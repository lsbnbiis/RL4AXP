"""
Quantum Refinement Module (SQA) for AMP multi-point mutation optimization.

Architecture (from SQA_AMP_RL.md):
  PPO top candidates
    → QUBO matrix:
        h_i  (diagonal)  = BLOSUM62 substitution score        [physical lower bound]
        J_ij (off-diag)  = real PepBERT multi-layer attention [co-evolution upper bound]
    → APC correction (de-noise)
    → GPU-accelerated SQA solver (Trotter slices = 20)
    → Optimal multi-point mutation combination

All matrix construction (attention fusion, APC correction, sparse filtering,
QUBO assembly) and the annealing loop run as GPU tensor ops end to end — no
intermediate `.cpu().numpy()` round-trip.
"""

from __future__ import annotations

import math
import torch as T
import torch.nn.functional as F
from typing import List, Tuple

# ─────────────────────────── BLOSUM62 ───────────────────────────
# Standard BLOSUM62 substitution matrix (NCBI)
# Row / column order: A R N D C Q E G H I L K M F P S T W Y V
_B62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_B62_IDX   = {aa: i for i, aa in enumerate(_B62_ORDER)}

_BLOSUM62_RAW = [
    #  A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    [  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],  # A
    [ -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
    [ -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],  # N
    [ -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],  # D
    [  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
    [ -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],  # Q
    [ -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],  # E
    [  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],  # G
    [ -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],  # H
    [ -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],  # I
    [ -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],  # L
    [ -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],  # K
    [ -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],  # M
    [ -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],  # F
    [ -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],  # P
    [  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],  # S
    [  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],  # T
    [ -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],  # W
    [ -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],  # Y
    [  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],  # V
]


def blosum62_score(aa_from: str, aa_to: str) -> float:
    i = _B62_IDX.get(aa_from)
    j = _B62_IDX.get(aa_to)
    if i is None or j is None:
        return 0.0
    return float(_BLOSUM62_RAW[i][j])


# BLOSUM62 as a GPU-indexable tensor (built once, moved to the target device
# on demand — it's only 20x20).
_BLOSUM62_T = T.tensor(_BLOSUM62_RAW, dtype=T.float32)


# ─────────────────── PepBERT real multi-layer attention ───────────────────

def get_position_embeddings(
    model: T.nn.Module,
    tokenizer,
    peptide: str,
    device: T.device,
) -> T.Tensor:
    """
    Return per-residue embeddings for one peptide: shape (L, d_model).
    Strips the [SOS] and [EOS] tokens from the encoder output.
    """
    sos_id = tokenizer.token_to_id("[SOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    ids = [sos_id] + tokenizer.encode(peptide).ids + [eos_id]
    input_ids = T.tensor([ids], dtype=T.int64, device=device)
    encoder_mask = T.ones((1, 1, 1, input_ids.size(1)), dtype=T.int64, device=device)

    with T.no_grad():
        embeds = model.encode(input_ids, encoder_mask)   # (1, L+2, d_model)

    return embeds[0, 1:-1, :]   # (L, d_model)


def get_layer_attentions(
    model: T.nn.Module,
    tokenizer,
    peptide: str,
    device: T.device,
) -> Tuple[T.Tensor, T.Tensor]:
    """
    Single forward pass that returns both the per-residue embeddings and the
    real self-attention map fused across encoder layers (SQA_AMP_RL.md §二.3
    step ①). Each PepBERT `MultiHeadAttentionBlock` stashes its
    `(batch, h, L+2, L+2)` softmax scores on `self.attention_scores` as a
    side effect of `forward()`, so they can be read straight off
    `model.encoder.layers[i].self_attention_block` right after `encode()`.

    Layer selection skips the first and last encoder block and fuses the
    ones in between (the "second-to-last through fifth-to-last" window the
    spec describes for a 6-layer encoder generalizes to "all middle layers").

    Returns
    -------
    embeddings : (L, d_model)
    attention  : (L, L)  symmetrized, head- and layer-averaged
    """
    sos_id = tokenizer.token_to_id("[SOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    ids = [sos_id] + tokenizer.encode(peptide).ids + [eos_id]
    input_ids = T.tensor([ids], dtype=T.int64, device=device)
    encoder_mask = T.ones((1, 1, 1, input_ids.size(1)), dtype=T.int64, device=device)

    with T.no_grad():
        embeds = model.encode(input_ids, encoder_mask)   # (1, L+2, d_model)

    layers = model.encoder.layers
    n_layers = len(layers)
    lo, hi = 1, max(n_layers - 1, 2)   # skip first & last block
    fused = T.stack([
        layers[i].self_attention_block.attention_scores[0].mean(dim=0)   # heads -> (L+2, L+2)
        for i in range(lo, hi)
    ]).mean(dim=0)                                                       # layers -> (L+2, L+2)

    attn = fused[1:-1, 1:-1]          # drop SOS/EOS
    attn = (attn + attn.T) / 2.0      # symmetrize (step ②) — raw attention isn't symmetric

    return embeds[0, 1:-1, :], attn


def build_attention_matrix(attention: T.Tensor) -> T.Tensor:
    """
    APC-correct and sparsify a symmetrized position-position attention
    matrix from `get_layer_attentions`, entirely as GPU tensor ops.

    Steps (from SQA_AMP_RL.md §二.3):
      2. APC correction  A_APC = A_sym - (row_mean × col_mean) / total_mean
      3. Sparse filter   zero out weakest 50 % of interactions
    """
    row_m   = attention.mean(dim=1, keepdim=True)
    col_m   = attention.mean(dim=0, keepdim=True)
    total_m = attention.mean()

    if T.abs(total_m) > 1e-8:
        A_apc = attention - (row_m * col_m) / total_m
    else:
        A_apc = attention.clone()

    threshold = T.quantile(A_apc.abs().flatten(), 0.5)
    A_apc = T.where(A_apc.abs() < threshold, T.zeros_like(A_apc), A_apc)

    return A_apc


# ──────────────────────── QUBO builder ────────────────────────

def build_qubo(
    seq: str,
    candidates: List[Tuple[int, str]],
    attention_matrix: T.Tensor,
    alpha: float = 1.0,
    lambda_hydro: float = 0.0,
    target_hydro: float = 0.4,
) -> Tuple[T.Tensor, T.Tensor]:
    """
    Build QUBO matrix for multi-point mutation selection, fully vectorized
    on whatever device `attention_matrix` already lives on.

    Energy: E(x) = Σ h_i x_i  +  Σ_{i<j} J_ij x_i x_j  +  λ * hydro_penalty

    h_i   = −BLOSUM62(orig→new)   [lower h → more favourable substitution]
    J_ij  = −α · A_APC[pos_i, pos_j]  [negative → co-operative reward]

    Returns
    -------
    h : (N,)   linear coefficients
    J : (N, N) quadratic coefficients (symmetric, zero diagonal)
    """
    HYDROPHOBIC = set("AFILMVW")
    device = attention_matrix.device
    N = len(candidates)

    if N == 0:
        return T.zeros(0, device=device), T.zeros((0, 0), device=device)

    blosum62_t = _BLOSUM62_T.to(device)
    pos_idx  = T.tensor([c[0] for c in candidates], dtype=T.long, device=device)
    orig_idx = T.tensor([_B62_IDX[seq[c[0]]] for c in candidates], dtype=T.long, device=device)
    aa_idx   = T.tensor([_B62_IDX[c[1]] for c in candidates], dtype=T.long, device=device)

    h = -blosum62_t[orig_idx, aa_idx]
    J = -alpha * attention_matrix[pos_idx][:, pos_idx]
    J = J - T.diag(T.diag(J))   # zero the diagonal

    # Optional hydrophobicity constraint
    if lambda_hydro > 0 and len(seq) > 0:
        L = len(seq)
        cur_h = sum(1 for aa in seq if aa in HYDROPHOBIC) / L
        delta = T.tensor(
            [(1 if aa_i in HYDROPHOBIC else 0) - (1 if seq[pos_i] in HYDROPHOBIC else 0)
             for pos_i, aa_i in candidates],
            dtype=T.float32, device=device,
        )
        h = h + 2 * lambda_hydro * delta / L * (cur_h - target_hydro)
        J = J + lambda_hydro * T.outer(delta, delta) / (L * L) * (1 - T.eye(N, device=device))

    return h, J


# ─────────────────── GPU-accelerated SQA solver ───────────────

def sqa_solve(
    h: T.Tensor,
    J: T.Tensor,
    n_trotter: int = 20,
    n_steps: int = 500,
    T_start: float = 5.0,
    T_end: float = 0.01,
    Gamma_start: float = 5.0,
    device: T.device | None = None,
) -> T.Tensor:
    """
    GPU-accelerated Simulated Quantum Annealing (SQA).

    Minimises  E(x) = h·x + x·J·x  over  x ∈ {0,1}^N.

    Implementation (from heardware_software_implementation.md §三):
    - All Trotter-slice spins stored on GPU VRAM
    - Metropolis acceptance vectorised over spins
    - Transverse-field coupling: J_⊥ = −T/2 · ln(tanh(Γ/(P·T)))
    - Trotter Slices fixed at n_trotter (default 20)

    Returns
    -------
    x : (N,) int32 tensor — selected mutations (1 = apply, 0 = skip)
    """
    N = h.shape[0]
    if device is None:
        device = h.device if N > 0 else (
            T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")
        )

    if N == 0:
        return T.zeros(0, dtype=T.int32, device=device)

    h_t   = h.to(device=device, dtype=T.float32)
    J_sym = ((J + J.T) / 2.0).to(device=device, dtype=T.float32)

    # Initialise P Trotter slices with random binary states: (P, N)
    spins = T.randint(0, 2, (n_trotter, N), dtype=T.float32, device=device)

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        T_curr = max(T_start * (T_end / T_start) ** frac, 1e-7)
        Gamma  = Gamma_start * (1.0 - frac)

        # Inter-Trotter coupling strength (scalar temperature schedule, not a matrix op)
        arg = Gamma / (n_trotter * T_curr)
        if arg > 1e-8:
            tanh_arg = math.tanh(min(arg, 20.0))
            J_perp = -0.5 * T_curr * math.log(max(tanh_arg, 1e-12))
        else:
            J_perp = 0.0
        J_perp_t = T.tensor(J_perp, dtype=T.float32, device=device)

        # Sweep all Trotter slices in random order
        for k in T.randperm(n_trotter, device=device).tolist():
            k_prev = (k - 1) % n_trotter
            k_next = (k + 1) % n_trotter

            classical = h_t + J_sym @ spins[k]                        # (N,)
            quantum   = J_perp_t * (spins[k_prev] + spins[k_next])   # (N,)

            # delta_E = energy change when flipping spin i
            # flip 1→0: ΔE = -(classical_i + quantum_i)
            # flip 0→1: ΔE = +(classical_i + quantum_i)
            delta_E = (1.0 - 2.0 * spins[k]) * (classical + quantum)

            accept = T.rand(N, device=device) < T.clamp(
                T.exp(-delta_E / T_curr), max=1.0
            )
            spins[k] = T.where(accept, 1.0 - spins[k], spins[k])

    # Pick the Trotter slice with lowest classical energy
    with T.no_grad():
        energies = T.stack([
            (h_t * spins[k]).sum() + 0.5 * (spins[k] @ J_sym @ spins[k])
            for k in range(n_trotter)
        ])
    best_k = int(energies.argmin().item())
    return spins[best_k].to(T.int32)


# ──────────────────────── High-level API ──────────────────────

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def get_top_positions(
    actor1: T.nn.Module,
    state: T.Tensor,
    n: int = 8,
) -> List[int]:
    """
    Return the n positions ranked highest by the trained Actor1 policy.
    state: (1, state_dim) tensor already on the actor's device.
    """
    with T.no_grad():
        probs = actor1(state).squeeze(0)   # (seq_len,)
    return T.topk(probs, min(n, probs.shape[0])).indices.cpu().tolist()


def get_top_aas(
    actor2: T.nn.Module,
    state: T.Tensor,
    position: int,
    n: int = 3,
) -> List[str]:
    """
    Return the n amino acids ranked highest by Actor2 for a given position.
    """
    pos_t = T.tensor([position], dtype=T.long, device=next(actor2.parameters()).device)
    with T.no_grad():
        probs = actor2(state, pos_t).squeeze(0)   # (20,)
    top_idx = T.topk(probs, min(n, probs.shape[0])).indices.cpu().tolist()
    return [AMINO_ACIDS[i] for i in top_idx]


def refine_peptide(
    seq: str,
    actor1: T.nn.Module,
    actor2: T.nn.Module,
    pepbert_model: T.nn.Module,
    pepbert_tokenizer,
    device: T.device,
    n_positions: int = 8,
    n_aas_per_pos: int = 3,
    n_trotter: int = 20,
    n_steps: int = 500,
    alpha: float = 1.0,
) -> Tuple[str, List[Tuple[int, str, str]]]:
    """
    Full SQA refinement pipeline for a single peptide.

    1. Get PepBERT embeddings + real multi-layer attention
    2. Actor1 → top positions; Actor2 → top AAs per position
    3. Build QUBO (BLOSUM62 + attention)
    4. SQA solve → binary selection
    5. Apply selected mutations

    Returns
    -------
    refined_seq   : new sequence after multi-point mutations
    mutations     : list of (position, original_aa, new_aa) applied
    """
    # 1. Embeddings + attention matrix (single forward pass)
    embeds, raw_attn = get_layer_attentions(pepbert_model, pepbert_tokenizer, seq, device)
    att_mat = build_attention_matrix(raw_attn)                # (L, L)

    # 2. Candidate mutations from Actor1/Actor2
    norm = F.normalize(embeds, dim=-1)
    state = norm.mean(dim=0, keepdim=True)                    # (1, d_model)  — light proxy

    top_pos = get_top_positions(actor1, state.to(next(actor1.parameters()).device), n=n_positions)

    candidates: List[Tuple[int, str]] = []
    for pos in top_pos:
        orig_aa = seq[pos]
        top_aas = get_top_aas(actor2, state.to(next(actor2.parameters()).device), pos, n=n_aas_per_pos)
        for aa in top_aas:
            if aa != orig_aa:
                candidates.append((pos, aa))

    if not candidates:
        return seq, []

    # 3. QUBO
    h, J = build_qubo(seq, candidates, att_mat, alpha=alpha)

    # 4. SQA
    selection = sqa_solve(h, J, n_trotter=n_trotter, n_steps=n_steps, device=device).cpu().tolist()

    # 5. Apply
    seq_list = list(seq)
    applied: dict[int, str] = {}
    mutations: List[Tuple[int, str, str]] = []
    for i, x in enumerate(selection):
        if x == 1:
            pos, new_aa = candidates[i]
            if pos not in applied:
                applied[pos]  = new_aa
                mutations.append((pos, seq[pos], new_aa))
                seq_list[pos] = new_aa

    return "".join(seq_list), mutations


def batch_refine(
    seqs: List[str],
    actor1: T.nn.Module,
    actor2: T.nn.Module,
    pepbert_model: T.nn.Module,
    pepbert_tokenizer,
    device: T.device,
    **kwargs,
) -> List[Tuple[str, List[Tuple[int, str, str]]]]:
    """Run refine_peptide for a list of sequences."""
    results = []
    for seq in seqs:
        refined, muts = refine_peptide(
            seq, actor1, actor2, pepbert_model, pepbert_tokenizer, device, **kwargs
        )
        results.append((refined, muts))
    return results
