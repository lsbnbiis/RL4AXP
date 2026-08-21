import os
import torch as T
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import logging
import tensorflow as tf

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from tokenizers import Tokenizer
from hem_prediction.lysispeptica import build_transformer
from hem_prediction._utils import CustomModel, GlobalMinPooling1D
from hem_prediction._utils import pc6_8d_encode, add_conc_on_pepbert_array

### pepBERT encoding

# 1) recall tokenizer.json and load the tokenizer
tokenizer_path = os.path.join("./hem_prediction", "pepbert_small", "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)

# 2) recall model weights (has downloaded from hf)
weights_path = os.path.join("./hem_prediction", "pepbert_small", "tmodel_16.pt")

# 3) hyperprams
config = {"seq_len": 52, "d_model": 160}

# 4) Initialize the model structure and load the weights
# Use CPU to avoid CUDA context conflict between TF/Keras and PyTorch.
# PepBERT-small (160-dim, 52 tokens) is fast enough on CPU.
DEVICE = T.device("cpu")

pbert_model = build_transformer(
    src_vocab_size=tokenizer.get_vocab_size(),
    src_seq_len=config["seq_len"],
    d_model=config["d_model"]
)
state = T.load(weights_path, weights_only=True, map_location=T.device(DEVICE))
# default is weights_only=False, will raise a FutureWarning
pbert_model.load_state_dict(state["model_state_dict"])
pbert_model.to(DEVICE)
pbert_model.eval()

# Cache for loaded Keras models — load once, reuse across calls
_keras_model_cache: dict = {}

def _load_keras_model(mdpath: str):

    if mdpath not in _keras_model_cache:
        _keras_model_cache[mdpath] = keras.models.load_model(
            mdpath,
            custom_objects={
                "CustomModel": CustomModel,
                "GlobalMinPooling1D": GlobalMinPooling1D
            },
            compile=False
        )

    return _keras_model_cache[mdpath]

PAD_ID = tokenizer.token_to_id("[PAD]")
SOS_ID = tokenizer.token_to_id("[SOS]")
EOS_ID = tokenizer.token_to_id("[EOS]")

def pbert_encode(seqli, tar_len, batch_size: int = 64):
    """
    Batch pepBERT encoding: processes sequences in mini-batches through the
    Transformer encoder, giving a large speedup over one-by-one inference.
    Returns a list of (seq_len, d_model) numpy arrays (no padding tokens).
    """
    opli = []

    for start in range(0, len(seqli), batch_size):
        batch_seqs = seqli[start: start + batch_size]

        # Tokenise and pad every sequence in the mini-batch
        batch_ids = []
        for seq in batch_seqs:
            ids = [SOS_ID] + tokenizer.encode(seq).ids + [EOS_ID]
            ids = ids + [PAD_ID] * (tar_len - len(seq))
            batch_ids.append(ids)

        input_ids = T.tensor(batch_ids, dtype=T.int64).to(DEVICE) # (B, tar_len)

        with T.no_grad():
            encoder_mask = (input_ids != PAD_ID).unsqueeze(1).unsqueeze(2).long()
            emb = pbert_model.encode(input_ids, encoder_mask) # (B, tar_len, d_model)
            emb_no_cls = emb[:, 1:-1, :].cpu().numpy() # strip [SOS]/[EOS]

        for j in range(emb_no_cls.shape[0]): opli.append(emb_no_cls[j])

    return opli # opli can't be numpy array, cause will add conc into 161 depth

def read_fasta(filepath: str):

    ids, concs, seqs = [], [], []
    seq_id = None
    seq_conc = None

    # ---------- Parse FASTA ----------
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
              
                # Parse new header
                header = line[1:]
                # Ex: seq1|Conc=14.3
                parts = header.split("|")
                seq_id = parts[0]
                ids.append(seq_id)
                p2 = parts[1].split("=")[1]

                try: seq_conc = float(p2)
                except: seq_conc = 50 # assign default conc 50 ug/ml
                concs.append(seq_conc)

            else: seqs.append(line)

    return ids, seqs, concs

def read_fasta_slice(filepath: str, window: int = 49):
    """
    Read FASTA and output 3 lists:
      - ids: fragment IDs (e.g. seq1_1_10)
      - concs: float concentrations
      - seqs: fragment sequences
    """
    ids, concs, seqs = [], [], []
    seq_id = None
    seq_conc = None
    seq_lines = []

    # ---------- Parse FASTA ----------
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):

                # Yield previous record
                if seq_id is not None:
                    full_seq = "".join(seq_lines)
                    _slice_sequence(seq_id, seq_conc, full_seq, window, ids, concs, seqs)

                # Parse new header
                header = line[1:]
                # Example: seq1|Conc=14.3
                parts = header.split("|")
                seq_id = parts[0]
                p2 = parts[1].split("=")[1]

                try: seq_conc = float(p2)
                except: seq_conc = 50 # assign default conc 50 ug/ml
                seq_lines = []

            else: seq_lines.append(line)

        # Final record
        if seq_id is not None:
            full_seq = "".join(seq_lines)
            _slice_sequence(seq_id, seq_conc, full_seq, window, ids, concs, seqs)

    return ids, seqs, concs

def _slice_sequence(seq_id, conc, full_seq, window, ids, concs, seqs):
    """
    Slice sequence using non-overlapping sliding windows.
    Append results to ids, concs, seqs lists.
    """
    seq_len = len(full_seq)

    # Case 1: sequence <= window → keep original
    if seq_len <= window:
        ids.append(seq_id)
        concs.append(conc)
        seqs.append(full_seq)
        return

    # Case 2: slice into fragments
    start = 1  # human-readable index
    while start <= seq_len:
        end = min(start + window - 1, seq_len)
        frag = full_seq[start-1:end] # Python index is 0-based
        frag_id = f"{seq_id}_{start}_{end}"
        ids.append(frag_id)
        concs.append(conc)
        seqs.append(frag)
        start = end + 1

def encoded_policy(mdpath, X_input):
    """Run a single Keras model on pre-computed input features."""
    keras_model = _load_keras_model(mdpath)
    pred_probs = keras_model.predict(X_input, verbose=0)
    # pred_probs shape: (N, 2) — return positive-class column
    return pred_probs[:, 1]

def ensemble_prob(prob_li):

    prob_li = np.array(prob_li)
    ens_prob = prob_li.mean(axis=0)

    return(ens_prob)

### main predict function, with 2 inputs, sequence and concentration

def predict(seqli, ugmlli, pbert_batch_size: int = 64):
    """
    Predict hemolytic probability for a list of sequences.

    Optimisations vs. the original:
      1. pepBERT encoder runs in mini-batches (pbert_batch_size) instead of
         one sequence at a time — major speedup for large inputs.
      2. Each distinct encoding is computed only once and shared across all
         ensemble models that use it (avoids re-running pepBERT twice).
      3. Keras models are loaded once and cached in _keras_model_cache —
         repeated calls to predict() skip the model-loading overhead.
      4. The ensemble models run in parallel via ThreadPoolExecutor.
    """
    md_policy = {
        5: [("md845613_5240chatt.keras", "pepbert_um")],

        10: [
            ("p791_836_cnn_zs_5544.keras", "pc6zs"),
            ("p798_796_cnn2_zs_5545.keras", "pc6zs"),
            ("p763_843_5950_3p1bn_ugml2std.keras", "pepbert_ugml"),
            ("p843_750_5041chatt_ugml2std.keras", "pepbert_ugml")
        ],

        20: [("md742811_1bnMLP.keras", "pepbert_um"), ("md749791_2p2bnMLP.keras", "pepbert_um")],
        30: [("md871776_3bnMLP.keras", "pepbert_um")]
    }

    thr_id = 10  # fixed threshold

    # Filter to only models whose files are present on disk
    _model_dir = os.path.join("./hem_prediction", "lysispeptica_models_thr10")
    policies = [
        (fname, enc) for fname, enc in md_policy[thr_id]
        if os.path.isfile(os.path.join(_model_dir, fname))
    ]
    if not policies:
        raise FileNotFoundError(
            f"No HEM ensemble models found in {_model_dir}. "
            "Please ensure at least one .keras model file is present."
        )

    needed_encodings = {enc for _, enc in policies}

    # --- Step 1: pre-compute all required encodings (deduplicated) ---
    # pepBERT is the expensive step; compute once and reuse across models.
    precomputed: dict = {}

    pbert_needed = {enc for enc in needed_encodings if enc.startswith("pepbert")}
    if pbert_needed:
        d160li = pbert_encode(seqli, 49, batch_size=pbert_batch_size)
        for enc in pbert_needed:
            precomputed[enc] = add_conc_on_pepbert_array(enc, d160li, seqli, ugmlli)

    for enc in needed_encodings - pbert_needed:
        precomputed[enc] = pc6_8d_encode(seqli, ugmlli, enc, 49)

    # --- Step 2: run all ensemble models in parallel ---
    def run_model(p):
        fname, encoding = p
        mdpath = os.path.join("./hem_prediction", "lysispeptica_models_thr10", fname)
        return encoded_policy(mdpath, precomputed[encoding])

    ens_li = [run_model(p) for p in policies]

    return np.array(ens_li).mean(axis=0)

def get_hem_probs(peptides: list[str], ugmlli: list[float] | None = None) -> T.Tensor:
    
    if ugmlli is None: ugmlli = [50.0] * len(peptides)
    
    for i, p in enumerate(peptides):
        if len(p) == 0: raise ValueError(f"Empty peptide at index {i}")
        if len(p) > 49: raise ValueError(f"Peptide too long at index {i}: len={len(p)}, seq={p}")
    
    hem_probs: np.ndarray = predict(peptides, ugmlli)

    return T.tensor(hem_probs, dtype=T.float32, device=DEVICE)

def test() -> None:

    # example input
    seqli = ["VRRFPWWYPFLRR", "WRPGRWWRPGRWWRPGFGGGRGGPGRW", "GWWRRTVAKVRA"]
    ugmlli = [136.2, 30, 50]

    prob_list = get_hem_probs(seqli, ugmlli)
    print(prob_list) # [0.80606806 0.33022696 0.16366142]

    import time

    # --- 500-sequence throughput test ---
    peptides_500 = ["FLTHILKGLFTAGKVIHGLIHRRRHG"] * 500

    # Warmup
    for _ in range(3): get_hem_probs(peptides_500)

    times = []
    for _ in range(20):

        t1 = time.time()
        probs = get_hem_probs(peptides_500)
        t2 = time.time()

        times.append(t2 - t1)

    print(probs.shape)

    print(f"Stable inference time for 500 sequences: {np.mean(times):.4f} ± {np.std(times):.4f} sec")