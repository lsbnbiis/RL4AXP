import numpy as np
import config
import torch as T

from amp_prediction.inference import get_amp_probs
from hem_prediction.inference import get_hem_probs
from acp_prediction.inference import get_acp_probs
from afp_prediction.inference import get_afp_probs
from avp_prediction.inference import get_avp_probs
from mrsa_prediction.inference import get_mrsa_probs
from peptide_optimization.encoding import PeptideEncoder
from peptide_optimization.design_rules_v2_1 import soft_rule_features, hard_filter_pass

_PROB_FNS = {
    "AMP": get_amp_probs,
    "HEM": lambda peptides: get_hem_probs(peptides, [config.HEM_CONCENTRATION] * len(peptides)),
    "ACP": get_acp_probs,
    "AFP": get_afp_probs,
    "AVP": get_avp_probs,
    "MRSA": get_mrsa_probs,
}

# +1 → maximise probability, -1 → minimise probability
_MODEL_DIRECTIONS = {
    "AMP": +1,
    "HEM": -1,
    "ACP": +1,
    "AFP": +1,
    "AVP": +1,
    "MRSA": +1,
}

def _get_probs_batch_dedup(prob_fn, peptides: list[str], device: T.device) -> T.Tensor:
    if len(peptides) == 0:
        return T.empty(0, device=device)
    unique_peps, inv = np.unique(peptides, return_inverse=True)
    if len(unique_peps) == len(peptides):
        return prob_fn(peptides).to(device)
    unique_probs = prob_fn(list(unique_peps)).to(device)
    if unique_probs.dim() == 0:
        unique_probs = unique_probs.unsqueeze(0)
    inv_t = T.tensor(inv, dtype=T.int64, device=device)
    return unique_probs[inv_t]

def _heuristic_reward_single(seq: str, c_terminal: str = "CONH2") -> float:

    passed, hard_details = hard_filter_pass(seq, c_terminal=c_terminal)
    soft = soft_rule_features(seq, c_terminal=c_terminal)

    consec_hydro = float(hard_details["max_consecutive_hydrophobic"])
    consec_identical = float(hard_details["max_identical_residue_run"])

    penalty = 0.0
    penalty += 0.25 * max(0.0, consec_hydro - 3.0)
    penalty += 0.20 * max(0.0, consec_identical - 2.0)
    penalty += 0.25 * max(0, seq.count("W") - 3)
    bonus = 0.10 if c_terminal == "CONH2" else 0.0

    hard_penalty = 0.0 if passed else -2.0

    feature_score = (
        0.50 * soft["net_charge_score"]
        + 0.45 * soft["hydrophobicity_score"]
        + 0.35 * soft["amphipathicity_score"]
        + 0.30 * soft["basic_fraction_score"]
        + 0.25 * soft["aggregation_control_score"]
        + 0.30 * soft["selectivity_proxy_score"]
        + 0.30 * soft["length_score"]
    )

    HEURISTIC_SCALE = 0.3

    return HEURISTIC_SCALE * (feature_score - penalty + bonus + hard_penalty)

def _heuristic_rewards_batch(peptides: list[str], device: T.device) -> T.Tensor:

    if len(peptides) == 0:
        return T.empty(0, device=device)
    unique_peps, inv = np.unique(peptides, return_inverse=True)
    unique_scores = [_heuristic_reward_single(p) for p in unique_peps]
    inv_t = T.tensor(inv, dtype=T.int64, device=device)
    unique_t = T.tensor(unique_scores, dtype=T.float32, device=device)
    return unique_t[inv_t]

class Environment:

    def __init__(self) -> None:

        assert len(config.REWARD_MODELS) >= 1, "REWARD_MODELS must contain at least one model."
        assert all(m in _PROB_FNS for m in config.REWARD_MODELS), \
            f"Unknown model(s) in REWARD_MODELS. Valid choices: {list(_PROB_FNS)}"

        self.encoder = PeptideEncoder()
        self.seq_len = len(config.TARGET_PEPTIDE)
        self.reward_models = list(config.REWARD_MODELS)

        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.a2_to_aa = {idx: aa for idx, aa in enumerate(self.amino_acids)}
        self.aa_to_a2 = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.device = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")

        self.peptides_1 = [config.TARGET_PEPTIDE] * config.N_PARALLELS

        self.probs_1 = {m: _get_probs_batch_dedup(_PROB_FNS[m], self.peptides_1, self.device) for m in self.reward_models}
        self.heuristic_1 = _heuristic_rewards_batch(self.peptides_1, self.device)

        self.states_1 = self.encoder.encode(self.peptides_1)
        self.state_dim = self.states_1.shape[1]

        self.n_action1 = self.seq_len
        self.n_action2 = len(self.amino_acids)

    def get_action_masks(self, action1s: T.Tensor) -> T.Tensor:
        """
        Generate action2 mask prohibiting mutating to the same amino acid (No-op mutation).
        action1s: (N,) Tensor containing selected positions (0..seq_len-1).
        Returns: (N, 20) Boolean Tensor on self.device (True = allowed, False = forbidden).
        """
        a1_list = action1s.tolist() if isinstance(action1s, T.Tensor) else list(action1s)
        N = len(self.peptides_curr)
        masks = T.ones((N, len(self.amino_acids)), dtype=T.bool, device=self.device)
        for i, (pep, a1) in enumerate(zip(self.peptides_curr, a1_list)):
            curr_aa = pep[a1]
            curr_aa_idx = self.aa_to_a2[curr_aa]
            masks[i, curr_aa_idx] = False
        return masks

    def reset(self) -> T.Tensor:

        self.done = False
        self.time_step = 1

        self.peptides_curr = self.peptides_1.copy()
        self.peptides_prev = self.peptides_1.copy()

        self.probs_curr = {m: self.probs_1[m].clone() for m in self.reward_models}
        self.probs_prev = {m: self.probs_1[m].clone() for m in self.reward_models}

        self.heuristic_curr = self.heuristic_1.clone()
        self.heuristic_prev = self.heuristic_1.clone()

        return self.states_1

    def step(self, action1s: T.Tensor, action2s: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:

        action1s = action1s.tolist()
        action2s = action2s.tolist()

        self.peptides_prev = self.peptides_curr.copy()
        new_aas = [self.a2_to_aa[a2] for a2 in action2s]
        self.peptides_curr = [
            p[:a1] + aa + p[a1 + 1:]
            for p, a1, aa in zip(self.peptides_curr, action1s, new_aas)
        ]

        for i, p in enumerate(self.peptides_curr):
            assert len(p) == self.seq_len, f"Peptide {i} length changed: {len(p)} != {self.seq_len}, seq={p}"

        for m in self.reward_models:
            self.probs_prev[m] = self.probs_curr[m].clone()
            self.probs_curr[m] = _get_probs_batch_dedup(_PROB_FNS[m], self.peptides_curr, self.device)

        self.heuristic_prev = self.heuristic_curr.clone()
        self.heuristic_curr = _heuristic_rewards_batch(self.peptides_curr, self.device)

        if self.time_step == config.TIME_HORIZON:
            self.peptides_T = self.peptides_curr.copy()
            self.done = True
        else:
            self.time_step += 1

        return self.encoder.encode(self.peptides_curr), self._get_rewards(), self.done

    def _get_rewards(self) -> T.Tensor:

        heuristic_step = self.heuristic_curr - self.heuristic_prev

        reward = heuristic_step
        for m in self.reward_models:
            d = _MODEL_DIRECTIONS[m]
            w = config.REWARD_WEIGHTS.get(m, 1.0)
            reward = reward + w * d * (self.probs_curr[m] - self.probs_prev[m])

        # Extra penalty whenever HEM probability exceeds the safe threshold.
        # This creates a hard floor that fires every step, not just at terminal.
        if "HEM" in self.reward_models:
            excess = T.clamp(self.probs_curr["HEM"] - config.HEM_THRESHOLD, min=0.0)
            reward = reward - config.HEM_PENALTY_SCALE * excess

        if self.done:
            heuristic_final = self.heuristic_curr - self.heuristic_1
            reward = reward + heuristic_final

            for m in self.reward_models:
                d = _MODEL_DIRECTIONS[m]
                w = config.REWARD_WEIGHTS.get(m, 1.0)
                curr = self.probs_curr[m]
                T_diff = curr - self.probs_1[m]
                # opt_factor: "distance to optimum" — small when already near the target
                opt_factor = (1 - curr) if d > 0 else curr
                factor = T.where(d * T_diff > 0, opt_factor, 1 - opt_factor)
                score = w * d * T_diff / T.clamp(factor, min=1e-2)
                reward = reward + score

        return reward.cpu()
