"""Design rules and scoring helpers for AMP / therapeutic peptide design (v2.1).

Key fixes from v2:
- aggregation_risk_score now has a usable gradient and consistent direction
- selectivity_proxy_score is actually implemented in soft_rule_features
- selectivity_index is normalized before entering reward aggregation
- pH-aware charge features are exposed directly in soft_rule_features
- aromatic hard filter kept, but softened slightly for W-rich exploration

Notes:
- These are heuristic defaults, not universal laws.
- Tune thresholds to your organism, assay, mechanism, and peptide family.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import math


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

RESIDUE_PROPERTIES: Dict[str, Dict[str, object]] = {
    "A": {"name": "Ala",  "charge": 0,  "class": "hydrophobic",           "helix": "medium"},
    "C": {"name": "Cys",  "charge": 0,  "class": "special",               "helix": "medium"},
    "D": {"name": "Asp",  "charge": -1, "class": "acidic",                "helix": "low"},
    "E": {"name": "Glu",  "charge": -1, "class": "acidic",                "helix": "high"},
    "F": {"name": "Phe",  "charge": 0,  "class": "hydrophobic_aromatic",  "helix": "medium"},
    "G": {"name": "Gly",  "charge": 0,  "class": "special_flexible",      "helix": "low"},
    "H": {"name": "His",  "charge": 0,  "class": "conditional_cationic",  "helix": "medium"},
    "I": {"name": "Ile",  "charge": 0,  "class": "hydrophobic",           "helix": "medium"},
    "K": {"name": "Lys",  "charge": +1, "class": "basic",                 "helix": "high"},
    "L": {"name": "Leu",  "charge": 0,  "class": "hydrophobic",           "helix": "high"},
    "M": {"name": "Met",  "charge": 0,  "class": "hydrophobic",           "helix": "high"},
    "N": {"name": "Asn",  "charge": 0,  "class": "polar",                 "helix": "low"},
    "P": {"name": "Pro",  "charge": 0,  "class": "special_helix_breaker", "helix": "very_low"},
    "Q": {"name": "Gln",  "charge": 0,  "class": "polar",                 "helix": "medium"},
    "R": {"name": "Arg",  "charge": +1, "class": "basic",                 "helix": "medium"},
    "S": {"name": "Ser",  "charge": 0,  "class": "polar",                 "helix": "low"},
    "T": {"name": "Thr",  "charge": 0,  "class": "polar",                 "helix": "low"},
    "V": {"name": "Val",  "charge": 0,  "class": "hydrophobic",           "helix": "medium"},
    "W": {"name": "Trp",  "charge": 0,  "class": "hydrophobic_aromatic",  "helix": "medium"},
    "Y": {"name": "Tyr",  "charge": 0,  "class": "polar_aromatic",        "helix": "medium"},
}

HYDROPHOBIC_SET = set("AFILMVW")
BASIC_SET = set("KR")
ACIDIC_SET = set("DE")
AROMATIC_SET = set("FWY")
HELIX_BREAKERS = set("PG")
AGGREGATION_PRONE_AROMATIC = set("FWY")

# Eisenberg consensus hydrophobicity scale (Eisenberg et al. 1984), used for
# the hydrophobic-moment / amphipathicity calculation below.
EISENBERG_HYDROPHOBICITY: Dict[str, float] = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

# Typical mean hydrophobic moment (Eisenberg scale, per residue) at which an
# alpha-helical AMP is considered clearly amphipathic; used to normalize
# hydrophobic_moment() into a 0..1 score.
AMPHIPATHIC_MOMENT_TARGET = 0.35

HARD_FILTERS = {
    "length_min": 10,
    "length_max": 30,
    "net_charge_min": 2,
    "net_charge_max": 9,
    "max_consecutive_hydrophobic": 4,
    "max_identical_residue_run": 3,
    # softened from 2 -> 3 for W-rich exploration while keeping soft penalty active
    "max_consecutive_aromatic": 3,
    "c_terminal_options": ["COOH", "CONH2"],
    "allow_d_amino_acid": False,
    "allow_noncanonical": False,
}

SOFT_TARGETS = {
    "preferred_net_charge_window": (4, 6),
    "preferred_length_window": (12, 24),
    "preferred_arg_fraction_max": 0.35,
    "preferred_trp_count_max": 3,
    "preferred_hydrophobic_fraction_range": (0.30, 0.55),
    "preferred_basic_fraction_range": (0.20, 0.45),
    "preferred_consecutive_aromatic_max": 1,
}

DEFAULT_REWARD_WEIGHTS = {
    "amp_activity_score": 2.0,
    "hemolysis_score": -1.5,
    "serum_stability_score": 0.8,
    "protease_stability_score": 0.8,
    "amphipathicity_score": 0.6,
    "novelty_score": 0.4,
    "aggregation_risk": -0.7,
    "synthesis_penalty": -0.5,
    "selectivity_index": 1.0,
}

HIS_PKA = 6.0

@dataclass(frozen=True)
class DesignRule:
    rule_id: str
    category: str
    rule_name: str
    heuristic: str
    common_operation: str
    expected_effect: str
    main_risk: str
    reward_or_filter_hint: str
    priority: str = "high"

DESIGN_RULES: List[DesignRule] = [
    DesignRule(
        "R001", "charge", "Net positive charge",
        "Most AMP leads benefit from net positive charge, often starting around +2 to +9.",
        "Increase Lys/Arg and reduce Asp/Glu.",
        "Improves attraction to anionic bacterial membranes.",
        "Too much charge can increase nonspecific interactions and salt sensitivity.",
        "Use hard bounds + soft target window around +4 to +6.",
    ),
    DesignRule(
        "R002", "charge", "Lys-to-Arg substitution",
        "Arg often binds membrane anionic groups more strongly than Lys, but not always more selectively.",
        "Test local K->R substitutions instead of full replacement.",
        "Can improve membrane binding and potency.",
        "May increase hemolysis or mammalian cytotoxicity.",
        "Add soft bonus for moderate Arg fraction; penalize excess Arg.",
    ),
    DesignRule(
        "R003", "charge", "Histidine as pH switch",
        "His gains positive charge in acidic environments; evaluate charge at pH 7.4 and 6.0.",
        "Introduce 1-3 His for pH-responsive variants.",
        "Potential infection-site or tumor-microenvironment responsiveness.",
        "Neutral pH potency may drop.",
        "Optional branch objective for pH-sensitive projects.",
        priority="medium",
    ),
    DesignRule(
        "R018", "selectivity", "Selectivity Index (SI)",
        "SI = MHC / MIC; higher values indicate a better therapeutic window.",
        "Calculate from paired mammalian-toxicity and antibacterial-potency assays; target SI > 10.",
        "Guides lead selection toward lower host toxicity at useful potency.",
        "SI alone does not capture kinetics or resistance behaviour.",
        "Normalize SI before reward aggregation; do not feed raw SI directly into reward.",
    ),
    DesignRule(
        "R019", "aggregation", "Aromatic clustering / aggregation risk",
        "Consecutive aromatic residues can drive pi-stacking aggregation and poor solubility.",
        "Avoid long aromatic clusters; keep soft pressure above 1 and hard fail above 3.",
        "Reduces aggregation-related activity loss and off-target effects.",
        "Over-removing aromatics can reduce membrane anchoring.",
        "Use both soft penalty and hard filter.",
    ),
]

def validate_sequence(seq: str) -> None:
    invalid = set(seq) - set(AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Invalid residues found: {sorted(invalid)}")

def calculate_net_charge(
    seq: str,
    c_terminal: str = "COOH",
    n_terminal_free: bool = True,
    pH: float = 7.4,
) -> float:
    validate_sequence(seq)
    charge = 0.0

    for aa in seq:
        if aa in {"K", "R"}:
            charge += 1.0
        elif aa in {"D", "E"}:
            charge -= 1.0

    his_fractional_charge = 1.0 / (1.0 + 10 ** (pH - HIS_PKA))
    charge += seq.count("H") * his_fractional_charge

    if n_terminal_free:
        charge += 1.0
    if c_terminal == "COOH":
        charge -= 1.0
    elif c_terminal != "CONH2":
        raise ValueError("c_terminal must be 'COOH' or 'CONH2'")

    return charge

def residue_fraction(seq: str, residue_set: set) -> float:
    validate_sequence(seq)
    if not seq:
        return 0.0
    return sum(aa in residue_set for aa in seq) / len(seq)

def hydrophobic_fraction(seq: str) -> float:
    return residue_fraction(seq, HYDROPHOBIC_SET)

def hydrophobic_moment(seq: str, delta_deg: float = 100.0) -> float:
    """Mean alpha-helical hydrophobic moment (Eisenberg 1982).

    delta_deg=100 degrees is the per-residue helical-wheel angle for an
    alpha-helix (3.6 residues/turn); higher moment -> more amphipathic.
    """
    validate_sequence(seq)
    if not seq:
        return 0.0
    delta = math.radians(delta_deg)
    sin_sum = sum(EISENBERG_HYDROPHOBICITY[aa] * math.sin(i * delta) for i, aa in enumerate(seq))
    cos_sum = sum(EISENBERG_HYDROPHOBICITY[aa] * math.cos(i * delta) for i, aa in enumerate(seq))
    return math.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(seq)

def basic_fraction(seq: str) -> float:
    return residue_fraction(seq, BASIC_SET)

def arg_fraction(seq: str) -> float:
    validate_sequence(seq)
    return seq.count("R") / len(seq) if seq else 0.0

def trp_count(seq: str) -> int:
    validate_sequence(seq)
    return seq.count("W")

def max_consecutive_run(seq: str, residue_set: set | None = None) -> int:
    validate_sequence(seq)
    best = 0
    current = 0
    for aa in seq:
        in_set = True if residue_set is None else aa in residue_set
        if in_set:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best

def max_identical_residue_run(seq: str) -> int:
    validate_sequence(seq)
    if not seq:
        return 0
    best = 1
    current = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best

def hard_filter_pass(
    seq: str,
    c_terminal: str = "COOH",
    pH: float = 7.4,
) -> Tuple[bool, Dict[str, object]]:
    validate_sequence(seq)
    details = {
        "length": len(seq),
        "net_charge": calculate_net_charge(seq, c_terminal=c_terminal, pH=pH),
        "max_consecutive_hydrophobic": max_consecutive_run(seq, HYDROPHOBIC_SET),
        "max_identical_residue_run": max_identical_residue_run(seq),
        "max_consecutive_aromatic": max_consecutive_run(seq, AGGREGATION_PRONE_AROMATIC),
    }
    passed = (
        HARD_FILTERS["length_min"] <= details["length"] <= HARD_FILTERS["length_max"]
        and HARD_FILTERS["net_charge_min"] <= details["net_charge"] <= HARD_FILTERS["net_charge_max"]
        and details["max_consecutive_hydrophobic"] <= HARD_FILTERS["max_consecutive_hydrophobic"]
        and details["max_identical_residue_run"] <= HARD_FILTERS["max_identical_residue_run"]
        and details["max_consecutive_aromatic"] <= HARD_FILTERS["max_consecutive_aromatic"]
    )
    return passed, details

def window_score(value: float, low: float, high: float, decay_width_factor: float = 1.0) -> float:
    if low <= value <= high:
        return 1.0
    width = max(high - low, 1e-6)
    decay_scale = width * decay_width_factor
    dist = (low - value) if value < low else (value - high)
    return max(0.0, 1.0 - dist / decay_scale)

def normalize_selectivity_index(si: float, target_si: float = 10.0) -> float:
    """Map raw SI to 0..1 with SI=target_si approaching 1.0."""
    if si <= 0:
        return 0.0
    return min(1.0, si / target_si)

def soft_rule_features(
    seq: str,
    c_terminal: str = "COOH",
    pH: float = 7.4,
) -> Dict[str, float]:
    validate_sequence(seq)

    length = len(seq)
    charge = calculate_net_charge(seq, c_terminal=c_terminal, pH=pH)
    charge_pH74 = calculate_net_charge(seq, c_terminal=c_terminal, pH=7.4)
    charge_pH60 = calculate_net_charge(seq, c_terminal=c_terminal, pH=6.0)
    hydro = hydrophobic_fraction(seq)
    basic = basic_fraction(seq)
    argf = arg_fraction(seq)
    wcount = trp_count(seq)
    max_arom = max_consecutive_run(seq, AGGREGATION_PRONE_AROMATIC)

    aggregation_risk_score = min(1.0, max(0.0, (max_arom - 1) / 2.0))
    aggregation_control_score = 1.0 - aggregation_risk_score

    moment = hydrophobic_moment(seq)
    amphipathicity_score = min(1.0, moment / AMPHIPATHIC_MOMENT_TARGET)

    selectivity_proxy_score = (
        0.40 * window_score(charge, *SOFT_TARGETS["preferred_net_charge_window"])
        + 0.35 * window_score(hydro, *SOFT_TARGETS["preferred_hydrophobic_fraction_range"])
        + 0.25 * window_score(basic, *SOFT_TARGETS["preferred_basic_fraction_range"])
    )

    return {
        "length_score": window_score(length, *SOFT_TARGETS["preferred_length_window"]),
        "net_charge_score": window_score(charge, *SOFT_TARGETS["preferred_net_charge_window"]),
        "hydrophobic_fraction": hydro,
        "hydrophobicity_score": window_score(hydro, *SOFT_TARGETS["preferred_hydrophobic_fraction_range"]),
        "basic_fraction": basic,
        "basic_fraction_score": window_score(basic, *SOFT_TARGETS["preferred_basic_fraction_range"]),
        "arg_fraction": argf,
        "arg_fraction_score": 1.0 if argf <= SOFT_TARGETS["preferred_arg_fraction_max"] else 0.0,
        "trp_count": float(wcount),
        "trp_count_score": 1.0 if wcount <= SOFT_TARGETS["preferred_trp_count_max"] else 0.0,
        "max_consecutive_aromatic": float(max_arom),
        "aggregation_risk_score": aggregation_risk_score,
        "aggregation_control_score": aggregation_control_score,
        "hydrophobic_moment": moment,
        "amphipathicity_score": amphipathicity_score,
        "selectivity_proxy_score": selectivity_proxy_score,
        "charge_pH7_4": charge_pH74,
        "charge_pH6_0": charge_pH60,
        "delta_charge_acidic": charge_pH60 - charge_pH74,
    }

def example_reward(
    amp_activity_score: float,
    hemolysis_score: float,
    serum_stability_score: float,
    protease_stability_score: float,
    amphipathicity_score: float,
    novelty_score: float,
    aggregation_risk: float,
    synthesis_penalty: float,
    selectivity_index: float = 0.0,
    weights: Dict[str, float] | None = None,
    normalize_si: bool = True,
) -> float:
    w = DEFAULT_REWARD_WEIGHTS if weights is None else weights
    si_term = normalize_selectivity_index(selectivity_index) if normalize_si else selectivity_index

    return (
        w["amp_activity_score"] * amp_activity_score
        + w["hemolysis_score"] * hemolysis_score
        + w["serum_stability_score"] * serum_stability_score
        + w["protease_stability_score"] * protease_stability_score
        + w["amphipathicity_score"] * amphipathicity_score
        + w["novelty_score"] * novelty_score
        + w["aggregation_risk"] * aggregation_risk
        + w["synthesis_penalty"] * synthesis_penalty
        + w.get("selectivity_index", 1.0) * si_term
    )

def rules_as_dicts() -> List[Dict[str, object]]:
    return [asdict(rule) for rule in DESIGN_RULES]

if __name__ == "__main__":

    demo_seq = "KWLKLLKKWLKLLKK"
    
    for ph in (7.4, 6.0):
        passed, hard_details = hard_filter_pass(demo_seq, c_terminal="CONH2", pH=ph)
        soft_details = soft_rule_features(demo_seq, c_terminal="CONH2", pH=ph)
        print(f"\n=== pH {ph} ===")
        print("Sequence      :", demo_seq)
        print("Hard filter   :", passed)
        print("Hard details  :", hard_details)
        print("Soft features :")
        for k, v in soft_details.items():
            print(f"  {k:<30} {v}")
