BETA = 1e-3
GAMMA = 0.99
EPSILON = 0.2
LAMBDA = 0.95

TIME_HORIZON = 5
N_EPISODES = 100_000
ENCODING_SCHEME = "PepBERT-large"

BUFFER_SIZE = 2048 * TIME_HORIZON
BATCH_SIZE = 128
N_EPOCHS = 4

AGENTS_HIDDEN_DIM = 256
AGENTS_DROPOUT_RATE = 0.1
AGENTS_LR = 2e-5
AGENTS_WEIGHT_DECAY = 1e-2
AGENTS_LR_STEP_SIZE = 3
AGENTS_LR_GAMMA = 0.7

N_PARALLELS = 200
RANDOM_SEED = 3407
CHECKPOINT_INTERVAL = 1000

TARGET_PEPTIDE = "RVKRVWPLVIRTVIAGYNLYRAIKKK"

REWARD_MODELS = ["ACP", "AFP", "AMP", "AVP", "HEM", "MRSA"]

HEM_CONCENTRATION = 50.0  # μg/mL, passed to HEM hemolysis predictor

# Per-model reward weights.
# HEM weight is set higher to compensate for the 4:1 structural imbalance
# (4 activity models maximising vs 1 hemolysis model minimising).
REWARD_WEIGHTS = {
    "AMP": 1.0,
    "ACP": 0.6,
    "AFP": 0.6,
    "AVP": 0.6,
    "HEM": 2.5,
    "MRSA": 1.0,
}

# Extra step-wise penalty when HEM probability exceeds this threshold.
# Applied on top of the weighted delta signal every step.
HEM_THRESHOLD     = 0.3   # 30% hemolysis at HEM_CONCENTRATION
HEM_PENALTY_SCALE = 1.0   # penalty magnitude per unit above threshold

# ── SQA Quantum Refinement (from SQA_AMP_RL.md §一) ──────────────
USE_SQA         = True    # enable SQA refinement at checkpoints
SQA_N_TROTTER   = 20      # Trotter slices (path-integral dimension)
SQA_N_STEPS     = 500     # annealing iterations
SQA_ALPHA       = 1.0     # attention → J_ij coupling strength
SQA_N_POSITIONS = 8       # top Actor1 positions to consider
SQA_N_AAS       = 3       # top Actor2 amino acids per position
