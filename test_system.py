"""Quick system validation script."""
import faulthandler
faulthandler.enable()
import gpu_setup
import warnings
warnings.filterwarnings("ignore")

import config
config.N_PARALLELS  = 2
config.TIME_HORIZON = 2

from peptide_optimization._utils import set_seeds, get_save_dir
from peptide_optimization.environment import Environment
from peptide_optimization.ppo import PPO
from peptide_optimization.sqa import refine_peptide

set_seeds()
print("Step 1: Environment init...")
env = Environment()
print("state_dim:", env.state_dim)

print("Step 2: PPO init...")
save_dir = get_save_dir()
ppo = PPO(env.state_dim, env.n_action1, env.n_action2, save_dir)
print("PPO OK")

print("Step 3: Episode...")
states = env.reset()
for step in range(config.TIME_HORIZON):
    a1, a2, lp1, lp2, pv = ppo.choose_actions(states)
    next_states, rewards, done = env.step(a1, a2)
    print("  step%d reward=%.4f done=%s" % (step + 1, float(rewards[0]), done))
    if done:
        break
    states = next_states.clone()

print("Final peptide:", env.peptides_T[0])

print("Step 4: SQA refinement...")
if config.ENCODING_SCHEME in ("PepBERT-large",):
    pepbert_model, pepbert_tokenizer = env.encoder.pepbert_large_model, env.encoder.pepbert_large_tokenizer
else:
    pepbert_model, pepbert_tokenizer = env.encoder.pepbert_small_model, env.encoder.pepbert_small_tokenizer

refined, mutations = refine_peptide(
    env.peptides_T[0], ppo.actor1, ppo.actor2, pepbert_model, pepbert_tokenizer, env.device,
    n_positions=4, n_aas_per_pos=2, n_trotter=4, n_steps=10,
)
assert len(refined) == len(env.peptides_T[0]), "SQA refinement changed sequence length"
print("Refined peptide:", refined, "mutations:", mutations)

print("=== SYSTEM TEST PASSED ===")
