"""Quick system validation script."""
import faulthandler
faulthandler.enable()
import gpu_setup
import warnings
warnings.filterwarnings("ignore")

import config
config.N_PARALLELS  = 4
config.TIME_HORIZON = 2
config.BUFFER_SIZE  = 8

from peptide_optimization._utils import set_seeds, get_save_dir
from peptide_optimization.environment import Environment
from peptide_optimization.ppo import PPO
from peptide_optimization.sqa import refine_peptide

set_seeds()
print("Step 1: Environment init...")
env = Environment()
print("state_dim:", env.state_dim, "models:", env.reward_models)
assert "MRSA" in env.reward_models, "MRSA missing from reward models!"

print("Step 2: PPO init...")
save_dir = get_save_dir()
ppo = PPO(env.state_dim, env.n_action1, env.n_action2, save_dir)
print("PPO OK")

print("Step 3: Rollout with Action Masking...")
trjs = {
    "states": [], "action1s": [], "action2s": [], "action_masks": [], "rewards": [],
    "log_prob1s": [], "log_prob2s": [], "pred_values": []
}
states = env.reset()
for step in range(config.TIME_HORIZON):
    orig_peps = env.peptides_curr.copy()
    a1, a2, lp1, lp2, pv, masks = ppo.choose_actions(states, action_mask_fn=env.get_action_masks)
    
    # Verify action masking (no identity mutations)
    for i, (pep, pos, new_aa_idx) in enumerate(zip(orig_peps, a1.tolist(), a2.tolist())):
        new_aa = env.a2_to_aa[new_aa_idx]
        assert pep[pos] != new_aa, f"Action mask failed: pep[{pos}] was {pep[pos]} and remained {new_aa}"
        assert masks[i, env.aa_to_a2[pep[pos]]] == False, "Mask should be False for current amino acid"

    next_states, rewards, done = env.step(a1, a2)
    print("  step%d reward=%.4f done=%s" % (step + 1, float(rewards[0]), done))
    
    trjs["states"].append(states)
    trjs["action1s"].append(a1)
    trjs["action2s"].append(a2)
    trjs["action_masks"].append(masks)
    trjs["rewards"].append(rewards)
    trjs["log_prob1s"].append(lp1)
    trjs["log_prob2s"].append(lp2)
    trjs["pred_values"].append(pv)

    if done:
        break
    states = next_states.clone()

print("Final peptide:", env.peptides_T[0])

print("Step 4: PPO Learn with Action Masks...")
ppo.buffer.store_trjs(trjs)
losses = ppo.learn()
print(f"Actor1 Loss: {losses[0]:.4f}, Actor2 Loss: {losses[1]:.4f}, Critic Loss: {losses[2]:.4f}")

print("Step 5: SQA refinement...")
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
