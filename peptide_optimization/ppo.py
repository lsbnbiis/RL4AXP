import os
import config
import torch as T

from peptide_optimization.buffer import ReplayBuffer
from torch.distributions.categorical import Categorical
from peptide_optimization.actors_critic import Actor1, Actor2, Critic

DEVICE = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")

class PPO:

    def __init__(self, state_dim: int, n_action1: int, n_action2: int, save_dir: str) -> None:

        self.n_action1 = n_action1
        self.n_action2 = n_action2

        actor1_path = os.path.join(save_dir, "ppo_actor1.pt")
        actor2_path = os.path.join(save_dir, "ppo_actor2.pt")
        critic_path = os.path.join(save_dir, "ppo_critic.pt")

        self.actor1 = Actor1(state_dim, n_action1, actor1_path)
        self.actor2 = Actor2(state_dim, n_action1, n_action2, actor2_path)
        self.critic = Critic(state_dim, critic_path)

        self.buffer = ReplayBuffer()

    def save_agent(self) -> None:

        self.actor1.save_model()
        self.actor2.save_model()
        self.critic.save_model()

    def load_agent(self) -> None:

        self.actor1.load_model()
        self.actor2.load_model()
        self.critic.load_model()

    def choose_actions(self, states: T.Tensor, action_mask_fn=None) -> tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor | None]:

        with T.no_grad():

            probs: T.Tensor = self.actor1(states) # (N, D)
            dists = Categorical(probs.cpu())
            action1s: T.Tensor = dists.sample() # (N, )
            log_prob1s: T.Tensor = dists.log_prob(action1s) # (N, )

            action1s_ = action1s.to(DEVICE)
            action_masks = action_mask_fn(action1s) if action_mask_fn is not None else None
            mask_ = action_masks.to(DEVICE) if action_masks is not None else None
            probs = self.actor2(states, action1s_, action_mask=mask_) # (N, D)
            dists = Categorical(probs.cpu())
            action2s: T.Tensor = dists.sample() # (N, )
            log_prob2s: T.Tensor = dists.log_prob(action2s) # (N, )

            pred_values: T.Tensor = self.critic(states)
            pred_values = pred_values.squeeze(1).cpu()

        return action1s, action2s, log_prob1s, log_prob2s, pred_values, action_masks
    
    def learn(self) -> tuple[float, float, float, float, float]:

        states, action1s, action2s, action_masks, old_log_prob1s, old_log_prob2s, returns, gaes = self.buffer.get_train_data()

        for _ in range(config.N_EPOCHS):
            actor1_epoch_loss = actor2_epoch_loss = critic_epoch_loss = 0
            entropy1_epoch_bonus = entropy2_epoch_bonus = 0

            for batch_indices in self.buffer.get_batch_indices():

                states_batch = states[batch_indices].to(DEVICE)
                action1s_batch = action1s[batch_indices].to(DEVICE)
                action2s_batch = action2s[batch_indices].to(DEVICE)
                mask_batch = action_masks[batch_indices].to(DEVICE) if action_masks is not None else None
                old_log_prob1s_batch = old_log_prob1s[batch_indices].to(DEVICE)
                old_log_prob2s_batch = old_log_prob2s[batch_indices].to(DEVICE)
                returns_batch = returns[batch_indices].to(DEVICE)
                gaes_batch = gaes[batch_indices].to(DEVICE)

                dists = Categorical(self.actor1(states_batch))
                new_log_prob1s_batch: T.Tensor = dists.log_prob(action1s_batch)
                entropy_bonus1: T.Tensor = dists.entropy().mean()

                ratio1 = new_log_prob1s_batch.exp() / old_log_prob1s_batch.exp()
                surrogate1 = T.clamp(ratio1, 1 - config.EPSILON, 1 + config.EPSILON)
                actor1_loss = -T.min(ratio1 * gaes_batch, surrogate1 * gaes_batch).mean()

                dists = Categorical(self.actor2(states_batch, action1s_batch, action_mask=mask_batch))
                new_log_prob2s_batch: T.Tensor = dists.log_prob(action2s_batch)
                entropy_bonus2: T.Tensor = dists.entropy().mean()

                ratio2 = new_log_prob2s_batch.exp() / old_log_prob2s_batch.exp()
                surrogate2 = T.clamp(ratio2, 1 - config.EPSILON, 1 + config.EPSILON)
                actor2_loss = -T.min(ratio2 * gaes_batch, surrogate2 * gaes_batch).mean()

                pred_values_batch: T.Tensor = self.critic(states_batch)
                critic_loss = 0.5 * (returns_batch - pred_values_batch.squeeze(1)).pow(2).mean()

                # entropy_bonus1 /= T.log(T.tensor(self.n_action1, dtype=T.float32).to(DEVICE))
                # entropy_bonus2 /= T.log(T.tensor(self.n_action2, dtype=T.float32).to(DEVICE))
                entropy_bonus = config.BETA * (entropy_bonus1 + entropy_bonus2)

                loss = actor1_loss + actor2_loss + critic_loss - entropy_bonus

                self.actor1.optimizer.zero_grad()
                self.actor2.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                loss.backward()

                self.actor1.optimizer.step()
                self.actor2.optimizer.step()
                self.critic.optimizer.step()

                actor1_epoch_loss += actor1_loss.item()
                actor2_epoch_loss += actor2_loss.item()
                critic_epoch_loss += critic_loss.item()
                entropy1_epoch_bonus += entropy_bonus1.item()
                entropy2_epoch_bonus += entropy_bonus2.item()

        actor1_epoch_loss /= config.N_EPOCHS
        actor2_epoch_loss /= config.N_EPOCHS
        critic_epoch_loss /= config.N_EPOCHS
        entropy1_epoch_bonus /= config.N_EPOCHS
        entropy2_epoch_bonus /= config.N_EPOCHS

        self.actor1.scheduler.step()
        self.actor2.scheduler.step()
        self.critic.scheduler.step()
        self.buffer.clear()

        return actor1_epoch_loss, actor2_epoch_loss, critic_epoch_loss, entropy1_epoch_bonus, entropy2_epoch_bonus

    def get_lr(self) -> float:

        return self.actor1.optimizer.param_groups[0]["lr"]