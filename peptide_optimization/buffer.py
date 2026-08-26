import config
import torch as T
import numpy as np

class ReplayBuffer:

    def __init__(self) -> None:

        self.data: dict[str, list[T.Tensor]] = {
            "states": [], "action1s": [], "action2s": [], "action_masks": [],
            "log_prob1s": [], "log_prob2s": [], "pred_values": [], "returns": [], "gaes": []
        }

    def store_trjs(self, trjs: dict[str, list[T.Tensor]]) -> None:
        
        states = T.stack(trjs["states"], dim=1)
        rewards = T.stack(trjs["rewards"], dim=1)
        pred_values = T.stack(trjs["pred_values"], dim=1)

        self.data["states"].extend(states.reshape(-1, states.shape[-1]).cpu())
        self.data["action1s"].extend(T.stack(trjs["action1s"], dim=1).flatten())
        self.data["action2s"].extend(T.stack(trjs["action2s"], dim=1).flatten())
        if "action_masks" in trjs and len(trjs["action_masks"]) > 0:
            action_masks = T.stack(trjs["action_masks"], dim=1)
            self.data["action_masks"].extend(action_masks.reshape(-1, action_masks.shape[-1]).cpu())
        self.data["log_prob1s"].extend(T.stack(trjs["log_prob1s"], dim=1).flatten())
        self.data["log_prob2s"].extend(T.stack(trjs["log_prob2s"], dim=1).flatten())
        self.data["pred_values"].extend(pred_values.flatten())
        
        self.data["returns"].extend(self._calculate_returns(rewards))
        self.data["gaes"].extend(self._calculate_gae(rewards, pred_values))

    def _calculate_returns(self, rewards: T.Tensor) -> T.Tensor:

        returns = T.zeros_like(rewards) # (N, T)
        for n in range(config.N_PARALLELS):

            ret = 0
            for t in reversed(range(config.TIME_HORIZON)):
                ret = rewards[n, t] + config.GAMMA * ret
                returns[n, t] = ret
        
        return returns.flatten() # (N x T, )
    
    def _calculate_gae(self, rewards: T.Tensor, pred_values: T.Tensor) -> T.Tensor:

        pred_values = T.cat([pred_values, T.zeros(config.N_PARALLELS, 1)], dim=1)

        advantages = T.zeros_like(rewards) # (N, T)
        next_advantages = T.zeros(config.N_PARALLELS)

        for t in reversed(range(config.TIME_HORIZON)):
            delta = rewards[:, t] + config.GAMMA * pred_values[:, t + 1] - pred_values[:, t]
            next_advantages = delta + config.GAMMA * config.LAMBDA * next_advantages
            advantages[:, t] = next_advantages

        return advantages.flatten() # (N x T, )
    
    def get_train_data(self) -> tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor | None]:

        states = T.stack(self.data["states"])
        action1s = T.stack(self.data["action1s"])
        action2s = T.stack(self.data["action2s"])
        action_masks = T.stack(self.data["action_masks"]) if len(self.data["action_masks"]) == len(self.data["states"]) else None
        old_log_prob1s = T.stack(self.data["log_prob1s"])
        old_log_prob2s = T.stack(self.data["log_prob2s"])
        returns = T.stack(self.data["returns"])

        gaes = T.stack(self.data["gaes"])
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return states, action1s, action2s, action_masks, old_log_prob1s, old_log_prob2s, returns, gaes

    def get_batch_indices(self) -> list[list[int]]:

        n_data = len(self.data["states"])
        all_indices = np.random.permutation(n_data)

        return [all_indices[i:i + config.BATCH_SIZE] for i in range(0, n_data, config.BATCH_SIZE)]

    def clear(self) -> None:
        
        self.data: dict[str, list[T.Tensor]] = {
            "states": [], "action1s": [], "action2s": [], "action_masks": [],
            "log_prob1s": [], "log_prob2s": [], "pred_values": [], "returns": [], "gaes": []
        }
    