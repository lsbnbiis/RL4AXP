import os
import config
import torch as T
import pandas as pd

from tqdm import tqdm
from peptide_optimization.ppo import PPO
from peptide_optimization._utils import *
from peptide_optimization.environment import Environment

class Framework:

    def __init__(self) -> None:

        set_seeds()
        self.save_dir = get_save_dir()
        save_config(config, self.save_dir)

        self.env = Environment()
        self.device = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")

        self.state_dim = self.env.state_dim
        self.n_action1 = self.env.n_action1
        self.n_action2 = self.env.n_action2

        self.agent = PPO(self.state_dim, self.n_action1, self.n_action2, self.save_dir)

        prob_cols = [f"{m}-Prob_T" for m in config.REWARD_MODELS]
        self.exp_results_df = pd.DataFrame(
            columns=["Episode", "Peptide_T"] + prob_cols + ["Heuristic_T", "Cumulative-Reward", "Action1s", "Action2s"]
        )

    def train(self, on_episode_end=None, stop_event=None, resume: bool = False) -> None:

        if not resume:
            self.episode = 0
            self.loss_func_data = {"actor1_loss": [], "actor2_loss": [], "critic_loss": [], "entropy1": [], "entropy2": []}
            self.lr_data = []
        elif not hasattr(self, "loss_func_data"):
            self.loss_func_data = {"actor1_loss": [], "actor2_loss": [], "critic_loss": [], "entropy1": [], "entropy2": []}
            self.lr_data = []

        with tqdm(total=config.N_EPISODES, initial=self.episode, desc="Training Peptide Optimizer", unit="episode") as bar:

            while self.episode < config.N_EPISODES:

                if stop_event is not None and stop_event.is_set():
                    break

                self.trjs = {
                    "states": [], "action1s": [], "action2s": [], "rewards": [],
                    "log_prob1s": [], "log_prob2s": [], "pred_values": []
                }

                states = self.env.reset()
                while True:

                    action1s, action2s, log_prob1s, log_prob2s, pred_values = self.agent.choose_actions(states)
                    next_states, rewards, done = self.env.step(action1s, action2s)
                    self._update_trjs(states, action1s, action2s, rewards, log_prob1s, log_prob2s, pred_values)

                    if done:

                        self.agent.buffer.store_trjs(self.trjs)

                        if len(self.agent.buffer.data["states"]) >= config.BUFFER_SIZE:

                            actor1_loss, actor2_loss, critic_loss, entropy1, entropy2 = self.agent.learn()
                            self._update_loss_data(actor1_loss, actor2_loss, critic_loss, entropy1, entropy2)
                            self.lr_data.append(self.agent.get_lr())

                        break

                    else:

                        states = next_states.clone()

                self._update_exp_results_df()
                self.episode += config.N_PARALLELS
                bar.update(config.N_PARALLELS)

                if on_episode_end is not None:
                    on_episode_end(
                        self.episode,
                        self.exp_results_df.copy(),
                        {k: list(v) for k, v in self.loss_func_data.items()},
                        list(self.lr_data),
                    )

                if self.episode % config.CHECKPOINT_INTERVAL == 0:

                    self.exp_results_df.to_csv(os.path.join(self.save_dir, "exp_results.csv"), index=False)
                    self._plot_exp_results()
                    self.agent.save_agent()

    def _update_trjs(
            self, states: T.Tensor, action1s: T.Tensor, action2s: T.Tensor, rewards: T.Tensor,
            log_prob1s: T.Tensor, log_prob2s: T.Tensor, pred_values: T.Tensor
        ) -> None:

        self.trjs["states"].append(states)
        self.trjs["action1s"].append(action1s)
        self.trjs["action2s"].append(action2s)
        self.trjs["rewards"].append(rewards)
        self.trjs["log_prob1s"].append(log_prob1s)
        self.trjs["log_prob2s"].append(log_prob2s)
        self.trjs["pred_values"].append(pred_values)

    def _update_loss_data(self, actor1_loss: float, actor2_loss: float, critic_loss: float, entropy1: float, entropy2: float) -> None:

        self.loss_func_data["actor1_loss"].append(actor1_loss)
        self.loss_func_data["actor2_loss"].append(actor2_loss)
        self.loss_func_data["critic_loss"].append(critic_loss)
        self.loss_func_data["entropy1"].append(entropy1)
        self.loss_func_data["entropy2"].append(entropy2)

    def _update_exp_results_df(self) -> None:

        peptides_T = self.env.peptides_T
        rewards = T.stack(self.trjs["rewards"], dim=1)
        cumulative_rewards = rewards.sum(axis=1).tolist()

        probs_T = {m: self.env.probs_curr[m].cpu().tolist() for m in config.REWARD_MODELS}
        heuristic_T = self.env.heuristic_curr.cpu().tolist()
        action1s = T.stack(self.trjs["action1s"], dim=1).tolist()
        action2s = T.stack(self.trjs["action2s"], dim=1).tolist()

        new_rows = []
        for n in range(config.N_PARALLELS):

            new_rows.append({
                "Episode": f"{self.episode + n + 1:06d}",
                "Peptide_T": peptides_T[n],
                **{f"{m}-Prob_T": f"{probs_T[m][n]:.4f}" for m in config.REWARD_MODELS},
                "Heuristic_T": f"{heuristic_T[n]:+07.4f}",
                "Cumulative-Reward": f"{cumulative_rewards[n]:+06.4f}",
                "Action1s": "|".join(f"{a:02d}" for a in action1s[n]),
                "Action2s": "|".join(f"{a:02d}" for a in action2s[n]),
            })

        if len(self.exp_results_df) == 0:
            self.exp_results_df = pd.DataFrame(new_rows)
        else:
            self.exp_results_df = pd.concat([self.exp_results_df, pd.DataFrame(new_rows)], ignore_index=True)

    def _plot_exp_results(self) -> None:

        plot_single_smooth(
            y_data=self.exp_results_df["Cumulative-Reward"].astype(float).to_list(),
            x_label="Episodes", y_label="Cumulative Reward", title="Cumulative Reward Across Episodes",
            fig_name="cumulative_reward", save_dir=self.save_dir
        )

        probs_cols = [f"{m}-Prob_T" for m in config.REWARD_MODELS]
        probs_data = self.exp_results_df[probs_cols].astype(float).values.T
        legends = [f"{m} Probability" for m in config.REWARD_MODELS]

        plot_multip_smooth(
            y_data=probs_data, x_label="Episodes", y_label="Probability",
            title="Peptide Activity / Hemolysis Probability Across Episodes",
            fig_name="probs_score", legends=legends, save_dir=self.save_dir
        )

        plot_single_smooth(
            y_data=self.exp_results_df["Heuristic_T"].astype(float).to_list(),
            x_label="Episodes", y_label="Heuristic Score", title="Heuristic Score of Optimized Peptides Across Episodes",
            fig_name="heuristic_score", save_dir=self.save_dir
        )

        plot_single_smooth(
            y_data=self.loss_func_data["actor1_loss"], x_label="Learning Steps", y_label="Actor1 Loss",
            title="Actor1 Loss Across Learning Steps", fig_name="actor1_loss", save_dir=self.save_dir, sigma=1
        )

        plot_single_smooth(
            y_data=self.loss_func_data["actor2_loss"], x_label="Learning Steps", y_label="Actor2 Loss",
            title="Actor2 Loss Across Learning Steps", fig_name="actor2_loss", save_dir=self.save_dir, sigma=1
        )

        plot_single_smooth(
            y_data=self.loss_func_data["critic_loss"], x_label="Learning Steps", y_label="Critic Loss",
            title="Critic Loss Across Learning Steps", fig_name="critic_loss", save_dir=self.save_dir, sigma=1
        )

        plot_single_smooth(
            y_data=self.loss_func_data["entropy1"], x_label="Learning Steps", y_label="Entropy Bonus1",
            title="Entropy Bonus1 Across Learning Steps", fig_name="entropy1", save_dir=self.save_dir, sigma=1
        )

        plot_single_smooth(
            y_data=self.loss_func_data["entropy2"], x_label="Learning Steps", y_label="Entropy Bonus2",
            title="Entropy Bonus2 Across Learning Steps", fig_name="entropy2", save_dir=self.save_dir, sigma=1
        )

        plot_single_smooth(
            y_data=self.lr_data, x_label="Learning Steps", y_label="Learning Rate",
            title="Learning Rate Across Learning Steps", fig_name="learning_rate", save_dir=self.save_dir, sigma=1
        )
