"""MAPPO trainer — Wang et al. (2025) Eqs. 23-33, v2 spec."""

import csv
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from RL_Methods.MAPPO.buffer import MAPPORolloutBuffer


def lr_schedule(initial_lr: float, decay_rate: float = 3):
    def func(progress_remaining: float) -> float:
        return initial_lr * math.exp(-(1 - progress_remaining) ** 2 * decay_rate)
    return func


class MAPPO:
    """Multi-Agent PPO with shared GCAPSEncoder and centralised critic.

    References
    ----------
    Wang et al. (2025) https://doi.org/10.1007/s00521-024-10654-9  (Eqs. 23-33)
    Yu   et al. (2022) https://arxiv.org/abs/2103.01955             (parameter sharing)
    """

    def __init__(self, env, policy, cfg, n_sect, n_tie, n_load):
        self.env    = env
        self.policy = policy
        self.cfg    = cfg
        self.device = policy._device

        # Per-bus initial switch/load statuses (sectionalizing: closed, tie: open, loads: served)
        self._n_sect = n_sect
        self._n_tie  = n_tie
        self._n_load = n_load

        n_agents = policy.n_agents
        self.buffer = MAPPORolloutBuffer(
            n_steps=cfg.n_steps,
            n_agents=n_agents,
            obs_dim=policy.obs_dim,
            global_dim=policy.features_dim * 2,
            device=self.device,
        )

        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=cfg.learning_rate,
            eps=1e-5
        )

        self.tb_writer = SummaryWriter(log_dir=cfg.logger + f"MAPPO_GCAPS_{cfg.bus_size}bus")
        self._global_step = 0
        self._conv_rows: list = []   # accumulates per-update metrics for CSV/NPZ

    # ── default initial actions after reset ─────────────────────────────────

    def _default_actions(self):
        act = np.zeros(self.policy.n_agents, dtype=np.float32)
        act[:self._n_sect]                      = 1.0   # sectionalizing closed
        act[self._n_sect + self._n_tie:]        = 1.0   # loads served
        return act

    # ── observation → tensor (unbatched) ────────────────────────────────────

    def _obs_to_tensor(self, obs):
        return {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
                for k, v in obs.items()}

    # ── rollout collection ───────────────────────────────────────────────────

    def collect_rollouts(self):
        self.buffer.reset()
        obs, _ = self.env.reset()
        current_actions = torch.tensor(self._default_actions()).unsqueeze(0).to(self.device)

        done = False
        for _ in range(self.cfg.n_steps):
            obs_t = self._obs_to_tensor(obs)

            actions, log_probs, values, agent_obs, global_state = \
                self.policy.get_actions(obs_t, current_actions)

            joint_action = actions.squeeze(0).cpu().numpy().astype(int)
            next_obs, reward, terminated, truncated, _ = self.env.step(joint_action)
            done = terminated or truncated

            # One-step TD: V(S') = 0 if episode ended, else compute from next obs
            if done:
                next_val = torch.zeros(1, 1, device=self.device)
            else:
                obs_next_t = self._obs_to_tensor(next_obs)
                with torch.no_grad():
                    F_n, F_g, F_c = self.policy.encode(obs_next_t)
                    next_val = self.policy.critic(torch.cat([F_g, F_c], dim=-1))

            mask = obs_t['ActionMasking'].squeeze(0)

            self.buffer.add(
                agent_obs.squeeze(0),
                actions.squeeze(0),
                reward,
                values.squeeze(0),
                next_val.squeeze(0),
                log_probs.squeeze(0),
                mask,
                global_state.squeeze(0),
                done,
            )

            current_actions = actions.detach()
            self._global_step += 1

            if done:
                obs, _ = self.env.reset()
                current_actions = torch.tensor(self._default_actions()).unsqueeze(0).to(self.device)
            else:
                obs = next_obs

        self.buffer.compute_advantages(gamma=self.cfg.gamma)

    # ── policy update ────────────────────────────────────────────────────────

    def update(self):
        clip_eps         = 0.2
        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy    = 0.0
        n_batches        = 0

        for _ in range(self.cfg.n_epochs):
            for agent_obs, actions, advantages, returns, old_log_probs, masks, global_states \
                    in self.buffer.get_batches(self.cfg.batch_size):

                log_probs, values, entropy = self.policy.evaluate_actions(
                    agent_obs, actions, global_states
                )

                # ── actor loss (Wang et al. 2025, Eqs. 28-31) ───────────────
                ratio  = torch.exp(log_probs - old_log_probs)          # (B, n_agents)
                surr1  = ratio * advantages
                surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

                active = (1.0 - masks)                                  # active agents only
                denom  = active.sum().clamp(min=1)

                actor_loss = -(torch.min(surr1, surr2) * active).sum() / denom

                # ── critic loss (Wang et al. 2025, Eq. 32) ──────────────────
                value_targets = returns.mean(dim=-1, keepdim=True)     # (B, 1)
                value_loss    = F.mse_loss(values, value_targets)

                # ── entropy bonus ────────────────────────────────────────────
                entropy_loss  = -(entropy * active).sum() / denom

                loss = actor_loss + self.cfg.val_coef * value_loss + self.cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy    += (-entropy_loss).item()
                n_batches        += 1

        n_batches = max(n_batches, 1)
        mean_reward = float(self.buffer.rewards.mean())
        metrics = {
            "actor_loss":  total_actor_loss / n_batches,
            "value_loss":  total_value_loss / n_batches,
            "entropy":     total_entropy    / n_batches,
            "mean_reward": mean_reward,
        }
        self.tb_writer.add_scalar("train/actor_loss",  metrics["actor_loss"],  self._global_step)
        self.tb_writer.add_scalar("train/value_loss",  metrics["value_loss"],  self._global_step)
        self.tb_writer.add_scalar("train/entropy",     metrics["entropy"],     self._global_step)
        self.tb_writer.add_scalar("train/mean_reward", metrics["mean_reward"], self._global_step)
        return metrics

    # ── main training loop ───────────────────────────────────────────────────

    def learn(self, total_timesteps, save_path, save_freq):
        n_updates    = total_timesteps // self.cfg.n_steps
        steps_done   = 0
        next_save    = save_freq

        print(f"MAPPO training: {total_timesteps} steps / {n_updates} updates")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        csv_path = save_path + "_convergence.csv"
        _csv_fields = ["steps", "update", "actor_loss", "value_loss", "entropy", "mean_reward"]
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=_csv_fields)
        csv_writer.writeheader()

        for update in range(1, n_updates + 1):
            progress = 1.0 - steps_done / total_timesteps
            for g in self.optimizer.param_groups:
                g['lr'] = lr_schedule(self.cfg.learning_rate)(progress)

            self.collect_rollouts()
            metrics    = self.update()
            steps_done += self.cfg.n_steps

            row = {"steps": steps_done, "update": update, **metrics}
            csv_writer.writerow(row)
            csv_file.flush()
            self._conv_rows.append(row)

            print(f"Update {update}/{n_updates} | steps {steps_done} | "
                  f"actor_loss {metrics['actor_loss']:.4f} | reward {metrics['mean_reward']:.4f}")

            if steps_done >= next_save:
                ckpt = save_path + f"_step{steps_done}"
                torch.save(self.policy.state_dict(), ckpt + ".pt")
                print(f"  Checkpoint saved: {ckpt}.pt")
                next_save += save_freq

        csv_file.close()

        # save NPZ for easy numpy loading
        arr = {k: np.array([r[k] for r in self._conv_rows]) for k in _csv_fields}
        np.savez(save_path + "_convergence.npz", **arr)

        torch.save(self.policy.state_dict(), save_path + "_final.pt")
        print(f"Training complete. Model saved to {save_path}_final.pt")
        print(f"Convergence data : {csv_path}")
        self.tb_writer.close()
