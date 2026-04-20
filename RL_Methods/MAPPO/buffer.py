import numpy as np
import torch


class MAPPORolloutBuffer:
    """Per-agent rollout buffer for MAPPO (Wang et al. 2025, Eq. 23).

    Stores n_steps transitions. Each transition contains:
      - agent_obs   : per-agent observation  (n_agents, obs_dim)
      - actions     : binary actions          (n_agents,)
      - rewards     : shared scalar reward
      - values      : critic value estimate   (scalar, same for all agents)
      - log_probs   : per-agent log-prob      (n_agents,)
      - masks       : outage mask mu_i        (n_agents,)
      - global_state: F_graph + F_context     (global_dim,)
    """

    def __init__(self, n_steps, n_agents, obs_dim, global_dim, device='cpu'):
        self.n_steps    = n_steps
        self.n_agents   = n_agents
        self.obs_dim    = obs_dim
        self.global_dim = global_dim
        self.device     = device
        self.reset()

    def reset(self):
        self.agent_obs    = np.zeros((self.n_steps, self.n_agents, self.obs_dim),  dtype=np.float32)
        self.actions      = np.zeros((self.n_steps, self.n_agents),                dtype=np.float32)
        self.rewards      = np.zeros(self.n_steps,                                 dtype=np.float32)
        self.values       = np.zeros(self.n_steps,                                 dtype=np.float32)
        self.log_probs    = np.zeros((self.n_steps, self.n_agents),                dtype=np.float32)
        self.masks        = np.zeros((self.n_steps, self.n_agents),                dtype=np.float32)
        self.global_states = np.zeros((self.n_steps, self.global_dim),             dtype=np.float32)
        self.dones        = np.zeros(self.n_steps,                                 dtype=np.float32)
        self.advantages   = np.zeros((self.n_steps, self.n_agents),                dtype=np.float32)
        self.returns      = np.zeros((self.n_steps, self.n_agents),                dtype=np.float32)
        self.pos  = 0
        self.full = False

    # ── numpy helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # ── add one step ─────────────────────────────────────────────────────────

    def add(self, agent_obs, actions, reward, value, log_probs, masks, global_state, done):
        self.agent_obs[self.pos]     = self._np(agent_obs)
        self.actions[self.pos]       = self._np(actions)
        self.rewards[self.pos]       = float(reward)
        self.values[self.pos]        = float(self._np(value).squeeze())
        self.log_probs[self.pos]     = self._np(log_probs)
        self.masks[self.pos]         = self._np(masks)
        self.global_states[self.pos] = self._np(global_state)
        self.dones[self.pos]         = float(done)
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True

    # ── advantage computation (Wang et al. 2025, Eq. 27) ────────────────────

    def compute_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Per-agent GAE (Schulman et al. 2016).

        delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_hat_t   = delta_t + (gamma * lambda) * (1 - done_t) * A_hat_{t+1}

        lambda=1.0 recovers Wang et al. (2025) Eq. 27 (truncated return).
        lambda=0.95 is the standard PPO setting (stable-baselines3 default).
        """
        last_val    = float(self._np(last_value).squeeze())
        gae         = 0.0

        for t in reversed(range(self.n_steps)):
            not_done    = 1.0 - self.dones[t]
            next_val    = last_val if t == self.n_steps - 1 else self.values[t + 1]
            next_val   *= not_done                             # zero bootstrap at episode end

            delta       = self.rewards[t] + gamma * next_val - self.values[t]
            gae         = delta + gamma * gae_lambda * not_done * gae

            self.advantages[t] = gae                          # broadcast to all agents
            self.returns[t]    = gae + self.values[t]         # V + A = G_t

    # ── mini-batch generator ─────────────────────────────────────────────────

    def get_batches(self, batch_size):
        indices = np.random.permutation(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start:start + batch_size]
            yield (
                torch.tensor(self.agent_obs[idx],    dtype=torch.float32).to(self.device),
                torch.tensor(self.actions[idx],      dtype=torch.float32).to(self.device),
                torch.tensor(self.advantages[idx],   dtype=torch.float32).to(self.device),
                torch.tensor(self.returns[idx],      dtype=torch.float32).to(self.device),
                torch.tensor(self.log_probs[idx],    dtype=torch.float32).to(self.device),
                torch.tensor(self.masks[idx],        dtype=torch.float32).to(self.device),
                torch.tensor(self.global_states[idx],dtype=torch.float32).to(self.device),
            )
