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

    def compute_advantages(self, last_value, gamma=0.99):
        """Per-agent truncated advantage: A_i_t = G_t - V(o_i_t).

        Bootstrap is zeroed at episode boundaries so that done=True steps
        do not propagate value estimates from the next episode's start state.
        """
        last_val = float(self._np(last_value).squeeze())

        # If the rollout ended mid-episode, bootstrap from last_value;
        # if it ended at a terminal step, last_val is already 0 (set by caller).
        next_val = last_val
        for t in reversed(range(self.n_steps)):
            # At a terminal step the episode ended — no future value to bootstrap.
            if self.dones[t]:
                next_val = 0.0
            G_t = self.rewards[t] + gamma * next_val
            self.returns[t]    = G_t                           # broadcast to all agents
            self.advantages[t] = G_t - self.values[t]         # per-agent advantage
            next_val = self.values[t]

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
