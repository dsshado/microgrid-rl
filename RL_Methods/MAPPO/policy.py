import torch
import torch.nn as nn
from RL_Methods.MAPPO.encoder import GCAPSEncoder


class MAPPOActor(nn.Module):
    """Shared-parameter actor — same weights for all agents.
    Input : per-agent observation  (B, n_agents, obs_dim)
    Output: Bernoulli logit         (B, n_agents)
    """

    def __init__(self, obs_dim, features_dim=128):
        super().__init__()
        h = features_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 4 * h),
            nn.Tanh(),
            nn.Linear(4 * h, 4 * h),
            nn.Tanh(),
            nn.Linear(4 * h, 2 * h),
            nn.Tanh(),
            nn.Linear(2 * h, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)          # (B, n_agents)

    def get_dist(self, obs):
        return torch.distributions.Bernoulli(logits=self.forward(obs))


class MAPPOCritic(nn.Module):
    """Centralised critic — takes concatenated F_graph + F_context.
    Input : global state  (B, 2 * features_dim)
    Output: scalar value  (B, 1)
    """

    def __init__(self, features_dim=128):
        super().__init__()
        h = features_dim
        self.net = nn.Sequential(
            nn.Linear(2 * h, 4 * h),
            nn.Tanh(),
            nn.Linear(4 * h, 4 * h),
            nn.Tanh(),
            nn.Linear(4 * h, 1)
        )

    def forward(self, global_state):
        return self.net(global_state)             # (B, 1)


class MAPPOPolicy(nn.Module):
    """Full MAPPO policy: GCAPSEncoder + shared Actor + centralised Critic.

    Per-agent observation:
        o_i = [delta_i, mu_i, O_i, F_node_i, F_graph, F_context]
        shape: (3 + 3 * features_dim,)
    """

    def __init__(self, n_agents, features_dim=128, node_dim=3,
                 context_input_dim=140, agent_bus_indices=None, device='cpu'):
        super().__init__()
        self.n_agents     = n_agents
        self.features_dim = features_dim
        self.obs_dim      = 3 + 3 * features_dim  # local(3) + F_node + F_graph + F_context

        self.encoder = GCAPSEncoder(
            features_dim=features_dim,
            node_dim=node_dim,
            n_dim=features_dim,
            context_input_dim=context_input_dim
        )
        self.actor  = MAPPOActor(obs_dim=self.obs_dim, features_dim=features_dim)
        self.critic = MAPPOCritic(features_dim=features_dim)

        if agent_bus_indices is not None:
            self.register_buffer('agent_bus_idx',
                                 torch.tensor(agent_bus_indices, dtype=torch.long))
        else:
            self.register_buffer('agent_bus_idx',
                                 torch.zeros(n_agents, dtype=torch.long))

        self._device = device
        self.to(device)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _to_tensor(self, obs_dict):
        out = {}
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.float().to(self._device)
            else:
                out[k] = torch.tensor(v, dtype=torch.float32).to(self._device)
            if out[k].dim() == 1:
                out[k] = out[k].unsqueeze(0)   # add batch dim
        return out

    def encode(self, obs_dict):
        data = self._to_tensor(obs_dict)
        return self.encoder(data)              # F_nodes, F_graph, F_context

    def build_agent_obs(self, obs_dict, F_nodes, F_graph, F_context, current_actions):
        """Build (B, n_agents, obs_dim) observation tensor."""
        B = F_graph.shape[0]
        mask         = self._to_tensor(obs_dict)['ActionMasking']  # (B, n_agents)
        cur_act      = current_actions.float().to(self._device)     # (B, n_agents)

        bus_idx      = self.agent_bus_idx.to(self._device)          # (n_agents,)
        F_node_agents = F_nodes[:, bus_idx, :]                      # (B, n_agents, h)
        F_graph_exp  = F_graph.unsqueeze(1).expand(B, self.n_agents, -1)
        F_ctx_exp    = F_context.unsqueeze(1).expand(B, self.n_agents, -1)

        delta_i = cur_act.unsqueeze(-1)   # (B, n_agents, 1)
        mu_i    = mask.unsqueeze(-1)       # (B, n_agents, 1)
        O_i     = mu_i.clone()

        return torch.cat([delta_i, mu_i, O_i, F_node_agents, F_graph_exp, F_ctx_exp], dim=-1)

    # ── forward passes ───────────────────────────────────────────────────────

    def get_actions(self, obs_dict, current_actions, deterministic=False):
        """Collect actions, log-probs, and values for one environment step."""
        with torch.no_grad():
            F_nodes, F_graph, F_context = self.encode(obs_dict)
            agent_obs   = self.build_agent_obs(obs_dict, F_nodes, F_graph, F_context, current_actions)
            dist        = self.actor.get_dist(agent_obs)
            actions     = dist.probs.gt(0.5).float() if deterministic else dist.sample()
            log_probs   = dist.log_prob(actions)
            global_state = torch.cat([F_graph, F_context], dim=-1)
            values      = self.critic(global_state)                 # (B, 1)

            # Outage masking: faulted agents keep current device status
            mask    = self._to_tensor(obs_dict)['ActionMasking']
            actions = torch.where(mask.bool(), current_actions.float().to(self._device), actions)

        return actions, log_probs, values, agent_obs, global_state

    def evaluate_actions(self, agent_obs, actions, global_state):
        """Re-evaluate stored (agent_obs, actions) for PPO update."""
        dist      = self.actor.get_dist(agent_obs)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        values    = self.critic(global_state)
        return log_probs, values, entropy
