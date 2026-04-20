"""GCAPS feature extractor for PPO (stable-baselines3 compatible)."""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GCAPSExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 128,
                 n_dim: int = 128,
                 n_p: int = 3,
                 node_dim: int = 3,
                 n_K: int = 2,
                 gnn_type: str = 'GCAPS'):
        super().__init__(observation_space, features_dim)
        self.n_dim    = n_dim
        self.n_p      = n_p
        self.n_K      = n_K
        self.node_dim = node_dim

        self.init_embed  = nn.Linear(node_dim, n_dim * n_p)
        self.W_L_1_G1    = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G2    = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G3    = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_F         = nn.Linear(n_dim * n_p, features_dim)

        self.full_context_nn = nn.Sequential(
            nn.Linear(140, 2 * features_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * features_dim, features_dim)
        )
        self.switch_encoder = nn.Sequential(
            nn.Linear(130, 2 * features_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * features_dim, features_dim)
        )
        self.final_encoder = nn.Sequential(
            nn.Linear(2 * features_dim, 4 * features_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * features_dim, features_dim)
        )

    def forward(self, data):
        X  = data['NodeFeat(BusVoltage)']   # (B, N, 3)
        A  = data['Adjacency']               # (B, N, N)
        B, N, _ = X.size()

        D  = torch.mul(
            torch.eye(N).expand(B, N, N).to(A.device),
            A.sum(-1)[:, None].expand(B, N, N)
        )
        L  = D - A
        L2 = torch.matmul(L, L)

        F0  = self.init_embed(X)
        inp = torch.cat([F0, torch.matmul(L, F0), torch.matmul(L2, F0)], dim=-1)

        g1 = self.W_L_1_G1(inp)
        g2 = self.W_L_1_G2(inp)
        g3 = self.W_L_1_G3(inp)

        F1      = torch.cat([g1, g2, g3], dim=-1)
        h       = self.W_F(F1)

        sw_emb  = self.switch_encoder(h.permute(0, 2, 1))
        context = self.full_context_nn(
            torch.cat([data['EnergySupp'], data['VoltageViolation'], data['EdgeFeat(Branchflow)']], dim=-1)
        )
        return self.final_encoder(torch.cat([sw_emb.mean(dim=1), context], dim=-1))


def get_extractor(gnn_type: str):
    return GCAPSExtractor
