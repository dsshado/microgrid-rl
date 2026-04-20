import torch
import torch.nn as nn


class GCAPSEncoder(nn.Module):
    """GCAPS encoder exposing per-node, graph-level, and context embeddings for MAPPO."""

    def __init__(self, features_dim=128, node_dim=3, n_dim=128, n_p=3, n_K=2, context_input_dim=140):
        super().__init__()
        self.n_dim        = n_dim
        self.n_p          = n_p
        self.n_K          = n_K
        self.features_dim = features_dim

        self.init_embed  = nn.Linear(node_dim, n_dim * n_p)
        self.W_L_1_G1    = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G2    = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G3    = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_F         = nn.Linear(n_dim * n_p, features_dim)

        self.context_nn = nn.Sequential(
            nn.Linear(context_input_dim, 2 * features_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * features_dim, features_dim)
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

        F0  = self.init_embed(X)                                             # (B, N, n_dim*n_p)
        inp = torch.cat([F0, torch.matmul(L, F0), torch.matmul(L2, F0)], dim=-1)

        g1 = self.W_L_1_G1(inp)
        g2 = self.W_L_1_G2(inp)
        g3 = self.W_L_1_G3(inp)

        F1      = torch.cat([g1, g2, g3], dim=-1)   # (B, N, n_dim*n_p)
        F_nodes = self.W_F(F1)                       # (B, N, features_dim)
        F_graph = F_nodes.mean(dim=1)                # (B, features_dim)

        F_context = self.context_nn(
            torch.cat([data['EnergySupp'], data['VoltageViolation'], data['EdgeFeat(Branchflow)']], dim=-1)
        )                                            # (B, features_dim)

        return F_nodes, F_graph, F_context
