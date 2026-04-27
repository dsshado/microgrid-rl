import math
import random
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from random import sample, uniform
from math import ceil

from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import *
from Environments.DSSdirect_34bus_loadandswitching.state_action_reward import *

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.WARNING)


class DSS_OutCtrl_Env(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        print("Initializing 34-bus env")
        self.DSSCktObj, G_init_local, conv_flag = initialize()
        self.outedges = []
        self.G        = G_init_local.copy()

        self.action_space = spaces.MultiBinary(n_actions)
        self.observation_space = spaces.Dict({
            "EnergySupp":           spaces.Box(low=0,    high=2,    shape=(1,),                                          dtype=np.float32),
            "NodeFeat(BusVoltage)": spaces.Box(low=0,    high=2,    shape=(len(G_init_local.nodes()), 3),                dtype=np.float32),
            "EdgeFeat(Branchflow)": spaces.Box(low=0,    high=2,    shape=(len(G_init_local.edges()),),                  dtype=np.float32),
            "Adjacency":            spaces.Box(low=0,    high=1,    shape=(len(G_init_local.nodes()), len(G_init_local.nodes())), dtype=np.float32),
            "VoltageViolation":     spaces.Box(low=0,    high=1000, shape=(1,),                                          dtype=np.float32),
            "ConvergenceViolation": spaces.Box(low=0,    high=1,    shape=(1,),                                          dtype=np.float32),
            "ActionMasking":        spaces.Box(low=0,    high=1,    shape=(n_actions,),                                  dtype=np.float32),
        })
        print("Env initialized")

    def step(self, action):
        try:
            self.DSSCktObj, self.G = take_action(action, self.outedges)
            obs    = get_state(self.DSSCktObj, self.G, self.outedges)
            reward = get_reward(obs)
        except Exception:
            obs    = get_state(self.DSSCktObj, self.G, self.outedges)
            reward = np.array([0.0])

        terminated = True
        truncated  = False
        info       = {"is_success": terminated, "episode": {"r": reward, "l": 1}}
        return obs, float(reward[0]), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.DSSCktObj, G_init_local, conv_flag = initialize()
        self.G = G_init_local.copy()

        # Fixed critical outage scenario: pass options={'fixed_outages': [('832','858'), ...]}
        if options and 'fixed_outages' in options:
            out_edges = []
            for (u, v) in options['fixed_outages']:
                if G_init_local.has_edge(u, v):
                    out_edges.append((u, v))
                elif G_init_local.has_edge(v, u):
                    out_edges.append((v, u))
            if len(out_edges) == 0:
                return self.reset()
        else:
            max_rad      = nx.diameter(G_init_local)
            max_percfail = 0.5
            nd           = random.choice(list(G_init_local.nodes()))
            rad          = ceil(uniform(0, max_rad / 2))

            Gsub      = nx.ego_graph(G_base, nd, radius=rad, undirected=False)
            sub_edges = list(Gsub.edges())
            if len(sub_edges) == 0:
                return self.reset()
            out_perc  = uniform(0, max_percfail)
            N_out     = max(1, math.ceil(len(sub_edges) * out_perc))
            out_edges = sample(sub_edges, k=N_out)

        for o_e in out_edges:
            (u, v)      = o_e
            branch_name = G_init_local.edges[o_e]['label'][0]
            self.DSSCktObj.dss.Text.Command(f'Open {branch_name} term=1')
            try:
                self.DSSCktObj.dss.Solution.Solve()
            except Exception:
                return self.reset()

        self.G.remove_edges_from(out_edges)
        self.outedges = out_edges

        obs = get_state(self.DSSCktObj, self.G, self.outedges)
        if np.isnan(obs["EdgeFeat(Branchflow)"]).any() or obs["NodeFeat(BusVoltage)"].max() > 100000:
            return self.reset()
        return obs, {}

    def render(self):
        pass
