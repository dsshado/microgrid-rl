"""Train PPO+GCAPS or MAPPO+GCAPS on the IEEE 123-bus network.

Usage:
    python train.py --algo MAPPO
    python train.py --algo PPO
"""

import sys
import os
import math
import torch
from typing import Callable

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))          # RL_Training/
_ROOT = os.path.dirname(_HERE)                               # MAPPO_Outage_Management/
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

# ── imports ───────────────────────────────────────────────────────────────────
from Environments.DSSdirect_123bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
from Environments.DSSdirect_123bus_loadandswitching.DSS_Initialize import (
    node_list, AllSwitches, dispatch_loads, Load_Buses, n_actions
)
from Configs.training_config import get_training_config


# ── learning rate schedule ────────────────────────────────────────────────────
def lr_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value * math.exp(-(1 - progress_remaining) ** 2 * 3)
    return func


# ── agent-to-bus mapping for MAPPO ───────────────────────────────────────────
def build_agent_bus_mapping():
    """Map each agent (switch/load) to its bus index in node_list."""
    mapping = []
    for sw in AllSwitches:
        bus = sw['from bus']
        mapping.append(node_list.index(bus) if bus in node_list else 0)
    for load_name in dispatch_loads:
        bus = Load_Buses.get(load_name, node_list[0])
        mapping.append(node_list.index(bus) if bus in node_list else 0)
    return mapping


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    cfg = get_training_config()
    if cfg.device:
        device_str = cfg.device
    elif torch.cuda.is_available() and not cfg.no_cuda:
        device_str = "cuda:0"
    else:
        device_str = "cpu"
    device = torch.device(device_str)

    print(f"Algorithm  : {cfg.algo}")
    print(f"Device     : {device}")
    print(f"Total steps: {cfg.total_steps}")

    save_path = cfg.model_save + f"{cfg.algo}_GCAPS_123bus"

    # ── create environment ────────────────────────────────────────────────────
    env = DSS_OutCtrl_Env()

    # ══════════════════════════════════════════════════════════════════════════
    if cfg.algo == 'PPO':
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from stable_baselines3.common.utils import set_random_seed
        from Policies.bus_123.Feature_Extractor import get_extractor
        from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy

        def make_env(rank, seed=0):
            def _init():
                e = DSS_OutCtrl_Env()
                e.reset(seed=seed + rank)
                return e
            set_random_seed(seed)
            return _init

        vec_env = SubprocVecEnv([make_env(i) for i in range(cfg.num_cpu)])

        net_arch = dict(
            pi=[cfg.features_dim, 2 * cfg.features_dim, 2 * cfg.features_dim, cfg.features_dim],
            vf=[cfg.features_dim, 2 * cfg.features_dim, 2 * cfg.features_dim, cfg.features_dim],
        )
        policy_kwargs = dict(
            features_extractor_class=get_extractor('GCAPS'),
            features_extractor_kwargs=dict(features_dim=cfg.features_dim, node_dim=3, gnn_type='GCAPS'),
            net_arch=net_arch,
        )
        checkpoint_cb = CheckpointCallback(
            save_freq=cfg.save_freq,
            save_path=cfg.model_save,
            name_prefix=f"PPO_GCAPS_123bus"
        )
        model = PPO(
            policy=ActorCriticGCAPSPolicy,
            env=vec_env,
            tensorboard_log=cfg.logger + "PPO_GCAPS_123bus",
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            learning_rate=lr_schedule(cfg.learning_rate),
            ent_coef=cfg.ent_coef,
            device=device_str,
        )
        model.learn(total_timesteps=cfg.total_steps, callback=checkpoint_cb)
        model.save(save_path + "_final")
        print(f"PPO model saved to {save_path}_final")

    # ══════════════════════════════════════════════════════════════════════════
    elif cfg.algo == 'MAPPO':
        from RL_Methods.MAPPO.policy import MAPPOPolicy
        from RL_Methods.MAPPO.mappo  import MAPPO

        agent_bus_indices = build_agent_bus_mapping()
        policy = MAPPOPolicy(
            n_agents=n_actions,
            features_dim=cfg.features_dim,
            node_dim=3,
            context_input_dim=140,
            agent_bus_indices=agent_bus_indices,
            device=device_str,
        )

        trainer = MAPPO(env=env, policy=policy, cfg=cfg)
        trainer.learn(
            total_timesteps=cfg.total_steps,
            save_path=save_path,
            save_freq=cfg.save_freq,
        )
