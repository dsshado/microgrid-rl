"""Evaluate and compare PPO+GCAPS vs MAPPO+GCAPS on IEEE test networks.

Usage:
    # Full evaluation + paper figures (paper figs ON by default, 500 eps, saves to Drive)
    python test.py --bus_size 34
                   --ppo_model /content/drive/MyDrive/microgrid_models/PPO_GCAPS_34bus_final
                   --mappo_model /content/drive/MyDrive/microgrid_models/MAPPO_GCAPS_34bus_final.pt

    # Re-plot from saved results (no re-evaluation)
    python test.py --bus_size 34 --load_results
                   --ppo_model <path>  --mappo_model <path>

    # Training convergence (needs log dirs)
    python test.py --bus_size 34
                   --ppo_log /content/drive/.../ppo_logs
                   --mappo_log /content/drive/.../MAPPO_GCAPS_34bus

    # N-fault scalability test (N=1,2,3,4 × 500 episodes)
    python test.py --bus_size 34 --n_fault_test

    # Re-plot N-fault from saved results
    python test.py --bus_size 34 --n_fault_test --load_results
"""

import sys
import os
import json
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman'],
    'mathtext.fontset':   'stix',
    'font.size':          15,
    'axes.labelsize':     15,
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'legend.fontsize':    15,
    'lines.linewidth':    1.5,
    'lines.markersize':   7,
    'errorbar.capsize':   4,
})

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

# ── environment detection ─────────────────────────────────────────────────────
_ON_COLAB = os.path.exists('/content')

if _ON_COLAB:
    _D_PLOT_DIR    = '/content/drive/MyDrive/microgrid_models/plots'
    _D_DATA_DIR    = '/content/drive/MyDrive/microgrid_models/plots/data'
    _D_PPO_MODEL   = '/content/drive/MyDrive/microgrid_models/PPO_GCAPS_34bus_final'
    _D_MAPPO_MODEL = '/content/drive/MyDrive/microgrid_models/MAPPO_GCAPS_34bus_final.pt'
    _D_PPO_LOG     = '/content/drive/MyDrive/microgrid_models/ppo_logs'
    _D_MAPPO_LOG   = '/content/drive/MyDrive/microgrid_models/MAPPO_GCAPS_34bus'
else:
    _D_PLOT_DIR    = os.path.join(_HERE, 'Results', 'plots')
    _D_DATA_DIR    = os.path.join(_HERE, 'TrainedModels')
    _D_PPO_MODEL   = os.path.join(_HERE, 'TrainedModels', 'PPO_GCAPS_34bus_final')
    _D_MAPPO_MODEL = os.path.join(_HERE, 'TrainedModels', 'MAPPO_GCAPS_34bus_final.pt')
    _D_PPO_LOG     = os.path.join(_HERE, 'TrainedModels', 'ppo_logs')
    _D_MAPPO_LOG   = os.path.join(_HERE, 'TrainedModels', 'MAPPO_logs')


# ── environment + network info loader ────────────────────────────────────────
def load_env_and_info(bus_size: int):
    if bus_size == 34:
        from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
        from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import (
            node_list, AllSwitches, dispatch_loads, Load_Buses, n_actions
        )
    else:
        from Environments.DSSdirect_123bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
        from Environments.DSSdirect_123bus_loadandswitching.DSS_Initialize import (
            node_list, AllSwitches, dispatch_loads, Load_Buses, n_actions
        )
    env = DSS_OutCtrl_Env()
    return env, node_list, AllSwitches, dispatch_loads, Load_Buses, n_actions


# ── agent-bus mapping ─────────────────────────────────────────────────────────
def build_agent_bus_mapping(node_list, AllSwitches, dispatch_loads, Load_Buses):
    mapping = []
    for sw in AllSwitches:
        bus = sw['from bus']
        mapping.append(node_list.index(bus) if bus in node_list else 0)
    for load_name in dispatch_loads:
        bus = Load_Buses.get(load_name, node_list[0])
        mapping.append(node_list.index(bus) if bus in node_list else 0)
    return mapping


# ── default initial actions ───────────────────────────────────────────────────
def _default_actions(bus_size: int):
    if bus_size == 34:
        return np.array([1]*5 + [0]*4 + [1]*10, dtype=np.float32)
    else:
        return np.array([1]*13 + [0]*9 + [1]*19, dtype=np.float32)


# ── metric extraction ────────────────────────────────────────────────────────
def _extract_metrics(obs):
    return float(obs['EnergySupp'].sum()), float(obs['VoltageViolation'].sum())


# ── save / load helpers ───────────────────────────────────────────────────────
def save_results(results, save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)
    for algo, data in results.items():
        path = os.path.join(save_dir, f"{tag}_{algo}.npz")
        np.savez(path, **data)
        print(f"  Saved {algo} results: {path}")


def load_results(save_dir, tag, algos=('PPO', 'MAPPO')):
    results = {}
    for algo in algos:
        path = os.path.join(save_dir, f"{tag}_{algo}.npz")
        data = np.load(path)
        results[algo] = {k: data[k] for k in data.files}
        print(f"  Loaded {algo} results: {path}")
    return results


def save_invalid_scenarios(scenarios, save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{tag}_invalid_scenarios.json")
    with open(path, 'w') as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Saved invalid scenarios: {path}")


def load_invalid_scenarios(save_dir, tag):
    path = os.path.join(save_dir, f"{tag}_invalid_scenarios.json")
    with open(path) as f:
        return json.load(f)


# ── PPO evaluation ────────────────────────────────────────────────────────────
def evaluate_ppo(model_path, n_episodes, bus_size):
    from stable_baselines3 import PPO
    from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy

    env, _, _, _, _, _ = load_env_and_info(bus_size)
    model = PPO.load(model_path, env=env)

    rewards, energy_supps, volt_viols = [], [], []
    invalid_count, invalid_scenarios = 0, []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, _ = env.step(action)
        en, vv = _extract_metrics(obs)
        if abs(en) > 10 or abs(vv) > 10:
            invalid_count += 1
            invalid_scenarios.append({
                'episode':  ep + 1,
                'outedges': [list(e) for e in env.outedges],
                'reward':   float(reward),
                'energy':   en,
                'volt_viol': vv,
            })
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
        if (ep + 1) % 10 == 0:
            print(f"  PPO  ep {ep+1:4d}/{n_episodes}  reward={reward:.3f}")

    print(f"  PPO  invalid episodes: {invalid_count}/{n_episodes}")
    return np.array(rewards), np.array(energy_supps), np.array(volt_viols), invalid_scenarios


# ── MAPPO evaluation ──────────────────────────────────────────────────────────
def evaluate_mappo(model_path, n_episodes, bus_size, device_str):
    from RL_Methods.MAPPO.policy import MAPPOPolicy

    env, node_list, AllSwitches, dispatch_loads, Load_Buses, n_actions = \
        load_env_and_info(bus_size)
    edge_dim          = env.observation_space['EdgeFeat(Branchflow)'].shape[0]
    context_input_dim = 1 + 1 + edge_dim
    agent_bus_indices = build_agent_bus_mapping(node_list, AllSwitches, dispatch_loads, Load_Buses)

    policy = MAPPOPolicy(
        n_agents=n_actions, features_dim=128, node_dim=3,
        context_input_dim=context_input_dim,
        agent_bus_indices=agent_bus_indices, device=device_str,
    )
    state_dict = torch.load(model_path, map_location=device_str)
    policy.load_state_dict(state_dict)
    policy.eval()

    rewards, energy_supps, volt_viols = [], [], []
    invalid_count = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        current_actions = torch.tensor(
            _default_actions(bus_size), dtype=torch.float32
        ).unsqueeze(0).to(device_str)
        actions, _, _, _, _ = policy.get_actions(obs, current_actions, deterministic=True)
        action_np = actions.squeeze(0).cpu().numpy().astype(int)
        obs, reward, _, _, _ = env.step(action_np)
        en, vv = _extract_metrics(obs)
        if abs(en) > 10 or abs(vv) > 10:
            invalid_count += 1
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
        if (ep + 1) % 10 == 0:
            print(f"  MAPPO ep {ep+1:4d}/{n_episodes}  reward={reward:.3f}")

    print(f"  MAPPO invalid episodes: {invalid_count}/{n_episodes}")
    return np.array(rewards), np.array(energy_supps), np.array(volt_viols)


# ── invalid scenario replay ───────────────────────────────────────────────────
def evaluate_ppo_on_scenarios(model_path, scenarios, bus_size):
    from stable_baselines3 import PPO
    from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env

    env   = DSS_OutCtrl_Env()
    model = PPO.load(model_path, env=env)
    rewards, energy_supps, volt_viols = [], [], []

    for i, scenario in enumerate(scenarios):
        outedges = scenario['outedges'] if isinstance(scenario, dict) else scenario
        obs, _ = env.reset(options={'fixed_outages': [tuple(e) for e in outedges]})
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, _ = env.step(action)
        en, vv = _extract_metrics(obs)
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
        print(f"  PPO  invalid-replay {i+1}/{len(scenarios)}  reward={reward:.3f}  energy={en:.3f}  volt_viol={vv:.3f}")
    return np.array(rewards), np.array(energy_supps), np.array(volt_viols)


def evaluate_mappo_on_scenarios(model_path, scenarios, bus_size, device_str):
    from RL_Methods.MAPPO.policy import MAPPOPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
    from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import (
        node_list, AllSwitches, dispatch_loads, Load_Buses, n_actions
    )

    env               = DSS_OutCtrl_Env()
    edge_dim          = env.observation_space['EdgeFeat(Branchflow)'].shape[0]
    context_input_dim = 1 + 1 + edge_dim
    agent_bus_indices = build_agent_bus_mapping(node_list, AllSwitches, dispatch_loads, Load_Buses)

    policy = MAPPOPolicy(
        n_agents=n_actions, features_dim=128, node_dim=3,
        context_input_dim=context_input_dim,
        agent_bus_indices=agent_bus_indices, device=device_str,
    )
    state_dict = torch.load(model_path, map_location=device_str)
    policy.load_state_dict(state_dict)
    policy.eval()

    rewards, energy_supps, volt_viols = [], [], []
    for i, scenario in enumerate(scenarios):
        outedges = scenario['outedges'] if isinstance(scenario, dict) else scenario
        obs, _ = env.reset(options={'fixed_outages': [tuple(e) for e in outedges]})
        current_actions = torch.tensor(
            _default_actions(bus_size), dtype=torch.float32
        ).unsqueeze(0).to(device_str)
        actions, _, _, _, _ = policy.get_actions(obs, current_actions, deterministic=True)
        action_np = actions.squeeze(0).cpu().numpy().astype(int)
        obs, reward, _, _, _ = env.step(action_np)
        en, vv = _extract_metrics(obs)
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
        print(f"  MAPPO invalid-replay {i+1}/{len(scenarios)}  reward={reward:.3f}  energy={en:.3f}  volt_viol={vv:.3f}")
    return np.array(rewards), np.array(energy_supps), np.array(volt_viols)


# ── critical case evaluation ──────────────────────────────────────────────────
CRITICAL_OUTAGES_34 = [('832', '858'), ('852', '854'), ('834', '860')]


def evaluate_ppo_critical(model_path, n_episodes, bus_size):
    from stable_baselines3 import PPO
    from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env

    env   = DSS_OutCtrl_Env()
    model = PPO.load(model_path, env=env)
    rewards, energy_supps, volt_viols = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(options={'fixed_outages': CRITICAL_OUTAGES_34})
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, _ = env.step(action)
        en, vv = _extract_metrics(obs)
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
    print(f"  PPO  critical mean reward: {np.mean(rewards):.4f}")
    return np.array(rewards), np.array(energy_supps), np.array(volt_viols)


def evaluate_mappo_critical(model_path, n_episodes, bus_size, device_str):
    from RL_Methods.MAPPO.policy import MAPPOPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
    from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import (
        node_list, AllSwitches, dispatch_loads, Load_Buses, n_actions
    )

    env               = DSS_OutCtrl_Env()
    edge_dim          = env.observation_space['EdgeFeat(Branchflow)'].shape[0]
    context_input_dim = 1 + 1 + edge_dim
    agent_bus_indices = build_agent_bus_mapping(node_list, AllSwitches, dispatch_loads, Load_Buses)

    policy = MAPPOPolicy(
        n_agents=n_actions, features_dim=128, node_dim=3,
        context_input_dim=context_input_dim,
        agent_bus_indices=agent_bus_indices, device=device_str,
    )
    state_dict = torch.load(model_path, map_location=device_str)
    policy.load_state_dict(state_dict)
    policy.eval()

    rewards, energy_supps, volt_viols = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(options={'fixed_outages': CRITICAL_OUTAGES_34})
        current_actions = torch.tensor(
            _default_actions(bus_size), dtype=torch.float32
        ).unsqueeze(0).to(device_str)
        actions, _, _, _, _ = policy.get_actions(obs, current_actions, deterministic=True)
        action_np = actions.squeeze(0).cpu().numpy().astype(int)
        obs, reward, _, _, _ = env.step(action_np)
        en, vv = _extract_metrics(obs)
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
    print(f"  MAPPO critical mean reward: {np.mean(rewards):.4f}")
    return np.array(rewards), np.array(energy_supps), np.array(volt_viols)


# ── N-fault scalability test ─────────────────────────────────────────────────
def _fault_candidates(bus_size):
    """Non-switch edges that can be faulted."""
    if bus_size == 34:
        from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import (
            AllSwitches, G_init
        )
    else:
        from Environments.DSSdirect_123bus_loadandswitching.DSS_Initialize import (
            AllSwitches, G_init
        )
    sw_set = set()
    for sw in AllSwitches:
        sw_set.add((sw['from bus'], sw['to bus']))
        sw_set.add((sw['to bus'], sw['from bus']))
    return [(u, v) for u, v in G_init.edges()
            if (u, v) not in sw_set and (v, u) not in sw_set]


def evaluate_ppo_n_faults(model_path, n_faults, n_episodes, bus_size, fixed_scenarios=None):
    from stable_baselines3 import PPO
    from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env

    env   = DSS_OutCtrl_Env()
    model = PPO.load(model_path, env=env)

    rewards, energy_supps, volt_viols, invalid_mask = [], [], [], []
    invalid_count = 0

    for ep in range(n_episodes):
        outages = fixed_scenarios[ep] if fixed_scenarios is not None else \
                  __import__('random').sample(_fault_candidates(bus_size), min(n_faults, len(_fault_candidates(bus_size))))
        obs, _    = env.reset(options={'fixed_outages': outages})
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, _ = env.step(action)
        en, vv = _extract_metrics(obs)
        is_inv = abs(en) > 10 or abs(vv) > 10
        if is_inv:
            invalid_count += 1
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
        invalid_mask.append(is_inv)

    print(f"  PPO  N={n_faults}: mean_reward={np.mean(rewards):.3f}  invalid={invalid_count}/{n_episodes}")
    return (np.array(rewards), np.array(energy_supps), np.array(volt_viols),
            invalid_count, np.array(invalid_mask, dtype=bool))


def evaluate_mappo_n_faults(model_path, n_faults, n_episodes, bus_size, device_str,
                             fixed_scenarios=None):
    from RL_Methods.MAPPO.policy import MAPPOPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
    from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import (
        node_list, AllSwitches, dispatch_loads, Load_Buses, n_actions
    )

    env               = DSS_OutCtrl_Env()
    edge_dim          = env.observation_space['EdgeFeat(Branchflow)'].shape[0]
    context_input_dim = 1 + 1 + edge_dim
    agent_bus_indices = build_agent_bus_mapping(node_list, AllSwitches, dispatch_loads, Load_Buses)

    policy = MAPPOPolicy(
        n_agents=n_actions, features_dim=128, node_dim=3,
        context_input_dim=context_input_dim,
        agent_bus_indices=agent_bus_indices, device=device_str,
    )
    state_dict = torch.load(model_path, map_location=device_str)
    policy.load_state_dict(state_dict)
    policy.eval()

    rewards, energy_supps, volt_viols, invalid_mask = [], [], [], []
    invalid_count = 0

    for ep in range(n_episodes):
        outages = fixed_scenarios[ep] if fixed_scenarios is not None else \
                  __import__('random').sample(_fault_candidates(bus_size), min(n_faults, len(_fault_candidates(bus_size))))
        obs, _ = env.reset(options={'fixed_outages': outages})
        current_actions = torch.tensor(
            _default_actions(bus_size), dtype=torch.float32
        ).unsqueeze(0).to(device_str)
        actions, _, _, _, _ = policy.get_actions(obs, current_actions, deterministic=True)
        action_np = actions.squeeze(0).cpu().numpy().astype(int)
        obs, reward, _, _, _ = env.step(action_np)
        en, vv = _extract_metrics(obs)
        is_inv = abs(en) > 10 or abs(vv) > 10
        if is_inv:
            invalid_count += 1
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
        invalid_mask.append(is_inv)

    print(f"  MAPPO N={n_faults}: mean_reward={np.mean(rewards):.3f}  invalid={invalid_count}/{n_episodes}")
    return (np.array(rewards), np.array(energy_supps), np.array(volt_viols),
            invalid_count, np.array(invalid_mask, dtype=bool))


def plot_n_fault_comparison(n_fault_results, save_dir, fmt):
    ns = sorted(n_fault_results.keys())
    os.makedirs(save_dir, exist_ok=True)

    # PPO's invalid mask — used for both algos so means are on the same episode set
    ppo_masks = [~n_fault_results[n]['PPO'].get('invalid_mask',
                  np.zeros(len(n_fault_results[n]['PPO']['reward']), dtype=bool))
                 for n in ns]

    # ── Figure A: mean energy, mean reward, mean voltage violation ────────────
    perf_configs = [
        ('energy_supp', 'Mean Energy Supplied (p.u.)'),
        ('reward',      'Mean Reward'),
        ('volt_viol',   'Mean Voltage Violation'),
    ]
    fig_a, axes_a = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, ylabel) in zip(axes_a, perf_configs):
        for algo, color, marker in [('PPO', 'steelblue', 'o'), ('MAPPO', 'tomato', 's')]:
            data  = [n_fault_results[n][algo][metric] for n in ns]
            means = [np.mean(d[m]) if m.any() else 0 for d, m in zip(data, ppo_masks)]
            stds  = [np.std(d[m])  if m.any() else 0 for d, m in zip(data, ppo_masks)]
            ax.errorbar(ns, means, yerr=stds, marker=marker, color=color,
                        linewidth=2, markersize=8, capsize=5, label=f'{algo}+GCAPS')
        ax.set_xlabel('Number of Simultaneous Faults (N)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(ns)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig_a.tight_layout()
    fname_a = os.path.join(save_dir, f'fig_n_fault_performance.{fmt}')
    fig_a.savefig(fname_a, dpi=300, bbox_inches='tight')
    plt.close(fig_a)
    print(f"  Saved: {fname_a}")

    # ── Figure B: failure rate ────────────────────────────────────────────────
    fig_b, ax_b = plt.subplots(figsize=(6, 5))
    for algo, color, marker in [('PPO', 'steelblue', 'o'), ('MAPPO', 'tomato', 's')]:
        vals = [float(n_fault_results[n][algo]['fail_rate']) * 100 for n in ns]
        ax_b.plot(ns, vals, marker=marker, color=color, linewidth=2,
                  markersize=8, label=f'{algo}+GCAPS')
        for x, y in zip(ns, vals):
            ax_b.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                          xytext=(0, 8), ha='center', fontsize=9, color=color)
    ax_b.set_xlabel('Number of Simultaneous Faults (N)')
    ax_b.set_ylabel('Failure Rate (%)')
    ax_b.set_xticks(ns)
    ax_b.set_ylim(bottom=0)
    ax_b.legend()
    ax_b.grid(True, alpha=0.3)
    fig_b.tight_layout()
    fname_b = os.path.join(save_dir, f'fig_n_fault_failure_rate.{fmt}')
    fig_b.savefig(fname_b, dpi=300, bbox_inches='tight')
    plt.close(fig_b)
    print(f"  Saved: {fname_b}")


# ── single-episode inspection (for paper figures) ─────────────────────────────
def inspect_ppo_scenario(model_path, outages, bus_size):
    from stable_baselines3 import PPO
    from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
    from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import (
        node_list, AllSwitches, dispatch_loads, n_actions, G_init, generators
    )

    env   = DSS_OutCtrl_Env()
    model = PPO.load(model_path, env=env)
    obs, _ = env.reset(options={'fixed_outages': outages})
    mask = obs['ActionMasking'].copy()
    action, _ = model.predict(obs, deterministic=True)
    post_obs, _, _, _, _ = env.step(action)

    defaults = _default_actions(bus_size)
    n_sw   = len(AllSwitches)
    n_sect = int(defaults[:n_sw].sum())
    n_tie  = n_sw - n_sect

    meta = {
        'node_list': node_list, 'AllSwitches': AllSwitches,
        'n_sect': n_sect, 'n_tie': n_tie,
        'dispatch_loads': dispatch_loads, 'mask': mask,
        'G': G_init, 'generators': generators,
    }
    return action, post_obs, meta, list(env.outedges)


def inspect_mappo_scenario(model_path, outages, bus_size, device_str):
    from RL_Methods.MAPPO.policy import MAPPOPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
    from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import (
        node_list, AllSwitches, dispatch_loads, Load_Buses, n_actions, G_init, generators
    )

    env               = DSS_OutCtrl_Env()
    edge_dim          = env.observation_space['EdgeFeat(Branchflow)'].shape[0]
    context_input_dim = 1 + 1 + edge_dim
    agent_bus_indices = build_agent_bus_mapping(node_list, AllSwitches, dispatch_loads, Load_Buses)

    policy = MAPPOPolicy(
        n_agents=n_actions, features_dim=128, node_dim=3,
        context_input_dim=context_input_dim,
        agent_bus_indices=agent_bus_indices, device=device_str,
    )
    state_dict = torch.load(model_path, map_location=device_str)
    policy.load_state_dict(state_dict)
    policy.eval()

    obs, _ = env.reset(options={'fixed_outages': outages})
    mask = obs['ActionMasking'].copy()
    current_actions = torch.tensor(
        _default_actions(bus_size), dtype=torch.float32
    ).unsqueeze(0).to(device_str)
    actions, _, _, _, _ = policy.get_actions(obs, current_actions, deterministic=True)
    action_np = actions.squeeze(0).cpu().numpy().astype(int)
    post_obs, _, _, _, _ = env.step(action_np)

    defaults = _default_actions(bus_size)
    n_sw   = len(AllSwitches)
    n_sect = int(defaults[:n_sw].sum())
    n_tie  = n_sw - n_sect

    meta = {
        'node_list': node_list, 'AllSwitches': AllSwitches,
        'n_sect': n_sect, 'n_tie': n_tie,
        'dispatch_loads': dispatch_loads, 'mask': mask,
        'G': G_init, 'generators': generators,
    }
    return action_np, post_obs, meta, list(env.outedges)


# ── paper-style: network topology graph ──────────────────────────────────────
def plot_network_topology(actions_dict, post_obs_dict, outedges, meta,
                          save_dir, fmt, suffix='', title=''):
    import networkx as nx

    G          = meta['G']
    node_list  = meta['node_list']
    AllSwitches = meta['AllSwitches']
    n_sw       = meta['n_sect'] + meta['n_tie']

    # Build lookup: (from_bus, to_bus) -> switch index  (both directions)
    sw_lookup = {}
    for i, sw in enumerate(AllSwitches):
        fb, tb = sw['from bus'], sw['to bus']
        sw_lookup[(fb, tb)] = i
        sw_lookup[(tb, fb)] = i

    # Faulted edge set (both directions)
    fault_set = set()
    for e in outedges:
        fault_set.add((str(e[0]), str(e[1])))
        fault_set.add((str(e[1]), str(e[0])))

    # Layout — kamada-kawai gives clean planar layout for radial networks
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    n_algos = len(actions_dict)
    fig, axes = plt.subplots(1, n_algos,
                             figsize=(11 * n_algos, 9), squeeze=False)

    for ax_idx, (ax, (algo, action)) in enumerate(zip(axes[0], actions_dict.items())):
        voltages = post_obs_dict[algo]['NodeFeat(BusVoltage)']
        v_mean   = voltages.mean(axis=1)  # average across phases per bus

        # Node colours
        node_colors = []
        for i, bus in enumerate(G.nodes()):
            idx = node_list.index(bus) if bus in node_list else -1
            v   = v_mean[idx] if idx >= 0 else 0.0
            if v < 0.01:
                node_colors.append('#b0b0b0')   # grey  — de-energized
            elif v < 0.95 or v > 1.05:
                node_colors.append('#e05c5c')   # red   — voltage violation
            else:
                node_colors.append('#74c476')   # green — normal

        # Classify edges
        normal_edges     = []
        closed_sw_edges  = []
        open_sw_edges    = []
        fault_edges      = []

        for u, v in G.edges():
            key = (str(u), str(v))
            if key in fault_set:
                fault_edges.append((u, v))
            elif key in sw_lookup or (str(v), str(u)) in sw_lookup:
                idx = sw_lookup.get(key) or sw_lookup.get((str(v), str(u)))
                if idx < len(action) and action[idx] == 1:
                    closed_sw_edges.append((u, v))
                else:
                    open_sw_edges.append((u, v))
            else:
                normal_edges.append((u, v))

        nx.draw_networkx_edges(G, pos, edgelist=normal_edges,
                               ax=ax, edge_color='#333333', width=1.5)
        nx.draw_networkx_edges(G, pos, edgelist=closed_sw_edges,
                               ax=ax, edge_color='#2ca02c', width=3)
        nx.draw_networkx_edges(G, pos, edgelist=open_sw_edges,
                               ax=ax, edge_color='#ff7f0e', width=2.5,
                               style='dashed')
        nx.draw_networkx_edges(G, pos, edgelist=fault_edges,
                               ax=ax, edge_color='#d62728', width=3,
                               style='dashed')

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=350, edgecolors='black', linewidths=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')

        # Overlay DER symbols
        gf_forming_buses = [g['bus'] for g in meta['generators'] if g['Gridforming'] == 'Yes']
        gf_feeding_buses = [g['bus'] for g in meta['generators'] if g['Gridforming'] == 'No']

        def _der_xy(buses):
            xs = [pos[b][0] for b in buses if b in pos]
            ys = [pos[b][1] for b in buses if b in pos]
            return xs, ys

        xf, yf = _der_xy(gf_forming_buses)
        xp, yp = _der_xy(gf_feeding_buses)
        if xf:
            ax.scatter(xf, yf, marker='*', s=400, color='gold',
                       edgecolors='black', linewidths=0.8, zorder=6)
        if xp:
            ax.scatter(xp, yp, marker='D', s=160, color='dodgerblue',
                       edgecolors='black', linewidths=0.8, zorder=6)

        ax.set_title(algo, fontsize=15, fontweight='bold', pad=8)
        ax.axis('off')

        # Only draw legend on first subplot
        if ax_idx == 0:
            legend = [
                Line2D([0], [0], color='#333333', lw=2,   label='Line (closed)'),
                Line2D([0], [0], color='#2ca02c', lw=3,   label='Switch (closed)'),
                Line2D([0], [0], color='#ff7f0e', lw=2.5, linestyle='--', label='Switch (open)'),
                Line2D([0], [0], color='#d62728', lw=3,   linestyle='--', label='Faulted line'),
                Patch(facecolor='#74c476', edgecolor='black', label='Normal voltage'),
                Patch(facecolor='#e05c5c', edgecolor='black', label='Voltage violation'),
                Patch(facecolor='#b0b0b0', edgecolor='black', label='De-energized'),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                       markeredgecolor='black', markersize=14, label='Grid-forming DER'),
                Line2D([0], [0], marker='D', color='w', markerfacecolor='dodgerblue',
                       markeredgecolor='black', markersize=9,  label='Grid-feeding DER'),
            ]
            ax.legend(handles=legend, loc='lower left', fontsize=12, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    tag   = f'_{suffix}' if suffix else ''
    fname = os.path.join(save_dir, f'fig_network_topology{tag}.{fmt}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── print invalid scenario details ───────────────────────────────────────────
def print_invalid_scenarios(invalid_scenarios):
    if not invalid_scenarios:
        return
    print(f"\n{'='*55}")
    print(f"  PPO Failed Scenarios — {len(invalid_scenarios)} episode(s)")
    print(f"{'='*55}")
    for i, sc in enumerate(invalid_scenarios):
        ep       = sc['episode']
        outedges = sc['outedges']
        reward   = sc['reward']
        energy   = sc['energy']
        vv       = sc['volt_viol']
        lines    = ', '.join(f"{u}-{v}" for u, v in outedges)
        print(f"  Episode {ep:4d}  |  Faulted lines: {lines}")
        print(f"             |  reward={reward:.3f}  energy={energy:.1f}  volt_viol={vv:.1f}")
    print(f"{'='*55}\n")


# ── standard comparison plots ─────────────────────────────────────────────────
def print_comparison(results, bus_size, label=''):
    header = f"{'Metric':<25} {'PPO+GCAPS':>15} {'MAPPO+GCAPS':>15}"
    title  = f"IEEE {bus_size}-bus results" + (f" — {label}" if label else "")
    print(f"\n{title}")
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for label_m, key, stat in [
        ("Mean Reward",      "reward",      "mean"),
        ("Std  Reward",      "reward",      "std"),
        ("Mean Energy Supp", "energy_supp", "mean"),
        ("Std  Energy Supp", "energy_supp", "std"),
        ("Mean Volt Viol",   "volt_viol",   "mean"),
        ("Std  Volt Viol",   "volt_viol",   "std"),
    ]:
        ppo_val   = getattr(np, stat)(results['PPO'][key])
        mappo_val = getattr(np, stat)(results['MAPPO'][key])
        print(f"  {label_m:<23} {ppo_val:>15.4f} {mappo_val:>15.4f}")
    print("=" * len(header) + "\n")


def plot_comparison(results, bus_size, save_dir, fmt='pdf'):
    os.makedirs(save_dir, exist_ok=True)
    metrics = {
        'Reward':             ('reward',      'Episode Reward'),
        'Energy Supplied':    ('energy_supp', 'Energy Supplied (p.u.)'),
        'Voltage Violations': ('volt_viol',   'Voltage Violations'),
    }
    for title, (key, ylabel) in metrics.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        for algo, color in [('PPO', 'steelblue'), ('MAPPO', 'tomato')]:
            vals = results[algo][key]
            ax.plot(vals, alpha=0.4, color=color, linewidth=0.8)
            ax.axhline(vals.mean(), linestyle='--', color=color,
                       label=f"{algo}+GCAPS (mean={vals.mean():.3f})")
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fname = os.path.join(save_dir, f"{key}_comparison.{fmt}")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved plot: {fname}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (title, (key, ylabel)) in zip(axes, metrics.items()):
        data = [results['PPO'][key], results['MAPPO'][key]]
        bp   = ax.boxplot(data, labels=['PPO+GCAPS', 'MAPPO+GCAPS'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['steelblue', 'tomato']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fname = os.path.join(save_dir, f'boxplot_summary.{fmt}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {fname}")


# ── paper-style Figure 1: decision variables heatmap ─────────────────────────
def plot_decision_heatmap(actions_dict, meta, save_dir, fmt, suffix=''):
    n_sect = meta['n_sect']
    n_tie  = meta['n_tie']
    n_sw   = n_sect + n_tie
    mask   = meta['mask']

    sw_labels   = [f'sw{i+1}' for i in range(n_sw)]
    load_labels = list(meta['dispatch_loads'])
    algos       = list(actions_dict.keys())
    n_algos     = len(algos)

    sw_data   = np.array([actions_dict[a][:n_sw] for a in algos], dtype=float)
    load_data = np.array([actions_dict[a][n_sw:] for a in algos], dtype=float)

    fig, axes = plt.subplots(
        1, 2, figsize=(max(10, n_sw * 1.1 + len(load_labels) * 0.7), max(2, n_algos + 1.5)),
        gridspec_kw={'width_ratios': [n_sw, len(load_labels)]}
    )

    for ax, data, labels, section_title in [
        (axes[0], sw_data,   sw_labels,   'switch status'),
        (axes[1], load_data, load_labels, 'load status'),
    ]:
        ax.imshow(data, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(n_algos))
        ax.set_yticklabels(algos, fontsize=10)
        for i in range(n_algos + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1.5)
        for j in range(len(labels) + 1):
            ax.axvline(j - 0.5, color='white', linewidth=1.5)

    # Red border on outage (masked) switches
    for j in range(n_sw):
        if mask[j] < 0.5:
            axes[0].add_patch(plt.Rectangle(
                (j - 0.5, -0.5), 1, n_algos,
                fill=False, edgecolor='red', linewidth=2.5, zorder=5
            ))

    # Legend: closed/open/outage
    legend_patches = [
        Patch(facecolor='#08306b', edgecolor='white', label='Closed (1)'),
        Patch(facecolor='#f0f0f0', edgecolor='gray',  label='Open (0)'),
        Patch(facecolor='white',   edgecolor='red', linewidth=2, label='Outage switch'),
    ]
    axes[1].legend(handles=legend_patches, loc='upper right', fontsize=11,
                   framealpha=0.9, bbox_to_anchor=(1, 1))

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    tag   = f'_{suffix}' if suffix else ''
    fname = os.path.join(save_dir, f'fig1_decision_heatmap{tag}.{fmt}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── paper-style Figure 2: voltage profile ────────────────────────────────────
def plot_voltage_profile(voltages_dict, node_list, save_dir, fmt, title='', suffix=''):
    n_algos = len(voltages_dict)
    fig, axes = plt.subplots(n_algos, 1, figsize=(14, 4 * n_algos), squeeze=False)

    phase_styles = [
        ('Phase a', 's', 'red'),
        ('Phase b', '^', 'blue'),
        ('Phase c', 'o', 'green'),
    ]
    x    = np.arange(len(node_list))
    step = max(1, len(node_list) // 20)

    for ax_idx, (ax, (algo, voltages)) in enumerate(zip(axes.flatten(), voltages_dict.items())):
        for ph_idx, (ph_label, marker, color) in enumerate(phase_styles):
            v      = voltages[:, ph_idx]
            active = v > 0.01

            if active.any():
                ax.scatter(x[active], v[active], marker=marker, color=color,
                           label=ph_label, s=25, zorder=3)

        ax.axhline(1.05, linestyle='--', color='gray', linewidth=0.8, alpha=0.7)
        ax.axhline(0.95, linestyle='--', color='gray', linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Buses')
        ax.set_ylabel('Voltage (in per unit)')
        ax.set_xticks(x[::step])
        ax.set_xticklabels(
            [node_list[i] for i in x[::step]], rotation=45, ha='right'
        )
        ax.set_title(algo, fontsize=13, fontweight='bold')
        if ax_idx == 0:
            ax.legend(loc='upper left', ncol=3)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    tag   = f'_{suffix}' if suffix else ''
    fname = os.path.join(save_dir, f'fig2_voltage_profile{tag}.{fmt}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── valid-episode mean bar chart ─────────────────────────────────────────────
def plot_valid_mean_bar(results, invalid_scenarios, save_dir, fmt):
    """Bar chart of mean metrics excluding PPO-crashed episodes."""
    # Build set of invalid episode indices (0-based)
    invalid_idx = set()
    for sc in invalid_scenarios:
        invalid_idx.add(sc['episode'] - 1)   # episode is 1-based

    n = len(results['PPO']['reward'])
    valid_mask = np.array([i not in invalid_idx for i in range(n)])
    n_valid    = valid_mask.sum()
    n_invalid  = n - n_valid

    metrics = [
        ('Mean Reward',        'reward',      'Reward'),
        ('Mean Energy Supp',   'energy_supp', 'Energy Supplied (p.u.)'),
        ('Mean Volt Violation', 'volt_viol',  'Voltage Violation'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, (title, key, ylabel) in zip(axes, metrics):
        ppo_vals   = results['PPO'][key][valid_mask]
        mappo_vals = results['MAPPO'][key][valid_mask]

        ppo_mean,   ppo_std   = ppo_vals.mean(),   ppo_vals.std()
        mappo_mean, mappo_std = mappo_vals.mean(), mappo_vals.std()

        bars = ax.bar(['PPO+GCAPS', 'MAPPO+GCAPS'],
                      [ppo_mean, mappo_mean],
                      yerr=[ppo_std, mappo_std],
                      color=['steelblue', 'tomato'], alpha=0.85,
                      capsize=6, error_kw={'linewidth': 1.5})

        for bar, mean in zip(bars, [ppo_mean, mappo_mean]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ppo_std * 0.05,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f'fig_valid_mean_bar.{fmt}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── paper-style Figure 3: energy bar chart ───────────────────────────────────
def plot_energy_bar(results_by_scenario, save_dir, fmt):
    scenarios = list(results_by_scenario.keys())
    x         = np.arange(len(scenarios))
    width     = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 2.2), 5))

    ppo_vals, mappo_vals = [], []
    for sc in scenarios:
        r = results_by_scenario[sc]
        ppo_vals.append(float(np.median(r['PPO']['energy_supp'])))
        mappo_vals.append(float(np.median(r['MAPPO']['energy_supp'])))

    bars_p = ax.bar(x - width/2, ppo_vals,   width, label='PPO+GCAPS',
                    color='steelblue', alpha=0.85)
    bars_m = ax.bar(x + width/2, mappo_vals, width, label='MAPPO+GCAPS',
                    color='tomato',    alpha=0.85)

    for bar in list(bars_p) + list(bars_m):
        h = bar.get_height()
        if abs(h) < 5:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Median Energy Supplied (p.u.)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f'fig3_energy_bar.{fmt}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── paper-style Figure 4: training convergence ───────────────────────────────
def plot_training_convergence(ppo_log, mappo_log, save_dir, fmt):
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False

    # PPO: SB3 monitor CSV
    if ppo_log:
        csvs = glob.glob(os.path.join(ppo_log, '**', 'monitor.csv'), recursive=True)
        if csvs:
            import pandas as pd
            df = pd.read_csv(csvs[0], skiprows=1)
            df['cumsteps'] = df['l'].cumsum()
            window = max(1, len(df) // 50)
            smooth = df['r'].rolling(window, min_periods=1).mean()
            ax.plot(df['cumsteps'] / 1e6, smooth,
                    color='steelblue', label='PPO+GCAPS', linewidth=1.5)
            plotted = True
        else:
            print(f"  No monitor.csv found under {ppo_log}")

    # MAPPO: TensorBoard events
    if mappo_log:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(mappo_log)
            ea.Reload()
            tag = 'rollout/ep_rew_mean'
            if tag in ea.Tags().get('scalars', []):
                events = ea.Scalars(tag)
                steps  = [e.step / 1e6 for e in events]
                vals   = [e.value for e in events]
                ax.plot(steps, vals, color='tomato', label='MAPPO+GCAPS', linewidth=1.5)
                plotted = True
        except Exception as e:
            print(f"  Could not load MAPPO TensorBoard log: {e}")

    if not plotted:
        print("  No training logs found — skipping convergence plot.")
        plt.close(fig)
        return

    ax.set_xlabel('Steps (× 10⁶)')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f'fig4_training_convergence.{fmt}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate PPO+GCAPS vs MAPPO+GCAPS")
    parser.add_argument('--bus_size',     type=int,  default=34, choices=[34, 123])
    parser.add_argument('--ppo_model',    type=str,  default=_D_PPO_MODEL)
    parser.add_argument('--mappo_model',  type=str,  default=_D_MAPPO_MODEL)
    parser.add_argument('--n_episodes',   type=int,  default=500)
    parser.add_argument('--no_cuda',      action='store_true')
    parser.add_argument('--plot_dir',     type=str,  default=_D_PLOT_DIR)
    parser.add_argument('--fig_format',   type=str,  default='pdf',
                        choices=['pdf', 'eps', 'png', 'svg'])
    parser.add_argument('--data_dir',     type=str,  default=_D_DATA_DIR,
                        help='Directory for saved .npz / .json result files')
    parser.add_argument('--load_results', action='store_true',
                        help='Skip evaluation and re-plot from saved .npz files')
    parser.add_argument('--no_paper_figs', action='store_true',
                        help='Disable paper-style figures (paper figs are on by default)')
    parser.add_argument('--ppo_log',      type=str,  default=_D_PPO_LOG,
                        help='PPO training log dir (for convergence plot)')
    parser.add_argument('--mappo_log',    type=str,  default=_D_MAPPO_LOG,
                        help='MAPPO TensorBoard log dir (for convergence plot)')
    parser.add_argument('--n_fault_test', action='store_true',
                        help='Run N-fault scalability test (N=1,2,3,4, 500 episodes each)')
    parser.add_argument('--n_fault_only', action='store_true',
                        help='Skip all other evaluations and run only the N-fault test')
    args = parser.parse_args()

    device_str = 'cuda:0' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    data_dir   = args.data_dir

    print(f"Bus size   : {args.bus_size}-bus")
    print(f"Device     : {device_str}")
    print(f"Episodes   : {args.n_episodes}")
    print(f"Fig format : {args.fig_format}")

    # ── random / invalid / critical / paper figures (skipped by --n_fault_only) ─
    invalid_scenarios = []
    results           = {}
    if not args.n_fault_only:
        # ── random episode evaluation ─────────────────────────────────────────
        _random_loaded = False
        if args.load_results:
            try:
                print("\n[Load] Loading saved results ...")
                results           = load_results(data_dir, tag='random')
                invalid_scenarios = load_invalid_scenarios(data_dir, tag='random')
                _random_loaded    = True
                print(f"  Loaded {len(invalid_scenarios)} invalid scenario(s)")
            except FileNotFoundError as _e:
                print(f"  [Load] Not found ({_e.filename}) — running evaluation instead.")

        if not _random_loaded:
            print(f"PPO  model : {args.ppo_model}")
            print(f"MAPPO model: {args.mappo_model}")

            print("\n[1/2] Evaluating PPO+GCAPS ...")
            r, en, vv, invalid_scenarios = evaluate_ppo(
                args.ppo_model, args.n_episodes, args.bus_size)
            results['PPO'] = {'reward': r, 'energy_supp': en, 'volt_viol': vv}

            print("\n[2/2] Evaluating MAPPO+GCAPS ...")
            r, en, vv = evaluate_mappo(
                args.mappo_model, args.n_episodes, args.bus_size, device_str)
            results['MAPPO'] = {'reward': r, 'energy_supp': en, 'volt_viol': vv}

            save_results(results, data_dir, tag='random')
            save_invalid_scenarios(invalid_scenarios, data_dir, tag='random')

        print_comparison(results, args.bus_size, label='Random Episodes')
        if args.no_paper_figs:
            plot_comparison(results, args.bus_size, args.plot_dir, fmt=args.fig_format)
        print_invalid_scenarios(invalid_scenarios)

        # ── invalid scenario replay ───────────────────────────────────────────
        inv_results = None
        if invalid_scenarios:
            _inv_loaded = False
            if args.load_results:
                try:
                    inv_results = load_results(data_dir, tag='invalid_replay')
                    _inv_loaded = True
                    print("\n[Load] Loaded saved invalid replay results.")
                except Exception:
                    print("\n[Load] No saved invalid replay results — will re-run.")

            if not _inv_loaded:
                print(f"\n[Invalid Replay] Replaying {len(invalid_scenarios)} PPO-failed scenario(s) ...")
                inv_ppo_r,   inv_ppo_en,   inv_ppo_vv   = evaluate_ppo_on_scenarios(
                    args.ppo_model, invalid_scenarios, args.bus_size)
                inv_mappo_r, inv_mappo_en, inv_mappo_vv = evaluate_mappo_on_scenarios(
                    args.mappo_model, invalid_scenarios, args.bus_size, device_str)
                inv_results = {
                    'PPO':   {'reward': inv_ppo_r,   'energy_supp': inv_ppo_en,   'volt_viol': inv_ppo_vv},
                    'MAPPO': {'reward': inv_mappo_r, 'energy_supp': inv_mappo_en, 'volt_viol': inv_mappo_vv},
                }
                save_results(inv_results, data_dir, tag='invalid_replay')

            print_comparison(inv_results, args.bus_size, label='PPO-Failed Scenarios Replay')
            if args.no_paper_figs:
                plot_comparison(inv_results, args.bus_size,
                                os.path.join(args.plot_dir, 'invalid_replay'), fmt=args.fig_format)

        # ── critical case (34-bus only) ───────────────────────────────────────
        crit_results = None
        if args.bus_size == 34:
            _crit_loaded = False
            if args.load_results:
                try:
                    print("\n[Load] Loading saved critical results ...")
                    crit_results = load_results(data_dir, tag='critical')
                    _crit_loaded = True
                except FileNotFoundError as _e:
                    print(f"  [Load] Not found ({_e.filename}) — running critical evaluation instead.")

            if not _crit_loaded:
                print("\n[Critical] Evaluating PPO+GCAPS on fixed outage scenario ...")
                cr_ppo_r, cr_ppo_en, cr_ppo_vv = evaluate_ppo_critical(
                    args.ppo_model, args.n_episodes, args.bus_size)
                print("[Critical] Evaluating MAPPO+GCAPS on fixed outage scenario ...")
                cr_mappo_r, cr_mappo_en, cr_mappo_vv = evaluate_mappo_critical(
                    args.mappo_model, args.n_episodes, args.bus_size, device_str)
                crit_results = {
                    'PPO':   {'reward': cr_ppo_r,   'energy_supp': cr_ppo_en,   'volt_viol': cr_ppo_vv},
                    'MAPPO': {'reward': cr_mappo_r, 'energy_supp': cr_mappo_en, 'volt_viol': cr_mappo_vv},
                }
                save_results(crit_results, data_dir, tag='critical')

            print(f"\nCritical outage scenario: lines 832-858, 852-854, 834-860")
            print_comparison(crit_results, args.bus_size, label='Critical Outage')
            if args.no_paper_figs:
                plot_comparison(crit_results, args.bus_size,
                                os.path.join(args.plot_dir, 'critical'), fmt=args.fig_format)
        else:
            print("\n[Critical] Critical case only supported for 34-bus. Skipping.")

    # ── paper-style figures ───────────────────────────────────────────────────
    if not args.n_fault_only and not args.no_paper_figs and args.bus_size == 34:
        print("\n[Paper Figures] Generating paper-style figures ...")
        fig_dir = os.path.join(args.plot_dir, 'paper_figures')
        try:
            print("  Running single-episode inspection on critical scenario ...")
            ppo_action,   ppo_post_obs,   meta,     crit_outedges = inspect_ppo_scenario(
                args.ppo_model, CRITICAL_OUTAGES_34, args.bus_size)
            mappo_action, mappo_post_obs, _,         _            = inspect_mappo_scenario(
                args.mappo_model, CRITICAL_OUTAGES_34, args.bus_size, device_str)

            crit_title = 'Critical Outage (lines 832-858, 852-854, 834-860)'

            # Decision heatmap — critical
            plot_decision_heatmap(
                {'PPO+GCAPS': ppo_action, 'MAPPO+GCAPS': mappo_action},
                meta, fig_dir, args.fig_format, suffix='critical'
            )
            # Voltage profile — critical
            plot_voltage_profile(
                {'PPO+GCAPS':   ppo_post_obs['NodeFeat(BusVoltage)'],
                 'MAPPO+GCAPS': mappo_post_obs['NodeFeat(BusVoltage)']},
                meta['node_list'], fig_dir, args.fig_format,
                title=crit_title, suffix='critical'
            )
            # Network topology graph — critical
            plot_network_topology(
                {'PPO+GCAPS': ppo_action, 'MAPPO+GCAPS': mappo_action},
                {'PPO+GCAPS': ppo_post_obs, 'MAPPO+GCAPS': mappo_post_obs},
                crit_outedges, meta, fig_dir, args.fig_format,
                suffix='critical', title=crit_title
            )

            # Same three figures for first PPO-failed (invalid) scenario
            if invalid_scenarios:
                sc       = invalid_scenarios[0]
                outedges = [tuple(e) for e in sc['outedges']]
                lines    = ', '.join(f"{u}-{v}" for u, v in sc['outedges'])
                inv_title = f'PPO-Failed Scenario (ep {sc["episode"]}: lines {lines})'
                print(f"  Running inspection on invalid scenario (ep {sc['episode']}: lines {lines}) ...")
                ppo_inv_action,   ppo_inv_obs,   inv_meta, inv_outedges = inspect_ppo_scenario(
                    args.ppo_model, outedges, args.bus_size)
                mappo_inv_action, mappo_inv_obs, _,        _            = inspect_mappo_scenario(
                    args.mappo_model, outedges, args.bus_size, device_str)
                plot_decision_heatmap(
                    {'PPO+GCAPS': ppo_inv_action, 'MAPPO+GCAPS': mappo_inv_action},
                    inv_meta, fig_dir, args.fig_format, suffix='invalid'
                )
                plot_voltage_profile(
                    {'PPO+GCAPS':   ppo_inv_obs['NodeFeat(BusVoltage)'],
                     'MAPPO+GCAPS': mappo_inv_obs['NodeFeat(BusVoltage)']},
                    inv_meta['node_list'], fig_dir, args.fig_format,
                    title=inv_title, suffix='invalid'
                )
                plot_network_topology(
                    {'PPO+GCAPS': ppo_inv_action, 'MAPPO+GCAPS': mappo_inv_action},
                    {'PPO+GCAPS': ppo_inv_obs, 'MAPPO+GCAPS': mappo_inv_obs},
                    inv_outedges, inv_meta, fig_dir, args.fig_format,
                    suffix='invalid', title=inv_title
                )

            # Valid-episode mean bar chart
            plot_valid_mean_bar(results, invalid_scenarios, fig_dir, args.fig_format)

            # Fig 3: training convergence (optional — needs log dirs)
            if args.ppo_log or args.mappo_log:
                plot_training_convergence(
                    args.ppo_log, args.mappo_log, fig_dir, args.fig_format)
            else:
                print("  Skipping convergence plot — provide --ppo_log and/or --mappo_log")

        except Exception as _e:
            import traceback
            print(f"\n[Paper Figures] ERROR: {_e}")
            traceback.print_exc()

    # ── N-fault scalability test ──────────────────────────────────────────────
    if args.n_fault_test or args.n_fault_only:
        import random
        print("\n[N-Fault Test] Running scalability test (N=1,2,3,4) ...")
        N_VALUES    = [1, 2, 3, 4]
        n_fault_eps = args.n_episodes
        n_fault_results = {}
        candidates  = _fault_candidates(args.bus_size)

        for n_f in N_VALUES:
            tag_nf = f'nfault_{n_f}'
            n_fault_results[n_f] = {}
            _loaded = False

            if args.load_results:
                try:
                    r = load_results(data_dir, tag=tag_nf)
                    n_fault_results[n_f] = r
                    _loaded = True
                    print(f"  [Load] Loaded N={n_f} results from {data_dir}")
                except FileNotFoundError:
                    print(f"  [Load] N={n_f} not found — running evaluation.")

            if not _loaded:
                # shared scenarios so PPO and MAPPO face identical fault sets
                random.seed(42)
                shared = [random.sample(candidates, min(n_f, len(candidates)))
                          for _ in range(n_fault_eps)]

                print(f"\n  [N={n_f}] PPO evaluation ({n_fault_eps} episodes) ...")
                ppo_r, ppo_en, ppo_vv, ppo_inv, ppo_mask = evaluate_ppo_n_faults(
                    args.ppo_model, n_f, n_fault_eps, args.bus_size,
                    fixed_scenarios=shared)

                print(f"  [N={n_f}] MAPPO evaluation ({n_fault_eps} episodes) ...")
                mpo_r, mpo_en, mpo_vv, mpo_inv, mpo_mask = evaluate_mappo_n_faults(
                    args.mappo_model, n_f, n_fault_eps, args.bus_size, device_str,
                    fixed_scenarios=shared)

                n_fault_results[n_f] = {
                    'PPO':   {
                        'reward':       ppo_r,
                        'energy_supp':  ppo_en,
                        'volt_viol':    ppo_vv,
                        'fail_rate':    np.array([ppo_inv / n_fault_eps]),
                        'invalid_mask': ppo_mask,
                    },
                    'MAPPO': {
                        'reward':       mpo_r,
                        'energy_supp':  mpo_en,
                        'volt_viol':    mpo_vv,
                        'fail_rate':    np.array([mpo_inv / n_fault_eps]),
                        'invalid_mask': mpo_mask,
                    },
                }
                save_results(n_fault_results[n_f], data_dir, tag=tag_nf)

        # Plot the comparison figure
        nf_fig_dir = os.path.join(args.plot_dir, 'paper_figures')
        plot_n_fault_comparison(n_fault_results, nf_fig_dir, args.fig_format)
        print("[N-Fault Test] Done.")

    print("Done.")
