"""Evaluate and compare PPO+GCAPS vs MAPPO+GCAPS on IEEE test networks.

Usage:
    # Full evaluation + paper figures
    python test.py --bus_size 34
                   --ppo_model ../RL_Training/Trained_Models/PPO_GCAPS_34bus_final
                   --mappo_model ../RL_Training/Trained_Models/MAPPO_GCAPS_34bus_final.pt
                   --fig_format pdf --paper_figs

    # Re-plot from saved results (no re-evaluation)
    python test.py --bus_size 34 --load_results --paper_figs --fig_format eps
                   --ppo_model <path>  --mappo_model <path>

    # Training convergence (needs log dirs)
    python test.py ... --paper_figs
                   --ppo_log /content/drive/.../ppo_logs
                   --mappo_log /content/drive/.../MAPPO_GCAPS_34bus
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

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)


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


# ── single-episode inspection (for paper figures) ─────────────────────────────
def inspect_ppo_scenario(model_path, outages, bus_size):
    from stable_baselines3 import PPO
    from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
    from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import (
        node_list, AllSwitches, sectional_swt, tie_swt, dispatch_loads, n_actions
    )

    env   = DSS_OutCtrl_Env()
    model = PPO.load(model_path, env=env)
    obs, _ = env.reset(options={'fixed_outages': outages})
    mask = obs['ActionMasking'].copy()
    action, _ = model.predict(obs, deterministic=True)
    post_obs, _, _, _, _ = env.step(action)
    meta = {
        'node_list': node_list, 'AllSwitches': AllSwitches,
        'sectional_swt': sectional_swt, 'tie_swt': tie_swt,
        'dispatch_loads': dispatch_loads, 'mask': mask,
    }
    return action, post_obs, meta


def inspect_mappo_scenario(model_path, outages, bus_size, device_str):
    from RL_Methods.MAPPO.policy import MAPPOPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
    from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import (
        node_list, AllSwitches, sectional_swt, tie_swt, dispatch_loads, Load_Buses, n_actions
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
    meta = {
        'node_list': node_list, 'AllSwitches': AllSwitches,
        'sectional_swt': sectional_swt, 'tie_swt': tie_swt,
        'dispatch_loads': dispatch_loads, 'mask': mask,
    }
    return action_np, post_obs, meta


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
        ax.set_title(f'PPO vs MAPPO — {title} ({bus_size}-bus)')
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
        ax.set_title(title)
    fig.suptitle(f'PPO+GCAPS vs MAPPO+GCAPS — IEEE {bus_size}-bus')
    fig.tight_layout()
    fname = os.path.join(save_dir, f'boxplot_summary.{fmt}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {fname}")


# ── paper-style Figure 1: decision variables heatmap ─────────────────────────
def plot_decision_heatmap(actions_dict, meta, save_dir, fmt):
    n_sect = len(meta['sectional_swt'])
    n_tie  = len(meta['tie_swt'])
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
        ax.set_title(section_title, fontsize=10)
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

    fig.suptitle(
        'Decision Variables — Switch and Load Status\n(red border = outage switch, light=0 open, dark=1 closed)',
        fontsize=11
    )
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f'fig1_decision_heatmap.{fmt}')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── paper-style Figure 2: voltage profile ────────────────────────────────────
def plot_voltage_profile(voltages_dict, node_list, save_dir, fmt, title=''):
    n_algos = len(voltages_dict)
    fig, axes = plt.subplots(n_algos, 1, figsize=(14, 4 * n_algos), squeeze=False)

    phase_styles = [
        ('Phase a', 's', 'red'),
        ('Phase b', '^', 'blue'),
        ('Phase c', 'o', 'green'),
    ]
    x    = np.arange(len(node_list))
    step = max(1, len(node_list) // 20)

    for ax, (algo, voltages) in zip(axes.flatten(), voltages_dict.items()):
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
        ax.set_title(f'{algo} — {title}')
        ax.set_xticks(x[::step])
        ax.set_xticklabels(
            [node_list[i] for i in x[::step]], rotation=45, ha='right', fontsize=7
        )
        ax.legend(loc='lower right', fontsize=8, ncol=3)
        ax.set_ylim(0.85, 1.15)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f'fig2_voltage_profile.{fmt}')
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
    ax.set_title('Energy Served Comparison — PPO+GCAPS vs MAPPO+GCAPS')
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
    ax.set_title('Training Convergence — 34-bus Network')
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
    parser.add_argument('--bus_size',     type=int,  default=123, choices=[34, 123])
    parser.add_argument('--ppo_model',    type=str,
                        default='../RL_Training/Trained_Models/PPO_GCAPS_123bus_final')
    parser.add_argument('--mappo_model',  type=str,
                        default='../RL_Training/Trained_Models/MAPPO_GCAPS_123bus_final.pt')
    parser.add_argument('--n_episodes',   type=int,  default=100)
    parser.add_argument('--no_cuda',      action='store_true')
    parser.add_argument('--plot_dir',     type=str,  default='Results/plots')
    parser.add_argument('--fig_format',   type=str,  default='pdf',
                        choices=['pdf', 'eps', 'png', 'svg'])
    parser.add_argument('--load_results', action='store_true',
                        help='Skip evaluation and re-plot from saved .npz files')
    parser.add_argument('--paper_figs',   action='store_true',
                        help='Generate paper-style figures (heatmap, voltage, energy bar, convergence)')
    parser.add_argument('--ppo_log',      type=str,  default=None,
                        help='PPO training log dir (for convergence plot)')
    parser.add_argument('--mappo_log',    type=str,  default=None,
                        help='MAPPO TensorBoard log dir (for convergence plot)')
    args = parser.parse_args()

    device_str = 'cuda:0' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    data_dir   = os.path.join(args.plot_dir, 'data')

    print(f"Bus size   : {args.bus_size}-bus")
    print(f"Device     : {device_str}")
    print(f"Episodes   : {args.n_episodes}")
    print(f"Fig format : {args.fig_format}")

    # ── random episode evaluation ─────────────────────────────────────────────
    if args.load_results:
        print("\n[Load] Loading saved results ...")
        results           = load_results(data_dir, tag='random')
        invalid_scenarios = load_invalid_scenarios(data_dir, tag='random')
        print(f"  Loaded {len(invalid_scenarios)} invalid scenario(s)")
    else:
        print(f"PPO  model : {args.ppo_model}")
        print(f"MAPPO model: {args.mappo_model}")
        results = {}

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
    plot_comparison(results, args.bus_size, args.plot_dir, fmt=args.fig_format)
    print_invalid_scenarios(invalid_scenarios)

    # ── invalid scenario replay ───────────────────────────────────────────────
    inv_results = None
    if invalid_scenarios:
        print(f"\n[Invalid Replay] Replaying {len(invalid_scenarios)} PPO-failed scenario(s) ...")
        inv_ppo_r,   inv_ppo_en,   inv_ppo_vv   = evaluate_ppo_on_scenarios(
            args.ppo_model, invalid_scenarios, args.bus_size)
        inv_mappo_r, inv_mappo_en, inv_mappo_vv = evaluate_mappo_on_scenarios(
            args.mappo_model, invalid_scenarios, args.bus_size, device_str)
        inv_results = {
            'PPO':   {'reward': inv_ppo_r,   'energy_supp': inv_ppo_en,   'volt_viol': inv_ppo_vv},
            'MAPPO': {'reward': inv_mappo_r, 'energy_supp': inv_mappo_en, 'volt_viol': inv_mappo_vv},
        }
        print_comparison(inv_results, args.bus_size, label='PPO-Failed Scenarios Replay')
        save_results(inv_results, data_dir, tag='invalid_replay')
        plot_comparison(inv_results, args.bus_size,
                        os.path.join(args.plot_dir, 'invalid_replay'), fmt=args.fig_format)

    # ── critical case (34-bus only) ───────────────────────────────────────────
    crit_results = None
    if args.bus_size == 34:
        if args.load_results:
            print("\n[Load] Loading saved critical results ...")
            crit_results = load_results(data_dir, tag='critical')
        else:
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
        plot_comparison(crit_results, args.bus_size,
                        os.path.join(args.plot_dir, 'critical'), fmt=args.fig_format)
    else:
        print("\n[Critical] Critical case only supported for 34-bus. Skipping.")

    # ── paper-style figures ───────────────────────────────────────────────────
    if args.paper_figs and args.bus_size == 34:
        print("\n[Paper Figures] Generating paper-style figures ...")
        fig_dir = os.path.join(args.plot_dir, 'paper_figures')

        print("  Running single-episode inspection on critical scenario ...")
        ppo_action,   ppo_post_obs,   meta = inspect_ppo_scenario(
            args.ppo_model, CRITICAL_OUTAGES_34, args.bus_size)
        mappo_action, mappo_post_obs, _    = inspect_mappo_scenario(
            args.mappo_model, CRITICAL_OUTAGES_34, args.bus_size, device_str)

        # Fig 1: decision variables heatmap
        plot_decision_heatmap(
            {'PPO+GCAPS': ppo_action, 'MAPPO+GCAPS': mappo_action},
            meta, fig_dir, args.fig_format
        )

        # Fig 2: voltage profile per bus/phase
        plot_voltage_profile(
            {'PPO+GCAPS':   ppo_post_obs['NodeFeat(BusVoltage)'],
             'MAPPO+GCAPS': mappo_post_obs['NodeFeat(BusVoltage)']},
            meta['node_list'], fig_dir, args.fig_format,
            title='Critical Outage (lines 832-858, 852-854, 834-860)'
        )

        # Fig 3: energy served bar chart (median, robust to outliers)
        energy_scenarios = {'Random Episodes': results}
        if crit_results is not None:
            energy_scenarios['Critical Outage'] = crit_results
        if inv_results is not None:
            energy_scenarios['Failed Scenario Replay'] = inv_results
        plot_energy_bar(energy_scenarios, fig_dir, args.fig_format)

        # Fig 4: training convergence (optional — needs log dirs)
        if args.ppo_log or args.mappo_log:
            plot_training_convergence(
                args.ppo_log, args.mappo_log, fig_dir, args.fig_format)
        else:
            print("  Skipping convergence plot — provide --ppo_log and/or --mappo_log")

    print("Done.")
