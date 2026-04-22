"""Evaluate and compare PPO+GCAPS vs MAPPO+GCAPS on IEEE test networks.

Usage:
    python test.py --bus_size 34  --ppo_model ../RL_Training/Trained_Models/PPO_GCAPS_34bus_final
                                  --mappo_model ../RL_Training/Trained_Models/MAPPO_GCAPS_34bus_final.pt
    python test.py --bus_size 123 --ppo_model ../RL_Training/Trained_Models/PPO_GCAPS_123bus_final
                                  --mappo_model ../RL_Training/Trained_Models/MAPPO_GCAPS_123bus_final.pt
"""

import sys
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))          # RL_Testing/
_ROOT = os.path.dirname(_HERE)                               # MAPPO_Outage_Management/
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
    """Sectionalizing: closed(1), tie: open(0), loads: served(1)."""
    if bus_size == 34:
        return np.array([1]*5 + [0]*4 + [1]*10, dtype=np.float32)
    else:
        return np.array([1]*13 + [0]*9 + [1]*19, dtype=np.float32)


# ── metric extraction from obs ────────────────────────────────────────────────
def _extract_metrics(obs):
    energy_supp = float(obs['EnergySupp'].sum())
    volt_viol   = float(obs['VoltageViolation'].sum())
    return energy_supp, volt_viol


# ── PPO evaluation ────────────────────────────────────────────────────────────
def evaluate_ppo(model_path: str, n_episodes: int, bus_size: int):
    from stable_baselines3 import PPO
    from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy

    env, _, _, _, _, _ = load_env_and_info(bus_size)
    model = PPO.load(model_path, env=env)

    rewards, energy_supps, volt_viols = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        en, vv = _extract_metrics(obs)
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
        if (ep + 1) % 10 == 0:
            print(f"  PPO  ep {ep+1:4d}/{n_episodes}  reward={reward:.3f}")

    return np.array(rewards), np.array(energy_supps), np.array(volt_viols)


# ── MAPPO evaluation ──────────────────────────────────────────────────────────
def evaluate_mappo(model_path: str, n_episodes: int, bus_size: int, device_str: str):
    from RL_Methods.MAPPO.policy import MAPPOPolicy

    env, node_list, AllSwitches, dispatch_loads, Load_Buses, n_actions = \
        load_env_and_info(bus_size)

    # Auto-detect context_input_dim from env
    edge_dim          = env.observation_space['EdgeFeat(Branchflow)'].shape[0]
    context_input_dim = 1 + 1 + edge_dim

    agent_bus_indices = build_agent_bus_mapping(node_list, AllSwitches, dispatch_loads, Load_Buses)
    policy = MAPPOPolicy(
        n_agents=n_actions,
        features_dim=128,
        node_dim=3,
        context_input_dim=context_input_dim,
        agent_bus_indices=agent_bus_indices,
        device=device_str,
    )
    state_dict = torch.load(model_path, map_location=device_str)
    policy.load_state_dict(state_dict)
    policy.eval()

    rewards, energy_supps, volt_viols = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        current_actions = torch.tensor(
            _default_actions(bus_size), dtype=torch.float32
        ).unsqueeze(0).to(device_str)

        actions, _, _, _, _ = policy.get_actions(obs, current_actions, deterministic=True)
        action_np = actions.squeeze(0).cpu().numpy().astype(int)

        obs, reward, terminated, truncated, info = env.step(action_np)
        en, vv = _extract_metrics(obs)
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
        if (ep + 1) % 10 == 0:
            print(f"  MAPPO ep {ep+1:4d}/{n_episodes}  reward={reward:.3f}")

    return np.array(rewards), np.array(energy_supps), np.array(volt_viols)


# ── printing results ──────────────────────────────────────────────────────────
def print_comparison(results: dict, bus_size: int):
    header = f"{'Metric':<25} {'PPO+GCAPS':>15} {'MAPPO+GCAPS':>15}"
    print(f"\nIEEE {bus_size}-bus results")
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    metrics = [
        ("Mean Reward",      "reward",      "mean"),
        ("Std  Reward",      "reward",      "std"),
        ("Mean Energy Supp", "energy_supp", "mean"),
        ("Std  Energy Supp", "energy_supp", "std"),
        ("Mean Volt Viol",   "volt_viol",   "mean"),
        ("Std  Volt Viol",   "volt_viol",   "std"),
    ]
    for label, key, stat in metrics:
        ppo_val   = getattr(np, stat)(results['PPO'][key])
        mappo_val = getattr(np, stat)(results['MAPPO'][key])
        print(f"  {label:<23} {ppo_val:>15.4f} {mappo_val:>15.4f}")
    print("=" * len(header) + "\n")


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_comparison(results: dict, bus_size: int, save_dir: str):
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
        fname = os.path.join(save_dir, f"{key}_comparison.png")
        fig.savefig(fname, dpi=150)
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
    fname = os.path.join(save_dir, 'boxplot_summary.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {fname}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate PPO+GCAPS vs MAPPO+GCAPS")
    parser.add_argument('--bus_size',    type=int, default=123, choices=[34, 123])
    parser.add_argument('--ppo_model',   type=str,
                        default='../RL_Training/Trained_Models/PPO_GCAPS_123bus_final')
    parser.add_argument('--mappo_model', type=str,
                        default='../RL_Training/Trained_Models/MAPPO_GCAPS_123bus_final.pt')
    parser.add_argument('--n_episodes',  type=int, default=100)
    parser.add_argument('--no_cuda',     action='store_true')
    parser.add_argument('--plot_dir',    type=str, default='Results/plots')
    args = parser.parse_args()

    device_str = 'cuda:0' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    print(f"Bus size   : {args.bus_size}-bus")
    print(f"Device     : {device_str}")
    print(f"Episodes   : {args.n_episodes}")
    print(f"PPO  model : {args.ppo_model}")
    print(f"MAPPO model: {args.mappo_model}")

    results = {}

    print("\n[1/2] Evaluating PPO+GCAPS ...")
    r, en, vv = evaluate_ppo(args.ppo_model, args.n_episodes, args.bus_size)
    results['PPO'] = {'reward': r, 'energy_supp': en, 'volt_viol': vv}

    print("\n[2/2] Evaluating MAPPO+GCAPS ...")
    r, en, vv = evaluate_mappo(args.mappo_model, args.n_episodes, args.bus_size, device_str)
    results['MAPPO'] = {'reward': r, 'energy_supp': en, 'volt_viol': vv}

    print_comparison(results, args.bus_size)
    plot_comparison(results, args.bus_size, args.plot_dir)
    print("Done.")
