"""Evaluate and compare PPO+GCAPS vs MAPPO+GCAPS on IEEE test networks.

Usage:
    # Run full evaluation and save results
    python test.py --bus_size 34  --ppo_model ../RL_Training/Trained_Models/PPO_GCAPS_34bus_final
                                  --mappo_model ../RL_Training/Trained_Models/MAPPO_GCAPS_34bus_final.pt
                                  --fig_format pdf

    # Re-plot from saved results without re-running models
    python test.py --bus_size 34 --load_results --plot_dir Results/plots --fig_format eps
"""

import sys
import os
import json
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


# ── save / load helpers ───────────────────────────────────────────────────────
def save_results(results: dict, save_dir: str, tag: str):
    os.makedirs(save_dir, exist_ok=True)
    for algo, data in results.items():
        path = os.path.join(save_dir, f"{tag}_{algo}.npz")
        np.savez(path, **data)
        print(f"  Saved {algo} results: {path}")


def load_results(save_dir: str, tag: str, algos=('PPO', 'MAPPO')):
    results = {}
    for algo in algos:
        path = os.path.join(save_dir, f"{tag}_{algo}.npz")
        data = np.load(path)
        results[algo] = {k: data[k] for k in data.files}
        print(f"  Loaded {algo} results: {path}")
    return results


def save_invalid_scenarios(scenarios: list, save_dir: str, tag: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{tag}_invalid_scenarios.json")
    with open(path, 'w') as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Saved invalid scenarios: {path}")


def load_invalid_scenarios(save_dir: str, tag: str):
    path = os.path.join(save_dir, f"{tag}_invalid_scenarios.json")
    with open(path) as f:
        return json.load(f)


# ── PPO evaluation ────────────────────────────────────────────────────────────
def evaluate_ppo(model_path: str, n_episodes: int, bus_size: int):
    from stable_baselines3 import PPO
    from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy

    env, _, _, _, _, _ = load_env_and_info(bus_size)
    model = PPO.load(model_path, env=env)

    rewards, energy_supps, volt_viols = [], [], []
    invalid_count    = 0
    invalid_scenarios = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        en, vv = _extract_metrics(obs)
        if abs(en) > 10 or abs(vv) > 10:
            invalid_count += 1
            invalid_scenarios.append([list(e) for e in env.outedges])
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
        if (ep + 1) % 10 == 0:
            print(f"  PPO  ep {ep+1:4d}/{n_episodes}  reward={reward:.3f}")

    print(f"  PPO  invalid episodes: {invalid_count}/{n_episodes}")
    return np.array(rewards), np.array(energy_supps), np.array(volt_viols), invalid_scenarios


# ── MAPPO evaluation ──────────────────────────────────────────────────────────
def evaluate_mappo(model_path: str, n_episodes: int, bus_size: int, device_str: str):
    from RL_Methods.MAPPO.policy import MAPPOPolicy

    env, node_list, AllSwitches, dispatch_loads, Load_Buses, n_actions = \
        load_env_and_info(bus_size)

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
    invalid_count = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        current_actions = torch.tensor(
            _default_actions(bus_size), dtype=torch.float32
        ).unsqueeze(0).to(device_str)

        actions, _, _, _, _ = policy.get_actions(obs, current_actions, deterministic=True)
        action_np = actions.squeeze(0).cpu().numpy().astype(int)

        obs, reward, terminated, truncated, info = env.step(action_np)
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
def evaluate_ppo_on_scenarios(model_path: str, scenarios: list, bus_size: int):
    """Run PPO on each saved invalid outage scenario."""
    from stable_baselines3 import PPO
    from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env

    env   = DSS_OutCtrl_Env()
    model = PPO.load(model_path, env=env)

    rewards, energy_supps, volt_viols = [], [], []
    for i, outedges in enumerate(scenarios):
        obs, _ = env.reset(options={'fixed_outages': [tuple(e) for e in outedges]})
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        en, vv = _extract_metrics(obs)
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
        print(f"  PPO  invalid-replay {i+1}/{len(scenarios)}  reward={reward:.3f}  energy={en:.3f}  volt_viol={vv:.3f}")
    return np.array(rewards), np.array(energy_supps), np.array(volt_viols)


def evaluate_mappo_on_scenarios(model_path: str, scenarios: list, bus_size: int, device_str: str):
    """Run MAPPO on each saved invalid outage scenario."""
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
    for i, outedges in enumerate(scenarios):
        obs, _ = env.reset(options={'fixed_outages': [tuple(e) for e in outedges]})
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
        print(f"  MAPPO invalid-replay {i+1}/{len(scenarios)}  reward={reward:.3f}  energy={en:.3f}  volt_viol={vv:.3f}")
    return np.array(rewards), np.array(energy_supps), np.array(volt_viols)


# ── critical case evaluation ─────────────────────────────────────────────────
CRITICAL_OUTAGES_34 = [('832', '858'), ('852', '854'), ('834', '860')]


def evaluate_ppo_critical(model_path: str, n_episodes: int, bus_size: int):
    from stable_baselines3 import PPO
    from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy
    from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env

    env   = DSS_OutCtrl_Env()
    model = PPO.load(model_path, env=env)

    rewards, energy_supps, volt_viols = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(options={'fixed_outages': CRITICAL_OUTAGES_34})
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        en, vv = _extract_metrics(obs)
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
    print(f"  PPO  critical mean reward: {np.mean(rewards):.4f}")
    return np.array(rewards), np.array(energy_supps), np.array(volt_viols)


def evaluate_mappo_critical(model_path: str, n_episodes: int, bus_size: int, device_str: str):
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
        obs, reward, terminated, truncated, info = env.step(action_np)
        en, vv = _extract_metrics(obs)
        rewards.append(float(reward))
        energy_supps.append(en)
        volt_viols.append(vv)
    print(f"  MAPPO critical mean reward: {np.mean(rewards):.4f}")
    return np.array(rewards), np.array(energy_supps), np.array(volt_viols)


# ── printing results ──────────────────────────────────────────────────────────
def print_comparison(results: dict, bus_size: int, label: str = ''):
    header = f"{'Metric':<25} {'PPO+GCAPS':>15} {'MAPPO+GCAPS':>15}"
    title  = f"IEEE {bus_size}-bus results" + (f" — {label}" if label else "")
    print(f"\n{title}")
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
    for label_m, key, stat in metrics:
        ppo_val   = getattr(np, stat)(results['PPO'][key])
        mappo_val = getattr(np, stat)(results['MAPPO'][key])
        print(f"  {label_m:<23} {ppo_val:>15.4f} {mappo_val:>15.4f}")
    print("=" * len(header) + "\n")


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_comparison(results: dict, bus_size: int, save_dir: str, fmt: str = 'pdf'):
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
                        choices=['pdf', 'eps', 'png', 'svg'],
                        help='Figure file format (default: pdf)')
    parser.add_argument('--load_results', action='store_true',
                        help='Skip evaluation and re-plot from saved .npz files')
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
        results = load_results(data_dir, tag='random')
        invalid_scenarios = load_invalid_scenarios(data_dir, tag='random')
        print(f"  Loaded {len(invalid_scenarios)} invalid scenario(s)")
    else:
        print(f"PPO  model : {args.ppo_model}")
        print(f"MAPPO model: {args.mappo_model}")
        results = {}

        print("\n[1/2] Evaluating PPO+GCAPS ...")
        r, en, vv, invalid_scenarios = evaluate_ppo(args.ppo_model, args.n_episodes, args.bus_size)
        results['PPO'] = {'reward': r, 'energy_supp': en, 'volt_viol': vv}

        print("\n[2/2] Evaluating MAPPO+GCAPS ...")
        r, en, vv = evaluate_mappo(args.mappo_model, args.n_episodes, args.bus_size, device_str)
        results['MAPPO'] = {'reward': r, 'energy_supp': en, 'volt_viol': vv}

        save_results(results, data_dir, tag='random')
        save_invalid_scenarios(invalid_scenarios, data_dir, tag='random')

    print_comparison(results, args.bus_size, label='Random Episodes')
    plot_comparison(results, args.bus_size, args.plot_dir, fmt=args.fig_format)

    # ── invalid scenario replay ───────────────────────────────────────────────
    if invalid_scenarios:
        print(f"\n[Invalid Replay] Replaying {len(invalid_scenarios)} PPO-failed scenario(s) with both models ...")
        inv_ppo_r,   inv_ppo_en,   inv_ppo_vv   = evaluate_ppo_on_scenarios(
            args.ppo_model, invalid_scenarios, args.bus_size
        )
        inv_mappo_r, inv_mappo_en, inv_mappo_vv = evaluate_mappo_on_scenarios(
            args.mappo_model, invalid_scenarios, args.bus_size, device_str
        )
        inv_results = {
            'PPO':   {'reward': inv_ppo_r,   'energy_supp': inv_ppo_en,   'volt_viol': inv_ppo_vv},
            'MAPPO': {'reward': inv_mappo_r, 'energy_supp': inv_mappo_en, 'volt_viol': inv_mappo_vv},
        }
        print_comparison(inv_results, args.bus_size, label='PPO-Failed Scenarios Replay')
        save_results(inv_results, data_dir, tag='invalid_replay')

    # ── critical case evaluation (34-bus only) ────────────────────────────────
    if args.bus_size == 34:
        if args.load_results:
            print("\n[Load] Loading saved critical results ...")
            crit_results = load_results(data_dir, tag='critical')
        else:
            print("\n[Critical] Evaluating PPO+GCAPS on fixed outage scenario ...")
            cr_ppo_r, cr_ppo_en, cr_ppo_vv = evaluate_ppo_critical(
                args.ppo_model, args.n_episodes, args.bus_size
            )
            print("[Critical] Evaluating MAPPO+GCAPS on fixed outage scenario ...")
            cr_mappo_r, cr_mappo_en, cr_mappo_vv = evaluate_mappo_critical(
                args.mappo_model, args.n_episodes, args.bus_size, device_str
            )
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
        print("\n[Critical] Critical case evaluation only supported for 34-bus. Skipping.")

    print("Done.")
