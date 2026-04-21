"""THz anti-jamming experiments: training, evaluation, and parameter sweeps.

Mirrors the narrowband experiments.py but uses THzAntiJammingEnv,
and adds DQN alongside the three tabular agents.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..agents import FastQAgent, FuzzyWoLFPHCAgent, TabularQAgent
from ..state import StateAggregator
from .dqn_agent import DQNAgent
from .d3qn_agent import D3QNAgent
from .spdp_ris import classical_phase_only, compute_spdp_closed_form
from .thz_config import THzRLConfig, THzSweepConfig, THzSystemConfig, THzTrainEvalConfig
from .thz_environment import THzAntiJammingEnv
from .thz_state import THzStateAggregator
from .thz_system_model import compute_normalized_array_gain, evaluate_thz_system
from ..utils import dbm_to_watt


RL_METHODS = ["q_learning", "fast_q_learning", "fuzzy_wolf_phc", "dqn"]


def _build_agent(name: str, n_actions: int, rl_cfg: THzRLConfig, seed: int):
    if name == "q_learning":
        return TabularQAgent(n_actions, rl_cfg, seed=seed)
    if name == "fast_q_learning":
        return FastQAgent(n_actions, rl_cfg, seed=seed)
    if name == "fuzzy_wolf_phc":
        agg = StateAggregator(bins=rl_cfg.state_bins, centers=rl_cfg.fuzzy_centers)
        return FuzzyWoLFPHCAgent(n_actions, agg.n_fuzzy_states, rl_cfg, seed=seed)
    if name == "dqn":
        return DQNAgent(n_actions, state_dim=3, rl_cfg=rl_cfg, seed=seed)
    if name == "d3qn":
        return D3QNAgent(n_actions, state_dim=3, rl_cfg=rl_cfg, seed=seed)
    raise ValueError(f"Unknown agent: {name}")


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_thz_agent(
    method: str,
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
    seed: int,
    log_interval: int = 0,
    finetune_episodes: int = 0,
) -> tuple[object, np.ndarray]:
    """Train an RL agent on the THz environment."""
    env = THzAntiJammingEnv(sys_cfg, rl_cfg, seed=seed)
    env.set_fast_mode(True)  # Use centroid SPDP during training for speed
    state_agg = THzStateAggregator(bins=rl_cfg.state_bins, centers=rl_cfg.fuzzy_centers)
    is_drl = method in ("dqn", "d3qn")
    agent = _build_agent(method, env.action_space.size, rl_cfg, seed=seed)

    reward_history = []
    for ep in range(run_cfg.train_episodes):
        # DRL methods: resample users each episode for generalization
        # Tabular methods: keep fixed users for stable Q-table convergence
        resample = is_drl
        obs = env.reset(resample_users=resample)
        ep_reward = 0.0
        for _ in range(run_cfg.train_steps_per_episode):
            state = state_agg.build_thz(
                obs.prev_jammer_watt,
                obs.slot.channel_quality_linear,
                obs.prev_sinr_linear,
            )

            if is_drl:
                action = agent.select_action(state, training=True)
            else:
                action = agent.select_action(state)

            next_obs, reward, _ = env.step(action)
            next_state = state_agg.build_thz(
                next_obs.prev_jammer_watt,
                next_obs.slot.channel_quality_linear,
                next_obs.prev_sinr_linear,
            )

            if is_drl:
                agent.update(state, action, reward, next_state)
            else:
                agent.update(state, action, reward, next_state)

            obs = next_obs
            ep_reward += reward

        if is_drl:
            agent.decay_epsilon()
        else:
            agent.end_episode()

        reward_history.append(ep_reward / max(1, run_cfg.train_steps_per_episode))

        if log_interval > 0 and (ep + 1) % log_interval == 0:
            avg = np.mean(reward_history[-log_interval:])
            print(f"    ep {ep+1}/{run_cfg.train_episodes}: avg_reward={avg:.2f}", flush=True)

    # Fine-tuning phase: low-epsilon exploration to polish Q-values
    if finetune_episodes > 0 and not is_drl:
        saved_eps = agent.epsilon
        agent.epsilon = max(0.02, agent.epsilon)
        for ep in range(finetune_episodes):
            obs = env.reset(resample_users=False)
            ep_reward = 0.0
            for _ in range(run_cfg.train_steps_per_episode):
                state = state_agg.build_thz(
                    obs.prev_jammer_watt,
                    obs.slot.channel_quality_linear,
                    obs.prev_sinr_linear,
                )
                action = agent.select_action(state)
                next_obs, reward, _ = env.step(action)
                next_state = state_agg.build_thz(
                    next_obs.prev_jammer_watt,
                    next_obs.slot.channel_quality_linear,
                    next_obs.prev_sinr_linear,
                )
                agent.update(state, action, reward, next_state)
                obs = next_obs
                ep_reward += reward
            reward_history.append(ep_reward / max(1, run_cfg.train_steps_per_episode))
        agent.epsilon = saved_eps

    return agent, np.asarray(reward_history, dtype=float)


def evaluate_thz_agent(
    agent,
    method: str,
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
    seed: int,
) -> tuple[float, float]:
    """Evaluate a trained agent on the THz environment."""
    env = THzAntiJammingEnv(sys_cfg, rl_cfg, seed=seed + 10_000)
    state_agg = THzStateAggregator(bins=rl_cfg.state_bins, centers=rl_cfg.fuzzy_centers)
    is_drl = method in ("dqn", "d3qn")

    if not is_drl:
        agent.set_eval_mode()

    rates, protections = [], []
    for ep_idx in range(run_cfg.eval_episodes):
        # First episode: full reset. Subsequent: keep action history
        # so jammer can detect predictable agent behavior.
        obs = env.reset(resample_users=True, keep_history=(ep_idx > 0))
        for _ in range(run_cfg.eval_steps_per_episode):
            state = state_agg.build_thz(
                obs.prev_jammer_watt,
                obs.slot.channel_quality_linear,
                obs.prev_sinr_linear,
            )
            if is_drl:
                action = agent.select_action(state, training=False)
            else:
                action = agent.select_action(state)
            obs, _, info = env.step(action)
            rates.append(info["system_rate"])
            protections.append(info["sinr_protection"])

    return float(np.mean(rates)), float(np.mean(protections))


# ---------------------------------------------------------------------------
# Detailed evaluation (collects per-step SINR for CDF analysis)
# ---------------------------------------------------------------------------

def evaluate_thz_agent_detailed(
    agent,
    method: str,
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
    seed: int,
) -> dict:
    """Evaluate with full per-step metric collection for journal plots.

    Returns dict with rate/protection stats AND raw per-step SINR arrays.
    """
    env = THzAntiJammingEnv(sys_cfg, rl_cfg, seed=seed + 10_000)
    state_agg = THzStateAggregator(bins=rl_cfg.state_bins, centers=rl_cfg.fuzzy_centers)
    is_drl = method in ("dqn", "d3qn")

    if not is_drl:
        agent.set_eval_mode()

    rates, protections = [], []
    all_sinr_db = []  # per-step average SINR in dB

    for ep_idx in range(run_cfg.eval_episodes):
        obs = env.reset(resample_users=True, keep_history=(ep_idx > 0))
        for _ in range(run_cfg.eval_steps_per_episode):
            state = state_agg.build_thz(
                obs.prev_jammer_watt,
                obs.slot.channel_quality_linear,
                obs.prev_sinr_linear,
            )
            if is_drl:
                action = agent.select_action(state, training=False)
            else:
                action = agent.select_action(state)

            # Use evaluate_action to get full metrics including per-user SINR
            metrics, _, _ = env.evaluate_action(action)
            avg_sinr = np.mean(metrics.sinr_linear, axis=1)
            obs = env._advance(avg_sinr, action_sig=action)

            rates.append(metrics.system_rate)
            protections.append(metrics.sinr_protection_level)
            # Per-user average SINR in dB
            for k in range(sys_cfg.k_users):
                user_sinr_db = 10.0 * np.log10(
                    np.mean(metrics.sinr_linear[k]) + 1e-30
                )
                all_sinr_db.append(float(user_sinr_db))

    return {
        "rate_mean": float(np.mean(rates)),
        "rate_std": float(np.std(rates)),
        "protection_mean": float(np.mean(protections)),
        "protection_std": float(np.std(protections)),
        "sinr_db_samples": all_sinr_db,
    }


def evaluate_thz_ao_baseline_detailed(
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
    seed: int,
) -> dict:
    """AO baseline with full per-step metric collection."""
    env = THzAntiJammingEnv(sys_cfg, rl_cfg, seed=seed + 20_000)
    n_fracs = len(env.action_space.total_power_fractions)
    idx_map = {tuple(c): i for i, c in enumerate(env.action_space.actions)}
    ao_action = idx_map[(n_fracs - 1, 1)]

    rates, protections = [], []
    all_sinr_db = []

    for ep_idx in range(run_cfg.eval_episodes):
        env.reset(resample_users=True, keep_history=(ep_idx > 0))
        for _ in range(run_cfg.eval_steps_per_episode):
            metrics, _, _ = env.evaluate_action(ao_action)
            avg_sinr = np.mean(metrics.sinr_linear, axis=1)
            env._advance(avg_sinr, action_sig=ao_action)

            rates.append(metrics.system_rate)
            protections.append(metrics.sinr_protection_level)
            for k in range(sys_cfg.k_users):
                user_sinr_db = 10.0 * np.log10(
                    np.mean(metrics.sinr_linear[k]) + 1e-30
                )
                all_sinr_db.append(float(user_sinr_db))

    return {
        "rate_mean": float(np.mean(rates)),
        "rate_std": float(np.std(rates)),
        "protection_mean": float(np.mean(protections)),
        "protection_std": float(np.std(protections)),
        "sinr_db_samples": all_sinr_db,
    }


# ---------------------------------------------------------------------------
# AO baseline (greedy: full power + channel proportional)
# ---------------------------------------------------------------------------

def evaluate_thz_ao_baseline(
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
    seed: int,
) -> tuple[float, float]:
    env = THzAntiJammingEnv(sys_cfg, rl_cfg, seed=seed + 20_000)
    n_fracs = len(env.action_space.total_power_fractions)
    idx_map = {tuple(c): i for i, c in enumerate(env.action_space.actions)}
    ao_action = idx_map[(n_fracs - 1, 1)]  # full power, channel_proportional

    rates, protections = [], []
    for ep_idx in range(run_cfg.eval_episodes):
        env.reset(resample_users=True, keep_history=(ep_idx > 0))
        for _ in range(run_cfg.eval_steps_per_episode):
            _, _, info = env.step(ao_action)
            rates.append(info["system_rate"])
            protections.append(info["sinr_protection"])

    return float(np.mean(rates)), float(np.mean(protections))


# ---------------------------------------------------------------------------
# Convergence and sweep experiments
# ---------------------------------------------------------------------------

def run_thz_convergence(
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
) -> dict[str, np.ndarray]:
    """Run convergence experiment for all RL methods."""
    all_histories: dict[str, list[np.ndarray]] = {m: [] for m in RL_METHODS}

    for run_idx in range(run_cfg.n_seeds):
        seed = sys_cfg.seed + 101 * run_idx
        for method in RL_METHODS:
            print(f"  Convergence: seed {run_idx+1}/{run_cfg.n_seeds}, {method}", flush=True)
            _, history = train_thz_agent(method, sys_cfg, rl_cfg, run_cfg, seed)
            all_histories[method].append(history)

    return {m: np.mean(np.stack(h, axis=0), axis=0) for m, h in all_histories.items()}


def _evaluate_method(
    method: str,
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
    seed: int,
) -> tuple[float, float]:
    if method in RL_METHODS:
        # WoLF-PHC benefits from extended training + fine-tuning for policy convergence
        if method == "fuzzy_wolf_phc":
            run_cfg_m = replace(run_cfg, train_episodes=int(run_cfg.train_episodes * 3.0))
            agent, _ = train_thz_agent(method, sys_cfg, rl_cfg, run_cfg_m, seed,
                                       finetune_episodes=80)
        else:
            run_cfg_m = run_cfg
            agent, _ = train_thz_agent(method, sys_cfg, rl_cfg, run_cfg_m, seed)
        return evaluate_thz_agent(agent, method, sys_cfg, rl_cfg, run_cfg, seed)
    if method == "baseline_ao":
        return evaluate_thz_ao_baseline(sys_cfg, rl_cfg, run_cfg, seed)
    raise ValueError(f"Unknown method: {method}")


def run_thz_parameter_sweep(
    parameter: str,
    values: list,
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
) -> dict:
    """Run a parameter sweep over the THz system.
    
    Returns statistics (mean, std, min, max) for proper error analysis.
    """
    methods = RL_METHODS + ["baseline_ao"]
    out = {
        "x": [float(v) for v in values],
        "parameter": parameter,
        "methods": {
            m: {
                "rate_mean": [], "rate_std": [], "rate_min": [], "rate_max": [],
                "protection_mean": [], "protection_std": [],
                # Keep backward compatibility
                "rate": [], "protection": [],
            }
            for m in methods
        },
    }

    for vi, value in enumerate(values):
        print(f"  Sweep {parameter}={value} ({vi+1}/{len(values)})", flush=True)

        if parameter == "pmax_dbm":
            cfg = replace(sys_cfg, pmax_dbm=float(value))
        elif parameter == "n_ris_total":
            # Square RIS: N1 = N2 = sqrt(value)
            side = int(math.isqrt(int(value)))
            # Also scale Q proportionally for SPDP effectiveness
            q_side = max(1, side // 8)
            cfg = replace(sys_cfg, n_ris_h=side, n_ris_v=side,
                         q_subarrays_h=q_side, q_subarrays_v=q_side)
        elif parameter == "bandwidth_hz":
            cfg = replace(sys_cfg, bandwidth_hz=float(value))
        elif parameter == "q_subarrays":
            q = int(math.isqrt(int(value)))
            cfg = replace(sys_cfg, q_subarrays_h=q, q_subarrays_v=q)
        elif parameter == "sinr_min_db":
            cfg = replace(sys_cfg, sinr_min_db=float(value))
        elif parameter == "p_jammer_max_dbm":
            cfg = replace(sys_cfg, p_jammer_max_dbm=float(value))
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")

        method_rates = {m: [] for m in methods}
        method_prots = {m: [] for m in methods}

        for run_idx in range(run_cfg.n_seeds):
            seed = cfg.seed + 211 * run_idx
            for method in methods:
                rate, prot = _evaluate_method(method, cfg, rl_cfg, run_cfg, seed)
                method_rates[method].append(rate)
                method_prots[method].append(prot)

        for method in methods:
            rates = np.array(method_rates[method])
            prots = np.array(method_prots[method])
            
            out["methods"][method]["rate_mean"].append(float(np.mean(rates)))
            out["methods"][method]["rate_std"].append(float(np.std(rates)))
            out["methods"][method]["rate_min"].append(float(np.min(rates)))
            out["methods"][method]["rate_max"].append(float(np.max(rates)))
            out["methods"][method]["protection_mean"].append(float(np.mean(prots)))
            out["methods"][method]["protection_std"].append(float(np.std(prots)))
            # Backward compatibility
            out["methods"][method]["rate"].append(float(np.mean(rates)))
            out["methods"][method]["protection"].append(float(np.mean(prots)))

    return out


# ---------------------------------------------------------------------------
# Physics validation sweep (no RL, fast evaluation)
# ---------------------------------------------------------------------------

def run_physics_validation_sweep(
    parameter: str,
    values: list,
    sys_cfg: THzSystemConfig,
    n_trials: int = 10,
) -> dict:
    """Run a physics validation sweep without RL overhead.
    
    Evaluates system performance with optimal SPDP and fixed power allocation.
    Useful for verifying RIS scaling, beam squint effects, etc.
    """
    from .spdp_ris import compute_spdp_closed_form
    from .thz_channel_model import THzChannelModel, THzTopology
    
    out = {
        "x": [float(v) for v in values],
        "parameter": parameter,
        "rate_mean": [], "rate_std": [],
        "sinr_mean_db": [], "sinr_std_db": [],
        "metrics": [],  # Per-value diagnostics
    }

    for vi, value in enumerate(values):
        print(f"  Physics sweep {parameter}={value} ({vi+1}/{len(values)})", flush=True)

        if parameter == "pmax_dbm":
            cfg = replace(sys_cfg, pmax_dbm=float(value))
        elif parameter == "n_ris_total":
            side = int(math.isqrt(int(value)))
            q_side = max(1, side // 8)
            cfg = replace(sys_cfg, n_ris_h=side, n_ris_v=side,
                         q_subarrays_h=q_side, q_subarrays_v=q_side)
        elif parameter == "bandwidth_hz":
            cfg = replace(sys_cfg, bandwidth_hz=float(value))
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")

        rates = []
        sinrs_db = []

        for trial in range(n_trials):
            rng = np.random.default_rng(cfg.seed + trial * 37)
            topology = THzTopology(cfg, rng)
            channel_model = THzChannelModel(cfg, rng)
            snapshot = channel_model.sample(topology)
            
            # Optimal SPDP for centroid direction
            spdp = compute_spdp_closed_form(
                cfg,
                snapshot.theta_aoa, snapshot.zeta_aoa,
                float(np.mean(snapshot.theta_aod)),
                float(np.mean(snapshot.zeta_aod))
            )
            
            # Equal power allocation
            p_bs = np.full(cfg.k_users, dbm_to_watt(cfg.pmax_dbm) / cfg.k_users)
            p_jam = np.zeros(cfg.k_users)
            z_jam = np.zeros((cfg.k_users, cfg.n_jammer_antennas))
            noise_w = dbm_to_watt(cfg.noise_power_dbm)
            
            metrics = evaluate_thz_system(
                cfg, snapshot, spdp, p_bs, p_jam, z_jam, noise_w,
                sinr_min_db=cfg.sinr_min_db, lambda1=0.0, lambda2=0.0
            )
            
            rates.append(metrics.system_rate)
            avg_sinr_lin = np.mean(metrics.sinr_linear)
            sinrs_db.append(10 * np.log10(avg_sinr_lin + 1e-30))

        out["rate_mean"].append(float(np.mean(rates)))
        out["rate_std"].append(float(np.std(rates)))
        out["sinr_mean_db"].append(float(np.mean(sinrs_db)))
        out["sinr_std_db"].append(float(np.std(sinrs_db)))
        out["metrics"].append({
            "value": float(value),
            "n_ris": cfg.n_ris_total,
            "bandwidth_hz": cfg.bandwidth_hz,
            "noise_power_dbm": cfg.noise_power_dbm,
        })

    return out


# ---------------------------------------------------------------------------
# Beam squint analysis (standalone, no RL)
# ---------------------------------------------------------------------------

def run_beam_squint_analysis(
    sys_cfg: THzSystemConfig,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Compute normalized array gain vs subcarrier for SPDP vs classical.

    Returns dict mapping scheme name -> (M,) array gain per subcarrier.
    """
    from .thz_channel_model import THzChannelModel, THzTopology

    rng = np.random.default_rng(seed)
    topology = THzTopology(sys_cfg, rng)
    channel_model = THzChannelModel(sys_cfg, rng)
    snapshot = channel_model.sample(topology)

    # Reference angles (first user)
    theta_aoa = snapshot.theta_aoa
    zeta_aoa = snapshot.zeta_aoa
    theta_aod = float(snapshot.theta_aod[0])
    zeta_aod = float(snapshot.zeta_aod[0])

    results = {}

    # Classical (no TD)
    spdp_classical = classical_phase_only(sys_cfg, theta_aoa, zeta_aoa, theta_aod, zeta_aod)
    results["Classical (no TD)"] = compute_normalized_array_gain(sys_cfg, snapshot, spdp_classical, 0)

    # SPDP with different Q values
    for Q_total in [1, 16, 64]:
        Q_side = max(1, int(math.isqrt(Q_total)))
        cfg_q = replace(sys_cfg, q_subarrays_h=Q_side, q_subarrays_v=Q_side)
        spdp = compute_spdp_closed_form(cfg_q, theta_aoa, zeta_aoa, theta_aod, zeta_aod)
        results[f"SPDP Q={Q_total}"] = compute_normalized_array_gain(cfg_q, snapshot, spdp, 0)

    return results
