from __future__ import annotations

from dataclasses import replace

import numpy as np

from .agents import FastQAgent, FuzzyWoLFPHCAgent, TabularQAgent
from .baselines import AOGreedyBaseline, NoIRSPowerOnlyBaseline
from .config import RLConfig, SystemConfig, TrainEvalConfig
from .environment import IRSAntiJammingEnv
from .state import StateAggregator


def _build_agent(name: str, env: IRSAntiJammingEnv, rl_cfg: RLConfig, seed: int):
    if name == "q_learning":
        return TabularQAgent(env.action_space.size, rl_cfg, seed=seed)
    if name == "fast_q_learning":
        return FastQAgent(env.action_space.size, rl_cfg, seed=seed)
    if name == "fuzzy_wolf_phc":
        agg = StateAggregator(bins=rl_cfg.state_bins, centers=rl_cfg.fuzzy_centers)
        return FuzzyWoLFPHCAgent(env.action_space.size, agg.n_fuzzy_states, rl_cfg, seed=seed)
    raise ValueError(f"Unsupported agent: {name}")


def train_rl_agent(
    method: str,
    sys_cfg: SystemConfig,
    rl_cfg: RLConfig,
    run_cfg: TrainEvalConfig,
    seed: int,
) -> tuple[object, np.ndarray]:
    env = IRSAntiJammingEnv(sys_cfg, rl_cfg, seed=seed)
    state_agg = StateAggregator(bins=rl_cfg.state_bins, centers=rl_cfg.fuzzy_centers)
    agent = _build_agent(method, env, rl_cfg, seed=seed)

    reward_history = []
    for _ in range(run_cfg.train_episodes):
        obs = env.reset(resample_users=False)
        ep_reward = 0.0
        for _ in range(run_cfg.train_steps_per_episode):
            state = state_agg.build(obs.prev_jammer_watt, obs.slot.channel_quality_linear, obs.prev_sinr_linear)
            action = agent.select_action(state)
            next_obs, reward, _ = env.step(action)
            next_state = state_agg.build(
                next_obs.prev_jammer_watt,
                next_obs.slot.channel_quality_linear,
                next_obs.prev_sinr_linear,
            )
            agent.update(state, action, reward, next_state)
            obs = next_obs
            ep_reward += reward

        agent.end_episode()
        reward_history.append(ep_reward / max(1, run_cfg.train_steps_per_episode))

    return agent, np.asarray(reward_history, dtype=float)


def evaluate_rl_agent(
    agent,
    sys_cfg: SystemConfig,
    rl_cfg: RLConfig,
    run_cfg: TrainEvalConfig,
    seed: int,
) -> tuple[float, float]:
    env = IRSAntiJammingEnv(sys_cfg, rl_cfg, seed=seed + 10_000)
    state_agg = StateAggregator(bins=rl_cfg.state_bins, centers=rl_cfg.fuzzy_centers)
    agent.set_eval_mode()

    rates: list[float] = []
    protections: list[float] = []

    for _ in range(run_cfg.eval_episodes):
        obs = env.reset(resample_users=True)
        for _ in range(run_cfg.eval_steps_per_episode):
            state = state_agg.build(obs.prev_jammer_watt, obs.slot.channel_quality_linear, obs.prev_sinr_linear)
            action = agent.select_action(state)
            obs, _, info = env.step(action)
            rates.append(info["system_rate"])
            protections.append(info["sinr_protection"])

    return float(np.mean(rates)), float(np.mean(protections))


def evaluate_ao_baseline(
    sys_cfg: SystemConfig,
    rl_cfg: RLConfig,
    run_cfg: TrainEvalConfig,
    seed: int,
) -> tuple[float, float]:
    env = IRSAntiJammingEnv(sys_cfg, rl_cfg, seed=seed + 20_000)
    baseline = AOGreedyBaseline()

    rates: list[float] = []
    protections: list[float] = []
    for _ in range(run_cfg.eval_episodes):
        _ = env.reset(resample_users=True)
        baseline.reset()
        for _ in range(run_cfg.eval_steps_per_episode):
            action = baseline.select_action(env)
            _, _, info = env.step(action)
            rates.append(info["system_rate"])
            protections.append(info["sinr_protection"])

    return float(np.mean(rates)), float(np.mean(protections))


def evaluate_no_irs_baseline(
    sys_cfg: SystemConfig,
    rl_cfg: RLConfig,
    run_cfg: TrainEvalConfig,
    seed: int,
) -> tuple[float, float]:
    env = IRSAntiJammingEnv(sys_cfg, rl_cfg, seed=seed + 30_000)
    baseline = NoIRSPowerOnlyBaseline()

    rates: list[float] = []
    protections: list[float] = []
    for _ in range(run_cfg.eval_episodes):
        _ = env.reset(resample_users=True)
        for _ in range(run_cfg.eval_steps_per_episode):
            _, info = baseline.run_step(env)
            rates.append(info["system_rate"])
            protections.append(info["sinr_protection"])

    return float(np.mean(rates)), float(np.mean(protections))


def run_convergence_experiment(
    sys_cfg: SystemConfig,
    rl_cfg: RLConfig,
    run_cfg: TrainEvalConfig,
) -> dict[str, np.ndarray]:
    methods = ["q_learning", "fast_q_learning", "fuzzy_wolf_phc"]
    all_histories: dict[str, list[np.ndarray]] = {m: [] for m in methods}

    for run_idx in range(run_cfg.n_seeds):
        seed = sys_cfg.seed + 101 * run_idx
        for method in methods:
            _, history = train_rl_agent(method, sys_cfg, rl_cfg, run_cfg, seed=seed)
            all_histories[method].append(history)

    return {m: np.mean(np.stack(h, axis=0), axis=0) for m, h in all_histories.items()}


def _evaluate_method_for_value(
    method: str,
    sys_cfg: SystemConfig,
    rl_cfg: RLConfig,
    run_cfg: TrainEvalConfig,
    seed: int,
) -> tuple[float, float]:
    if method in {"q_learning", "fast_q_learning", "fuzzy_wolf_phc"}:
        agent, _ = train_rl_agent(method, sys_cfg, rl_cfg, run_cfg, seed=seed)
        return evaluate_rl_agent(agent, sys_cfg, rl_cfg, run_cfg, seed=seed)
    if method == "baseline_ao":
        return evaluate_ao_baseline(sys_cfg, rl_cfg, run_cfg, seed=seed)
    if method == "no_irs_power":
        return evaluate_no_irs_baseline(sys_cfg, rl_cfg, run_cfg, seed=seed)
    raise ValueError(f"Unknown method: {method}")


def run_parameter_sweep(
    parameter: str,
    values: list[float] | list[int],
    sys_cfg: SystemConfig,
    rl_cfg: RLConfig,
    run_cfg: TrainEvalConfig,
) -> dict[str, dict[str, list[float]]]:
    methods = ["fuzzy_wolf_phc", "fast_q_learning", "baseline_ao", "no_irs_power"]
    out = {
        "x": [float(v) for v in values],
        "methods": {m: {"rate": [], "protection": []} for m in methods},
    }

    for value in values:
        if parameter == "pmax_dbm":
            cfg = replace(sys_cfg, pmax_dbm=float(value))
        elif parameter == "m_ris_elements":
            cfg = replace(sys_cfg, m_ris_elements=int(value))
        elif parameter == "sinr_min_db":
            cfg = replace(sys_cfg, sinr_min_db=float(value))
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")

        method_rates = {m: [] for m in methods}
        method_prots = {m: [] for m in methods}

        for run_idx in range(run_cfg.n_seeds):
            seed = cfg.seed + 211 * run_idx
            for method in methods:
                rate, prot = _evaluate_method_for_value(method, cfg, rl_cfg, run_cfg, seed)
                method_rates[method].append(rate)
                method_prots[method].append(prot)

        for method in methods:
            out["methods"][method]["rate"].append(float(np.mean(method_rates[method])))
            out["methods"][method]["protection"].append(float(np.mean(method_prots[method])))

    return out
