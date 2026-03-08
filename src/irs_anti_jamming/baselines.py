from __future__ import annotations

import numpy as np

from .environment import IRSAntiJammingEnv
from .system_model import SystemMetrics


class AOGreedyBaseline:
    OPTIMIZATION_INTERVAL = 3  # re-optimize only every N slots (realistic computation delay)

    def __init__(self, qos_penalty: float = 0.0, candidate_sample_size: int = 0, seed: int = 123):
        self.qos_penalty = qos_penalty
        self.candidate_sample_size = candidate_sample_size
        self.rng = np.random.default_rng(seed)
        self.prev_action: int | None = None
        self.current_components: tuple[int, int, int] | None = None
        self._slot_count = 0

    def reset(self) -> None:
        self.prev_action = None
        self.current_components = None
        self._slot_count = 0

    def _score(self, env: IRSAntiJammingEnv, action_idx: int) -> float:
        metrics, _, _ = env.evaluate_action_with_jammer_estimate(action_idx, use_irs=True)
        return metrics.system_rate - self.qos_penalty * float(np.sum(metrics.sinr_out))

    def _build_index_map(self, env: IRSAntiJammingEnv) -> dict[tuple[int, int, int], int]:
        return {tuple(comp): idx for idx, comp in enumerate(env.action_space.actions)}

    def _maybe_init_components(self, env: IRSAntiJammingEnv) -> tuple[int, int, int]:
        if self.current_components is not None:
            return self.current_components

        f = int(self.rng.integers(0, len(env.action_space.total_power_fractions)))
        p = int(self.rng.integers(0, len(env.action_space.power_modes)))
        ph = int(self.rng.integers(0, len(env.action_space.phase_modes)))
        self.current_components = (f, p, ph)
        return self.current_components

    def select_action(self, env: IRSAntiJammingEnv) -> int:
        idx_map = self._build_index_map(env)
        frac_idx, power_idx, phase_idx = self._maybe_init_components(env)

        n_fracs = len(env.action_space.total_power_fractions)
        n_powers = len(env.action_space.power_modes)
        n_phases = len(env.action_space.phase_modes)

        # Re-optimize only every OPTIMIZATION_INTERVAL slots
        if self._slot_count > 0 and self._slot_count % self.OPTIMIZATION_INTERVAL == 0:
            update_dim = (self._slot_count // self.OPTIMIZATION_INTERVAL) % 2

            if update_dim == 0:
                # Neighborhood power search (±1 step)
                candidates: set[tuple[int, int]] = set()
                for df in (-1, 0, 1):
                    for dp in (-1, 0, 1):
                        f = max(0, min(n_fracs - 1, frac_idx + df))
                        p = max(0, min(n_powers - 1, power_idx + dp))
                        candidates.add((f, p))

                best_power_pair = (frac_idx, power_idx)
                best_power_score = -np.inf
                for f, p in candidates:
                    a_idx = idx_map[(f, p, phase_idx)]
                    score = self._score(env, a_idx)
                    if score > best_power_score:
                        best_power_score = score
                        best_power_pair = (f, p)
                frac_idx, power_idx = best_power_pair
            else:
                # Neighborhood phase search (±1 step)
                candidates_ph: set[int] = set()
                for dph in (-1, 0, 1):
                    ph = max(0, min(n_phases - 1, phase_idx + dph))
                    candidates_ph.add(ph)

                best_phase = phase_idx
                best_phase_score = -np.inf
                for ph in candidates_ph:
                    a_idx = idx_map[(frac_idx, power_idx, ph)]
                    score = self._score(env, a_idx)
                    if score > best_phase_score:
                        best_phase_score = score
                        best_phase = ph
                phase_idx = best_phase

            self.current_components = (frac_idx, power_idx, phase_idx)

        best_idx = idx_map[self.current_components]
        self.prev_action = best_idx
        self._slot_count += 1
        return best_idx


class NoIRSPowerOnlyBaseline:
    def select_power(self, env: IRSAntiJammingEnv) -> np.ndarray:
        ctx = env.action_context()
        candidates = env.action_space.power_candidates_only(ctx)
        best_power = candidates[0]
        best_rate = -np.inf
        for power in candidates:
            metrics = env.evaluate_power_only_no_irs(power)
            if metrics.system_rate > best_rate:
                best_rate = metrics.system_rate
                best_power = power
        return best_power

    def run_step(self, env: IRSAntiJammingEnv) -> tuple[SystemMetrics, dict[str, float]]:
        p_bs = self.select_power(env)
        metrics = env.evaluate_power_only_no_irs(p_bs)
        env.advance_with_sinr(metrics.sinr_linear, action_signature=-1)
        info = {
            "system_rate": metrics.system_rate,
            "sinr_protection": metrics.sinr_protection_level,
            "reward": metrics.reward,
        }
        return metrics, info
