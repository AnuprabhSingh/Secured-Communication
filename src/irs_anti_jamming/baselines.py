from __future__ import annotations

import numpy as np

from .action_space import optimize_irs_phases
from .environment import IRSAntiJammingEnv
from .system_model import SystemMetrics, evaluate_system
from .utils import dbm_to_watt, project_to_simplex


class AOGreedyBaseline:
    """Baseline 1 [39]: AO-style optimizer using estimated jammer state.

    Per the paper, this method:
    - Uses estimated (previous) jammer state only (model mismatch)
    - Maximizes rate greedily (no SINR penalty in its objective)
    - Uses alternating optimization: fix beamforming → optimize power;
      fix power → optimize IRS phases. We model this as:
      * Full Pmax power
      * Channel-proportional power allocation (matches AO's waterfilling)
      * Full AO phase optimization (10 iterations — more than RL's 5)
    """

    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def select_action(self, env: IRSAntiJammingEnv) -> int:
        """Pick an action via AO heuristic.

        AO uses full power with channel-proportional allocation.
        The IRS phases are automatically optimized by the HybridActionSpace.
        """
        n_fracs = len(env.action_space.total_power_fractions)
        # Full Pmax
        frac_idx = n_fracs - 1
        # channel_proportional
        power_idx = 1

        idx_map = {tuple(comp): idx for idx, comp in enumerate(env.action_space.actions)}
        return idx_map[(frac_idx, power_idx)]


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
