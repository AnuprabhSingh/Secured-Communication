from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .channel_model import ChannelSnapshot
from .utils import linear_to_db, normalize, project_to_simplex


@dataclass(slots=True)
class ActionContext:
    snapshot: ChannelSnapshot
    pmax_watt: float
    sinr_min_db: float
    prev_sinr_linear: np.ndarray
    channel_quality_linear: np.ndarray


class JointActionSpace:
    def __init__(self, k_users: int, m_ris_elements: int, seed: int):
        self.k_users = k_users
        self.m_ris_elements = m_ris_elements
        self.rng = np.random.default_rng(seed)

        self.total_power_fractions = [0.40, 0.55, 0.70, 0.85, 1.0]
        self.power_modes = ["equal", "channel_proportional", "inverse_channel", "sinr_deficit"]
        self.phase_modes = [
            "all_zero",
            "fixed_random_1",
            "fixed_random_2",
            "align_weakest_user",
            "align_weighted_sum",
        ]

        self._fixed_phase_1 = self._random_quantized_phase()
        self._fixed_phase_2 = self._random_quantized_phase()

        self.actions: list[tuple[int, int, int]] = list(
            product(range(len(self.total_power_fractions)), range(len(self.power_modes)), range(len(self.phase_modes)))
        )

    @property
    def size(self) -> int:
        return len(self.actions)

    def _random_quantized_phase(self, levels: int = 8) -> np.ndarray:
        if self.m_ris_elements <= 0:
            return np.zeros(0, dtype=float)
        bins = self.rng.integers(0, levels, size=self.m_ris_elements)
        return 2.0 * np.pi * bins / levels

    def _direct_mrt_beamformers(self, snapshot: ChannelSnapshot) -> np.ndarray:
        w = np.zeros((self.k_users, snapshot.g_bu.shape[1]), dtype=np.complex128)
        for k in range(self.k_users):
            w[k] = normalize(snapshot.g_bu[k])
        return w

    def _decode_powers(self, fraction_idx: int, mode_idx: int, ctx: ActionContext) -> np.ndarray:
        total = self.total_power_fractions[fraction_idx] * ctx.pmax_watt
        mode = self.power_modes[mode_idx]

        if mode == "equal":
            weights = np.full(self.k_users, 1.0 / self.k_users)
        elif mode == "channel_proportional":
            weights = project_to_simplex(ctx.channel_quality_linear)
        elif mode == "inverse_channel":
            weights = project_to_simplex(1.0 / np.maximum(ctx.channel_quality_linear, 1e-12))
        else:
            deficit = np.maximum(ctx.sinr_min_db - linear_to_db(np.maximum(ctx.prev_sinr_linear, 1e-12)), 0.0)
            weights = project_to_simplex(deficit + 1e-3)

        return total * weights

    def _decode_phases(self, phase_idx: int, p_bs_watt: np.ndarray, ctx: ActionContext) -> np.ndarray:
        if self.m_ris_elements <= 0:
            return np.zeros(0, dtype=float)

        mode = self.phase_modes[phase_idx]
        if mode == "all_zero":
            return np.zeros(self.m_ris_elements, dtype=float)
        if mode == "fixed_random_1":
            return self._fixed_phase_1.copy()
        if mode == "fixed_random_2":
            return self._fixed_phase_2.copy()

        w = self._direct_mrt_beamformers(ctx.snapshot)

        if mode == "align_weakest_user":
            weakest = int(np.argmin(ctx.prev_sinr_linear))
            coeff = np.conj(ctx.snapshot.g_ru[weakest]) * (ctx.snapshot.G @ w[weakest])
            theta = -np.angle(coeff + 1e-12)
            return np.mod(theta, 2.0 * np.pi)

        weighted_coeff = np.zeros(self.m_ris_elements, dtype=np.complex128)
        weights = project_to_simplex(np.maximum(p_bs_watt, 1e-12))
        for k in range(self.k_users):
            weighted_coeff += weights[k] * np.conj(ctx.snapshot.g_ru[k]) * (ctx.snapshot.G @ w[k])
        theta = -np.angle(weighted_coeff + 1e-12)
        return np.mod(theta, 2.0 * np.pi)

    def decode(self, action_idx: int, ctx: ActionContext) -> tuple[np.ndarray, np.ndarray]:
        frac_idx, mode_idx, phase_idx = self.actions[action_idx]
        p_bs = self._decode_powers(frac_idx, mode_idx, ctx)
        theta = self._decode_phases(phase_idx, p_bs, ctx)
        return p_bs, theta

    def power_candidates_only(self, ctx: ActionContext) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for frac_idx in range(len(self.total_power_fractions)):
            for mode_idx in range(len(self.power_modes)):
                out.append(self._decode_powers(frac_idx, mode_idx, ctx))
        return out
