from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel_model import ChannelSnapshot
from .system_model import build_phi_vector, compute_maxsinr_beamformers, effective_channels
from .utils import db_to_linear, linear_to_db, normalize, project_to_simplex


@dataclass(slots=True)
class ActionContext:
    snapshot: ChannelSnapshot
    pmax_watt: float
    sinr_min_db: float
    prev_sinr_linear: np.ndarray
    channel_quality_linear: np.ndarray
    noise_watt: float


def optimize_irs_phases(
    snapshot: ChannelSnapshot,
    p_bs_watt: np.ndarray,
    m_ris_elements: int,
    n_ao_iters: int = 6,
    noise_watt: float = 3.16e-14,
    sinr_min_linear: float = 10.0,
) -> np.ndarray:
    """IRS phase optimization via alternating optimization with max-SINR BF.

    Uses two strategies and picks the best:
    1. Standard sum-power weighted AO
    2. SINR-deficit weighted AO (protects weak users)
    """
    if m_ris_elements <= 0:
        return np.zeros(0, dtype=float)

    k_users = p_bs_watt.shape[0]
    G = snapshot.G
    g_ru = snapshot.g_ru

    best_theta = np.zeros(m_ris_elements, dtype=float)
    best_rate = -np.inf

    # === Strategy 1: Standard sum-rate AO ===
    theta = np.zeros(m_ris_elements, dtype=float)
    for _ in range(n_ao_iters):
        phi = build_phi_vector(theta)
        h_eff = effective_channels(snapshot, phi, use_irs=True)
        w = compute_maxsinr_beamformers(h_eff, p_bs_watt, noise_watt)
        composite = np.zeros(m_ris_elements, dtype=np.complex128)
        for k in range(k_users):
            composite += np.sqrt(p_bs_watt[k]) * np.conj(g_ru[k]) * (G @ w[k])
        theta = np.mod(-np.angle(composite + 1e-15), 2.0 * np.pi)

    # Evaluate
    phi = build_phi_vector(theta)
    h_eff = effective_channels(snapshot, phi, use_irs=True)
    w = compute_maxsinr_beamformers(h_eff, p_bs_watt, noise_watt)
    rate = 0.0
    for k in range(k_users):
        dk = p_bs_watt[k] * abs(h_eff[k] @ w[k]) ** 2
        interf = noise_watt
        for j in range(k_users):
            if j != k:
                interf += p_bs_watt[j] * abs(h_eff[k] @ w[j]) ** 2
        rate += np.log2(1.0 + dk / max(interf, 1e-30))
    if rate > best_rate:
        best_rate = rate
        best_theta = theta.copy()

    # === Strategy 2: SINR-deficit weighted AO ===
    theta = np.zeros(m_ris_elements, dtype=float)
    for _ in range(n_ao_iters):
        phi = build_phi_vector(theta)
        h_eff = effective_channels(snapshot, phi, use_irs=True)
        w = compute_maxsinr_beamformers(h_eff, p_bs_watt, noise_watt)

        # Compute per-user SINR for weighting
        sinrs = np.zeros(k_users)
        for k in range(k_users):
            dk = p_bs_watt[k] * abs(h_eff[k] @ w[k]) ** 2
            interf = noise_watt
            for j in range(k_users):
                if j != k:
                    interf += p_bs_watt[j] * abs(h_eff[k] @ w[j]) ** 2
            sinrs[k] = dk / max(interf, 1e-30)

        # Weight: inverse SINR (help weakest users more)
        inv_sinr = 1.0 / np.maximum(sinrs, 1e-6)
        user_weights = np.sqrt(p_bs_watt) * inv_sinr

        composite = np.zeros(m_ris_elements, dtype=np.complex128)
        for k in range(k_users):
            composite += user_weights[k] * np.conj(g_ru[k]) * (G @ w[k])
        theta = np.mod(-np.angle(composite + 1e-15), 2.0 * np.pi)

    # Evaluate
    phi = build_phi_vector(theta)
    h_eff = effective_channels(snapshot, phi, use_irs=True)
    w = compute_maxsinr_beamformers(h_eff, p_bs_watt, noise_watt)
    rate = 0.0
    for k in range(k_users):
        dk = p_bs_watt[k] * abs(h_eff[k] @ w[k]) ** 2
        interf = noise_watt
        for j in range(k_users):
            if j != k:
                interf += p_bs_watt[j] * abs(h_eff[k] @ w[j]) ** 2
        rate += np.log2(1.0 + dk / max(interf, 1e-30))
    if rate > best_rate:
        best_rate = rate
        best_theta = theta.copy()

    return best_theta


class HybridActionSpace:
    """Hybrid action space: RL chooses power allocation, AO optimizes IRS phases."""

    def __init__(self, k_users: int, m_ris_elements: int, seed: int,
                 n_ao_iters: int = 6, sinr_min_db: float = 10.0):
        self.k_users = k_users
        self.m_ris_elements = m_ris_elements
        self.rng = np.random.default_rng(seed)
        self.n_ao_iters = n_ao_iters
        self.sinr_min_linear = float(db_to_linear(sinr_min_db))

        self.total_power_fractions = [0.3, 0.45, 0.6, 0.75, 0.85, 1.0]
        self.power_modes = [
            "equal",
            "channel_proportional",
            "inverse_channel",
            "sinr_deficit",
            "waterfilling",
        ]

        self.actions: list[tuple[int, int]] = [
            (f, m)
            for f in range(len(self.total_power_fractions))
            for m in range(len(self.power_modes))
        ]

    @property
    def size(self) -> int:
        return len(self.actions)

    def _decode_powers(self, fraction_idx: int, mode_idx: int, ctx: ActionContext) -> np.ndarray:
        total = self.total_power_fractions[fraction_idx] * ctx.pmax_watt
        mode = self.power_modes[mode_idx]

        if mode == "equal":
            weights = np.full(self.k_users, 1.0 / self.k_users)
        elif mode == "channel_proportional":
            weights = project_to_simplex(ctx.channel_quality_linear)
        elif mode == "inverse_channel":
            weights = project_to_simplex(1.0 / np.maximum(ctx.channel_quality_linear, 1e-12))
        elif mode == "waterfilling":
            cq = np.maximum(ctx.channel_quality_linear, 1e-12)
            wf = np.log1p(cq)
            weights = project_to_simplex(wf)
        else:
            deficit = np.maximum(
                ctx.sinr_min_db - linear_to_db(np.maximum(ctx.prev_sinr_linear, 1e-12)),
                0.0,
            )
            weights = project_to_simplex(deficit + 1e-3)

        return total * weights

    def decode(self, action_idx: int, ctx: ActionContext) -> tuple[np.ndarray, np.ndarray]:
        frac_idx, mode_idx = self.actions[action_idx]
        p_bs = self._decode_powers(frac_idx, mode_idx, ctx)

        theta = optimize_irs_phases(
            snapshot=ctx.snapshot,
            p_bs_watt=p_bs,
            m_ris_elements=self.m_ris_elements,
            n_ao_iters=self.n_ao_iters,
            noise_watt=ctx.noise_watt,
            sinr_min_linear=self.sinr_min_linear,
        )
        return p_bs, theta

    def power_candidates_only(self, ctx: ActionContext) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for frac_idx in range(len(self.total_power_fractions)):
            for mode_idx in range(len(self.power_modes)):
                out.append(self._decode_powers(frac_idx, mode_idx, ctx))
        return out
