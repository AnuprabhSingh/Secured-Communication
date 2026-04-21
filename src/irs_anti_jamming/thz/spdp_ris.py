"""Sub-connected Phase-Delay-Phase (SPDP) RIS architecture.

Implements the wideband RIS design from:
  Su et al., "Wideband Precoding for RIS-Aided THz Communications",
  IEEE Trans. Commun., 2023.  (Lemma 1, Eqs. 24-26)

The RIS has N = N1*N2 elements arranged as a UPA, divided into
Q = Q1*Q2 sub-arrays.  Each element has two layers of phase shifters;
each sub-array shares one time-delay (TD) module.

Signal flow per element:
  incoming -> Layer-1 phase -> sub-array TD -> Layer-2 phase -> re-radiated
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .thz_config import THzSystemConfig


@dataclass(slots=True)
class SPDPResponse:
    """SPDP configuration for all RIS elements."""
    theta_1: np.ndarray        # (N,) first-layer phases (complex, unit modulus)
    theta_2: np.ndarray        # (N,) second-layer phases
    td_delays: np.ndarray      # (Q,) time delays per sub-array (seconds)
    q_map: np.ndarray          # (N,) sub-array index for each element


def _build_index_maps(cfg: THzSystemConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute mappings from global element index to sub-array and local indices.

    Returns:
        q_map: (N,)  sub-array flat index for each element
        k1_map: (N,) local horizontal index within sub-array
        k2_map: (N,) local vertical index within sub-array
    """
    N1, N2 = cfg.n_ris_h, cfg.n_ris_v
    K1, K2 = cfg.k1, cfg.k2
    Q2 = cfg.q_subarrays_v
    N = N1 * N2

    q_map = np.empty(N, dtype=np.int32)
    k1_map = np.empty(N, dtype=np.int32)
    k2_map = np.empty(N, dtype=np.int32)

    for n1 in range(N1):
        for n2 in range(N2):
            n = n1 * N2 + n2
            q1 = n1 // K1
            q2 = n2 // K2
            q_map[n] = q1 * Q2 + q2
            k1_map[n] = n1 % K1
            k2_map[n] = n2 % K2

    return q_map, k1_map, k2_map


def compute_spdp_closed_form(
    cfg: THzSystemConfig,
    theta_aoa: float,
    zeta_aoa: float,
    theta_aod: float,
    zeta_aod: float,
) -> SPDPResponse:
    """Closed-form SPDP design for a single reference direction (Lemma 1).

    Args:
        cfg: system configuration
        theta_aoa, zeta_aoa: RIS AoA elevation and azimuth (from BS)
        theta_aod, zeta_aod: RIS AoD elevation and azimuth (toward UE)

    Returns:
        SPDPResponse with phase-shifter values and TD delays.
    """
    N1, N2 = cfg.n_ris_h, cfg.n_ris_v
    Q1, Q2 = cfg.q_subarrays_h, cfg.q_subarrays_v
    K1, K2 = cfg.k1, cfg.k2
    f_c = cfg.center_freq_hz
    Q = Q1 * Q2
    N = N1 * N2

    # Direction cosines
    alpha1 = math.sin(theta_aoa) * math.cos(zeta_aoa)
    beta1 = math.sin(theta_aoa) * math.sin(zeta_aoa)
    alpha2 = math.sin(theta_aod) * math.cos(zeta_aod)
    beta2 = math.sin(theta_aod) * math.sin(zeta_aod)

    # Time delays per sub-array  (Eq. 24)
    td_delays = np.zeros(Q, dtype=float)
    for q1 in range(Q1):
        for q2 in range(Q2):
            q = q1 * Q2 + q2
            td_delays[q] = (1.0 / (2.0 * f_c)) * (
                (q1 * K1 - (K1 - 1) / 2.0) * (alpha1 + alpha2)
                + (q2 * K2 - (K2 - 1) / 2.0) * (beta1 + beta2)
            )

    # Phase shifts per element
    q_map, k1_map, k2_map = _build_index_maps(cfg)

    # Layer 1: align to AoA  (Eq. 25)
    phase1 = -math.pi * (k1_map.astype(float) * alpha1 + k2_map.astype(float) * beta1)
    theta_1 = np.exp(1j * phase1)

    # Layer 2: steer toward AoD  (Eq. 26)
    phase2 = -math.pi * (k1_map.astype(float) * alpha2 + k2_map.astype(float) * beta2)
    theta_2 = np.exp(1j * phase2)

    # Optional low-resolution quantization
    if cfg.phase_bits > 0:
        n_levels = 2 ** cfg.phase_bits
        grid = np.linspace(0, 2 * math.pi, n_levels, endpoint=False)
        for arr in (theta_1, theta_2):
            angles = np.angle(arr) % (2 * math.pi)
            idx = np.argmin(np.abs(angles[:, None] - grid[None, :]), axis=1)
            arr[:] = np.exp(1j * grid[idx])

    return SPDPResponse(theta_1=theta_1, theta_2=theta_2,
                        td_delays=td_delays, q_map=q_map)


def spdp_reflection_vector(spdp: SPDPResponse, f_m: float) -> np.ndarray:
    """Compute the frequency-dependent RIS reflection vector at subcarrier f_m.

    phi_n(f_m) = Theta2(n) * exp(-j 2pi f_m t_{q(n)}) * Theta1(n)

    Args:
        spdp: SPDP configuration
        f_m: subcarrier frequency (Hz)

    Returns:
        (N,) complex reflection coefficients (unit modulus per element).
    """
    td_per_element = spdp.td_delays[spdp.q_map]  # (N,)
    td_phase = np.exp(-1j * 2.0 * math.pi * f_m * td_per_element)
    return spdp.theta_2 * td_phase * spdp.theta_1


def classical_phase_only(
    cfg: THzSystemConfig,
    theta_aoa: float,
    zeta_aoa: float,
    theta_aod: float,
    zeta_aod: float,
) -> SPDPResponse:
    """Classical narrowband (frequency-independent) phase-only RIS.

    Uses the optimal phases at center frequency f_c only (no TD modules).
    This is the baseline that suffers from beam squint at wideband THz.
    """
    N1, N2 = cfg.n_ris_h, cfg.n_ris_v
    N = N1 * N2

    alpha1 = math.sin(theta_aoa) * math.cos(zeta_aoa)
    beta1 = math.sin(theta_aoa) * math.sin(zeta_aoa)
    alpha2 = math.sin(theta_aod) * math.cos(zeta_aod)
    beta2 = math.sin(theta_aod) * math.sin(zeta_aod)

    q_map, k1_map, k2_map = _build_index_maps(cfg)

    # Classical: combine both layers into a single frequency-flat phase
    # (equivalent to SPDP with td=0)
    n1_global = np.arange(N1, dtype=float)
    n2_global = np.arange(N2, dtype=float)
    n1_grid, n2_grid = np.meshgrid(n1_global, n2_global, indexing="ij")
    n1_flat = n1_grid.ravel()
    n2_flat = n2_grid.ravel()

    phase_combined = -math.pi * (
        n1_flat * (alpha1 + alpha2) + n2_flat * (beta1 + beta2)
    )
    phi = np.exp(1j * phase_combined)

    td_zeros = np.zeros(cfg.n_subarrays_total, dtype=float)

    return SPDPResponse(
        theta_1=phi,
        theta_2=np.ones(N, dtype=complex),
        td_delays=td_zeros,
        q_map=q_map,
    )


def optimize_spdp_multiuser(
    cfg: THzSystemConfig,
    theta_aoa: float,
    zeta_aoa: float,
    theta_aod_arr: np.ndarray,
    zeta_aod_arr: np.ndarray,
    rate_eval_fn=None,
    p_bs_watt: np.ndarray | None = None,
    noise_watt: float = 0.0,
    snapshot=None,
) -> SPDPResponse:
    """Multi-user SPDP: evaluate K+1 candidates, pick the best.

    Candidates: centroid AoD + each user's individual AoD.
    If rate_eval_fn is provided, uses it to score each candidate;
    otherwise returns the centroid-based SPDP.

    Args:
        cfg: system config
        theta_aoa, zeta_aoa: RIS AoA from BS
        theta_aod_arr: (K,) per-user AoD elevation
        zeta_aod_arr: (K,) per-user AoD azimuth
        rate_eval_fn: optional callable(spdp) -> float for scoring
        p_bs_watt, noise_watt, snapshot: passed through to rate_eval_fn

    Returns:
        Best SPDPResponse among candidates.
    """
    K = theta_aod_arr.shape[0]

    # Build candidate directions: centroid + per-user
    candidates = [
        (float(np.mean(theta_aod_arr)), float(np.mean(zeta_aod_arr))),
    ]
    for k in range(K):
        candidates.append((float(theta_aod_arr[k]), float(zeta_aod_arr[k])))

    if rate_eval_fn is None:
        # No scoring function: return centroid
        return compute_spdp_closed_form(cfg, theta_aoa, zeta_aoa,
                                        candidates[0][0], candidates[0][1])

    best_rate = -np.inf
    best_spdp = None
    for theta_ref, zeta_ref in candidates:
        spdp = compute_spdp_closed_form(cfg, theta_aoa, zeta_aoa, theta_ref, zeta_ref)
        rate = rate_eval_fn(spdp)
        if rate > best_rate:
            best_rate = rate
            best_spdp = spdp

    return best_spdp
