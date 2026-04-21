"""Hybrid beamforming for wideband THz BS with sub-connected architecture.

The BS has N_t antennas and N_RF RF chains organized in P = N_RF sub-arrays.
Each sub-array has N_t/P antennas and one time-delay (TD) module.

Architecture (Su et al. Sec IV-C, Eqs. 37-39):
  W_m = F_RF(f_m) @ F_BB[m]

  F_RF(f_m) = W_u * diag(exp(-j 2pi f_m t^BS))    (N_t, N_RF)
  F_BB[m]   = max-SINR digital precoder             (N_RF, K)

W_u is block-diagonal with per-sub-array steering vector partitions.
"""
from __future__ import annotations

import math

import numpy as np

from .thz_config import SPEED_OF_LIGHT, THzSystemConfig
from ..utils import normalize


def design_analog_precoder(
    cfg: THzSystemConfig,
    phi_bs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Design the sub-connected analog precoder W_u and BS TD delays.

    W_u = blkdiag(w_{u,1}, ..., w_{u,P})  shape (N_t, P)
    Each w_{u,p} is the partition of the BS steering vector at f_c.

    TD delays t_p^BS provide frequency-dependent beam-squint compensation
    across the P sub-arrays (Eq. 38).

    Returns:
        W_u: (N_t, P) block-diagonal analog precoder (phase-only)
        td_bs: (P,) time delays per BS sub-array
    """
    N_t = cfg.n_bs_antennas
    P = cfg.n_rf_chains
    N_per = cfg.n_bs_per_subarray  # N_t / P
    f_c = cfg.center_freq_hz
    d = cfg.antenna_spacing

    # Full BS steering vector at center frequency toward RIS (unnormalized)
    n_arr = np.arange(N_t, dtype=float)
    phase_full = 2.0 * math.pi * d * math.sin(phi_bs) / SPEED_OF_LIGHT * f_c * n_arr
    steering_full = np.exp(1j * phase_full)

    # Block-diagonal: partition into P sub-arrays
    W_u = np.zeros((N_t, P), dtype=np.complex128)
    for p in range(P):
        start = p * N_per
        end = start + N_per
        W_u[start:end, p] = steering_full[start:end]

    # Project to unit-modulus (phase-only constraint)
    mask = np.abs(W_u) > 1e-12
    W_u[mask] = np.exp(1j * np.angle(W_u[mask]))
    W_u[~mask] = 0.0
    # Normalize columns
    for p in range(P):
        col_norm = np.linalg.norm(W_u[:, p])
        if col_norm > 1e-12:
            W_u[:, p] /= col_norm

    # TD delays for BS beam-squint compensation (Eq. 38)
    sin_phi = math.sin(phi_bs)
    td_bs = np.zeros(P, dtype=float)
    for p in range(P):
        # Center antenna index of sub-array p
        center_idx = p * N_per + (N_per - 1) / 2.0
        td_bs[p] = d * sin_phi * center_idx / SPEED_OF_LIGHT

    return W_u, td_bs


def analog_precoder_at_freq(
    W_u: np.ndarray,
    td_bs: np.ndarray,
    f_m: float,
) -> np.ndarray:
    """Compute the frequency-dependent analog precoder at subcarrier f_m.

    F_RF(f_m) = W_u * diag(exp(-j 2pi f_m t^BS))

    Returns:
        (N_t, N_RF) frequency-dependent analog precoder.
    """
    td_phase = np.exp(-1j * 2.0 * math.pi * f_m * td_bs)  # (P,)
    return W_u * td_phase[np.newaxis, :]  # (N_t, P) broadcast


def compute_digital_precoder(
    H_eff_BB: np.ndarray,
    p_bs_watt: np.ndarray,
    noise_watt: float,
) -> np.ndarray:
    """Max-SINR digital precoder in the reduced N_RF-dimensional baseband.

    For each user k:
      R_k = sigma^2 I + sum_{i!=k} P_i h_i^H h_i
      F_BB[:,k] = R_k^{-1} h_k / ||R_k^{-1} h_k||

    Args:
        H_eff_BB: (K, N_RF) effective baseband channel (after analog precoding)
        p_bs_watt: (K,) per-user power allocation
        noise_watt: noise power

    Returns:
        F_BB: (N_RF, K) digital precoder
    """
    K, N_RF = H_eff_BB.shape
    F_BB = np.zeros((N_RF, K), dtype=np.complex128)

    for k in range(K):
        R_k = noise_watt * np.eye(N_RF, dtype=np.complex128)
        for i in range(K):
            if i == k:
                continue
            hi = H_eff_BB[i].conj()  # (N_RF,)
            R_k += p_bs_watt[i] * np.outer(hi, hi.conj())

        hk = H_eff_BB[k].conj()
        try:
            R_inv_h = np.linalg.solve(R_k, hk)
        except np.linalg.LinAlgError:
            R_inv_h = np.linalg.lstsq(R_k, hk, rcond=None)[0]

        F_BB[:, k] = normalize(R_inv_h)

    return F_BB
