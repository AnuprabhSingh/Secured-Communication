"""Wideband THz system model: per-subcarrier SINR, rate, and reward.

Computes the effective channel through SPDP-RIS and hybrid beamforming,
then evaluates SINR across all subcarriers and users.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .thz_config import THzSystemConfig
from .thz_channel_model import THzChannelSnapshot
from .spdp_ris import SPDPResponse, spdp_reflection_vector
from .hybrid_beamforming import (
    design_analog_precoder,
    analog_precoder_at_freq,
    compute_digital_precoder,
)
from ..utils import db_to_linear

EPS = 1e-30


@dataclass(slots=True)
class THzSystemMetrics:
    sinr_linear: np.ndarray         # (K, M_eval) per-user per-subcarrier SINR
    rates_per_user: np.ndarray      # (K,) average rate per user (over subcarriers)
    system_rate: float              # sum_k R_k
    sinr_out: np.ndarray            # (K, M_eval) binary outage
    sinr_protection_level: float    # percentage
    reward: float
    W: np.ndarray                   # (M_eval, N_t, K) hybrid precoders (for diagnostics)


def effective_channel_subcarrier(
    snapshot: THzChannelSnapshot,
    spdp: SPDPResponse,
    m: int,
) -> np.ndarray:
    """Effective BS-to-user channel at subcarrier index m for all users.

    h_eff_{k,m} = h_bu_{k,m} + h_{k,m}^T * diag(phi(f_m)) * G_m

    The direct BS-UE link (h_bu, NLoS Rayleigh) provides per-user spatial
    diversity needed for multi-user interference suppression.  The RIS-
    reflected path provides the dominant signal power.

    Returns:
        H_eff: (K, N_t) effective channel matrix at subcarrier m.
    """
    f_m = snapshot.freqs[m]
    phi_m = spdp_reflection_vector(spdp, f_m)   # (N,)
    G_m = snapshot.G[m]                           # (N, N_t)

    # RIS-reflected path: (K, N) * (N,) -> (K, N), then @ (N, N_t) -> (K, N_t)
    H_ris = (snapshot.h_ru[m] * phi_m[np.newaxis, :]) @ G_m

    # Direct BS-UE path (NLoS)
    H_direct = snapshot.h_bu[m]  # (K, N_t)

    return H_ris + H_direct


def compute_hybrid_precoders(
    cfg: THzSystemConfig,
    snapshot: THzChannelSnapshot,
    spdp: SPDPResponse,
    p_bs_watt: np.ndarray,
    noise_watt: float,
    subcarrier_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute hybrid precoders W_m = F_RF(f_m) @ F_BB[m] for all (or selected) subcarriers.

    Args:
        cfg: system config
        snapshot: channel snapshot
        spdp: SPDP RIS configuration
        p_bs_watt: (K,) per-user power
        noise_watt: noise power
        subcarrier_indices: optional subset of subcarrier indices to evaluate

    Returns:
        W: (M_eval, N_t, K) hybrid precoder per subcarrier
    """
    if subcarrier_indices is None:
        subcarrier_indices = np.arange(cfg.n_subcarriers)

    M_eval = len(subcarrier_indices)
    N_t = cfg.n_bs_antennas
    K = cfg.k_users

    # Design analog precoder (frequency-flat base + TD)
    W_u, td_bs = design_analog_precoder(cfg, snapshot.phi_bs)

    W = np.zeros((M_eval, N_t, K), dtype=np.complex128)

    for idx, m in enumerate(subcarrier_indices):
        f_m = snapshot.freqs[m]

        # Frequency-dependent analog precoder
        F_RF_m = analog_precoder_at_freq(W_u, td_bs, f_m)  # (N_t, N_RF)

        # Effective channel through RIS
        H_eff_m = effective_channel_subcarrier(snapshot, spdp, m)  # (K, N_t)

        # Effective baseband channel
        H_eff_BB_m = H_eff_m @ F_RF_m  # (K, N_RF)

        # Digital precoder (max-SINR in N_RF-dim space)
        F_BB_m = compute_digital_precoder(H_eff_BB_m, p_bs_watt, noise_watt)  # (N_RF, K)

        # Full hybrid precoder
        W_m = F_RF_m @ F_BB_m  # (N_t, K)

        # Normalize each column to unit norm (direction only).
        # Power allocation P_k is applied separately in the SINR formula,
        # matching the narrowband convention: signal = P_k * |h_eff @ w_k|^2.
        for k in range(K):
            col_norm = np.linalg.norm(W_m[:, k])
            if col_norm > EPS:
                W_m[:, k] /= col_norm

        W[idx] = W_m

    return W


def evaluate_thz_system(
    cfg: THzSystemConfig,
    snapshot: THzChannelSnapshot,
    spdp: SPDPResponse,
    p_bs_watt: np.ndarray,
    p_jammer_watt: np.ndarray,
    z_jammer: np.ndarray,
    noise_watt: float,
    sinr_min_db: float,
    lambda1: float,
    lambda2: float,
    pmax_watt: float = 1.0,
    subcarrier_stride: int = 1,
) -> THzSystemMetrics:
    """Full wideband system evaluation.

    Computes per-subcarrier SINR for all users, aggregated rate,
    protection level, and RL reward.

    Args:
        cfg, snapshot, spdp: system setup
        p_bs_watt: (K,) per-user power allocation
        p_jammer_watt: (K,) jammer power per user
        z_jammer: (K, N_j) jammer precoders
        noise_watt: noise power per subcarrier
        sinr_min_db: SINR target
        lambda1, lambda2: reward weights
        pmax_watt: max BS power budget
        subcarrier_stride: evaluate every N-th subcarrier (speed vs accuracy)

    Returns:
        THzSystemMetrics
    """
    M = cfg.n_subcarriers
    K = cfg.k_users

    # Subcarrier indices to evaluate
    sc_indices = np.arange(0, M, subcarrier_stride)
    M_eval = len(sc_indices)

    # Compute hybrid precoders
    W = compute_hybrid_precoders(cfg, snapshot, spdp, p_bs_watt, noise_watt, sc_indices)

    # Pre-compute jammer terms (frequency-flat)
    jammer_terms = np.zeros(K, dtype=float)
    for k in range(K):
        jammer_terms[k] = p_jammer_watt[k] * np.abs(np.vdot(snapshot.h_ju[k], z_jammer[k])) ** 2

    # Per-subcarrier SINR
    sinr = np.zeros((K, M_eval), dtype=float)

    for idx, m in enumerate(sc_indices):
        H_eff_m = effective_channel_subcarrier(snapshot, spdp, m)  # (K, N_t)

        for k in range(K):
            signal = p_bs_watt[k] * np.abs(H_eff_m[k] @ W[idx, :, k]) ** 2
            iui = 0.0
            for i in range(K):
                if i == k:
                    continue
                iui += p_bs_watt[i] * np.abs(H_eff_m[k] @ W[idx, :, i]) ** 2
            sinr[k, idx] = signal / (iui + jammer_terms[k] + noise_watt + EPS)

    # Rate per user: averaged over evaluated subcarriers
    rates_per_user = np.mean(np.log2(1.0 + sinr), axis=1)  # (K,)
    system_rate = float(np.sum(rates_per_user))

    # Protection level
    sinr_min_lin = float(db_to_linear(sinr_min_db))
    sinr_out = (sinr < sinr_min_lin).astype(float)
    protection = 100.0 * float(1.0 - np.mean(sinr_out))

    # Reward: continuous SINR-deficit penalty (extends paper Eq. 7 for RL).
    # Binary outage creates a FLAT reward landscape when all users are in
    # outage — the agent cannot distinguish "almost meeting target" from
    # "far below target".  Continuous deficit gives smooth gradient:
    #   qos_penalty = mean_k[ max(0, SINR_target - avg_SINR_k) ] / |SINR_target|
    # This is 0 when all users meet the target, and scales proportionally
    # to how far below target each user falls.
    sinr_db = 10.0 * np.log10(np.maximum(sinr, EPS))
    avg_sinr_per_user_db = np.mean(sinr_db, axis=1)  # (K,)
    sinr_deficit = np.clip(sinr_min_db - avg_sinr_per_user_db, 0.0, 30.0)
    qos_penalty = float(np.mean(sinr_deficit) / max(abs(sinr_min_db), 1.0))
    power_frac = float(np.sum(p_bs_watt)) / max(pmax_watt, EPS)
    reward = system_rate - lambda1 * power_frac - lambda2 * qos_penalty

    return THzSystemMetrics(
        sinr_linear=sinr,
        rates_per_user=rates_per_user,
        system_rate=system_rate,
        sinr_out=sinr_out,
        sinr_protection_level=protection,
        reward=reward,
        W=W,
    )


def thz_channel_quality(
    cfg: THzSystemConfig,
    snapshot: THzChannelSnapshot,
    spdp: SPDPResponse,
    subcarrier_stride: int = 8,
) -> np.ndarray:
    """Average effective channel power across subcarriers for each user.

    Used for state feature computation and power allocation modes.

    Returns:
        (K,) channel quality per user.
    """
    M = cfg.n_subcarriers
    K = cfg.k_users
    sc_indices = np.arange(0, M, subcarrier_stride)
    quality = np.zeros(K, dtype=float)

    for m in sc_indices:
        H_eff_m = effective_channel_subcarrier(snapshot, spdp, m)  # (K, N_t)
        quality += np.sum(np.abs(H_eff_m) ** 2, axis=1)

    return quality / len(sc_indices)


def compute_normalized_array_gain(
    cfg: THzSystemConfig,
    snapshot: THzChannelSnapshot,
    spdp: SPDPResponse,
    user_idx: int = 0,
) -> np.ndarray:
    """Normalized array gain eta(f_m) per subcarrier for beam-squint analysis.

    Computes |h_{k,m}^T @ diag(phi(f_m)) @ G_m @ w|^2 normalized by the
    peak gain to show beam-squint effects.

    Returns:
        (M,) normalized array gain values.
    """
    M = cfg.n_subcarriers
    gains = np.zeros(M, dtype=float)

    for m in range(M):
        H_eff_m = effective_channel_subcarrier(snapshot, spdp, m)
        gains[m] = float(np.sum(np.abs(H_eff_m[user_idx]) ** 2))

    peak = np.max(gains)
    if peak > EPS:
        gains /= peak

    return gains


# ---------------------------------------------------------------------------
# Diagnostic functions for physics validation
# ---------------------------------------------------------------------------

def verify_ris_scaling(
    cfg: THzSystemConfig,
    snapshot: THzChannelSnapshot,
    spdp: SPDPResponse,
    user_idx: int = 0,
) -> dict:
    """Verify that RIS provides proper N^2 array gain scaling.
    
    For a cascaded RIS channel (BS→RIS→UE), the ideal coherent combining
    should give |h_ris|^2 ∝ N^2 where N is the number of RIS elements.
    
    Note: This function separates the RIS path from the direct BS-UE path
    for accurate scaling verification, since the direct path power is
    constant and can obscure RIS scaling at small N.
    
    Returns:
        Dict with diagnostic metrics including RIS path and total path gains.
    """
    N = cfg.n_ris_total
    M = cfg.n_subcarriers
    center_idx = M // 2
    
    f_m = snapshot.freqs[center_idx]
    phi_m = spdp_reflection_vector(spdp, f_m)
    G_m = snapshot.G[center_idx]
    h_ru_m = snapshot.h_ru[center_idx, user_idx, :]
    h_bu_m = snapshot.h_bu[center_idx, user_idx, :]
    
    # RIS path only (for clean N² scaling verification)
    H_ris = (h_ru_m * phi_m) @ G_m
    ris_power = float(np.sum(np.abs(H_ris) ** 2))
    
    # Direct path only
    direct_power = float(np.sum(np.abs(h_bu_m) ** 2))
    
    # Total effective channel (RIS + direct)
    H_eff = H_ris + h_bu_m
    total_power = float(np.sum(np.abs(H_eff) ** 2))
    
    # Individual component magnitudes for debugging
    g_norm = float(np.linalg.norm(G_m, 'fro') ** 2)
    h_ru_norm = float(np.linalg.norm(h_ru_m) ** 2)
    
    return {
        "n_ris_elements": N,
        "ris_channel_power": ris_power,
        "direct_channel_power": direct_power,
        "effective_channel_power": total_power,  # Total for backward compatibility
        "ris_to_direct_ratio_db": float(10 * np.log10(ris_power / (direct_power + EPS))),
        "bs_ris_channel_power": g_norm,
        "ris_ue_channel_power": h_ru_norm,
        "expected_scaling_factor": N ** 2,
        "log10_ris_power": float(np.log10(ris_power + EPS)),
        "log10_eff_power": float(np.log10(total_power + EPS)),
    }


def analyze_beam_squint(
    cfg: THzSystemConfig,
    snapshot: THzChannelSnapshot,
    spdp: SPDPResponse,
    user_idx: int = 0,
) -> dict:
    """Analyze beam squint effects across bandwidth.
    
    Computes array gain at center vs edge subcarriers to quantify
    the beam squint degradation.
    
    Returns:
        Dict with edge/center gain ratios and bandwidth-dependent metrics.
    """
    M = cfg.n_subcarriers
    gains = compute_normalized_array_gain(cfg, snapshot, spdp, user_idx)
    
    center_idx = M // 2
    edge_low_idx = 0
    edge_high_idx = M - 1
    
    center_gain = gains[center_idx]
    edge_low_gain = gains[edge_low_idx]
    edge_high_gain = gains[edge_high_idx]
    
    # 3dB bandwidth: fraction of subcarriers with gain > 0.5
    above_3db = np.sum(gains > 0.5) / M
    
    # Effective bandwidth (integral of normalized gain)
    effective_bw_ratio = float(np.mean(gains))
    
    return {
        "center_gain": float(center_gain),
        "edge_low_gain": float(edge_low_gain),
        "edge_high_gain": float(edge_high_gain),
        "edge_to_center_ratio_low": float(edge_low_gain / max(center_gain, EPS)),
        "edge_to_center_ratio_high": float(edge_high_gain / max(center_gain, EPS)),
        "fraction_above_3db": float(above_3db),
        "effective_bandwidth_ratio": effective_bw_ratio,
        "bandwidth_hz": cfg.bandwidth_hz,
        "fractional_bandwidth": cfg.bandwidth_hz / cfg.center_freq_hz,
    }


def compute_wideband_capacity(
    cfg: THzSystemConfig,
    sinr_per_subcarrier: np.ndarray,
) -> dict:
    """Compute proper wideband capacity metrics.
    
    Args:
        cfg: system config
        sinr_per_subcarrier: (K, M) SINR values per user per subcarrier
    
    Returns:
        Dict with various capacity metrics.
    """
    K, M_eval = sinr_per_subcarrier.shape
    B = cfg.bandwidth_hz
    
    # Spectral efficiency per subcarrier (bps/Hz)
    se_per_sc = np.log2(1.0 + sinr_per_subcarrier)  # (K, M_eval)
    
    # Average spectral efficiency per user
    avg_se_per_user = np.mean(se_per_sc, axis=1)  # (K,)
    
    # Total capacity = B * average_SE (since we assume uniform power per subcarrier)
    # For OFDM: C = (B/M) * sum_m log2(1 + SINR_m) = B * mean(log2(1 + SINR))
    capacity_per_user = B * avg_se_per_user  # bps
    total_capacity = float(np.sum(capacity_per_user))
    
    # Capacity per Hz (normalized by bandwidth)
    capacity_per_hz = total_capacity / B
    
    return {
        "total_capacity_bps": total_capacity,
        "total_capacity_gbps": total_capacity / 1e9,
        "capacity_per_user_gbps": (capacity_per_user / 1e9).tolist(),
        "avg_spectral_efficiency_bps_hz": float(np.mean(avg_se_per_user)),
        "sum_spectral_efficiency_bps_hz": float(np.sum(avg_se_per_user)),
    }
