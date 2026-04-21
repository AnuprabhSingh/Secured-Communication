from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel_model import ChannelSnapshot
from .utils import db_to_linear, normalize


@dataclass(slots=True)
class SystemMetrics:
    sinr_linear: np.ndarray
    rates: np.ndarray
    system_rate: float
    sinr_out: np.ndarray
    sinr_protection_level: float
    reward: float


def build_phi_vector(theta: np.ndarray) -> np.ndarray:
    return np.exp(1j * theta)


def effective_channels(snapshot: ChannelSnapshot, phi: np.ndarray, use_irs: bool) -> np.ndarray:
    if use_irs and phi.size > 0:
        reflective_part = (np.conj(snapshot.g_ru) * phi[None, :]) @ snapshot.G
        return reflective_part + np.conj(snapshot.g_bu)
    return np.conj(snapshot.g_bu)


def compute_mrt_beamformers(h_eff: np.ndarray) -> np.ndarray:
    w = np.zeros_like(h_eff, dtype=np.complex128)
    for k in range(h_eff.shape[0]):
        w[k] = normalize(np.conj(h_eff[k]))
    return w


def compute_maxsinr_beamformers(
    h_eff: np.ndarray,
    p_bs_watt: np.ndarray,
    noise_watt: float,
    h_ju: np.ndarray | None = None,
    p_jammer_watt: np.ndarray | None = None,
    z_jammer: np.ndarray | None = None,
) -> np.ndarray:
    """Max-SINR (MVDR) beamforming per paper reference [17].

    For each user k, the beamformer maximizes SINR_k by:
      w_k = R_k^{-1} h_k / ||R_k^{-1} h_k||
    where R_k = sum_{i!=k} P_i h_eff_k h_eff_k^H  +  jammer_cov  +  noise*I
    is the interference-plus-noise covariance at user k's receiver.

    Since the BS transmits to ALL users, R_k represents
    the interference seen at the BS when user k is the desired user.
    w_k steers the beam toward h_eff_k while nulling toward interferers.
    """
    k_users, n_antennas = h_eff.shape
    w = np.zeros_like(h_eff, dtype=np.complex128)

    for k in range(k_users):
        # Interference covariance (from other users' effective channels)
        R_k = noise_watt * np.eye(n_antennas, dtype=np.complex128)
        for i in range(k_users):
            if i == k:
                continue
            hi = np.conj(h_eff[i])  # conjugate since h_eff contains g^*
            R_k += p_bs_watt[i] * np.outer(hi, hi.conj())

        # Solve for optimal beamformer
        hk = np.conj(h_eff[k])
        try:
            R_inv_h = np.linalg.solve(R_k, hk)
        except np.linalg.LinAlgError:
            R_inv_h = np.linalg.lstsq(R_k, hk, rcond=None)[0]

        w[k] = normalize(R_inv_h)

    return w


def evaluate_system(
    snapshot: ChannelSnapshot,
    p_bs_watt: np.ndarray,
    theta: np.ndarray,
    p_jammer_watt: np.ndarray,
    z_jammer: np.ndarray,
    noise_watt: float,
    sinr_min_db: float,
    lambda1: float,
    lambda2: float,
    pmax_watt: float = 1.0,
    use_irs: bool = True,
) -> SystemMetrics:
    phi = build_phi_vector(theta)
    h_eff = effective_channels(snapshot, phi, use_irs=use_irs)
    beamformers = compute_maxsinr_beamformers(h_eff, p_bs_watt, noise_watt)

    k_users = h_eff.shape[0]
    sinr = np.zeros(k_users, dtype=float)

    for k in range(k_users):
        signal = p_bs_watt[k] * np.abs(h_eff[k] @ beamformers[k]) ** 2
        interference = 0.0
        for i in range(k_users):
            if i == k:
                continue
            interference += p_bs_watt[i] * np.abs(h_eff[k] @ beamformers[i]) ** 2

        jammer_term = p_jammer_watt[k] * np.abs(np.vdot(snapshot.h_ju[k], z_jammer[k])) ** 2
        sinr[k] = signal / (interference + jammer_term + noise_watt + 1e-12)

    rates = np.log2(1.0 + sinr)
    system_rate = float(np.sum(rates))

    sinr_min_linear = float(db_to_linear(sinr_min_db))
    sinr_out = (sinr < sinr_min_linear).astype(float)
    protection = float(100.0 * np.mean(1.0 - sinr_out))

    qos_violations = float(np.sum(sinr_out))
    # Power penalty per paper Eq. 7: lambda1 * sum(P_k)
    # Normalize by Pmax so penalty is in [0, K] range (comparable to rate)
    power_fraction = float(np.sum(p_bs_watt)) / max(pmax_watt, 1e-12)
    reward = system_rate - lambda1 * power_fraction - lambda2 * qos_violations

    return SystemMetrics(
        sinr_linear=sinr,
        rates=rates,
        system_rate=system_rate,
        sinr_out=sinr_out,
        sinr_protection_level=protection,
        reward=reward,
    )


def channel_quality(snapshot: ChannelSnapshot) -> np.ndarray:
    direct = np.sum(np.abs(snapshot.g_bu) ** 2, axis=1)
    cascaded = np.sum(np.abs(snapshot.g_ru @ snapshot.G) ** 2, axis=1)
    return direct + cascaded
