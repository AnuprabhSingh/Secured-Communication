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
    use_irs: bool = True,
) -> SystemMetrics:
    phi = build_phi_vector(theta)
    h_eff = effective_channels(snapshot, phi, use_irs=use_irs)
    beamformers = compute_mrt_beamformers(h_eff)

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

    reward = system_rate - lambda1 * float(np.sum(p_bs_watt)) - lambda2 * float(np.sum(sinr_out))
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
