from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SystemConfig
from .utils import complex_normal, db_to_linear


def _distance(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a_xy - b_xy, axis=-1)


def pathloss_gain_linear(distance_m: np.ndarray, beta: float, pl0_db: float, d0_m: float) -> np.ndarray:
    """Convert pathloss model (Eq.24) to linear channel gain.

    Paper Eq.24: PL = PL0 - 10*beta*log10(d/d0) gives channel gain in dB.
    At d=d0 the gain is PL0 dB; gain decreases as d grows.
    """
    d = np.maximum(distance_m, d0_m)
    gain_db = pl0_db - 10.0 * beta * np.log10(d / d0_m)
    return db_to_linear(gain_db)


def _angle_of_departure(from_xy: np.ndarray, to_xy: np.ndarray) -> float:
    vec = np.asarray(to_xy, dtype=float) - np.asarray(from_xy, dtype=float)
    return float(np.arctan2(vec[1], vec[0]))


def _ula_response(n_elements: int, angle_rad: float, wavelength_m: float, spacing_factor: float = 0.5) -> np.ndarray:
    if n_elements <= 0:
        return np.zeros(0, dtype=np.complex128)
    d = spacing_factor * wavelength_m
    n = np.arange(n_elements, dtype=float)
    phase = 2.0 * np.pi * d * np.sin(angle_rad) * n / max(1e-12, wavelength_m)
    return np.exp(1j * phase) / np.sqrt(n_elements)


def _rician_mix(gain_linear: float, los: np.ndarray, nlos: np.ndarray, k_db: float) -> np.ndarray:
    k_lin = float(db_to_linear(k_db))
    los_scale = np.sqrt(k_lin / (k_lin + 1.0))
    nlos_scale = np.sqrt(1.0 / (k_lin + 1.0))
    return np.sqrt(gain_linear) * (los_scale * los + nlos_scale * nlos)


@dataclass(slots=True)
class ChannelSnapshot:
    G: np.ndarray
    g_bu: np.ndarray
    g_ru: np.ndarray
    h_ju: np.ndarray


class Topology:
    def __init__(self, cfg: SystemConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.bs_xy = np.asarray(cfg.bs_position_xy, dtype=float)
        self.irs_xy = np.asarray(cfg.irs_position_xy, dtype=float)
        self.ue_xy = self._sample_points(cfg.ue_region_xyxy, cfg.k_users)
        self.jammer_xy = self._sample_points(cfg.jammer_region_xyxy, 1)[0]

    def _sample_points(self, region_xyxy: tuple[float, float, float, float], count: int) -> np.ndarray:
        x0, x1, y0, y1 = region_xyxy
        xs = self.rng.uniform(x0, x1, size=count)
        ys = self.rng.uniform(y0, y1, size=count)
        return np.column_stack([xs, ys])

    def resample_users(self) -> None:
        self.ue_xy = self._sample_points(self.cfg.ue_region_xyxy, self.cfg.k_users)

    def move_jammer(self) -> None:
        step = self.rng.normal(0.0, 2.0, size=2)
        x0, x1, y0, y1 = self.cfg.jammer_region_xyxy
        new_xy = self.jammer_xy + step
        new_xy[0] = float(np.clip(new_xy[0], x0, x1))
        new_xy[1] = float(np.clip(new_xy[1], y0, y1))
        self.jammer_xy = new_xy


class ChannelModel:
    def __init__(self, cfg: SystemConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def sample(self, topology: Topology) -> ChannelSnapshot:
        c = self.cfg

        d_br = float(np.linalg.norm(topology.bs_xy - topology.irs_xy))
        gain_br = float(
            pathloss_gain_linear(
                np.asarray([d_br]),
                beta=c.beta_br,
                pl0_db=c.pathloss_pl0_db,
                d0_m=c.pathloss_d0_m,
            )[0]
        )
        if c.enable_geometry_los:
            a_bs = _ula_response(
                c.n_bs_antennas,
                _angle_of_departure(topology.bs_xy, topology.irs_xy),
                wavelength_m=c.carrier_wavelength_m,
            )
            a_ris = _ula_response(
                c.m_ris_elements,
                _angle_of_departure(topology.irs_xy, topology.bs_xy),
                wavelength_m=c.carrier_wavelength_m,
            )
            g_los = np.outer(a_ris, np.conj(a_bs))
            g_nlos = complex_normal((c.m_ris_elements, c.n_bs_antennas), self.rng)
            G = _rician_mix(gain_br, g_los, g_nlos, c.rician_k_br_db)
        else:
            G = np.sqrt(gain_br) * complex_normal((c.m_ris_elements, c.n_bs_antennas), self.rng)

        d_bu = _distance(topology.ue_xy, np.broadcast_to(topology.bs_xy, topology.ue_xy.shape))
        gain_bu = pathloss_gain_linear(
            d_bu,
            beta=c.beta_bu,
            pl0_db=c.pathloss_pl0_db,
            d0_m=c.pathloss_d0_m,
        )
        if c.enable_geometry_los:
            g_bu = np.zeros((c.k_users, c.n_bs_antennas), dtype=np.complex128)
            nlos_bu = complex_normal((c.k_users, c.n_bs_antennas), self.rng)
            for k in range(c.k_users):
                a_bu = _ula_response(
                    c.n_bs_antennas,
                    _angle_of_departure(topology.bs_xy, topology.ue_xy[k]),
                    wavelength_m=c.carrier_wavelength_m,
                )
                g_bu[k] = _rician_mix(float(gain_bu[k]), a_bu, nlos_bu[k], c.rician_k_bu_db)
        else:
            g_bu = np.sqrt(gain_bu)[:, None] * complex_normal((c.k_users, c.n_bs_antennas), self.rng)

        d_ru = _distance(topology.ue_xy, np.broadcast_to(topology.irs_xy, topology.ue_xy.shape))
        gain_ru = pathloss_gain_linear(
            d_ru,
            beta=c.beta_ru,
            pl0_db=c.pathloss_pl0_db,
            d0_m=c.pathloss_d0_m,
        )
        if c.enable_geometry_los:
            g_ru = np.zeros((c.k_users, c.m_ris_elements), dtype=np.complex128)
            nlos_ru = complex_normal((c.k_users, c.m_ris_elements), self.rng)
            for k in range(c.k_users):
                a_ru = _ula_response(
                    c.m_ris_elements,
                    _angle_of_departure(topology.irs_xy, topology.ue_xy[k]),
                    wavelength_m=c.carrier_wavelength_m,
                )
                g_ru[k] = _rician_mix(float(gain_ru[k]), a_ru, nlos_ru[k], c.rician_k_ru_db)
        else:
            g_ru = np.sqrt(gain_ru)[:, None] * complex_normal((c.k_users, c.m_ris_elements), self.rng)

        d_ju = _distance(topology.ue_xy, np.broadcast_to(topology.jammer_xy, topology.ue_xy.shape))
        gain_ju = pathloss_gain_linear(
            d_ju,
            beta=c.beta_ju,
            pl0_db=c.pathloss_pl0_db,
            d0_m=c.pathloss_d0_m,
        )
        h_ju = np.sqrt(gain_ju)[:, None] * complex_normal((c.k_users, c.n_jammer_antennas), self.rng)

        return ChannelSnapshot(G=G, g_bu=g_bu, g_ru=g_ru, h_ju=h_ju)
