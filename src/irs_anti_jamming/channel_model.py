from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SystemConfig
from .utils import complex_normal, db_to_linear


def _distance(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a_xy - b_xy, axis=-1)


def pathloss_gain_linear(distance_m: np.ndarray, beta: float, pl0_db: float, d0_m: float) -> np.ndarray:
    d = np.maximum(distance_m, d0_m)
    pl_db = pl0_db - 10.0 * beta * np.log10(d / d0_m)
    return db_to_linear(pl_db)


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
        G = np.sqrt(gain_br) * complex_normal((c.m_ris_elements, c.n_bs_antennas), self.rng)

        d_bu = _distance(topology.ue_xy, np.broadcast_to(topology.bs_xy, topology.ue_xy.shape))
        gain_bu = pathloss_gain_linear(
            d_bu,
            beta=c.beta_bu,
            pl0_db=c.pathloss_pl0_db,
            d0_m=c.pathloss_d0_m,
        )
        g_bu = np.sqrt(gain_bu)[:, None] * complex_normal((c.k_users, c.n_bs_antennas), self.rng)

        d_ru = _distance(topology.ue_xy, np.broadcast_to(topology.irs_xy, topology.ue_xy.shape))
        gain_ru = pathloss_gain_linear(
            d_ru,
            beta=c.beta_ru,
            pl0_db=c.pathloss_pl0_db,
            d0_m=c.pathloss_d0_m,
        )
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
