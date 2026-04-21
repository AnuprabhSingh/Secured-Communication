from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import linear_to_db, watt_to_dbm


@dataclass(slots=True)
class StateRepresentation:
    features: np.ndarray
    discrete_id: int
    fuzzy_memberships: np.ndarray


class StateAggregator:
    def __init__(self, bins: int = 8, centers: tuple[float, float, float] = (0.0, 0.5, 1.0)):
        self.bins = bins
        self.centers = np.asarray(centers, dtype=float)
        self.n_centers = len(centers)
        self.n_fuzzy_states = self.n_centers**3
        self.width = 1.0 / max(1, self.n_centers - 1)

    def _normalize_features(
        self,
        prev_jammer_p_watt: np.ndarray,
        channel_quality_linear: np.ndarray,
        prev_sinr_linear: np.ndarray,
    ) -> np.ndarray:
        pj_dbm = watt_to_dbm(prev_jammer_p_watt)
        ch_db = linear_to_db(np.maximum(channel_quality_linear, 1e-12))
        sinr_db = linear_to_db(np.maximum(prev_sinr_linear, 1e-12))

        mean_pj_dbm = float(np.mean(pj_dbm))
        max_pj_dbm = float(np.max(pj_dbm))

        mean_ch_db = float(np.mean(ch_db))
        std_ch_db = float(np.std(ch_db))

        mean_sinr_db = float(np.mean(sinr_db))
        min_sinr_db = float(np.min(sinr_db))

        f_pj_mean = np.clip((mean_pj_dbm - 15.0) / 25.0, 0.0, 1.0)
        f_pj_max = np.clip((max_pj_dbm - 15.0) / 25.0, 0.0, 1.0)
        f_pj = 0.6 * f_pj_mean + 0.4 * f_pj_max

        f_ch_mean = np.clip((mean_ch_db + 100.0) / 60.0, 0.0, 1.0)
        f_ch_spread = np.clip(std_ch_db / 20.0, 0.0, 1.0)
        f_ch = 0.75 * f_ch_mean + 0.25 * (1.0 - f_ch_spread)

        f_sinr_mean = np.clip((mean_sinr_db + 10.0) / 40.0, 0.0, 1.0)
        f_sinr_min = np.clip((min_sinr_db + 10.0) / 40.0, 0.0, 1.0)
        f_sinr = 0.5 * f_sinr_mean + 0.5 * f_sinr_min
        return np.asarray([f_pj, f_ch, f_sinr], dtype=float)

    def _discrete_id(self, features: np.ndarray) -> int:
        idx = np.clip((features * self.bins).astype(int), 0, self.bins - 1)
        return int(idx[0] * self.bins * self.bins + idx[1] * self.bins + idx[2])

    def _triangular_memberships(self, x: float) -> np.ndarray:
        mu = np.maximum(1.0 - np.abs(x - self.centers) / max(self.width, 1e-9), 0.0)
        if mu.sum() <= 1e-12:
            best = int(np.argmin(np.abs(x - self.centers)))
            mu[best] = 1.0
        else:
            mu = mu / mu.sum()
        return mu

    def _fuzzy_memberships(self, features: np.ndarray) -> np.ndarray:
        mu0 = self._triangular_memberships(float(features[0]))
        mu1 = self._triangular_memberships(float(features[1]))
        mu2 = self._triangular_memberships(float(features[2]))

        memberships = np.zeros(self.n_fuzzy_states, dtype=float)
        idx = 0
        for i in range(self.n_centers):
            for j in range(self.n_centers):
                for k in range(self.n_centers):
                    memberships[idx] = mu0[i] * mu1[j] * mu2[k]
                    idx += 1

        total = memberships.sum()
        if total <= 1e-12:
            memberships[0] = 1.0
        else:
            memberships /= total
        return memberships

    def build(
        self,
        prev_jammer_p_watt: np.ndarray,
        channel_quality_linear: np.ndarray,
        prev_sinr_linear: np.ndarray,
    ) -> StateRepresentation:
        features = self._normalize_features(prev_jammer_p_watt, channel_quality_linear, prev_sinr_linear)
        sid = self._discrete_id(features)
        fuzzy = self._fuzzy_memberships(features)
        return StateRepresentation(features=features, discrete_id=sid, fuzzy_memberships=fuzzy)
