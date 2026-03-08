from __future__ import annotations

import numpy as np

from .config import SystemConfig
from .utils import dbm_to_watt, linear_to_db


class SmartJammer:
    def __init__(self, cfg: SystemConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def sample_powers_watt(self, prev_sinr_linear: np.ndarray, predictability: float = 0.0) -> np.ndarray:
        prev_sinr_db = linear_to_db(np.maximum(prev_sinr_linear, 1e-12))
        eta = float(np.clip(predictability, 0.0, 1.0))

        reactive = 1.0 * (prev_sinr_db - 3.0)
        base = 0.5 * (self.cfg.p_jammer_min_dbm + self.cfg.p_jammer_max_dbm)
        exploit_boost = 15.0 * eta
        noise = self.rng.normal(0.0, 3.0, size=self.cfg.k_users)
        pj_dbm = base + reactive + exploit_boost + noise
        pj_dbm = np.clip(pj_dbm, self.cfg.p_jammer_min_dbm, self.cfg.p_jammer_max_dbm)
        return np.asarray(dbm_to_watt(pj_dbm), dtype=float)

    def sample_precoders(self, h_ju: np.ndarray, predictability: float = 0.0) -> np.ndarray:
        z_rand = (
            self.rng.standard_normal((self.cfg.k_users, self.cfg.n_jammer_antennas))
            + 1j * self.rng.standard_normal((self.cfg.k_users, self.cfg.n_jammer_antennas))
        ) / np.sqrt(2.0)

        h_norm = np.linalg.norm(h_ju, axis=1, keepdims=True)
        h_norm = np.maximum(h_norm, 1e-12)
        z_target = h_ju / h_norm

        eta = float(np.clip(predictability, 0.0, 1.0))
        z = (1.0 - eta) * z_rand + eta * z_target

        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return z / norms
