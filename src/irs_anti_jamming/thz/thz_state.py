"""Wideband THz state features — adapted from narrowband state module.

Reuses the same 3-feature representation (jammer power, channel quality,
previous SINR) but with THz-appropriate normalization ranges.

The base StateAggregator hardcodes sub-6 GHz normalization ranges
(jammer 15-40 dBm, channel -100 to -40 dB).  At THz the jammer is
weaker (5-15 dBm), channel quality is much lower (-160 to -60 dB),
and the SINR operating point differs.  Without this override ALL
states collapse to the same discrete_id and the RL agent cannot
distinguish any situations.
"""
from __future__ import annotations

import numpy as np

from ..state import StateAggregator, StateRepresentation
from ..utils import linear_to_db, watt_to_dbm


class THzStateAggregator(StateAggregator):
    """State aggregator with THz-appropriate normalization ranges."""

    def __init__(
        self,
        bins: int = 8,
        centers: tuple[float, float, float] = (0.0, 0.5, 1.0),
        jammer_range_dbm: tuple[float, float] = (0.0, 25.0),
        channel_range_db: tuple[float, float] = (-160.0, -60.0),
        sinr_range_db: tuple[float, float] = (-20.0, 30.0),
    ):
        super().__init__(bins=bins, centers=centers)
        self._pj_lo, self._pj_hi = jammer_range_dbm
        self._ch_lo, self._ch_hi = channel_range_db
        self._sinr_lo, self._sinr_hi = sinr_range_db

    def _normalize_features(
        self,
        prev_jammer_p_watt: np.ndarray,
        channel_quality_linear: np.ndarray,
        prev_sinr_linear: np.ndarray,
    ) -> np.ndarray:
        """Normalize state features to [0, 1] using THz-specific ranges.

        Without this override the base class maps THz jammer power (5-15 dBm)
        into its [15, 40] dBm range, producing f_pj ≈ 0 for every state.
        Similarly, THz channel quality is orders of magnitude below the
        sub-6 GHz operating point, collapsing f_ch to a constant.
        """
        pj_dbm = watt_to_dbm(prev_jammer_p_watt)
        ch_db = linear_to_db(np.maximum(channel_quality_linear, 1e-20))
        sinr_db = linear_to_db(np.maximum(prev_sinr_linear, 1e-12))

        mean_pj = float(np.mean(pj_dbm))
        max_pj = float(np.max(pj_dbm))
        mean_ch = float(np.mean(ch_db))
        std_ch = float(np.std(ch_db))
        mean_sinr = float(np.mean(sinr_db))
        min_sinr = float(np.min(sinr_db))

        pj_range = max(self._pj_hi - self._pj_lo, 1.0)
        ch_range = max(self._ch_hi - self._ch_lo, 1.0)
        sinr_range = max(self._sinr_hi - self._sinr_lo, 1.0)

        # Jammer pressure (mean + peak)
        f_pj_mean = np.clip((mean_pj - self._pj_lo) / pj_range, 0.0, 1.0)
        f_pj_max = np.clip((max_pj - self._pj_lo) / pj_range, 0.0, 1.0)
        f_pj = float(0.6 * f_pj_mean + 0.4 * f_pj_max)

        # Channel quality (mean level + spread across users)
        f_ch_mean = np.clip((mean_ch - self._ch_lo) / ch_range, 0.0, 1.0)
        f_ch_spread = np.clip(std_ch / 30.0, 0.0, 1.0)
        f_ch = float(0.75 * f_ch_mean + 0.25 * (1.0 - f_ch_spread))

        # SINR quality (mean + worst-user)
        f_sinr_mean = np.clip((mean_sinr - self._sinr_lo) / sinr_range, 0.0, 1.0)
        f_sinr_min = np.clip((min_sinr - self._sinr_lo) / sinr_range, 0.0, 1.0)
        f_sinr = float(0.5 * f_sinr_mean + 0.5 * f_sinr_min)

        return np.asarray([f_pj, f_ch, f_sinr], dtype=float)

    def build_thz(
        self,
        prev_jammer_p_watt: np.ndarray,
        wideband_channel_quality: np.ndarray,
        prev_avg_sinr: np.ndarray,
    ) -> StateRepresentation:
        """Build state from wideband-averaged quantities.

        Args:
            prev_jammer_p_watt: (K,) jammer power per user (watts)
            wideband_channel_quality: (K,) mean |h_eff|^2 over subcarriers per user
            prev_avg_sinr: (K,) mean SINR over subcarriers per user (linear)

        Returns:
            StateRepresentation with discrete_id, features, fuzzy_memberships.
        """
        return super().build(prev_jammer_p_watt, wideband_channel_quality, prev_avg_sinr)
