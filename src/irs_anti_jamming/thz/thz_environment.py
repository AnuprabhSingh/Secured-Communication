"""Wideband THz anti-jamming environment.

Follows the same Gym-like API as the narrowband IRSAntiJammingEnv:
  reset() -> Observation
  step(action_idx) -> (Observation, reward, info)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .thz_config import THzSystemConfig, THzRLConfig
from .thz_channel_model import THzChannelModel, THzChannelSnapshot, THzTopology
from .spdp_ris import SPDPResponse, compute_spdp_closed_form, optimize_spdp_multiuser
from .thz_action_space import THzActionContext, THzHybridActionSpace
from .thz_system_model import (
    THzSystemMetrics,
    evaluate_thz_system,
    thz_channel_quality,
)
from ..jammer import SmartJammer
from ..utils import db_to_linear, dbm_to_watt


@dataclass(slots=True)
class THzSlotSample:
    snapshot: THzChannelSnapshot
    ref_spdp: SPDPResponse           # centroid SPDP for quality estimates
    p_jammer_watt: np.ndarray
    z_jammer: np.ndarray
    channel_quality_linear: np.ndarray


@dataclass(slots=True)
class THzObservation:
    slot: THzSlotSample
    prev_jammer_watt: np.ndarray
    prev_sinr_linear: np.ndarray


class THzAntiJammingEnv:
    """Wideband THz environment with SPDP-RIS and hybrid beamforming."""

    def __init__(self, sys_cfg: THzSystemConfig, rl_cfg: THzRLConfig,
                 seed: int | None = None):
        self.sys_cfg = sys_cfg
        self.rl_cfg = rl_cfg
        self.rng = np.random.default_rng(sys_cfg.seed if seed is None else seed)

        self.topology = THzTopology(sys_cfg, self.rng)
        self.channel_model = THzChannelModel(sys_cfg, self.rng)

        # Reuse narrowband SmartJammer with THz power bounds via a
        # lightweight config adapter
        self._jammer_cfg = _JammerConfigAdapter(sys_cfg)
        self.jammer = SmartJammer(self._jammer_cfg, self.rng)

        self.action_space = THzHybridActionSpace(
            cfg=sys_cfg,
            seed=int(self.rng.integers(0, 2**31 - 1)),
        )

        # Use proper thermal noise kTB instead of fixed value
        self.noise_watt = float(dbm_to_watt(sys_cfg.noise_power_dbm))

        K = sys_cfg.k_users
        self.prev_jammer_watt = np.full(
            K,
            float(dbm_to_watt(0.5 * (sys_cfg.p_jammer_min_dbm + sys_cfg.p_jammer_max_dbm))),
        )
        self.prev_sinr_linear = np.full(K, float(db_to_linear(sys_cfg.sinr_min_db)))
        self.current_slot: THzSlotSample | None = None

        self._action_history: list[int] = []
        self._action_history_window = 25

    def set_fast_mode(self, fast: bool) -> None:
        """Enable/disable fast SPDP mode (skip multiuser search during training)."""
        self.action_space.fast_mode = fast

    # --- Action predictability (same as narrowband) ---

    def _record_action(self, action_sig: int) -> None:
        self._action_history.append(int(action_sig))
        if len(self._action_history) > self._action_history_window:
            self._action_history = self._action_history[-self._action_history_window:]

    def _predictability_score(self) -> float:
        if len(self._action_history) <= 1:
            return 0.0
        history = np.array(self._action_history, dtype=int)
        unique, counts = np.unique(history, return_counts=True)
        dominant = float(np.max(counts)) / float(len(history))
        repeats = float(np.mean(history[1:] == history[:-1]))
        if len(unique) <= 1:
            return 1.0
        chance = 1.0 / max(1, len(unique))
        norm = (dominant - chance) / max(1e-12, 1.0 - chance)

        # Entropy-based component: low entropy ↔ concentrated action distribution
        # A deterministic agent visiting few actions has low entropy even with
        # varying states. This catches agents that argmax selects from a small
        # subset of actions. Max entropy = log2(n_actions).
        n_total = 42  # action space size
        probs = counts / float(len(history))
        entropy = -float(np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0))))
        max_entropy = np.log2(n_total)
        entropy_norm = 1.0 - min(1.0, entropy / max_entropy)  # 1 = predictable, 0 = random

        return float(np.clip(0.25 * repeats + 0.35 * np.clip(norm, 0, 1) + 0.4 * entropy_norm, 0, 1))

    # --- Channel sampling ---

    def _ref_spdp(self, snapshot: THzChannelSnapshot) -> SPDPResponse:
        """Centroid SPDP for channel quality estimation."""
        return compute_spdp_closed_form(
            self.sys_cfg,
            snapshot.theta_aoa, snapshot.zeta_aoa,
            float(np.mean(snapshot.theta_aod)),
            float(np.mean(snapshot.zeta_aod)),
        )

    def _sample_slot(self) -> THzSlotSample:
        self.topology.move_jammer()
        snapshot = self.channel_model.sample(self.topology)
        ref_spdp = self._ref_spdp(snapshot)

        cq = thz_channel_quality(self.sys_cfg, snapshot, ref_spdp,
                                  subcarrier_stride=max(1, self.sys_cfg.n_subcarriers // 16))

        predictability = self._predictability_score()
        p_jammer = self.jammer.sample_powers_watt(self.prev_sinr_linear,
                                                   predictability=predictability)
        z_jammer = self.jammer.sample_precoders(snapshot.h_ju,
                                                 predictability=predictability)

        return THzSlotSample(
            snapshot=snapshot, ref_spdp=ref_spdp,
            p_jammer_watt=p_jammer, z_jammer=z_jammer,
            channel_quality_linear=cq,
        )

    # --- Public API ---

    def reset(self, resample_users: bool = True, keep_history: bool = False) -> THzObservation:
        if resample_users:
            self.topology.resample_users()
        if not keep_history:
            self._action_history = []
        
        # Reset jammer state for consistent episode starts
        if hasattr(self.jammer, 'reset') and not keep_history:
            self.jammer.reset()

        K = self.sys_cfg.k_users
        self.prev_jammer_watt = np.full(
            K,
            float(dbm_to_watt(0.5 * (self.sys_cfg.p_jammer_min_dbm + self.sys_cfg.p_jammer_max_dbm))),
        )
        self.prev_sinr_linear = np.full(K, float(db_to_linear(self.sys_cfg.sinr_min_db)))
        self.current_slot = self._sample_slot()
        return THzObservation(
            slot=self.current_slot,
            prev_jammer_watt=self.prev_jammer_watt.copy(),
            prev_sinr_linear=self.prev_sinr_linear.copy(),
        )

    def _build_action_context(self) -> THzActionContext:
        assert self.current_slot is not None
        return THzActionContext(
            snapshot=self.current_slot.snapshot,
            pmax_watt=float(dbm_to_watt(self.sys_cfg.pmax_dbm)),
            sinr_min_db=self.sys_cfg.sinr_min_db,
            prev_sinr_linear=self.prev_sinr_linear,
            channel_quality_linear=self.current_slot.channel_quality_linear,
            noise_watt=self.noise_watt,
            ref_spdp=self.current_slot.ref_spdp,
        )

    def action_context(self) -> THzActionContext:
        return self._build_action_context()

    def evaluate_action(self, action_idx: int) -> tuple[THzSystemMetrics, np.ndarray, SPDPResponse]:
        ctx = self._build_action_context()
        p_bs, spdp = self.action_space.decode(action_idx, ctx)
        metrics = evaluate_thz_system(
            cfg=self.sys_cfg,
            snapshot=self.current_slot.snapshot,
            spdp=spdp,
            p_bs_watt=p_bs,
            p_jammer_watt=self.current_slot.p_jammer_watt,
            z_jammer=self.current_slot.z_jammer,
            noise_watt=self.noise_watt,
            sinr_min_db=self.sys_cfg.sinr_min_db,
            lambda1=self.rl_cfg.lambda1,
            lambda2=self.rl_cfg.lambda2,
            pmax_watt=float(dbm_to_watt(self.sys_cfg.pmax_dbm)),
            subcarrier_stride=self.sys_cfg.subcarrier_stride,
        )
        return metrics, p_bs, spdp

    def _advance(self, realized_sinr: np.ndarray, action_sig: int | None = None) -> THzObservation:
        if action_sig is not None:
            self._record_action(action_sig)
        self.prev_jammer_watt = self.current_slot.p_jammer_watt.copy()
        self.prev_sinr_linear = realized_sinr.copy()
        self.current_slot = self._sample_slot()
        return THzObservation(
            slot=self.current_slot,
            prev_jammer_watt=self.prev_jammer_watt.copy(),
            prev_sinr_linear=self.prev_sinr_linear.copy(),
        )

    def step(self, action_idx: int) -> tuple[THzObservation, float, dict[str, float]]:
        metrics, _, _ = self.evaluate_action(action_idx)
        # Average SINR per user across subcarriers
        avg_sinr = np.mean(metrics.sinr_linear, axis=1)
        obs = self._advance(avg_sinr, action_sig=action_idx)
        info = {
            "system_rate": metrics.system_rate,
            "sinr_protection": metrics.sinr_protection_level,
            "reward": metrics.reward,
        }
        return obs, metrics.reward, info


# ---------------------------------------------------------------------------
# Adapter to reuse SmartJammer with THzSystemConfig
# ---------------------------------------------------------------------------

class _JammerConfigAdapter:
    """Bridge THzSystemConfig attributes to what SmartJammer expects."""

    def __init__(self, cfg: THzSystemConfig):
        self.k_users = cfg.k_users
        self.n_jammer_antennas = cfg.n_jammer_antennas
        self.p_jammer_min_dbm = cfg.p_jammer_min_dbm
        self.p_jammer_max_dbm = cfg.p_jammer_max_dbm
