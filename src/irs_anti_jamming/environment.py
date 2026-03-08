from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_space import ActionContext, JointActionSpace
from .channel_model import ChannelModel, ChannelSnapshot, Topology
from .config import RLConfig, SystemConfig
from .jammer import SmartJammer
from .system_model import SystemMetrics, channel_quality, evaluate_system
from .utils import db_to_linear, dbm_to_watt


@dataclass(slots=True)
class SlotSample:
    snapshot: ChannelSnapshot
    p_jammer_watt: np.ndarray
    z_jammer: np.ndarray
    channel_quality_linear: np.ndarray


@dataclass(slots=True)
class Observation:
    slot: SlotSample
    prev_jammer_watt: np.ndarray
    prev_sinr_linear: np.ndarray


class IRSAntiJammingEnv:
    def __init__(self, sys_cfg: SystemConfig, rl_cfg: RLConfig, seed: int | None = None):
        self.sys_cfg = sys_cfg
        self.rl_cfg = rl_cfg
        self.rng = np.random.default_rng(sys_cfg.seed if seed is None else seed)

        self.topology = Topology(sys_cfg, self.rng)
        self.channel_model = ChannelModel(sys_cfg, self.rng)
        self.jammer = SmartJammer(sys_cfg, self.rng)
        self.action_space = JointActionSpace(
            k_users=sys_cfg.k_users,
            m_ris_elements=sys_cfg.m_ris_elements,
            seed=int(self.rng.integers(0, 2**31 - 1)),
        )

        self.noise_watt = float(dbm_to_watt(sys_cfg.noise_dbm))

        self.prev_jammer_watt = np.full(
            sys_cfg.k_users,
            float(dbm_to_watt(0.5 * (sys_cfg.p_jammer_min_dbm + sys_cfg.p_jammer_max_dbm))),
            dtype=float,
        )
        self.prev_sinr_linear = np.full(sys_cfg.k_users, float(db_to_linear(sys_cfg.sinr_min_db)), dtype=float)
        self.current_slot: SlotSample | None = None

        self._action_history: list[int] = []
        self._action_history_window = 20

    def _record_action(self, action_signature: int) -> None:
        self._action_history.append(int(action_signature))
        if len(self._action_history) > self._action_history_window:
            self._action_history = self._action_history[-self._action_history_window :]

    def _predictability_score(self) -> float:
        if len(self._action_history) <= 1:
            return 0.0

        history = np.asarray(self._action_history, dtype=int)
        unique, counts = np.unique(history, return_counts=True)
        dominant_ratio = float(np.max(counts)) / float(len(history))

        repeats = float(np.mean(history[1:] == history[:-1]))
        chance = 1.0 / max(1, len(unique))
        dominant_norm = (dominant_ratio - chance) / max(1e-12, 1.0 - chance)
        dominant_norm = float(np.clip(dominant_norm, 0.0, 1.0))

        return float(np.clip(0.5 * repeats + 0.5 * dominant_norm, 0.0, 1.0))

    def _sample_slot(self) -> SlotSample:
        self.topology.move_jammer()
        snapshot = self.channel_model.sample(self.topology)

        predictability = self._predictability_score()
        p_jammer = self.jammer.sample_powers_watt(self.prev_sinr_linear, predictability=predictability)
        z_jammer = self.jammer.sample_precoders(snapshot.h_ju, predictability=predictability)
        q = channel_quality(snapshot)
        return SlotSample(snapshot=snapshot, p_jammer_watt=p_jammer, z_jammer=z_jammer, channel_quality_linear=q)

    def reset(self, resample_users: bool = True) -> Observation:
        if resample_users:
            self.topology.resample_users()

        self._action_history = []

        self.prev_jammer_watt = np.full(
            self.sys_cfg.k_users,
            float(dbm_to_watt(0.5 * (self.sys_cfg.p_jammer_min_dbm + self.sys_cfg.p_jammer_max_dbm))),
            dtype=float,
        )
        self.prev_sinr_linear = np.full(
            self.sys_cfg.k_users,
            float(db_to_linear(self.sys_cfg.sinr_min_db)),
            dtype=float,
        )
        self.current_slot = self._sample_slot()
        return Observation(
            slot=self.current_slot,
            prev_jammer_watt=self.prev_jammer_watt.copy(),
            prev_sinr_linear=self.prev_sinr_linear.copy(),
        )

    def _build_action_context(self) -> ActionContext:
        assert self.current_slot is not None
        return ActionContext(
            snapshot=self.current_slot.snapshot,
            pmax_watt=float(dbm_to_watt(self.sys_cfg.pmax_dbm)),
            sinr_min_db=self.sys_cfg.sinr_min_db,
            prev_sinr_linear=self.prev_sinr_linear,
            channel_quality_linear=self.current_slot.channel_quality_linear,
        )

    def action_context(self) -> ActionContext:
        return self._build_action_context()

    def evaluate_action(self, action_idx: int, use_irs: bool = True) -> tuple[SystemMetrics, np.ndarray, np.ndarray]:
        assert self.current_slot is not None
        ctx = self._build_action_context()
        p_bs_watt, theta = self.action_space.decode(action_idx, ctx)
        metrics = evaluate_system(
            snapshot=self.current_slot.snapshot,
            p_bs_watt=p_bs_watt,
            theta=theta,
            p_jammer_watt=self.current_slot.p_jammer_watt,
            z_jammer=self.current_slot.z_jammer,
            noise_watt=self.noise_watt,
            sinr_min_db=self.sys_cfg.sinr_min_db,
            lambda1=self.rl_cfg.lambda1,
            lambda2=self.rl_cfg.lambda2,
            use_irs=use_irs,
        )
        return metrics, p_bs_watt, theta

    def evaluate_action_with_jammer_estimate(
        self,
        action_idx: int,
        use_irs: bool = True,
    ) -> tuple[SystemMetrics, np.ndarray, np.ndarray]:
        assert self.current_slot is not None
        ctx = self._build_action_context()
        p_bs_watt, theta = self.action_space.decode(action_idx, ctx)

        p_jammer_est = self.prev_jammer_watt.copy()
        z_est = np.ones(
            (self.sys_cfg.k_users, self.sys_cfg.n_jammer_antennas),
            dtype=np.complex128,
        ) / np.sqrt(self.sys_cfg.n_jammer_antennas)

        metrics = evaluate_system(
            snapshot=self.current_slot.snapshot,
            p_bs_watt=p_bs_watt,
            theta=theta,
            p_jammer_watt=p_jammer_est,
            z_jammer=z_est,
            noise_watt=self.noise_watt,
            sinr_min_db=self.sys_cfg.sinr_min_db,
            lambda1=self.rl_cfg.lambda1,
            lambda2=self.rl_cfg.lambda2,
            use_irs=use_irs,
        )
        return metrics, p_bs_watt, theta

    def evaluate_action_without_jammer(
        self,
        action_idx: int,
        use_irs: bool = True,
    ) -> tuple[SystemMetrics, np.ndarray, np.ndarray]:
        assert self.current_slot is not None
        ctx = self._build_action_context()
        p_bs_watt, theta = self.action_space.decode(action_idx, ctx)

        p_jammer_zero = np.zeros(self.sys_cfg.k_users, dtype=float)
        z_dummy = np.ones(
            (self.sys_cfg.k_users, self.sys_cfg.n_jammer_antennas),
            dtype=np.complex128,
        ) / np.sqrt(self.sys_cfg.n_jammer_antennas)

        metrics = evaluate_system(
            snapshot=self.current_slot.snapshot,
            p_bs_watt=p_bs_watt,
            theta=theta,
            p_jammer_watt=p_jammer_zero,
            z_jammer=z_dummy,
            noise_watt=self.noise_watt,
            sinr_min_db=self.sys_cfg.sinr_min_db,
            lambda1=self.rl_cfg.lambda1,
            lambda2=self.rl_cfg.lambda2,
            use_irs=use_irs,
        )
        return metrics, p_bs_watt, theta

    def evaluate_power_only_no_irs(self, p_bs_watt: np.ndarray) -> SystemMetrics:
        assert self.current_slot is not None
        theta = np.zeros(self.sys_cfg.m_ris_elements, dtype=float)
        return evaluate_system(
            snapshot=self.current_slot.snapshot,
            p_bs_watt=p_bs_watt,
            theta=theta,
            p_jammer_watt=self.current_slot.p_jammer_watt,
            z_jammer=self.current_slot.z_jammer,
            noise_watt=self.noise_watt,
            sinr_min_db=self.sys_cfg.sinr_min_db,
            lambda1=self.rl_cfg.lambda1,
            lambda2=self.rl_cfg.lambda2,
            use_irs=False,
        )

    def _advance(self, realized_sinr_linear: np.ndarray, action_signature: int | None = None) -> Observation:
        if action_signature is not None:
            self._record_action(action_signature)

        self.prev_jammer_watt = self.current_slot.p_jammer_watt.copy()
        self.prev_sinr_linear = realized_sinr_linear.copy()
        self.current_slot = self._sample_slot()
        return Observation(
            slot=self.current_slot,
            prev_jammer_watt=self.prev_jammer_watt.copy(),
            prev_sinr_linear=self.prev_sinr_linear.copy(),
        )

    def advance_with_sinr(self, realized_sinr_linear: np.ndarray, action_signature: int | None = None) -> Observation:
        return self._advance(realized_sinr_linear, action_signature=action_signature)

    def step(self, action_idx: int) -> tuple[Observation, float, dict[str, float]]:
        metrics, _, _ = self.evaluate_action(action_idx, use_irs=True)
        obs = self._advance(metrics.sinr_linear, action_signature=action_idx)
        info = {
            "system_rate": metrics.system_rate,
            "sinr_protection": metrics.sinr_protection_level,
            "reward": metrics.reward,
        }
        return obs, metrics.reward, info
