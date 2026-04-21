"""THz action space: RL chooses power allocation, SPDP + hybrid BF computed optimally."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .thz_config import THzSystemConfig
from .thz_channel_model import THzChannelSnapshot
from .spdp_ris import SPDPResponse, optimize_spdp_multiuser
from .thz_system_model import evaluate_thz_system, thz_channel_quality
from ..utils import db_to_linear, linear_to_db, project_to_simplex


@dataclass(slots=True)
class THzActionContext:
    snapshot: THzChannelSnapshot
    pmax_watt: float
    sinr_min_db: float
    prev_sinr_linear: np.ndarray     # (K,) average SINR per user from prev step
    channel_quality_linear: np.ndarray  # (K,) avg channel power
    noise_watt: float
    ref_spdp: SPDPResponse           # reference SPDP (centroid) for quality estimates


class THzHybridActionSpace:
    """RL selects power allocation (42 actions); SPDP + hybrid BF computed optimally."""

    def __init__(self, cfg: THzSystemConfig, seed: int, n_ao_candidates: int = 5):
        self.cfg = cfg
        self.k_users = cfg.k_users
        self.rng = np.random.default_rng(seed)
        self.n_ao_candidates = n_ao_candidates
        self.fast_mode = False  # Skip per-user SPDP search during training

        self.total_power_fractions = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]
        self.power_modes = [
            "equal",
            "channel_proportional",
            "inverse_channel",
            "sinr_deficit",
            "waterfilling",
            "max_protection",
        ]

        self.actions: list[tuple[int, int]] = [
            (f, m)
            for f in range(len(self.total_power_fractions))
            for m in range(len(self.power_modes))
        ]

    @property
    def size(self) -> int:
        return len(self.actions)  # 7 fractions x 6 modes = 42

    def _decode_powers(self, fraction_idx: int, mode_idx: int,
                       ctx: THzActionContext) -> np.ndarray:
        total = self.total_power_fractions[fraction_idx] * ctx.pmax_watt
        mode = self.power_modes[mode_idx]

        if mode == "equal":
            weights = np.full(self.k_users, 1.0 / self.k_users)
        elif mode == "channel_proportional":
            weights = project_to_simplex(ctx.channel_quality_linear)
        elif mode == "inverse_channel":
            weights = project_to_simplex(1.0 / np.maximum(ctx.channel_quality_linear, 1e-12))
        elif mode == "waterfilling":
            cq = np.maximum(ctx.channel_quality_linear, 1e-12)
            wf = np.log1p(cq)
            weights = project_to_simplex(wf)
        else:  # sinr_deficit
            deficit = np.maximum(
                ctx.sinr_min_db - linear_to_db(np.maximum(ctx.prev_sinr_linear, 1e-12)),
                0.0,
            )
            weights = project_to_simplex(deficit + 1e-3)

        if mode == "max_protection":
            # Focus power on users with worst SINR (inverse-SINR weighting)
            inv_sinr = 1.0 / np.maximum(ctx.prev_sinr_linear, 1e-12)
            weights = project_to_simplex(inv_sinr)

        return total * weights

    def decode(self, action_idx: int, ctx: THzActionContext) -> tuple[np.ndarray, SPDPResponse]:
        """Decode RL action into (power allocation, SPDP config).

        Returns:
            p_bs_watt: (K,) per-user power
            spdp: optimized SPDPResponse
        """
        frac_idx, mode_idx = self.actions[action_idx]
        p_bs = self._decode_powers(frac_idx, mode_idx, ctx)

        # Fast mode: use pre-computed centroid SPDP (no multiuser search)
        if self.fast_mode:
            return p_bs, ctx.ref_spdp

        # Multi-user SPDP: evaluate candidates and pick best
        def rate_eval_fn(spdp_candidate):
            metrics = evaluate_thz_system(
                cfg=self.cfg,
                snapshot=ctx.snapshot,
                spdp=spdp_candidate,
                p_bs_watt=p_bs,
                p_jammer_watt=np.zeros(self.k_users),
                z_jammer=np.zeros((self.k_users, self.cfg.n_jammer_antennas)),
                noise_watt=ctx.noise_watt,
                sinr_min_db=ctx.sinr_min_db,
                lambda1=0.0, lambda2=0.0,
                pmax_watt=ctx.pmax_watt,
                subcarrier_stride=max(1, self.cfg.n_subcarriers // 16),
            )
            return metrics.system_rate

        spdp = optimize_spdp_multiuser(
            cfg=self.cfg,
            theta_aoa=ctx.snapshot.theta_aoa,
            zeta_aoa=ctx.snapshot.zeta_aoa,
            theta_aod_arr=ctx.snapshot.theta_aod,
            zeta_aod_arr=ctx.snapshot.zeta_aod,
            rate_eval_fn=rate_eval_fn,
            p_bs_watt=p_bs,
            noise_watt=ctx.noise_watt,
            snapshot=ctx.snapshot,
        )

        return p_bs, spdp

    def power_candidates_only(self, ctx: THzActionContext) -> list[np.ndarray]:
        return [
            self._decode_powers(f, m, ctx)
            for f in range(len(self.total_power_fractions))
            for m in range(len(self.power_modes))
        ]
