from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SystemConfig:
    k_users: int = 4
    n_bs_antennas: int = 8
    n_jammer_antennas: int = 8
    m_ris_elements: int = 60

    pmax_dbm: float = 30.0
    sinr_min_db: float = 10.0
    noise_dbm: float = -105.0

    pathloss_pl0_db: float = 30.0
    pathloss_d0_m: float = 1.0
    beta_bu: float = 3.75
    beta_br: float = 2.2
    beta_ru: float = 2.2
    beta_ju: float = 2.5

    enable_geometry_los: bool = True
    carrier_wavelength_m: float = 0.1
    rician_k_br_db: float = 8.0
    rician_k_bu_db: float = 3.0
    rician_k_ru_db: float = 6.0

    p_jammer_min_dbm: float = 15.0
    p_jammer_max_dbm: float = 40.0

    bs_position_xy: tuple[float, float] = (0.0, 0.0)
    irs_position_xy: tuple[float, float] = (75.0, 100.0)
    ue_region_xyxy: tuple[float, float, float, float] = (50.0, 150.0, 0.0, 100.0)
    jammer_region_xyxy: tuple[float, float, float, float] = (50.0, 100.0, 0.0, 100.0)

    seed: int = 7


@dataclass(slots=True)
class RLConfig:
    alpha: float = 0.01   # Slightly higher for 30-action space (paper: 0.5e-2)
    gamma: float = 0.9

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    lambda1: float = 0.5  # Paper Eq.7 — power penalty (lower: less penalty for using power)
    lambda2: float = 3.0  # Paper Eq.7 — QoS penalty (higher: strongly penalize outage)

    xi_win: float = 0.01  # Paper [46],[47]
    xi_loss: float = 0.04  # Paper [46],[47]

    wolf_eval_temperature: float = 15.0  # Boltzmann temperature for WoLF-PHC eval softmax

    state_bins: int = 8
    fuzzy_centers: tuple[float, float, float] = (0.0, 0.5, 1.0)


@dataclass(slots=True)
class TrainEvalConfig:
    train_episodes: int = 1200
    train_steps_per_episode: int = 50
    eval_episodes: int = 50
    eval_steps_per_episode: int = 20
    n_seeds: int = 3


@dataclass(slots=True)
class SweepConfig:
    pmax_dbm_values: list[float] = field(
        default_factory=lambda: [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    )
    ris_elements_values: list[int] = field(default_factory=lambda: [20, 40, 60, 80, 100])
    sinr_target_db_values: list[float] = field(default_factory=lambda: [10.0, 15.0, 20.0, 25.0])
