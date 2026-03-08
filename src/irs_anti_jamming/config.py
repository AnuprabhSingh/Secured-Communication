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

    p_jammer_min_dbm: float = 15.0
    p_jammer_max_dbm: float = 40.0

    bs_position_xy: tuple[float, float] = (0.0, 0.0)
    irs_position_xy: tuple[float, float] = (75.0, 100.0)
    ue_region_xyxy: tuple[float, float, float, float] = (50.0, 150.0, 0.0, 100.0)
    jammer_region_xyxy: tuple[float, float, float, float] = (50.0, 100.0, 0.0, 100.0)

    seed: int = 7


@dataclass(slots=True)
class RLConfig:
    alpha: float = 0.008
    gamma: float = 0.9

    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995

    lambda1: float = 1.0
    lambda2: float = 2.0

    xi_win: float = 0.01
    xi_loss: float = 0.04

    state_bins: int = 8
    fuzzy_centers: tuple[float, float, float] = (0.0, 0.5, 1.0)


@dataclass(slots=True)
class TrainEvalConfig:
    train_episodes: int = 1200
    train_steps_per_episode: int = 30
    eval_episodes: int = 50
    eval_steps_per_episode: int = 10
    n_seeds: int = 3


@dataclass(slots=True)
class SweepConfig:
    pmax_dbm_values: list[float] = field(
        default_factory=lambda: [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    )
    ris_elements_values: list[int] = field(default_factory=lambda: [20, 40, 60, 80, 100])
    sinr_target_db_values: list[float] = field(default_factory=lambda: [10.0, 15.0, 20.0, 25.0])
