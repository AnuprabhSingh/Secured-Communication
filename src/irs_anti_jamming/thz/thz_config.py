"""Configuration dataclasses for wideband THz IRS anti-jamming system.

References:
  - Su et al., "Wideband Precoding for RIS-Aided THz Communications",
    IEEE Trans. Commun., 2023.
  - Yan et al., "Beamforming Analysis and Design for Wideband THz RIS
    Communications", IEEE JSAC, 2023.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SPEED_OF_LIGHT = 3.0e8  # m/s
BOLTZMANN_K = 1.380649e-23  # J/K


@dataclass(slots=True)
class THzSystemConfig:
    # --- Users and jammer ---
    k_users: int = 4
    n_jammer_antennas: int = 2     # portable THz jammer (small array)

    # --- BS hybrid array (sub-connected architecture) ---
    n_bs_antennas: int = 256       # N_t (ULA)
    n_rf_chains: int = 16          # N_RF  (also = number of BS sub-arrays P)

    # --- RIS SPDP parameters (UPA: N1 x N2) ---
    n_ris_h: int = 64              # N1 (horizontal)
    n_ris_v: int = 64              # N2 (vertical)
    q_subarrays_h: int = 8         # Q1 (horizontal sub-arrays)
    q_subarrays_v: int = 8         # Q2 (vertical sub-arrays)
    phase_bits: int = 0            # 0 = continuous; 1,2,3 = low-resolution

    # --- OFDM wideband ---
    n_subcarriers: int = 128       # M
    bandwidth_hz: float = 10.0e9   # B (Hz)
    center_freq_hz: float = 100.0e9  # f_c (Hz)

    # --- Power and QoS ---
    pmax_dbm: float = 40.0         # THz needs higher Tx power than sub-6GHz
    sinr_min_db: float = 5.0       # Realistic THz target (10 dB too aggressive)
    # noise_dbm is now computed via thermal noise kTB (see noise_power_dbm property)
    noise_figure_db: float = 10.0   # Receiver noise figure (dB)
    temperature_kelvin: float = 290.0  # System temperature (K)

    # --- Near-field / far-field ---
    use_near_field: bool = True     # Auto-switch to near-field when Fresnel distance exceeded
    force_far_field: bool = False   # Override: always use far-field (for comparison)

    # --- THz propagation ---
    molecular_abs_coeff_db_per_m: float = 5.157e-4
    # From ITU-R P.676-10 at ~100 GHz, 15 C, 101325 Pa, 7.5 g/m^3
    direct_link_penalty_db: float = 20.0  # Extra NLoS attenuation for direct BS-UE link (dB)
    # At THz, wall penetration loss is 20-40 dB; RIS provides bypass path

    # --- Geometry (3D, metres) ---
    bs_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ris_position: tuple[float, float, float] = (0.0, 2.0, 0.0)
    ue_region_min: tuple[float, float, float] = (8.0, 1.0, 0.0)
    ue_region_max: tuple[float, float, float] = (12.0, 3.0, 0.0)
    jammer_region_min: tuple[float, float, float] = (15.0, 0.0, 0.0)
    jammer_region_max: tuple[float, float, float] = (25.0, 4.0, 0.0)

    # --- Jammer (portable THz device: limited power and antenna count) ---
    p_jammer_min_dbm: float = 5.0
    p_jammer_max_dbm: float = 20.0  # THz jammer can be high-power dedicated device
    jammer_nlos_penalty_db: float = 5.0  # THz jammer partial NLoS attenuation

    # --- Training speed trade-off ---
    subcarrier_stride: int = 1     # evaluate every N-th subcarrier in RL loops

    seed: int = 7

    # ---- derived properties ----
    @property
    def n_ris_total(self) -> int:
        return self.n_ris_h * self.n_ris_v

    @property
    def n_subarrays_total(self) -> int:
        return self.q_subarrays_h * self.q_subarrays_v

    @property
    def k1(self) -> int:
        """Elements per sub-array, horizontal."""
        return self.n_ris_h // self.q_subarrays_h

    @property
    def k2(self) -> int:
        """Elements per sub-array, vertical."""
        return self.n_ris_v // self.q_subarrays_v

    @property
    def antenna_spacing(self) -> float:
        """Half-wavelength at center frequency (m)."""
        return SPEED_OF_LIGHT / (2.0 * self.center_freq_hz)

    @property
    def subcarrier_frequencies(self) -> np.ndarray:
        """Subcarrier frequencies f_m for m=0..M-1 (Eq. 1 of Su et al.)."""
        M = self.n_subcarriers
        m = np.arange(M)
        return self.center_freq_hz + (self.bandwidth_hz / M) * (m - (M - 1) / 2.0)

    @property
    def n_bs_per_subarray(self) -> int:
        """Antennas per BS sub-array."""
        return self.n_bs_antennas // self.n_rf_chains

    @property
    def noise_power_dbm(self) -> float:
        """Thermal noise power per subcarrier in dBm.
        
        Uses kTB formula: P_noise = k * T * B_sc + NF
        where B_sc = bandwidth_hz / n_subcarriers is per-subcarrier bandwidth.
        """
        import math
        b_subcarrier = self.bandwidth_hz / self.n_subcarriers
        thermal_noise_watt = BOLTZMANN_K * self.temperature_kelvin * b_subcarrier
        thermal_noise_dbm = 10.0 * math.log10(thermal_noise_watt) + 30.0
        return thermal_noise_dbm + self.noise_figure_db

    @property
    def total_noise_power_dbm(self) -> float:
        """Total thermal noise power across all subcarriers in dBm."""
        import math
        thermal_noise_watt = BOLTZMANN_K * self.temperature_kelvin * self.bandwidth_hz
        thermal_noise_dbm = 10.0 * math.log10(thermal_noise_watt) + 30.0
        return thermal_noise_dbm + self.noise_figure_db

    @property
    def wavelength(self) -> float:
        """Wavelength at center frequency (m)."""
        return SPEED_OF_LIGHT / self.center_freq_hz

    @property
    def ris_aperture(self) -> float:
        """RIS physical aperture size (m), assuming half-wavelength spacing."""
        max_dim = max(self.n_ris_h, self.n_ris_v)
        return max_dim * self.antenna_spacing

    @property
    def fresnel_distance(self) -> float:
        """Fresnel distance (Fraunhofer boundary) in metres.
        
        d_F = 2 * D^2 / lambda
        where D is the largest array dimension.
        """
        D = self.ris_aperture
        return 2.0 * D**2 / self.wavelength

    def is_near_field(self, distance: float) -> bool:
        """Check if a given distance is in the near-field region."""
        return distance < self.fresnel_distance


@dataclass(slots=True)
class THzRLConfig:
    alpha: float = 0.01
    gamma: float = 0.9

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.98      # Faster decay → exploit learned policy sooner

    lambda1: float = 0.5   # power penalty
    lambda2: float = 1.5   # QoS penalty (continuous SINR-deficit based)

    xi_win: float = 0.02
    xi_loss: float = 0.08

    wolf_eval_temperature: float = 15.0  # Boltzmann temperature for WoLF-PHC eval softmax

    state_bins: int = 8
    fuzzy_centers: tuple[float, float, float] = (0.0, 0.5, 1.0)

    # DQN-specific
    dqn_hidden1: int = 256
    dqn_hidden2: int = 128
    dqn_replay_size: int = 10_000
    dqn_batch_size: int = 64
    dqn_target_tau: float = 0.005
    dqn_lr: float = 1e-3


@dataclass(slots=True)
class THzTrainEvalConfig:
    train_episodes: int = 1200
    train_steps_per_episode: int = 50
    eval_episodes: int = 50
    eval_steps_per_episode: int = 20
    n_seeds: int = 3


@dataclass(slots=True)
class THzSweepConfig:
    pmax_dbm_values: list[float] = field(
        default_factory=lambda: [20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
    )
    n_ris_elements_values: list[int] = field(
        default_factory=lambda: [16, 64, 256, 1024]
    )
    bandwidth_values_hz: list[float] = field(
        default_factory=lambda: [0.1e9, 1.0e9, 5.0e9, 10.0e9]
    )
    q_subarray_values: list[int] = field(
        default_factory=lambda: [1, 16, 64]
    )
    sinr_target_db_values: list[float] = field(
        default_factory=lambda: [0.0, 5.0, 10.0, 15.0, 20.0]
    )
