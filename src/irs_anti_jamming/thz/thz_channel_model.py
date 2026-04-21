"""Wideband THz channel model with molecular absorption and beam squint.

Implements the LoS-dominant ray-based channel model for RIS-aided THz
communications, following the formulation from:
  Su et al., "Wideband Precoding for RIS-Aided THz Communications",
  IEEE Trans. Commun., 2023.  (Eqs. 1, 5-10)

Key THz-specific features vs. the narrowband sub-6GHz model:
  - Per-subcarrier channels G_m, h_{k,m} (frequency-selective)
  - Molecular absorption: exp(-0.5 * tau_abs * d)
  - Beam squint: steering vectors depend on f_m / f_c
  - LoS-only propagation (THz quasi-optical)
  - UPA (2D) at RIS instead of ULA (1D)
  - Near-field model for large RIS arrays (d < 2D²/λ)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .thz_config import SPEED_OF_LIGHT, THzSystemConfig
from ..utils import complex_normal

EPS = 1e-30


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _distance_3d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _angles_3d(from_xyz: np.ndarray, to_xyz: np.ndarray) -> tuple[float, float]:
    """Compute elevation theta and azimuth zeta from *from_xyz* toward *to_xyz*.

    Returns (theta, zeta) where:
      theta = elevation angle from z-axis (0..pi)
      zeta  = azimuth angle in xy-plane (-pi..pi)

    For a 2D-only scenario (z=0 everywhere) theta = pi/2 always.
    """
    vec = np.asarray(to_xyz, dtype=float) - np.asarray(from_xyz, dtype=float)
    d_xy = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    theta = math.atan2(d_xy, vec[2]) if abs(vec[2]) > EPS else math.pi / 2.0
    zeta = math.atan2(vec[1], vec[0])
    return theta, zeta


# ---------------------------------------------------------------------------
# THz propagation
# ---------------------------------------------------------------------------

def thz_path_gain(freqs: np.ndarray, distance: float,
                  tau_abs_db_per_m: float) -> np.ndarray:
    """Complex-magnitude path gain at each subcarrier.

    g(f_m, d) = c / (4 pi f_m d) * exp(-0.5 * tau_abs * d)   (Su Eq. 5)

    Args:
        freqs: subcarrier frequencies (M,)
        distance: propagation distance in metres
        tau_abs_db_per_m: molecular absorption coefficient in dB/m

    Returns:
        Real-valued gain array of shape (M,).
    """
    d = max(distance, 1e-3)  # clamp to 1 mm
    # Convert dB/m to Neper/m:  1 dB = ln(10)/10 Np
    tau_np = tau_abs_db_per_m * (math.log(10.0) / 10.0)
    fspl = SPEED_OF_LIGHT / (4.0 * math.pi * freqs * d)
    absorption = np.exp(-0.5 * tau_np * d)
    return fspl * absorption  # shape (M,)


# ---------------------------------------------------------------------------
# Steering vectors  (frequency-dependent → beam squint)
# ---------------------------------------------------------------------------

def ula_steering_thz(freqs: np.ndarray, n_elements: int,
                     angle_rad: float, d_spacing: float) -> np.ndarray:
    """Frequency-dependent ULA steering vectors (BS), unnormalized.

    b(f_m, phi) = [1, exp(j 2pi f_m/c d sin(phi)), ..., exp(j 2pi f_m/c (N-1) d sin(phi))]

    Unnormalized so that the physical array gain (N_t for coherent Tx)
    is preserved in the channel model.  Beamforming vector ||w||=1
    provides the correct power normalization separately.

    Args:
        freqs: (M,) subcarrier frequencies
        n_elements: number of antenna elements N_t
        angle_rad: departure/arrival angle phi
        d_spacing: antenna spacing (m)

    Returns:
        (M, N_t) complex steering matrix, one row per subcarrier.
    """
    n = np.arange(n_elements, dtype=float)  # (N_t,)
    base_phase = (2.0 * math.pi * d_spacing * math.sin(angle_rad) / SPEED_OF_LIGHT)
    phase = np.outer(freqs, n) * base_phase  # (M, N_t)
    # No 1/sqrt(N_t) normalization — keep physical array response so that
    # beamforming gains (N_t from BS, N from RIS) are preserved in the
    # effective channel.  Power normalization is done by the beamformer
    # (||w_k||=1), matching the narrowband convention.
    return np.exp(1j * phase)


def upa_steering_thz(freqs: np.ndarray, N1: int, N2: int,
                     theta: float, zeta: float,
                     d_spacing: float) -> np.ndarray:
    """Frequency-dependent UPA steering vectors (RIS).

    a(f_m, theta, zeta) = kron(a_h, a_v) / sqrt(N)   (Su Eq. 7)

    where a_h has N1 elements in horizontal with spatial freq sin(theta)cos(zeta),
          a_v has N2 elements in vertical with spatial freq sin(theta)sin(zeta).

    Args:
        freqs: (M,) subcarrier frequencies
        N1, N2: UPA dimensions
        theta: elevation angle
        zeta: azimuth angle
        d_spacing: element spacing (m)

    Returns:
        (M, N1*N2) complex steering matrix.
    """
    M = freqs.shape[0]
    N = N1 * N2

    alpha = math.sin(theta) * math.cos(zeta)  # horizontal spatial freq
    beta = math.sin(theta) * math.sin(zeta)   # vertical spatial freq

    n1 = np.arange(N1, dtype=float)  # (N1,)
    n2 = np.arange(N2, dtype=float)  # (N2,)

    # Phase per frequency: (2pi/c) * f_m * d * n * spatial_freq
    base_h = (2.0 * math.pi * d_spacing * alpha / SPEED_OF_LIGHT)  # scalar
    base_v = (2.0 * math.pi * d_spacing * beta / SPEED_OF_LIGHT)

    # a_h: (M, N1)
    phase_h = np.outer(freqs, n1) * base_h
    a_h = np.exp(1j * phase_h)

    # a_v: (M, N2)
    phase_v = np.outer(freqs, n2) * base_v
    a_v = np.exp(1j * phase_v)

    # Kronecker product per subcarrier: (M, N1*N2)
    # a[m] = kron(a_h[m], a_v[m])
    # Efficient: a_h[:, :, None] * a_v[:, None, :] then reshape
    result = (a_h[:, :, np.newaxis] * a_v[:, np.newaxis, :]).reshape(M, N)
    # No 1/sqrt(N) normalization — keep physical array response.
    # RIS coherent combining gain (N in power) is preserved naturally.
    return result


# ---------------------------------------------------------------------------
# Near-field steering vectors for large RIS arrays
# ---------------------------------------------------------------------------

def upa_steering_near_field(
    freqs: np.ndarray,
    N1: int, N2: int,
    source_xyz: np.ndarray,
    ris_center_xyz: np.ndarray,
    d_spacing: float,
) -> np.ndarray:
    """Near-field UPA steering vectors accounting for per-element distances.

    In the near-field (Fresnel region), the plane-wave assumption breaks down.
    Each RIS element sees a different path length to the source, causing
    spherical wavefront curvature.

    r_n = ||source - element_n||
    a_n(f_m) = exp(-j 2pi f_m r_n / c) / r_n

    Args:
        freqs: (M,) subcarrier frequencies
        N1, N2: UPA dimensions (horizontal, vertical)
        source_xyz: (3,) source position
        ris_center_xyz: (3,) RIS center position
        d_spacing: element spacing (m)

    Returns:
        (M, N1*N2) complex steering matrix with per-element path loss and phase.
    """
    M = freqs.shape[0]
    N = N1 * N2

    # Generate element positions relative to RIS center
    # Elements indexed (n1, n2) for n1 in [0,N1), n2 in [0,N2)
    # Position offset: ((n1 - (N1-1)/2) * d, (n2 - (N2-1)/2) * d, 0)
    n1_idx = np.arange(N1, dtype=float) - (N1 - 1) / 2.0
    n2_idx = np.arange(N2, dtype=float) - (N2 - 1) / 2.0
    n1_grid, n2_grid = np.meshgrid(n1_idx, n2_idx, indexing="ij")

    # Element positions (N, 3) - assuming RIS in xy-plane, elements along x and y
    element_offsets = np.zeros((N, 3), dtype=float)
    element_offsets[:, 0] = (n1_grid * d_spacing).ravel()  # x-offset
    element_offsets[:, 1] = (n2_grid * d_spacing).ravel()  # y-offset

    element_xyz = ris_center_xyz + element_offsets  # (N, 3)

    # Distance from source to each element
    diff = source_xyz - element_xyz  # (N, 3)
    distances = np.linalg.norm(diff, axis=1)  # (N,)
    distances = np.maximum(distances, 1e-6)  # avoid division by zero

    # Near-field steering: per-element phase and amplitude
    # a_n(f_m) = (1/r_n) * exp(-j 2pi f_m r_n / c)
    # The 1/r_n gives proper near-field amplitude variation (spherical spreading)
    result = np.zeros((M, N), dtype=np.complex128)
    for m, f_m in enumerate(freqs):
        phase = -2.0 * math.pi * f_m * distances / SPEED_OF_LIGHT
        result[m, :] = np.exp(1j * phase) / distances

    # Normalize by reference distance (center element) to keep sensible magnitudes
    ref_dist = float(np.linalg.norm(source_xyz - ris_center_xyz))
    if ref_dist > EPS:
        result *= ref_dist

    return result


# ---------------------------------------------------------------------------
# THz Topology
# ---------------------------------------------------------------------------

class THzTopology:
    """3D topology for THz system: BS, RIS, users, jammer."""

    def __init__(self, cfg: THzSystemConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.bs_xyz = np.asarray(cfg.bs_position, dtype=float)
        self.ris_xyz = np.asarray(cfg.ris_position, dtype=float)
        self.ue_xyz = self._sample_points(cfg.ue_region_min, cfg.ue_region_max, cfg.k_users)
        self.jammer_xyz = self._sample_points(cfg.jammer_region_min, cfg.jammer_region_max, 1)[0]

    def _sample_points(self, region_min: tuple, region_max: tuple, count: int) -> np.ndarray:
        lo = np.asarray(region_min, dtype=float)
        hi = np.asarray(region_max, dtype=float)
        points = np.column_stack([
            self.rng.uniform(lo[i], max(hi[i], lo[i] + 1e-6), size=count)
            for i in range(3)
        ])
        return points

    def resample_users(self) -> None:
        self.ue_xyz = self._sample_points(
            self.cfg.ue_region_min, self.cfg.ue_region_max, self.cfg.k_users
        )

    def move_jammer(self) -> None:
        step = self.rng.normal(0.0, 0.5, size=3)  # smaller steps for THz (closer distances)
        lo = np.asarray(self.cfg.jammer_region_min, dtype=float)
        hi = np.asarray(self.cfg.jammer_region_max, dtype=float)
        self.jammer_xyz = np.clip(self.jammer_xyz + step, lo, hi)

    # precomputed angles
    def angle_bs_to_ris(self) -> float:
        """Azimuth angle at BS toward RIS (for ULA)."""
        vec = self.ris_xyz - self.bs_xyz
        return float(np.arctan2(vec[1], vec[0]))

    def angles_ris_aoa(self) -> tuple[float, float]:
        """(theta, zeta) of AoA at RIS from BS direction."""
        return _angles_3d(self.ris_xyz, self.bs_xyz)

    def angles_ris_aod(self, user_xyz: np.ndarray) -> tuple[float, float]:
        """(theta, zeta) of AoD at RIS toward a user."""
        return _angles_3d(self.ris_xyz, user_xyz)


# ---------------------------------------------------------------------------
# Channel snapshot
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class THzChannelSnapshot:
    """Per-subcarrier channel matrices for one time slot."""
    G: np.ndarray           # (M, N, N_t) BS-RIS per subcarrier
    h_ru: np.ndarray        # (M, K, N)   RIS-UE per subcarrier per user
    h_bu: np.ndarray        # (M, K, N_t) direct BS-UE per subcarrier (NLoS)
    h_ju: np.ndarray        # (K, N_j)    jammer-UE (frequency-flat)
    freqs: np.ndarray       # (M,)        subcarrier frequencies
    phi_bs: float           # BS transmit angle toward RIS
    theta_aoa: float        # RIS AoA elevation (from BS)
    zeta_aoa: float         # RIS AoA azimuth
    theta_aod: np.ndarray   # (K,) RIS AoD elevation per user
    zeta_aod: np.ndarray    # (K,) RIS AoD azimuth per user


# ---------------------------------------------------------------------------
# Channel model
# ---------------------------------------------------------------------------

class THzChannelModel:
    """Generates wideband THz channel snapshots.

    BS-RIS and RIS-UE links are LoS-only (THz quasi-optical propagation).
    A weak direct BS-UE link (NLoS Rayleigh) provides per-user spatial
    diversity for multi-user interference suppression.
    Jammer-UE link is Rayleigh (no RIS assistance).
    
    Supports both far-field (plane-wave) and near-field (spherical-wave)
    channel models based on Fresnel distance.
    """

    def __init__(self, cfg: THzSystemConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def _use_near_field(self, distance: float) -> bool:
        """Determine whether to use near-field model for this distance."""
        if self.cfg.force_far_field:
            return False
        if not self.cfg.use_near_field:
            return False
        return self.cfg.is_near_field(distance)

    def sample(self, topology: THzTopology) -> THzChannelSnapshot:
        c = self.cfg
        freqs = c.subcarrier_frequencies  # (M,)
        M = c.n_subcarriers
        N = c.n_ris_total
        N_t = c.n_bs_antennas
        K = c.k_users
        d_ant = c.antenna_spacing

        # --- BS-RIS channel G_m (LoS only) ---
        d_br = _distance_3d(topology.bs_xyz, topology.ris_xyz)
        phi_bs = topology.angle_bs_to_ris()
        theta_aoa, zeta_aoa = topology.angles_ris_aoa()

        # Check near-field condition for BS-RIS link
        use_nf_br = self._use_near_field(d_br)
        
        if use_nf_br:
            # Near-field: spherical wavefront from BS to each RIS element
            a_ris = upa_steering_near_field(
                freqs, c.n_ris_h, c.n_ris_v,
                topology.bs_xyz, topology.ris_xyz, d_ant
            )
            # Path gain is already incorporated in near-field steering
            gain_br = np.ones(M)
            # Apply molecular absorption
            tau_np = c.molecular_abs_coeff_db_per_m * (math.log(10.0) / 10.0)
            absorption = math.exp(-0.5 * tau_np * d_br)
            a_ris *= absorption
        else:
            # Far-field: plane-wave approximation
            gain_br = thz_path_gain(freqs, d_br, c.molecular_abs_coeff_db_per_m)  # (M,)
            a_ris = upa_steering_thz(freqs, c.n_ris_h, c.n_ris_v, theta_aoa, zeta_aoa, d_ant)
        
        tau_br = d_br / SPEED_OF_LIGHT
        delay_phase = np.exp(-1j * 2.0 * math.pi * freqs * tau_br)  # (M,)

        # BS side ULA steering (always far-field since BS array is smaller)
        b_bs = ula_steering_thz(freqs, N_t, phi_bs, d_ant)

        G = np.zeros((M, N, N_t), dtype=np.complex128)
        for m in range(M):
            if use_nf_br:
                # Near-field: a_ris already has distance-dependent amplitude
                scalar = delay_phase[m]
            else:
                scalar = gain_br[m] * delay_phase[m]
            G[m] = scalar * np.outer(a_ris[m], b_bs[m].conj())

        # --- RIS-UE channels h_{k,m} (LoS only, per user) ---
        theta_aod = np.zeros(K)
        zeta_aod = np.zeros(K)
        h_ru = np.zeros((M, K, N), dtype=np.complex128)

        for k in range(K):
            d_ru_k = _distance_3d(topology.ris_xyz, topology.ue_xyz[k])
            theta_aod[k], zeta_aod[k] = topology.angles_ris_aod(topology.ue_xyz[k])
            
            tau_ru_k = d_ru_k / SPEED_OF_LIGHT
            delay_k = np.exp(-1j * 2.0 * math.pi * freqs * tau_ru_k)
            
            use_nf_ru = self._use_near_field(d_ru_k)
            
            if use_nf_ru:
                # Near-field RIS-UE channel
                a_ris_k = upa_steering_near_field(
                    freqs, c.n_ris_h, c.n_ris_v,
                    topology.ue_xyz[k], topology.ris_xyz, d_ant
                )
                tau_np = c.molecular_abs_coeff_db_per_m * (math.log(10.0) / 10.0)
                absorption = math.exp(-0.5 * tau_np * d_ru_k)
                for m in range(M):
                    h_ru[m, k, :] = delay_k[m] * a_ris_k[m] * absorption
            else:
                # Far-field RIS-UE channel
                gain_ru_k = thz_path_gain(freqs, d_ru_k, c.molecular_abs_coeff_db_per_m)
                a_ris_k = upa_steering_thz(freqs, c.n_ris_h, c.n_ris_v,
                                            theta_aod[k], zeta_aod[k], d_ant)
                for m in range(M):
                    h_ru[m, k, :] = gain_ru_k[m] * delay_k[m] * a_ris_k[m]

        # --- Direct BS-UE channel (NLoS Rayleigh with extra path loss) ---
        # Provides per-user spatial diversity for MUI suppression.
        # At THz, direct link is typically blocked/heavily attenuated (hence RIS).
        from ..utils import db_to_linear
        direct_atten = math.sqrt(float(db_to_linear(-c.direct_link_penalty_db)))
        h_bu = np.zeros((M, K, N_t), dtype=np.complex128)
        for k in range(K):
            d_bu_k = _distance_3d(topology.bs_xyz, topology.ue_xyz[k])
            gain_bu_k = thz_path_gain(freqs, d_bu_k, c.molecular_abs_coeff_db_per_m)
            for m in range(M):
                h_bu[m, k, :] = (direct_atten * gain_bu_k[m] *
                                 complex_normal((N_t,), self.rng).ravel())

        # --- Jammer-UE channel (Rayleigh, frequency-flat, THz path loss at f_c) ---
        # Portable THz jammer: NLoS penalty models partial blockage typical at THz.
        h_ju = np.zeros((K, c.n_jammer_antennas), dtype=np.complex128)
        f_c = c.center_freq_hz
        jammer_atten = math.sqrt(float(db_to_linear(-c.jammer_nlos_penalty_db)))
        for k in range(K):
            d_ju_k = _distance_3d(topology.jammer_xyz, topology.ue_xyz[k])
            gain_ju_k = SPEED_OF_LIGHT / (4.0 * math.pi * f_c * max(d_ju_k, 1e-3))
            tau_abs_np = c.molecular_abs_coeff_db_per_m * (math.log(10.0) / 10.0)
            gain_ju_k *= math.exp(-0.5 * tau_abs_np * d_ju_k)
            h_ju[k] = jammer_atten * gain_ju_k * complex_normal((c.n_jammer_antennas,), self.rng).ravel()

        return THzChannelSnapshot(
            G=G, h_ru=h_ru, h_bu=h_bu, h_ju=h_ju, freqs=freqs,
            phi_bs=phi_bs, theta_aoa=theta_aoa, zeta_aoa=zeta_aoa,
            theta_aod=theta_aod, zeta_aod=zeta_aod,
        )
