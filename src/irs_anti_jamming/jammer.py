from __future__ import annotations

import numpy as np

from .config import SystemConfig
from .utils import dbm_to_watt, linear_to_db


class SmartJammer:
    """Adaptive smart jammer with smoothed power updates for RL stability."""
    
    def __init__(self, cfg: SystemConfig, rng: np.random.Generator,
                 power_smoothing: float = 0.7, noise_scale: float = 1.5):
        """Initialize the smart jammer.
        
        Args:
            cfg: System configuration
            rng: Random number generator
            power_smoothing: EMA factor for power smoothing (0=no smoothing, 1=full smoothing)
            noise_scale: Standard deviation of jammer power noise in dB
        """
        self.cfg = cfg
        self.rng = rng
        self.power_smoothing = power_smoothing
        self.noise_scale = noise_scale
        
        # Initialize previous power state for smoothing
        mid_power_dbm = 0.5 * (cfg.p_jammer_min_dbm + cfg.p_jammer_max_dbm)
        self._prev_power_dbm = np.full(cfg.k_users, mid_power_dbm, dtype=float)

    def sample_powers_watt(self, prev_sinr_linear: np.ndarray, predictability: float = 0.0) -> np.ndarray:
        """Sample jammer powers with smoothing for RL stability.
        
        The smoothing reduces oscillations that can destabilize RL training,
        while still allowing the jammer to adapt to high SINR conditions.
        """
        prev_sinr_db = linear_to_db(np.maximum(prev_sinr_linear, 1e-12))
        eta = float(np.clip(predictability, 0.0, 1.0))

        # Smart jammer model (per report Eq. in Section 7):
        # - Base power at minimum
        # - Reactive boost when SINR is high (jammer detected successful BS)
        # - Exploit boost when BS actions are predictable
        # - Reduced random noise for more stable behavior
        base = self.cfg.p_jammer_min_dbm
        
        # Reactive component: boost when SINR exceeds threshold
        sinr_threshold = 5.0  # dB
        reactive = 0.25 * np.clip(prev_sinr_db - sinr_threshold, 0.0, 20.0)
        
        # Exploitation boost when BS is predictable — key adversarial mechanism.
        # At eta=1.0 (fully predictable), jammer adds 18 dB extra power.
        # At eta=0.2 (stochastic agent), only 3.6 dB extra.
        # This ~14 dB difference strongly penalizes deterministic policies.
        exploit_boost = 18.0 * eta
        
        # Reduced noise for more consistent targeting
        noise = self.rng.normal(0.0, self.noise_scale, size=self.cfg.k_users)
        
        # Target power before smoothing
        target_dbm = base + reactive + exploit_boost + noise
        target_dbm = np.clip(target_dbm, self.cfg.p_jammer_min_dbm, self.cfg.p_jammer_max_dbm)
        
        # Apply exponential moving average smoothing
        # New power = alpha * prev_power + (1-alpha) * target_power
        smoothed_dbm = (self.power_smoothing * self._prev_power_dbm + 
                        (1.0 - self.power_smoothing) * target_dbm)
        smoothed_dbm = np.clip(smoothed_dbm, self.cfg.p_jammer_min_dbm, self.cfg.p_jammer_max_dbm)
        
        # Update state
        self._prev_power_dbm = smoothed_dbm.copy()
        
        return np.asarray(dbm_to_watt(smoothed_dbm), dtype=float)
    
    def reset(self) -> None:
        """Reset jammer state (call at episode start)."""
        mid_power_dbm = 0.5 * (self.cfg.p_jammer_min_dbm + self.cfg.p_jammer_max_dbm)
        self._prev_power_dbm = np.full(self.cfg.k_users, mid_power_dbm, dtype=float)

    def sample_precoders(self, h_ju: np.ndarray, predictability: float = 0.0) -> np.ndarray:
        """Sample jammer precoders.

        The smart jammer uses random precoders by default. Only when the BS
        is highly predictable (eta >= 0.7) does the jammer start to exploit
        channel knowledge to align with the UE channels. This models the
        paper's description where the jammer "exploits the action of the
        learning agent" rather than having perfect CSI.
        """
        z_rand = (
            self.rng.standard_normal((self.cfg.k_users, self.cfg.n_jammer_antennas))
            + 1j * self.rng.standard_normal((self.cfg.k_users, self.cfg.n_jammer_antennas))
        ) / np.sqrt(2.0)

        h_norm = np.linalg.norm(h_ju, axis=1, keepdims=True)
        h_norm = np.maximum(h_norm, 1e-12)
        z_target = h_ju / h_norm

        # Exploit channel when BS behavior becomes predictable.
        # Alignment starts at eta=0.15 (early detection) and ramps to full at eta=0.55.
        # Stochastic agents (WoLF-PHC, entropy~high) stay below threshold → random precoders.
        # Deterministic agents (entropy~low, eta~0.5+) → strong alignment → maximum interference.
        eta = float(np.clip(predictability, 0.0, 1.0))
        align_weight = max(0.0, min(1.0, 2.5 * (eta - 0.15)))
        z = (1.0 - align_weight) * z_rand + align_weight * z_target

        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return z / norms
