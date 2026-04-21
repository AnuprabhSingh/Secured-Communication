#!/usr/bin/env python3
"""Validate THz RIS anti-jamming physics implementation.

This script runs diagnostic tests to verify:
1. RIS scaling: SINR increases with N (16 → 4096 elements)
2. Beam squint: Rate degrades with bandwidth for classical RIS
3. Near-field effects: Active for large RIS arrays
4. Noise power: Proper kTB thermal noise calculation
5. Averaging: Consistent, monotonic results across parameters
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from irs_anti_jamming.thz.thz_config import THzSystemConfig, BOLTZMANN_K
from irs_anti_jamming.thz.thz_channel_model import THzChannelModel, THzTopology
from irs_anti_jamming.thz.spdp_ris import compute_spdp_closed_form, classical_phase_only
from irs_anti_jamming.thz.thz_system_model import (
    evaluate_thz_system,
    verify_ris_scaling,
    analyze_beam_squint,
    compute_wideband_capacity,
)
from irs_anti_jamming.utils import dbm_to_watt


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_noise_power() -> bool:
    """Verify thermal noise calculation follows kTB formula."""
    print_header("TEST 1: Noise Power (kTB)")
    
    cfg = THzSystemConfig()
    
    # Expected thermal noise per subcarrier
    b_sc = cfg.bandwidth_hz / cfg.n_subcarriers
    expected_thermal_w = BOLTZMANN_K * cfg.temperature_kelvin * b_sc
    expected_thermal_dbm = 10 * np.log10(expected_thermal_w) + 30
    expected_noise_dbm = expected_thermal_dbm + cfg.noise_figure_db
    
    actual_noise_dbm = cfg.noise_power_dbm
    
    print(f"  Bandwidth: {cfg.bandwidth_hz/1e9:.1f} GHz")
    print(f"  Subcarriers: {cfg.n_subcarriers}")
    print(f"  Per-SC bandwidth: {b_sc/1e6:.2f} MHz")
    print(f"  Temperature: {cfg.temperature_kelvin} K")
    print(f"  Noise Figure: {cfg.noise_figure_db} dB")
    print(f"  Expected thermal noise (per SC): {expected_thermal_dbm:.2f} dBm")
    print(f"  Expected total noise (per SC): {expected_noise_dbm:.2f} dBm")
    print(f"  Actual noise_power_dbm: {actual_noise_dbm:.2f} dBm")
    
    passed = abs(actual_noise_dbm - expected_noise_dbm) < 0.1
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_ris_scaling() -> bool:
    """Verify SINR increases with RIS size (N^2 scaling expected)."""
    print_header("TEST 2: RIS Array Gain Scaling")
    
    n_values = [16, 64, 256, 1024, 4096]
    results = []
    
    # Single-user test for clean N² scaling (no MUI)
    # Use classical RIS (no SPDP TD) for narrowband to isolate pure array gain
    base_cfg_single = THzSystemConfig(
        k_users=1,  # Single user to isolate RIS gain
        pmax_dbm=30.0,
        bandwidth_hz=0.1e9,  # Very narrow band - no beam squint
        force_far_field=True,  # Use far-field for clean scaling test
    )
    
    print(f"  Testing N = {n_values}")
    print(f"  Power: {base_cfg_single.pmax_dbm} dBm, Bandwidth: {base_cfg_single.bandwidth_hz/1e9:.2f} GHz")
    print(f"  Single-user, narrowband test with classical RIS (no TD)")
    print()
    
    channel_powers = []
    sinr_values = []
    
    for N_total in n_values:
        side = int(np.sqrt(N_total))
        cfg = replace(base_cfg_single, n_ris_h=side, n_ris_v=side,
                     q_subarrays_h=1, q_subarrays_v=1)  # Single sub-array = classical
        
        # Average over multiple realizations for stability
        trial_sinrs = []
        trial_powers = []
        
        for trial in range(5):
            rng_trial = np.random.default_rng(42 + trial)
            topology = THzTopology(cfg, rng_trial)
            channel_model = THzChannelModel(cfg, rng_trial)
            snapshot = channel_model.sample(topology)
            
            # Use classical phase-only RIS (cleaner for testing array gain)
            from irs_anti_jamming.thz.spdp_ris import classical_phase_only
            spdp = classical_phase_only(
                cfg,
                snapshot.theta_aoa, snapshot.zeta_aoa,
                float(snapshot.theta_aod[0]), float(snapshot.zeta_aod[0])
            )
            
            p_bs = np.array([dbm_to_watt(cfg.pmax_dbm)])
            p_jam = np.zeros(1)
            z_jam = np.zeros((1, cfg.n_jammer_antennas))
            noise_w = dbm_to_watt(cfg.noise_power_dbm)
            
            metrics = evaluate_thz_system(
                cfg, snapshot, spdp, p_bs, p_jam, z_jam, noise_w,
                sinr_min_db=10.0, lambda1=0.0, lambda2=0.0
            )
            
            trial_sinrs.append(np.mean(metrics.sinr_linear))
            
            scaling_info = verify_ris_scaling(cfg, snapshot, spdp, user_idx=0)
            # Use RIS path power for scaling verification (direct path is constant)
            trial_powers.append(scaling_info['ris_channel_power'])
        
        avg_sinr = np.mean(trial_sinrs)
        avg_power = np.mean(trial_powers)
        channel_powers.append(avg_power)
        sinr_values.append(avg_sinr)
        
        avg_sinr_db = 10 * np.log10(avg_sinr + 1e-30)
        results.append((N_total, avg_sinr_db, metrics.system_rate))
        
        print(f"  N={N_total:5d}: SINR={avg_sinr_db:6.2f} dB, "
              f"|h_ris|²={avg_power:.2e}")
    
    # Check monotonicity
    sinrs_db = [r[1] for r in results]
    monotonic = all(sinrs_db[i] < sinrs_db[i+1] for i in range(len(sinrs_db)-1))
    
    # Check scaling by fitting
    # For N² scaling: |h|² ∝ N² → log10(|h|²) = 2*log10(N) + const
    log_n = np.log10(n_values)
    log_power = np.log10(channel_powers)
    slope, intercept = np.polyfit(log_n, log_power, 1)
    
    # Also check SINR scaling (should be close to channel power scaling since no interference)
    log_sinr = np.log10(sinr_values)
    sinr_slope, _ = np.polyfit(log_n, log_sinr, 1)
    
    gain_16_to_4096 = sinrs_db[-1] - sinrs_db[0]
    
    print()
    print(f"  Monotonic: {monotonic}")
    print(f"  Total SINR gain (N=16→4096): {gain_16_to_4096:.1f} dB")
    print(f"  Channel power scaling slope: {slope:.2f} (expect ~2.0 for N²)")
    print(f"  SINR scaling slope: {sinr_slope:.2f}")
    
    # Pass if slope > 1.5 (between N and N² scaling) and SINR gain > 15 dB
    passed = monotonic and slope > 1.5 and gain_16_to_4096 > 15
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_beam_squint() -> bool:
    """Verify beam squint effect: rate decreases with bandwidth."""
    print_header("TEST 3: Beam Squint / Bandwidth Effect")
    
    bandwidths_ghz = [0.1, 1.0, 5.0, 10.0, 20.0]
    results_spdp = []
    results_classical = []
    
    rng = np.random.default_rng(42)
    base_cfg = THzSystemConfig(n_ris_h=32, n_ris_v=32)  # 1024 elements
    
    print(f"  Testing bandwidths: {bandwidths_ghz} GHz")
    print(f"  RIS size: {base_cfg.n_ris_total} elements")
    print()
    
    for bw_ghz in bandwidths_ghz:
        cfg = replace(base_cfg, bandwidth_hz=bw_ghz * 1e9)
        
        topology = THzTopology(cfg, rng)
        channel_model = THzChannelModel(cfg, rng)
        snapshot = channel_model.sample(topology)
        
        # SPDP (beam-squint compensated)
        spdp = compute_spdp_closed_form(
            cfg, snapshot.theta_aoa, snapshot.zeta_aoa,
            float(snapshot.theta_aod[0]), float(snapshot.zeta_aod[0])
        )
        
        # Classical (no TD compensation)
        classical = classical_phase_only(
            cfg, snapshot.theta_aoa, snapshot.zeta_aoa,
            float(snapshot.theta_aod[0]), float(snapshot.zeta_aod[0])
        )
        
        p_bs = np.full(cfg.k_users, dbm_to_watt(cfg.pmax_dbm) / cfg.k_users)
        p_jam = np.zeros(cfg.k_users)
        z_jam = np.zeros((cfg.k_users, cfg.n_jammer_antennas))
        noise_w = dbm_to_watt(cfg.noise_power_dbm)
        
        # Evaluate SPDP
        metrics_spdp = evaluate_thz_system(
            cfg, snapshot, spdp, p_bs, p_jam, z_jam, noise_w,
            sinr_min_db=10.0, lambda1=0.0, lambda2=0.0
        )
        
        # Evaluate Classical
        metrics_classical = evaluate_thz_system(
            cfg, snapshot, classical, p_bs, p_jam, z_jam, noise_w,
            sinr_min_db=10.0, lambda1=0.0, lambda2=0.0
        )
        
        squint_info = analyze_beam_squint(cfg, snapshot, spdp, user_idx=0)
        
        results_spdp.append((bw_ghz, metrics_spdp.system_rate))
        results_classical.append((bw_ghz, metrics_classical.system_rate))
        
        print(f"  B={bw_ghz:5.1f} GHz: SPDP={metrics_spdp.system_rate:.3f}, "
              f"Classical={metrics_classical.system_rate:.3f}, "
              f"Edge/Center={squint_info['edge_to_center_ratio_low']:.3f}")
    
    # Check that classical degrades more than SPDP with increasing bandwidth
    spdp_rates = [r[1] for r in results_spdp]
    classical_rates = [r[1] for r in results_classical]
    
    spdp_degradation = spdp_rates[0] - spdp_rates[-1]  # 0.1 GHz vs 20 GHz
    classical_degradation = classical_rates[0] - classical_rates[-1]
    
    print()
    print(f"  SPDP rate degradation (0.1→20 GHz): {spdp_degradation:.3f}")
    print(f"  Classical rate degradation: {classical_degradation:.3f}")
    print(f"  SPDP should degrade less than Classical")
    
    # SPDP should maintain better performance at wide bandwidths
    passed = classical_degradation > spdp_degradation * 0.8  # Classical degrades more
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_near_field() -> bool:
    """Verify near-field model activates for large RIS."""
    print_header("TEST 4: Near-Field Model")
    
    base_cfg = THzSystemConfig()
    
    print(f"  Center frequency: {base_cfg.center_freq_hz/1e9:.0f} GHz")
    print(f"  Wavelength: {base_cfg.wavelength*1e3:.3f} mm")
    print()
    
    test_cases = [
        (4, 4),      # 16 elements
        (16, 16),    # 256 elements
        (32, 32),    # 1024 elements
        (64, 64),    # 4096 elements
        (128, 128),  # 16384 elements
    ]
    
    for n1, n2 in test_cases:
        cfg = replace(base_cfg, n_ris_h=n1, n_ris_v=n2)
        d_fresnel = cfg.fresnel_distance
        d_bs_ris = np.linalg.norm(
            np.array(cfg.bs_position) - np.array(cfg.ris_position)
        )
        
        is_nf = cfg.is_near_field(d_bs_ris)
        status = "NEAR-FIELD" if is_nf else "far-field"
        
        print(f"  N={n1*n2:5d}: Fresnel dist={d_fresnel:.3f} m, "
              f"BS-RIS={d_bs_ris:.2f} m → {status}")
    
    # Large arrays should trigger near-field
    cfg_large = replace(base_cfg, n_ris_h=64, n_ris_v=64)
    d_bs_ris = np.linalg.norm(
        np.array(cfg_large.bs_position) - np.array(cfg_large.ris_position)
    )
    
    passed = cfg_large.is_near_field(d_bs_ris)
    print()
    print(f"  4096-element RIS triggers near-field: {passed}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_monotonic_averaging() -> bool:
    """Verify results are monotonic with proper averaging."""
    print_header("TEST 5: Monotonic Results with Averaging")
    
    powers_dbm = [10, 15, 20, 25, 30, 35, 40]
    n_trials = 5
    
    base_cfg = THzSystemConfig(n_ris_h=16, n_ris_v=16)
    
    print(f"  Testing powers: {powers_dbm} dBm")
    print(f"  Averaging over {n_trials} trials")
    print()
    
    results = []
    
    for p_dbm in powers_dbm:
        cfg = replace(base_cfg, pmax_dbm=float(p_dbm))
        
        trial_rates = []
        for trial in range(n_trials):
            rng = np.random.default_rng(42 + trial)
            topology = THzTopology(cfg, rng)
            channel_model = THzChannelModel(cfg, rng)
            snapshot = channel_model.sample(topology)
            
            spdp = compute_spdp_closed_form(
                cfg, snapshot.theta_aoa, snapshot.zeta_aoa,
                float(snapshot.theta_aod[0]), float(snapshot.zeta_aod[0])
            )
            
            p_bs = np.full(cfg.k_users, dbm_to_watt(cfg.pmax_dbm) / cfg.k_users)
            p_jam = np.zeros(cfg.k_users)
            z_jam = np.zeros((cfg.k_users, cfg.n_jammer_antennas))
            noise_w = dbm_to_watt(cfg.noise_power_dbm)
            
            metrics = evaluate_thz_system(
                cfg, snapshot, spdp, p_bs, p_jam, z_jam, noise_w,
                sinr_min_db=10.0, lambda1=0.0, lambda2=0.0
            )
            trial_rates.append(metrics.system_rate)
        
        avg_rate = np.mean(trial_rates)
        std_rate = np.std(trial_rates)
        results.append((p_dbm, avg_rate, std_rate))
        
        print(f"  P={p_dbm:2d} dBm: Rate={avg_rate:.3f} ± {std_rate:.3f}")
    
    # Check monotonicity
    rates = [r[1] for r in results]
    monotonic = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    
    print()
    print(f"  Monotonic: {monotonic}")
    print(f"  RESULT: {'PASS' if monotonic else 'FAIL'}")
    return monotonic


def main():
    print("\n" + "=" * 60)
    print("  THz RIS Anti-Jamming Physics Validation")
    print("=" * 60)
    
    tests = [
        ("Noise Power (kTB)", test_noise_power),
        ("RIS Array Gain Scaling", test_ris_scaling),
        ("Beam Squint / Bandwidth", test_beam_squint),
        ("Near-Field Model", test_near_field),
        ("Monotonic Averaging", test_monotonic_averaging),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ERROR: {e}")
    
    print_header("SUMMARY")
    
    for name, passed, error in results:
        status = "PASS" if passed else ("ERROR" if error else "FAIL")
        print(f"  {name}: {status}")
        if error:
            print(f"    → {error}")
    
    n_passed = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    print()
    print(f"  Total: {n_passed}/{n_total} tests passed")
    
    return 0 if n_passed == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
