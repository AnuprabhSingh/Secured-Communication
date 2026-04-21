#!/usr/bin/env python3
"""Run parameter sweeps and collect journal-quality metrics.

Produces sweep_results.json with:
  - Real parameter sweep data for P_max, N_RIS, SINR_min, P_jammer
  - SINR CDF data from detailed evaluation
  - Runtime comparison per method
  - No-IRS baseline data

Usage:
    python scripts/run_journal_sweeps.py [--fast]

    --fast : Use fewer sweep points & shorter training (quick sanity check)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.irs_anti_jamming.thz.thz_config import (
    THzRLConfig,
    THzSystemConfig,
    THzTrainEvalConfig,
)
from src.irs_anti_jamming.thz.thz_experiments import (
    RL_METHODS,
    _build_agent,
    evaluate_thz_agent_detailed,
    evaluate_thz_ao_baseline_detailed,
    run_thz_parameter_sweep,
    train_thz_agent,
)

ALL_METHODS = RL_METHODS + ["baseline_ao"]


# ── Runtime measurement ──────────────────────────────────────────────

def measure_runtime(
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
    n_seeds: int = 2,
) -> dict[str, dict]:
    """Measure wall-clock training time per method."""
    results = {}

    for method in RL_METHODS:
        times = []
        for run_idx in range(n_seeds):
            seed = sys_cfg.seed + 77 * run_idx
            train_eps = run_cfg.train_episodes
            ft_eps = 0
            if method == "fuzzy_wolf_phc":
                train_eps = int(train_eps * 3.0)
                ft_eps = 80

            run_cfg_m = replace(run_cfg, train_episodes=train_eps)
            t0 = time.perf_counter()
            train_thz_agent(method, sys_cfg, rl_cfg, run_cfg_m, seed,
                            finetune_episodes=ft_eps)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"    {method} seed {run_idx}: {elapsed:.1f}s")

        results[method] = {
            "time_mean_s": float(np.mean(times)),
            "time_std_s": float(np.std(times)),
            "train_episodes": train_eps,
        }

    # AO baseline: no training, just measure eval time
    t0 = time.perf_counter()
    evaluate_thz_ao_baseline_detailed(sys_cfg, rl_cfg, run_cfg, sys_cfg.seed)
    elapsed = time.perf_counter() - t0
    results["baseline_ao"] = {
        "time_mean_s": elapsed,
        "time_std_s": 0.0,
        "train_episodes": 0,
    }

    return results


# ── SINR CDF collection ─────────────────────────────────────────────

def collect_sinr_cdf(
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
    n_seeds: int = 2,
) -> dict[str, list[float]]:
    """Train each method and collect per-user SINR samples for CDF."""
    cdf_data: dict[str, list[float]] = {}

    for method in RL_METHODS:
        all_sinr = []
        for run_idx in range(n_seeds):
            seed = sys_cfg.seed + 53 * run_idx
            train_eps = run_cfg.train_episodes
            ft_eps = 0
            if method == "fuzzy_wolf_phc":
                train_eps = int(train_eps * 3.0)
                ft_eps = 80
            run_cfg_m = replace(run_cfg, train_episodes=train_eps)

            print(f"    CDF: {method} seed {run_idx+1}/{n_seeds}", flush=True)
            agent, _ = train_thz_agent(method, sys_cfg, rl_cfg, run_cfg_m, seed,
                                       finetune_episodes=ft_eps)
            detail = evaluate_thz_agent_detailed(
                agent, method, sys_cfg, rl_cfg, run_cfg, seed
            )
            all_sinr.extend(detail["sinr_db_samples"])
        cdf_data[method] = all_sinr

    # AO baseline
    all_sinr = []
    for run_idx in range(n_seeds):
        seed = sys_cfg.seed + 53 * run_idx
        detail = evaluate_thz_ao_baseline_detailed(sys_cfg, rl_cfg, run_cfg, seed)
        all_sinr.extend(detail["sinr_db_samples"])
    cdf_data["baseline_ao"] = all_sinr

    return cdf_data


# ── No-IRS baseline ─────────────────────────────────────────────────

def evaluate_no_irs_baseline(
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
    n_seeds: int = 2,
) -> dict:
    """Evaluate with minimal RIS (1×1 = 1 element) as No-IRS proxy."""
    # 1×1 RIS with 1 subarray → negligible reflection gain
    cfg_no_irs = replace(sys_cfg, n_ris_h=1, n_ris_v=1,
                         q_subarrays_h=1, q_subarrays_v=1)
    rates, prots = [], []
    for run_idx in range(n_seeds):
        seed = cfg_no_irs.seed + 31 * run_idx
        detail = evaluate_thz_ao_baseline_detailed(cfg_no_irs, rl_cfg, run_cfg, seed)
        rates.append(detail["rate_mean"])
        prots.append(detail["protection_mean"])

    return {
        "rate_mean": float(np.mean(rates)),
        "rate_std": float(np.std(rates)),
        "protection_mean": float(np.mean(prots)),
        "protection_std": float(np.std(prots)),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    fast = "--fast" in sys.argv

    output_dir = Path(__file__).resolve().parents[1] / "outputs_wolf_v2"
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "sweep_results.json"

    # ── System configs ───────────────────────────────────────────────
    # The full 256-RIS/64-BS system takes ~5s/step (matrix ops scale as
    # O(N_BS^2 * N_RIS)).  For parameter sweeps that need hundreds of
    # training runs, use a medium-sized system (3ms/step) that still
    # captures the correct physical trends.  Base evaluation uses the
    # full system (already in paper_results.json).

    base_cfg = THzSystemConfig()  # full system for CDF/runtime measurement

    # Medium system for sweep efficiency (same physics, ~1000x faster)
    sweep_cfg = THzSystemConfig(
        n_bs_antennas=32,
        n_rf_chains=8,
        n_ris_h=16,   # 256 RIS elements (16x16)
        n_ris_v=16,
        q_subarrays_h=4,
        q_subarrays_v=4,
        n_subcarriers=32,
        subcarrier_stride=4,
    )
    rl_cfg = THzRLConfig()

    if fast:
        run_cfg = THzTrainEvalConfig(
            train_episodes=100,
            train_steps_per_episode=15,
            eval_episodes=15,
            eval_steps_per_episode=10,
            n_seeds=1,
        )
        pmax_vals = [25.0, 32.0, 40.0]
        nris_vals = [16, 64, 144, 256]
        sinr_vals = [3.0, 10.0, 20.0]
        pjam_vals = [5.0, 12.0, 20.0]
        cdf_seeds = 1
    else:
        run_cfg = THzTrainEvalConfig(
            train_episodes=150,
            train_steps_per_episode=20,
            eval_episodes=20,
            eval_steps_per_episode=15,
            n_seeds=2,
        )
        pmax_vals = [25.0, 30.0, 35.0, 40.0]
        nris_vals = [16, 64, 144, 256]
        sinr_vals = [3.0, 5.0, 10.0, 15.0, 20.0]
        pjam_vals = [5.0, 10.0, 15.0, 20.0]
        cdf_seeds = 2

    results: dict = {"fast_mode": fast}

    # ── 1. Parameter sweeps ───────────────────────────────────────────
    sweep_cfgs = [
        ("pmax_dbm", pmax_vals, "P_max (dBm)"),
        ("n_ris_total", nris_vals, "N_RIS"),
        ("sinr_min_db", sinr_vals, "SINR_min (dB)"),
        ("p_jammer_max_dbm", pjam_vals, "P_jammer (dBm)"),
    ]

    for param, values, label in sweep_cfgs:
        print(f"\n{'='*60}")
        print(f"  Sweep: {label}")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        results[f"sweep_{param}"] = run_thz_parameter_sweep(
            param, values, sweep_cfg, rl_cfg, run_cfg,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.0f}s")

    # ── 2. SINR CDF ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Collecting SINR CDF data")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    results["sinr_cdf"] = collect_sinr_cdf(sweep_cfg, rl_cfg, run_cfg, n_seeds=cdf_seeds)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.0f}s")

    # ── 3. Runtime comparison ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Measuring runtime per method")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    results["runtime"] = measure_runtime(sweep_cfg, rl_cfg, run_cfg, n_seeds=cdf_seeds)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.0f}s")

    # ── 4. No-IRS baseline ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  No-IRS baseline evaluation")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    results["no_irs_baseline"] = evaluate_no_irs_baseline(
        sweep_cfg, rl_cfg, run_cfg, n_seeds=cdf_seeds,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.0f}s")

    # ── Save ──────────────────────────────────────────────────────────
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  Saved: {out_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
