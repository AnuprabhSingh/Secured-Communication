#!/usr/bin/env python3
"""Generate paper results for the joint AO (3-block) formulation.

Runs all 5 methods on paper-scale config (N=64, M=256, Msc=64) over 3 seeds
and saves results to outputs_joint_ao/paper_results.json.

Usage:
    .venv/bin/python3 scripts/run_joint_ao_results.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from irs_anti_jamming.thz.thz_config import (
    THzRLConfig,
    THzSystemConfig,
    THzTrainEvalConfig,
)
from irs_anti_jamming.thz.thz_experiments import (
    RL_METHODS,
    evaluate_thz_agent,
    evaluate_thz_ao_baseline,
    train_thz_agent,
)

# ---------------------------------------------------------------------------
# Config (matches paper Table I)
# ---------------------------------------------------------------------------
SYS_CFG = THzSystemConfig(
    n_bs_antennas=64,
    n_rf_chains=8,
    n_ris_h=16,
    n_ris_v=16,
    q_subarrays_h=8,
    q_subarrays_v=8,
    n_subcarriers=64,
    k_users=4,
    n_jammer_antennas=2,
    subcarrier_stride=4,
    seed=0,
)
RL_CFG = THzRLConfig()
RUN_CFG = THzTrainEvalConfig(
    train_episodes=400,
    train_steps_per_episode=20,
    eval_episodes=25,
    eval_steps_per_episode=10,
)
SEEDS = [0, 1, 2]
METHODS = ["q_learning", "fast_q_learning", "fuzzy_wolf_phc", "dqn"]

OUTPUT_DIR = PROJECT_ROOT / "outputs_joint_ao"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    results: dict[str, list] = {m: [] for m in METHODS}
    results["ao_baseline"] = []

    total_start = time.time()

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")
        cfg = replace(SYS_CFG, seed=seed)

        # AO baseline (no training needed)
        ao_rate, ao_prot = evaluate_thz_ao_baseline(cfg, RL_CFG, RUN_CFG, seed=seed)
        results["ao_baseline"].append({"rate": ao_rate, "protection": ao_prot})
        print(f"  AO baseline: rate={ao_rate:.3f}, protection={ao_prot:.1f}%")

        for method in METHODS:
            t0 = time.time()
            agent, _ = train_thz_agent(method, cfg, RL_CFG, RUN_CFG, seed=seed)
            rate, prot = evaluate_thz_agent(agent, method, cfg, RL_CFG, RUN_CFG, seed=seed)
            elapsed = time.time() - t0
            results[method].append({"rate": rate, "protection": prot, "time": elapsed})
            print(f"  {method:20s}: rate={rate:.3f}, protection={prot:.1f}%  [{elapsed:.1f}s]")

    print(f"\nTotal time: {time.time() - total_start:.1f}s")

    # Compute mean ± std
    summary = {}
    for method, entries in results.items():
        rates = [e["rate"] for e in entries]
        prots = [e["protection"] for e in entries]
        summary[method] = {
            "rate_mean": float(np.mean(rates)),
            "rate_std": float(np.std(rates)),
            "protection_mean": float(np.mean(prots)),
            "protection_std": float(np.std(prots)),
        }

    print("\n" + "="*60)
    print("SUMMARY (mean ± std over 3 seeds)")
    print("="*60)
    for m, s in summary.items():
        print(f"  {m:20s}: {s['rate_mean']:.2f} ± {s['rate_std']:.2f} bps/Hz  |  "
              f"{s['protection_mean']:.1f} ± {s['protection_std']:.1f}%")

    out_path = OUTPUT_DIR / "paper_results.json"
    with open(out_path, "w") as f:
        json.dump({"per_seed": results, "summary": summary}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
