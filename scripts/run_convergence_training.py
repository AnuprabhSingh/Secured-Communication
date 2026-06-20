#!/usr/bin/env python3
"""Run multi-seed training to collect REAL per-episode reward histories.

Uses the same paper-scale system config (N=64, M=256, Msc=64) as the main
results run.  Saves reward histories to outputs_joint_ao/convergence_data.json.
The generate_joint_ao_plots.py script reads this file to produce Fig 4.

Usage:
    .venv/bin/python3 scripts/run_convergence_training.py

Estimated run time (on an Apple Silicon MacBook Air):
    5 seeds × 600 episodes × 4 methods ≈ 35–45 minutes.
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
from irs_anti_jamming.thz.thz_experiments import train_thz_agent

# ---------------------------------------------------------------------------
# Config — identical to run_joint_ao_results.py (paper Table I)
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

# 600 episodes: ~50% more than the 400 used in evaluation, so convergence
# is clearly visible in the tail of the curves.
RUN_CFG = THzTrainEvalConfig(
    train_episodes=600,
    train_steps_per_episode=20,
    eval_episodes=0,   # no evaluation needed here
    eval_steps_per_episode=0,
)

SEEDS = [0, 1, 2, 3, 4]   # 5 independent seeds → meaningful confidence bands

METHODS = ["q_learning", "fast_q_learning", "fuzzy_wolf_phc", "dqn"]

OUTPUT_DIR  = PROJECT_ROOT / "outputs_joint_ao"
OUTPUT_FILE = OUTPUT_DIR / "convergence_data.json"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Accumulate: histories[method] = list of arrays, one per seed
    histories: dict[str, list[list[float]]] = {m: [] for m in METHODS}

    grand_start = time.time()

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})")
        print(f"{'='*60}")
        cfg = replace(SYS_CFG, seed=seed)

        for method in METHODS:
            t0 = time.time()
            print(f"  Training {method} ...", end="", flush=True)
            _, reward_history = train_thz_agent(
                method, cfg, RL_CFG, RUN_CFG, seed=seed,
                log_interval=100,
            )
            elapsed = time.time() - t0
            print(f"  done in {elapsed:.1f}s  "
                  f"(final 50-ep avg reward = "
                  f"{float(np.mean(reward_history[-50:])):.3f})")
            histories[method].append([float(v) for v in reward_history])

    grand_elapsed = time.time() - grand_start
    print(f"\nTotal training time: {grand_elapsed/60:.1f} min")

    # Compute per-method statistics for a quick sanity check
    print("\nPer-method final reward (mean ± std over seeds, last 50 ep):")
    for method in METHODS:
        finals = [np.mean(h[-50:]) for h in histories[method]]
        print(f"  {method:20s}: {np.mean(finals):.3f} ± {np.std(finals):.3f}")

    # Save
    out = {
        "config": {
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "train_episodes": RUN_CFG.train_episodes,
            "train_steps_per_episode": RUN_CFG.train_steps_per_episode,
            "n_bs_antennas": SYS_CFG.n_bs_antennas,
            "n_ris_elements": SYS_CFG.n_ris_h * SYS_CFG.n_ris_v,
            "n_subcarriers": SYS_CFG.n_subcarriers,
        },
        "histories": histories,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
