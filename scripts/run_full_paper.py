#!/usr/bin/env python3
"""Full paper re-run: 5 seeds × 600 training episodes.

Produces:
  outputs_joint_ao/paper_results.json      — evaluation metrics for all plots
  outputs_joint_ao/convergence_data.json   — per-episode reward histories for Fig 4
  outputs_joint_ao/training_log.txt        — full timestamped log (tail -f this file)

Every print goes to both stdout and the log file simultaneously.

Usage:
    .venv/bin/python3 scripts/run_full_paper.py

Estimated time:  ~60–80 minutes on an Apple Silicon MacBook Air
   (5 seeds × [q 94s + fast_q 93s + fuzzy 95s + dqn 125s] ≈ 68 min training
    + ~10 min evaluation = ~78 min total)
"""
from __future__ import annotations

import datetime
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
    evaluate_thz_agent,
    evaluate_thz_agent_detailed,
    evaluate_thz_ao_baseline,
    train_thz_agent,
)

# ---------------------------------------------------------------------------
# Config — paper Table I
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

TRAIN_CFG = THzTrainEvalConfig(
    train_episodes=600,          # was 400 — more training → better convergence
    train_steps_per_episode=20,
    eval_episodes=0,             # training only pass
    eval_steps_per_episode=0,
)
EVAL_CFG = THzTrainEvalConfig(
    train_episodes=0,
    train_steps_per_episode=0,
    eval_episodes=30,            # was 25 — more eval episodes → tighter stats
    eval_steps_per_episode=10,
)

SEEDS   = [0, 1, 2, 3, 4]      # 5 seeds (was 3)
METHODS = ["q_learning", "fast_q_learning", "fuzzy_wolf_phc", "dqn"]

OUTPUT_DIR        = PROJECT_ROOT / "outputs_joint_ao"
LOG_FILE          = OUTPUT_DIR / "training_log.txt"
RESULTS_FILE      = OUTPUT_DIR / "paper_results.json"
CONVERGENCE_FILE  = OUTPUT_DIR / "convergence_data.json"

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Tee-logger: writes to stdout AND log file simultaneously
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, path: Path):
        self._file = open(path, "w", buffering=1)  # line-buffered
        self._stdout = sys.stdout

    def write(self, text: str) -> None:
        self._stdout.write(text)
        self._file.write(text)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _ts() -> str:
    """Current time stamp string for log lines."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def _eta(elapsed: float, done: int, total: int) -> str:
    """Human-readable ETA string."""
    if done == 0:
        return "--:--"
    remaining = elapsed / done * (total - done)
    h, rem = divmod(int(remaining), 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    tee = Tee(LOG_FILE)
    sys.stdout = tee

    grand_start = time.time()
    total_tasks = len(SEEDS) * len(METHODS)   # training tasks (AO is cheap)
    tasks_done  = 0

    # Timing estimates from the previous 400-ep run (per method, per seed)
    time_estimates_400ep = {
        "q_learning":      80.0,
        "fast_q_learning": 83.0,
        "fuzzy_wolf_phc":  88.0,
        "dqn":             122.0,
    }
    # Scale for 600 ep
    est_per_method = {m: v * 600 / 400 for m, v in time_estimates_400ep.items()}
    est_total = sum(est_per_method[m] for m in METHODS) * len(SEEDS)

    print(f"[{_ts()}] ============================================================")
    print(f"[{_ts()}]  Full paper re-run  — 5 seeds × 600 episodes")
    print(f"[{_ts()}]  Log file: {LOG_FILE}")
    print(f"[{_ts()}]  Estimated total time: {est_total/60:.0f} min")
    print(f"[{_ts()}] ============================================================\n")
    print(f"[{_ts()}]  Config:")
    print(f"[{_ts()}]    Seeds              : {SEEDS}")
    print(f"[{_ts()}]    Train episodes     : {TRAIN_CFG.train_episodes}")
    print(f"[{_ts()}]    Steps/episode      : {TRAIN_CFG.train_steps_per_episode}")
    print(f"[{_ts()}]    Eval episodes      : {EVAL_CFG.eval_episodes}")
    print(f"[{_ts()}]    BS antennas (N)    : {SYS_CFG.n_bs_antennas}")
    print(f"[{_ts()}]    RIS elements (M)   : {SYS_CFG.n_ris_h * SYS_CFG.n_ris_v}")
    print(f"[{_ts()}]    Subcarriers (Msc)  : {SYS_CFG.n_subcarriers}")
    print(f"[{_ts()}]    Users (K)          : {SYS_CFG.k_users}")
    print()

    # Storage
    eval_results: dict[str, list] = {m: [] for m in METHODS}
    eval_results["ao_baseline"] = []
    conv_histories: dict[str, list[list[float]]] = {m: [] for m in METHODS}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n[{_ts()}] {'='*56}")
        print(f"[{_ts()}]  Seed {seed}  ({seed_idx+1}/{len(SEEDS)})")
        elapsed_so_far = time.time() - grand_start
        print(f"[{_ts()}]  Elapsed: {elapsed_so_far/60:.1f} min  |  "
              f"ETA: {_eta(elapsed_so_far, tasks_done, total_tasks)}")
        print(f"[{_ts()}] {'='*56}")
        cfg = replace(SYS_CFG, seed=seed)

        # -- AO baseline (deterministic, no training) --
        print(f"[{_ts()}]  AO baseline ...", end="", flush=True)
        t0 = time.time()
        ao_rate, ao_prot = evaluate_thz_ao_baseline(cfg, RL_CFG, EVAL_CFG, seed=seed)
        print(f"  {time.time()-t0:.1f}s  "
              f"rate={ao_rate:.3f}  prot={ao_prot:.1f}%")
        eval_results["ao_baseline"].append({"rate": ao_rate, "protection": ao_prot})

        # -- RL methods --
        for method in METHODS:
            print(f"\n[{_ts()}]  [{method}]  training ...", flush=True)
            t0 = time.time()
            agent, reward_history = train_thz_agent(
                method, cfg, RL_CFG, TRAIN_CFG, seed=seed,
                log_interval=100,
            )
            train_time = time.time() - t0
            final_50 = float(np.mean(reward_history[-50:]))
            print(f"[{_ts()}]  [{method}]  training done  "
                  f"{train_time:.1f}s  "
                  f"final-50-ep-reward={final_50:.3f}")
            conv_histories[method].append([float(v) for v in reward_history])

            print(f"[{_ts()}]  [{method}]  evaluating ...", end="", flush=True)
            t1 = time.time()
            rate, prot = evaluate_thz_agent(
                agent, method, cfg, RL_CFG, EVAL_CFG, seed=seed
            )
            eval_time = time.time() - t1
            total_time = time.time() - t0
            print(f"  {eval_time:.1f}s  "
                  f"rate={rate:.3f}  prot={prot:.1f}%  "
                  f"(total {total_time:.1f}s)")
            eval_results[method].append({
                "rate": rate,
                "protection": prot,
                "time": train_time,
            })

            tasks_done += 1
            elapsed_so_far = time.time() - grand_start
            print(f"[{_ts()}]  Progress: {tasks_done}/{total_tasks} tasks  |  "
                  f"Elapsed: {elapsed_so_far/60:.1f} min  |  "
                  f"ETA: {_eta(elapsed_so_far, tasks_done, total_tasks)}")

        # -- Checkpoint: save partial results after each seed --
        _save_results(eval_results, conv_histories, partial=True)
        print(f"\n[{_ts()}]  Checkpoint saved after seed {seed}.")

    grand_elapsed = time.time() - grand_start
    print(f"\n[{_ts()}] {'='*56}")
    print(f"[{_ts()}]  Training complete — {grand_elapsed/60:.1f} min total")
    print(f"[{_ts()}] {'='*56}")

    _save_results(eval_results, conv_histories, partial=False)

    # Print final summary table
    summary = _compute_summary(eval_results)
    print(f"\n[{_ts()}]  FINAL SUMMARY  (mean ± std over {len(SEEDS)} seeds)")
    print(f"[{_ts()}]  {'Method':<22}  {'Rate (bps/Hz)':>18}  {'SINR Protection':>18}")
    print(f"[{_ts()}]  {'-'*22}  {'-'*18}  {'-'*18}")
    for m, s in summary.items():
        print(f"[{_ts()}]  {m:<22}  "
              f"{s['rate_mean']:6.3f} ± {s['rate_std']:.3f}       "
              f"{s['protection_mean']:5.1f} ± {s['protection_std']:.1f}%")

    sys.stdout = tee._stdout
    tee.close()
    print(f"\nAll done. Results in {OUTPUT_DIR}/")


def _compute_summary(eval_results: dict) -> dict:
    summary = {}
    for method, entries in eval_results.items():
        rates = [e["rate"] for e in entries]
        prots = [e["protection"] for e in entries]
        summary[method] = {
            "rate_mean":        float(np.mean(rates)),
            "rate_std":         float(np.std(rates)),
            "protection_mean":  float(np.mean(prots)),
            "protection_std":   float(np.std(prots)),
        }
    return summary


def _save_results(eval_results: dict, conv_histories: dict,
                  partial: bool = False) -> None:
    """Save both output files (called after each seed as a checkpoint)."""
    summary = _compute_summary(eval_results)

    # paper_results.json  (read by generate_joint_ao_plots for Figs 5–13)
    per_seed_out: dict[str, list] = {}
    for method, entries in eval_results.items():
        per_seed_out[method] = entries
    with open(RESULTS_FILE, "w") as f:
        json.dump({"per_seed": per_seed_out, "summary": summary}, f, indent=2)

    # convergence_data.json  (read by generate_joint_ao_plots for Fig 4)
    # Only include methods that have at least one history recorded
    completed = {m: h for m, h in conv_histories.items() if h}
    seeds_done = max((len(v) for v in completed.values()), default=0)
    with open(CONVERGENCE_FILE, "w") as f:
        json.dump({
            "config": {
                "n_seeds":                seeds_done,
                "seeds":                  SEEDS[:seeds_done],
                "train_episodes":         TRAIN_CFG.train_episodes,
                "train_steps_per_episode": TRAIN_CFG.train_steps_per_episode,
                "n_bs_antennas":          SYS_CFG.n_bs_antennas,
                "n_ris_elements":         SYS_CFG.n_ris_h * SYS_CFG.n_ris_v,
                "n_subcarriers":          SYS_CFG.n_subcarriers,
                "partial":                partial,
            },
            "histories": completed,
        }, f, indent=2)


if __name__ == "__main__":
    main()
