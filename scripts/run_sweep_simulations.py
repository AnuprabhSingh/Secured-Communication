#!/usr/bin/env python3
"""Run all parameter sweeps needed for Figs 5, 6, 7, 11, 12.

Sweeps performed:
  fig5  – P_max sweep : [25, 32, 40] dBm
  fig6  – N_RIS sweep : [16, 64, 144, 256] elements (square RIS: sqrt x sqrt)
  fig7  – SINR target : [3, 5, 10, 15, 20] dB
  fig11 – SINR CDF    : detailed evaluation at paper config (collects per-step SINR)
  fig12 – Jammer pwr  : [5, 10, 15, 20] dBm max jammer power

Each sweep point trains 5 seeds × 4 RL methods from scratch at that config,
then evaluates. AO baseline is also evaluated at every sweep point.

Outputs:
  outputs_joint_ao/sweep_pmax.json
  outputs_joint_ao/sweep_nris.json
  outputs_joint_ao/sweep_sinr_target.json
  outputs_joint_ao/sweep_sinr_cdf.json
  outputs_joint_ao/sweep_pjammer.json
  outputs_joint_ao/sweep_log.txt  ← tail -f this for live progress

Estimated time:
  fig5  (3 pts): ~2.5 h
  fig6  (4 pts): ~3.5 h  (16-element RIS is fast; 256 matches paper ≈ reuses)
  fig7  (5 pts): ~4 h
  fig11 (1 pt) : ~15 min (evaluation only, re-trains once at paper config)
  fig12 (4 pts): ~3 h
  TOTAL: ~13-14 h  ← run overnight / leave running

Usage:
    .venv/bin/python3 scripts/run_sweep_simulations.py
"""
from __future__ import annotations

import datetime
import json
import math
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
    evaluate_thz_ao_baseline_detailed,
    train_thz_agent,
)

# ---------------------------------------------------------------------------
# Base config — paper Table I
# ---------------------------------------------------------------------------
BASE_CFG = THzSystemConfig(
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
    train_episodes=600,
    train_steps_per_episode=20,
    eval_episodes=0,
    eval_steps_per_episode=0,
)
EVAL_CFG = THzTrainEvalConfig(
    train_episodes=0,
    train_steps_per_episode=0,
    eval_episodes=30,
    eval_steps_per_episode=10,
)

SEEDS   = [0, 1, 2, 3, 4]
METHODS = ["q_learning", "fast_q_learning", "fuzzy_wolf_phc", "dqn"]

OUTPUT_DIR = PROJECT_ROOT / "outputs_joint_ao"
LOG_FILE   = OUTPUT_DIR / "sweep_log.txt"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------
PMAX_DBM_VALUES        = [25.0, 32.0, 40.0]          # Fig 5
NRIS_VALUES            = [16, 64, 144, 256]            # Fig 6  (total elements, square)
SINR_TARGET_DB_VALUES  = [3.0, 5.0, 10.0, 15.0, 20.0] # Fig 7
PJAMMER_MAX_DBM_VALUES = [5.0, 10.0, 15.0, 20.0]      # Fig 12

# ---------------------------------------------------------------------------
# Tee-logger
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, path: Path):
        self._file   = open(path, "w", buffering=1)
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
    return datetime.datetime.now().strftime("%H:%M:%S")


def _eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "--:--"
    remaining = elapsed / done * (total - done)
    h, rem = divmod(int(remaining), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Core: train + evaluate one method at one sweep point over all seeds
# ---------------------------------------------------------------------------
def _run_one_point(
    label: str,
    cfg: THzSystemConfig,
    task_counter: list,   # [done, total]
    grand_start: float,
    detailed: bool = False,
) -> dict:
    """Train all 4 methods + AO baseline at cfg over all seeds.

    Returns:
        { "ao_baseline": [{"rate":..,"protection":..}, ...],
          "q_learning":  [{"rate":..,"protection":..,"time":..}, ...],
          ...
          "sinr_db_samples": {method: [all per-user SINR db across seeds]}  (detailed only)
        }
    """
    results: dict[str, list] = {m: [] for m in METHODS}
    results["ao_baseline"] = []
    sinr_db_all: dict[str, list[float]] = {m: [] for m in METHODS + ["ao_baseline"]}

    for seed_idx, seed in enumerate(SEEDS):
        seed_cfg = replace(cfg, seed=seed)

        # AO baseline
        if detailed:
            ao_det = evaluate_thz_ao_baseline_detailed(seed_cfg, RL_CFG, EVAL_CFG, seed=seed)
            results["ao_baseline"].append({
                "rate": ao_det["rate_mean"],
                "protection": ao_det["protection_mean"],
            })
            sinr_db_all["ao_baseline"].extend(ao_det["sinr_db_samples"])
        else:
            ao_rate, ao_prot = evaluate_thz_ao_baseline(seed_cfg, RL_CFG, EVAL_CFG, seed=seed)
            results["ao_baseline"].append({"rate": ao_rate, "protection": ao_prot})

        for method in METHODS:
            t0 = time.time()
            agent, _ = train_thz_agent(
                method, seed_cfg, RL_CFG, TRAIN_CFG, seed=seed, log_interval=0
            )
            train_time = time.time() - t0

            if detailed:
                det = evaluate_thz_agent_detailed(
                    agent, method, seed_cfg, RL_CFG, EVAL_CFG, seed=seed
                )
                results[method].append({
                    "rate": det["rate_mean"],
                    "protection": det["protection_mean"],
                    "time": train_time,
                })
                sinr_db_all[method].extend(det["sinr_db_samples"])
            else:
                rate, prot = evaluate_thz_agent(
                    agent, method, seed_cfg, RL_CFG, EVAL_CFG, seed=seed
                )
                results[method].append({
                    "rate": rate, "protection": prot, "time": train_time,
                })

            task_counter[0] += 1
            elapsed = time.time() - grand_start
            print(f"[{_ts()}]    {label} | seed {seed} | {method}: "
                  f"rate={results[method][-1]['rate']:.3f}  "
                  f"prot={results[method][-1]['protection']:.1f}%  "
                  f"[{train_time:.0f}s train]  "
                  f"| progress {task_counter[0]}/{task_counter[1]}  "
                  f"ETA {_eta(elapsed, task_counter[0], task_counter[1])}",
                  flush=True)

    out: dict = {"per_seed": results}
    if detailed:
        out["sinr_db_samples"] = sinr_db_all
    return out


def _summary(per_seed: dict[str, list]) -> dict:
    s = {}
    for m, entries in per_seed.items():
        rates = [e["rate"] for e in entries]
        prots = [e["protection"] for e in entries]
        s[m] = {
            "rate_mean": float(np.mean(rates)),
            "rate_std":  float(np.std(rates)),
            "protection_mean": float(np.mean(prots)),
            "protection_std":  float(np.std(prots)),
        }
    return s


# ---------------------------------------------------------------------------
# Individual sweep runners
# ---------------------------------------------------------------------------

def run_pmax_sweep(task_counter: list, grand_start: float) -> None:
    print(f"\n[{_ts()}] {'='*60}")
    print(f"[{_ts()}]  SWEEP: P_max  values={PMAX_DBM_VALUES} dBm")
    print(f"[{_ts()}] {'='*60}")
    sweep_results = []
    for pmax in PMAX_DBM_VALUES:
        cfg = replace(BASE_CFG, pmax_dbm=pmax)
        print(f"\n[{_ts()}]  --- P_max = {pmax} dBm ---")
        point = _run_one_point(f"Pmax={pmax:.0f}dBm", cfg, task_counter, grand_start)
        sweep_results.append({
            "pmax_dbm": pmax,
            "per_seed": point["per_seed"],
            "summary": _summary(point["per_seed"]),
        })
        # Checkpoint after each point
        with open(OUTPUT_DIR / "sweep_pmax.json", "w") as f:
            json.dump({"sweep": sweep_results}, f, indent=2)
        print(f"[{_ts()}]  Checkpoint: sweep_pmax.json updated.")
    print(f"[{_ts()}]  P_max sweep DONE.")


def run_nris_sweep(task_counter: list, grand_start: float) -> None:
    print(f"\n[{_ts()}] {'='*60}")
    print(f"[{_ts()}]  SWEEP: N_RIS  values={NRIS_VALUES} elements")
    print(f"[{_ts()}] {'='*60}")
    sweep_results = []
    for n_total in NRIS_VALUES:
        # Keep square RIS; adjust subarrays to stay valid
        side = int(math.isqrt(n_total))
        # sub-array count: at most side (must divide side evenly)
        q_h = min(8, side)
        while side % q_h != 0:
            q_h -= 1
        q_v = q_h
        cfg = replace(BASE_CFG,
                      n_ris_h=side, n_ris_v=side,
                      q_subarrays_h=q_h, q_subarrays_v=q_v)
        print(f"\n[{_ts()}]  --- N_RIS = {n_total}  ({side}×{side}, "
              f"sub-arrays {q_h}×{q_v}) ---")
        point = _run_one_point(f"NRIS={n_total}", cfg, task_counter, grand_start)
        sweep_results.append({
            "n_ris": n_total,
            "per_seed": point["per_seed"],
            "summary": _summary(point["per_seed"]),
        })
        with open(OUTPUT_DIR / "sweep_nris.json", "w") as f:
            json.dump({"sweep": sweep_results}, f, indent=2)
        print(f"[{_ts()}]  Checkpoint: sweep_nris.json updated.")
    print(f"[{_ts()}]  N_RIS sweep DONE.")


def run_sinr_target_sweep(task_counter: list, grand_start: float) -> None:
    print(f"\n[{_ts()}] {'='*60}")
    print(f"[{_ts()}]  SWEEP: SINR target  values={SINR_TARGET_DB_VALUES} dB")
    print(f"[{_ts()}] {'='*60}")
    sweep_results = []
    for sinr_db in SINR_TARGET_DB_VALUES:
        cfg = replace(BASE_CFG, sinr_min_db=sinr_db)
        print(f"\n[{_ts()}]  --- SINR target = {sinr_db} dB ---")
        point = _run_one_point(f"SINRtgt={sinr_db}dB", cfg, task_counter, grand_start)
        sweep_results.append({
            "sinr_target_db": sinr_db,
            "per_seed": point["per_seed"],
            "summary": _summary(point["per_seed"]),
        })
        with open(OUTPUT_DIR / "sweep_sinr_target.json", "w") as f:
            json.dump({"sweep": sweep_results}, f, indent=2)
        print(f"[{_ts()}]  Checkpoint: sweep_sinr_target.json updated.")
    print(f"[{_ts()}]  SINR target sweep DONE.")


def run_sinr_cdf(task_counter: list, grand_start: float) -> None:
    """Detailed evaluation at paper config to collect per-user SINR samples."""
    print(f"\n[{_ts()}] {'='*60}")
    print(f"[{_ts()}]  SINR CDF: detailed evaluation at paper config")
    print(f"[{_ts()}] {'='*60}")
    point = _run_one_point("CDF", BASE_CFG, task_counter, grand_start, detailed=True)
    summary = _summary(point["per_seed"])
    out = {
        "per_seed": point["per_seed"],
        "summary": summary,
        "sinr_db_samples": point["sinr_db_samples"],
    }
    with open(OUTPUT_DIR / "sweep_sinr_cdf.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{_ts()}]  SINR CDF data saved.")


def run_pjammer_sweep(task_counter: list, grand_start: float) -> None:
    print(f"\n[{_ts()}] {'='*60}")
    print(f"[{_ts()}]  SWEEP: Jammer power  values={PJAMMER_MAX_DBM_VALUES} dBm")
    print(f"[{_ts()}] {'='*60}")
    sweep_results = []
    for pj in PJAMMER_MAX_DBM_VALUES:
        # min jammer power = max - 10 dB range, floor at 0
        pj_min = max(0.0, pj - 10.0)
        cfg = replace(BASE_CFG, p_jammer_max_dbm=pj, p_jammer_min_dbm=pj_min)
        print(f"\n[{_ts()}]  --- P_jammer_max = {pj} dBm ---")
        point = _run_one_point(f"Pjam={pj:.0f}dBm", cfg, task_counter, grand_start)
        sweep_results.append({
            "pjammer_max_dbm": pj,
            "per_seed": point["per_seed"],
            "summary": _summary(point["per_seed"]),
        })
        with open(OUTPUT_DIR / "sweep_pjammer.json", "w") as f:
            json.dump({"sweep": sweep_results}, f, indent=2)
        print(f"[{_ts()}]  Checkpoint: sweep_pjammer.json updated.")
    print(f"[{_ts()}]  Jammer power sweep DONE.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    tee = Tee(LOG_FILE)
    sys.stdout = tee

    # Total training tasks: count all (seed, method, sweep_point) combinations
    n_pmax   = len(PMAX_DBM_VALUES)        * len(SEEDS) * len(METHODS)
    n_nris   = len(NRIS_VALUES)            * len(SEEDS) * len(METHODS)
    n_sinrt  = len(SINR_TARGET_DB_VALUES)  * len(SEEDS) * len(METHODS)
    n_cdf    = 1                           * len(SEEDS) * len(METHODS)
    n_pjam   = len(PJAMMER_MAX_DBM_VALUES) * len(SEEDS) * len(METHODS)
    total    = n_pmax + n_nris + n_sinrt + n_cdf + n_pjam

    task_counter = [0, total]   # mutable so helpers can increment
    grand_start  = time.time()

    # Per-task time estimate (seconds): from observed 600-ep run
    avg_task_s = (94 + 94 + 97 + 152) / 4  # mean across 4 methods
    est_h = total * avg_task_s / 3600

    print(f"[{_ts()}] {'='*60}")
    print(f"[{_ts()}]  Sweep simulations for Figs 5, 6, 7, 11, 12")
    print(f"[{_ts()}]  Log file: {LOG_FILE}")
    print(f"[{_ts()}]  Total training tasks: {total}")
    print(f"[{_ts()}]  Estimated time: {est_h:.1f} h")
    print(f"[{_ts()}] {'='*60}")
    print(f"[{_ts()}]  Task breakdown:")
    print(f"[{_ts()}]    Fig 5  (P_max sweep)  : {n_pmax}  tasks")
    print(f"[{_ts()}]    Fig 6  (N_RIS sweep)  : {n_nris}  tasks")
    print(f"[{_ts()}]    Fig 7  (SINR target)  : {n_sinrt} tasks")
    print(f"[{_ts()}]    Fig 11 (SINR CDF)     : {n_cdf}   tasks")
    print(f"[{_ts()}]    Fig 12 (Jammer power) : {n_pjam}  tasks")
    print()

    run_pmax_sweep(task_counter, grand_start)
    run_nris_sweep(task_counter, grand_start)
    run_sinr_target_sweep(task_counter, grand_start)
    run_sinr_cdf(task_counter, grand_start)
    run_pjammer_sweep(task_counter, grand_start)

    grand_elapsed = time.time() - grand_start
    print(f"\n[{_ts()}] {'='*60}")
    print(f"[{_ts()}]  ALL SWEEPS COMPLETE — {grand_elapsed/3600:.2f} h total")
    print(f"[{_ts()}]  Results in {OUTPUT_DIR}/")
    print(f"[{_ts()}]  Run: .venv/bin/python3 scripts/generate_joint_ao_plots.py")
    print(f"[{_ts()}] {'='*60}")

    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    main()
