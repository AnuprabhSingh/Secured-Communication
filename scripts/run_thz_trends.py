#!/usr/bin/env python3
"""Main CLI for wideband THz IRS anti-jamming experiments.

Usage:
  python scripts/run_thz_trends.py --profile quick
  python scripts/run_thz_trends.py --profile balanced
  python scripts/run_thz_trends.py --profile full

Produces:
  - Convergence plot (reward vs episode for all RL methods + DQN)
  - Parameter sweeps: P_max, N_RIS, bandwidth, Q (TD modules)
  - Beam squint analysis plot
  - results.json with all numerical data
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from irs_anti_jamming.thz.thz_config import (
    THzRLConfig,
    THzSweepConfig,
    THzSystemConfig,
    THzTrainEvalConfig,
)
from irs_anti_jamming.thz.thz_experiments import (
    RL_METHODS,
    run_beam_squint_analysis,
    run_thz_convergence,
    run_thz_parameter_sweep,
)


# ---- Profiles ----

PROFILES = {
    "quick": {
        "train_episodes": 100,
        "train_steps_per_episode": 10,
        "eval_episodes": 10,
        "eval_steps_per_episode": 5,
        "n_seeds": 1,
        "subcarrier_stride": 16,
        # Reduced array sizes for speed
        "n_bs_antennas": 32,
        "n_rf_chains": 4,
        "n_ris_h": 8,
        "n_ris_v": 8,
        "q_subarrays_h": 4,
        "q_subarrays_v": 4,
        "n_subcarriers": 32,
    },
    "balanced": {
        "train_episodes": 400,
        "train_steps_per_episode": 20,
        "eval_episodes": 20,
        "eval_steps_per_episode": 10,
        "n_seeds": 2,
        "subcarrier_stride": 8,
        "n_bs_antennas": 64,
        "n_rf_chains": 8,
        "n_ris_h": 16,
        "n_ris_v": 16,
        "q_subarrays_h": 4,
        "q_subarrays_v": 4,
        "n_subcarriers": 64,
    },
    "full": {
        "train_episodes": 1200,
        "train_steps_per_episode": 50,
        "eval_episodes": 50,
        "eval_steps_per_episode": 20,
        "n_seeds": 3,
        "subcarrier_stride": 4,
        "n_bs_antennas": 256,
        "n_rf_chains": 16,
        "n_ris_h": 64,
        "n_ris_v": 64,
        "q_subarrays_h": 8,
        "q_subarrays_v": 8,
        "n_subcarriers": 128,
    },
    "paper": {
        "train_episodes": 500,
        "train_steps_per_episode": 20,
        "eval_episodes": 25,
        "eval_steps_per_episode": 10,
        "n_seeds": 3,
        "subcarrier_stride": 8,
        "n_bs_antennas": 64,
        "n_rf_chains": 8,
        "n_ris_h": 16,
        "n_ris_v": 16,
        "q_subarrays_h": 4,
        "q_subarrays_v": 4,
        "n_subcarriers": 64,
    },
}


METHOD_LABELS = {
    "q_learning": "Classical Q-Learning",
    "fast_q_learning": "Fast Q-Learning",
    "fuzzy_wolf_phc": "Fuzzy WoLF-PHC",
    "dqn": "DQN",
    "baseline_ao": "AO Baseline",
}

METHOD_COLORS = {
    "q_learning": "C3",
    "fast_q_learning": "C1",
    "fuzzy_wolf_phc": "C0",
    "dqn": "C2",
    "baseline_ao": "C4",
}


# ---- Plotting helpers ----

def _moving_average(y: np.ndarray, window: int = 25) -> tuple[np.ndarray, np.ndarray]:
    if y.size == 0:
        return np.array([]), np.array([])
    w = max(1, min(window, y.size))
    if w == 1:
        return np.arange(1, y.size + 1, dtype=float), y
    kernel = np.ones(w) / w
    smoothed = np.convolve(y, kernel, mode="valid")
    return np.arange(w, y.size + 1, dtype=float), smoothed


def plot_convergence(data: dict[str, np.ndarray], out_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for method in RL_METHODS:
        if method not in data:
            continue
        y = data[method]
        x_raw = np.arange(1, y.size + 1, dtype=float)
        x_ma, y_ma = _moving_average(y, window=25)
        color = METHOD_COLORS.get(method, None)
        plt.plot(x_raw, y, linewidth=0.8, alpha=0.15, color=color)
        plt.plot(x_ma, y_ma, label=METHOD_LABELS[method], linewidth=2.0, color=color)

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("THz Wideband Anti-Jamming: Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "thz_convergence.png", dpi=220)
    plt.close()
    print(f"  Saved: {out_dir / 'thz_convergence.png'}")


def plot_sweep(sweep_data: dict, xlabel: str, title: str, out_path: Path) -> None:
    x = sweep_data["x"]
    # Convert Hz to GHz for bandwidth labels
    if "bandwidth" in sweep_data.get("parameter", "").lower() or "bandwidth" in xlabel.lower():
        x = [v / 1e9 for v in x]
        xlabel = "Bandwidth (GHz)"
    methods = sweep_data["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for method, stats in methods.items():
        color = METHOD_COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)
        rate_mean = stats.get("rate_mean", stats["rate"])
        prot_mean = stats.get("protection_mean", stats["protection"])
        rate_std = stats.get("rate_std", None)
        prot_std = stats.get("protection_std", None)

        if rate_std and any(s > 0 for s in rate_std):
            axes[0].errorbar(x, rate_mean, yerr=rate_std, marker="o", linewidth=2,
                           label=label, color=color, capsize=3, capthick=1.5)
        else:
            axes[0].plot(x, rate_mean, marker="o", linewidth=2, label=label, color=color)

        if prot_std and any(s > 0 for s in prot_std):
            axes[1].errorbar(x, prot_mean, yerr=prot_std, marker="o", linewidth=2,
                           label=label, color=color, capsize=3, capthick=1.5)
        else:
            axes[1].plot(x, prot_mean, marker="o", linewidth=2, label=label, color=color)

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Average System Rate (bit/s/Hz)")
    axes[0].set_title(f"{title} — Rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("SINR Protection (%)")
    axes[1].set_title(f"{title} — Protection")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_beam_squint(data: dict[str, np.ndarray], out_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for scheme, gains in data.items():
        x = np.arange(len(gains))
        plt.plot(x, gains, linewidth=2, label=scheme)
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Normalized Array Gain")
    plt.title("Beam Squint: SPDP vs Classical Phase-Only RIS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(out_dir / "thz_beam_squint.png", dpi=220)
    plt.close()
    print(f"  Saved: {out_dir / 'thz_beam_squint.png'}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="THz Wideband Anti-Jamming Experiments")
    parser.add_argument("--profile", default="quick", choices=PROFILES.keys())
    parser.add_argument("--output-dir", default="outputs_thz")
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--beam-squint-only", action="store_true")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== THz Wideband Anti-Jamming Experiments (profile={args.profile}) ===")
    print(f"  BS: {profile['n_bs_antennas']} antennas, {profile['n_rf_chains']} RF chains")
    print(f"  RIS: {profile['n_ris_h']}x{profile['n_ris_v']} = {profile['n_ris_h']*profile['n_ris_v']} elements")
    print(f"  OFDM: {profile['n_subcarriers']} subcarriers, stride={profile['subcarrier_stride']}")
    print(f"  Training: {profile['train_episodes']} episodes, {profile['n_seeds']} seeds")
    print(f"  Output: {out_dir}")

    sys_cfg = THzSystemConfig(
        n_bs_antennas=profile["n_bs_antennas"],
        n_rf_chains=profile["n_rf_chains"],
        n_ris_h=profile["n_ris_h"],
        n_ris_v=profile["n_ris_v"],
        q_subarrays_h=profile["q_subarrays_h"],
        q_subarrays_v=profile["q_subarrays_v"],
        n_subcarriers=profile["n_subcarriers"],
        subcarrier_stride=profile["subcarrier_stride"],
    )
    rl_cfg = THzRLConfig()
    run_cfg = THzTrainEvalConfig(
        train_episodes=profile["train_episodes"],
        train_steps_per_episode=profile["train_steps_per_episode"],
        eval_episodes=profile["eval_episodes"],
        eval_steps_per_episode=profile["eval_steps_per_episode"],
        n_seeds=profile["n_seeds"],
    )

    results = {}

    # --- Beam squint analysis ---
    print("\n--- Beam Squint Analysis ---")
    bs_data = run_beam_squint_analysis(sys_cfg)
    plot_beam_squint(bs_data, out_dir)
    results["beam_squint"] = {k: v.tolist() for k, v in bs_data.items()}

    if args.beam_squint_only:
        _save_results(results, out_dir)
        return

    # --- Convergence ---
    print("\n--- Convergence Experiment ---")
    conv_data = run_thz_convergence(sys_cfg, rl_cfg, run_cfg)
    plot_convergence(conv_data, out_dir)
    results["convergence"] = {k: v.tolist() for k, v in conv_data.items()}

    if args.skip_sweeps:
        _save_results(results, out_dir)
        return

    # --- Parameter sweeps ---
    sweep_cfg = THzSweepConfig()

    print("\n--- Sweep: P_max ---")
    sweep_pmax = run_thz_parameter_sweep("pmax_dbm", sweep_cfg.pmax_dbm_values,
                                          sys_cfg, rl_cfg, run_cfg)
    plot_sweep(sweep_pmax, "P_max (dBm)", "THz: Rate vs Transmit Power",
               out_dir / "thz_sweep_pmax.png")
    results["sweep_pmax"] = sweep_pmax

    print("\n--- Sweep: N_RIS ---")
    sweep_ris = run_thz_parameter_sweep("n_ris_total", sweep_cfg.n_ris_elements_values,
                                         sys_cfg, rl_cfg, run_cfg)
    plot_sweep(sweep_ris, "N_RIS (elements)", "THz: Rate vs RIS Size",
               out_dir / "thz_sweep_nris.png")
    results["sweep_nris"] = sweep_ris

    print("\n--- Sweep: Bandwidth ---")
    sweep_bw = run_thz_parameter_sweep("bandwidth_hz", sweep_cfg.bandwidth_values_hz,
                                        sys_cfg, rl_cfg, run_cfg)
    plot_sweep(sweep_bw, "Bandwidth (GHz)",
               "THz: Rate vs Bandwidth (Beam Squint Impact)",
               out_dir / "thz_sweep_bandwidth.png")
    results["sweep_bandwidth"] = sweep_bw

    print("\n--- Sweep: SINR Target ---")
    sweep_sinr = run_thz_parameter_sweep("sinr_min_db", sweep_cfg.sinr_target_db_values,
                                          sys_cfg, rl_cfg, run_cfg)
    plot_sweep(sweep_sinr, "SINR Target (dB)", "THz: Rate vs SINR Target",
               out_dir / "thz_sweep_sinr.png")
    results["sweep_sinr"] = sweep_sinr

    _save_results(results, out_dir)
    print(f"\n=== Done. All outputs in {out_dir} ===")


def _save_results(results: dict, out_dir: Path) -> None:
    path = out_dir / "thz_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
