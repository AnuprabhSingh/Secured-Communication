#!/usr/bin/env python3
"""Paper-quality THz simulations: convergence + parameter sweeps + publication plots.

Runs all experiments needed for a full paper:
  1. Convergence comparison (4 RL methods + AO baseline)
  2. Parameter sweeps: Pmax, N_RIS, Bandwidth, SINR target
  3. Beam squint analysis
  4. Generates 8 publication-quality figures with error bars

Usage:
  python scripts/run_paper_simulations.py                    # paper profile
  python scripts/run_paper_simulations.py --profile balanced  # faster
  python scripts/run_paper_simulations.py --skip-sweeps       # convergence only
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
    evaluate_thz_agent,
    evaluate_thz_ao_baseline,
    run_beam_squint_analysis,
    run_thz_convergence,
    run_thz_parameter_sweep,
    train_thz_agent,
)


# ============================================================================
# Profiles
# ============================================================================

PROFILES = {
    "balanced": {
        "train_episodes": 400,
        "train_steps_per_episode": 20,
        "eval_episodes": 50,
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


# ============================================================================
# Plotting constants
# ============================================================================

METHOD_LABELS = {
    "q_learning": "Classical Q-Learning",
    "fast_q_learning": "Fast Q-Learning [19]",
    "fuzzy_wolf_phc": "Enhanced Fuzzy WoLF-PHC (Proposed)",
    "dqn": "Vanilla DQN [Mnih15]",
    "d3qn": "D3QN-PER [Wang16]",
    "baseline_ao": "AO Baseline [39]",
}

METHOD_COLORS = {
    "q_learning": "#d62728",      # red
    "fast_q_learning": "#ff7f0e",  # orange
    "fuzzy_wolf_phc": "#1f77b4",   # blue (proposed — prominent)
    "dqn": "#8c564b",             # brown
    "d3qn": "#9467bd",            # purple
    "baseline_ao": "#7f7f7f",      # grey
}

METHOD_MARKERS = {
    "q_learning": "s",
    "fast_q_learning": "D",
    "fuzzy_wolf_phc": "o",
    "dqn": "^",
    "d3qn": "p",
    "baseline_ao": "v",
}

# Publication font sizes
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ============================================================================
# Helper functions
# ============================================================================

def moving_average(y: np.ndarray, window: int = 25) -> tuple[np.ndarray, np.ndarray]:
    if y.size == 0:
        return np.array([]), np.array([])
    w = max(1, min(window, y.size))
    if w == 1:
        return np.arange(1, y.size + 1, dtype=float), y
    kernel = np.ones(w) / w
    smoothed = np.convolve(y, kernel, mode="valid")
    return np.arange(w, y.size + 1, dtype=float), smoothed


def elapsed_str(t0: float) -> str:
    dt = time.time() - t0
    if dt < 60:
        return f"{dt:.0f}s"
    return f"{dt/60:.1f}min"


# ============================================================================
# Extended convergence with per-seed curves (for confidence bands)
# ============================================================================

def run_convergence_with_bands(
    sys_cfg: THzSystemConfig,
    rl_cfg: THzRLConfig,
    run_cfg: THzTrainEvalConfig,
) -> dict:
    """Run convergence and return per-seed histories for confidence bands."""
    all_histories: dict[str, list[list[float]]] = {m: [] for m in RL_METHODS}
    eval_results: dict[str, list[tuple[float, float]]] = {m: [] for m in RL_METHODS + ["baseline_ao"]}

    for run_idx in range(run_cfg.n_seeds):
        seed = sys_cfg.seed + 101 * run_idx
        for method in RL_METHODS:
            # Deep RL methods need more training than tabular;
            # WoLF-PHC benefits from extended training + fine-tuning for policy convergence
            if method == "dqn":
                method_cfg = replace(run_cfg, train_episodes=int(run_cfg.train_episodes * 2.5))
                ft_eps = 50
            elif method == "fuzzy_wolf_phc":
                method_cfg = replace(run_cfg, train_episodes=int(run_cfg.train_episodes * 3.0))
                ft_eps = 100
            else:
                method_cfg = run_cfg
                ft_eps = 0
            print(f"  Convergence: seed {run_idx+1}/{run_cfg.n_seeds}, {method} ({method_cfg.train_episodes} eps)", flush=True)
            # DRL agents and WoLF-PHC get fine-tuning to close train/eval gap
            agent, history = train_thz_agent(method, sys_cfg, rl_cfg, method_cfg, seed,
                                             log_interval=50, finetune_episodes=ft_eps)
            all_histories[method].append(history.tolist())

            # Evaluate after training
            rate, prot = evaluate_thz_agent(agent, method, sys_cfg, rl_cfg, run_cfg, seed)
            eval_results[method].append((rate, prot))
            print(f"    → rate={rate:.2f}, protection={prot:.1f}%", flush=True)

        # AO baseline
        rate, prot = evaluate_thz_ao_baseline(sys_cfg, rl_cfg, run_cfg, seed)
        eval_results["baseline_ao"].append((rate, prot))
        print(f"  Convergence: seed {run_idx+1}/{run_cfg.n_seeds}, baseline_ao → rate={rate:.2f}, prot={prot:.1f}%", flush=True)

    return {
        "histories": all_histories,
        "eval": {
            m: {
                "rate_mean": float(np.mean([r for r, _ in v])),
                "rate_std": float(np.std([r for r, _ in v])),
                "protection_mean": float(np.mean([p for _, p in v])),
                "protection_std": float(np.std([p for _, p in v])),
                "rates": [r for r, _ in v],
                "protections": [p for _, p in v],
            }
            for m, v in eval_results.items()
        },
    }


# ============================================================================
# Plot: Convergence with confidence bands
# ============================================================================

def plot_convergence_bands(conv_data: dict, out_dir: Path) -> None:
    """Convergence plot with mean ± std shading across seeds."""
    fig, ax = plt.subplots(figsize=(8, 5))

    window = 25
    for method in RL_METHODS:
        histories = conv_data["histories"][method]
        if not histories:
            continue

        arr = np.array(histories)  # (n_seeds, n_episodes)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)

        x_ma, mean_ma = moving_average(mean_curve, window)
        _, std_ma = moving_average(std_curve, window)

        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]

        # Raw traces (faint)
        for seed_curve in histories:
            ax.plot(np.arange(1, len(seed_curve)+1), seed_curve,
                    linewidth=0.4, alpha=0.08, color=color)

        # Smoothed mean
        ax.plot(x_ma, mean_ma, label=label, linewidth=2.2, color=color)
        # Confidence band (±1 std)
        if arr.shape[0] > 1:
            ax.fill_between(x_ma, mean_ma - std_ma, mean_ma + std_ma,
                           alpha=0.15, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward per Episode")
    ax.set_title("THz Wideband Anti-Jamming: RL Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_convergence.png")
    plt.close(fig)
    print(f"  Saved: fig_convergence.png")


# ============================================================================
# Plot: Evaluation bar chart (rate + protection)
# ============================================================================

def plot_evaluation_bars(conv_data: dict, out_dir: Path) -> None:
    """Bar chart with error bars for trained agent evaluation."""
    eval_data = conv_data["eval"]
    methods = RL_METHODS + ["baseline_ao"]
    methods = [m for m in methods if m in eval_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(methods))
    width = 0.6

    rates = [eval_data[m]["rate_mean"] for m in methods]
    rate_errs = [eval_data[m]["rate_std"] for m in methods]
    prots = [eval_data[m]["protection_mean"] for m in methods]
    prot_errs = [eval_data[m]["protection_std"] for m in methods]
    colors = [METHOD_COLORS.get(m, "gray") for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    bars1 = ax1.bar(x, rates, width, yerr=rate_errs, capsize=4,
                    color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("System Rate (bits/s/Hz)")
    ax1.set_title("Average System Rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax1.grid(True, alpha=0.2, axis="y")
    for bar, val in zip(bars1, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(x, prots, width, yerr=prot_errs, capsize=4,
                    color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("SINR Protection Level (%)")
    ax2.set_title("SINR Protection Level")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax2.grid(True, alpha=0.2, axis="y")
    ax2.set_ylim(0, 105)
    for bar, val in zip(bars2, prots):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("THz Anti-Jamming: Trained Agent Evaluation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_evaluation.png")
    plt.close(fig)
    print(f"  Saved: fig_evaluation.png")


# ============================================================================
# Plot: Parameter sweep with error bars
# ============================================================================

def plot_sweep(sweep_data: dict, xlabel: str, title_tag: str,
               out_path: Path) -> None:
    """Sweep plot with error bars from multiple seeds."""
    x = sweep_data["x"]
    param = sweep_data.get("parameter", "")

    # Convert bandwidth from Hz to GHz for display
    if "bandwidth" in param.lower():
        x = [v / 1e9 for v in x]
        xlabel = "Bandwidth (GHz)"
    # Convert N_RIS to log-friendly label
    if "n_ris" in param.lower():
        xlabel = r"$N_\mathrm{RIS}$ (elements)"

    methods = sweep_data["methods"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for method, stats in methods.items():
        color = METHOD_COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)
        marker = METHOD_MARKERS.get(method, "o")

        rate_mean = stats.get("rate_mean", stats["rate"])
        prot_mean = stats.get("protection_mean", stats["protection"])
        rate_std = stats.get("rate_std", None)
        prot_std = stats.get("protection_std", None)

        if rate_std and any(s > 0 for s in rate_std):
            ax1.errorbar(x, rate_mean, yerr=rate_std, marker=marker, linewidth=2,
                        label=label, color=color, capsize=3, capthick=1.5, markersize=6)
        else:
            ax1.plot(x, rate_mean, marker=marker, linewidth=2, label=label, color=color, markersize=6)

        if prot_std and any(s > 0 for s in prot_std):
            ax2.errorbar(x, prot_mean, yerr=prot_std, marker=marker, linewidth=2,
                        label=label, color=color, capsize=3, capthick=1.5, markersize=6)
        else:
            ax2.plot(x, prot_mean, marker=marker, linewidth=2, label=label, color=color, markersize=6)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Average System Rate (bits/s/Hz)")
    ax1.set_title(f"{title_tag} — Rate")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("SINR Protection (%)")
    ax2.set_title(f"{title_tag} — Protection")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Use log scale for N_RIS
    if "n_ris" in param.lower():
        ax1.set_xscale("log", base=2)
        ax2.set_xscale("log", base=2)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ============================================================================
# Plot: Beam squint analysis
# ============================================================================

def plot_beam_squint(data: dict[str, np.ndarray], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {
        "Classical (no TD)": {"ls": "--", "lw": 2.5, "color": "#d62728"},
        "SPDP Q=1": {"ls": ":", "lw": 2.0, "color": "#ff7f0e"},
        "SPDP Q=16": {"ls": "-.", "lw": 2.0, "color": "#2ca02c"},
        "SPDP Q=64": {"ls": "-", "lw": 2.5, "color": "#1f77b4"},
    }
    for scheme, gains in data.items():
        x = np.arange(len(gains))
        s = styles.get(scheme, {"ls": "-", "lw": 2, "color": None})
        ax.plot(x, gains, label=scheme, **s)

    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Normalized Array Gain")
    ax.set_title("Beam Squint Compensation: SPDP vs Classical Phase-Only RIS")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_beam_squint.png")
    plt.close(fig)
    print(f"  Saved: fig_beam_squint.png")


# ============================================================================
# Plot: Combined summary (4 subplots)
# ============================================================================

def plot_summary(conv_data: dict, sweep_pmax: dict, sweep_nris: dict,
                 out_dir: Path) -> None:
    """4-panel summary figure for paper overview."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Convergence
    ax = axes[0, 0]
    for method in RL_METHODS:
        histories = conv_data["histories"][method]
        if not histories:
            continue
        mean_curve = np.mean(histories, axis=0)
        x_ma, y_ma = moving_average(mean_curve, 25)
        ax.plot(x_ma, y_ma, label=METHOD_LABELS[method], linewidth=2,
                color=METHOD_COLORS[method])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("(a) RL Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (b) Evaluation bars
    ax = axes[0, 1]
    eval_data = conv_data["eval"]
    methods = [m for m in RL_METHODS + ["baseline_ao"] if m in eval_data]
    x_pos = np.arange(len(methods))
    rates = [eval_data[m]["rate_mean"] for m in methods]
    colors = [METHOD_COLORS.get(m, "gray") for m in methods]
    labels = [METHOD_LABELS.get(m, m).split(" (")[0].split(" [")[0] for m in methods]
    ax.bar(x_pos, rates, 0.6, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("System Rate (bits/s/Hz)")
    ax.set_title("(b) Trained Agent Rate")
    ax.grid(True, alpha=0.2, axis="y")

    # (c) Pmax sweep
    ax = axes[1, 0]
    for method, stats in sweep_pmax["methods"].items():
        rate_mean = stats.get("rate_mean", stats["rate"])
        ax.plot(sweep_pmax["x"], rate_mean, marker=METHOD_MARKERS.get(method, "o"),
                linewidth=2, label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, None), markersize=5)
    ax.set_xlabel(r"$P_\mathrm{max}$ (dBm)")
    ax.set_ylabel("System Rate (bits/s/Hz)")
    ax.set_title(r"(c) Rate vs $P_\mathrm{max}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # (d) N_RIS sweep
    ax = axes[1, 1]
    x_ris = sweep_nris["x"]
    for method, stats in sweep_nris["methods"].items():
        rate_mean = stats.get("rate_mean", stats["rate"])
        ax.plot(x_ris, rate_mean, marker=METHOD_MARKERS.get(method, "o"),
                linewidth=2, label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, None), markersize=5)
    ax.set_xlabel(r"$N_\mathrm{RIS}$ (elements)")
    ax.set_ylabel("System Rate (bits/s/Hz)")
    ax.set_title(r"(d) Rate vs $N_\mathrm{RIS}$")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle("THz RIS-Aided Anti-Jamming: Simulation Results", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig_summary.png")
    plt.close(fig)
    print(f"  Saved: fig_summary.png")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper-quality THz simulations")
    parser.add_argument("--profile", default="paper", choices=PROFILES.keys())
    parser.add_argument("--output-dir", default="outputs_paper")
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--convergence-only", action="store_true")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0_global = time.time()

    print(f"\n{'='*70}")
    print(f"  THz Paper-Quality Simulations (profile={args.profile})")
    print(f"{'='*70}")
    print(f"  BS:  {profile['n_bs_antennas']} antennas, {profile['n_rf_chains']} RF chains")
    print(f"  RIS: {profile['n_ris_h']}x{profile['n_ris_v']} = {profile['n_ris_h']*profile['n_ris_v']} elements (SPDP)")
    print(f"  OFDM: {profile['n_subcarriers']} subcarriers, stride={profile['subcarrier_stride']}")
    print(f"  Training: {profile['train_episodes']} episodes x {profile['train_steps_per_episode']} steps, {profile['n_seeds']} seeds")
    print(f"  Evaluation: {profile['eval_episodes']} episodes x {profile['eval_steps_per_episode']} steps")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}\n")

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

    results = {
        "profile": args.profile,
        "config": {
            "n_bs_antennas": sys_cfg.n_bs_antennas,
            "n_rf_chains": sys_cfg.n_rf_chains,
            "n_ris": sys_cfg.n_ris_total,
            "n_subcarriers": sys_cfg.n_subcarriers,
            "bandwidth_ghz": sys_cfg.bandwidth_hz / 1e9,
            "center_freq_ghz": sys_cfg.center_freq_hz / 1e9,
            "pmax_dbm": sys_cfg.pmax_dbm,
            "sinr_min_db": sys_cfg.sinr_min_db,
            "noise_power_dbm": sys_cfg.noise_power_dbm,
            "n_actions": 42,
            "train_episodes": run_cfg.train_episodes,
            "n_seeds": run_cfg.n_seeds,
        },
    }

    # ---- Beam Squint Analysis ----
    print(">>> Beam Squint Analysis")
    t0 = time.time()
    bs_data = run_beam_squint_analysis(sys_cfg)
    plot_beam_squint(bs_data, out_dir)
    results["beam_squint"] = {k: v.tolist() for k, v in bs_data.items()}
    print(f"    Done ({elapsed_str(t0)})\n")

    if args.convergence_only:
        # Quick: just convergence, no sweeps
        args.skip_sweeps = True

    # ---- Convergence with confidence bands ----
    print(">>> Convergence Experiment (with per-seed histories)")
    t0 = time.time()
    conv_data = run_convergence_with_bands(sys_cfg, rl_cfg, run_cfg)
    plot_convergence_bands(conv_data, out_dir)
    plot_evaluation_bars(conv_data, out_dir)
    results["convergence"] = {
        "histories": {m: [list(h) for h in hs] for m, hs in conv_data["histories"].items()},
        "eval": conv_data["eval"],
    }
    print(f"    Done ({elapsed_str(t0)})\n")

    # ---- Parameter Sweeps ----
    sweep_pmax = None
    sweep_nris = None

    if not args.skip_sweeps:
        sweep_cfg = THzSweepConfig()

        # Sweep: P_max
        print(">>> Sweep: P_max")
        t0 = time.time()
        sweep_pmax = run_thz_parameter_sweep(
            "pmax_dbm", sweep_cfg.pmax_dbm_values, sys_cfg, rl_cfg, run_cfg)
        plot_sweep(sweep_pmax, r"$P_\mathrm{max}$ (dBm)", "THz: Transmit Power",
                   out_dir / "fig_sweep_pmax.png")
        results["sweep_pmax"] = sweep_pmax
        print(f"    Done ({elapsed_str(t0)})\n")

        # Sweep: N_RIS
        print(">>> Sweep: N_RIS")
        t0 = time.time()
        sweep_nris = run_thz_parameter_sweep(
            "n_ris_total", sweep_cfg.n_ris_elements_values, sys_cfg, rl_cfg, run_cfg)
        plot_sweep(sweep_nris, r"$N_\mathrm{RIS}$", "THz: RIS Size",
                   out_dir / "fig_sweep_nris.png")
        results["sweep_nris"] = sweep_nris
        print(f"    Done ({elapsed_str(t0)})\n")

        # Sweep: Bandwidth
        print(">>> Sweep: Bandwidth")
        t0 = time.time()
        sweep_bw = run_thz_parameter_sweep(
            "bandwidth_hz", sweep_cfg.bandwidth_values_hz, sys_cfg, rl_cfg, run_cfg)
        plot_sweep(sweep_bw, "Bandwidth (GHz)", "THz: Bandwidth (Beam Squint Impact)",
                   out_dir / "fig_sweep_bandwidth.png")
        results["sweep_bandwidth"] = sweep_bw
        print(f"    Done ({elapsed_str(t0)})\n")

        # Sweep: SINR target
        print(">>> Sweep: SINR Target")
        t0 = time.time()
        sweep_sinr = run_thz_parameter_sweep(
            "sinr_min_db", sweep_cfg.sinr_target_db_values, sys_cfg, rl_cfg, run_cfg)
        plot_sweep(sweep_sinr, r"$\gamma_\mathrm{min}$ (dB)", "THz: SINR Target",
                   out_dir / "fig_sweep_sinr.png")
        results["sweep_sinr"] = sweep_sinr
        print(f"    Done ({elapsed_str(t0)})\n")

        # Summary 4-panel figure
        if sweep_pmax and sweep_nris:
            plot_summary(conv_data, sweep_pmax, sweep_nris, out_dir)

    # ---- Save all results ----
    results_path = out_dir / "paper_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n>>> Results saved to: {results_path}")

    total_time = time.time() - t0_global
    print(f"\n{'='*70}")
    print(f"  ALL DONE — Total time: {elapsed_str(t0_global)}")
    print(f"  Outputs in: {out_dir}")
    print(f"  Figures: {len(list(out_dir.glob('fig_*.png')))} files")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
