#!/usr/bin/env python3
"""Fast paper-quality THz simulation: convergence + 4 sweeps + beam squint.

Designed to complete in ~15-20 minutes on a MacBook Air by:
  - Using fast_mode SPDP during training (centroid only)
  - Using 32-BS/64-RIS quick config for sweeps (parameters varied anyway)
  - Using 64-BS/256-RIS for convergence (the showpiece figure)
  - Minimizing sweep points to essentials (4-5 per sweep)
  - 3 seeds everywhere for error bars
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from irs_anti_jamming.thz.thz_config import (
    THzRLConfig, THzSweepConfig, THzSystemConfig, THzTrainEvalConfig,
)
from irs_anti_jamming.thz.thz_experiments import (
    RL_METHODS,
    evaluate_thz_agent,
    evaluate_thz_ao_baseline,
    run_beam_squint_analysis,
    train_thz_agent,
)

# ═══════════════════════════════════════════════════════════════════
# Configurations
# ═══════════════════════════════════════════════════════════════════

# Convergence: higher quality (the main figure)
CONV_SYS = THzSystemConfig(
    n_bs_antennas=64, n_rf_chains=8,
    n_ris_h=16, n_ris_v=16,
    q_subarrays_h=4, q_subarrays_v=4,
    n_subcarriers=64, subcarrier_stride=8,
)
CONV_RUN = THzTrainEvalConfig(
    train_episodes=300, train_steps_per_episode=15,
    eval_episodes=20, eval_steps_per_episode=10,
    n_seeds=3,
)

# Sweeps: use SAME array config as convergence (parameter being swept is the only variable)
SWEEP_SYS = THzSystemConfig(
    n_bs_antennas=64, n_rf_chains=8,
    n_ris_h=16, n_ris_v=16,
    q_subarrays_h=4, q_subarrays_v=4,
    n_subcarriers=64, subcarrier_stride=8,
)
SWEEP_RUN = THzTrainEvalConfig(
    train_episodes=200, train_steps_per_episode=10,
    eval_episodes=15, eval_steps_per_episode=8,
    n_seeds=2,
)

RL_CFG = THzRLConfig()

# Sweep values (minimal but meaningful)
SWEEP_PMAX = [20.0, 30.0, 40.0, 45.0]
SWEEP_NRIS = [16, 64, 256, 1024]
SWEEP_BW = [0.5e9, 2.0e9, 5.0e9, 10.0e9]
SWEEP_SINR = [0.0, 5.0, 10.0, 15.0]

ALL_METHODS = RL_METHODS + ["baseline_ao"]

# ═══════════════════════════════════════════════════════════════════
# Plot styling
# ═══════════════════════════════════════════════════════════════════

METHOD_LABELS = {
    "q_learning": "Classical Q-Learning",
    "fast_q_learning": "Fast Q-Learning [19]",
    "fuzzy_wolf_phc": "Fuzzy WoLF-PHC (Proposed)",
    "dqn": "DQN",
    "baseline_ao": "AO Baseline [39]",
}
METHOD_COLORS = {
    "q_learning": "#e74c3c",
    "fast_q_learning": "#e67e22",
    "fuzzy_wolf_phc": "#2980b9",
    "dqn": "#27ae60",
    "baseline_ao": "#8e44ad",
}
METHOD_MARKERS = {
    "q_learning": "s",
    "fast_q_learning": "^",
    "fuzzy_wolf_phc": "o",
    "dqn": "D",
    "baseline_ao": "v",
}


def _ma(y, w=20):
    """Moving average."""
    if y.size < w:
        return np.arange(1, y.size + 1, dtype=float), y
    k = np.ones(w) / w
    s = np.convolve(y, k, mode="valid")
    return np.arange(w, y.size + 1, dtype=float), s


# ═══════════════════════════════════════════════════════════════════
# Helper: run one sweep (train + eval for each method/seed/value)
# ═══════════════════════════════════════════════════════════════════

def run_sweep(param_name, values, sys_cfg, rl_cfg, run_cfg, cfg_mutator):
    """Run a parameter sweep. Returns structured results with stats."""
    out = {
        "x": [float(v) for v in values],
        "parameter": param_name,
        "methods": {m: {"rate_mean": [], "rate_std": [], "prot_mean": [], "prot_std": []}
                    for m in ALL_METHODS},
    }
    for vi, val in enumerate(values):
        cfg = cfg_mutator(sys_cfg, val)
        print(f"    {param_name}={val} ({vi+1}/{len(values)})", flush=True)

        seed_rates = {m: [] for m in ALL_METHODS}
        seed_prots = {m: [] for m in ALL_METHODS}

        for si in range(run_cfg.n_seeds):
            seed = cfg.seed + 211 * si
            for method in ALL_METHODS:
                if method == "baseline_ao":
                    r, p = evaluate_thz_ao_baseline(cfg, rl_cfg, run_cfg, seed)
                else:
                    agent, _ = train_thz_agent(method, cfg, rl_cfg, run_cfg, seed,
                                               log_interval=0)
                    r, p = evaluate_thz_agent(agent, method, cfg, rl_cfg, run_cfg, seed)
                seed_rates[method].append(r)
                seed_prots[method].append(p)

        for m in ALL_METHODS:
            out["methods"][m]["rate_mean"].append(float(np.mean(seed_rates[m])))
            out["methods"][m]["rate_std"].append(float(np.std(seed_rates[m])))
            out["methods"][m]["prot_mean"].append(float(np.mean(seed_prots[m])))
            out["methods"][m]["prot_std"].append(float(np.std(seed_prots[m])))
    return out


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    out_dir = PROJECT_ROOT / "outputs_paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.time()

    print("=" * 70)
    print("  THz Paper-Quality Simulation (fast version)")
    print("=" * 70)
    print(f"  Convergence: {CONV_SYS.n_bs_antennas} BS, {CONV_SYS.n_ris_total} RIS, "
          f"{CONV_RUN.train_episodes} ep × {CONV_RUN.n_seeds} seeds")
    print(f"  Sweeps:      {SWEEP_SYS.n_bs_antennas} BS, {SWEEP_SYS.n_ris_total} RIS, "
          f"{SWEEP_RUN.train_episodes} ep × {SWEEP_RUN.n_seeds} seeds")
    print(f"  Output: {out_dir}")
    print("=" * 70, flush=True)

    # ──────────────────────────────────────────────────────────────
    # 1. BEAM SQUINT ANALYSIS (instant)
    # ──────────────────────────────────────────────────────────────
    print("\n[1/6] Beam Squint Analysis...", flush=True)
    bs_data = run_beam_squint_analysis(CONV_SYS)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {
        "Classical (no TD)": ("--", "red", 2.5),
        "SPDP Q=1": (":", "orange", 2.0),
        "SPDP Q=16": ("-.", "green", 2.0),
        "SPDP Q=64": ("-", "blue", 2.5),
    }
    for scheme, gains in bs_data.items():
        ls, c, lw = styles.get(scheme, ("-", "gray", 1.5))
        ax.plot(np.arange(len(gains)), gains, ls=ls, color=c, linewidth=lw, label=scheme)
    ax.set_xlabel("Subcarrier Index", fontsize=12)
    ax.set_ylabel("Normalized Array Gain", fontsize=12)
    ax.set_title("Beam Squint Compensation: SPDP vs Classical RIS", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_beam_squint.png", dpi=250)
    plt.close(fig)
    results["beam_squint"] = {k: v.tolist() for k, v in bs_data.items()}
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

    # ──────────────────────────────────────────────────────────────
    # 2. CONVERGENCE (main figure — 3 seeds, error bands)
    # ──────────────────────────────────────────────────────────────
    print("\n[2/6] Convergence Experiment...", flush=True)
    t_conv = time.time()

    # Collect per-seed histories for error bands
    all_histories = {m: [] for m in RL_METHODS}
    eval_results = {m: {"rates": [], "prots": []} for m in ALL_METHODS}

    for si in range(CONV_RUN.n_seeds):
        seed = CONV_SYS.seed + 101 * si
        print(f"  Seed {si+1}/{CONV_RUN.n_seeds}:", flush=True)
        for method in RL_METHODS:
            print(f"    {method}...", end=" ", flush=True)
            agent, history = train_thz_agent(
                method, CONV_SYS, RL_CFG, CONV_RUN, seed, log_interval=100)
            all_histories[method].append(history)
            r, p = evaluate_thz_agent(agent, method, CONV_SYS, RL_CFG, CONV_RUN, seed)
            eval_results[method]["rates"].append(r)
            eval_results[method]["prots"].append(p)
            print(f"rate={r:.2f}, prot={p:.1f}%", flush=True)

        # AO baseline
        print(f"    baseline_ao...", end=" ", flush=True)
        r, p = evaluate_thz_ao_baseline(CONV_SYS, RL_CFG, CONV_RUN, seed)
        eval_results["baseline_ao"]["rates"].append(r)
        eval_results["baseline_ao"]["prots"].append(p)
        print(f"rate={r:.2f}, prot={p:.1f}%", flush=True)

    print(f"  Convergence done ({time.time()-t_conv:.0f}s)", flush=True)

    # --- Convergence plot with error bands ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in RL_METHODS:
        stacked = np.stack(all_histories[method], axis=0)  # (n_seeds, n_episodes)
        mean_h = np.mean(stacked, axis=0)
        std_h = np.std(stacked, axis=0)
        x_ma, y_ma = _ma(mean_h, w=20)
        _, std_ma = _ma(std_h, w=20)
        color = METHOD_COLORS[method]
        ax.plot(x_ma, y_ma, label=METHOD_LABELS[method], linewidth=2.2, color=color)
        ax.fill_between(x_ma, y_ma - std_ma, y_ma + std_ma, alpha=0.15, color=color)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average Reward per Episode", fontsize=12)
    ax.set_title("THz Wideband Anti-Jamming: RL Convergence (64 BS, 256 RIS)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_convergence.png", dpi=250)
    plt.close(fig)

    # --- Evaluation bar chart ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    methods_order = ["q_learning", "fast_q_learning", "fuzzy_wolf_phc", "dqn", "baseline_ao"]
    x_pos = np.arange(len(methods_order))
    for i, m in enumerate(methods_order):
        rates = eval_results[m]["rates"]
        prots = eval_results[m]["prots"]
        color = METHOD_COLORS[m]
        ax1.bar(i, np.mean(rates), yerr=np.std(rates), color=color,
                capsize=5, edgecolor="black", linewidth=0.5)
        ax1.text(i, np.mean(rates) + np.std(rates) + 0.2, f"{np.mean(rates):.1f}",
                ha="center", fontsize=9, fontweight="bold")
        ax2.bar(i, np.mean(prots), yerr=np.std(prots), color=color,
                capsize=5, edgecolor="black", linewidth=0.5)
        ax2.text(i, np.mean(prots) + np.std(prots) + 1, f"{np.mean(prots):.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([METHOD_LABELS[m] for m in methods_order], rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("Average System Rate (bits/s/Hz)", fontsize=11)
    ax1.set_title("Trained Agent: System Rate", fontsize=12)
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([METHOD_LABELS[m] for m in methods_order], rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("SINR Protection Level (%)", fontsize=11)
    ax2.set_title("Trained Agent: SINR Protection", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")
    fig.suptitle("THz Anti-Jamming: Trained Agent Evaluation", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_evaluation.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Store convergence results
    results["convergence"] = {
        m: np.mean(np.stack(all_histories[m], axis=0), axis=0).tolist()
        for m in RL_METHODS
    }
    results["convergence_per_seed"] = {
        m: [h.tolist() for h in all_histories[m]] for m in RL_METHODS
    }
    results["evaluation"] = {
        m: {"rate_mean": float(np.mean(eval_results[m]["rates"])),
            "rate_std": float(np.std(eval_results[m]["rates"])),
            "prot_mean": float(np.mean(eval_results[m]["prots"])),
            "prot_std": float(np.std(eval_results[m]["prots"])),
            "rates": eval_results[m]["rates"],
            "prots": eval_results[m]["prots"]}
        for m in ALL_METHODS
    }

    # ──────────────────────────────────────────────────────────────
    # 3-6. PARAMETER SWEEPS
    # ──────────────────────────────────────────────────────────────
    def mutate_pmax(cfg, val):
        return replace(cfg, pmax_dbm=float(val))

    def mutate_nris(cfg, val):
        side = int(math.isqrt(int(val)))
        q_side = max(1, side // 4)
        return replace(cfg, n_ris_h=side, n_ris_v=side,
                      q_subarrays_h=q_side, q_subarrays_v=q_side)

    def mutate_bw(cfg, val):
        return replace(cfg, bandwidth_hz=float(val))

    def mutate_sinr(cfg, val):
        return replace(cfg, sinr_min_db=float(val))

    sweeps = [
        ("pmax_dbm", SWEEP_PMAX, mutate_pmax,
         "$P_{max}$ (dBm)", "Rate vs Transmit Power"),
        ("n_ris_total", SWEEP_NRIS, mutate_nris,
         "$N_{RIS}$ (elements)", "Rate vs RIS Size"),
        ("bandwidth_hz", SWEEP_BW, mutate_bw,
         "Bandwidth (GHz)", "Rate vs Bandwidth"),
        ("sinr_min_db", SWEEP_SINR, mutate_sinr,
         "SINR Target (dB)", "Rate vs SINR Target"),
    ]

    for idx, (param, values, mutator, xlabel, title_base) in enumerate(sweeps, start=3):
        print(f"\n[{idx}/6] Sweep: {param}...", flush=True)
        t_sw = time.time()
        sweep_data = run_sweep(param, values, SWEEP_SYS, RL_CFG, SWEEP_RUN, mutator)
        print(f"  Done ({time.time()-t_sw:.0f}s)", flush=True)

        # Plot
        x_vals = sweep_data["x"]
        if "bandwidth" in param:
            x_vals = [v / 1e9 for v in x_vals]  # Hz → GHz

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for m in ALL_METHODS:
            color = METHOD_COLORS[m]
            marker = METHOD_MARKERS[m]
            label = METHOD_LABELS[m]
            d = sweep_data["methods"][m]
            ax1.errorbar(x_vals, d["rate_mean"], yerr=d["rate_std"],
                        marker=marker, linewidth=2, label=label, color=color,
                        capsize=4, capthick=1.5, markersize=7)
            ax2.errorbar(x_vals, d["prot_mean"], yerr=d["prot_std"],
                        marker=marker, linewidth=2, label=label, color=color,
                        capsize=4, capthick=1.5, markersize=7)
        ax1.set_xlabel(xlabel, fontsize=12)
        ax1.set_ylabel("System Rate (bits/s/Hz)", fontsize=11)
        ax1.set_title(f"THz: {title_base}", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        ax2.set_xlabel(xlabel, fontsize=12)
        ax2.set_ylabel("SINR Protection (%)", fontsize=11)
        ax2.set_title(f"THz: Protection vs {xlabel}", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        fig.tight_layout()

        fig_name = f"fig_sweep_{param}.png"
        fig.savefig(out_dir / fig_name, dpi=250)
        plt.close(fig)
        print(f"  Saved: {fig_name}", flush=True)

        results[f"sweep_{param}"] = sweep_data

    # ──────────────────────────────────────────────────────────────
    # Save all results
    # ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    results["meta"] = {
        "convergence_config": {
            "n_bs": CONV_SYS.n_bs_antennas, "n_ris": CONV_SYS.n_ris_total,
            "episodes": CONV_RUN.train_episodes, "seeds": CONV_RUN.n_seeds,
        },
        "sweep_config": {
            "n_bs": SWEEP_SYS.n_bs_antennas, "n_ris": SWEEP_SYS.n_ris_total,
            "episodes": SWEEP_RUN.train_episodes, "seeds": SWEEP_RUN.n_seeds,
        },
        "total_time_seconds": elapsed,
    }
    with open(out_dir / "paper_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  DONE in {elapsed/60:.1f} minutes")
    print(f"  Output: {out_dir}")
    print(f"  Files: fig_beam_squint.png, fig_convergence.png, fig_evaluation.png")
    print(f"         fig_sweep_pmax_dbm.png, fig_sweep_n_ris_total.png")
    print(f"         fig_sweep_bandwidth_hz.png, fig_sweep_sinr_min_db.png")
    print(f"         paper_results.json")
    print(f"{'='*70}")

    # Print summary table
    print("\n  === Evaluation Summary ===")
    print(f"  {'Method':<30} {'Rate':>10} {'Protection':>12}")
    print(f"  {'-'*52}")
    for m in methods_order:
        d = results["evaluation"][m]
        print(f"  {METHOD_LABELS[m]:<30} {d['rate_mean']:>8.2f}±{d['rate_std']:.2f} "
              f"{d['prot_mean']:>9.1f}±{d['prot_std']:.1f}%")


if __name__ == "__main__":
    main()
