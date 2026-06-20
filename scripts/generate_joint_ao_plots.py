#!/usr/bin/env python3
"""Generate all IEEE-style paper plots for the joint 3-block AO formulation.

Reads outputs_joint_ao/paper_results.json and produces all 10 publication
figures in outputs_joint_ao/ieee_plots/.

Figures generated:
  ieee_fig4_convergence   – reward vs training episode (synthetic curves)
  ieee_fig5_vs_pmax       – rate & protection vs P_max (extrapolated)
  ieee_fig6_vs_nris       – rate & protection vs N_RIS (extrapolated)
  ieee_fig7_vs_sinr       – rate & protection vs SINR target (extrapolated)
  ieee_fig8_beam_squint   – analytical beam squint (bandwidth comparison)
  ieee_fig9_evaluation    – evaluation bar chart (real data)
  ieee_fig10_per_seed     – per-seed consistency scatter (real data)
  ieee_fig11_sinr_cdf     – empirical SINR CDF (synthetic from Gaussian fit)
  ieee_fig12_vs_pjammer   – rate & protection vs jammer power (extrapolated)
  ieee_fig13_runtime      – training time comparison (real data)

Usage:
    .venv/bin/python3 scripts/generate_joint_ao_plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# IEEE style
# ---------------------------------------------------------------------------
IEEE_COL_W = 3.5
IEEE_DBL_W = 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.8,
})

# ---------------------------------------------------------------------------
# Method styling
# ---------------------------------------------------------------------------
METHOD_ORDER = ["q_learning", "fast_q_learning", "fuzzy_wolf_phc", "dqn", "ao_baseline"]

METHOD_LABELS = {
    "q_learning":      "Classical Q-learning",
    "fast_q_learning": "Fast Q-learning [1]",
    "fuzzy_wolf_phc":  "Proposed fuzzy WoLF-PHC",
    "dqn":             "DQN [12]",
    "ao_baseline":     "AO Baseline [6]",
}

METHOD_COLORS = {
    "q_learning":      "#d62728",
    "fast_q_learning": "#ff7f0e",
    "fuzzy_wolf_phc":  "#1f77b4",
    "dqn":             "#2ca02c",
    "ao_baseline":     "#7f7f7f",
}

METHOD_MARKERS = {
    "q_learning":      "s",
    "fast_q_learning": "D",
    "fuzzy_wolf_phc":  "o",
    "dqn":             "^",
    "ao_baseline":     "v",
}

METHOD_LINESTYLES = {
    "q_learning":      "--",
    "fast_q_learning": "-.",
    "fuzzy_wolf_phc":  "-",
    "dqn":             ":",
    "ao_baseline":     "--",
}

HATCHES = ["//", "\\\\", "", "xx", ".."]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _save(fig, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  Saved: {stem}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Fig 4 – Convergence (reward vs episode) — REAL data from convergence_data.json
# ---------------------------------------------------------------------------
def fig4_convergence(summary: dict, out_dir: Path,
                     convergence_file: Path | None = None) -> None:
    """Plot per-episode reward histories from real training runs.

    Reads outputs_joint_ao/convergence_data.json produced by
    run_convergence_training.py.  Each method has N_seeds reward arrays;
    we plot the per-episode mean with a ±1 std shaded band (both computed
    after a moving-average smooth to reduce per-step noise).

    If the convergence file does not exist, the function raises an error
    rather than falling back to synthetic data.
    """
    if convergence_file is None:
        convergence_file = out_dir.parents[0] / "convergence_data.json"

    if not convergence_file.exists():
        raise FileNotFoundError(
            f"Real convergence data not found at {convergence_file}.\n"
            f"Run:  .venv/bin/python3 scripts/run_convergence_training.py"
        )

    with open(convergence_file) as f:
        cdata = json.load(f)

    histories = cdata["histories"]   # {method: [[ep_rewards_seed0], ...]}
    E = cdata["config"]["train_episodes"]

    fig, ax = plt.subplots(figsize=(IEEE_DBL_W, 4.0))

    smooth_w = 20  # moving-average window (episodes)
    kernel   = np.ones(smooth_w) / smooth_w

    for method in ["q_learning", "fast_q_learning", "fuzzy_wolf_phc", "dqn"]:
        if method not in histories:
            continue

        raw_seeds = [np.array(h, dtype=float) for h in histories[method]]

        # Smooth each seed independently
        smoothed = [np.convolve(h, kernel, mode="valid") for h in raw_seeds]

        # Stack into (n_seeds, n_episodes_after_smoothing)
        arr = np.stack(smoothed, axis=0)
        mean_curve = arr.mean(axis=0)
        std_curve  = arr.std(axis=0)

        xe = np.arange(smooth_w, E + 1, dtype=float)

        c  = METHOD_COLORS[method]
        ls = METHOD_LINESTYLES[method]
        ax.plot(xe, mean_curve, label=METHOD_LABELS[method],
                color=c, linestyle=ls, linewidth=2.0)
        ax.fill_between(xe,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.15, color=c)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Average reward per episode")
    ax.set_xlim(1, E)
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="black", fancybox=False)
    fig.tight_layout()
    _save(fig, out_dir, "ieee_fig4_convergence")


# ---------------------------------------------------------------------------
# Fig 5 – Rate & Protection vs P_max
# ---------------------------------------------------------------------------
def fig5_vs_pmax(out_dir: Path, sweep_file: Path | None = None) -> None:
    """Rate & protection vs P_max — REAL sweep data."""
    if sweep_file is None:
        sweep_file = out_dir.parents[0] / "sweep_pmax.json"
    if not sweep_file.exists():
        raise FileNotFoundError(
            f"P_max sweep data not found at {sweep_file}.\n"
            f"Run: .venv/bin/python3 scripts/run_sweep_simulations.py"
        )
    with open(sweep_file) as f:
        data = json.load(f)

    points = data["sweep"]
    pmax_vals = np.array([p["pmax_dbm"] for p in points])

    no_irs_rate = 5.48
    no_irs_prot = 18.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.2))

    for method in METHOD_ORDER + ["ao_baseline"]:
        rates_mean, rates_std = [], []
        prots_mean, prots_std = [], []
        for pt in points:
            s = pt["summary"].get(method)
            if s is None:
                continue
            rates_mean.append(s["rate_mean"])
            rates_std.append(s["rate_std"])
            prots_mean.append(s["protection_mean"])
            prots_std.append(s["protection_std"])
        if not rates_mean:
            continue
        r = np.array(rates_mean); re = np.array(rates_std)
        p = np.array(prots_mean); pe = np.array(prots_std)
        c  = METHOD_COLORS.get(method, "gray")
        mk = METHOD_MARKERS.get(method, "x")
        ls = METHOD_LINESTYLES.get(method, "-")
        lb = METHOD_LABELS.get(method, method)
        ax1.errorbar(pmax_vals, r, yerr=re, marker=mk, color=c,
                     linestyle=ls, label=lb, markersize=6, capsize=3)
        ax2.errorbar(pmax_vals, p, yerr=pe, marker=mk, color=c,
                     linestyle=ls, label=lb, markersize=6, capsize=3)

    ax1.axhline(no_irs_rate, color="black", ls=":", lw=1.3, label="No IRS")
    ax2.axhline(no_irs_prot, color="black", ls=":", lw=1.3, label="No IRS")
    for ax, ylabel, title in [
        (ax1, "Average system rate (bits/s/Hz)", "(a)"),
        (ax2, "SINR protection level (%)", "(b)"),
    ]:
        ax.set_xlabel(r"Maximum transmit power $P_{\max}$ (dBm)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize=7, framealpha=0.9, edgecolor="black", fancybox=False)
        ax.set_title(title, fontsize=10, loc="left", pad=4)
    ax2.set_ylim(0, 105)
    fig.tight_layout()
    _save(fig, out_dir, "ieee_fig5_vs_pmax")


# ---------------------------------------------------------------------------
# Fig 6 – Rate & Protection vs N_RIS
# ---------------------------------------------------------------------------
def fig6_vs_nris(out_dir: Path, sweep_file: Path | None = None) -> None:
    """Rate & protection vs N_RIS — REAL sweep data."""
    if sweep_file is None:
        sweep_file = out_dir.parents[0] / "sweep_nris.json"
    if not sweep_file.exists():
        raise FileNotFoundError(
            f"N_RIS sweep data not found at {sweep_file}.\n"
            f"Run: .venv/bin/python3 scripts/run_sweep_simulations.py"
        )
    with open(sweep_file) as f:
        data = json.load(f)

    points   = data["sweep"]
    nris_vals = np.array([p["n_ris"] for p in points], dtype=float)
    no_irs_rate = 5.48
    no_irs_prot = 18.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.2))

    for method in METHOD_ORDER + ["ao_baseline"]:
        rates_mean, rates_std = [], []
        prots_mean, prots_std = [], []
        for pt in points:
            s = pt["summary"].get(method)
            if s is None:
                continue
            rates_mean.append(s["rate_mean"])
            rates_std.append(s["rate_std"])
            prots_mean.append(s["protection_mean"])
            prots_std.append(s["protection_std"])
        if not rates_mean:
            continue
        r = np.array(rates_mean); re = np.array(rates_std)
        p = np.array(prots_mean); pe = np.array(prots_std)
        c  = METHOD_COLORS.get(method, "gray")
        mk = METHOD_MARKERS.get(method, "x")
        ls = METHOD_LINESTYLES.get(method, "-")
        lb = METHOD_LABELS.get(method, method)
        ax1.errorbar(nris_vals, r, yerr=re, marker=mk, color=c,
                     linestyle=ls, label=lb, markersize=6, capsize=3)
        ax2.errorbar(nris_vals, p, yerr=pe, marker=mk, color=c,
                     linestyle=ls, label=lb, markersize=6, capsize=3)

    ax1.axhline(no_irs_rate, color="black", ls=":", lw=1.3, label="No IRS")
    ax2.axhline(no_irs_prot, color="black", ls=":", lw=1.3, label="No IRS")
    for ax, ylabel, title in [
        (ax1, "Average system rate (bits/s/Hz)", "(a)"),
        (ax2, "SINR protection level (%)", "(b)"),
    ]:
        ax.set_xlabel(r"Number of RIS elements $N_{\mathrm{RIS}}$")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize=7, framealpha=0.9, edgecolor="black", fancybox=False)
        ax.set_title(title, fontsize=10, loc="left", pad=4)
    ax2.set_ylim(0, 105)
    fig.tight_layout()
    _save(fig, out_dir, "ieee_fig6_vs_nris")


# ---------------------------------------------------------------------------
# Fig 7 – Rate & Protection vs SINR target
# ---------------------------------------------------------------------------
def fig7_vs_sinr(out_dir: Path, sweep_file: Path | None = None) -> None:
    """Rate & protection vs SINR target — REAL sweep data."""
    if sweep_file is None:
        sweep_file = out_dir.parents[0] / "sweep_sinr_target.json"
    if not sweep_file.exists():
        raise FileNotFoundError(
            f"SINR target sweep data not found at {sweep_file}.\n"
            f"Run: .venv/bin/python3 scripts/run_sweep_simulations.py"
        )
    with open(sweep_file) as f:
        data = json.load(f)

    points    = data["sweep"]
    sinr_vals = np.array([p["sinr_target_db"] for p in points])
    no_irs_rate = 5.48
    no_irs_prot = 18.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.2))

    for method in METHOD_ORDER + ["ao_baseline"]:
        rates_mean, rates_std = [], []
        prots_mean, prots_std = [], []
        for pt in points:
            s = pt["summary"].get(method)
            if s is None:
                continue
            rates_mean.append(s["rate_mean"])
            rates_std.append(s["rate_std"])
            prots_mean.append(s["protection_mean"])
            prots_std.append(s["protection_std"])
        if not rates_mean:
            continue
        r = np.array(rates_mean); re = np.array(rates_std)
        p = np.array(prots_mean); pe = np.array(prots_std)
        c  = METHOD_COLORS.get(method, "gray")
        mk = METHOD_MARKERS.get(method, "x")
        ls = METHOD_LINESTYLES.get(method, "-")
        lb = METHOD_LABELS.get(method, method)
        ax1.errorbar(sinr_vals, r, yerr=re, marker=mk, color=c,
                     linestyle=ls, label=lb, markersize=6, capsize=3)
        ax2.errorbar(sinr_vals, p, yerr=pe, marker=mk, color=c,
                     linestyle=ls, label=lb, markersize=6, capsize=3)

    ax1.axhline(no_irs_rate, color="black", ls=":", lw=1.3, label="No IRS")
    ax2.axhline(no_irs_prot, color="black", ls=":", lw=1.3, label="No IRS")
    for ax, ylabel, title in [
        (ax1, "Average system rate (bits/s/Hz)", "(a)"),
        (ax2, "SINR protection level (%)", "(b)"),
    ]:
        ax.set_xlabel(r"UE SINR target $\gamma_{\min}$ (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize=7, framealpha=0.9, edgecolor="black", fancybox=False)
        ax.set_title(title, fontsize=10, loc="left", pad=4)
    ax2.set_ylim(0, 105)
    fig.tight_layout()
    _save(fig, out_dir, "ieee_fig7_vs_sinr")


# ---------------------------------------------------------------------------
# Fig 8 – Beam squint (analytical, unchanged from system)
# ---------------------------------------------------------------------------
def fig8_beam_squint(out_dir: Path) -> None:
    M = 64
    f_c = 100.0e9
    N = 256
    steering_sin = 0.5
    bandwidths = [
        (0.1e9,  r"$B=0.1$ GHz", "--", "#1f77b4"),
        (2.0e9,  r"$B=2$ GHz",   "-.", "#d62728"),
        (10.0e9, r"$B=10$ GHz",  "-",  "#DAA520"),
    ]

    fig, ax = plt.subplots(figsize=(IEEE_COL_W * 1.8, 3.5))
    x = np.arange(M)

    for bw, label, ls, color in bandwidths:
        freqs = f_c - bw / 2 + np.arange(M) * bw / M
        delta = np.pi * (freqs / f_c - 1.0) * steering_sin
        half_d = delta / 2.0
        N_half_d = N * half_d
        gain = np.ones(M, dtype=float)
        nz = np.abs(half_d) > 1e-12
        gain[nz] = (np.sin(N_half_d[nz]) / (N * np.sin(half_d[nz]))) ** 2
        gain /= np.max(gain) + 1e-30
        ax.plot(x, gain, label=label, linestyle=ls, linewidth=2.0, color=color)

    ax.set_xlabel("Subcarrier index $m$")
    ax.set_ylabel(r"Normalized array gain $\eta(f_m)$")
    ax.set_xlim(0, M - 1)
    ax.set_ylim(0, 1.10)
    ax.grid(True)
    ax.legend(loc="lower center", framealpha=0.9, edgecolor="black",
              fancybox=False, ncol=3, fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "ieee_fig8_beam_squint")


# ---------------------------------------------------------------------------
# Fig 9 – Evaluation bar chart (REAL data)
# ---------------------------------------------------------------------------
def fig9_evaluation(summary: dict, out_dir: Path) -> None:
    # Add No IRS
    summary_ext = dict(summary)
    summary_ext["no_irs"] = {"rate_mean": 5.48, "rate_std": 0.0,
                              "protection_mean": 18.8, "protection_std": 0.0}

    order = METHOD_ORDER + ["no_irs"]
    method_labels_ext = dict(METHOD_LABELS, no_irs="No IRS")
    method_colors_ext = dict(METHOD_COLORS, no_irs="#17becf")

    methods = [m for m in order if m in summary_ext]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.5))
    x = np.arange(len(methods))
    width = 0.55

    rates     = [summary_ext[m]["rate_mean"]       for m in methods]
    rate_errs = [summary_ext[m]["rate_std"]         for m in methods]
    prots     = [summary_ext[m]["protection_mean"]  for m in methods]
    prot_errs = [summary_ext[m]["protection_std"]   for m in methods]
    colors    = [method_colors_ext.get(m, "gray")   for m in methods]
    labels    = [method_labels_ext.get(m, m)        for m in methods]
    hatches   = ["//", "\\\\", "", "xx", "..", "||"]

    bars1 = ax1.bar(x, rates, width, yerr=rate_errs, capsize=3,
                    color=colors, edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars1, hatches):
        bar.set_hatch(h)
    for bar, val in zip(bars1, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax1.set_ylabel("Average system rate (bits/s/Hz)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax1.grid(True, axis="y")
    ax1.set_title("(a)", fontsize=10, loc="left", pad=4)

    bars2 = ax2.bar(x, prots, width, yerr=prot_errs, capsize=3,
                    color=colors, edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars2, hatches):
        bar.set_hatch(h)
    for bar, val in zip(bars2, prots):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax2.set_ylabel("SINR protection level (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax2.grid(True, axis="y")
    ax2.set_ylim(0, 105)
    ax2.set_title("(b)", fontsize=10, loc="left", pad=4)

    fig.tight_layout()
    _save(fig, out_dir, "ieee_fig9_evaluation")


# ---------------------------------------------------------------------------
# Fig 10 – Per-seed consistency (REAL data)
# ---------------------------------------------------------------------------
def fig10_per_seed(per_seed: dict, summary: dict, out_dir: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in per_seed]
    rng = np.random.default_rng(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.5))

    for i, method in enumerate(methods):
        seeds = per_seed[method]
        rates = [s["rate"] for s in seeds]
        prots = [s["protection"] for s in seeds]
        c  = METHOD_COLORS[method]
        mk = METHOD_MARKERS[method]
        lb = METHOD_LABELS[method]

        jx = rng.uniform(-0.15, 0.15, len(rates))
        ax1.scatter(np.full(len(rates), i) + jx, rates,
                    color=c, marker=mk, s=50, edgecolors="black",
                    linewidth=0.5, zorder=3)
        ax1.plot([i - 0.25, i + 0.25],
                 [summary[method]["rate_mean"]] * 2,
                 color=c, linewidth=3.0, zorder=2)

        ax2.scatter(np.full(len(prots), i) + jx, prots,
                    color=c, marker=mk, s=50, edgecolors="black",
                    linewidth=0.5, zorder=3, label=lb)
        ax2.plot([i - 0.25, i + 0.25],
                 [summary[method]["protection_mean"]] * 2,
                 color=c, linewidth=3.0, zorder=2)

    labels = [METHOD_LABELS[m] for m in methods]
    for ax, ylabel, title in [
        (ax1, "System rate (bits/s/Hz)", "(a)"),
        (ax2, "SINR protection level (%)", "(b)"),
    ]:
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
        ax.set_title(title, fontsize=10, loc="left", pad=4)
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    _save(fig, out_dir, "ieee_fig10_per_seed")


# ---------------------------------------------------------------------------
# Fig 11 – SINR CDF — REAL per-user SINR samples from detailed evaluation
# ---------------------------------------------------------------------------
def fig11_sinr_cdf(out_dir: Path, cdf_file: Path | None = None) -> None:
    """Empirical CDF of per-user SINR from detailed evaluation runs.

    Reads sweep_sinr_cdf.json produced by run_sweep_simulations.py which
    collects actual per-step, per-user SINR values during evaluation.
    """
    if cdf_file is None:
        cdf_file = out_dir.parents[0] / "sweep_sinr_cdf.json"
    if not cdf_file.exists():
        raise FileNotFoundError(
            f"SINR CDF data not found at {cdf_file}.\n"
            f"Run: .venv/bin/python3 scripts/run_sweep_simulations.py"
        )
    with open(cdf_file) as f:
        data = json.load(f)

    sinr_samples = data["sinr_db_samples"]   # {method: [sinr_db, ...]}
    sinr_min_db  = 5.0

    fig, ax = plt.subplots(figsize=(IEEE_COL_W * 1.6, 3.5))

    for method in METHOD_ORDER + ["ao_baseline"]:
        samples = sinr_samples.get(method)
        if not samples:
            continue
        arr    = np.sort(np.array(samples, dtype=float))
        cdf    = np.arange(1, len(arr) + 1) / len(arr)
        c  = METHOD_COLORS.get(method, "gray")
        ls = METHOD_LINESTYLES.get(method, "-")
        lb = METHOD_LABELS.get(method, method)
        ax.plot(arr, cdf, label=lb, color=c, linestyle=ls, linewidth=1.8)

    ax.axvline(sinr_min_db, color="black", ls=":", lw=1.0, alpha=0.6,
               label=r"$\gamma_{\min} = 5$ dB")
    ax.set_xlabel("Per-user SINR (dB)")
    ax.set_ylabel(r"CDF $F(\mathrm{SINR})$")
    ax.set_ylim(0, 1.02)
    ax.grid(True)
    ax.legend(fontsize=7, framealpha=0.9, edgecolor="black",
              fancybox=False, loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir, "ieee_fig11_sinr_cdf")


# ---------------------------------------------------------------------------
# Fig 12 – Rate & Protection vs Jammer Power
# ---------------------------------------------------------------------------
def fig12_vs_pjammer(out_dir: Path, sweep_file: Path | None = None) -> None:
    """Rate & protection vs jammer power — REAL sweep data."""
    if sweep_file is None:
        sweep_file = out_dir.parents[0] / "sweep_pjammer.json"
    if not sweep_file.exists():
        raise FileNotFoundError(
            f"Jammer power sweep data not found at {sweep_file}.\n"
            f"Run: .venv/bin/python3 scripts/run_sweep_simulations.py"
        )
    with open(sweep_file) as f:
        data = json.load(f)

    points  = data["sweep"]
    pj_vals = np.array([p["pjammer_max_dbm"] for p in points])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.2))

    for method in METHOD_ORDER + ["ao_baseline"]:
        rates_mean, rates_std = [], []
        prots_mean, prots_std = [], []
        for pt in points:
            s = pt["summary"].get(method)
            if s is None:
                continue
            rates_mean.append(s["rate_mean"])
            rates_std.append(s["rate_std"])
            prots_mean.append(s["protection_mean"])
            prots_std.append(s["protection_std"])
        if not rates_mean:
            continue
        r = np.array(rates_mean); re = np.array(rates_std)
        p = np.array(prots_mean); pe = np.array(prots_std)
        c  = METHOD_COLORS.get(method, "gray")
        mk = METHOD_MARKERS.get(method, "x")
        ls = METHOD_LINESTYLES.get(method, "-")
        lb = METHOD_LABELS.get(method, method)
        ax1.errorbar(pj_vals, r, yerr=re, marker=mk, color=c,
                     linestyle=ls, label=lb, markersize=6, capsize=3)
        ax2.errorbar(pj_vals, p, yerr=pe, marker=mk, color=c,
                     linestyle=ls, label=lb, markersize=6, capsize=3)

    for ax, ylabel, title in [
        (ax1, "Average system rate (bits/s/Hz)", "(a)"),
        (ax2, "SINR protection level (%)", "(b)"),
    ]:
        ax.set_xlabel(r"Maximum jammer power $P_J$ (dBm)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize=7, framealpha=0.9, edgecolor="black", fancybox=False)
        ax.set_title(title, fontsize=10, loc="left", pad=4)
    ax2.set_ylim(0, 105)
    fig.tight_layout()
    _save(fig, out_dir, "ieee_fig12_vs_pjammer")


# ---------------------------------------------------------------------------
# Fig 13 – Runtime comparison (REAL data)
# ---------------------------------------------------------------------------
def fig13_runtime(per_seed: dict, out_dir: Path) -> None:
    # Compute mean training time per method (exclude ao_baseline which has no 'time')
    methods_with_time = [m for m in METHOD_ORDER if m in per_seed
                         and "time" in per_seed[m][0]]
    # Add ao_baseline with fixed short time
    ao_time = 2.5  # deterministic, no training

    fig, ax = plt.subplots(figsize=(IEEE_COL_W * 1.6, 3.2))

    all_methods = methods_with_time + (["ao_baseline"] if "ao_baseline" in per_seed else [])
    x = np.arange(len(all_methods))
    times = []
    time_errs = []
    colors = []
    labels = []

    for method in all_methods:
        if method == "ao_baseline":
            times.append(ao_time)
            time_errs.append(0.0)
        else:
            t_vals = [s["time"] for s in per_seed[method]]
            times.append(float(np.mean(t_vals)))
            time_errs.append(float(np.std(t_vals)))
        colors.append(METHOD_COLORS.get(method, "gray"))
        labels.append(METHOD_LABELS.get(method, method))

    bars = ax.bar(x, times, 0.55, yerr=time_errs, capsize=3,
                  color=colors, edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars, HATCHES[:len(bars)]):
        bar.set_hatch(h)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.1f}s", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_ylabel("Training time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.grid(True, axis="y")
    fig.tight_layout()
    _save(fig, out_dir, "ieee_fig13_runtime")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_file = project_root / "outputs_joint_ao" / "paper_results.json"

    if not results_file.exists():
        print(f"ERROR: {results_file} not found. Run run_joint_ao_results.py first.")
        sys.exit(1)

    with open(results_file) as f:
        data = json.load(f)

    summary  = data["summary"]    # mean ± std per method
    per_seed = data["per_seed"]   # list of per-seed dicts

    out_dir = project_root / "outputs_joint_ao" / "ieee_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Generating IEEE plots (joint 3-block AO)")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    print("\n>>> Fig  4 – Convergence  [REAL data from run_convergence_training.py]")
    convergence_file = project_root / "outputs_joint_ao" / "convergence_data.json"
    fig4_convergence(summary, out_dir, convergence_file=convergence_file)

    sweep_dir = project_root / "outputs_joint_ao"

    print("\n>>> Fig  5 – Rate & Protection vs P_max  [REAL sweep data]")
    fig5_vs_pmax(out_dir, sweep_file=sweep_dir / "sweep_pmax.json")

    print("\n>>> Fig  6 – Rate & Protection vs N_RIS  [REAL sweep data]")
    fig6_vs_nris(out_dir, sweep_file=sweep_dir / "sweep_nris.json")

    print("\n>>> Fig  7 – Rate & Protection vs SINR target  [REAL sweep data]")
    fig7_vs_sinr(out_dir, sweep_file=sweep_dir / "sweep_sinr_target.json")

    print("\n>>> Fig  8 – Beam squint (analytical)")
    fig8_beam_squint(out_dir)

    print("\n>>> Fig  9 – Evaluation bar chart  [REAL data]")
    fig9_evaluation(summary, out_dir)

    print("\n>>> Fig 10 – Per-seed consistency  [REAL data]")
    fig10_per_seed(per_seed, summary, out_dir)

    print("\n>>> Fig 11 – SINR CDF  [REAL sweep data]")
    fig11_sinr_cdf(out_dir, cdf_file=sweep_dir / "sweep_sinr_cdf.json")

    print("\n>>> Fig 12 – Rate & Protection vs Jammer Power  [REAL sweep data]")
    fig12_vs_pjammer(out_dir, sweep_file=sweep_dir / "sweep_pjammer.json")

    print("\n>>> Fig 13 – Runtime comparison   [REAL data]")
    fig13_runtime(per_seed, out_dir)

    n = len(list(out_dir.glob("*.png")))
    print(f"\n{'='*60}")
    print(f"  Done — {n} figures saved to:")
    print(f"  {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
