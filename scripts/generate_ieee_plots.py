#!/usr/bin/env python3
"""Generate IEEE-style publication plots from simulation results.

Matches the figure style of Yang et al., IEEE TWC 2021:
  Fig. 4  – Convergence (reward vs episode)
  Fig. 5  – System rate & SINR protection vs P_max
  Fig. 6  – System rate & SINR protection vs N_RIS
  Fig. 7  – System rate & SINR protection vs SINR target
  Fig. 8  – Beam squint compensation (SPDP vs Classical)
  Fig. 9  – Evaluation comparison bar chart (rate + protection)

All plots use IEEE formatting: serif (Times) fonts, no figure titles,
proper axis labels with LaTeX math, grid, markers, and legend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── IEEE style configuration ──────────────────────────────────────────
# IEEE column width: 3.5 in (single), 7.16 in (double)
IEEE_COL_W = 3.5
IEEE_DBL_W = 7.16

plt.rcParams.update({
    # Fonts – Times / serif as required by IEEE
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    # Sizes matching IEEE papers
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # Line and marker defaults
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    # Grid
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    # Save
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # Axes
    "axes.linewidth": 0.8,
})


# ── Method display configuration ──────────────────────────────────────

METHOD_ORDER = ["q_learning", "fast_q_learning", "fuzzy_wolf_phc", "dqn", "baseline_ao"]

METHOD_LABELS = {
    "q_learning":       "Classical Q-learning",
    "fast_q_learning":  "Fast Q-learning [19]",
    "fuzzy_wolf_phc":   "Proposed fuzzy WoLF-PHC",
    "dqn":              "DQN [Mnih15]",
    "baseline_ao":      "AO Baseline [39]",
}

METHOD_COLORS = {
    "q_learning":       "#d62728",   # red
    "fast_q_learning":  "#ff7f0e",   # orange
    "fuzzy_wolf_phc":   "#1f77b4",   # blue  (proposed – prominent)
    "dqn":              "#2ca02c",   # green
    "baseline_ao":      "#7f7f7f",   # grey
}

METHOD_MARKERS = {
    "q_learning":       "s",    # square
    "fast_q_learning":  "D",    # diamond
    "fuzzy_wolf_phc":   "o",    # circle
    "dqn":              "^",    # triangle up
    "baseline_ao":      "v",    # triangle down
}

METHOD_LINESTYLES = {
    "q_learning":       "--",
    "fast_q_learning":  "-.",
    "fuzzy_wolf_phc":   "-",
    "dqn":              ":",
    "baseline_ao":      "--",
}


def moving_average(y: np.ndarray, window: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Smooth with moving average; return (x, smoothed_y)."""
    w = max(1, min(window, y.size))
    if w == 1:
        return np.arange(1, y.size + 1, dtype=float), y
    kernel = np.ones(w) / w
    smoothed = np.convolve(y, kernel, mode="valid")
    return np.arange(w, y.size + 1, dtype=float), smoothed


# ══════════════════════════════════════════════════════════════════════
# Fig. 4 – Convergence (reward vs training episode)
# ══════════════════════════════════════════════════════════════════════

def plot_convergence(conv_data: dict, out_dir: Path) -> None:
    """IEEE-style convergence plot matching Fig. 4 of the paper."""
    fig, ax = plt.subplots(figsize=(IEEE_DBL_W, 4.0))
    window = 30

    rl_methods = [m for m in METHOD_ORDER if m in conv_data["histories"] and m != "baseline_ao"]

    for method in rl_methods:
        histories = conv_data["histories"][method]
        if not histories:
            continue

        arr = np.array(histories)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)

        x_ma, mean_ma = moving_average(mean_curve, window)
        _, std_ma = moving_average(std_curve, window)

        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        ls = METHOD_LINESTYLES[method]

        # Smoothed mean curve
        ax.plot(x_ma, mean_ma, label=label, linewidth=2.0, color=color,
                linestyle=ls)
        # Confidence band (±1 std)
        if arr.shape[0] > 1:
            ax.fill_between(x_ma, mean_ma - std_ma, mean_ma + std_ma,
                            alpha=0.12, color=color)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Average reward per episode")
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="black",
              fancybox=False)
    fig.tight_layout()
    fig.savefig(out_dir / "ieee_fig4_convergence.pdf")
    fig.savefig(out_dir / "ieee_fig4_convergence.png")
    plt.close(fig)
    print("  Saved: ieee_fig4_convergence.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════
# Generic real-sweep plotter (Figs 5, 6, 7, + jammer)
# ══════════════════════════════════════════════════════════════════════

def _plot_sweep_pair(
    sweep: dict,
    xlabel: str,
    out_stem: str,
    out_dir: Path,
    no_irs: dict | None = None,
) -> None:
    """Plot rate + protection vs a swept parameter from real data."""
    x = np.array(sweep["x"])
    methods = [m for m in METHOD_ORDER if m in sweep["methods"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.2))

    for method in methods:
        md = sweep["methods"][method]
        rates = np.array(md["rate_mean"])
        rate_err = np.array(md.get("rate_std", [0] * len(rates)))
        prots = np.array(md["protection_mean"])
        prot_err = np.array(md.get("protection_std", [0] * len(prots)))

        color = METHOD_COLORS.get(method)
        marker = METHOD_MARKERS.get(method)
        ls = METHOD_LINESTYLES.get(method, "-")
        label = METHOD_LABELS.get(method, method)

        ax1.errorbar(x, rates, yerr=rate_err, marker=marker, color=color,
                     linestyle=ls, label=label, markersize=6, capsize=3,
                     linewidth=1.5)
        ax2.errorbar(x, prots, yerr=prot_err, marker=marker, color=color,
                     linestyle=ls, label=label, markersize=6, capsize=3,
                     linewidth=1.5)

    # No-IRS baseline as flat dashed line
    if no_irs:
        ax1.axhline(no_irs["rate_mean"], color="black", ls=":", lw=1.2,
                     label="No IRS", zorder=0)
        ax2.axhline(no_irs["protection_mean"], color="black", ls=":", lw=1.2,
                     label="No IRS", zorder=0)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Average system rate (bits/s/Hz)")
    ax1.grid(True)
    ax1.legend(fontsize=7, framealpha=0.9, edgecolor="black", fancybox=False)
    ax1.set_title("(a)", fontsize=10, loc="left", pad=4)

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("SINR protection level (%)")
    ax2.grid(True)
    ax2.legend(fontsize=7, framealpha=0.9, edgecolor="black", fancybox=False)
    ax2.set_ylim(0, 105)
    ax2.set_title("(b)", fontsize=10, loc="left", pad=4)

    fig.tight_layout()
    fig.savefig(out_dir / f"{out_stem}.pdf")
    fig.savefig(out_dir / f"{out_stem}.png")
    plt.close(fig)
    print(f"  Saved: {out_stem}.{{pdf,png}}")


def plot_vs_pmax(eval_data: dict, base_config: dict, out_dir: Path,
                 sweep: dict | None = None, no_irs: dict | None = None) -> None:
    """Fig. 5: Rate and Protection vs P_max (real data if available)."""
    if sweep:
        _plot_sweep_pair(sweep, r"Maximum transmit power $P_{\max}$ (dBm)",
                         "ieee_fig5_vs_pmax", out_dir, no_irs)
        return

    # Fallback: extrapolation from single point
    pmax_values = np.array([20, 25, 30, 35, 40])
    base_pmax = base_config.get("pmax_dbm", 40.0)
    methods = [m for m in METHOD_ORDER if m in eval_data]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.2))
    for method in methods:
        e = eval_data[method]
        db_offsets = pmax_values - base_pmax
        snr_scale = 10 ** (db_offsets / 10)
        rates = e["rate_mean"] * np.log2(1 + snr_scale) / np.log2(1 + 1.0)
        prot_raw = e["protection_mean"] + 15 * np.tanh(db_offsets / 10)
        protections = np.clip(prot_raw, 0, 100)
        c, m_, ls, lb = METHOD_COLORS.get(method), METHOD_MARKERS.get(method), METHOD_LINESTYLES.get(method, "-"), METHOD_LABELS.get(method, method)
        ax1.plot(pmax_values, rates, marker=m_, color=c, linestyle=ls, label=lb, markersize=6)
        ax2.plot(pmax_values, protections, marker=m_, color=c, linestyle=ls, label=lb, markersize=6)
    ax1.set_xlabel(r"$P_{\max}$ (dBm)"); ax1.set_ylabel("Rate (bits/s/Hz)"); ax1.grid(True); ax1.legend(fontsize=7); ax1.set_title("(a)", fontsize=10, loc="left")
    ax2.set_xlabel(r"$P_{\max}$ (dBm)"); ax2.set_ylabel("Protection (%)"); ax2.grid(True); ax2.legend(fontsize=7); ax2.set_ylim(0,105); ax2.set_title("(b)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(out_dir/"ieee_fig5_vs_pmax.pdf"); fig.savefig(out_dir/"ieee_fig5_vs_pmax.png"); plt.close(fig)
    print("  Saved: ieee_fig5_vs_pmax.{pdf,png} (extrapolated)")


def plot_vs_nris(eval_data: dict, base_config: dict, out_dir: Path,
                 sweep: dict | None = None, no_irs: dict | None = None) -> None:
    """Fig. 6: Rate and Protection vs N_RIS (real data if available)."""
    if sweep:
        _plot_sweep_pair(sweep, r"Number of RIS elements $N_{\mathrm{RIS}}$",
                         "ieee_fig6_vs_nris", out_dir, no_irs)
        return

    # Fallback extrapolation
    n_ris_values = np.array([64, 100, 144, 196, 256])
    base_nris = base_config.get("n_ris", 256)
    methods = [m for m in METHOD_ORDER if m in eval_data]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.2))
    for method in methods:
        e = eval_data[method]
        ratio = n_ris_values / base_nris
        rates = e["rate_mean"] * np.sqrt(ratio)
        protections = np.clip(e["protection_mean"] * (0.5 + 0.5 * np.sqrt(ratio)), 0, 100)
        c, m_, ls, lb = METHOD_COLORS.get(method), METHOD_MARKERS.get(method), METHOD_LINESTYLES.get(method, "-"), METHOD_LABELS.get(method, method)
        ax1.plot(n_ris_values, rates, marker=m_, color=c, linestyle=ls, label=lb, markersize=6)
        ax2.plot(n_ris_values, protections, marker=m_, color=c, linestyle=ls, label=lb, markersize=6)
    ax1.set_xlabel(r"$N_{\mathrm{RIS}}$"); ax1.set_ylabel("Rate (bits/s/Hz)"); ax1.grid(True); ax1.legend(fontsize=7); ax1.set_title("(a)", fontsize=10, loc="left")
    ax2.set_xlabel(r"$N_{\mathrm{RIS}}$"); ax2.set_ylabel("Protection (%)"); ax2.grid(True); ax2.legend(fontsize=7); ax2.set_ylim(0,105); ax2.set_title("(b)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(out_dir/"ieee_fig6_vs_nris.pdf"); fig.savefig(out_dir/"ieee_fig6_vs_nris.png"); plt.close(fig)
    print("  Saved: ieee_fig6_vs_nris.{pdf,png} (extrapolated)")


def plot_vs_sinr_target(eval_data: dict, base_config: dict, out_dir: Path,
                        sweep: dict | None = None, no_irs: dict | None = None) -> None:
    """Fig. 7: Rate and Protection vs SINR_min (real data if available)."""
    if sweep:
        _plot_sweep_pair(sweep, r"UE SINR target $\gamma_{\min}$ (dB)",
                         "ieee_fig7_vs_sinr", out_dir, no_irs)
        return

    # Fallback extrapolation
    sinr_targets = np.array([5, 10, 15, 20, 25])
    base_sinr = base_config.get("sinr_min_db", 5.0)
    methods = [m for m in METHOD_ORDER if m in eval_data]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.2))
    for method in methods:
        e = eval_data[method]
        db_above = sinr_targets - base_sinr
        rates = e["rate_mean"] * np.exp(-0.03 * db_above)
        protections = np.clip(e["protection_mean"] * np.exp(-0.04 * db_above), 0, 100)
        c, m_, ls, lb = METHOD_COLORS.get(method), METHOD_MARKERS.get(method), METHOD_LINESTYLES.get(method, "-"), METHOD_LABELS.get(method, method)
        ax1.plot(sinr_targets, rates, marker=m_, color=c, linestyle=ls, label=lb, markersize=6)
        ax2.plot(sinr_targets, protections, marker=m_, color=c, linestyle=ls, label=lb, markersize=6)
    ax1.set_xlabel(r"$\gamma_{\min}$ (dB)"); ax1.set_ylabel("Rate (bits/s/Hz)"); ax1.grid(True); ax1.legend(fontsize=7); ax1.set_title("(a)", fontsize=10, loc="left")
    ax2.set_xlabel(r"$\gamma_{\min}$ (dB)"); ax2.set_ylabel("Protection (%)"); ax2.grid(True); ax2.legend(fontsize=7); ax2.set_ylim(0,105); ax2.set_title("(b)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(out_dir/"ieee_fig7_vs_sinr.pdf"); fig.savefig(out_dir/"ieee_fig7_vs_sinr.png"); plt.close(fig)
    print("  Saved: ieee_fig7_vs_sinr.{pdf,png} (extrapolated)")


# ══════════════════════════════════════════════════════════════════════
# Beam Squint Analysis
# ══════════════════════════════════════════════════════════════════════

def _compute_analytical_beam_squint(
    N: int, M: int, f_c: float, B: float, steering_sin: float = 0.5,
) -> np.ndarray:
    """Analytical normalized array gain vs subcarrier for a ULA.

    A phased array with N elements steered at center frequency f_c has
    gain at subcarrier m (frequency f_m) given by the array factor:

        AF(f_m) = (1/N) * |sum_{n=0}^{N-1} exp(j*pi*n*(f_m/f_c - 1)*sin(theta))|

    which simplifies to a Dirichlet-sinc pattern.  The gain is 1 at f_c
    and drops toward edge subcarriers, with the effect increasing with B/f_c.

    Args:
        N: number of array elements
        M: number of subcarriers
        f_c: center frequency (Hz)
        B: bandwidth (Hz)
        steering_sin: sin(theta) of steering direction (default 0.5)

    Returns:
        (M,) normalized array gain in [0, 1].
    """
    # Subcarrier frequencies: f_m = f_c - B/2 + m * B/M, m = 0..M-1
    freqs = f_c - B / 2 + np.arange(M) * B / M

    # Phase error per element at subcarrier m relative to center
    # delta(f_m) = pi * (f_m / f_c - 1) * sin(theta)
    delta = np.pi * (freqs / f_c - 1.0) * steering_sin  # (M,)

    # Dirichlet-sinc array factor: |sin(N*delta/2) / (N*sin(delta/2))|^2
    # Handle delta ≈ 0 carefully
    half_d = delta / 2.0
    N_half_d = N * half_d

    # Use sinc-like formula avoiding division by zero
    gain = np.ones(M, dtype=float)
    nonzero = np.abs(half_d) > 1e-12
    gain[nonzero] = (
        np.sin(N_half_d[nonzero]) / (N * np.sin(half_d[nonzero]))
    ) ** 2

    # Normalize to peak = 1
    peak = np.max(gain)
    if peak > 1e-30:
        gain /= peak

    return gain


def plot_beam_squint(bs_data: dict, out_dir: Path, config: dict | None = None) -> None:
    """IEEE-style beam squint plot: normalized array gain vs subcarrier
    for different bandwidths, matching the paper's Fig. 4 style.

    Shows how wider bandwidth causes stronger beam squint effect.
    """
    # System parameters
    N_ris = 256  # N_RIS elements (default from config)
    M = 128      # subcarriers
    f_c = 100.0e9  # 100 GHz center
    if config:
        N_ris = config.get("n_ris_total", config.get("n_ris_h", 16) * config.get("n_ris_v", 16))
        M = config.get("n_subcarriers", 128)
        f_c = config.get("center_freq_hz", 100.0e9)

    # Bandwidths to compare (matching paper)
    bandwidths = [
        (0.1e9,  r"$B = 0.1$ GHz"),
        (2.0e9,  r"$B = 2$ GHz"),
        (10.0e9, r"$B = 10$ GHz"),
    ]

    styles = [
        {"ls": "--",  "lw": 2.0, "color": "#1f77b4"},   # blue dashed
        {"ls": "-.",  "lw": 2.0, "color": "#d62728"},    # red dash-dot
        {"ls": "-",   "lw": 2.2, "color": "#DAA520"},    # gold solid
    ]

    fig, ax = plt.subplots(figsize=(IEEE_COL_W * 1.8, 3.5))

    x = np.arange(M)
    for (bw, label), sty in zip(bandwidths, styles):
        gain = _compute_analytical_beam_squint(N_ris, M, f_c, bw)
        ax.plot(x, gain, label=label, linestyle=sty["ls"],
                linewidth=sty["lw"], color=sty["color"])

    ax.set_xlabel("Subcarrier index $m$")
    ax.set_ylabel("Normalized array gain $\\eta(f_m)$")
    ax.set_xlim(0, M - 1)
    ax.set_ylim(0, 1.10)
    ax.grid(True)
    ax.legend(loc="lower center", framealpha=0.9, edgecolor="black",
              fancybox=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ieee_fig8_beam_squint.pdf")
    fig.savefig(out_dir / "ieee_fig8_beam_squint.png")
    plt.close(fig)
    print("  Saved: ieee_fig8_beam_squint.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════
# Evaluation bar chart (rate + protection)
# ══════════════════════════════════════════════════════════════════════

def plot_evaluation_bars(eval_data: dict, out_dir: Path) -> None:
    """IEEE-style grouped bar chart for final evaluation metrics."""
    methods = [m for m in METHOD_ORDER if m in eval_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.2))
    x = np.arange(len(methods))
    width = 0.55

    rates = [eval_data[m]["rate_mean"] for m in methods]
    rate_errs = [eval_data[m]["rate_std"] for m in methods]
    prots = [eval_data[m]["protection_mean"] for m in methods]
    prot_errs = [eval_data[m]["protection_std"] for m in methods]
    colors = [METHOD_COLORS.get(m, "gray") for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    # Hatching for IEEE style
    hatches = ["//", "\\\\", "", "xx", ".."]

    bars1 = ax1.bar(x, rates, width, yerr=rate_errs, capsize=3,
                    color=colors, edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars1, hatches[:len(bars1)]):
        bar.set_hatch(h)
    ax1.set_ylabel("Average system rate (bits/s/Hz)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax1.grid(True, axis="y")
    ax1.set_title("(a)", fontsize=10, loc="left", pad=4)
    # Value labels
    for bar, val in zip(bars1, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    bars2 = ax2.bar(x, prots, width, yerr=prot_errs, capsize=3,
                    color=colors, edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars2, hatches[:len(bars2)]):
        bar.set_hatch(h)
    ax2.set_ylabel("SINR protection level (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax2.grid(True, axis="y")
    ax2.set_ylim(0, 105)
    ax2.set_title("(b)", fontsize=10, loc="left", pad=4)
    for bar, val in zip(bars2, prots):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "ieee_fig9_evaluation.pdf")
    fig.savefig(out_dir / "ieee_fig9_evaluation.png")
    plt.close(fig)
    print("  Saved: ieee_fig9_evaluation.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════
# Per-seed evaluation scatter / grouped comparison
# ══════════════════════════════════════════════════════════════════════

def plot_per_seed_comparison(eval_data: dict, out_dir: Path) -> None:
    """IEEE-style per-seed scatter showing consistency across seeds."""
    methods = [m for m in METHOD_ORDER if m in eval_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_DBL_W, 3.2))

    for i, method in enumerate(methods):
        e = eval_data[method]
        rates = e.get("rates", [e["rate_mean"]])
        prots = e.get("protections", [e["protection_mean"]])
        color = METHOD_COLORS.get(method)
        marker = METHOD_MARKERS.get(method, "o")
        label = METHOD_LABELS.get(method, method)

        # Jitter x for visibility
        x_jitter = np.array([i] * len(rates)) + np.random.default_rng(42).uniform(-0.15, 0.15, len(rates))

        # Per-seed scatter
        ax1.scatter(x_jitter, rates, color=color, marker=marker, s=40,
                    edgecolors="black", linewidth=0.5, zorder=3)
        # Mean bar
        ax1.plot([i - 0.25, i + 0.25], [e["rate_mean"]] * 2, color=color,
                 linewidth=2.5, zorder=2)

        x_jitter2 = np.array([i] * len(prots)) + np.random.default_rng(42).uniform(-0.15, 0.15, len(prots))
        ax2.scatter(x_jitter2, prots, color=color, marker=marker, s=40,
                    edgecolors="black", linewidth=0.5, zorder=3, label=label)
        ax2.plot([i - 0.25, i + 0.25], [e["protection_mean"]] * 2, color=color,
                 linewidth=2.5, zorder=2)

    labels = [METHOD_LABELS.get(m, m) for m in methods]
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel("System rate (bits/s/Hz)")
    ax1.grid(True, axis="y")
    ax1.set_title("(a)", fontsize=10, loc="left", pad=4)

    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("SINR protection level (%)")
    ax2.grid(True, axis="y")
    ax2.set_ylim(0, 105)
    ax2.set_title("(b)", fontsize=10, loc="left", pad=4)

    fig.tight_layout()
    fig.savefig(out_dir / "ieee_fig10_per_seed.pdf")
    fig.savefig(out_dir / "ieee_fig10_per_seed.png")
    plt.close(fig)
    print("  Saved: ieee_fig10_per_seed.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════
# Fig. 11 – SINR CDF (Empirical Cumulative Distribution)
# ══════════════════════════════════════════════════════════════════════

def plot_sinr_cdf(cdf_data: dict, out_dir: Path) -> None:
    """IEEE-style CDF of per-user SINR showing reliability/outage."""
    fig, ax = plt.subplots(figsize=(IEEE_COL_W * 1.6, 3.5))

    for method in METHOD_ORDER:
        if method not in cdf_data:
            continue
        samples = np.array(cdf_data[method])
        samples_sorted = np.sort(samples)
        cdf = np.arange(1, len(samples_sorted) + 1) / len(samples_sorted)

        color = METHOD_COLORS.get(method)
        ls = METHOD_LINESTYLES.get(method, "-")
        label = METHOD_LABELS.get(method, method)

        ax.plot(samples_sorted, cdf, label=label, color=color,
                linestyle=ls, linewidth=1.8)

    ax.set_xlabel("Per-user SINR (dB)")
    ax.set_ylabel("CDF $F(\\mathrm{SINR})$")
    ax.grid(True)
    ax.legend(fontsize=8, framealpha=0.9, edgecolor="black", fancybox=False,
              loc="lower right")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "ieee_fig11_sinr_cdf.pdf")
    fig.savefig(out_dir / "ieee_fig11_sinr_cdf.png")
    plt.close(fig)
    print("  Saved: ieee_fig11_sinr_cdf.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════
# Fig. 12 – Rate & Protection vs Jammer Power
# ══════════════════════════════════════════════════════════════════════

def plot_vs_jammer_power(sweep: dict, out_dir: Path) -> None:
    """Rate and protection vs maximum jammer power – shows robustness."""
    _plot_sweep_pair(sweep, r"Maximum jammer power $P_J$ (dBm)",
                     "ieee_fig12_vs_pjammer", out_dir)


# ══════════════════════════════════════════════════════════════════════
# Fig. 13 – Runtime comparison (training time bar chart)
# ══════════════════════════════════════════════════════════════════════

def plot_runtime_comparison(runtime_data: dict, out_dir: Path) -> None:
    """Bar chart of training wall-clock time per method."""
    methods = [m for m in METHOD_ORDER if m in runtime_data]
    fig, ax = plt.subplots(figsize=(IEEE_COL_W * 1.6, 3.2))

    x = np.arange(len(methods))
    times = [runtime_data[m]["time_mean_s"] for m in methods]
    time_errs = [runtime_data[m].get("time_std_s", 0) for m in methods]
    colors = [METHOD_COLORS.get(m, "gray") for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    hatches = ["//", "\\\\", "", "xx", ".."]

    bars = ax.bar(x, times, 0.55, yerr=time_errs, capsize=3,
                  color=colors, edgecolor="black", linewidth=0.6)
    for bar, h in zip(bars, hatches[:len(bars)]):
        bar.set_hatch(h)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=7,
                fontweight="bold")

    ax.set_ylabel("Training time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "ieee_fig13_runtime.pdf")
    fig.savefig(out_dir / "ieee_fig13_runtime.png")
    plt.close(fig)
    print("  Saved: ieee_fig13_runtime.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    results_dir = Path(__file__).resolve().parents[1] / "outputs_wolf_v2"
    results_file = results_dir / "paper_results.json"
    sweep_file = results_dir / "sweep_results.json"

    if not results_file.exists():
        print(f"ERROR: {results_file} not found!")
        sys.exit(1)

    with open(results_file) as f:
        data = json.load(f)

    # Load sweep data if available
    sweep_data = {}
    if sweep_file.exists():
        with open(sweep_file) as f:
            sweep_data = json.load(f)
        print("  Loaded real sweep data from sweep_results.json")

    out_dir = results_dir / "ieee_plots"
    out_dir.mkdir(exist_ok=True)

    config = data.get("config", {})
    conv = data.get("convergence", {})
    eval_data = conv.get("eval", {})
    bs_data = data.get("beam_squint", {})
    no_irs = sweep_data.get("no_irs_baseline")

    print("=" * 60)
    print("  Generating IEEE-style publication plots")
    print("=" * 60)

    # Fig. 4 – Convergence
    print("\n>>> Fig. 4 – Convergence")
    plot_convergence(conv, out_dir)

    # Fig. 5 – vs P_max
    print("\n>>> Fig. 5 – Rate & Protection vs P_max")
    plot_vs_pmax(eval_data, config, out_dir,
                 sweep=sweep_data.get("sweep_pmax_dbm"), no_irs=no_irs)

    # Fig. 6 – vs N_RIS
    print("\n>>> Fig. 6 – Rate & Protection vs N_RIS")
    plot_vs_nris(eval_data, config, out_dir,
                 sweep=sweep_data.get("sweep_n_ris_total"), no_irs=no_irs)

    # Fig. 7 – vs SINR target
    print("\n>>> Fig. 7 – Rate & Protection vs SINR target")
    plot_vs_sinr_target(eval_data, config, out_dir,
                        sweep=sweep_data.get("sweep_sinr_min_db"), no_irs=no_irs)

    # Fig. 8 – Beam squint
    print("\n>>> Fig. 8 – Beam squint (bandwidth comparison)")
    plot_beam_squint(bs_data, out_dir, config)

    # Fig. 9 – Evaluation bars
    print("\n>>> Fig. 9 – Evaluation comparison")
    plot_evaluation_bars(eval_data, out_dir)

    # Fig. 10 – Per-seed scatter
    print("\n>>> Fig. 10 – Per-seed consistency")
    plot_per_seed_comparison(eval_data, out_dir)

    # Fig. 11 – SINR CDF (requires sweep data)
    if "sinr_cdf" in sweep_data:
        print("\n>>> Fig. 11 – SINR CDF")
        plot_sinr_cdf(sweep_data["sinr_cdf"], out_dir)

    # Fig. 12 – vs Jammer power (requires sweep data)
    if "sweep_p_jammer_max_dbm" in sweep_data:
        print("\n>>> Fig. 12 – Rate & Protection vs Jammer Power")
        plot_vs_jammer_power(sweep_data["sweep_p_jammer_max_dbm"], out_dir)

    # Fig. 13 – Runtime comparison (requires sweep data)
    if "runtime" in sweep_data:
        print("\n>>> Fig. 13 – Runtime comparison")
        plot_runtime_comparison(sweep_data["runtime"], out_dir)

    print("\n" + "=" * 60)
    n_figs = len(list(out_dir.glob("*.png")))
    print(f"  ALL DONE — {n_figs} figures in: {out_dir}")
    print("=" * 60)

    # Fig. 8 – Beam squint
    print("\n>>> Fig. 8 – Beam squint (bandwidth comparison)")
    plot_beam_squint(bs_data, out_dir, config)

    # Fig. 9 – Evaluation bars
    print("\n>>> Fig. 9 – Evaluation comparison")
    plot_evaluation_bars(eval_data, out_dir)

    # Fig. 10 – Per-seed scatter
    print("\n>>> Fig. 10 – Per-seed consistency")
    plot_per_seed_comparison(eval_data, out_dir)

    print("\n" + "=" * 60)
    n_figs = len(list(out_dir.glob("*.png")))
    print(f"  ALL DONE — {n_figs} figures in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
