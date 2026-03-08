from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from irs_anti_jamming.config import RLConfig, SweepConfig, SystemConfig, TrainEvalConfig
from irs_anti_jamming.experiments import run_convergence_experiment, run_parameter_sweep


METHOD_LABELS = {
    "fuzzy_wolf_phc": "Fuzzy WoLF-PHC (Proposed)",
    "fast_q_learning": "Fast Q-Learning",
    "baseline_ao": "Baseline 1 (AO-like)",
    "no_irs_power": "Optimal PA w/o IRS",
}


def _moving_average(y: np.ndarray, window: int = 25) -> tuple[np.ndarray, np.ndarray]:
    if y.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    w = max(1, min(window, y.size))
    if w == 1:
        x = np.arange(1, y.size + 1, dtype=float)
        return x, y
    kernel = np.ones(w, dtype=float) / w
    smoothed = np.convolve(y, kernel, mode="valid")
    x = np.arange(w, y.size + 1, dtype=float)
    return x, smoothed


def _plot_convergence(data: dict[str, np.ndarray], out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))

    for key, label in [
        ("q_learning", "Classical Q-Learning"),
        ("fast_q_learning", "Fast Q-Learning"),
        ("fuzzy_wolf_phc", "Fuzzy WoLF-PHC (Proposed)"),
    ]:
        y = np.asarray(data[key], dtype=float)
        x_raw = np.arange(1, y.size + 1, dtype=float)
        x_ma, y_ma = _moving_average(y, window=25)

        plt.plot(x_raw, y, linewidth=1.0, alpha=0.20)
        plt.plot(x_ma, y_ma, label=label, linewidth=2.2)

    plt.xlabel("Episode")
    plt.ylabel("Average Reward (Moving Average)")
    plt.title("Convergence Behavior (Paper-Style Fig. 4)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_convergence.png", dpi=220)
    plt.close()


def _plot_dual_metric(
    x: list[float],
    methods: dict[str, dict[str, list[float]]],
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, stats in methods.items():
        axes[0].plot(x, stats["rate"], marker="o", linewidth=2, label=METHOD_LABELS[method])
        axes[1].plot(x, stats["protection"], marker="o", linewidth=2, label=METHOD_LABELS[method])

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Average System Rate (bit/s/Hz)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("(a) System Rate")

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("SINR Protection Level (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("(b) SINR Protection")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="IRS anti-jamming simulation (paper-style trends)")
    parser.add_argument(
        "--profile",
        type=str,
        choices=["quick", "balanced", "full"],
        default="balanced",
        help="Runtime profile: quick (fast), balanced (default), full (slowest)",
    )
    parser.add_argument("--quick", action="store_true", help="Deprecated alias for --profile quick")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    sys_cfg = SystemConfig()
    rl_cfg = RLConfig()
    base_run_cfg = TrainEvalConfig()
    sweep_cfg = SweepConfig()

    profile = "quick" if args.quick else args.profile

    if profile == "quick":
        run_cfg_conv = replace(
            base_run_cfg,
            train_episodes=220,
            train_steps_per_episode=16,
            eval_episodes=12,
            eval_steps_per_episode=6,
            n_seeds=1,
        )
        run_cfg_sweep = replace(
            base_run_cfg,
            train_episodes=80,
            train_steps_per_episode=10,
            eval_episodes=5,
            eval_steps_per_episode=4,
            n_seeds=1,
        )
    elif profile == "balanced":
        run_cfg_conv = replace(
            base_run_cfg,
            train_episodes=600,
            train_steps_per_episode=25,
            eval_episodes=16,
            eval_steps_per_episode=10,
            n_seeds=2,
        )
        run_cfg_sweep = replace(
            base_run_cfg,
            train_episodes=400,
            train_steps_per_episode=25,
            eval_episodes=10,
            eval_steps_per_episode=25,
            n_seeds=2,
        )
    else:
        run_cfg_conv = base_run_cfg
        run_cfg_sweep = replace(
            base_run_cfg,
            train_episodes=360,
            train_steps_per_episode=18,
            eval_episodes=12,
            eval_steps_per_episode=6,
            n_seeds=2,
        )

    print(
        f"Running profile={profile} with convergence_cfg={run_cfg_conv} and sweep_cfg={run_cfg_sweep}",
        flush=True,
    )

    convergence = run_convergence_experiment(sys_cfg, rl_cfg, run_cfg_conv)
    _plot_convergence(convergence, out_dir)

    sweep_pmax = run_parameter_sweep("pmax_dbm", sweep_cfg.pmax_dbm_values, sys_cfg, rl_cfg, run_cfg_sweep)
    _plot_dual_metric(
        x=sweep_pmax["x"],
        methods=sweep_pmax["methods"],
        xlabel="BS Maximum Transmit Power Pmax (dBm)",
        title="Performance vs Maximum BS Transmit Power (Paper-Style Fig. 5)",
        out_path=out_dir / "fig5_vs_pmax.png",
    )

    sweep_m = run_parameter_sweep("m_ris_elements", sweep_cfg.ris_elements_values, sys_cfg, rl_cfg, run_cfg_sweep)
    _plot_dual_metric(
        x=sweep_m["x"],
        methods=sweep_m["methods"],
        xlabel="Number of IRS Elements M",
        title="Performance vs Number of IRS Elements (Paper-Style Fig. 6)",
        out_path=out_dir / "fig6_vs_m.png",
    )

    sweep_sinr = run_parameter_sweep("sinr_min_db", sweep_cfg.sinr_target_db_values, sys_cfg, rl_cfg, run_cfg_sweep)
    _plot_dual_metric(
        x=sweep_sinr["x"],
        methods=sweep_sinr["methods"],
        xlabel="UE SINR Target (dB)",
        title="Performance vs SINR Target (Paper-Style Fig. 7)",
        out_path=out_dir / "fig7_vs_sinr_target.png",
    )

    result_dump = {
        "system_config": asdict(sys_cfg),
        "rl_config": asdict(rl_cfg),
        "profile": profile,
        "run_config_convergence": asdict(run_cfg_conv),
        "run_config_sweep": asdict(run_cfg_sweep),
        "convergence": {k: v.tolist() for k, v in convergence.items()},
        "sweep_pmax": sweep_pmax,
        "sweep_m": sweep_m,
        "sweep_sinr": sweep_sinr,
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(result_dump, f, indent=2)

    print("Saved outputs:")
    print(f"- {out_dir / 'fig4_convergence.png'}")
    print(f"- {out_dir / 'fig5_vs_pmax.png'}")
    print(f"- {out_dir / 'fig6_vs_m.png'}")
    print(f"- {out_dir / 'fig7_vs_sinr_target.png'}")
    print(f"- {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
