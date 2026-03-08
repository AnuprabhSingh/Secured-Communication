# IRS-Assisted Anti-Jamming Communications Simulation

This project implements a modular Python simulation based on the IEEE paper: “Intelligent Reflecting Surface Assisted Anti-Jamming Communications: A Fast Reinforcement Learning Approach.”

The implementation follows the paper’s system model, optimization objective, and RL formulation.

## Implemented Paper Components

- **BS–RIS–UE model** with channels:
  - BS→IRS: `G ∈ C^(M×N)`
  - BS→UE-k: `g_bu,k^H ∈ C^(1×N)`
  - IRS→UE-k: `g_ru,k^H ∈ C^(1×M)`
  - Jammer→UE-k: `h_J,k^H ∈ C^(1×N_J)`
- **RIS reflecting matrix**:
  - `Φ = diag(e^(jθ_1), …, e^(jθ_M))`, with `|Φ_m| = 1`
- **Signal and interference model**:
  - received signal with desired term + inter-user interference + jamming + noise
- **SINR (Eq. 3) and achievable rate (Eq. 4 objective)**
- **Joint optimization variables**:
  - BS transmit powers `{P_k}`
  - RIS phase shifts `{θ_m}`
- **Reward function (Eq. 7–8)**:
  - `r = Σ log2(1+SINR_k) - λ1 ΣP_k - λ2 ΣSINR_out,k`
- **RL algorithms**:
  - Classical Q-learning
  - Fast Q-learning
  - Fuzzy WoLF-PHC with fuzzy state aggregation (Eq. 14–22)

## Project Structure

- `src/irs_anti_jamming/config.py`: paper/system/RL parameters
- `src/irs_anti_jamming/channel_model.py`: topology + channel generation + pathloss Eq. (24)
- `src/irs_anti_jamming/jammer.py`: smart jammer power/precoder model
- `src/irs_anti_jamming/system_model.py`: SINR/rate/reward calculations
- `src/irs_anti_jamming/state.py`: state construction + fuzzy state aggregation
- `src/irs_anti_jamming/action_space.py`: joint power + RIS-phase action construction
- `src/irs_anti_jamming/agents.py`: Q-learning / fast Q-learning / fuzzy WoLF-PHC
- `src/irs_anti_jamming/baselines.py`: AO-like baseline + no-IRS baseline
- `src/irs_anti_jamming/experiments.py`: training/evaluation/sweeps
- `scripts/run_paper_trends.py`: figure-generation entry point

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
# IRS-Assisted Anti-Jamming Communications (Simulation)

This repository contains a modular Python simulation inspired by the paper:

**Intelligent Reflecting Surface Assisted Anti-Jamming Communications: A Fast Reinforcement Learning Approach**.

It models a BS-IRS-UE system under smart jamming and compares RL-based anti-jamming control against classical baselines.

## What is implemented

- Multi-user downlink with BS, IRS, UEs, and a moving jammer
- Channel generation with pathloss and small-scale fading
- Joint action design for BS power allocation and IRS phase control
- Reward-driven RL for anti-jamming adaptation
- Methods:
  - Classical Q-Learning
  - Fast Q-Learning
  - Fuzzy WoLF-PHC (proposed method)
  - AO-like baseline
  - No-IRS power-only baseline
- Paper-style trend generation:
  - Convergence curve
  - Performance vs max BS power
  - Performance vs number of IRS elements
  - Performance vs SINR target

## Repository layout

```text
.
├── scripts/
│   └── run_paper_trends.py        # Main entry point for figure generation
├── src/irs_anti_jamming/
│   ├── config.py                  # System/RL/train/sweep configs
│   ├── channel_model.py           # Topology + channel realization
│   ├── jammer.py                  # Smart jammer model
│   ├── system_model.py            # SINR/rate/reward evaluation
│   ├── state.py                   # State aggregation + fuzzy memberships
│   ├── action_space.py            # Joint action decoding (power + phase)
│   ├── agents.py                  # RL agent implementations
│   ├── baselines.py               # AO-like and no-IRS baselines
│   ├── environment.py             # Simulation environment
│   ├── experiments.py             # Training/evaluation/sweep orchestration
│   └── utils.py                   # Unit conversion and helper math
├── requirements.txt
└── README.md
```

## Environment setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are minimal (`numpy`, `matplotlib`).

## How to run

### 1) Quick profile (fast sanity check)

```bash
python scripts/run_paper_trends.py --profile quick --output outputs_quick
```

### 2) Balanced profile (recommended default)

```bash
python scripts/run_paper_trends.py --profile balanced --output outputs
```

You can omit `--profile balanced` because balanced is the default.

### 3) Full profile (slowest, highest budget)

```bash
python scripts/run_paper_trends.py --profile full --output outputs_full
```

## Generated outputs

Each run writes the following files to the selected output directory:

- `fig4_convergence.png` — RL convergence trends
- `fig5_vs_pmax.png` — system rate and protection vs BS max power
- `fig6_vs_m.png` — system rate and protection vs IRS elements
- `fig7_vs_sinr_target.png` — system rate and protection vs SINR target
- `results.json` — raw numeric data for all plots

## Simulation pipeline (high level)

1. Sample topology/channel and jammer behavior per slot.
2. Build compact state from prior jammer power, channel quality, and prior SINR.
3. Agent selects a joint action (power profile + IRS phase mode).
4. Evaluate SINR/rate/reward from physical-layer model.
5. Update policy/Q-values, advance environment, and repeat.
6. Aggregate across episodes and seeds, then sweep key parameters.

## Main configs to tune

- `src/irs_anti_jamming/config.py`
  - `SystemConfig`: antenna counts, user count, jammer power bounds, SINR target
  - `RLConfig`: learning rate, discount factor, epsilon schedule, fuzzy/WoLF params
  - `TrainEvalConfig`: training and evaluation episode budgets
  - `SweepConfig`: x-axis values for Figs 5–7

- `scripts/run_paper_trends.py`
  - Runtime profiles (`quick`, `balanced`, `full`)
  - Output folder naming

## Reproducibility notes

- Seeded RNG is used throughout the environment/components.
- Results can still vary slightly across Python/numpy versions and hardware.
- For stable comparisons, keep profile, seed count, and configs fixed between runs.

## Troubleshooting

- **Import errors**: run commands from project root and ensure venv is activated.
- **No output files**: verify the script finished without interruption and check terminal logs.
- **Slow runtime**: use `--profile quick` first, then move to `balanced` or `full`.

## Reference

If you use this simulation in reports or coursework, cite the original IEEE paper above.
