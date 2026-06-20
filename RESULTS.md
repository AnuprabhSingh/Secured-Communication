# Understanding Your Results

This guide explains the results generated after running `python run_all.py`.

---

## Where Are My Results?

All results are saved in: **`outputs_joint_ao/`**

```
outputs_joint_ao/
├── paper_results.json              # Main results (all methods, all metrics)
├── sweep_*.json                    # Parameter sweep data (5 files)
├── training_log.txt                # Training convergence logs
├── sweep_log.txt                   # Sweep execution log
└── ieee_plots/                     # 10 publication-quality figures
    ├── ieee_fig9_evaluation.pdf    # Main results comparison
    ├── ieee_fig5_vs_pmax.pdf       # Transmit power sweep
    ├── ieee_fig6_vs_nris.pdf       # RIS size sweep
    ├── ieee_fig7_vs_sinr.pdf       # QoS threshold sweep
    ├── ieee_fig12_vs_pjammer.pdf   # Jammer power sweep
    └── (+ 5 more figures)
```

---

## Quick Results Lookup

### Step 1: View the Summary
```bash
python3 << 'EOF'
import json
with open("outputs_joint_ao/paper_results.json") as f:
    data = json.load(f)
    summary = data["summary"]
    for method, stats in summary.items():
        print(f"{method}:")
        print(f"  Rate: {stats['rate_mean']:.2f} ± {stats['rate_std']:.2f} bits/s/Hz")
        print(f"  Protection: {stats['protection_mean']:.1f}% ± {stats['protection_std']:.1f}%")
        print()
EOF
```

**Expected output:**
```
q_learning:
  Rate: 8.25 ± 1.28 bits/s/Hz
  Protection: 49.0% ± 14.4%

fast_q_learning:
  Rate: 8.05 ± 1.68 bits/s/Hz
  Protection: 45.5% ± 18.7%

fuzzy_wolf_phc:
  Rate: 11.87 ± 0.55 bits/s/Hz    ⭐ YOUR PROPOSED METHOD
  Protection: 81.6% ± 3.9%

dqn:
  Rate: 8.23 ± 2.40 bits/s/Hz
  Protection: 49.1% ± 25.7%

ao_baseline:
  Rate: 7.89 ± 0.78 bits/s/Hz
  Protection: 43.2% ± 7.7%
```

### Step 2: Compare to Paper

| Metric | Paper Value | Your Run |
|--------|-------------|----------|
| **Fuzzy WoLF-PHC Rate** | 11.87 bits/s/Hz | ✅ Check `paper_results.json` |
| **Protection** | 81.6% | ✅ Check `paper_results.json` |
| **vs AO Baseline** | +50% rate, +89% protection | ✅ Calculate from JSON |
| **Variance** | ~0.55 (lowest) | ✅ Check `rate_std` |

---

## Understanding Key Metrics

### System Rate (bits/s/Hz)
**What it means:** How much data (in bits) you can transmit per second per Hz of bandwidth
**Formula:** `R = Σ log₂(1 + SINR_k)` averaged over all subcarriers
**Good values:** 
- No defense: ~5.5 bits/s/Hz
- Deterministic AO: ~7.9 bits/s/Hz
- **Our WoLF-PHC: 11.87 bits/s/Hz** ✅

**Why it matters:** Higher rate = more data gets through despite jamming

---

### SINR Protection (%)
**What it means:** What fraction of subcarriers meet the quality requirement (5 dB SINR)
**Formula:** `η = (# subcarriers with SINR ≥ γ_min) / (total subcarriers) × 100%`
**Good values:**
- No defense: ~18.8%
- Deterministic AO: ~43.2%
- **Our WoLF-PHC: 81.6%** ✅

**Why it matters:** Higher protection = more robust link, fewer retransmissions needed

---

### Seed Variance
**What it means:** How consistent is the algorithm across different random network topologies?
**Metric:** Standard deviation (±std) across 5 random seeds
**Good values (low std = better consistency):**
- DQN: ±2.40 bits/s/Hz (high variance, unstable)
- AO Baseline: ±0.78 bits/s/Hz (moderate)
- **Our WoLF-PHC: ±0.55 bits/s/Hz** ✅ (most stable)

**Why it matters:** Low variance = algorithm works reliably in different scenarios

---

## Interpreting Individual Plots

### Figure 9: Evaluation Comparison (Main Result)
**File:** `outputs_joint_ao/ieee_plots/ieee_fig9_evaluation.pdf`

Two bars per method:
- **Blue bar** = System rate (bits/s/Hz)
- **Orange bar** = SINR protection (%)

What to look for:
- ✅ WoLF-PHC has tallest blue AND orange bars
- ✅ WoLF-PHC error bars are smallest
- ❌ Other methods have larger error bars = less consistent

---

### Figure 5: Rate vs. Transmit Power
**File:** `outputs_joint_ao/ieee_plots/ieee_fig5_vs_pmax.pdf`

Shows: "How does performance scale with more power?"

What to look for:
- ✅ WoLF-PHC line is steepest (scales best with power)
- ✅ AO baseline plateaus (hits saturation)
- ❌ Q-learning is noisy (high variance)

---

### Figure 6: Rate vs. RIS Array Size
**File:** `outputs_joint_ao/ieee_plots/ieee_fig6_vs_nris.pdf`

Shows: "How much does RIS size matter?"

What to look for:
- ✅ All methods improve with more RIS elements
- ✅ WoLF-PHC remains on top across all sizes
- ✅ Benefit saturates around 256 elements

---

### Figure 7: Rate vs. SINR Target
**File:** `outputs_joint_ao/ieee_plots/ieee_fig7_vs_sinr.pdf`

Shows: "How robust is the system to higher QoS requirements?"

What to look for:
- ✅ WoLF-PHC maintains lead even at high SINR targets (10 dB)
- ✅ All methods collapse at 20 dB (physical limit)

---

### Figure 12: Rate vs. Jammer Power (Regime Transition)
**File:** `outputs_joint_ao/ieee_plots/ieee_fig12_vs_pjammer.pdf`

Shows: "How does each method handle increasingly powerful jammers?"

**Key insight:** Clear regime transition!
- At low jammer power (5-10 dBm): AO baseline competitive
- At high jammer power (15-20 dBm): **WoLF-PHC dominates** ✅

Why? Mixed strategy (randomization) is only valuable against smart jammers
- Weak jammer: Simple deterministic control works fine
- Strong jammer: Unpredictability is your best defense

---

## Paper Results Summary Table

Your reproduction should produce these numbers:

| Method | Rate (bits/s/Hz) | Protection (%) | Status |
|--------|------------------|-----------------|--------|
| No IRS | 5.48 | 18.8% | Baseline (no defense) |
| AO Baseline | 7.89 ± 0.78 | 43.2% ± 7.7% | Deterministic, deterministic |
| Classical Q-Learning | 8.25 ± 1.28 | 49.0% ± 14.4% | Tabular, high variance |
| DQN | 8.23 ± 2.40 | 49.1% ± 25.7% | Deep RL, very unstable |
| Fast Q-Learning | 8.05 ± 1.68 | 45.5% ± 18.7% | Tabular + boost, moderate variance |
| **Fuzzy WoLF-PHC** | **11.87 ± 0.55** | **81.6% ± 3.9%** | **⭐ Our proposal** |
| **Improvement over AO** | **+50%** | **+89%** | **+12× lower variance** |

---

## Parameter Sweep Results

### `sweep_pmax.json`
Performance vs. transmit power: P_max ∈ {25, 32, 40} dBm
- Check: WoLF-PHC should show steepest improvement curve
- Look for: Rate jumps from 1.72 bits/s/Hz (25 dBm) → 11.87 (40 dBm)

### `sweep_nris.json`
Performance vs. RIS array size: N_RIS ∈ {16, 64, 144, 256}
- Check: Diminishing returns (saturation near 256 elements)
- Look for: WoLF-PHC maintains lead at all sizes

### `sweep_sinr_target.json`
Performance vs. QoS threshold: γ_min ∈ {3, 5, 10, 15, 20} dB
- Check: Graceful degradation with stricter targets
- Look for: WoLF-PHC: 91.8% (3dB) → 31.4% (10dB) → 0% (20dB)

### `sweep_pjammer.json`
Performance vs. jammer power: P_J ∈ {5, 10, 15, 20} dBm
- **Key plot:** Regime transition (WoLF-PHC dominates at P_J ≥ 15 dBm)
- Look for: Crossover point ~10-15 dBm where WoLF-PHC becomes superior

### `sweep_sinr_cdf.json`
Empirical SINR distribution across all subcarriers and users
- Check: WoLF-PHC CDF shifted right (higher SINRs)
- Look for: Fewer users below 5 dB threshold

---

## Checking Reproducibility

### Are your numbers close to the paper?

```bash
# Extract just WoLF-PHC results
python3 << 'EOF'
import json
with open("outputs_joint_ao/paper_results.json") as f:
    data = json.load(f)
    wolf = data["summary"]["fuzzy_wolf_phc"]
    print(f"Rate:       {wolf['rate_mean']:.2f} ± {wolf['rate_std']:.2f}")
    print(f"Protection: {wolf['protection_mean']:.1f}% ± {wolf['protection_std']:.1f}%")
    print(f"\nTarget from paper:")
    print(f"Rate:       11.87 ± 0.55")
    print(f"Protection: 81.6% ± 3.9%")
EOF
```

**Acceptable ranges:**
- Rate: 11.5 - 12.2 bits/s/Hz ✅
- Protection: 80% - 83% ✅
- Variance: ±0.4 - ±0.7 bits/s/Hz ✅

Small differences (±5%) are OK due to:
- Different hardware (faster CPU = slightly different random walk)
- Different OS (timing variations)
- Different Python version (minor numerical differences)

Large differences (>10%) suggest:
- ❌ Wrong configuration (check `src/irs_anti_jamming/thz/thz_config.py`)
- ❌ Incomplete training (check training_log.txt for episode count)
- ❌ Corrupted results file

---

## Customizing Experiments

### Change System Configuration
Edit: `src/irs_anti_jamming/thz/thz_config.py`

Example: Use smaller RIS (64 elements instead of 256)
```python
# Line ~30
n_ris_h=8,      # Change from 16
n_ris_v=8,      # Change from 16
```

Then rerun:
```bash
python scripts/run_joint_ao_results.py
```

### Change Training Duration
Edit: `scripts/run_joint_ao_results.py`

Example: Train for 300 episodes instead of 600
```python
# Line ~80
episodes=300,   # Change from 600
```

### Train Single Method Only
```bash
python3 << 'EOF'
from scripts.run_joint_ao_results.py import *
# Modify RL_METHODS to include only fuzzy_wolf_phc
EOF
```

---

## Still Have Questions?

1. **"What does SPDP-RIS do?"** → See Figure 8 (beam squint compensation)
2. **"Why 27 fuzzy states?"** → See README.md Section 8
3. **"How is power allocated?"** → See README.md Section 7
4. **"What's WoLF-PHC?"** → See README.md Section 9

---

**Congratulations on reproducing the results! 🎉**
