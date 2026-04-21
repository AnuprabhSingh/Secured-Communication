from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


METHODS = ["fuzzy_wolf_phc", "fast_q_learning", "baseline_ao", "no_irs_power"]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def is_monotonic(values: list[float], direction: str, atol: float) -> bool:
    if len(values) <= 1:
        return True
    if direction == "inc":
        return all(values[i + 1] >= values[i] - atol for i in range(len(values) - 1))
    if direction == "dec":
        return all(values[i + 1] <= values[i] + atol for i in range(len(values) - 1))
    raise ValueError(f"Unsupported direction: {direction}")


def frac(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def run_checks(results: dict, atol: float, required_ratio: float) -> list[CheckResult]:
    checks: list[CheckResult] = []

    sp = results["sweep_pmax"]["methods"]
    sm = results["sweep_m"]["methods"]
    ss = results["sweep_sinr"]["methods"]

    # 1) Fig.5 trend expectations: rate and protection should improve with Pmax.
    pmax_rate_ok = {
        m: is_monotonic(sp[m]["rate"], "inc", atol=atol) for m in METHODS
    }
    checks.append(
        CheckResult(
            name="pmax_rate_monotonic",
            passed=all(pmax_rate_ok.values()),
            detail=str(pmax_rate_ok),
        )
    )

    pmax_prot_ok = {
        m: is_monotonic(sp[m]["protection"], "inc", atol=atol) for m in METHODS
    }
    checks.append(
        CheckResult(
            name="pmax_protection_monotonic",
            passed=all(pmax_prot_ok.values()),
            detail=str(pmax_prot_ok),
        )
    )

    # 2) Fig.6 trend expectations: IRS methods improve with M; no-IRS mostly flat vs M.
    m_rate_irs_ok = {
        m: is_monotonic(sm[m]["rate"], "inc", atol=atol) for m in METHODS if m != "no_irs_power"
    }
    checks.append(
        CheckResult(
            name="m_rate_irs_monotonic",
            passed=all(m_rate_irs_ok.values()),
            detail=str(m_rate_irs_ok),
        )
    )

    no_irs_rate_span = max(sm["no_irs_power"]["rate"]) - min(sm["no_irs_power"]["rate"])
    checks.append(
        CheckResult(
            name="m_rate_no_irs_flat",
            passed=no_irs_rate_span <= 0.4,
            detail=f"span={no_irs_rate_span:.4f} (threshold<=0.4)",
        )
    )

    # 3) Fig.7 trend expectation: rate generally declines as SINR target increases.
    sinr_rate_ok = {
        m: is_monotonic(ss[m]["rate"], "dec", atol=atol) for m in METHODS
    }
    checks.append(
        CheckResult(
            name="sinr_target_rate_monotonic",
            passed=all(sinr_rate_ok.values()),
            detail=str(sinr_rate_ok),
        )
    )

    # 4) Ranking expectations from paper narrative.
    # Fig.5 text: proposed and baseline1 strong on rate; all IRS should beat no-IRS.
    total = len(results["sweep_pmax"]["x"])
    pass_count = 0
    for i in range(total):
        fuzzy = sp["fuzzy_wolf_phc"]["rate"][i]
        fast = sp["fast_q_learning"]["rate"][i]
        ao = sp["baseline_ao"]["rate"][i]
        no_irs = sp["no_irs_power"]["rate"][i]
        if fuzzy >= fast - atol and ao >= no_irs - atol and fuzzy >= no_irs - atol:
            pass_count += 1
    ratio = frac(pass_count, total)
    checks.append(
        CheckResult(
            name="pmax_rate_ranking",
            passed=ratio >= required_ratio,
            detail=f"pass={pass_count}/{total} ratio={ratio:.2f} required>={required_ratio:.2f}",
        )
    )

    # Fig.6 text: fuzzy should lead rate; no-IRS should be worst rate.
    total = len(results["sweep_m"]["x"])
    pass_count = 0
    for i in range(total):
        vals = {m: sm[m]["rate"][i] for m in METHODS}
        if vals["fuzzy_wolf_phc"] >= max(vals["fast_q_learning"], vals["baseline_ao"], vals["no_irs_power"]) - atol and vals[
            "no_irs_power"
        ] <= min(vals["fuzzy_wolf_phc"], vals["fast_q_learning"], vals["baseline_ao"]) + atol:
            pass_count += 1
    ratio = frac(pass_count, total)
    checks.append(
        CheckResult(
            name="m_rate_ranking",
            passed=ratio >= required_ratio,
            detail=f"pass={pass_count}/{total} ratio={ratio:.2f} required>={required_ratio:.2f}",
        )
    )

    # Fig.7 text: fuzzy and fast should outperform baseline1 on protection.
    total = len(results["sweep_sinr"]["x"])
    pass_count = 0
    for i in range(total):
        fuzzy = ss["fuzzy_wolf_phc"]["protection"][i]
        fast = ss["fast_q_learning"]["protection"][i]
        ao = ss["baseline_ao"]["protection"][i]
        if fuzzy >= ao - atol and fast >= ao - atol:
            pass_count += 1
    ratio = frac(pass_count, total)
    checks.append(
        CheckResult(
            name="sinr_protection_ranking",
            passed=ratio >= required_ratio,
            detail=f"pass={pass_count}/{total} ratio={ratio:.2f} required>={required_ratio:.2f}",
        )
    )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Check scientific reproduction quality from results.json")
    parser.add_argument("--results", type=str, default="outputs/results.json", help="Path to results.json")
    parser.add_argument(
        "--required-ratio",
        type=float,
        default=0.8,
        help="Required pass ratio for ranking checks (0 to 1)",
    )
    parser.add_argument("--atol", type=float, default=0.5, help="Absolute tolerance for monotonic/ranking comparisons")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"ERROR: results file not found: {path}")
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    checks = run_checks(data, atol=args.atol, required_ratio=args.required_ratio)

    passed = sum(int(c.passed) for c in checks)
    total = len(checks)

    print(f"Scientific Reproduction Check: {path}")
    print("=" * 72)
    for c in checks:
        flag = "PASS" if c.passed else "FAIL"
        print(f"[{flag}] {c.name}: {c.detail}")

    print("-" * 72)
    print(f"Summary: {passed}/{total} checks passed")

    if passed == total:
        print("Overall verdict: PASS (scientific reproduction criteria satisfied)")
        return 0

    print("Overall verdict: FAIL (tune baseline/method fidelity before extension work)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())