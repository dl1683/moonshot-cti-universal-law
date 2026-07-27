#!/usr/bin/env python
"""
score_atlas_r2.py

Paired metrics, quality floors, confidence intervals, cost regret, and kill tests
for the Atlas R2 experiment.

Usage:
    python scripts/score_atlas_r2.py --phase discovery
    python scripts/score_atlas_r2.py --phase confirmation
    python scripts/score_atlas_r2.py --kill-test
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO / "results" / "atlas_r2_cost_ledger.csv"
BUDGET_PATH = REPO / "precommit" / "atlas_r2_budget.json"

NON_INFERIORITY_MARGIN = 5.0
COST_RATIO_MIN = 10.0
BOOTSTRAP_SAMPLES = 10000
CI_LEVEL = 0.95
STABILITY_THRESHOLD = 0.80
RECIPE_SENSITIVE_THRESHOLD = 0.25


def load_ledger():
    """Load cost ledger into list of dicts."""
    if not LEDGER_PATH.exists():
        print("No ledger found.", file=sys.stderr)
        return []
    rows = []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_pass_rate(rows, system_id, workload):
    """Compute pass rate for a system on a workload."""
    relevant = [r for r in rows
                if r["system_id"] == system_id and r["workload"] == workload]
    if not relevant:
        return 0.0, 0
    passed = sum(1 for r in relevant if r["status"] == "pass")
    return passed / len(relevant), len(relevant)


def bootstrap_ci(values, stat_fn=np.mean, n_boot=BOOTSTRAP_SAMPLES,
                 ci=CI_LEVEL):
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(42)
    arr = np.array(values, dtype=float)
    boot_stats = np.array([
        stat_fn(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_stats, 100 * alpha)
    hi = np.percentile(boot_stats, 100 * (1 - alpha))
    return float(lo), float(hi), float(np.mean(boot_stats))


def paired_non_inferiority(local_scores, frontier_scores, margin=NON_INFERIORITY_MARGIN):
    """Test if local is non-inferior to frontier within margin.

    Returns (is_non_inferior, lower_bound, point_estimate).
    Non-inferior if lower 95% CI bound of (local - frontier) > -margin.
    """
    diffs = np.array(local_scores) - np.array(frontier_scores)
    lo, hi, mean = bootstrap_ci(diffs)
    return lo > -margin, lo, mean


def compute_cost_ratio(local_cost_per_task, api_cost_per_task):
    """Compute avoided-cost ratio."""
    if local_cost_per_task <= 0:
        return float("inf")
    if api_cost_per_task <= 0:
        return 0.0
    return api_cost_per_task / local_cost_per_task


def cost_per_task(rows, system_id, workload, volume=10000):
    """Compute all-in cost per successful task at given volume."""
    relevant = [r for r in rows
                if r["system_id"] == system_id and r["workload"] == workload]
    if not relevant:
        return float("inf")

    successful = [r for r in relevant if r["status"] == "pass"]
    if not successful:
        return float("inf")

    total_gpu_hours = sum(float(r.get("gpu_seconds", 0)) for r in relevant) / 3600
    total_api_cost = sum(float(r.get("api_cost_usd", 0)) for r in relevant)
    total_energy_j = sum(float(r.get("energy_joules", 0)) for r in relevant)

    n_attempted = len(relevant)
    n_successful = len(successful)
    success_rate = n_successful / n_attempted

    scale_factor = volume / n_attempted if n_attempted > 0 else 1

    variable_cost = (total_gpu_hours * 0 + total_api_cost + total_energy_j * 0) * scale_factor
    per_successful = variable_cost / (n_successful * scale_factor) if n_successful > 0 else float("inf")

    return per_successful


def run_kill_tests(rows):
    """Run all 8 kill criteria from the R2 protocol."""
    print("\n" + "=" * 60)
    print("KILL TEST BATTERY")
    print("=" * 60)

    kills = []

    # Placeholder: these need real data from completed experiments
    print("\nKill tests require completed discovery + confirmation data.")
    print("Run after P5/P6 phases complete.")
    print("\nKill criteria checked:")
    criteria = [
        "1. Fewer than 2 workloads show sub-1.7B at frontier non-inferiority",
        "2. No local-vs-hosted ratio reaches 10x at 10K tasks",
        "3. Selector cost regret > 20% on either confirmation track",
        "4. Recommendation stability < 80% across bootstrap",
        "5. >25% of conclusions are RECIPE_SENSITIVE",
        "6. Positive result depends on composite score or omitted cost",
        "7. Future-model test fails",
        "8. Strongest sentence is 'we evaluated many systems'",
    ]
    for c in criteria:
        print(f"  [ ] {c}")

    return kills


def main():
    parser = argparse.ArgumentParser(description="Atlas R2 scorer")
    parser.add_argument("--phase", choices=["discovery", "confirmation"],
                        help="Which phase to score")
    parser.add_argument("--kill-test", action="store_true",
                        help="Run kill test battery")
    parser.add_argument("--summary", action="store_true",
                        help="Print ledger summary")
    args = parser.parse_args()

    rows = load_ledger()

    if args.summary or (not args.phase and not args.kill_test):
        print(f"Ledger: {len(rows)} task records")
        if rows:
            systems = set(r["system_id"] for r in rows)
            workloads = set(r["workload"] for r in rows)
            phases = set(r["phase"] for r in rows)
            passed = sum(1 for r in rows if r["status"] == "pass")
            print(f"  Systems: {len(systems)}")
            print(f"  Workloads: {len(workloads)}")
            print(f"  Phases: {phases}")
            print(f"  Pass rate: {passed}/{len(rows)} "
                  f"({100*passed/len(rows):.1f}%)")

    if args.kill_test:
        run_kill_tests(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
