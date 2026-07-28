#!/usr/bin/env python
"""
score_atlas_r2.py

Paired metrics, quality floors, confidence intervals, cost accounting,
and kill tests for the Atlas R2.1 experiment.

Protocol: R2.1 (precommit/atlas_r2_protocol_r2_1.md)

Usage:
    python scripts/score_atlas_r2.py --summary
    python scripts/score_atlas_r2.py --gate-a
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
TASK_RECORDS_DIR = REPO / "results" / "cti_atlas_r2_task_records"
BUDGET_PATH = REPO / "precommit" / "atlas_r2_budget.json"
SYSTEMS_CONFIG = REPO / "configs" / "atlas_r2_systems.yaml"

PROTOCOL_REVISION = "r2.1"
NON_INFERIORITY_MARGIN = 0.05
COST_RATIO_MIN = 10.0
BOOTSTRAP_SAMPLES = 10000
CI_LEVEL = 0.95
STABILITY_THRESHOLD = 0.80
RECIPE_SENSITIVE_THRESHOLD = 0.25

LOCAL_HOURLY_RATE = 0.518
CAPITAL_RATE_PER_HOUR = 0.50
ELECTRICITY_RATE_PER_HOUR = 0.018


def load_ledger():
    if not LEDGER_PATH.exists():
        print("No ledger found.", file=sys.stderr)
        return []
    rows = []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_task_records(system_id, phase="P1", workload="W-D2"):
    """Load authoritative task records for a system."""
    path = (TASK_RECORDS_DIR
            / f"cti_atlas_r2_{PROTOCOL_REVISION}_{phase}_{workload}_{system_id}.json")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_systems_config():
    import yaml
    with open(SYSTEMS_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_pass_rate_from_records(system_id, phase="P1", workload="W-D2"):
    """Compute pass rate from authoritative task records (not ledger)."""
    records = load_task_records(system_id, phase, workload)
    task_records = {k: v for k, v in records.items() if not k.startswith("__")}
    if not task_records:
        return 0.0, 0
    passed = sum(1 for r in task_records.values() if r["status"] == "pass")
    return passed / len(task_records), len(task_records)


def bootstrap_ci(values, stat_fn=np.mean, n_boot=BOOTSTRAP_SAMPLES,
                 ci=CI_LEVEL):
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


def bootstrap_ci_clustered(values, cluster_ids, stat_fn=np.mean,
                           n_boot=BOOTSTRAP_SAMPLES, ci=CI_LEVEL):
    """Bootstrap CI with cluster resampling (e.g. by query_id)."""
    rng = np.random.default_rng(42)
    clusters = defaultdict(list)
    for v, cid in zip(values, cluster_ids):
        clusters[cid].append(v)
    cluster_keys = list(clusters.keys())
    n_clusters = len(cluster_keys)

    boot_stats = []
    for _ in range(n_boot):
        sampled_keys = rng.choice(cluster_keys, size=n_clusters, replace=True)
        sampled_values = []
        for k in sampled_keys:
            sampled_values.extend(clusters[k])
        boot_stats.append(stat_fn(np.array(sampled_values)))

    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_stats, 100 * alpha)
    hi = np.percentile(boot_stats, 100 * (1 - alpha))
    return float(lo), float(hi), float(np.mean(boot_stats))


def paired_non_inferiority(local_scores, frontier_scores,
                           margin=NON_INFERIORITY_MARGIN):
    """Test non-inferiority: lower 95% CI of (local - frontier) > -margin."""
    diffs = np.array(local_scores) - np.array(frontier_scores)
    lo, hi, mean = bootstrap_ci(diffs)
    return lo > -margin, lo, mean


def cost_per_task_local(rows, system_id, workload, volume=10000):
    """All-in cost per successful task for a local system at given volume.

    Cost = capital_amortization + electricity.
    """
    relevant = [r for r in rows
                if r["system_id"] == system_id and r["workload"] == workload
                ]
    if not relevant:
        return float("inf")

    successful = [r for r in relevant if r["status"] == "pass"]
    if not successful:
        return float("inf")

    total_gpu_hours = sum(float(r.get("gpu_seconds", 0)) for r in relevant) / 3600
    n_attempted = len(relevant)
    n_successful = len(successful)

    scale_factor = volume / n_attempted if n_attempted > 0 else 1
    total_hours_at_volume = total_gpu_hours * scale_factor

    variable_cost = total_hours_at_volume * LOCAL_HOURLY_RATE
    per_successful = variable_cost / (n_successful * scale_factor)

    return per_successful


def cost_per_task_api(rows, system_id, workload, volume=10000):
    """All-in cost per successful task for an API system at given volume."""
    relevant = [r for r in rows
                if r["system_id"] == system_id and r["workload"] == workload
                ]
    if not relevant:
        return float("inf")

    successful = [r for r in relevant if r["status"] == "pass"]
    if not successful:
        return float("inf")

    total_api_cost = sum(float(r.get("api_cost_usd", 0)) for r in relevant)
    n_attempted = len(relevant)
    n_successful = len(successful)

    scale_factor = volume / n_attempted if n_attempted > 0 else 1
    variable_cost = total_api_cost * scale_factor
    per_successful = variable_cost / (n_successful * scale_factor)

    return per_successful


def compute_cost_ratio(local_cost, api_cost):
    if local_cost <= 0:
        return float("inf")
    if api_cost <= 0:
        return 0.0
    return api_cost / local_cost


def run_gate_a(rows):
    """Gate A: 2 per family -- highest quality + cheapest within 10pt of family best.

    50/50 workload-level aggregate (W-D2 and W-D3 weighted equally).
    """
    print("\n" + "=" * 60)
    print("GATE A (R2.1)")
    print("=" * 60)

    cfg = load_systems_config()
    local = cfg.get("local_checkpoints", {})

    families = defaultdict(list)
    for sys_id, spec in local.items():
        wd2_rate, wd2_n = compute_pass_rate_from_records(sys_id, "P1", "W-D2")
        wd3_rate, wd3_n = compute_pass_rate_from_records(sys_id, "P1", "W-D3")

        if wd2_n == 0 and wd3_n == 0:
            print(f"  INCOMPLETE: {sys_id} has no R2.1 data")
            continue

        composite = 0.5 * wd2_rate + 0.5 * wd3_rate

        families[spec["family"]].append({
            "system_id": sys_id,
            "params_b": spec["params_b"],
            "wd2_rate": wd2_rate,
            "wd2_n": wd2_n,
            "wd3_rate": wd3_rate,
            "wd3_n": wd3_n,
            "composite": composite,
        })

    anchors = []
    for family, members in sorted(families.items()):
        members.sort(key=lambda x: x["composite"], reverse=True)
        best = members[0]
        anchors.append(best)
        print(f"\n  {family}: best = {best['system_id']} "
              f"(composite={best['composite']:.3f})")

        within_10pt = [m for m in members[1:]
                       if best["composite"] - m["composite"] <= 0.10]
        if within_10pt:
            cheapest = min(within_10pt, key=lambda x: x["params_b"])
            if cheapest["system_id"] != best["system_id"]:
                anchors.append(cheapest)
                print(f"           cheapest = {cheapest['system_id']} "
                      f"(composite={cheapest['composite']:.3f})")

    print(f"\n  Gate A output: {len(anchors)} anchors")
    for a in anchors:
        print(f"    {a['system_id']:20s}  W-D2={a['wd2_rate']:.1%} "
              f"W-D3={a['wd3_rate']:.1%}  "
              f"composite={a['composite']:.3f}")

    return anchors


def run_kill_tests(rows):
    """Run kill criteria from the R2 protocol."""
    print("\n" + "=" * 60)
    print("KILL TEST BATTERY (R2.1)")
    print("=" * 60)

    print("\nKill tests require completed discovery + confirmation data.")
    print("Run after P5/P6 phases complete.")
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

    return []


def print_summary(rows):
    """Print summary from ledger (budget) and task records (science)."""
    gpu_hours = sum(float(r.get("gpu_seconds", 0)) for r in rows) / 3600
    print(f"\nBudget ledger: {len(rows)} total records, {gpu_hours:.2f} GPU-hours")

    if TASK_RECORDS_DIR.exists():
        import yaml
        cfg = load_systems_config()
        local_systems = list(cfg.get("local_checkpoints", {}).keys())

        print(f"\nR2.1 Task Records (authoritative):")
        for workload in ["W-D2", "W-D3"]:
            wl_total = 0
            wl_passed = 0
            for sys_id in local_systems:
                rate, n = compute_pass_rate_from_records(sys_id, "P1", workload)
                if n > 0:
                    passed = int(rate * n)
                    wl_total += n
                    wl_passed += passed
                    print(f"  {sys_id:20s} {workload}: "
                          f"{passed}/{n} ({rate:.1%})")
            if wl_total > 0:
                print(f"  {'TOTAL':20s} {workload}: "
                      f"{wl_passed}/{wl_total} ({100*wl_passed/wl_total:.1f}%)")
    else:
        print("\n  No R2.1 task records yet.")


def main():
    parser = argparse.ArgumentParser(description="Atlas R2.1 scorer")
    parser.add_argument("--gate-a", action="store_true",
                        help="Run Gate A selection")
    parser.add_argument("--kill-test", action="store_true",
                        help="Run kill test battery")
    parser.add_argument("--summary", action="store_true",
                        help="Print ledger summary")
    args = parser.parse_args()

    rows = load_ledger()

    if args.summary or (not args.gate_a and not args.kill_test):
        print_summary(rows)

    if args.gate_a:
        run_gate_a(rows)

    if args.kill_test:
        run_kill_tests(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
