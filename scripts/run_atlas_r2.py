#!/usr/bin/env python
"""
run_atlas_r2.py

Budget-enforcing Atlas R2 runner.
Refuses to run confirmation without a sealed selector.
Tracks GPU-hours, API dollars, and energy per task.

Usage:
    python scripts/run_atlas_r2.py --phase P0
    python scripts/run_atlas_r2.py --phase P1 --workload W-D2
    python scripts/run_atlas_r2.py --phase P6 --workload W-C1  # requires sealed selector
"""

import argparse
import json
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUDGET_PATH = REPO / "precommit" / "atlas_r2_budget.json"
SELECTOR_PATH = REPO / "precommit" / "atlas_r2_selector.json"
LEDGER_PATH = REPO / "results" / "atlas_r2_cost_ledger.csv"

CONFIRMATION_PHASES = {"P6"}
LEDGER_HEADER = (
    "timestamp,phase,workload,system_id,task_id,status,"
    "gpu_seconds,wall_seconds,input_tokens,output_tokens,"
    "api_cost_usd,energy_joules,peak_memory_mb\n"
)


def load_budget():
    if not BUDGET_PATH.exists():
        print("ABORT: precommit/atlas_r2_budget.json missing", file=sys.stderr)
        sys.exit(1)
    with open(BUDGET_PATH, encoding="utf-8") as f:
        return json.load(f)


def check_budget_remaining(budget, phase):
    """Check if the phase has budget remaining."""
    phases = budget.get("gpu_budget", {}).get("phases", {})
    phase_key = next((k for k in phases if k.startswith(phase)), None)
    if phase_key is None:
        print(f"ABORT: Phase {phase} not in budget", file=sys.stderr)
        sys.exit(1)

    allocated = phases[phase_key]["allocated"]

    spent = 0.0
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH, encoding="utf-8") as f:
            for line in f:
                if line.startswith("timestamp"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 7 and parts[1] == phase:
                    spent += float(parts[6]) / 3600.0

    remaining = allocated - spent
    if remaining <= 0:
        print(f"ABORT: Phase {phase} budget exhausted "
              f"({spent:.2f}h spent of {allocated:.2f}h)", file=sys.stderr)
        sys.exit(1)

    print(f"Phase {phase}: {remaining:.2f}h remaining of {allocated:.2f}h")
    return remaining


def check_total_budget(budget):
    """Check total GPU-hours across all phases."""
    binding_max = budget["gpu_budget"]["binding_maximum"]
    total_spent = 0.0
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH, encoding="utf-8") as f:
            for line in f:
                if line.startswith("timestamp"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    total_spent += float(parts[6]) / 3600.0

    if total_spent >= binding_max:
        print(f"ABORT: Total budget exhausted "
              f"({total_spent:.2f}h of {binding_max}h)", file=sys.stderr)
        sys.exit(1)

    print(f"Total: {total_spent:.2f}h spent of {binding_max}h")
    return binding_max - total_spent


def check_api_budget(budget):
    """Check total API spend."""
    api_max = budget["api_budget"]["binding_maximum"]
    total_api = 0.0
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH, encoding="utf-8") as f:
            for line in f:
                if line.startswith("timestamp"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 11:
                    cost = parts[10]
                    if cost:
                        total_api += float(cost)

    if total_api >= api_max:
        print(f"ABORT: API budget exhausted "
              f"(${total_api:.2f} of ${api_max})", file=sys.stderr)
        sys.exit(1)

    print(f"API: ${total_api:.2f} spent of ${api_max}")
    return api_max - total_api


def check_confirmation_seal(phase):
    """Confirmation phases require a sealed selector."""
    if phase not in CONFIRMATION_PHASES:
        return

    if not SELECTOR_PATH.exists():
        print(f"ABORT: Phase {phase} is confirmation but "
              f"precommit/atlas_r2_selector.json does not exist.\n"
              f"Freeze the selector before running confirmation.",
              file=sys.stderr)
        sys.exit(1)

    with open(SELECTOR_PATH, encoding="utf-8") as f:
        content = f.read()
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    print(f"Selector sealed: SHA256={digest[:16]}...")


def init_ledger():
    """Create the cost ledger CSV if it does not exist."""
    if not LEDGER_PATH.exists():
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER_PATH, "w", encoding="utf-8") as f:
            f.write(LEDGER_HEADER)
        print(f"Created {LEDGER_PATH}")


def log_task(phase, workload, system_id, task_id, status,
             gpu_seconds=0, wall_seconds=0, input_tokens=0, output_tokens=0,
             api_cost_usd=0, energy_joules=0, peak_memory_mb=0):
    """Append a single task result to the cost ledger."""
    ts = datetime.now(timezone.utc).isoformat()
    row = (f"{ts},{phase},{workload},{system_id},{task_id},{status},"
           f"{gpu_seconds},{wall_seconds},{input_tokens},{output_tokens},"
           f"{api_cost_usd},{energy_joules},{peak_memory_mb}\n")
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(row)


def main():
    parser = argparse.ArgumentParser(description="Atlas R2 budget-enforcing runner")
    parser.add_argument("--phase", required=True,
                        help="Phase to run (P0-P7)")
    parser.add_argument("--workload", default=None,
                        help="Workload ID (W-D1, W-D2, W-D3, W-C1, W-C2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check budgets without executing")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Atlas R2 Runner - Phase {args.phase}")
    if args.workload:
        print(f"Workload: {args.workload}")
    print("=" * 60)

    budget = load_budget()

    check_budget_remaining(budget, args.phase)
    check_total_budget(budget)
    check_api_budget(budget)
    check_confirmation_seal(args.phase)
    init_ledger()

    if args.dry_run:
        print("\nDRY RUN: Budget checks passed. No execution.")
        return 0

    print(f"\nPhase {args.phase} ready for execution.")
    print("Execution logic not yet implemented - this is the P0 skeleton.")
    print("Next: implement per-phase task dispatch, model loading, and scoring.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
