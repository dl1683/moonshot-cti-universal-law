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
import gc
import json
import hashlib
import re
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

    if args.phase == "P1" and args.workload == "W-D2":
        return run_p1_mkqa(budget)

    print(f"\nPhase {args.phase} ready for execution.")
    print(f"Dispatch not yet implemented for {args.phase}/{args.workload}.")
    return 0


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
RESULTS_DIR = REPO / "results"


def _strip_think(text):
    """Remove Qwen3 <think>...</think> blocks from model output."""
    return THINK_RE.sub("", text).strip()


def run_p1_mkqa(budget):
    """P1: Raw W-D2 screen — 9 checkpoints x 320 MKQA episodes."""
    sys.path.insert(0, str(REPO / "src"))
    import torch
    import yaml
    from cti_atlas_inference import load_model, generate
    from cti_atlas_workloads import load_mkqa, format_mkqa_prompt, score_mkqa
    from cti_energy_meter import EnergyMeter

    with open(REPO / "configs" / "atlas_r2_systems.yaml", encoding="utf-8") as f:
        systems_cfg = yaml.safe_load(f)

    local_systems = list(systems_cfg["local_checkpoints"].keys())
    episodes = load_mkqa(n_queries=40)
    episodes_with_answers = [ep for ep in episodes if ep["answers"]]

    print(f"\nP1 W-D2: {len(local_systems)} systems x "
          f"{len(episodes_with_answers)} scorable episodes "
          f"(of {len(episodes)} total)")

    all_summaries = {}

    for sys_id in local_systems:
        print(f"\n{'='*50}")
        print(f"System: {sys_id}")
        print(f"{'='*50}")

        model, tok, spec = load_model(sys_id)
        meter = EnergyMeter()

        passed = 0
        total = 0
        f1_sum = 0.0
        task_results = []
        meter.start()

        for i, ep in enumerate(episodes_with_answers):
            prompt = format_mkqa_prompt(ep)
            t0 = time.perf_counter()
            result = generate(model, tok, prompt, max_new_tokens=64)
            gpu_seconds = time.perf_counter() - t0

            clean_text = _strip_think(result["text"])
            score = score_mkqa(clean_text, ep)
            total += 1
            f1_sum += score["f1"]
            if score["status"] == "pass":
                passed += 1

            log_task(
                phase="P1", workload="W-D2", system_id=sys_id,
                task_id=ep["task_id"], status=score["status"],
                gpu_seconds=round(gpu_seconds, 3),
                wall_seconds=result["wall_seconds"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
            )

            task_results.append({
                "task_id": ep["task_id"],
                "lang": ep["lang"],
                "status": score["status"],
                "exact_match": score["exact_match"],
                "f1": score["f1"],
                "gpu_seconds": round(gpu_seconds, 3),
                "output_tokens": result["output_tokens"],
            })

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(episodes_with_answers)}] "
                      f"pass={passed}/{total} "
                      f"({100*passed/total:.1f}%)")

        meter.stop()
        energy = meter.summary()

        summary = {
            "system_id": sys_id,
            "hf_id": spec["hf_id"],
            "params_b": spec["params_b"],
            "family": spec["family"],
            "pass_rate": round(passed / total, 4) if total else 0,
            "mean_f1": round(f1_sum / total, 4) if total else 0,
            "passed": passed,
            "total": total,
            "energy": energy,
            "tasks": task_results,
        }
        all_summaries[sys_id] = summary

        print(f"  Final: {passed}/{total} "
              f"({100*passed/total:.1f}%) | "
              f"F1={summary['mean_f1']:.3f} | "
              f"{energy['energy_joules']:.0f}J | "
              f"{energy['mean_power_watts']:.0f}W avg")

        del model, tok
        torch.cuda.empty_cache()
        gc.collect()

    out_path = RESULTS_DIR / "atlas_r2_p1_mkqa_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nResults written to {out_path}")

    print("\n" + "=" * 60)
    print("P1 W-D2 SUMMARY")
    print("=" * 60)
    for sid, s in sorted(all_summaries.items(),
                         key=lambda x: x[1]["pass_rate"], reverse=True):
        print(f"  {sid:20s}  pass={s['pass_rate']:.1%}  "
              f"F1={s['mean_f1']:.3f}  "
              f"{s['energy']['energy_joules']:.0f}J  "
              f"({s['params_b']}B)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
