#!/usr/bin/env python
"""
run_atlas_r2.py

Budget-enforcing Atlas R2.1 runner with authoritative task records,
segment-level energy tracking, and proportional energy allocation.

Protocol: R2.1 (precommit/atlas_r2_protocol_r2_1.md)

Usage:
    python scripts/run_atlas_r2.py --phase P1 --workload W-D2
    python scripts/run_atlas_r2.py --phase P1 --workload W-D2 --system qwen3_0.6b
    python scripts/run_atlas_r2.py --phase P1 --workload W-D3
    python scripts/run_atlas_r2.py --dry-run --phase P1 --workload W-D2
"""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUDGET_PATH = REPO / "precommit" / "atlas_r2_budget.json"
SELECTOR_PATH = REPO / "precommit" / "atlas_r2_selector.json"
LEDGER_PATH = REPO / "results" / "atlas_r2_cost_ledger.csv"
TASK_RECORDS_DIR = REPO / "results" / "cti_atlas_r2_task_records"
RESULTS_DIR = REPO / "results"

PROTOCOL_REVISION = "r2.1"
CONFIRMATION_PHASES = {"P6"}
LEDGER_HEADER = (
    "timestamp,phase,workload,system_id,task_id,status,"
    "gpu_seconds,wall_seconds,input_tokens,output_tokens,"
    "api_cost_usd,energy_joules,peak_memory_mb\n"
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text):
    """Remove Qwen3 <think>...</think> blocks from model output."""
    return THINK_RE.sub("", text).strip()


def _run_id():
    return f"run_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Budget checks
# ---------------------------------------------------------------------------

def load_budget():
    if not BUDGET_PATH.exists():
        print("ABORT: precommit/atlas_r2_budget.json missing", file=sys.stderr)
        sys.exit(1)
    with open(BUDGET_PATH, encoding="utf-8") as f:
        return json.load(f)


def check_budget_remaining(budget, phase):
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
    if not LEDGER_PATH.exists():
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER_PATH, "w", encoding="utf-8") as f:
            f.write(LEDGER_HEADER)
        print(f"Created {LEDGER_PATH}")


# ---------------------------------------------------------------------------
# Task record I/O
# ---------------------------------------------------------------------------

def _task_records_path(phase, workload, system_id):
    return (TASK_RECORDS_DIR
            / f"cti_atlas_r2_{PROTOCOL_REVISION}_{phase}_{workload}_{system_id}.json")


def _load_task_records(phase, workload, system_id):
    path = _task_records_path(phase, workload, system_id)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_task_records(records, phase, workload, system_id):
    TASK_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    path = _task_records_path(phase, workload, system_id)
    fd, tmp = tempfile.mkstemp(dir=str(TASK_RECORDS_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=True)
        for attempt in range(5):
            try:
                os.replace(tmp, str(path))
                return
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.5)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _log_ledger(phase, workload, system_id, task_id, status,
                gpu_seconds=0, wall_seconds=0, input_tokens=0, output_tokens=0,
                api_cost_usd=0, energy_joules=0, peak_memory_mb=0):
    """Append to cost ledger (13-column format, budget tracking only)."""
    ts = datetime.now(timezone.utc).isoformat()
    row = (f"{ts},{phase},{workload},{system_id},{task_id},{status},"
           f"{gpu_seconds},{wall_seconds},{input_tokens},{output_tokens},"
           f"{api_cost_usd},{energy_joules},{peak_memory_mb}\n")
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(row)


# ---------------------------------------------------------------------------
# Summary recomputation from task records
# ---------------------------------------------------------------------------

def _recompute_mkqa_summary(records, spec):
    """Recompute all W-D2 metrics from authoritative task records."""
    task_list = [v for k, v in records.items() if not k.startswith("__")]
    total = len(task_list)
    if total == 0:
        return {"system_id": spec.get("system_id", ""),
                "total": 0, "passed": 0, "pass_rate": 0.0}

    passed = sum(1 for t in task_list if t["status"] == "pass")

    sys.path.insert(0, str(REPO / "src"))
    from cti_atlas_workloads import mkqa_language_macro_average
    lang_stats = mkqa_language_macro_average(task_list)

    return {
        "system_id": spec.get("system_id", ""),
        "hf_id": spec.get("hf_id", ""),
        "params_b": spec.get("params_b", 0),
        "family": spec.get("family", ""),
        "protocol_revision": PROTOCOL_REVISION,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4),
        "macro_f1": lang_stats["macro_f1"],
        "macro_em": lang_stats["macro_em"],
        "language_breakdown": {k: v for k, v in lang_stats.items()
                               if k not in ("macro_f1", "macro_em")},
    }


def _recompute_policybench_summary(records, spec):
    """Recompute all W-D3 metrics from authoritative task records."""
    task_list = [v for k, v in records.items() if not k.startswith("__")]
    total = len(task_list)
    if total == 0:
        return {"system_id": spec.get("system_id", ""),
                "total": 0}

    parse_valid = sum(1 for t in task_list if t.get("parse_valid", False))
    all_correct = sum(1 for t in task_list if t.get("all_correct", False))
    household_scores = [t.get("household_score", 0.0) for t in task_list]
    macro_score = sum(household_scores) / len(household_scores)
    passed = sum(1 for t in task_list if t["status"] == "pass")

    return {
        "system_id": spec.get("system_id", ""),
        "hf_id": spec.get("hf_id", ""),
        "params_b": spec.get("params_b", 0),
        "family": spec.get("family", ""),
        "protocol_revision": PROTOCOL_REVISION,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4),
        "macro_household_score": round(macro_score, 4),
        "parse_valid_rate": round(parse_valid / total, 4),
        "all_correct_rate": round(all_correct / total, 4),
    }


# ---------------------------------------------------------------------------
# P1 W-D2: MKQA raw screen
# ---------------------------------------------------------------------------

def run_p1_mkqa(budget, system_filter=None):
    """P1: Raw W-D2 screen -- 9 checkpoints x 320 MKQA episodes (R2.1)."""
    sys.path.insert(0, str(REPO / "src"))
    import torch
    import yaml
    from cti_atlas_inference import load_model, generate
    from cti_atlas_workloads import (
        load_mkqa, format_mkqa_prompt, score_mkqa, gold_answer_hash,
    )
    from cti_energy_meter import EnergyMeter

    with open(REPO / "configs" / "atlas_r2_systems.yaml", encoding="utf-8") as f:
        systems_cfg = yaml.safe_load(f)

    local_systems = list(systems_cfg["local_checkpoints"].keys())
    if system_filter:
        if system_filter not in local_systems:
            print(f"Unknown system: {system_filter}", file=sys.stderr)
            return 1
        local_systems = [system_filter]

    episodes = load_mkqa(n_queries=40)
    print(f"\nP1 W-D2 (R2.1): {len(local_systems)} systems x "
          f"{len(episodes)} episodes")

    all_summaries = {}
    run_id = _run_id()
    print(f"Run ID: {run_id}")

    for sys_id in local_systems:
        print(f"\n{'='*50}")
        print(f"System: {sys_id}")
        print(f"{'='*50}")

        records = _load_task_records("P1", "W-D2", sys_id)
        done_ids = set(records.keys())
        remaining = [ep for ep in episodes if ep["task_id"] not in done_ids]

        if not remaining:
            print(f"  All {len(episodes)} tasks done, recomputing summary")
            spec = systems_cfg["local_checkpoints"][sys_id]
            spec["system_id"] = sys_id
            all_summaries[sys_id] = _recompute_mkqa_summary(records, spec)
            _print_mkqa_summary(all_summaries[sys_id])
            continue

        if done_ids:
            print(f"  Resuming: {len(done_ids)} done, {len(remaining)} remaining")

        model, tok, spec = load_model(sys_id)
        spec["system_id"] = sys_id
        meter = EnergyMeter()
        segment_id = f"seg_{uuid.uuid4().hex[:8]}"

        gen_times = []
        pending_records = []

        meter.start()

        for i, ep in enumerate(remaining):
            prompt = format_mkqa_prompt(ep)

            torch.cuda.synchronize()
            t_gen_start = time.perf_counter()
            result = generate(model, tok, prompt, max_new_tokens=64)
            torch.cuda.synchronize()
            t_gen_end = time.perf_counter()
            gen_time = t_gen_end - t_gen_start

            raw_output = result["text"]
            clean_text = _strip_think(raw_output)

            if result["is_empty"]:
                status = "empty_output"
                score_result = {"exact_match": False, "f1": 0.0,
                                "status": "empty_output",
                                "scorer_version": "r2.1.0"}
            else:
                score_result = score_mkqa(clean_text, ep)
                status = score_result["status"]

            task_record = {
                "task_id": ep["task_id"],
                "query_id": ep["query_id"],
                "lang": ep["lang"],
                "protocol_revision": PROTOCOL_REVISION,
                "run_id": run_id,
                "raw_output": raw_output.encode("ascii",
                                                errors="replace").decode(),
                "cleaned_prediction": clean_text.encode("ascii",
                                                        errors="replace").decode(),
                "exact_match": score_result["exact_match"],
                "f1": score_result["f1"],
                "status": status,
                "scorer_version": score_result.get("scorer_version", ""),
                "gold_answer_hash": gold_answer_hash(ep, "W-D2"),
                "model_revision": spec.get("hf_revision", ""),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "wall_seconds": result["wall_seconds"],
                "gpu_seconds": round(gen_time, 3),
                "segment_id": segment_id,
                "timed_out": result["timed_out"],
            }

            records[ep["task_id"]] = task_record
            gen_times.append((ep["task_id"], gen_time))
            pending_records.append(task_record)

            _log_ledger(
                phase="P1", workload="W-D2", system_id=sys_id,
                task_id=ep["task_id"], status=status,
                gpu_seconds=round(gen_time, 3),
                wall_seconds=result["wall_seconds"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
            )

            _save_task_records(records, "P1", "W-D2", sys_id)

            if (i + 1) % 40 == 0 or i == 0 or i == len(remaining) - 1:
                done_now = len(done_ids) + i + 1
                passed_now = sum(1 for r in records.values()
                                 if r["status"] == "pass")
                print(f"  [{done_now}/{len(episodes)}] "
                      f"pass={passed_now}/{done_now} "
                      f"({100*passed_now/done_now:.1f}%)")

        meter.stop()
        energy = meter.summary()

        total_gen_time = sum(gt for _, gt in gen_times)
        if total_gen_time > 0:
            for tid, gt in gen_times:
                allocated = energy["energy_joules"] * (gt / total_gen_time)
                records[tid]["allocated_energy_joules"] = round(allocated, 4)

        segment_record = {
            "segment_id": segment_id,
            "system_id": sys_id,
            "run_id": run_id,
            "tasks_in_segment": [tid for tid, _ in gen_times],
            "segment_energy_joules": energy["energy_joules"],
            "segment_duration_seconds": energy["duration_seconds"],
            "mean_power_watts": energy["mean_power_watts"],
            "peak_memory_mb": energy["peak_memory_mb"],
            "samples": energy["samples"],
        }
        records["__segment__" + segment_id] = segment_record

        _save_task_records(records, "P1", "W-D2", sys_id)

        summary = _recompute_mkqa_summary(records, spec)
        summary["energy"] = energy
        all_summaries[sys_id] = summary
        _print_mkqa_summary(summary)

        del model, tok
        torch.cuda.empty_cache()
        gc.collect()

    _write_summary("atlas_r2_p1_mkqa_r2_1.json", all_summaries)
    _print_final_mkqa_table(all_summaries)
    return 0


def _print_mkqa_summary(s):
    print(f"  pass={s['passed']}/{s['total']} ({s['pass_rate']:.1%}) | "
          f"F1={s['macro_f1']:.3f} | EM={s['macro_em']:.3f}")


def _print_final_mkqa_table(summaries):
    print("\n" + "=" * 60)
    print("P1 W-D2 SUMMARY (R2.1)")
    print("=" * 60)
    for sid, s in sorted(summaries.items(),
                         key=lambda x: x[1].get("pass_rate", 0), reverse=True):
        print(f"  {sid:20s}  pass={s['pass_rate']:.1%}  "
              f"F1={s.get('macro_f1', 0):.3f}  "
              f"EM={s.get('macro_em', 0):.3f}  "
              f"({s.get('params_b', '?')}B)")


# ---------------------------------------------------------------------------
# P1 W-D3: PolicyBench raw screen
# ---------------------------------------------------------------------------

def run_p1_policybench(budget, system_filter=None):
    """P1: Raw W-D3 screen -- 9 checkpoints x 100 households (R2.1)."""
    sys.path.insert(0, str(REPO / "src"))
    import torch
    import yaml
    from cti_atlas_inference import load_model, generate
    from cti_atlas_workloads import (
        load_policybench, format_policybench_prompt, score_policybench,
        gold_answer_hash,
    )
    from cti_energy_meter import EnergyMeter

    with open(REPO / "configs" / "atlas_r2_systems.yaml", encoding="utf-8") as f:
        systems_cfg = yaml.safe_load(f)

    local_systems = list(systems_cfg["local_checkpoints"].keys())
    if system_filter:
        if system_filter not in local_systems:
            print(f"Unknown system: {system_filter}", file=sys.stderr)
            return 1
        local_systems = [system_filter]

    episodes = load_policybench(n_households=100)
    print(f"\nP1 W-D3 (R2.1): {len(local_systems)} systems x "
          f"{len(episodes)} households")

    all_summaries = {}
    run_id = _run_id()
    print(f"Run ID: {run_id}")

    for sys_id in local_systems:
        print(f"\n{'='*50}")
        print(f"System: {sys_id}")
        print(f"{'='*50}")

        records = _load_task_records("P1", "W-D3", sys_id)
        done_ids = set(k for k in records if not k.startswith("__"))
        remaining = [ep for ep in episodes if ep["task_id"] not in done_ids]

        if not remaining:
            print(f"  All {len(episodes)} households done, recomputing summary")
            spec = systems_cfg["local_checkpoints"][sys_id]
            spec["system_id"] = sys_id
            all_summaries[sys_id] = _recompute_policybench_summary(records, spec)
            continue

        if done_ids:
            print(f"  Resuming: {len(done_ids)} done, {len(remaining)} remaining")

        model, tok, spec = load_model(sys_id)
        spec["system_id"] = sys_id
        meter = EnergyMeter()
        segment_id = f"seg_{uuid.uuid4().hex[:8]}"

        gen_times = []
        meter.start()

        for i, ep in enumerate(remaining):
            prompt = format_policybench_prompt(ep)

            torch.cuda.synchronize()
            t_gen_start = time.perf_counter()
            result = generate(model, tok, prompt, max_new_tokens=384)
            torch.cuda.synchronize()
            t_gen_end = time.perf_counter()
            gen_time = t_gen_end - t_gen_start

            raw_output = result["text"]
            clean_text = _strip_think(raw_output)

            if result["is_empty"]:
                status = "empty_output"
                score_result = {
                    "parse_valid": False, "fields_correct": 0,
                    "fields_total": len(ep["fields"]),
                    "household_score": 0.0, "all_correct": False,
                    "status": "empty_output", "field_results": {},
                    "scorer_version": "r2.1.0",
                }
            else:
                score_result = score_policybench(clean_text, ep)
                status = score_result["status"]

            task_record = {
                "task_id": ep["task_id"],
                "scenario_id": ep["scenario_id"],
                "protocol_revision": PROTOCOL_REVISION,
                "run_id": run_id,
                "raw_output": raw_output.encode("ascii",
                                                errors="replace").decode(),
                "cleaned_prediction": clean_text.encode("ascii",
                                                        errors="replace").decode(),
                "parse_valid": score_result["parse_valid"],
                "fields_correct": score_result["fields_correct"],
                "fields_total": score_result["fields_total"],
                "household_score": score_result["household_score"],
                "all_correct": score_result["all_correct"],
                "status": status,
                "field_results": score_result.get("field_results", {}),
                "scorer_version": score_result.get("scorer_version", ""),
                "gold_answer_hash": gold_answer_hash(ep, "W-D3"),
                "model_revision": spec.get("hf_revision", ""),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "wall_seconds": result["wall_seconds"],
                "gpu_seconds": round(gen_time, 3),
                "segment_id": segment_id,
                "timed_out": result["timed_out"],
            }

            records[ep["task_id"]] = task_record
            gen_times.append((ep["task_id"], gen_time))

            _log_ledger(
                phase="P1", workload="W-D3", system_id=sys_id,
                task_id=ep["task_id"], status=status,
                gpu_seconds=round(gen_time, 3),
                wall_seconds=result["wall_seconds"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
            )

            _save_task_records(records, "P1", "W-D3", sys_id)

            if (i + 1) % 20 == 0 or i == 0 or i == len(remaining) - 1:
                done_now = len(done_ids) + i + 1
                passed_now = sum(
                    1 for k, r in records.items()
                    if not k.startswith("__") and r["status"] == "pass"
                )
                print(f"  [{done_now}/{len(episodes)}] "
                      f"pass={passed_now}/{done_now} "
                      f"({100*passed_now/done_now:.1f}%)")

        meter.stop()
        energy = meter.summary()

        total_gen_time = sum(gt for _, gt in gen_times)
        if total_gen_time > 0:
            for tid, gt in gen_times:
                allocated = energy["energy_joules"] * (gt / total_gen_time)
                records[tid]["allocated_energy_joules"] = round(allocated, 4)

        segment_record = {
            "segment_id": segment_id,
            "system_id": sys_id,
            "run_id": run_id,
            "tasks_in_segment": [tid for tid, _ in gen_times],
            "segment_energy_joules": energy["energy_joules"],
            "segment_duration_seconds": energy["duration_seconds"],
            "mean_power_watts": energy["mean_power_watts"],
            "peak_memory_mb": energy["peak_memory_mb"],
            "samples": energy["samples"],
        }
        records["__segment__" + segment_id] = segment_record

        _save_task_records(records, "P1", "W-D3", sys_id)

        summary = _recompute_policybench_summary(records, spec)
        summary["energy"] = energy
        all_summaries[sys_id] = summary

        print(f"  pass={summary['passed']}/{summary['total']} "
              f"({summary['pass_rate']:.1%}) | "
              f"score={summary['macro_household_score']:.3f} | "
              f"parse={summary['parse_valid_rate']:.1%}")

        del model, tok
        torch.cuda.empty_cache()
        gc.collect()

    _write_summary("atlas_r2_p1_policybench_r2_1.json", all_summaries)

    print("\n" + "=" * 60)
    print("P1 W-D3 SUMMARY (R2.1)")
    print("=" * 60)
    for sid, s in sorted(all_summaries.items(),
                         key=lambda x: x[1].get("pass_rate", 0), reverse=True):
        print(f"  {sid:20s}  pass={s['pass_rate']:.1%}  "
              f"score={s.get('macro_household_score', 0):.3f}  "
              f"parse={s.get('parse_valid_rate', 0):.1%}  "
              f"({s.get('params_b', '?')}B)")

    return 0


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _write_summary(filename, summaries):
    out_path = RESULTS_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=True)
    print(f"\nSummary written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Atlas R2.1 budget-enforcing runner")
    parser.add_argument("--phase", required=True,
                        help="Phase to run (P0-P7)")
    parser.add_argument("--workload", default=None,
                        help="Workload ID (W-D1, W-D2, W-D3, W-C1, W-C2)")
    parser.add_argument("--system", default=None,
                        help="Run only this system ID (for resuming)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check budgets without executing")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Atlas R2.1 Runner - Phase {args.phase}")
    if args.workload:
        print(f"Workload: {args.workload}")
    print(f"Protocol: {PROTOCOL_REVISION}")
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
        return run_p1_mkqa(budget, system_filter=args.system)
    if args.phase == "P1" and args.workload == "W-D3":
        return run_p1_policybench(budget, system_filter=args.system)

    print(f"\nPhase {args.phase} ready for execution.")
    print(f"Dispatch not yet implemented for {args.phase}/{args.workload}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
