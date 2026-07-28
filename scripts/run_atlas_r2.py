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


def _thermal_gate(system_id, gpu_index=0, pause_temp_c=85,
                  resume_temp_c=78, poll_seconds=5,
                  sensor_failure_seconds=60):
    """Block between tasks until the GPU has safe thermal headroom."""
    import pynvml

    if resume_temp_c >= pause_temp_c:
        raise ValueError("resume_temp_c must be lower than pause_temp_c")

    def log(event, **fields):
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        print(
            f"{datetime.now(timezone.utc).isoformat()} THERMAL "
            f"event={event} system={system_id} {details}".rstrip(),
            flush=True,
        )

    initialized = False
    handle = None
    cooling = False

    try:
        while True:
            try:
                if not initialized:
                    pynvml.nvmlInit()
                    initialized = True
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                temp_c = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError as exc:
                log(
                    "sensor_error",
                    gpu_index=gpu_index,
                    error=type(exc).__name__,
                    assumed_hot=True,
                    retry_seconds=sensor_failure_seconds,
                )
                cooling = True
                if initialized:
                    try:
                        pynvml.nvmlShutdown()
                    except pynvml.NVMLError:
                        pass
                    initialized = False
                    handle = None
                time.sleep(sensor_failure_seconds)
                continue

            if not cooling and temp_c <= pause_temp_c:
                return

            if cooling and temp_c < resume_temp_c:
                log("resume", temp_c=temp_c, resume_below_c=resume_temp_c)
                return

            if not cooling:
                cooling = True
                log(
                    "pause",
                    temp_c=temp_c,
                    pause_above_c=pause_temp_c,
                    resume_below_c=resume_temp_c,
                )
            else:
                log("cooling", temp_c=temp_c,
                    resume_below_c=resume_temp_c)

            time.sleep(poll_seconds)
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass


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

def _task_records_path(phase, workload, system_id,
                       protocol_revision=PROTOCOL_REVISION):
    return (TASK_RECORDS_DIR
            / f"cti_atlas_r2_{protocol_revision}_{phase}_{workload}_{system_id}.json")


def _load_task_records(phase, workload, system_id,
                       protocol_revision=PROTOCOL_REVISION):
    path = _task_records_path(phase, workload, system_id,
                              protocol_revision)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_task_records(records, phase, workload, system_id,
                       protocol_revision=PROTOCOL_REVISION):
    TASK_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    path = _task_records_path(phase, workload, system_id,
                              protocol_revision)
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
            if i + 1 < len(remaining):
                _thermal_gate(sys_id)

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
            result = generate(model, tok, prompt, max_new_tokens=384,
                               timeout_seconds=120)
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
            if i + 1 < len(remaining):
                _thermal_gate(sys_id)

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
# R2.4 W-D3: PolicyBench structured generation (NRI-scored)
# ---------------------------------------------------------------------------

R2_4_REVISION = "r2.4"
R2_4_PANEL_REVISION = "r2.2"
R2_4_MAX_NEW_TOKENS = 448
R2_4_WATCHDOG_SECONDS = 120.0
R2_4_LATENCY_FLOOR_SECONDS = 30.0
R2_4_SYSTEM_GPU_CAP_SECONDS = 10_800.0
R2_4_TOTAL_GPU_CAP_SECONDS = 97_200.0

R2_4_PREVALENCE_PATH = REPO / "data" / "policybench" / "r2_2_prevalence.json"
R2_4_CHALLENGE_PATH = REPO / "data" / "policybench" / "r2_2_challenge.json"
R2_4_MANIFEST_PATH = REPO / "data" / "policybench" / "r2_2_panel_manifest.json"

R2_4_SMOKE_SYSTEMS = ["falcon_h1_0.5b", "gemma3_12b"]
R2_4_SMOKE_SALT_P = "atlas-r2.2-d3-smoke-p"
R2_4_SMOKE_SALT_C = "atlas-r2.2-d3-smoke-c"


def _compute_contract_fingerprint():
    """Hash of protocol + scorer + inference + runner + panel hashes."""
    paths = [
        REPO / "precommit" / "atlas_r2_protocol_r2_2.md",
        REPO / "precommit" / "atlas_r2_protocol_r2_3.md",
        REPO / "precommit" / "atlas_r2_protocol_r2_4.md",
        REPO / "src" / "cti_atlas_workloads.py",
        REPO / "src" / "cti_atlas_inference.py",
        REPO / "scripts" / "run_atlas_r2.py",
    ]
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    manifest = json.loads(R2_4_MANIFEST_PATH.read_bytes())
    h.update(manifest["prevalence"]["hash"].encode())
    h.update(manifest["challenge"]["hash"].encode())
    return h.hexdigest()


def _build_r2_4_panel_rows():
    """Load sealed panels and wrap with panel metadata."""
    sys.path.insert(0, str(REPO / "src"))
    from cti_atlas_workloads import load_r2_2_panel

    prevalence = load_r2_2_panel(R2_4_PREVALENCE_PATH)
    challenge = load_r2_2_panel(R2_4_CHALLENGE_PATH)

    rows = []
    for hh in prevalence:
        _preflight_household(hh)
        rows.append({"panel_id": "P", "stratum": None, "household": hh})
    for hh in challenge:
        _preflight_household(hh)
        rows.append({
            "panel_id": "C",
            "stratum": hh.get("stratum"),
            "household": hh,
        })
    return rows


def _preflight_household(hh):
    """Verify field sorting and hash."""
    fields = hh["fields"]
    gold = hh["gold_array"]
    sorted_fields = sorted(fields, key=lambda x: x.encode("utf-8"))
    if fields != sorted_fields:
        raise ValueError(f"Fields not UTF-8 sorted: {hh.get('scenario_id')}")
    if len(fields) != len(gold):
        raise ValueError(f"Fields/gold length mismatch: {hh.get('scenario_id')}")
    expected_hash = hashlib.sha256(
        "|".join(fields).encode("utf-8")).hexdigest()
    if hh.get("field_order_hash") and hh["field_order_hash"] != expected_hash:
        raise ValueError(f"Field order hash mismatch: {hh.get('scenario_id')}")


def _make_task_id(panel_id, hh):
    """Build R2.4 task ID from panel and household identity."""
    identity_string = hh["identity_string"]
    identity_hash = hashlib.sha256(
        identity_string.encode("utf-8")).hexdigest()
    return f"W-D3:{panel_id}:{identity_hash}"


def _select_smoke_households(panel_rows):
    """Select 24 smoke households per R2.2 Section 6."""
    p_rows = [r for r in panel_rows if r["panel_id"] == "P"]
    c_rows = [r for r in panel_rows if r["panel_id"] == "C"]

    def rank_key(salt, hh):
        identity_string = hh["identity_string"]
        return hashlib.sha256(
            (salt + "|" + identity_string).encode("utf-8")).hexdigest()

    p_sorted = sorted(p_rows, key=lambda r: rank_key(
        R2_4_SMOKE_SALT_P, r["household"]))
    smoke_p = p_sorted[:8]

    strata = {}
    for r in c_rows:
        s = r["stratum"]
        strata.setdefault(s, []).append(r)

    smoke_c = []
    for s_name in sorted(strata.keys()):
        s_rows = sorted(strata[s_name], key=lambda r: rank_key(
            R2_4_SMOKE_SALT_C, r["household"]))
        smoke_c.extend(s_rows[:4])

    return smoke_p + smoke_c


def _get_package_versions():
    """Collect pinned package versions for provenance."""
    import importlib.metadata
    pkgs = {}
    for name in ["torch", "transformers", "bitsandbytes",
                 "accelerate", "tokenizers", "nvidia-ml-py"]:
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pkgs[name] = "not_installed"
    pkgs["python"] = sys.version.split()[0]
    return pkgs


def run_p1_policybench_r2_4(budget, system_filter=None, smoke=False):
    """P1 W-D3 R2.4: 9 systems x 300 households with NRI scoring."""
    sys.path.insert(0, str(REPO / "src"))
    import yaml
    from cti_atlas_inference import SupervisedWorker
    from cti_atlas_workloads import (
        R2_2_SCORER_VERSION,
        format_r2_2_prompt, parse_r2_2_output,
        score_r2_2_household, _strip_think_block,
    )

    with open(REPO / "configs" / "atlas_r2_systems.yaml",
              encoding="utf-8") as f:
        systems_cfg = yaml.safe_load(f)

    local_systems = list(systems_cfg["local_checkpoints"].keys())

    if smoke:
        if system_filter and system_filter not in R2_4_SMOKE_SYSTEMS:
            print(f"Smoke only allows: {R2_4_SMOKE_SYSTEMS}",
                  file=sys.stderr)
            return 1
        if system_filter:
            local_systems = [system_filter]
        else:
            local_systems = [s for s in R2_4_SMOKE_SYSTEMS
                             if s in local_systems]
    elif system_filter:
        if system_filter not in local_systems:
            print(f"Unknown system: {system_filter}", file=sys.stderr)
            return 1
        local_systems = [system_filter]

    contract_fp = _compute_contract_fingerprint()
    panel_rows = _build_r2_4_panel_rows()
    pkg_versions = _get_package_versions()

    if smoke:
        selected_rows = _select_smoke_households(panel_rows)
        mode_label = "SMOKE"
    else:
        selected_rows = panel_rows
        mode_label = "FULL"

    manifest = json.loads(R2_4_MANIFEST_PATH.read_bytes())
    panel_hashes = {
        "P": manifest["prevalence"]["hash"],
        "C": manifest["challenge"]["hash"],
    }

    print(f"\nP1 W-D3 R2.4 ({mode_label}): {len(local_systems)} systems x "
          f"{len(selected_rows)} households")
    print(f"Contract: {contract_fp[:16]}...")
    print(f"Scorer: {R2_2_SCORER_VERSION}")

    run_id = _run_id()
    all_summaries = {}

    for sys_id in local_systems:
        print(f"\n{'='*50}")
        print(f"System: {sys_id} ({mode_label})")
        print(f"{'='*50}")

        records = _load_task_records(
            "P1", "W-D3", sys_id, protocol_revision=R2_4_REVISION)

        done_ids = {
            k for k, v in records.items()
            if not k.startswith("__")
            and v.get("execution_state") == "terminal"
            and v.get("protocol_revision") == R2_4_REVISION
            and v.get("contract_fingerprint") == contract_fp
        }

        remaining = []
        for row in selected_rows:
            hh = row["household"]
            task_id = _make_task_id(row["panel_id"], hh)
            if task_id not in done_ids:
                remaining.append(row)

        if not remaining:
            print(f"  All {len(selected_rows)} households done, "
                  f"recomputing summary")
            spec = systems_cfg["local_checkpoints"][sys_id]
            all_summaries[sys_id] = _recompute_r2_4_summary(
                records, spec, panel_rows, selected_rows)
            continue

        if done_ids:
            print(f"  Resuming: {len(done_ids)} done, "
                  f"{len(remaining)} remaining")

        gpu_spent = sum(
            v.get("gpu_seconds", 0)
            for k, v in records.items()
            if not k.startswith("__")
            and v.get("execution_state") == "terminal"
        )

        print(f"  Starting SupervisedWorker for {sys_id}...")
        worker = SupervisedWorker(sys_id)
        worker.start()
        spec = systems_cfg["local_checkpoints"][sys_id]
        scorer_hash = hashlib.sha256(
            (REPO / "src" / "cti_atlas_workloads.py").read_bytes()
        ).hexdigest()

        from cti_energy_meter import EnergyMeter
        meter = EnergyMeter()
        segment_id = f"seg_{uuid.uuid4().hex[:8]}"
        gen_times = []
        meter.start()

        for i, row in enumerate(remaining):
            hh = row["household"]
            panel_id = row["panel_id"]
            task_id = _make_task_id(panel_id, hh)
            identity_string = hh["identity_string"]
            identity_hash = hashlib.sha256(
                identity_string.encode("utf-8")).hexdigest()

            if gpu_spent + R2_4_WATCHDOG_SECONDS > R2_4_SYSTEM_GPU_CAP_SECONDS:
                print(f"  BUDGET: system cap reached at {gpu_spent:.0f}s")
                records[task_id] = {
                    "task_id": task_id,
                    "execution_state": "INCOMPLETE_BUDGET",
                    "protocol_revision": R2_4_REVISION,
                }
                _save_task_records(records, "P1", "W-D3", sys_id,
                                   protocol_revision=R2_4_REVISION)
                break

            _thermal_gate(sys_id)

            prompt = format_r2_2_prompt(hh)
            prompt_hash = hashlib.sha256(
                prompt.encode("utf-8")).hexdigest()

            if not worker.alive:
                print(f"  Respawning worker after watchdog kill...")
                worker.respawn()

            t0 = time.perf_counter()
            result = worker.generate(
                prompt, R2_4_MAX_NEW_TOKENS, 0.0,
                R2_4_WATCHDOG_SECONDS)
            user_wall = time.perf_counter() - t0
            gen_time = result.get("wall_seconds", user_wall)

            clean = _strip_think_block(result["text"])
            parsed, schema_valid, error_code = parse_r2_2_output(
                clean, expected_length=len(hh["fields"]),
                fields=hh["fields"])

            scores = score_r2_2_household(parsed, hh)

            qualifying = schema_valid and not result.get(
                "timed_out", False)
            latency_ok = user_wall >= R2_4_LATENCY_FLOOR_SECONDS

            gold_minified = json.dumps(
                hh["gold_array"], separators=(",", ":"))
            gold_hash = hashlib.sha256(
                gold_minified.encode("utf-8")).hexdigest()

            task_record = {
                "task_id": task_id,
                "phase": "P1",
                "workload": "W-D3",
                "system_id": sys_id,
                "protocol_revision": R2_4_REVISION,
                "panel_protocol_revision": R2_4_PANEL_REVISION,
                "panel_id": panel_id,
                "stratum": row["stratum"],
                "canonical_identity": hh.get("canonical_identity"),
                "canonical_identity_hash": identity_hash,
                "scenario_id": hh.get("scenario_id"),
                "gold_answer_hash": gold_hash,
                "fields": hh["fields"],
                "field_order_hash": hashlib.sha256(
                    "|".join(hh["fields"]).encode("utf-8")).hexdigest(),
                "expected_length": len(hh["fields"]),
                "prompt_sha256": prompt_hash,
                "contract_fingerprint": contract_fp,
                "raw_output": result["text"],
                "cleaned_prediction": clean,
                "parsed_prediction": parsed,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "eos_reached": result.get("eos_reached", False),
                "cap_hit": result.get("cap_hit", False),
                "generation_stop_reason": result.get(
                    "stop_reason", "unknown"),
                "wall_seconds": result["wall_seconds"],
                "user_wall_seconds": round(user_wall, 3),
                "latency_floor_met": latency_ok,
                "watchdog_abort": result.get("timed_out", False),
                "retry_count": 0,
                "retry_reason": None,
                "schema_valid": schema_valid,
                "schema_error_code": error_code,
                "qualifying_completion": qualifying,
                "execution_state": "terminal",
                "status": ("QUALIFYING_COMPLETION" if qualifying
                           else f"INVALID:{error_code or 'TIMEOUT'}"),
                "fields_correct": scores["n_correct"],
                "fields_total": scores["n_fields"],
                "household_agreement": scores["agreement"],
                "all_correct": (scores["n_correct"] == scores["n_fields"]),
                "field_results": scores["fields"],
                "scorer_version": R2_2_SCORER_VERSION,
                "scorer_sha256": scorer_hash,
                "model_id": spec.get("hf_id"),
                "model_revision": spec.get("hf_revision", ""),
                "tokenizer_revision": spec.get("hf_revision", ""),
                "package_revisions": pkg_versions,
                "run_id": run_id,
                "segment_id": segment_id,
                "gpu_seconds": round(gen_time, 3),
                "attempts": [{
                    "attempt_index": 0,
                    "attempt_id": f"{task_id}#a0",
                    "state": "completed",
                    "max_new_tokens": R2_4_MAX_NEW_TOKENS,
                    "temperature": 0.0,
                    "timeout_seconds": R2_4_WATCHDOG_SECONDS,
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "gpu_seconds": round(gen_time, 3),
                    "eos_reached": result.get("eos_reached", False),
                    "cap_hit": result.get("cap_hit", False),
                    "watchdog_abort": result.get("timed_out", False),
                    "infrastructure_error_code": None,
                }],
            }

            records[task_id] = task_record
            gen_times.append((task_id, gen_time))
            gpu_spent += gen_time

            _log_ledger(
                phase="P1", workload="W-D3", system_id=sys_id,
                task_id=task_id,
                status=task_record["status"],
                gpu_seconds=round(gen_time, 3),
                wall_seconds=result["wall_seconds"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
            )

            _save_task_records(records, "P1", "W-D3", sys_id,
                               protocol_revision=R2_4_REVISION)

            if (i + 1) % 20 == 0 or i == 0 or i == len(remaining) - 1:
                done_now = len(done_ids) + i + 1
                qualifying_now = sum(
                    1 for k, v in records.items()
                    if not k.startswith("__")
                    and v.get("qualifying_completion"))
                print(f"  [{done_now}/{len(selected_rows)}] "
                      f"qualifying={qualifying_now}/{done_now} "
                      f"gpu={gpu_spent:.0f}s "
                      f"stop={result.get('stop_reason', '?')}")

        meter.stop()
        energy = meter.summary()

        total_gen = sum(gt for _, gt in gen_times)
        if total_gen > 0:
            for tid, gt in gen_times:
                if tid in records and not tid.startswith("__"):
                    allocated = energy["energy_joules"] * (gt / total_gen)
                    records[tid]["allocated_energy_joules"] = round(
                        allocated, 4)

        if gen_times:
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

        _save_task_records(records, "P1", "W-D3", sys_id,
                           protocol_revision=R2_4_REVISION)

        summary = _recompute_r2_4_summary(
            records, spec, panel_rows, selected_rows)
        all_summaries[sys_id] = summary

        qualifying_count = summary.get("qualifying_completions", 0)
        total_count = summary.get("terminal_households", 0)
        print(f"  qualifying={qualifying_count}/{total_count} | "
              f"agreement={summary.get('agreement', 0):.3f}")

        worker.shutdown()
        gc.collect()

    summary_file = f"atlas_r2_p1_policybench_r2_4.json"
    _write_summary(summary_file, all_summaries)

    print(f"\n{'='*60}")
    print(f"P1 W-D3 R2.4 SUMMARY ({mode_label})")
    print(f"{'='*60}")
    for sid, s in sorted(all_summaries.items()):
        print(f"  {sid:20s}  "
              f"q={s.get('qualifying_completions', 0)}/"
              f"{s.get('terminal_households', 0)}  "
              f"agree={s.get('agreement', 0):.3f}")

    if smoke:
        smoke_pass = _check_smoke_pass(all_summaries, selected_rows)
        print(f"\nSMOKE RESULT: {'PASS' if smoke_pass else 'FAIL'}")
        return 0 if smoke_pass else 1

    return 0


def _check_smoke_pass(summaries, selected_rows):
    """Check smoke test pass criteria per R2.2 Section 6."""
    if len(summaries) < 2:
        print("  FAIL: fewer than 2 smoke systems completed")
        return False

    for sid in R2_4_SMOKE_SYSTEMS:
        s = summaries.get(sid)
        if not s:
            print(f"  FAIL: {sid} not in summaries")
            return False
        terminal = s.get("terminal_households", 0)
        if terminal != len(selected_rows):
            print(f"  FAIL: {sid} has {terminal}/{len(selected_rows)} "
                  f"terminal")
            return False

    gemma_s = summaries.get("gemma3_12b", {})
    gemma_q = gemma_s.get("qualifying_completions", 0)
    if gemma_q < 23:
        print(f"  FAIL: gemma3_12b qualifying={gemma_q} < 23")
        return False

    for sid in R2_4_SMOKE_SYSTEMS:
        s = summaries.get(sid, {})
        if s.get("cap_hit_count", 0) > 0:
            print(f"  FAIL: {sid} has cap hits")
            return False

    return True


def _recompute_r2_4_summary(records, spec, all_panel_rows,
                            expected_rows):
    """Recompute R2.4 summary with NRI metrics from task records."""
    tasks = {
        k: v for k, v in records.items()
        if not k.startswith("__")
        and v.get("execution_state") == "terminal"
    }

    terminal = len(tasks)
    qualifying = sum(1 for v in tasks.values()
                     if v.get("qualifying_completion"))

    agreements = [v.get("household_agreement", 0) for v in tasks.values()]
    mean_agreement = (sum(agreements) / len(agreements)
                      if agreements else 0)

    cap_hits = sum(1 for v in tasks.values() if v.get("cap_hit"))
    watchdog_aborts = sum(1 for v in tasks.values()
                         if v.get("watchdog_abort"))

    nri_by_panel = {}
    for panel_id in ["P", "C"]:
        panel_tasks = {k: v for k, v in tasks.items()
                       if v.get("panel_id") == panel_id}

        elig_rescue_n = 0
        elig_rescue_d = 0
        elig_harm_n = 0
        elig_harm_d = 0
        amt_rescue_n = 0
        amt_rescue_d = 0
        amt_harm_n = 0
        amt_harm_d = 0

        for v in panel_tasks.values():
            for fr in v.get("field_results", []):
                zbc = fr.get("zero_baseline_correct", False)
                correct = fr.get("correct", False)
                ftype = fr.get("type", "")
                rescue = fr.get("rescue", False)
                harm = fr.get("harm", False)

                if ftype == "eligibility":
                    if not zbc:
                        elig_rescue_d += 1
                        if rescue:
                            elig_rescue_n += 1
                    if zbc:
                        elig_harm_d += 1
                        if harm:
                            elig_harm_n += 1
                elif ftype == "amount":
                    if not zbc:
                        amt_rescue_d += 1
                        if rescue:
                            amt_rescue_n += 1
                    if zbc:
                        amt_harm_d += 1
                        if harm:
                            amt_harm_n += 1

        def safe_rate(n, d):
            return round(n / d, 6) if d > 0 else None

        elig_rr = safe_rate(elig_rescue_n, elig_rescue_d)
        elig_hr = safe_rate(elig_harm_n, elig_harm_d)
        elig_nri = (round(elig_rr - elig_hr, 6)
                    if elig_rr is not None and elig_hr is not None
                    else None)

        amt_rr = safe_rate(amt_rescue_n, amt_rescue_d)
        amt_hr = safe_rate(amt_harm_n, amt_harm_d)
        amt_nri = (round(amt_rr - amt_hr, 6)
                   if amt_rr is not None and amt_hr is not None
                   else None)

        macro_nri = (round(0.5 * elig_nri + 0.5 * amt_nri, 6)
                     if elig_nri is not None and amt_nri is not None
                     else None)

        nri_by_panel[panel_id] = {
            "eligibility": {
                "rescue_numerator": elig_rescue_n,
                "rescue_denominator": elig_rescue_d,
                "rescue_rate": elig_rr,
                "harm_numerator": elig_harm_n,
                "harm_denominator": elig_harm_d,
                "harm_rate": elig_hr,
                "nri": elig_nri,
            },
            "amount": {
                "rescue_numerator": amt_rescue_n,
                "rescue_denominator": amt_rescue_d,
                "rescue_rate": amt_rr,
                "harm_numerator": amt_harm_n,
                "harm_denominator": amt_harm_d,
                "harm_rate": amt_hr,
                "nri": amt_nri,
            },
            "macro_nri": macro_nri,
            "expected": (100 if panel_id == "P" else 200),
            "terminal": len(panel_tasks),
            "qualifying": sum(1 for v in panel_tasks.values()
                              if v.get("qualifying_completion")),
        }

    return {
        "system_id": spec.get("system_id", spec.get("hf_id", "")),
        "protocol_revision": R2_4_REVISION,
        "scorer_version": "r2.4.0",
        "terminal_households": terminal,
        "qualifying_completions": qualifying,
        "agreement": round(mean_agreement, 4),
        "cap_hit_count": cap_hits,
        "watchdog_abort_count": watchdog_aborts,
        "panels": nri_by_panel,
    }


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
    parser = argparse.ArgumentParser(
        description="Atlas R2 budget-enforcing runner")
    parser.add_argument("--phase", required=True,
                        help="Phase to run (P0-P7)")
    parser.add_argument("--workload", default=None,
                        help="Workload ID (W-D1, W-D2, W-D3, W-C1, W-C2)")
    parser.add_argument("--system", default=None,
                        help="Run only this system ID (for resuming)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check budgets without executing")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test (W-D3 only, 2 systems x 24 HH)")
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
        if args.smoke:
            return run_p1_policybench_r2_4(
                budget, system_filter=args.system, smoke=True)
        return run_p1_policybench_r2_4(
            budget, system_filter=args.system)

    print(f"\nPhase {args.phase} ready for execution.")
    print(f"Dispatch not yet implemented for {args.phase}/{args.workload}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
