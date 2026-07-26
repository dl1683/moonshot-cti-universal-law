"""Geometry Admission Test: Stage A orchestrator.

Runs the full Stage A pipeline:
1. Capacity training (teacher + 3 Transformer students + 3 GRU students)
2. Trace extraction (raw R + observable connection for all banks)
3. Numerical gates audit
4. Timing budget projection
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from cti_geometry_admission_automaton import (
    DEVELOPMENT_KEY_JSON,
    key_from_json,
    generate_all_eval_sets,
    generate_anchors,
    partition_anchors_into_banks,
    audit_edge_coverage,
    hash_eval_set,
    collate_fn,
)
from cti_geometry_admission_models import (
    create_teacher,
    count_parameters,
)
from cti_geometry_admission_trainer import (
    train_one_run,
    check_capacity_gates,
    RUNS,
)
from cti_geometry_admission_extraction import (
    extract_hidden_states,
    extract_raw_trace,
    extract_observable_connection,
    generate_perturbations,
    check_numerical_gates,
    serialize_traces,
    TEACHER_DEPTH_LAYERS,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_a"


TRANSFORMER_RUNS = [r for r in RUNS if r["arch"] != "gru"]
GRU_RUNS = [r for r in RUNS if r["arch"] == "gru"]


def run_capacity_training(device: torch.device, include_gru: bool = False) -> list[dict]:
    """Run capacity training. GRU deferred until Transformer Stage C passes."""
    key = key_from_json(DEVELOPMENT_KEY_JSON)
    eval_sets = generate_all_eval_sets(key, seed=42)

    runs_to_execute = TRANSFORMER_RUNS + (GRU_RUNS if include_gru else [])
    if not include_gru:
        print("[Stage A-T] GRU runs deferred — Transformer-only capacity training.")

    summaries = []
    for run_cfg in runs_to_execute:
        print(f"\n{'='*60}")
        print(f"Capacity training: {run_cfg['name']}")
        print(f"{'='*60}")
        torch.cuda.reset_peak_memory_stats()
        summary = train_one_run(run_cfg, key, eval_sets, device)
        summaries.append(summary)

    gates = check_capacity_gates(summaries)
    gates_path = RESULTS_DIR / "capacity_summary.json"
    with open(gates_path, "w") as f:
        json.dump(gates, f, indent=2)

    return summaries


def run_extraction(device: torch.device) -> dict:
    """Extract raw and observable traces from trained teacher on all 32 banks."""
    key = key_from_json(DEVELOPMENT_KEY_JSON)

    teacher_path = RESULTS_DIR / "teacher" / "model_final.pt"
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher model not found: {teacher_path}")

    teacher = create_teacher().to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher.eval()

    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)

    coverage = audit_edge_coverage(key, anchors)
    anchor_manifest = {
        "n_anchors": len(anchors),
        "n_banks": len(banks),
        "bank_size": len(banks[0]),
        "edge_coverage": coverage,
    }
    with open(RESULTS_DIR / "anchor_manifest.json", "w") as f:
        json.dump(anchor_manifest, f, indent=2)

    all_raw_hashes = []
    all_obs_hashes = []
    all_numerical = []
    extraction_times = []
    obs_phase_timings = []

    for bank_idx, bank in enumerate(banks):
        print(f"\nExtracting bank {bank_idx}/{len(banks)}...")
        t0 = time.time()

        ticks = extract_hidden_states(teacher, bank, device, TEACHER_DEPTH_LAYERS)
        raw_transitions = extract_raw_trace(ticks, list(range(len(TEACHER_DEPTH_LAYERS))))

        perturbations = []
        for anchor in bank:
            perturbed = generate_perturbations(anchor, key)
            perturbations.append(perturbed)

        obs_transitions = extract_observable_connection(
            teacher, bank, perturbations, device, TEACHER_DEPTH_LAYERS,
        )

        first_key = next(iter(obs_transitions))
        if "timings" in obs_transitions[first_key]:
            obs_phase_timings.append(obs_transitions[first_key]["timings"])

        numerical = check_numerical_gates(raw_transitions, obs_transitions)
        all_numerical.append({"bank": bank_idx, "gates": numerical})

        anchor_hashes = [a["hash"] for a in bank]
        raw_hash, obs_hash = serialize_traces(
            raw_transitions, obs_transitions, bank_idx, anchor_hashes,
            RESULTS_DIR,
        )
        all_raw_hashes.append(raw_hash)
        all_obs_hashes.append(obs_hash)

        dt = time.time() - t0
        extraction_times.append(dt)
        print(f"  Bank {bank_idx}: {dt:.1f}s, numerical={'PASS' if numerical['all_pass'] else 'FAIL'}")

    t0 = time.time()
    raw_hashes_2 = []
    obs_hashes_2 = []
    for bank_idx, bank in enumerate(banks):
        ticks = extract_hidden_states(teacher, bank, device, TEACHER_DEPTH_LAYERS)
        raw_transitions = extract_raw_trace(ticks, list(range(len(TEACHER_DEPTH_LAYERS))))

        perturbations = []
        for anchor in bank:
            perturbed = generate_perturbations(anchor, key)
            perturbations.append(perturbed)
        obs_transitions = extract_observable_connection(
            teacher, bank, perturbations, device, TEACHER_DEPTH_LAYERS,
        )

        anchor_hashes = [a["hash"] for a in bank]
        raw_hash, obs_hash = serialize_traces(
            raw_transitions, obs_transitions, bank_idx, anchor_hashes,
            RESULTS_DIR / "repeat_check",
        )
        raw_hashes_2.append(raw_hash)
        obs_hashes_2.append(obs_hash)

    repeat_dt = time.time() - t0
    repeat_match_raw = all(h1 == h2 for h1, h2 in zip(all_raw_hashes, raw_hashes_2))
    repeat_match_obs = all(h1 == h2 for h1, h2 in zip(all_obs_hashes, obs_hashes_2))

    raw_manifest = {
        "bank_hashes": {str(i): h for i, h in enumerate(all_raw_hashes)},
        "repeat_match": repeat_match_raw,
    }
    obs_manifest = {
        "bank_hashes": {str(i): h for i, h in enumerate(all_obs_hashes)},
        "repeat_match": repeat_match_obs,
    }

    with open(RESULTS_DIR / "raw_trace_manifest.json", "w") as f:
        json.dump(raw_manifest, f, indent=2)
    with open(RESULTS_DIR / "observable_trace_manifest.json", "w") as f:
        json.dump(obs_manifest, f, indent=2)

    numerical_audit = {
        "banks": all_numerical,
        "all_pass": all(n["gates"]["all_pass"] for n in all_numerical),
        "repeat_match_raw": repeat_match_raw,
        "repeat_match_obs": repeat_match_obs,
    }
    with open(RESULTS_DIR / "numerical_audit.json", "w") as f:
        json.dump(numerical_audit, f, indent=2)

    phase_timing_summary = {}
    if obs_phase_timings:
        for key in obs_phase_timings[0]:
            vals = [t[key] for t in obs_phase_timings if key in t]
            phase_timing_summary[key] = {
                "mean_s": float(np.mean(vals)),
                "total_s": float(np.sum(vals)),
            }

    with open(RESULTS_DIR / "extraction_phase_timings.json", "w") as f:
        json.dump(phase_timing_summary, f, indent=2)

    return {
        "extraction_times": extraction_times,
        "repeat_time": repeat_dt,
        "all_numerical_pass": numerical_audit["all_pass"],
        "repeat_match_raw": repeat_match_raw,
        "repeat_match_obs": repeat_match_obs,
        "phase_timings": phase_timing_summary,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("STAGE A: CAPACITY TRAINING")
    print("=" * 60)
    summaries = run_capacity_training(device)

    gates_path = RESULTS_DIR / "capacity_summary.json"
    with open(gates_path) as f:
        gates = json.load(f)

    if not gates.get("stage_a_pass"):
        print("\nSTAGE A CAPACITY: FAIL")
        print("Cannot proceed to extraction.")
        return

    print("\nSTAGE A CAPACITY: PASS")

    print("\n" + "=" * 60)
    print("STAGE A: TRACE EXTRACTION")
    print("=" * 60)
    extraction = run_extraction(device)

    print("\n" + "=" * 60)
    print("STAGE A RESULTS")
    print("=" * 60)
    print(f"Numerical gates: {'PASS' if extraction['all_numerical_pass'] else 'FAIL'}")
    print(f"Repeat match (raw): {extraction['repeat_match_raw']}")
    print(f"Repeat match (obs): {extraction['repeat_match_obs']}")
    print(f"Mean extraction time per bank: {np.mean(extraction['extraction_times']):.1f}s")

    total_training = sum(s["wall_seconds"] for s in summaries)
    total_extraction = sum(extraction["extraction_times"]) + extraction["repeat_time"]

    timing = {
        "training_seconds": total_training,
        "extraction_seconds": total_extraction,
        "total_stage_a_seconds": total_training + total_extraction,
        "total_stage_a_hours": (total_training + total_extraction) / 3600,
    }
    with open(RESULTS_DIR / "timing_budget.json", "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\nTotal Stage A time: {timing['total_stage_a_hours']:.2f} GPU-hours")

    launch_gate = (
        gates.get("stage_a_pass", False)
        and extraction["all_numerical_pass"]
        and extraction["repeat_match_raw"]
        and extraction["repeat_match_obs"]
    )
    print(f"\nLaunch gate for Stage B: {'PASS' if launch_gate else 'FAIL'}")


if __name__ == "__main__":
    main()
