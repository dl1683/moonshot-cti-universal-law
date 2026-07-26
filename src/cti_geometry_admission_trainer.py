"""Geometry Admission Test: Stage A capacity training.

Teacher (seed 101), Transformer student (seeds 201-203), GRU student (seeds 301-303).
All hyperparameters from the Stage A specification.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from cti_geometry_admission_automaton import (
    DEVELOPMENT_KEY_JSON,
    key_from_json,
    generate_all_eval_sets,
    hash_eval_set,
    AutomatonTrainDataset,
    collate_fn,
)
from cti_geometry_admission_models import (
    create_teacher,
    create_transformer_student,
    create_gru_student,
    count_parameters,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_a"

RUNS = [
    {"name": "teacher",         "arch": "teacher",    "seed": 101, "lr": 3e-4},
    {"name": "transformer_s1",  "arch": "t_student",  "seed": 201, "lr": 5e-4},
    {"name": "transformer_s2",  "arch": "t_student",  "seed": 202, "lr": 5e-4},
    {"name": "transformer_s3",  "arch": "t_student",  "seed": 203, "lr": 5e-4},
    {"name": "gru_s1",          "arch": "gru",        "seed": 301, "lr": 1e-3},
    {"name": "gru_s2",          "arch": "gru",        "seed": 302, "lr": 1e-3},
    {"name": "gru_s3",          "arch": "gru",        "seed": 303, "lr": 1e-3},
]

MAX_STEPS = 5000
BATCH_SIZE = 512
WARMUP_STEPS = 250
EVAL_INTERVAL = 250
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.95)
EPS = 1e-8
COSINE_MIN_RATIO = 0.1

import hashlib
_stream_hash = hashlib.sha256(b"GAT_STAGE_A_TRAIN_STREAM_V1").digest()
TRAIN_STREAM_SEED = int.from_bytes(_stream_hash[:8], "little")


def cosine_lr(step: int, warmup: int, total: int, peak_lr: float, min_ratio: float) -> float:
    if step < warmup:
        return peak_lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def create_model(arch: str):
    if arch == "teacher":
        return create_teacher()
    elif arch == "t_student":
        return create_transformer_student()
    elif arch == "gru":
        return create_gru_student()
    raise ValueError(f"Unknown arch: {arch}")


@torch.no_grad()
def evaluate(model, eval_examples, device, batch_size=512):
    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(eval_examples), batch_size):
        batch_examples = eval_examples[start:start + batch_size]
        batch = collate_fn(batch_examples)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(dtype=torch.bfloat16):
            out = model(input_ids, attention_mask)
        preds = out["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]

    model.train()
    return correct / total if total > 0 else 0.0


def _build_config_data(run_cfg: dict, key, eval_sets: dict, model) -> dict:
    import hashlib as _hl
    import platform
    return {
        "run_config": run_cfg,
        "key_hash": _hl.sha256(json.dumps(key if isinstance(key, dict) else {"array": "omitted"}, sort_keys=True).encode()).hexdigest(),
        "eval_set_sizes": {k: len(v) for k, v in eval_sets.items()},
        "train_stream_seed": TRAIN_STREAM_SEED,
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "warmup": WARMUP_STEPS,
        "grad_clip": GRAD_CLIP,
        "weight_decay": WEIGHT_DECAY,
        "param_count": count_parameters(model),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _compute_config_hash(run_cfg: dict, key, eval_sets: dict, model) -> str:
    import hashlib as _hl
    config_data = _build_config_data(run_cfg, key, eval_sets, model)
    return _hl.sha256(json.dumps(config_data, sort_keys=True).encode()).hexdigest()


def _write_config_hash(run_dir: Path, run_cfg: dict, key, eval_sets: dict, model):
    """Write config identity hash before first optimizer step."""
    config_data = _build_config_data(run_cfg, key, eval_sets, model)
    config_hash = hashlib.sha256(json.dumps(config_data, sort_keys=True).encode()).hexdigest()

    with open(run_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    with open(run_dir / "config.sha256", "w") as f:
        f.write(config_hash)

    return config_hash


def train_one_run(run_cfg: dict, key, eval_sets: dict, device: torch.device):
    name = run_cfg["name"]
    arch = run_cfg["arch"]
    seed = run_cfg["seed"]
    peak_lr = run_cfg["lr"]

    run_dir = RESULTS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    log_path = run_dir / "training_log.jsonl"
    summary_path = run_dir / "summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        if summary.get("status") == "complete":
            print(f"[{name}] Already complete, skipping.")
            return summary

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = create_model(arch).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler()

    start_step = 0
    if checkpoint_path.exists():
        existing_hash_path = run_dir / "config.sha256"
        if existing_hash_path.exists():
            with open(existing_hash_path) as f:
                old_hash = f.read().strip()
            new_hash = _compute_config_hash(run_cfg, key, eval_sets, model)
            if old_hash != new_hash:
                print(f"[{name}] Config hash mismatch — restarting from step 0.")
                checkpoint_path.unlink()
            else:
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scaler.load_state_dict(ckpt["scaler"])
                start_step = ckpt["step"]
                print(f"[{name}] Resuming from step {start_step}")
        else:
            print(f"[{name}] No config hash for existing checkpoint — restarting.")
            checkpoint_path.unlink()

    config_hash = _write_config_hash(run_dir, run_cfg, key, eval_sets, model)
    print(f"[{name}] Config hash: {config_hash[:16]}...")

    dataset = AutomatonTrainDataset(key, seed=TRAIN_STREAM_SEED, max_length=16)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
        num_workers=0, pin_memory=False,
    )

    log_file = open(log_path, "a", encoding="utf-8")
    model.train()
    loader_iter = iter(loader)
    t0 = time.time()
    step_times = []

    best_eval = {"in_range": 0.0, "extrapolation": 0.0, "direct_edges": 0}
    eval_history = []

    for step in range(start_step, MAX_STEPS):
        step_t0 = time.time()
        lr = cosine_lr(step, WARMUP_STEPS, MAX_STEPS, peak_lr, COSINE_MIN_RATIO)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = next(loader_iter)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast(dtype=torch.bfloat16):
            out = model(input_ids, attention_mask)
            loss = F.cross_entropy(out["logits"], labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        step_time = time.time() - step_t0
        step_times.append(step_time)

        if step % 50 == 0:
            print(f"[{name}] step {step}/{MAX_STEPS} loss={loss.item():.4f} lr={lr:.6f} dt={step_time:.3f}s")

        if (step + 1) % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            acc_in = evaluate(model, eval_sets["dev_in_range"], device)
            acc_ext = evaluate(model, eval_sets["dev_extrapolation"], device)
            acc_edges = evaluate(model, eval_sets["direct_edges"], device)
            n_edges = int(acc_edges * len(eval_sets["direct_edges"]))
            acc_stress = evaluate(model, eval_sets["stress_long"], device)

            eval_result = {
                "step": step + 1,
                "in_range": acc_in,
                "extrapolation": acc_ext,
                "direct_edges_correct": n_edges,
                "direct_edges_total": len(eval_sets["direct_edges"]),
                "stress_long": acc_stress,
                "loss": loss.item(),
                "lr": lr,
                "wall_time": time.time() - t0,
            }
            eval_history.append(eval_result)
            log_file.write(json.dumps(eval_result) + "\n")
            log_file.flush()

            if acc_in > best_eval["in_range"]:
                best_eval["in_range"] = acc_in
            if acc_ext > best_eval["extrapolation"]:
                best_eval["extrapolation"] = acc_ext
            if n_edges > best_eval["direct_edges"]:
                best_eval["direct_edges"] = n_edges

            print(f"[{name}] EVAL step={step+1}: in_range={acc_in:.4f} extrap={acc_ext:.4f} "
                  f"edges={n_edges}/48 stress={acc_stress:.4f}")

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step + 1,
            }, checkpoint_path)

    log_file.close()
    wall_time = time.time() - t0

    summary = {
        "name": name,
        "arch": arch,
        "seed": seed,
        "lr": peak_lr,
        "params": count_parameters(model),
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "best_in_range": best_eval["in_range"],
        "best_extrapolation": best_eval["extrapolation"],
        "best_direct_edges": best_eval["direct_edges"],
        "final_in_range": eval_history[-1]["in_range"] if eval_history else 0,
        "final_extrapolation": eval_history[-1]["extrapolation"] if eval_history else 0,
        "final_direct_edges": eval_history[-1]["direct_edges_correct"] if eval_history else 0,
        "final_stress": eval_history[-1]["stress_long"] if eval_history else 0,
        "wall_seconds": wall_time,
        "median_step_ms": sorted(step_times)[len(step_times)//2] * 1000 if step_times else 0,
        "p95_step_ms": sorted(step_times)[int(len(step_times)*0.95)] * 1000 if step_times else 0,
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0,
        "eval_history": eval_history,
        "status": "complete",
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    torch.save(model.state_dict(), run_dir / "model_final.pt")

    return summary


def _two_eval_pass(summary: dict, in_range_thresh: float, extrap_thresh: float, edges_thresh: int) -> bool:
    """Both of the final two scheduled evaluations must pass thresholds."""
    history = summary.get("eval_history", [])
    if len(history) < 2:
        return False
    for e in history[-2:]:
        if e["in_range"] < in_range_thresh:
            return False
        if e["extrapolation"] < extrap_thresh:
            return False
        if e["direct_edges_correct"] < edges_thresh:
            return False
    return True


def check_capacity_gates(summaries: list[dict]) -> dict:
    teacher = [s for s in summaries if s["arch"] == "teacher"]
    t_students = [s for s in summaries if s["arch"] == "t_student"]
    gru_students = [s for s in summaries if s["arch"] == "gru"]

    gates = {}

    if teacher:
        t = teacher[0]
        two_eval_ok = _two_eval_pass(t, 0.995, 0.990, 48)
        gates["teacher"] = {
            "in_range_pass": t["final_in_range"] >= 0.995,
            "extrap_pass": t["final_extrapolation"] >= 0.990,
            "edges_pass": t["final_direct_edges"] == 48,
            "two_eval_pass": two_eval_ok,
            "in_range": t["final_in_range"],
            "extrapolation": t["final_extrapolation"],
            "direct_edges": t["final_direct_edges"],
        }
        gates["teacher"]["all_pass"] = all([
            gates["teacher"]["in_range_pass"],
            gates["teacher"]["extrap_pass"],
            gates["teacher"]["edges_pass"],
            gates["teacher"]["two_eval_pass"],
        ])

    for arch_name, students, label in [
        ("transformer", t_students, "t_student"),
        ("gru", gru_students, "gru"),
    ]:
        if not students:
            continue
        passing_seeds = [s for s in students
                         if s["final_in_range"] >= 0.990
                         and s["final_extrapolation"] >= 0.990
                         and s["final_direct_edges"] == 48
                         and _two_eval_pass(s, 0.990, 0.990, 48)]
        floor_ok = all(s["final_in_range"] >= 0.985 for s in students)
        gates[arch_name] = {
            "passing_seeds": len(passing_seeds),
            "seeds_required": 2,
            "floor_pass": floor_ok,
            "all_pass": len(passing_seeds) >= 2 and floor_ok,
            "per_seed": [{
                "name": s["name"],
                "in_range": s["final_in_range"],
                "extrapolation": s["final_extrapolation"],
                "direct_edges": s["final_direct_edges"],
            } for s in students],
        }

    gates["stage_a_pass"] = (
        gates.get("teacher", {}).get("all_pass", False)
        and gates.get("transformer", {}).get("all_pass", False)
    )
    gates["gru_pass"] = gates.get("gru", {}).get("all_pass", False)

    return gates


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    key = key_from_json(DEVELOPMENT_KEY_JSON)
    print("Generating frozen eval sets...")
    eval_sets = generate_all_eval_sets(key, seed=42)
    eval_hashes = {name: hash_eval_set(examples) for name, examples in eval_sets.items()}

    config = {
        "key": "development",
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "warmup_steps": WARMUP_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "grad_clip": GRAD_CLIP,
        "weight_decay": WEIGHT_DECAY,
        "betas": list(BETAS),
        "eps": EPS,
        "cosine_min_ratio": COSINE_MIN_RATIO,
        "eval_set_hashes": eval_hashes,
        "eval_set_sizes": {name: len(examples) for name, examples in eval_sets.items()},
        "runs": RUNS,
    }

    config_path = RESULTS_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    key_path = RESULTS_DIR / "development_key.json"
    with open(key_path, "w") as f:
        json.dump(DEVELOPMENT_KEY_JSON, f, indent=2)

    summaries = []
    for run_cfg in RUNS:
        print(f"\n{'='*60}")
        print(f"Starting: {run_cfg['name']} (arch={run_cfg['arch']}, seed={run_cfg['seed']})")
        print(f"{'='*60}")

        torch.cuda.reset_peak_memory_stats()
        summary = train_one_run(run_cfg, key, eval_sets, device)
        summaries.append(summary)

        print(f"\n[{run_cfg['name']}] Complete:")
        print(f"  In-range:      {summary['final_in_range']:.4f}")
        print(f"  Extrapolation: {summary['final_extrapolation']:.4f}")
        print(f"  Direct edges:  {summary['final_direct_edges']}/48")
        print(f"  Stress long:   {summary['final_stress']:.4f}")
        print(f"  Wall time:     {summary['wall_seconds']:.1f}s")
        print(f"  Peak GPU:      {summary['peak_gpu_mb']:.0f} MB")

    gates = check_capacity_gates(summaries)
    gates_path = RESULTS_DIR / "capacity_summary.json"
    with open(gates_path, "w") as f:
        json.dump(gates, f, indent=2)

    print(f"\n{'='*60}")
    print("CAPACITY GATE RESULTS")
    print(f"{'='*60}")
    print(json.dumps(gates, indent=2))

    if gates["stage_a_pass"]:
        print("\nSTAGE A: PASS - Teacher and Transformer student meet capacity gates.")
    else:
        print("\nSTAGE A: FAIL - Capacity gates not met.")

    if gates["gru_pass"]:
        print("GRU: PASS - GRU student meets capacity gates.")
    else:
        print("GRU: FAIL - GRU student does not meet capacity gates.")

    timing = {
        "runs": [{
            "name": s["name"],
            "wall_seconds": s["wall_seconds"],
            "median_step_ms": s["median_step_ms"],
            "p95_step_ms": s["p95_step_ms"],
            "peak_gpu_mb": s["peak_gpu_mb"],
        } for s in summaries],
        "total_wall_seconds": sum(s["wall_seconds"] for s in summaries),
        "total_gpu_hours": sum(s["wall_seconds"] for s in summaries) / 3600,
    }

    student_wall = sum(s["wall_seconds"] for s in summaries if s["arch"] != "teacher")
    avg_student_wall = student_wall / max(1, len([s for s in summaries if s["arch"] != "teacher"]))
    projected_stage_b_hours = (144 * avg_student_wall / 60) / 60
    timing["projected_stage_b_gpu_hours"] = projected_stage_b_hours
    timing["stage_b_budget_pass"] = projected_stage_b_hours <= 30

    timing_path = RESULTS_DIR / "timing_budget.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\nTiming budget:")
    print(f"  Total Stage A wall time: {timing['total_wall_seconds']:.0f}s ({timing['total_gpu_hours']:.2f} GPU-hours)")
    print(f"  Projected Stage B (144 runs): {projected_stage_b_hours:.1f} GPU-hours")
    print(f"  Stage B budget pass (<=30h): {timing['stage_b_budget_pass']}")


if __name__ == "__main__":
    main()
