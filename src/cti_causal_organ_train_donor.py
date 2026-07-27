"""Donor capacity training for Causal Skill Organ admission test.

Protocol: CSO_ADMISSION_V1 (locked Jul 26 2026).
Gate: >=99.5% exact acc lengths 1-12, >=99.0% lengths 13-32,
      >=99.0% excluded compositions.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cti_causal_register_transducer import (
    MOD, NUM_REGS,
    init_partitions, verify_precommit,
    generate_training_example,
    generate_length_extrapolation,
    generate_excluded_bigram,
    generate_withheld_trigram,
    generate_held_out_state,
    generate_full_intersection,
    execute_program,
)
from cti_causal_organ_models import (
    create_donor, count_parameters,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "causal_organ"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
SMOKE_CHECKPOINT_DIR = RESULTS_DIR / "checkpoints_smoke"

TORCH_SEED = 2026

MAX_STEPS = 50_000
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
EVAL_EVERY = 500
CHECKPOINT_EVERY = 2000
COOLDOWN_SECONDS = 10

SMOKE_STEPS = 500
SMOKE_EVAL_EVERY = 100

CAPACITY_GATES = {
    "train_lengths": 0.995,
    "length_extrapolation": 0.990,
    "excluded_bigram": 0.990,
}


def make_batch(rng, partitions, batch_size, generator_fn, **kwargs):
    """Generate a batch of examples using the given generator function."""
    init_states = []
    ops_list = []
    targets = []
    max_len = 0

    for _ in range(batch_size):
        ex = generator_fn(rng, partitions, **kwargs)
        init_states.append(ex["init_state"])
        ops_list.append(ex["ops"])
        targets.append(ex["final_state"])
        max_len = max(max_len, len(ex["ops"]))

    init_regs = torch.tensor(
        np.array(init_states), dtype=torch.long, device=DEVICE
    )
    target_regs = torch.tensor(
        np.array(targets), dtype=torch.long, device=DEVICE
    )

    ops_padded = np.zeros((batch_size, max_len), dtype=np.int64)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    for i, op_seq in enumerate(ops_list):
        ops_padded[i, :len(op_seq)] = op_seq
        mask[i, :len(op_seq)] = True

    ops = torch.tensor(ops_padded, dtype=torch.long, device=DEVICE)
    ops_mask = torch.tensor(mask, dtype=torch.bool, device=DEVICE)

    return init_regs, ops, ops_mask, target_regs


def compute_loss_and_acc(model, init_regs, ops, ops_mask, target_regs):
    """Forward pass + per-register cross-entropy loss + exact accuracy."""
    logits, _ = model(init_regs, ops, ops_mask=ops_mask)

    loss = sum(
        F.cross_entropy(logits[r], target_regs[:, r])
        for r in range(NUM_REGS)
    ) / NUM_REGS

    preds = torch.stack([lg.argmax(dim=-1) for lg in logits], dim=1)
    exact_match = (preds == target_regs).all(dim=1).float().mean()

    return loss, exact_match.item()


def lr_schedule(step, warmup, max_steps, base_lr):
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def evaluate(model, rng, partitions, n_batches=10, batch_size=128):
    """Evaluate on all strata."""
    model.eval()
    results = {}

    strata = {
        "train_lengths": (generate_training_example, {}),
        "length_extrapolation": (generate_length_extrapolation, {}),
        "excluded_bigram": (generate_excluded_bigram, {}),
        "withheld_trigram": (generate_withheld_trigram, {}),
        "held_out_state": (generate_held_out_state, {}),
        "full_intersection": (generate_full_intersection, {}),
    }

    with torch.no_grad():
        for name, (gen_fn, kwargs) in strata.items():
            total_correct = 0
            total_count = 0
            for _ in range(n_batches):
                init_regs, ops, ops_mask, target_regs = make_batch(
                    rng, partitions, batch_size, gen_fn, **kwargs
                )
                _, acc = compute_loss_and_acc(
                    model, init_regs, ops, ops_mask, target_regs
                )
                total_correct += acc * batch_size
                total_count += batch_size
            results[name] = total_correct / total_count

    model.train()
    return results


def check_gates(eval_results):
    """Check if capacity gates are met."""
    passed = True
    for gate, threshold in CAPACITY_GATES.items():
        acc = eval_results.get(gate, 0.0)
        status = "PASS" if acc >= threshold else "FAIL"
        if acc < threshold:
            passed = False
        print(f"  {gate}: {acc:.4f} (>= {threshold}) [{status}]")
    return passed


def save_checkpoint(model, optimizer, step, eval_results, path,
                    rng_state=None, precommit_hash=None, scaler=None,
                    best_acc=None, eval_rng_state=None):
    """Save training checkpoint with RNG state for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "eval_results": eval_results,
        "torch_rng_state": torch.random.get_rng_state(),
    }
    if DEVICE.type == "cuda":
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state()
    if rng_state is not None:
        ckpt["numpy_rng_state"] = rng_state
    if precommit_hash is not None:
        ckpt["precommit_integrity_sha256"] = precommit_hash
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    if best_acc is not None:
        ckpt["best_acc"] = best_acc
    if eval_rng_state is not None:
        ckpt["eval_rng_state"] = eval_rng_state
    torch.save(ckpt, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path, expected_precommit_hash=None,
                    scaler=None):
    """Load training checkpoint. Returns dict with step, numpy_rng_state,
    best_acc, eval_rng_state."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if expected_precommit_hash is not None:
        saved_hash = ckpt.get("precommit_integrity_sha256")
        if saved_hash is not None and saved_hash != expected_precommit_hash:
            raise ValueError(
                f"Checkpoint precommit hash mismatch.\n"
                f"  Checkpoint: {saved_hash}\n"
                f"  Current:    {expected_precommit_hash}\n"
                "Precommit changed since checkpoint was saved."
            )
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "torch_rng_state" in ckpt:
        torch.random.set_rng_state(ckpt["torch_rng_state"].cpu())
    if DEVICE.type == "cuda" and "cuda_rng_state" in ckpt:
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    step = ckpt["step"]
    print(f"  Resumed from step {step}: {path}")
    return {
        "step": step,
        "numpy_rng_state": ckpt.get("numpy_rng_state"),
        "best_acc": ckpt.get("best_acc", 0.0),
        "eval_rng_state": ckpt.get("eval_rng_state"),
    }


def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint file in the given directory."""
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("donor_step_*.pt"))
    return ckpts[-1] if ckpts else None


def _get_precommit_hash() -> str:
    """Read integrity hash from the precommit artifact."""
    precommit_path = RESULTS_DIR / "precommit.json"
    with open(precommit_path) as f:
        return json.load(f)["integrity_sha256"]


def train_donor(max_steps=MAX_STEPS, smoke=False):
    """Train the donor model."""
    ckpt_dir = SMOKE_CHECKPOINT_DIR if smoke else CHECKPOINT_DIR

    if smoke:
        max_steps = SMOKE_STEPS
        eval_every = SMOKE_EVAL_EVERY
        checkpoint_every = SMOKE_STEPS
        print("=" * 60)
        print("DONOR CAPACITY TRAINING: SMOKE TEST")
        print("=" * 60)
    else:
        eval_every = EVAL_EVERY
        checkpoint_every = CHECKPOINT_EVERY
        print("=" * 60)
        print("DONOR CAPACITY TRAINING: FULL RUN")
        print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Max steps: {max_steps}")
    print()

    partitions = init_partitions()
    verify_precommit(partitions)
    precommit_hash = _get_precommit_hash()
    print()

    torch.manual_seed(TORCH_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed(TORCH_SEED)

    model = create_donor()
    model = model.to(DEVICE)
    n_params = count_parameters(model)
    print(f"Total parameters: {n_params:,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    start_step = 0
    rng = np.random.RandomState(42)
    eval_rng = np.random.RandomState(9999)
    best_acc = 0.0
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt is not None and not smoke:
        ckpt_data = load_checkpoint(
            model, optimizer, latest_ckpt,
            expected_precommit_hash=precommit_hash,
            scaler=scaler,
        )
        start_step = ckpt_data["step"]
        if ckpt_data["numpy_rng_state"] is not None:
            rng.set_state(ckpt_data["numpy_rng_state"])
        else:
            rng = np.random.RandomState(42 + start_step)
        best_acc = ckpt_data["best_acc"]
        if ckpt_data["eval_rng_state"] is not None:
            eval_rng.set_state(ckpt_data["eval_rng_state"])

    gate_passed = False
    eval_results = {}
    t0 = time.time()

    for step in range(start_step, max_steps):
        for pg in optimizer.param_groups:
            pg["lr"] = lr_schedule(step, WARMUP_STEPS, max_steps, LR)

        init_regs, ops, ops_mask, target_regs = make_batch(
            rng, partitions, BATCH_SIZE, generate_training_example
        )

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            loss, acc = compute_loss_and_acc(
                model, init_regs, ops, ops_mask, target_regs
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  step {step:6d} | loss {loss.item():.4f} | "
                  f"train_acc {acc:.4f} | lr {lr_now:.2e} | "
                  f"{elapsed:.0f}s")

        if (step + 1) % eval_every == 0 or step == max_steps - 1:
            print(f"\n--- Eval at step {step + 1} ---")
            eval_results = evaluate(model, eval_rng, partitions)
            for name, acc_val in eval_results.items():
                print(f"  {name}: {acc_val:.4f}")
            print("\nCapacity gates:")
            gate_passed = check_gates(eval_results)
            if gate_passed:
                print("  >>> ALL GATES PASSED <<<")
            print()

            train_acc = eval_results.get("train_lengths", 0.0)
            if train_acc > best_acc:
                best_acc = train_acc
                save_checkpoint(
                    model, optimizer, step + 1, eval_results,
                    ckpt_dir / "donor_best.pt",
                    rng_state=rng.get_state(),
                    precommit_hash=precommit_hash,
                    scaler=scaler,
                    best_acc=best_acc,
                    eval_rng_state=eval_rng.get_state(),
                )

        if (step + 1) % checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, step + 1, eval_results,
                ckpt_dir / f"donor_step_{step + 1:06d}.pt",
                rng_state=rng.get_state(),
                precommit_hash=precommit_hash,
                scaler=scaler,
                best_acc=best_acc,
                eval_rng_state=eval_rng.get_state(),
            )
            if DEVICE.type == "cuda" and COOLDOWN_SECONDS > 0:
                time.sleep(COOLDOWN_SECONDS)

        if gate_passed and not smoke:
            print(f"\nCapacity gates passed at step {step + 1}. "
                  f"Continuing to convergence...")

    final_eval = evaluate(model, eval_rng, partitions, n_batches=20)
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (20 batches per stratum)")
    print("=" * 60)
    for name, acc_val in final_eval.items():
        print(f"  {name}: {acc_val:.4f}")
    print("\nFinal capacity gates:")
    final_pass = check_gates(final_eval)

    save_checkpoint(
        model, optimizer, max_steps, final_eval,
        ckpt_dir / "donor_final.pt",
        rng_state=rng.get_state(),
        precommit_hash=precommit_hash,
        scaler=scaler,
        best_acc=best_acc,
        eval_rng_state=eval_rng.get_state(),
    )

    result = {
        "model": "RecurrentTransformerDonor",
        "params": n_params,
        "max_steps": max_steps,
        "final_eval": final_eval,
        "gates_passed": final_pass,
        "smoke": smoke,
        "device": str(DEVICE),
        "elapsed_seconds": time.time() - t0,
        "precommit_integrity_sha256": precommit_hash,
        "torch_seed": TORCH_SEED,
    }

    result_path = RESULTS_DIR / (
        "donor_smoke_result.json" if smoke else "donor_capacity_result.json"
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {result_path}")

    return model, final_eval, final_pass


if __name__ == "__main__":
    import sys

    smoke = "--smoke" in sys.argv
    train_donor(smoke=smoke)
