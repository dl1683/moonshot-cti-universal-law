"""Geometry Admission Test: restricted-supervision student training (installer).

Trains a student using only 64 calibration examples + teacher artifact.
No teacher weights, hidden states, logits, or withheld labels.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from cti_geometry_admission_automaton import (
    key_from_json,
    generate_calibration_set,
    generate_withheld_eval_set,
    generate_anchors,
    partition_anchors_into_banks,
    generate_bank_order_permutation,
    collate_fn,
    hash_eval_set,
)
from cti_geometry_admission_models import (
    create_transformer_student,
    create_gru_student,
    count_parameters,
)
from cti_geometry_admission_geometry import (
    compute_student_R_sequence,
    compute_student_G_sequence,
    loss_raw_R,
    loss_observable_R,
    loss_static_G,
    loss_smoothness,
)

STUDENT_DEPTH_LAYERS = [0, 1, 2, 3, 4, 5, 6]
MAX_STEPS = 5000
BATCH_SIZE_TASK = 64
WARMUP_STEPS = 250
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.95)
EPS = 1e-8
COSINE_MIN_RATIO = 0.1
EVAL_INTERVAL = 250

ARMS = [
    "raw_correct",
    "raw_wrong",
    "raw_haar",
    "obs_correct",
    "obs_wrong",
    "obs_haar",
    "static_g",
    "smoothness",
    "no_auxiliary",
]


def cosine_lr(step, warmup, total, peak_lr, min_ratio):
    if step < warmup:
        return peak_lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def load_teacher_artifacts(artifact_dir: Path, n_banks: int = 32):
    """Load teacher R targets, G targets, and observable artifacts."""
    raw_targets = {}
    obs_targets = {}
    g_targets = {}

    for bank_idx in range(n_banks):
        bank_dir = artifact_dir / f"bank_{bank_idx:03d}"

        raw_path = bank_dir / "raw_trace.json"
        if raw_path.exists():
            with open(raw_path) as f:
                raw_data = json.loads(f.read())
            raw_targets[bank_idx] = {}
            for j_str, t in raw_data["transitions"].items():
                raw_targets[bank_idx][int(j_str)] = np.array(t["R"], dtype=np.float32)

        obs_path = bank_dir / "observable_trace.json"
        if obs_path.exists():
            with open(obs_path) as f:
                obs_data = json.loads(f.read())
            obs_targets[bank_idx] = {}
            for j_str, t in obs_data["transitions"].items():
                obs_targets[bank_idx][int(j_str)] = {
                    "R_obs": np.array(t["R_obs"], dtype=np.float32),
                    "U_basis": np.array(t["U_basis"], dtype=np.float32),
                }

    return raw_targets, obs_targets, g_targets


def calibrate_coefficient(
    model,
    calibration_examples: list[dict],
    anchor_banks: list[list[dict]],
    arm: str,
    teacher_artifacts: dict,
    device: torch.device,
) -> float:
    """One-shot gradient-ratio coefficient calibration per R4 spec.

    Returns lambda_a = ||grad_task||_2 / median_b(||grad_aux(b)||_2).
    """
    model.train()
    model.zero_grad()

    cal_batch = collate_fn(calibration_examples)
    cal_ids = cal_batch["input_ids"].to(device)
    cal_mask = cal_batch["attention_mask"].to(device)
    cal_labels = cal_batch["labels"].to(device)

    with autocast(dtype=torch.bfloat16):
        out = model(cal_ids, cal_mask)
    task_loss = F.cross_entropy(out["logits"].float(), cal_labels)

    trunk_params = [p for name, p in model.named_parameters()
                    if "classifier" not in name and p.requires_grad]

    task_grads = torch.autograd.grad(
        task_loss, trunk_params, retain_graph=False, allow_unused=True,
    )
    g_task = torch.sqrt(sum(
        (g ** 2).sum() for g in task_grads if g is not None
    )).item()
    if not np.isfinite(g_task) or g_task == 0:
        return 1.0

    aux_norms = []
    for bank_idx, bank in enumerate(anchor_banks):
        model.zero_grad()
        aux_loss = compute_auxiliary_loss(
            model, bank, arm, teacher_artifacts, bank_idx, device,
        )
        if aux_loss is None or aux_loss.item() == 0:
            continue

        aux_grads = torch.autograd.grad(
            aux_loss, trunk_params, retain_graph=False, allow_unused=True,
        )
        g_aux = torch.sqrt(sum(
            (g ** 2).sum() for g in aux_grads if g is not None
        )).item()
        if g_aux > 0 and np.isfinite(g_aux):
            aux_norms.append(g_aux)

    if not aux_norms:
        return 1.0

    median_g_aux = float(np.median(aux_norms))
    if median_g_aux == 0:
        return 1.0

    lam = g_task / median_g_aux
    return lam


def compute_auxiliary_loss(
    model,
    bank_anchors: list[dict],
    arm: str,
    teacher_artifacts: dict,
    bank_idx: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Compute auxiliary loss for a given arm and bank."""
    if arm == "no_auxiliary":
        return None

    batch = collate_fn(bank_anchors)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    out = model(input_ids, attention_mask, return_hidden_states=True)
    hidden_states = out["hidden_states"]

    if arm in ("raw_correct", "raw_wrong", "raw_haar"):
        student_R = compute_student_R_sequence(
            hidden_states, STUDENT_DEPTH_LAYERS, attention_mask,
        )
        target_key = {"raw_correct": "raw", "raw_wrong": "raw_wrong", "raw_haar": "raw_haar"}[arm]
        targets = teacher_artifacts.get(target_key, {}).get(bank_idx, {})
        if not targets:
            return None
        teacher_R = [torch.tensor(targets[j], device=device, dtype=torch.float32)
                     for j in range(6)]
        return loss_raw_R(student_R, teacher_R)

    elif arm in ("obs_correct", "obs_wrong", "obs_haar"):
        student_R = compute_student_R_sequence(
            hidden_states, STUDENT_DEPTH_LAYERS, attention_mask,
        )
        target_key = {"obs_correct": "obs", "obs_wrong": "obs_wrong", "obs_haar": "obs_haar"}[arm]
        targets = teacher_artifacts.get(target_key, {}).get(bank_idx, {})
        if not targets:
            return None
        teacher_R_obs = [torch.tensor(targets[j]["R_obs"], device=device, dtype=torch.float32)
                        for j in range(6)]
        U_basis = [torch.tensor(targets[j]["U_basis"], device=device, dtype=torch.float32)
                   for j in range(6)]
        return loss_observable_R(student_R, teacher_R_obs, U_basis)

    elif arm == "static_g":
        student_G = compute_student_G_sequence(
            hidden_states, STUDENT_DEPTH_LAYERS, attention_mask,
        )
        targets = teacher_artifacts.get("static_g", {}).get(bank_idx, {})
        if not targets:
            return None
        teacher_G = [torch.tensor(targets[j], device=device, dtype=torch.float32)
                     for j in range(7)]
        return loss_static_G(student_G, teacher_G)

    elif arm == "smoothness":
        return loss_smoothness(hidden_states, STUDENT_DEPTH_LAYERS, attention_mask)

    return None


@torch.no_grad()
def evaluate_withheld(model, withheld_examples, device, batch_size=512):
    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(withheld_examples), batch_size):
        batch_ex = withheld_examples[start:start + batch_size]
        batch = collate_fn(batch_ex)
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


@torch.no_grad()
def centroid_probe(model, calibration_examples, probe_examples, device):
    """Frozen centroid probe: L2-normalize, average per class, cosine predict."""
    model.eval()

    cal_batch = collate_fn(calibration_examples)
    cal_ids = cal_batch["input_ids"].to(device)
    cal_mask = cal_batch["attention_mask"].to(device)
    with autocast(dtype=torch.bfloat16):
        out = model(cal_ids, cal_mask, return_hidden_states=True)
    last_idx = cal_mask.sum(dim=1) - 1
    H = out["hidden_states"][-1]
    cal_hidden = H[torch.arange(len(calibration_examples), device=device), last_idx].float()
    cal_hidden = F.normalize(cal_hidden, dim=-1)

    cal_labels = torch.tensor([ex["label"] for ex in calibration_examples], device=device)
    centroids = []
    for y in range(12):
        mask = cal_labels == y
        if mask.sum() > 0:
            centroid = cal_hidden[mask].mean(dim=0)
            centroids.append(F.normalize(centroid, dim=0))
        else:
            centroids.append(torch.zeros(cal_hidden.shape[1], device=device))
    centroids = torch.stack(centroids)

    probe_batch = collate_fn(probe_examples)
    probe_ids = probe_batch["input_ids"].to(device)
    probe_mask = probe_batch["attention_mask"].to(device)
    with autocast(dtype=torch.bfloat16):
        out = model(probe_ids, probe_mask, return_hidden_states=True)
    last_idx = probe_mask.sum(dim=1) - 1
    H = out["hidden_states"][-1]
    probe_hidden = H[torch.arange(len(probe_examples), device=device), last_idx].float()
    probe_hidden = F.normalize(probe_hidden, dim=-1)

    sims = probe_hidden @ centroids.T
    preds = sims.argmax(dim=-1)
    probe_labels = torch.tensor([ex["label"] for ex in probe_examples], device=device)
    correct = (preds == probe_labels).sum().item()
    model.train()
    return correct, len(probe_examples)


@torch.no_grad()
def evaluate_direct_edge_logits(model, direct_edges, device):
    """Return logits for all 48 direct edges as (48, 12) numpy array."""
    model.eval()
    batch = collate_fn(direct_edges)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with autocast(dtype=torch.bfloat16):
        out = model(input_ids, attention_mask)
    logits = out["logits"].float().cpu().numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite logits detected in direct edge evaluation")
    assert logits.shape == (48, 12), f"Expected (48, 12), got {logits.shape}"
    model.train()
    return logits


def train_installer_run(
    run_config: dict,
    teacher_artifacts: dict,
    calibration_examples: list[dict],
    withheld_examples: list[dict],
    direct_probes: list[dict],
    anchor_banks: list[list[dict]],
    bank_order: list[int],
    output_dir: Path,
    device: torch.device,
    seal_probes: bool = False,
):
    """Train one installer run (one arm, one key, one seed).

    seal_probes: if True, probe results go to sealed_probe_log.jsonl only,
    excluded from training_log.jsonl and summary.json. For CM-CKS sealed runs.
    """
    import hashlib as _hashlib
    name = run_config["name"]
    arm = run_config["arm"]
    seed = run_config["seed"]
    peak_lr = run_config.get("lr", 5e-4)
    arch = run_config.get("arch", "transformer")
    coeff = run_config.get("coefficient", 1.0)
    init_checkpoint = run_config.get("init_checkpoint")

    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        if summary.get("status") == "complete":
            print(f"[{name}] Already complete, skipping.")
            return summary

    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if arch == "gru":
            model = create_gru_student()
        else:
            model = create_transformer_student()
        state_dict = torch.load(init_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        init_hash = _hashlib.sha256(init_path.read_bytes()).hexdigest()
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if arch == "gru":
            model = create_gru_student()
        else:
            model = create_transformer_student()
        init_hash = "dynamic"

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler()

    cal_batch = collate_fn(calibration_examples)

    checkpoint_path = run_dir / "checkpoint.pt"
    log_path = run_dir / "training_log.jsonl"

    if checkpoint_path.exists():
        print(f"[{name}] Existing checkpoint found -- removing (frozen contract: always restart from step 0).")
        checkpoint_path.unlink()

    log_file = None
    sealed_log_file = None
    try:
        log_file = open(log_path, "w", encoding="utf-8")
        if seal_probes:
            sealed_log_path = run_dir / "sealed_probe_log.jsonl"
            sealed_log_file = open(sealed_log_path, "w", encoding="utf-8")
        model.train()
        t0 = time.time()
        eval_history = []

        cal_ids = cal_batch["input_ids"].to(device)
        cal_mask = cal_batch["attention_mask"].to(device)
        cal_labels = cal_batch["labels"].to(device)

        for step in range(MAX_STEPS):
            lr = cosine_lr(step, WARMUP_STEPS, MAX_STEPS, peak_lr, COSINE_MIN_RATIO)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                out = model(cal_ids, cal_mask)
            task_loss = F.cross_entropy(out["logits"].float(), cal_labels)

            if not torch.isfinite(task_loss):
                raise RuntimeError(f"[{name}] Non-finite task loss at step {step}")

            bank_idx = bank_order[step]
            bank = anchor_banks[bank_idx]

            if arm != "no_auxiliary":
                with torch.cuda.amp.autocast(enabled=False):
                    aux_loss = compute_auxiliary_loss(
                        model, bank, arm, teacher_artifacts, bank_idx, device,
                    )
                if aux_loss is None:
                    raise RuntimeError(
                        f"[{name}] Missing auxiliary target for arm={arm}, bank={bank_idx}"
                    )
                if not torch.isfinite(aux_loss):
                    raise RuntimeError(f"[{name}] Non-finite aux loss at step {step}")
                total_loss = task_loss + coeff * aux_loss
            else:
                total_loss = task_loss
                aux_loss = torch.tensor(0.0)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"[{name}] Non-finite gradient norm at step {step}")
            scaler.step(optimizer)
            scaler.update()

            if step % 100 == 0:
                with torch.no_grad():
                    preds = out["logits"].argmax(dim=-1)
                    cal_acc = (preds == cal_labels).float().mean().item()
                print(f"[{name}] step {step}/{MAX_STEPS} task={task_loss.item():.4f} "
                      f"aux={aux_loss.item():.4f} cal_acc={cal_acc:.4f} lr={lr:.6f}")

            if (step + 1) % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
                with torch.no_grad():
                    preds = out["logits"].argmax(dim=-1)
                    cal_acc = (preds == cal_labels).float().mean().item()

                eval_result = {
                    "step": step + 1,
                    "task_loss": task_loss.item(),
                    "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
                    "cal_accuracy": cal_acc,
                    "lr": lr,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                    "wall_time": time.time() - t0,
                }
                print(f"[{name}] EVAL step={step+1}: cal_acc={cal_acc:.4f}")

                eval_history.append(eval_result)
                log_file.write(json.dumps(eval_result) + "\n")
                log_file.flush()

                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step + 1,
                }, checkpoint_path)

    finally:
        if log_file is not None:
            log_file.close()
        if sealed_log_file is not None:
            sealed_log_file.close()

    wall_time = time.time() - t0

    summary = {
        "name": name,
        "arm": arm,
        "arch": arch,
        "seed": seed,
        "lr": peak_lr,
        "coefficient": coeff,
        "params": count_parameters(model),
        "max_steps": MAX_STEPS,
        "init_hash": init_hash,
        "sealed": seal_probes,
        "wall_seconds": wall_time,
        "eval_history": eval_history,
        "status": "complete",
    }

    torch.save(model.state_dict(), run_dir / "model_final.pt")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    print("Installer module loaded.")
    print(f"Arms: {ARMS}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Calibration batch size: {BATCH_SIZE_TASK}")
    print(f"Eval interval: {EVAL_INTERVAL}")
