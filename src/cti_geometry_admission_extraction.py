"""Geometry Admission Test: raw trace and observable connection extraction.

Implements Stage A sections 6-9: depth clock, raw R extraction, observable
connection via VJP/perturbation balancing, and numerical gates.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cti_geometry_admission_automaton import (
    DEVELOPMENT_KEY_JSON,
    key_from_json,
    generate_anchors,
    partition_anchors_into_banks,
    collate_fn,
    simulate_automaton,
    NUM_OPS,
)


TEACHER_DEPTH_LAYERS = [0, 2, 4, 6, 8, 10, 12]
STUDENT_DEPTH_LAYERS = [0, 1, 2, 3, 4, 5, 6]
BALANCED_TOP_K = 8
RIDGE_FACTOR = 1e-3
BALANCED_RIDGE = 1e-6 / 64
NUM_PERTURBATIONS = 4

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_a"


def center_and_normalize(H: np.ndarray) -> np.ndarray:
    """Center rows and Frobenius-normalize. H: (n_anchors, dim)."""
    n = H.shape[0]
    C = np.eye(n) - np.ones((n, n)) / n
    CH = C @ H
    norm = np.linalg.norm(CH, "fro") + 1e-12
    return np.sqrt(n) * CH / norm


def extract_raw_trace(
    checkpoints: list[np.ndarray],
    depth_layers: list[int],
) -> dict:
    """Extract raw R_j, Omega_j from depth-clock-aligned checkpoints.

    checkpoints: list of (n_anchors, dim) arrays at each layer.
    depth_layers: which layers form the 7 depth clock ticks (indices into checkpoints).
    Returns dict with R_j, Omega_j, G_j, numerical diagnostics for each transition j=0..5.
    """
    ticks = [checkpoints[i] for i in depth_layers]
    n_transitions = len(ticks) - 1
    transitions = {}

    for j in range(n_transitions):
        X_j = center_and_normalize(ticks[j])
        X_j1 = center_and_normalize(ticks[j + 1])

        G_j = X_j @ X_j.T
        A_j = (X_j1 - X_j) @ X_j.T

        eigvals, eigvecs = np.linalg.eigh(G_j)
        n = X_j.shape[0]
        trace_G = np.trace(G_j)
        ridge = RIDGE_FACTOR * trace_G / (n - 1)

        ridged_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + ridge))

        C = np.eye(n) - np.ones((n, n)) / n
        G_inv_sqrt = C @ eigvecs @ ridged_inv_sqrt @ eigvecs.T @ C

        R_j = G_inv_sqrt @ A_j @ G_inv_sqrt
        Omega_j = 0.5 * (R_j - R_j.T)

        cond = (eigvals[-1] + ridge) / (eigvals[0] + ridge) if eigvals[0] + ridge > 0 else float("inf")
        rank = int(np.sum(eigvals > 1e-10 * eigvals[-1]))

        transitions[j] = {
            "R": R_j,
            "Omega": Omega_j,
            "G_trace": float(trace_G),
            "ridge": float(ridge),
            "condition_number": float(cond),
            "numerical_rank": rank,
            "eigval_min": float(eigvals[0]),
            "eigval_max": float(eigvals[-1]),
        }

    return transitions


def generate_perturbations(
    anchor: dict,
    key: np.ndarray,
    protocol_id: str = "OCF_GAT_ANCHORS_V1",
) -> list[dict]:
    """Generate NUM_PERTURBATIONS key-independent perturbations for an anchor."""
    perturbed = []
    for k in range(NUM_PERTURBATIONS):
        hash_input = f"{anchor['hash']}||{k}"
        h = hashlib.sha256(hash_input.encode()).hexdigest()
        seed_int = int(h[:16], 16)
        rng = np.random.default_rng(seed_int)

        ops = list(anchor["ops"])
        if len(ops) == 0:
            continue

        pos = int(rng.integers(0, len(ops)))
        original_op = ops[pos]
        other_ops = [o for o in range(NUM_OPS) if o != original_op]
        new_op = other_ops[int(rng.integers(0, len(other_ops)))]
        ops[pos] = new_op

        label = simulate_automaton(key, anchor["s0"], np.array(ops))
        input_ids = [anchor["s0"]] + [op + 12 for op in ops]
        perturbed.append({
            "input_ids": input_ids,
            "label": label,
            "s0": anchor["s0"],
            "ops": ops,
            "length": len(ops),
            "perturbation_index": k,
            "position": pos,
            "original_op": original_op,
            "new_op": new_op,
        })

    return perturbed


def extract_hidden_states(
    model,
    examples: list[dict],
    device: torch.device,
    depth_layers: list[int],
) -> list[np.ndarray]:
    """Run model on examples and extract hidden states at depth clock ticks.

    Returns list of (n_examples, dim) arrays, one per depth tick.
    """
    batch = collate_fn(examples)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids, attention_mask, return_hidden_states=True)

    hidden_states = out["hidden_states"]
    last_idx = attention_mask.sum(dim=1) - 1

    ticks = []
    for layer_idx in depth_layers:
        H = hidden_states[layer_idx]
        final_token = H[torch.arange(H.shape[0], device=device), last_idx]
        ticks.append(final_token.float().cpu().numpy())

    return ticks


def extract_observable_connection(
    model,
    bank_anchors: list[dict],
    bank_perturbations: list[list[dict]],
    device: torch.device,
    depth_layers: list[int],
) -> dict:
    """Extract observable R_obs for a bank using VJP observability + perturbation controllability.

    Args:
        model: trained model with return_hidden_states
        bank_anchors: list of 64 anchor dicts
        bank_perturbations: list of 64 lists, each containing NUM_PERTURBATIONS perturbation dicts
        device: torch device
        depth_layers: layer indices for depth clock

    Returns: dict with R_obs, U basis, diagnostics per transition.
    """
    n = len(bank_anchors)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    batch = collate_fn(bank_anchors)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    last_idx = attention_mask.sum(dim=1) - 1

    out = model(input_ids, attention_mask, return_hidden_states=True)
    logits = out["logits"]
    hidden_states = out["hidden_states"]

    sorted_indices = logits.argsort(dim=-1, descending=True)
    top_class = sorted_indices[:, 0]
    runner_class = sorted_indices[:, 1]

    tick_states_np = []
    for layer_idx in depth_layers:
        H = hidden_states[layer_idx]
        final_token = H[torch.arange(n, device=device), last_idx]
        tick_states_np.append(final_token.float().cpu().numpy())

    vjp_grads = []
    for layer_idx in depth_layers:
        for p in model.parameters():
            p.requires_grad_(True)

        out_g = model(input_ids, attention_mask, return_hidden_states=True)
        logits_g = out_g["logits"]
        margins_g = logits_g[torch.arange(n, device=device), top_class] - \
                    logits_g[torch.arange(n, device=device), runner_class]

        target_state = out_g["hidden_states"][layer_idx]
        target_final = target_state[torch.arange(n, device=device), last_idx]
        target_final.retain_grad()
        margins_g.sum().backward(retain_graph=False)
        grad = target_final.grad.float().detach().cpu().numpy()
        vjp_grads.append(grad)

        for p in model.parameters():
            p.requires_grad_(False)
        model.zero_grad()

    pert_ticks_by_k = []
    for k in range(NUM_PERTURBATIONS):
        pert_batch_k = [bank_perturbations[i][k] for i in range(n)]
        ticks_k = extract_hidden_states(model, pert_batch_k, device, depth_layers)
        pert_ticks_by_k.append(ticks_k)

    transitions = {}
    for j in range(len(depth_layers) - 1):
        X_j = center_and_normalize(tick_states_np[j])
        X_j1 = center_and_normalize(tick_states_np[j + 1])

        G_j = X_j @ X_j.T
        A_j = (X_j1 - X_j) @ X_j.T
        eigvals, eigvecs = np.linalg.eigh(G_j)
        trace_G = np.trace(G_j)
        ridge = RIDGE_FACTOR * trace_G / (n - 1)
        ridged_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + ridge))
        C_mat = np.eye(n) - np.ones((n, n)) / n
        G_inv_sqrt = C_mat @ eigvecs @ ridged_inv_sqrt @ eigvecs.T @ C_mat
        R_j = G_inv_sqrt @ A_j @ G_inv_sqrt

        J_j = vjp_grads[j]
        J_j_centered = center_and_normalize(J_j)
        S_j = J_j_centered @ X_j.T
        W_o = S_j.T @ S_j
        W_o = W_o / (np.trace(W_o) + 1e-12)

        D_sum = np.zeros((n, n))
        for k in range(NUM_PERTURBATIONS):
            X_j_pert = center_and_normalize(pert_ticks_by_k[k][j])
            D_k = (X_j_pert - X_j) @ X_j.T
            D_sum += D_k.T @ D_k

        W_c = D_sum / (np.trace(D_sum) + 1e-12)

        W_c += BALANCED_RIDGE * np.eye(n)
        W_o += BALANCED_RIDGE * np.eye(n)

        eigvals_wc, eigvecs_wc = np.linalg.eigh(W_c)
        eigvals_wc = np.maximum(eigvals_wc, 1e-10)
        W_c_sqrt = eigvecs_wc @ np.diag(np.sqrt(eigvals_wc)) @ eigvecs_wc.T

        M_j = W_c_sqrt @ W_o @ W_c_sqrt
        M_j = 0.5 * (M_j + M_j.T)

        eig_M, vec_M = np.linalg.eigh(M_j)
        top_k_idx = np.argsort(eig_M)[-BALANCED_TOP_K:][::-1]
        V_top = vec_M[:, top_k_idx]

        U_raw = W_c_sqrt @ V_top
        U, _ = np.linalg.qr(U_raw)
        U = U[:, :BALANCED_TOP_K]

        for col in range(U.shape[1]):
            max_idx = np.argmax(np.abs(U[:, col]))
            if U[max_idx, col] < 0:
                U[:, col] *= -1

        R_obs = U.T @ R_j @ U

        orth_error = np.linalg.norm(U.T @ U - np.eye(BALANCED_TOP_K), "fro")
        rank_Wc = int(np.sum(np.linalg.eigvalsh(W_c) > 1e-10))
        rank_Wo = int(np.sum(np.linalg.eigvalsh(W_o) > 1e-10))

        transitions[j] = {
            "R_obs": R_obs,
            "U_basis": U,
            "R_full": R_j,
            "orthogonality_error": float(orth_error),
            "W_c_rank": rank_Wc,
            "W_o_rank": rank_Wo,
            "W_c_trace": float(np.trace(W_c)),
            "W_o_trace": float(np.trace(W_o)),
        }

    return transitions


def check_numerical_gates(raw_transitions: dict, obs_transitions: dict = None) -> dict:
    """Check all numerical gates from Stage A section 9."""
    gates = {}

    for j, t in raw_transitions.items():
        R = t["R"]
        gate = {
            "finite": bool(np.all(np.isfinite(R))),
            "numerical_rank_ge_48": t["numerical_rank"] >= 48,
            "condition_le_1e6": t["condition_number"] <= 1e6,
        }
        gates[f"raw_transition_{j}"] = gate

    if obs_transitions:
        for j, t in obs_transitions.items():
            gate = {
                "finite": bool(np.all(np.isfinite(t["R_obs"]))),
                "orthogonality_le_1e5": t["orthogonality_error"] <= 1e-5,
                "W_c_rank_ge_8": t["W_c_rank"] >= 8,
                "W_o_rank_ge_8": t["W_o_rank"] >= 8,
            }
            gates[f"obs_transition_{j}"] = gate

    all_pass = all(
        all(v for v in g.values()) for g in gates.values()
    )
    gates["all_pass"] = all_pass
    return gates


def serialize_traces(
    raw_transitions: dict,
    obs_transitions: dict,
    bank_idx: int,
    anchor_hashes: list[str],
    output_dir: Path,
) -> tuple[str, str]:
    """Serialize raw and observable traces to files. Returns SHA256 hashes."""
    bank_dir = output_dir / f"bank_{bank_idx:03d}"
    bank_dir.mkdir(parents=True, exist_ok=True)

    raw_data = {
        "bank_index": bank_idx,
        "anchor_hashes": anchor_hashes,
        "transitions": {},
    }
    for j, t in raw_transitions.items():
        raw_data["transitions"][str(j)] = {
            "R": t["R"].astype(np.float32).tolist(),
            "Omega": t["Omega"].astype(np.float32).tolist(),
            "G_trace": t["G_trace"],
            "ridge": t["ridge"],
            "condition_number": t["condition_number"],
            "numerical_rank": t["numerical_rank"],
        }

    raw_json = json.dumps(raw_data, sort_keys=True, separators=(",", ":"))
    raw_hash = hashlib.sha256(raw_json.encode()).hexdigest()
    with open(bank_dir / "raw_trace.json", "w") as f:
        f.write(raw_json)

    obs_data = {
        "bank_index": bank_idx,
        "anchor_hashes": anchor_hashes,
        "transitions": {},
    }
    for j, t in obs_transitions.items():
        obs_data["transitions"][str(j)] = {
            "R_obs": t["R_obs"].astype(np.float32).tolist(),
            "U_basis": t["U_basis"].astype(np.float32).tolist(),
            "orthogonality_error": t["orthogonality_error"],
            "W_c_rank": t["W_c_rank"],
            "W_o_rank": t["W_o_rank"],
        }

    obs_json = json.dumps(obs_data, sort_keys=True, separators=(",", ":"))
    obs_hash = hashlib.sha256(obs_json.encode()).hexdigest()
    with open(bank_dir / "observable_trace.json", "w") as f:
        f.write(obs_json)

    return raw_hash, obs_hash


if __name__ == "__main__":
    print("Extraction module loaded. Key functions:")
    print("  center_and_normalize(H) -> X")
    print("  extract_raw_trace(checkpoints, depth_layers) -> transitions")
    print("  extract_observable_connection(model, anchors, perturbations, device, layers) -> obs")
    print("  check_numerical_gates(raw, obs) -> gates")
    print("  serialize_traces(raw, obs, bank_idx, hashes, dir) -> (raw_sha, obs_sha)")

    H = np.random.randn(64, 384).astype(np.float32)
    X = center_and_normalize(H)
    print(f"\nCenter+normalize test: input {H.shape} -> output {X.shape}")
    print(f"  Mean row norm: {np.mean(np.linalg.norm(X, axis=1)):.6f}")
    print(f"  Column mean: {np.abs(X.mean(axis=0)).max():.2e}")
    print(f"  Frobenius: {np.linalg.norm(X, 'fro'):.4f} (expected: {np.sqrt(64):.4f})")

    checkpoints = [np.random.randn(64, 384).astype(np.float32) for _ in range(13)]
    raw = extract_raw_trace(checkpoints, TEACHER_DEPTH_LAYERS)
    print(f"\nRaw trace extraction: {len(raw)} transitions")
    for j, t in raw.items():
        print(f"  Transition {j}: R shape={t['R'].shape}, rank={t['numerical_rank']}, "
              f"cond={t['condition_number']:.1f}")
