"""Geometry Admission Test: differentiable geometry and loss functions.

Provides differentiable R computation, all five control losses, and
Haar-matched artifact construction. All matrix operations in float32
with autocast disabled per the Stage B/C specification.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn


RIDGE_FACTOR = 1e-3
BALANCED_TOP_K = 8


def center_and_normalize_torch(H: torch.Tensor) -> torch.Tensor:
    """Center rows and Frobenius-normalize. H: (n, dim). float32."""
    n = H.shape[0]
    mean = H.mean(dim=0, keepdim=True)
    CH = H - mean
    norm = torch.linalg.norm(CH, ord="fro").clamp(min=1e-12)
    return (n ** 0.5) * CH / norm


def compute_R_differentiable(
    X_j: torch.Tensor,
    X_j1: torch.Tensor,
) -> torch.Tensor:
    """Compute normalized generator R_j from centered/normalized X_j, X_{j+1}.

    All operations in float32. Gradients flow through the student's X.
    Returns R_j: (n, n).
    """
    n = X_j.shape[0]
    G_j = X_j @ X_j.T
    A_j = (X_j1 - X_j) @ X_j.T

    G_sym = 0.5 * (G_j + G_j.T)
    eigvals, eigvecs = torch.linalg.eigh(G_sym)

    trace_G = eigvals.sum()
    ridge = RIDGE_FACTOR * trace_G / (n - 1)

    eigvals_clamped = eigvals.clamp(min=0)
    ridged_inv_sqrt = 1.0 / torch.sqrt(eigvals_clamped + ridge)

    ones = torch.ones(n, 1, device=X_j.device, dtype=X_j.dtype)
    C = torch.eye(n, device=X_j.device, dtype=X_j.dtype) - ones @ ones.T / n

    G_inv_sqrt = C @ eigvecs @ torch.diag(ridged_inv_sqrt) @ eigvecs.T @ C
    R_j = G_inv_sqrt @ A_j @ G_inv_sqrt

    return R_j


def compute_student_R_sequence(
    hidden_states: list[torch.Tensor],
    depth_layers: list[int],
    attention_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Compute R_j for all transitions from student hidden states.

    hidden_states: list of (batch, seq_len, dim) tensors.
    Returns list of (n, n) R_j tensors for each transition.
    """
    n = attention_mask.shape[0]
    last_idx = attention_mask.sum(dim=1) - 1
    device = hidden_states[0].device

    ticks = []
    for layer_idx in depth_layers:
        H = hidden_states[layer_idx]
        final_token = H[torch.arange(n, device=device), last_idx]
        ticks.append(final_token.float())

    R_sequence = []
    for j in range(len(ticks) - 1):
        X_j = center_and_normalize_torch(ticks[j])
        X_j1 = center_and_normalize_torch(ticks[j + 1])
        R_j = compute_R_differentiable(X_j, X_j1)
        R_sequence.append(R_j)

    return R_sequence


def compute_student_G_sequence(
    hidden_states: list[torch.Tensor],
    depth_layers: list[int],
    attention_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Compute centered normalized Gram G_j at all checkpoints."""
    n = attention_mask.shape[0]
    last_idx = attention_mask.sum(dim=1) - 1
    device = hidden_states[0].device

    G_sequence = []
    for layer_idx in depth_layers:
        H = hidden_states[layer_idx]
        final_token = H[torch.arange(n, device=device), last_idx].float()
        X = center_and_normalize_torch(final_token)
        G = X @ X.T
        G_sequence.append(G)

    return G_sequence


def loss_raw_R(
    student_R_seq: list[torch.Tensor],
    teacher_R_seq: list[torch.Tensor],
) -> torch.Tensor:
    """Raw R MSE loss: (1/6) sum_j ||R^S_j - R^T_j||_F^2 / 64^2."""
    n_transitions = len(student_R_seq)
    n = student_R_seq[0].shape[0]
    total = torch.tensor(0.0, device=student_R_seq[0].device)
    for j in range(n_transitions):
        diff = student_R_seq[j] - teacher_R_seq[j]
        total = total + (diff ** 2).sum() / (n * n)
    return total / n_transitions


def loss_observable_R(
    student_R_seq: list[torch.Tensor],
    teacher_R_obs_seq: list[torch.Tensor],
    U_basis_seq: list[torch.Tensor],
) -> torch.Tensor:
    """Observable R MSE loss: (1/6) sum_j ||U^T R^S_j U - R^T_obs_j||_F^2 / 8^2."""
    n_transitions = len(student_R_seq)
    r = U_basis_seq[0].shape[1]
    total = torch.tensor(0.0, device=student_R_seq[0].device)
    for j in range(n_transitions):
        R_s_obs = U_basis_seq[j].T @ student_R_seq[j] @ U_basis_seq[j]
        diff = R_s_obs - teacher_R_obs_seq[j]
        total = total + (diff ** 2).sum() / (r * r)
    return total / n_transitions


def loss_static_G(
    student_G_seq: list[torch.Tensor],
    teacher_G_seq: list[torch.Tensor],
) -> torch.Tensor:
    """Static G MSE loss: (1/7) sum_j ||G^S_j - G^T_j||_F^2 / 64^2."""
    n_checkpoints = len(student_G_seq)
    n = student_G_seq[0].shape[0]
    total = torch.tensor(0.0, device=student_G_seq[0].device)
    for j in range(n_checkpoints):
        diff = student_G_seq[j] - teacher_G_seq[j]
        total = total + (diff ** 2).sum() / (n * n)
    return total / n_checkpoints


def loss_smoothness(
    hidden_states: list[torch.Tensor],
    depth_layers: list[int],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Generic smoothness loss: (1/6) sum_j ||X^S_{j+1} - X^S_j||_F^2 / 64."""
    n = attention_mask.shape[0]
    last_idx = attention_mask.sum(dim=1) - 1
    device = hidden_states[0].device

    ticks = []
    for layer_idx in depth_layers:
        H = hidden_states[layer_idx]
        final_token = H[torch.arange(n, device=device), last_idx].float()
        X = center_and_normalize_torch(final_token)
        ticks.append(X)

    n_transitions = len(ticks) - 1
    total = torch.tensor(0.0, device=device)
    for j in range(n_transitions):
        diff = ticks[j + 1] - ticks[j]
        total = total + (diff ** 2).sum() / n
    return total / n_transitions


def generate_haar_rotation_raw(
    n: int,
    bank_idx: int,
    seed_prefix: str = "GAT_HAAR_RAW_V1",
) -> np.ndarray:
    """Generate Haar-random orthogonal rotation in centered subspace.

    Returns Q: (n, n) orthogonal matrix that acts as identity on the mean
    direction and Haar-random in the (n-1)-dim centered subspace.
    """
    seed_str = f"{seed_prefix}||{bank_idx}"
    seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed_int)

    Z = rng.standard_normal((n - 1, n - 1))
    Q_c, R_qr = np.linalg.qr(Z)
    diag_sign = np.sign(np.diag(R_qr))
    diag_sign[diag_sign == 0] = 1.0
    Q_c = Q_c * diag_sign

    e = np.ones((n, 1)) / np.sqrt(n)
    E = np.linalg.svd(np.eye(n) - e @ e.T, full_matrices=False)[0][:, :n - 1]

    Q = e @ e.T + E @ Q_c @ E.T
    return Q.astype(np.float32)


def generate_haar_rotation_obs(
    r: int,
    bank_idx: int,
    seed_prefix: str = "GAT_HAAR_OBS_V1",
) -> np.ndarray:
    """Generate Haar-random O(r) rotation for observable candidate."""
    seed_str = f"{seed_prefix}||{bank_idx}"
    seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed_int)

    Z = rng.standard_normal((r, r))
    Q, R_qr = np.linalg.qr(Z)
    diag_sign = np.sign(np.diag(R_qr))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign
    return Q.astype(np.float32)


def apply_haar_to_raw_targets(
    teacher_R_seq: list[np.ndarray],
    Q: np.ndarray,
) -> list[np.ndarray]:
    """Apply Haar rotation to raw teacher R targets: Q R^T Q^T."""
    return [Q @ R @ Q.T for R in teacher_R_seq]


def apply_haar_to_obs_targets(
    teacher_R_obs_seq: list[np.ndarray],
    Q_obs: np.ndarray,
) -> list[np.ndarray]:
    """Apply Haar rotation to observable teacher R targets: Q R^T_obs Q^T."""
    return [Q_obs @ R @ Q_obs.T for R in teacher_R_obs_seq]


if __name__ == "__main__":
    import sys
    sys.stdout = open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False)

    n, d = 64, 384
    H1 = torch.randn(n, d)
    H2 = torch.randn(n, d)

    X1 = center_and_normalize_torch(H1)
    X2 = center_and_normalize_torch(H2)
    R = compute_R_differentiable(X1, X2)
    print(f"R shape: {R.shape}")
    print(f"R Frobenius: {torch.linalg.norm(R, 'fro'):.4f}")

    R_t = [torch.randn(n, n) for _ in range(6)]
    R_s = [torch.randn(n, n, requires_grad=True) for _ in range(6)]
    loss = loss_raw_R(R_s, R_t)
    print(f"Raw R loss: {loss.item():.6f}")
    loss.backward()
    print(f"Gradient flows: {R_s[0].grad is not None}")

    Q_raw = generate_haar_rotation_raw(n, bank_idx=0)
    print(f"\nHaar Q: {Q_raw.shape}")
    print(f"  Orthogonality error: {np.linalg.norm(Q_raw @ Q_raw.T - np.eye(n)):.2e}")
    print(f"  Preserves mean: {abs(Q_raw @ np.ones(n) / np.sqrt(n) - np.ones(n) / np.sqrt(n)).max():.2e}")

    Q_obs = generate_haar_rotation_obs(8, bank_idx=0)
    print(f"Haar Q_obs: {Q_obs.shape}")
    print(f"  Orthogonality error: {np.linalg.norm(Q_obs @ Q_obs.T - np.eye(8)):.2e}")
