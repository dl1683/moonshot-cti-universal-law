"""Geometry Admission Test: statistical analysis for Stage B/C.

Stage B selection metric, Stage C bootstrap and sign-flip tests,
PASS/FAIL/VOID verdict calculation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def stage_b_selection(
    results: dict,
) -> dict:
    """Stage B candidate selection from 18-run screen.

    results: dict with structure:
        results[key_idx][arm_name] = withheld_accuracy (float)
        key_idx in {0, 1}, arm_name in ARMS

    Returns selection decision with winner, eligibility, margins.
    """
    candidates = ["raw_correct", "obs_correct"]
    controls_raw = ["no_auxiliary", "smoothness", "static_g", "raw_wrong", "raw_haar"]
    controls_obs = ["no_auxiliary", "smoothness", "static_g", "obs_wrong", "obs_haar"]

    selection = {}
    for cand, controls in [("raw_correct", controls_raw), ("obs_correct", controls_obs)]:
        cand_label = "raw" if "raw" in cand else "obs"

        margins = []
        for k in range(2):
            cand_acc = results[k][cand]
            best_control = max(results[k][c] for c in controls)
            margin = cand_acc - best_control
            margins.append(margin)

        eligible = margins[0] > 0 and margins[1] > 0 and np.mean(margins) >= 0.10
        mean_acc = np.mean([results[k][cand] for k in range(2)])

        selection[cand_label] = {
            "withheld_acc": {k: results[k][cand] for k in range(2)},
            "margins": margins,
            "mean_margin": float(np.mean(margins)),
            "mean_acc": float(mean_acc),
            "eligible": bool(eligible),
        }

    eligible_candidates = [c for c, s in selection.items() if s["eligible"]]

    if not eligible_candidates:
        return {
            "winner": None,
            "verdict": "FAIL",
            "reason": "No eligible candidate",
            "selection": selection,
        }

    if len(eligible_candidates) == 1:
        winner = eligible_candidates[0]
    else:
        if selection["raw"]["mean_acc"] > selection["obs"]["mean_acc"]:
            winner = "raw"
        elif selection["obs"]["mean_acc"] > selection["raw"]["mean_acc"]:
            winner = "obs"
        else:
            winner = "obs"

    return {
        "winner": winner,
        "verdict": "PASS",
        "selection": selection,
    }


def stage_c_primary_statistic(
    results: dict,
    winner: str,
) -> dict:
    """Compute Delta_min: weakest controlled advantage across 8 keys, 3 seeds, 5 controls.

    results: dict with structure:
        results[key_idx][seed_idx][arm_name] = withheld_accuracy
        key_idx in 0..7, seed_idx in 0..2

    Returns dict with Delta_c for each control and Delta_min.
    """
    if winner == "raw":
        winner_arm = "raw_correct"
        controls = ["no_auxiliary", "smoothness", "static_g", "raw_wrong", "raw_haar"]
    else:
        winner_arm = "obs_correct"
        controls = ["no_auxiliary", "smoothness", "static_g", "obs_wrong", "obs_haar"]

    deltas = {}
    for c in controls:
        d_bar_k = []
        for k in range(8):
            d_seeds = []
            for s in range(3):
                d = results[k][s][winner_arm] - results[k][s][c]
                d_seeds.append(d)
            d_bar_k.append(np.mean(d_seeds))
        delta_c = np.mean(d_bar_k)
        deltas[c] = {
            "delta": float(delta_c),
            "per_key": [float(d) for d in d_bar_k],
        }

    delta_min = min(d["delta"] for d in deltas.values())
    delta_min_control = min(deltas, key=lambda c: deltas[c]["delta"])

    return {
        "delta_min": float(delta_min),
        "delta_min_control": delta_min_control,
        "deltas": deltas,
    }


def key_cluster_bootstrap(
    results: dict,
    winner: str,
    n_replicates: int = 100_000,
) -> dict:
    """Key-cluster bootstrap for Delta_min 95% CI.

    Resample 8 key indices with replacement, keep seeds/arms paired within key.
    """
    seed_hex = hashlib.sha256("GAT_STAGE_C_KEY_BOOTSTRAP_V1".encode()).hexdigest()
    seed_int = int(seed_hex[:16], 16)
    rng = np.random.default_rng(seed_int)

    if winner == "raw":
        winner_arm = "raw_correct"
        controls = ["no_auxiliary", "smoothness", "static_g", "raw_wrong", "raw_haar"]
    else:
        winner_arm = "obs_correct"
        controls = ["no_auxiliary", "smoothness", "static_g", "obs_wrong", "obs_haar"]

    d_bar_array = np.zeros((8, len(controls)))
    for k in range(8):
        for ci, c in enumerate(controls):
            d_seeds = []
            for s in range(3):
                d = results[k][s][winner_arm] - results[k][s][c]
                d_seeds.append(d)
            d_bar_array[k, ci] = np.mean(d_seeds)

    delta_min_bootstrap = np.zeros(n_replicates)
    for rep in range(n_replicates):
        sampled_keys = rng.integers(0, 8, size=8)
        sampled_d_bar = d_bar_array[sampled_keys]
        delta_per_control = sampled_d_bar.mean(axis=0)
        delta_min_bootstrap[rep] = delta_per_control.min()

    lcb_95 = float(np.percentile(delta_min_bootstrap, 2.5))
    ucb_95 = float(np.percentile(delta_min_bootstrap, 97.5))

    return {
        "n_replicates": n_replicates,
        "lcb_95": lcb_95,
        "ucb_95": ucb_95,
        "median": float(np.median(delta_min_bootstrap)),
        "mean": float(np.mean(delta_min_bootstrap)),
    }


def paired_sign_flip_test(
    results: dict,
    winner: str,
) -> dict:
    """Exact paired sign-flip test over 2^8 = 256 sign vectors.

    Computes joint-minimum statistic T_epsilon for each sign pattern.
    """
    if winner == "raw":
        winner_arm = "raw_correct"
        controls = ["no_auxiliary", "smoothness", "static_g", "raw_wrong", "raw_haar"]
    else:
        winner_arm = "obs_correct"
        controls = ["no_auxiliary", "smoothness", "static_g", "obs_wrong", "obs_haar"]

    d_bar_array = np.zeros((8, len(controls)))
    for k in range(8):
        for ci, c in enumerate(controls):
            d_seeds = []
            for s in range(3):
                d = results[k][s][winner_arm] - results[k][s][c]
                d_seeds.append(d)
            d_bar_array[k, ci] = np.mean(d_seeds)

    observed_delta_min = (d_bar_array.mean(axis=0)).min()

    count_ge = 0
    for bits in range(256):
        epsilon = np.array([(bits >> i) & 1 for i in range(8)]) * 2 - 1
        signed_d_bar = d_bar_array * epsilon[:, None]
        t_epsilon = (signed_d_bar.mean(axis=0)).min()
        if t_epsilon >= observed_delta_min:
            count_ge += 1

    p_value = count_ge / 256

    return {
        "observed_delta_min": float(observed_delta_min),
        "count_ge": count_ge,
        "total_permutations": 256,
        "p_value": float(p_value),
    }


def stage_c_verdict(
    primary: dict,
    bootstrap: dict,
    sign_flip: dict,
    probe_results: dict,
    protocol_checks: dict,
) -> dict:
    """Compute PASS/FAIL/VOID verdict for Stage C."""

    void_reasons = []
    if not protocol_checks.get("all_teachers_pass", True):
        void_reasons.append("Teacher capacity/extraction failure")
    if not protocol_checks.get("all_runs_complete", True):
        void_reasons.append("Incomplete runs")
    if not protocol_checks.get("hashes_verified", True):
        void_reasons.append("Hash verification failure")
    if not protocol_checks.get("no_forbidden_info", True):
        void_reasons.append("Forbidden information leak")
    if protocol_checks.get("key_commitment_mismatch", False):
        void_reasons.append("Key commitment mismatch")

    if void_reasons:
        return {
            "verdict": "VOID",
            "reasons": void_reasons,
        }

    fail_reasons = []

    if primary["delta_min"] < 0.20:
        fail_reasons.append(f"Delta_min={primary['delta_min']:.4f} < 0.20")

    if bootstrap["lcb_95"] <= 0.10:
        fail_reasons.append(f"Bootstrap LCB={bootstrap['lcb_95']:.4f} <= 0.10")

    if sign_flip["p_value"] > 0.05:
        fail_reasons.append(f"Sign-flip p={sign_flip['p_value']:.4f} > 0.05")

    for k in range(8):
        probe_acc = probe_results.get(k, {}).get("probe_acc", 0)
        if probe_acc < 0.70:
            fail_reasons.append(f"Key {k} probe acc={probe_acc:.4f} < 0.70")

    if fail_reasons:
        return {
            "verdict": "FAIL",
            "reasons": fail_reasons,
            "delta_min": primary["delta_min"],
            "lcb_95": bootstrap["lcb_95"],
            "p_value": sign_flip["p_value"],
        }

    return {
        "verdict": "PASS",
        "delta_min": primary["delta_min"],
        "lcb_95": bootstrap["lcb_95"],
        "p_value": sign_flip["p_value"],
        "claim": "Frozen artifact carried post-committed transition information "
                 "beyond five declared controls.",
    }


if __name__ == "__main__":
    print("Statistics module loaded.")
    print("Functions: stage_b_selection, stage_c_primary_statistic,")
    print("           key_cluster_bootstrap, paired_sign_flip_test, stage_c_verdict")

    rng = np.random.default_rng(42)
    mock_results = {}
    for k in range(2):
        mock_results[k] = {}
        for arm in ["raw_correct", "obs_correct", "no_auxiliary", "smoothness",
                     "static_g", "raw_wrong", "raw_haar", "obs_wrong", "obs_haar"]:
            if arm in ("raw_correct", "obs_correct"):
                mock_results[k][arm] = 0.60 + rng.random() * 0.30
            else:
                mock_results[k][arm] = 0.15 + rng.random() * 0.25

    sel = stage_b_selection(mock_results)
    print(f"\nMock Stage B selection: winner={sel['winner']}, verdict={sel['verdict']}")
    for cand, s in sel["selection"].items():
        print(f"  {cand}: eligible={s['eligible']}, mean_margin={s['mean_margin']:.4f}")
