"""Geometry Admission Test: statistical analysis for Stage B/C.

Stage B selection metric, Stage C bootstrap and sign-flip tests,
PASS/FAIL/VOID verdict calculation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


STAGE_B_PROTOCOL = "OCF_GAT_STAGE_B_STRUCTURAL_SCREEN_V1"
STAGE_B_SEEDS = [201, 202, 203]
STAGE_B_CANDIDATES = ["raw", "obs"]
STAGE_B_MIN_DELTA = 0.0
STAGE_B_FLOOR_DELTA = 0.10


def stage_b_structural_screen(
    withheld_accuracies: dict,
    protocol_checks: dict,
) -> dict:
    """Stage B structural screen: correct vs Haar for each seed x candidate.

    withheld_accuracies: {seed: {candidate: {"correct": acc, "haar": acc}}}
    protocol_checks: must contain all required fields; missing = VOID.

    Returns dict with verdict (STRUCTURAL_SCREEN_PASS/FAIL/VOID), winner, deltas.
    """
    required_checks = [
        "stage_a_artifacts_valid",
        "all_runs_complete",
        "initialization_hashes_paired",
        "coefficient_frozen",
        "artifact_hashes_valid",
        "no_forbidden_info",
        "all_losses_finite",
    ]
    void_reasons = []
    for check in required_checks:
        if check not in protocol_checks:
            void_reasons.append(f"Missing protocol check: {check}")
        elif not protocol_checks[check]:
            void_reasons.append(f"Failed protocol check: {check}")

    if void_reasons:
        return {"verdict": "STRUCTURAL_SCREEN_VOID", "reasons": void_reasons}

    selection = {}
    for cand in STAGE_B_CANDIDATES:
        deltas = []
        per_seed = {}
        for seed in STAGE_B_SEEDS:
            acc_correct = withheld_accuracies[seed][cand]["correct"]
            acc_haar = withheld_accuracies[seed][cand]["haar"]
            delta = acc_correct - acc_haar
            deltas.append(delta)
            per_seed[seed] = {
                "correct": float(acc_correct),
                "haar": float(acc_haar),
                "delta": float(delta),
            }

        min_delta = min(deltas)
        median_delta = float(np.median(deltas))
        mean_delta = float(np.mean(deltas))

        eligible = (
            min_delta > STAGE_B_MIN_DELTA
            and median_delta >= STAGE_B_FLOOR_DELTA
            and mean_delta >= STAGE_B_FLOOR_DELTA
        )

        from math import comb
        n = len(deltas)
        k = sum(1 for d in deltas if d > 0)
        sign_p = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)

        selection[cand] = {
            "per_seed": per_seed,
            "deltas": [float(d) for d in deltas],
            "min_delta": float(min_delta),
            "median_delta": median_delta,
            "mean_delta": mean_delta,
            "eligible": bool(eligible),
            "sign_test_p": float(sign_p),
        }

    eligible_candidates = [c for c, s in selection.items() if s["eligible"]]

    if not eligible_candidates:
        return {
            "winner": None,
            "verdict": "STRUCTURAL_SCREEN_FAIL",
            "reason": "No eligible candidate",
            "selection": selection,
        }

    if len(eligible_candidates) == 1:
        winner = eligible_candidates[0]
    else:
        raw_mean = selection["raw"]["mean_delta"]
        obs_mean = selection["obs"]["mean_delta"]
        if raw_mean > obs_mean:
            winner = "raw"
        else:
            winner = "obs"

    return {
        "winner": winner,
        "verdict": "STRUCTURAL_SCREEN_PASS",
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

    required_checks = [
        "all_teachers_pass",
        "all_runs_complete",
        "hashes_verified",
        "no_forbidden_info",
    ]
    void_reasons = []
    for check in required_checks:
        if check not in protocol_checks:
            void_reasons.append(f"Missing protocol check: {check}")
        elif not protocol_checks[check]:
            void_reasons.append(f"Failed: {check}")
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


def counterfactual_edge_crossover(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    changed_edges: list[dict],
) -> dict:
    """Compute bidirectional crossover on changed direct edges.

    logits_a: logits from student installed with artifact Z_A, shape (48, 12)
    logits_b: logits from student installed with artifact Z_B, shape (48, 12)
    changed_edges: list of 2 dicts with keys 'edge_index', 'base_output', 'partner_output'

    Returns per-edge margins and crossover success.
    """
    results = []
    for edge in changed_edges:
        idx = edge["edge_index"]
        y_a = edge["base_output"]
        y_b = edge["partner_output"]

        m_a = float(logits_a[idx, y_a] - logits_a[idx, y_b])
        m_b = float(logits_b[idx, y_b] - logits_b[idx, y_a])

        results.append({
            "edge_index": idx,
            "m_a": m_a,
            "m_b": m_b,
            "d": m_a + m_b,
            "a_correct": m_a > 0,
            "b_correct": m_b > 0,
            "crossover": m_a > 0 and m_b > 0,
        })

    return {
        "edges": results,
        "all_crossover": all(r["crossover"] for r in results),
        "mean_d": float(np.mean([r["d"] for r in results])),
    }


def unchanged_edge_stability(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    changed_edge_indices: list[int],
    drift_ceiling: float = 0.5,
) -> dict:
    """Check that unchanged direct edges remain stable between A and B artifacts.

    Uses total variation distance on softmax distributions.
    """
    n_edges = logits_a.shape[0]
    unchanged = [i for i in range(n_edges) if i not in changed_edge_indices]

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    drifts = []
    flips = 0
    for i in unchanged:
        p_a = softmax(logits_a[i])
        p_b = softmax(logits_b[i])
        tv = 0.5 * np.sum(np.abs(p_a - p_b))
        drifts.append(float(tv))
        if np.argmax(logits_a[i]) != np.argmax(logits_b[i]):
            flips += 1

    return {
        "mean_tv": float(np.mean(drifts)),
        "max_tv": float(np.max(drifts)),
        "flip_count": flips,
        "flip_rate": flips / len(unchanged) if unchanged else 0.0,
        "n_unchanged": len(unchanged),
        "stable": float(np.max(drifts)) <= drift_ceiling and flips <= 2,
    }


def cm_pair_success(
    crossover: dict,
    stability: dict,
    effect_floor: float = 0.5,
) -> dict:
    """Determine if a single CM-CKS pair passes all conjunctive criteria."""
    pass_crossover = crossover["all_crossover"]
    pass_effect = crossover["mean_d"] >= effect_floor
    pass_stability = stability["stable"]

    return {
        "success": pass_crossover and pass_effect and pass_stability,
        "pass_crossover": pass_crossover,
        "pass_effect": pass_effect,
        "pass_stability": pass_stability,
        "mean_d": crossover["mean_d"],
        "mean_tv": stability["mean_tv"],
    }


def cm_exact_sign_test(pair_successes: list[bool]) -> dict:
    """Exact sign test over CM-CKS pairs.

    For 8 pairs: P(X >= 7 | p=0.5) = 9/256 = 0.0352.
    """
    n = len(pair_successes)
    k = sum(pair_successes)

    from math import comb
    p_value = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)

    return {
        "n_pairs": n,
        "n_success": k,
        "p_value": float(p_value),
        "threshold": 7 if n == 8 else max(1, int(n * 0.875)),
        "pass": k >= (7 if n == 8 else max(1, int(n * 0.875))),
    }


def cm_cks_verdict(
    pair_results: list[dict],
    protocol_checks: dict,
) -> dict:
    """Compute CM-CKS PASS/FAIL/VOID verdict. Fail-closed: missing checks = VOID."""

    required_checks = [
        "all_teachers_pass",
        "all_runs_complete",
        "pairs_constructed_correctly",
        "calibration_hashes_match",
        "logits_finite",
    ]
    void_reasons = []
    if len(pair_results) != 8:
        void_reasons.append(f"Expected 8 pairs, got {len(pair_results)}")
    for check in required_checks:
        if check not in protocol_checks:
            void_reasons.append(f"Missing protocol check: {check}")
        elif not protocol_checks[check]:
            void_reasons.append(f"Failed: {check}")
    if void_reasons:
        return {"verdict": "VOID", "reasons": void_reasons}

    successes = [p["success"] for p in pair_results]
    sign_test = cm_exact_sign_test(successes)

    effects = [p["mean_d"] for p in pair_results]
    stabilities = [p["mean_tv"] for p in pair_results]

    fail_reasons = []
    if not sign_test["pass"]:
        fail_reasons.append(
            f"Sign test: {sign_test['n_success']}/{sign_test['n_pairs']} "
            f"< {sign_test['threshold']}"
        )
    if float(np.mean(effects)) < 0.5:
        fail_reasons.append(
            f"Aggregate effect floor: mean_d={float(np.mean(effects)):.4f} < 0.5"
        )
    if float(np.max(stabilities)) > 0.5:
        fail_reasons.append(
            f"Aggregate stability ceiling: max_tv={float(np.max(stabilities)):.4f} > 0.5"
        )

    if fail_reasons:
        return {
            "verdict": "FAIL",
            "reasons": fail_reasons,
            "n_success": sign_test["n_success"],
            "n_pairs": sign_test["n_pairs"],
            "p_value": sign_test["p_value"],
            "mean_effect": float(np.mean(effects)),
            "mean_stability": float(np.mean(stabilities)),
        }

    return {
        "verdict": "PASS",
        "n_success": sign_test["n_success"],
        "n_pairs": sign_test["n_pairs"],
        "p_value": sign_test["p_value"],
        "mean_effect": float(np.mean(effects)),
        "mean_stability": float(np.mean(stabilities)),
        "claim": "Calibration-matched counterfactual key swap produced "
                 "localized, correctly signed behavioral change.",
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
