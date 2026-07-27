"""Geometry Admission Test: Stage C-I orchestrator (CM-CKS sealed confirmation).

PERMANENTLY CLOSED (Jul 26, 2026).
Stage B STRUCTURAL_SCREEN_FAIL killed the geometry transfer thesis.
R11 steering converged on Causal Skill Organs (CSO) as successor.
This file is preserved for historical reference only.
"""
raise SystemExit(
    "GAT Stage C-I is PERMANENTLY CLOSED (Jul 26 2026). "
    "Stage B STRUCTURAL_SCREEN_FAIL killed the geometry transfer thesis. "
    "See results/geometry_admission/stage_b/decision.json and STATUS.md."
)

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cti_geometry_admission_automaton import (
    OP_NAMES,
    NUM_STATES,
    NUM_OPS,
    key_from_json,
    generate_key_from_seed,
    generate_sealed_key,
    generate_all_eval_sets,
    generate_calibration_set,
    generate_direct_edges,
    generate_anchors,
    partition_anchors_into_banks,
    generate_bank_order_permutation,
    paired_key_from_transposition,
    hash_eval_set,
    collate_fn,
)
from cti_geometry_admission_models import create_teacher, create_transformer_student, count_parameters
from cti_geometry_admission_trainer import train_one_run as train_teacher_run, _two_eval_pass
from cti_geometry_admission_extraction import (
    extract_hidden_states,
    extract_raw_trace,
    extract_observable_connection,
    generate_perturbations,
    TEACHER_DEPTH_LAYERS,
)
from cti_geometry_admission_installer import (
    calibrate_coefficient,
    train_installer_run,
    evaluate_direct_edge_logits,
)
from cti_geometry_admission_statistics import (
    counterfactual_edge_crossover,
    unchanged_edge_stability,
    cm_pair_success,
    cm_exact_sign_test,
    cm_cks_verdict,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_c_cm"
SECRETS_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "secrets"
STAGE_B_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_b"

NUM_SEALED_PAIRS = 8
INSTALLER_SEEDS = [501, 502]


def edge_index(state: int, op_name: str) -> int:
    return state * NUM_OPS + OP_NAMES.index(op_name)


def derive_sealed_partner(base_key_json: dict, pair_index: int, key_slot: int) -> tuple[dict, dict]:
    """Derive a deterministic transposition partner for a sealed base key."""
    calibrated_op = OP_NAMES[key_slot % NUM_OPS]
    other_ops = [op for op in OP_NAMES if op != calibrated_op]

    h = hashlib.sha256(
        f"GAT_STAGE_C_CM_PARTNER_{pair_index}".encode()
    ).digest()
    withheld_op = other_ops[int(h[0]) % len(other_ops)]
    u = int(h[1]) % NUM_STATES
    v = (u + 1 + int(h[2]) % (NUM_STATES - 1)) % NUM_STATES

    partner_key, metadata = paired_key_from_transposition(
        base_key_json, calibrated_op, withheld_op, u, v,
    )
    return partner_key, metadata


def generate_sealed_pairs() -> list[dict]:
    """Generate 8 independent sealed base/partner pairs."""
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = []
    commitments = []

    for i in range(NUM_SEALED_PAIRS):
        base_key_json, seed_bytes, seed_hash = generate_sealed_key(i)
        partner_key_json, pair_metadata = derive_sealed_partner(base_key_json, i, key_slot=0)

        pair_data = {
            "pair_index": i,
            "base_key_json": base_key_json,
            "partner_key_json": partner_key_json,
            "seed_bytes_hex": seed_bytes.hex(),
            "seed_hash": seed_hash,
            "pair_metadata": pair_metadata,
        }
        pairs.append(pair_data)
        commitments.append({
            "pair_index": i,
            "base_commitment": seed_hash,
            "partner_key_hash": pair_metadata["partner_key_hash"],
        })

        secret_path = SECRETS_DIR / f"cm_pair_{i:02d}_secret.json"
        with open(secret_path, "w") as f:
            json.dump(pair_data, f, indent=2)

    manifest_hash = hashlib.sha256(
        json.dumps(commitments, sort_keys=True).encode()
    ).hexdigest()
    with open(RESULTS_DIR / "pair_commitments.json", "w") as f:
        json.dump({"commitments": commitments, "manifest_hash": manifest_hash}, f, indent=2)

    return pairs


def extract_teacher_artifacts(key_json: dict, teacher_dir: Path, device: torch.device) -> dict:
    """Extract raw and observable artifacts from a trained teacher."""
    key = key_from_json(key_json)
    teacher_path = teacher_dir / "model_final.pt"
    teacher = create_teacher().to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher.eval()

    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)

    artifacts = {"raw": {}, "obs": {}}
    for bank_idx, bank in enumerate(banks):
        ticks = extract_hidden_states(teacher, bank, device, TEACHER_DEPTH_LAYERS)

        raw_transitions = extract_raw_trace(ticks, list(range(len(TEACHER_DEPTH_LAYERS))))
        artifacts["raw"][bank_idx] = {
            j: t["R"].astype(np.float32) for j, t in raw_transitions.items()
        }

        perturbations = [generate_perturbations(a, key) for a in bank]
        obs_transitions = extract_observable_connection(
            teacher, bank, perturbations, device, TEACHER_DEPTH_LAYERS,
        )
        artifacts["obs"][bank_idx] = {
            j: {
                "R_obs": t["R_obs"].astype(np.float32),
                "U_basis": t["U_basis"].astype(np.float32),
            } for j, t in obs_transitions.items()
        }

    del teacher
    torch.cuda.empty_cache()
    return artifacts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    decision_path = STAGE_B_DIR / "decision.json"
    if not decision_path.exists():
        print("ERROR: Stage B decision not found. Run Stage B first.")
        return
    with open(decision_path) as f:
        stage_b_decision = json.load(f)
    if stage_b_decision["verdict"] != "PASS":
        print(f"Stage B verdict: {stage_b_decision['verdict']} -- cannot proceed.")
        return

    winner = stage_b_decision["winner"]
    winner_arm = f"{winner}_correct"
    frozen_coeff = stage_b_decision["frozen_coefficients"][winner_arm]
    print(f"Winner: {winner}, arm: {winner_arm}, coefficient: {frozen_coeff:.6f}")

    source_files = [
        "cti_geometry_admission_automaton.py",
        "cti_geometry_admission_models.py",
        "cti_geometry_admission_trainer.py",
        "cti_geometry_admission_extraction.py",
        "cti_geometry_admission_geometry.py",
        "cti_geometry_admission_installer.py",
        "cti_geometry_admission_statistics.py",
        "cti_geometry_admission_stage_c.py",
    ]
    src_dir = Path(__file__).resolve().parent
    code_hashes = {}
    for sf in source_files:
        sf_path = src_dir / sf
        if sf_path.exists():
            code_hashes[sf] = hashlib.sha256(sf_path.read_bytes()).hexdigest()
    code_manifest = hashlib.sha256(
        json.dumps(code_hashes, sort_keys=True).encode()
    ).hexdigest()

    stage_c_config = {
        "winner": winner,
        "winner_arm": winner_arm,
        "frozen_coefficient": frozen_coeff,
        "n_pairs": NUM_SEALED_PAIRS,
        "installer_seeds": INSTALLER_SEEDS,
        "total_runs": NUM_SEALED_PAIRS * 2 * len(INSTALLER_SEEDS),
        "code_manifest": code_manifest,
        "code_hashes": code_hashes,
    }
    with open(RESULTS_DIR / "stage_c_config.json", "w") as f:
        json.dump(stage_c_config, f, indent=2)
    print(f"Stage C precommitment written. Code manifest: {code_manifest[:16]}...")

    print("\n" + "=" * 60)
    print("STAGE C-I: GENERATING SEALED PAIRS")
    print("=" * 60)

    commitments_path = RESULTS_DIR / "pair_commitments.json"
    if commitments_path.exists():
        print("Pair commitments already exist, loading secrets...")
        pairs = []
        for i in range(NUM_SEALED_PAIRS):
            secret_path = SECRETS_DIR / f"cm_pair_{i:02d}_secret.json"
            with open(secret_path) as f:
                pairs.append(json.load(f))
    else:
        pairs = generate_sealed_pairs()
    print(f"  {NUM_SEALED_PAIRS} sealed pairs generated and committed.")

    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)
    bank_order = generate_bank_order_permutation()

    print("\n" + "=" * 60)
    print(f"STAGE C-I: TRAINING {NUM_SEALED_PAIRS * 2} TEACHERS")
    print("=" * 60)

    import cti_geometry_admission_trainer as trainer_mod
    original_results_dir = trainer_mod.RESULTS_DIR
    trainer_mod.RESULTS_DIR = RESULTS_DIR

    teacher_pass = True
    for pi, pair in enumerate(pairs):
        for role, key_json in [("base", pair["base_key_json"]),
                                ("partner", pair["partner_key_json"])]:
            name = f"pair{pi:02d}_{role}_teacher"
            key = key_from_json(key_json)
            eval_sets = generate_all_eval_sets(key, seed=42)

            print(f"\n  Training {name}...")
            summary = train_teacher_run(
                {"name": name, "arch": "teacher", "seed": 101, "lr": 3e-4},
                key, eval_sets, device, allow_resume=True,
            )

            gate_ok = (
                summary["final_in_range"] >= 0.995
                and summary["final_extrapolation"] >= 0.990
                and summary["final_direct_edges"] == 48
                and _two_eval_pass(summary, 0.995, 0.990, 48)
            )
            if not gate_ok:
                print(f"  WARNING: {name} below capacity!")
                teacher_pass = False

    trainer_mod.RESULTS_DIR = original_results_dir

    if not teacher_pass:
        print("\nSTAGE C-I: VOID -- Teacher capacity failures.")
        with open(RESULTS_DIR / "verdict.json", "w") as f:
            json.dump({"verdict": "VOID", "reason": "Teacher capacity failures"}, f, indent=2)
        return

    print("\n" + "=" * 60)
    print("STAGE C-I: EXTRACTING ARTIFACTS")
    print("=" * 60)

    pair_artifacts = []
    for pi, pair in enumerate(pairs):
        print(f"\n  Pair {pi}: extracting base + partner artifacts...")
        base_art = extract_teacher_artifacts(
            pair["base_key_json"],
            RESULTS_DIR / f"pair{pi:02d}_base_teacher",
            device,
        )
        partner_art = extract_teacher_artifacts(
            pair["partner_key_json"],
            RESULTS_DIR / f"pair{pi:02d}_partner_teacher",
            device,
        )
        pair_artifacts.append({"base": base_art, "partner": partner_art})

    total_runs = NUM_SEALED_PAIRS * 2 * len(INSTALLER_SEEDS)
    print("\n" + "=" * 60)
    print(f"STAGE C-I: INSTALLER RUNS ({total_runs} total)")
    print("=" * 60)

    all_pair_results = []
    run_count = 0

    for pi, pair in enumerate(pairs):
        base_key = key_from_json(pair["base_key_json"])
        partner_key = key_from_json(pair["partner_key_json"])
        base_direct = generate_direct_edges(base_key)
        partner_direct = generate_direct_edges(partner_key)

        cal_base = generate_calibration_set(base_key, key_slot=0)
        cal_partner = generate_calibration_set(partner_key, key_slot=0)

        metadata = pair["pair_metadata"]
        changed = metadata["changed_edges"]
        changed_indices = [edge_index(e["state"], e["op"]) for e in changed]

        seed_pair_results = []

        for seed in INSTALLER_SEEDS:
            run_a_name = f"pair{pi:02d}_{winner}_A_s{seed}"
            run_b_name = f"pair{pi:02d}_{winner}_B_s{seed}"

            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {run_a_name}")
            train_installer_run(
                {"name": run_a_name, "arm": winner_arm, "seed": seed,
                 "lr": 5e-4, "arch": "transformer", "coefficient": frozen_coeff},
                {"raw": pair_artifacts[pi]["base"]["raw"],
                 "obs": pair_artifacts[pi]["base"]["obs"]},
                cal_base, [], base_direct, banks, bank_order, RESULTS_DIR, device,
                seal_probes=True,
            )

            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {run_b_name}")
            train_installer_run(
                {"name": run_b_name, "arm": winner_arm, "seed": seed,
                 "lr": 5e-4, "arch": "transformer", "coefficient": frozen_coeff},
                {"raw": pair_artifacts[pi]["partner"]["raw"],
                 "obs": pair_artifacts[pi]["partner"]["obs"]},
                cal_partner, [], partner_direct, banks, bank_order, RESULTS_DIR, device,
                seal_probes=True,
            )

            model_a = create_transformer_student().to(device)
            model_a.load_state_dict(torch.load(
                RESULTS_DIR / run_a_name / "model_final.pt",
                map_location=device, weights_only=True,
            ))
            la = evaluate_direct_edge_logits(model_a, base_direct, device)
            del model_a

            model_b = create_transformer_student().to(device)
            model_b.load_state_dict(torch.load(
                RESULTS_DIR / run_b_name / "model_final.pt",
                map_location=device, weights_only=True,
            ))
            lb = evaluate_direct_edge_logits(model_b, partner_direct, device)
            del model_b
            torch.cuda.empty_cache()

            crossover = counterfactual_edge_crossover(la, lb, changed)
            stability = unchanged_edge_stability(la, lb, changed_indices)
            pair_result = cm_pair_success(crossover, stability)

            seed_pair_results.append({
                "seed": seed,
                "crossover": crossover,
                "stability": stability,
                "pair_result": pair_result,
            })

            print(f"  crossover={crossover['all_crossover']}, "
                  f"mean_d={crossover['mean_d']:.4f}, "
                  f"stable={stability['stable']}, "
                  f"success={pair_result['success']}")

        agg_crossover_flags = [r["crossover"]["all_crossover"] for r in seed_pair_results]
        agg_mean_d = float(np.mean([r["crossover"]["mean_d"] for r in seed_pair_results]))
        agg_mean_tv = float(np.mean([r["stability"]["mean_tv"] for r in seed_pair_results]))
        agg_max_tv = float(np.max([r["stability"]["max_tv"] for r in seed_pair_results]))
        agg_flips = sum(r["stability"]["flip_count"] for r in seed_pair_results)

        avg_crossover = {"all_crossover": all(agg_crossover_flags), "mean_d": agg_mean_d}
        avg_stability = {"stable": agg_max_tv <= 0.5 and agg_flips <= 2,
                         "mean_tv": agg_mean_tv, "max_tv": agg_max_tv, "flip_count": agg_flips}
        pair_success = cm_pair_success(avg_crossover, avg_stability)

        pair_summary = {
            "pair_index": pi,
            "success": pair_success["success"],
            "mean_d": agg_mean_d,
            "mean_tv": agg_mean_tv,
            "pass_crossover": pair_success["pass_crossover"],
            "pass_effect": pair_success["pass_effect"],
            "pass_stability": pair_success["pass_stability"],
            "per_seed": [{
                "seed": r["seed"],
                "crossover_success": r["crossover"]["all_crossover"],
                "mean_d": r["crossover"]["mean_d"],
                "stable": r["stability"]["stable"],
                "success": r["pair_result"]["success"],
            } for r in seed_pair_results],
        }
        all_pair_results.append(pair_summary)
        print(f"\n  Pair {pi} aggregate: success={pair_success['success']}, mean_d={agg_mean_d:.4f}")

    print("\n" + "=" * 60)
    print("STAGE C-I: STATISTICAL ANALYSIS")
    print("=" * 60)

    cal_hashes_ok = True
    for pi, pair in enumerate(pairs):
        bk = key_from_json(pair["base_key_json"])
        pk = key_from_json(pair["partner_key_json"])
        cal_b = generate_calibration_set(bk, key_slot=0)
        cal_p = generate_calibration_set(pk, key_slot=0)
        h_b = hash_eval_set(cal_b)
        h_p = hash_eval_set(cal_p)
        if h_b != h_p:
            print(f"  ERROR: Pair {pi} calibration hash mismatch: {h_b[:16]} != {h_p[:16]}")
            cal_hashes_ok = False

    protocol_checks = {
        "all_teachers_pass": teacher_pass,
        "all_runs_complete": run_count == total_runs,
        "pairs_constructed_correctly": all(
            p["pair_metadata"]["calibration_identical"]
            and p["pair_metadata"]["num_differing_entries"] == 2
            for p in pairs
        ),
        "calibration_hashes_match": cal_hashes_ok,
    }

    verdict = cm_cks_verdict(all_pair_results, protocol_checks)

    successes = [p["success"] for p in all_pair_results]
    sign_test = cm_exact_sign_test(successes)

    full_analysis = {
        "winner": winner,
        "n_pairs": NUM_SEALED_PAIRS,
        "n_seeds": len(INSTALLER_SEEDS),
        "pair_results": all_pair_results,
        "sign_test": sign_test,
        "protocol_checks": protocol_checks,
        "verdict": verdict,
    }

    with open(RESULTS_DIR / "verdict.json", "w") as f:
        json.dump(full_analysis, f, indent=2)

    print(f"\nSign test: {sign_test['n_success']}/{sign_test['n_pairs']} "
          f"(p={sign_test['p_value']:.4f}, threshold={sign_test['threshold']})")

    print(f"\n{'=' * 60}")
    print(f"STAGE C-I VERDICT: {verdict['verdict']}")
    print(f"{'=' * 60}")

    if verdict["verdict"] == "PASS":
        print(f"  {sign_test['n_success']}/{sign_test['n_pairs']} pairs pass (>= 7/8)")
        print(f"  p-value = {sign_test['p_value']:.4f}")
        print(f"  mean effect = {verdict['mean_effect']:.4f}")
        print(f"  Claim: {verdict['claim']}")
    elif verdict["verdict"] == "FAIL":
        for r in verdict.get("reasons", []):
            print(f"  - {r}")
        print("\nCM-CKS effect not confirmed at 7/8 threshold.")
    else:
        for r in verdict.get("reasons", []):
            print(f"  - {r}")


if __name__ == "__main__":
    main()
