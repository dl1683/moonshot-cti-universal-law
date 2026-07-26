"""Geometry Admission Test: Stage C orchestrator.

144-run sealed confirmation: 8 keys x 3 seeds x 6 arms (winner + 5 controls).
Bootstrap CI + exact sign-flip test for PASS/FAIL/VOID.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cti_geometry_admission_automaton import (
    key_from_json,
    generate_key_from_seed,
    generate_sealed_key,
    generate_all_eval_sets,
    generate_calibration_set,
    generate_withheld_eval_set,
    generate_anchors,
    partition_anchors_into_banks,
    generate_bank_order_permutation,
    hash_eval_set,
    collate_fn,
)
from cti_geometry_admission_models import create_teacher, count_parameters
from cti_geometry_admission_trainer import train_one_run as train_teacher_capacity
from cti_geometry_admission_extraction import (
    extract_hidden_states,
    extract_raw_trace,
    extract_observable_connection,
    generate_perturbations,
    check_numerical_gates,
    serialize_traces,
    center_and_normalize,
    TEACHER_DEPTH_LAYERS,
)
from cti_geometry_admission_geometry import (
    generate_haar_rotation_raw,
    generate_haar_rotation_obs,
    apply_haar_to_raw_targets,
    apply_haar_to_obs_targets,
)
from cti_geometry_admission_installer import (
    calibrate_coefficient,
    train_installer_run,
    ARMS,
)
from cti_geometry_admission_statistics import (
    stage_c_primary_statistic,
    key_cluster_bootstrap,
    paired_sign_flip_test,
    stage_c_verdict,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_c"
SECRETS_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "secrets"

NUM_SEALED_KEYS = 8
NUM_SEEDS = 3
INSTALLER_SEEDS = [501, 502, 503]


def generate_and_commit_keys() -> list[dict]:
    """Generate 8 sealed keys, save seeds to secrets dir, commitments to results."""
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    keys = []
    commitments = []

    for i in range(NUM_SEALED_KEYS):
        key_json, seed_bytes, seed_hash = generate_sealed_key(i)
        keys.append({
            "key_index": i,
            "key_json": key_json,
            "seed_bytes_hex": seed_bytes.hex(),
            "seed_hash": seed_hash,
        })
        commitments.append({
            "key_index": i,
            "commitment": seed_hash,
        })

        secret_path = SECRETS_DIR / f"key_{i:02d}_secret.json"
        with open(secret_path, "w") as f:
            json.dump({
                "key_index": i,
                "seed_bytes_hex": seed_bytes.hex(),
                "seed_hash": seed_hash,
                "key_json": key_json,
            }, f, indent=2)

    commitment_path = RESULTS_DIR / "key_commitments.json"
    commitment_hash = hashlib.sha256(
        json.dumps(commitments, sort_keys=True).encode()
    ).hexdigest()
    with open(commitment_path, "w") as f:
        json.dump({
            "commitments": commitments,
            "manifest_hash": commitment_hash,
        }, f, indent=2)

    return keys


def verify_key_commitment(keys: list[dict]) -> bool:
    """Verify that all key commitments match."""
    for k in keys:
        seed_bytes = bytes.fromhex(k["seed_bytes_hex"])
        expected_hash = hashlib.sha256(seed_bytes).hexdigest()
        if expected_hash != k["seed_hash"]:
            return False
        regen_key = generate_key_from_seed(seed_bytes)
        if regen_key != k["key_json"]:
            return False
    return True


def train_teacher_for_key(key_json: dict, key_idx: int, device: torch.device) -> dict:
    """Train a teacher to capacity on the given key."""
    key = key_from_json(key_json)
    eval_sets = generate_all_eval_sets(key, seed=42)

    run_cfg = {
        "name": f"teacher_key{key_idx:02d}",
        "arch": "teacher",
        "seed": 101,
        "lr": 3e-4,
    }

    import cti_geometry_admission_trainer as trainer_mod
    original_dir = trainer_mod.RESULTS_DIR
    trainer_mod.RESULTS_DIR = RESULTS_DIR

    summary = train_teacher_capacity(run_cfg, key, eval_sets, device)

    trainer_mod.RESULTS_DIR = original_dir
    return summary


def extract_teacher_artifacts(key_json: dict, key_idx: int, device: torch.device) -> dict:
    """Extract raw R, observable R, and static G artifacts from trained teacher."""
    key = key_from_json(key_json)

    teacher_dir = RESULTS_DIR / f"teacher_key{key_idx:02d}"
    teacher_path = teacher_dir / "model_final.pt"
    teacher = create_teacher().to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher.eval()

    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)

    artifacts = {"raw": {}, "obs": {}, "static_g": {}}
    numerical_all_pass = True

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

        artifacts["static_g"][bank_idx] = {}
        for tick_idx in range(len(TEACHER_DEPTH_LAYERS)):
            X = center_and_normalize(ticks[tick_idx])
            G = (X @ X.T).astype(np.float32)
            artifacts["static_g"][bank_idx][tick_idx] = G

        numerical = check_numerical_gates(raw_transitions, obs_transitions)
        if not numerical["all_pass"]:
            numerical_all_pass = False

    del teacher
    torch.cuda.empty_cache()

    return artifacts, numerical_all_pass


def build_controls_for_key(
    correct_artifacts: dict,
    wrong_artifacts: dict,
) -> dict:
    """Build wrong-key and Haar-matched control targets for one key."""
    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)
    n_banks = len(banks)
    bank_size = len(banks[0])

    controls = {
        "raw_wrong": wrong_artifacts["raw"],
        "obs_wrong": {},
        "raw_haar": {},
        "obs_haar": {},
    }

    for bank_idx in range(n_banks):
        controls["obs_wrong"][bank_idx] = wrong_artifacts["obs"][bank_idx]

        Q_raw = generate_haar_rotation_raw(bank_size, bank_idx)
        Q_obs = generate_haar_rotation_obs(8, bank_idx)

        raw_correct = {j: correct_artifacts["raw"][bank_idx][j] for j in range(6)}
        haar_raw = apply_haar_to_raw_targets(
            [raw_correct[j] for j in range(6)], Q_raw,
        )
        controls["raw_haar"][bank_idx] = {j: haar_raw[j] for j in range(6)}

        obs_correct_list = [correct_artifacts["obs"][bank_idx][j]["R_obs"] for j in range(6)]
        haar_obs = apply_haar_to_obs_targets(obs_correct_list, Q_obs)
        controls["obs_haar"][bank_idx] = {
            j: {
                "R_obs": haar_obs[j],
                "U_basis": correct_artifacts["obs"][bank_idx][j]["U_basis"],
            } for j in range(6)
        }

    return controls


def select_arms_for_winner(winner: str) -> list[str]:
    """Return the 6 arms for sealed confirmation based on Stage B winner."""
    if winner == "raw":
        return ["raw_correct", "no_auxiliary", "smoothness", "static_g", "raw_wrong", "raw_haar"]
    else:
        return ["obs_correct", "no_auxiliary", "smoothness", "static_g", "obs_wrong", "obs_haar"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stage_b_dir = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_b"
    decision_path = stage_b_dir / "decision.json"
    if not decision_path.exists():
        print("ERROR: Stage B decision not found. Run Stage B first.")
        return

    with open(decision_path) as f:
        stage_b_decision = json.load(f)

    if stage_b_decision["verdict"] != "PASS":
        print(f"Stage B verdict: {stage_b_decision['verdict']} - cannot proceed to Stage C.")
        return

    winner = stage_b_decision["winner"]
    print(f"Stage B winner: {winner}")

    arms = select_arms_for_winner(winner)
    print(f"Arms for Stage C: {arms}")
    total_runs = NUM_SEALED_KEYS * NUM_SEEDS * len(arms)
    print(f"Total runs: {total_runs}")

    frozen_coeff_path = stage_b_dir / "frozen_coefficients.json"
    if not frozen_coeff_path.exists():
        print("ERROR: Frozen coefficients from Stage B not found.")
        return
    with open(frozen_coeff_path) as f:
        frozen_coefficients = json.load(f)
    print(f"Loaded frozen coefficients: {frozen_coefficients}")

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
    code_manifest_hash = hashlib.sha256(
        json.dumps(code_hashes, sort_keys=True).encode()
    ).hexdigest()

    stage_c_config = {
        "winner": winner,
        "arms": arms,
        "n_keys": NUM_SEALED_KEYS,
        "n_seeds": NUM_SEEDS,
        "seeds": INSTALLER_SEEDS,
        "total_runs": total_runs,
        "stage_b_decision_path": str(decision_path),
        "frozen_coefficients": frozen_coefficients,
        "code_manifest_hash": code_manifest_hash,
        "code_hashes": code_hashes,
    }

    with open(RESULTS_DIR / "stage_c_config.json", "w") as f:
        json.dump(stage_c_config, f, indent=2)
    print(f"Stage C precommitment written. Code manifest: {code_manifest_hash[:16]}...")

    print("\n" + "=" * 60)
    print("STAGE C: GENERATING SEALED KEYS")
    print("=" * 60)

    commitments_path = RESULTS_DIR / "key_commitments.json"
    if commitments_path.exists():
        print("Key commitments already exist, loading secrets...")
        keys = []
        for i in range(NUM_SEALED_KEYS):
            secret_path = SECRETS_DIR / f"key_{i:02d}_secret.json"
            with open(secret_path) as f:
                keys.append(json.load(f))
    else:
        keys = generate_and_commit_keys()

    assert verify_key_commitment(keys), "Key commitment verification failed!"
    print(f"  {NUM_SEALED_KEYS} sealed keys generated and committed.")

    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)
    bank_order = generate_bank_order_permutation()

    print("\n" + "=" * 60)
    print("STAGE C: TEACHER TRAINING + EXTRACTION (8 keys)")
    print("=" * 60)

    all_artifacts = {}
    teacher_pass = True

    for ki, key_data in enumerate(keys):
        key_json = key_data["key_json"]
        print(f"\n--- Key {ki} ---")

        print(f"  Training teacher...")
        summary = train_teacher_for_key(key_json, ki, device)
        from cti_geometry_admission_trainer import _two_eval_pass
        teacher_ok = (
            summary["final_in_range"] >= 0.995
            and summary["final_extrapolation"] >= 0.990
            and summary["final_direct_edges"] == 48
            and _two_eval_pass(summary, 0.995, 0.990, 48)
        )
        if not teacher_ok:
            print(f"  WARNING: Teacher for key {ki} below capacity threshold!")
            print(f"    in_range={summary['final_in_range']:.4f} "
                  f"extrap={summary['final_extrapolation']:.4f} "
                  f"edges={summary['final_direct_edges']}/48")
            teacher_pass = False

        print(f"  Extracting artifacts...")
        artifacts, numerical_pass = extract_teacher_artifacts(key_json, ki, device)
        all_artifacts[ki] = artifacts
        if not numerical_pass:
            print(f"  WARNING: Numerical gates failed for key {ki}!")
            teacher_pass = False

    if not teacher_pass:
        print("\nSTAGE C: VOID - Teacher capacity/extraction failures.")
        with open(RESULTS_DIR / "verdict.json", "w") as f:
            json.dump({"verdict": "VOID", "reason": "Teacher failures"}, f, indent=2)
        return

    print("\n" + "=" * 60)
    print("STAGE C: BUILDING CONTROLS")
    print("=" * 60)

    all_controls = {}
    for ki in range(NUM_SEALED_KEYS):
        wrong_ki = (ki + 1) % NUM_SEALED_KEYS
        all_controls[ki] = build_controls_for_key(
            all_artifacts[ki], all_artifacts[wrong_ki],
        )
        print(f"  Key {ki}: wrong-key source = key {wrong_ki}, Haar rotations built.")

    print("\n" + "=" * 60)
    print(f"STAGE C: INSTALLER RUNS ({total_runs} total)")
    print("=" * 60)

    all_results = {}
    run_count = 0

    for ki, key_data in enumerate(keys):
        key_json = key_data["key_json"]
        key = key_from_json(key_json)

        cal = generate_calibration_set(key, key_slot=ki)
        withheld, probes = generate_withheld_eval_set(key, key_slot=ki, key_index=ki)

        combined_artifacts = {
            "raw": all_artifacts[ki]["raw"],
            "obs": all_artifacts[ki]["obs"],
            "static_g": all_artifacts[ki]["static_g"],
            "raw_wrong": all_controls[ki]["raw_wrong"],
            "obs_wrong": all_controls[ki]["obs_wrong"],
            "raw_haar": all_controls[ki]["raw_haar"],
            "obs_haar": all_controls[ki]["obs_haar"],
        }

        all_results[ki] = {}

        for si, seed in enumerate(INSTALLER_SEEDS):
            all_results[ki][si] = {}

            for arm in arms:
                run_name = f"key{ki:02d}_s{seed}_{arm}"
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] {run_name}")

                coeff = frozen_coefficients.get(arm, 1.0)

                run_config = {
                    "name": run_name,
                    "arm": arm,
                    "seed": seed,
                    "lr": 5e-4,
                    "arch": "transformer",
                    "coefficient": coeff,
                }

                summary = train_installer_run(
                    run_config, combined_artifacts, cal, withheld, probes,
                    banks, bank_order, RESULTS_DIR, device,
                )

                all_results[ki][si][arm] = summary["final_withheld_acc"]
                print(f"  acc={summary['final_withheld_acc']:.4f} "
                      f"probe={summary['final_probe_correct']}/{summary['final_probe_total']}")

    raw_results_path = RESULTS_DIR / "all_results.json"
    with open(raw_results_path, "w") as f:
        json.dump({
            str(ki): {
                str(si): {arm: float(acc) for arm, acc in arms_dict.items()}
                for si, arms_dict in seeds.items()
            }
            for ki, seeds in all_results.items()
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("STAGE C: STATISTICAL ANALYSIS")
    print("=" * 60)

    primary = stage_c_primary_statistic(all_results, winner)
    print(f"Delta_min = {primary['delta_min']:.4f} (control: {primary['delta_min_control']})")
    for c, d in primary["deltas"].items():
        print(f"  {c}: delta = {d['delta']:.4f}")

    bootstrap = key_cluster_bootstrap(all_results, winner)
    print(f"\nBootstrap 95% CI: [{bootstrap['lcb_95']:.4f}, {bootstrap['ucb_95']:.4f}]")
    print(f"Bootstrap mean: {bootstrap['mean']:.4f}")

    sign_flip = paired_sign_flip_test(all_results, winner)
    print(f"\nSign-flip test: p = {sign_flip['p_value']:.6f}")
    print(f"Observed delta_min: {sign_flip['observed_delta_min']:.4f}")
    print(f"Count >= observed: {sign_flip['count_ge']}/256")

    probe_results = {}
    for ki in range(NUM_SEALED_KEYS):
        probe_accs = []
        winner_arm = f"{winner}_correct"
        for si in range(NUM_SEEDS):
            run_name = f"key{ki:02d}_s{INSTALLER_SEEDS[si]}_{winner_arm}"
            summary_path = RESULTS_DIR / run_name / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    s = json.load(f)
                probe_accs.append(s.get("final_probe_acc",
                    s.get("final_probe_correct", 0) / max(s.get("final_probe_total", 1), 1)))
        probe_results[ki] = {
            "probe_acc": float(np.mean(probe_accs)) if probe_accs else 0,
            "n_seeds": len(probe_accs),
        }

    from cti_geometry_admission_verify import verify_forbidden_channels
    forbidden_check = verify_forbidden_channels()
    if not forbidden_check["pass"]:
        print(f"  WARNING: Forbidden channel issues detected!")
        for issue in forbidden_check["issues"][:5]:
            print(f"    {issue['type']}: {issue.get('file', 'N/A')}")

    protocol_checks = {
        "all_teachers_pass": teacher_pass,
        "all_runs_complete": run_count == total_runs,
        "hashes_verified": verify_key_commitment(keys),
        "no_forbidden_info": forbidden_check["pass"],
        "forbidden_issues": forbidden_check["issues"] if not forbidden_check["pass"] else [],
    }

    verdict = stage_c_verdict(primary, bootstrap, sign_flip, probe_results, protocol_checks)

    full_analysis = {
        "winner": winner,
        "primary": primary,
        "bootstrap": bootstrap,
        "sign_flip": sign_flip,
        "probe_results": {str(k): v for k, v in probe_results.items()},
        "protocol_checks": protocol_checks,
        "verdict": verdict,
    }

    with open(RESULTS_DIR / "verdict.json", "w") as f:
        json.dump(full_analysis, f, indent=2)

    print("\n" + "=" * 60)
    print(f"STAGE C VERDICT: {verdict['verdict']}")
    print("=" * 60)

    if verdict["verdict"] == "PASS":
        print(f"  Delta_min = {verdict['delta_min']:.4f} >= 0.20")
        print(f"  LCB_95 = {verdict['lcb_95']:.4f} > 0.10")
        print(f"  p-value = {verdict['p_value']:.6f} <= 0.05")
        print(f"  Claim: {verdict['claim']}")
        print("\nThe frozen artifact carries post-committed transition information")
        print("beyond all five declared controls. Proceed to GRU conditional.")
    elif verdict["verdict"] == "FAIL":
        print("  Reasons:")
        for r in verdict.get("reasons", []):
            print(f"    - {r}")
        print("\nThe experiment does not support the transfer claim.")
    else:
        print("  VOID - protocol integrity compromised:")
        for r in verdict.get("reasons", []):
            print(f"    - {r}")


if __name__ == "__main__":
    main()
