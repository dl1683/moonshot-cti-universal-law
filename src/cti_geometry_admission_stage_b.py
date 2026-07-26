"""Geometry Admission Test: Stage B orchestrator.

18-run candidate screen: 2 dev keys x 9 arms.
Selects winner (raw R vs observable R) by withheld accuracy advantage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from cti_geometry_admission_automaton import (
    key_from_json,
    generate_all_eval_sets,
    generate_calibration_set,
    generate_withheld_eval_set,
    generate_anchors,
    partition_anchors_into_banks,
    generate_bank_order_permutation,
    generate_stage_b_dev_keys,
    hash_eval_set,
    collate_fn,
)
from cti_geometry_admission_models import create_teacher, count_parameters
from cti_geometry_admission_trainer import train_one_run as train_teacher_run
from cti_geometry_admission_extraction import (
    extract_hidden_states,
    extract_raw_trace,
    extract_observable_connection,
    generate_perturbations,
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
from cti_geometry_admission_statistics import stage_b_selection

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_b"
STAGE_A_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_a"


def train_teacher_for_key(key_json: dict, key_idx: int, device: torch.device) -> dict:
    """Train a teacher on the given key using Stage A protocol."""
    key = key_from_json(key_json)
    eval_sets = generate_all_eval_sets(key, seed=42)

    run_cfg = {
        "name": f"teacher_key{key_idx}",
        "arch": "teacher",
        "seed": 101,
        "lr": 3e-4,
    }

    from cti_geometry_admission_trainer import RESULTS_DIR as TRAINER_RESULTS_DIR
    import cti_geometry_admission_trainer as trainer_mod
    original_dir = trainer_mod.RESULTS_DIR
    trainer_mod.RESULTS_DIR = RESULTS_DIR

    summary = train_teacher_run(run_cfg, key, eval_sets, device)

    trainer_mod.RESULTS_DIR = original_dir
    return summary


def extract_teacher_artifacts(
    key_json: dict,
    key_idx: int,
    device: torch.device,
) -> dict:
    """Extract raw R, observable R, and static G artifacts from trained teacher."""
    key = key_from_json(key_json)

    teacher_dir = RESULTS_DIR / f"teacher_key{key_idx}"
    teacher_path = teacher_dir / "model_final.pt"
    teacher = create_teacher().to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher.eval()

    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)

    artifacts = {"raw": {}, "obs": {}, "static_g": {}}

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

        from cti_geometry_admission_extraction import center_and_normalize
        artifacts["static_g"][bank_idx] = {}
        for tick_idx in range(len(TEACHER_DEPTH_LAYERS)):
            X = center_and_normalize(ticks[tick_idx])
            G = (X @ X.T).astype(np.float32)
            artifacts["static_g"][bank_idx][tick_idx] = G

    return artifacts


def build_control_artifacts(
    artifacts_key0: dict,
    artifacts_key1: dict,
) -> tuple[dict, dict]:
    """Build wrong-key and Haar-matched control artifacts for both keys."""
    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)
    n_banks = len(banks)
    bank_size = len(banks[0])

    key0_controls = {
        "raw_wrong": artifacts_key1["raw"],
        "obs_wrong": {},
        "raw_haar": {},
        "obs_haar": {},
    }
    key1_controls = {
        "raw_wrong": artifacts_key0["raw"],
        "obs_wrong": {},
        "raw_haar": {},
        "obs_haar": {},
    }

    for bank_idx in range(n_banks):
        key0_controls["obs_wrong"][bank_idx] = artifacts_key1["obs"][bank_idx]
        key1_controls["obs_wrong"][bank_idx] = artifacts_key0["obs"][bank_idx]

        Q_raw = generate_haar_rotation_raw(bank_size, bank_idx)
        Q_obs = generate_haar_rotation_obs(8, bank_idx)

        raw_0 = {j: artifacts_key0["raw"][bank_idx][j] for j in range(6)}
        raw_1 = {j: artifacts_key1["raw"][bank_idx][j] for j in range(6)}
        haar_raw_0 = apply_haar_to_raw_targets(
            [raw_0[j] for j in range(6)], Q_raw,
        )
        haar_raw_1 = apply_haar_to_raw_targets(
            [raw_1[j] for j in range(6)], Q_raw,
        )
        key0_controls["raw_haar"][bank_idx] = {j: haar_raw_0[j] for j in range(6)}
        key1_controls["raw_haar"][bank_idx] = {j: haar_raw_1[j] for j in range(6)}

        obs_0_list = [artifacts_key0["obs"][bank_idx][j]["R_obs"] for j in range(6)]
        obs_1_list = [artifacts_key1["obs"][bank_idx][j]["R_obs"] for j in range(6)]
        haar_obs_0 = apply_haar_to_obs_targets(obs_0_list, Q_obs)
        haar_obs_1 = apply_haar_to_obs_targets(obs_1_list, Q_obs)
        key0_controls["obs_haar"][bank_idx] = {
            j: {
                "R_obs": haar_obs_0[j],
                "U_basis": artifacts_key0["obs"][bank_idx][j]["U_basis"],
            } for j in range(6)
        }
        key1_controls["obs_haar"][bank_idx] = {
            j: {
                "R_obs": haar_obs_1[j],
                "U_basis": artifacts_key1["obs"][bank_idx][j]["U_basis"],
            } for j in range(6)
        }

    return key0_controls, key1_controls


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dev_keys = generate_stage_b_dev_keys()
    with open(RESULTS_DIR / "development_keys.json", "w") as f:
        json.dump([{"slot": k["slot"], "derivation": k["derivation"],
                     "seed_hash": k["seed_hash"], "key_json": k["key_json"]}
                    for k in dev_keys], f, indent=2)

    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)
    bank_order = generate_bank_order_permutation()

    print("\n" + "=" * 60)
    print("STAGE B: TEACHER TRAINING")
    print("=" * 60)
    for ki, kd in enumerate(dev_keys):
        print(f"\nTraining teacher for key {ki}...")
        train_teacher_for_key(kd["key_json"], ki, device)

    print("\n" + "=" * 60)
    print("STAGE B: ARTIFACT EXTRACTION")
    print("=" * 60)
    artifacts = []
    for ki, kd in enumerate(dev_keys):
        print(f"\nExtracting artifacts for key {ki}...")
        art = extract_teacher_artifacts(kd["key_json"], ki, device)
        artifacts.append(art)

    print("\n" + "=" * 60)
    print("STAGE B: CONTROL ARTIFACTS")
    print("=" * 60)
    key0_controls, key1_controls = build_control_artifacts(artifacts[0], artifacts[1])
    controls = [key0_controls, key1_controls]

    print("\n" + "=" * 60)
    print("STAGE B: INSTALLER RUNS (18 total)")
    print("=" * 60)

    all_results = {0: {}, 1: {}}

    for ki, kd in enumerate(dev_keys):
        key = key_from_json(kd["key_json"])
        cal = generate_calibration_set(key, key_slot=ki)
        withheld, probes = generate_withheld_eval_set(key, key_slot=ki, key_index=ki)

        combined_artifacts = {
            "raw": artifacts[ki]["raw"],
            "obs": artifacts[ki]["obs"],
            "static_g": artifacts[ki]["static_g"],
            "raw_wrong": controls[ki]["raw_wrong"],
            "obs_wrong": controls[ki]["obs_wrong"],
            "raw_haar": controls[ki]["raw_haar"],
            "obs_haar": controls[ki]["obs_haar"],
        }

        for arm in ARMS:
            run_name = f"key{ki}_{arm}_s401"
            print(f"\nRunning: {run_name}")

            from cti_geometry_admission_installer import create_transformer_student
            torch.manual_seed(400)
            torch.cuda.manual_seed_all(400)
            ref_model = create_transformer_student().to(device)

            coeff = calibrate_coefficient(
                ref_model, cal, banks, arm, combined_artifacts, device,
            )
            del ref_model
            print(f"  Coefficient for {arm}: {coeff:.6f}")

            run_config = {
                "name": run_name,
                "arm": arm,
                "seed": 401,
                "lr": 5e-4,
                "arch": "transformer",
                "coefficient": coeff,
            }

            summary = train_installer_run(
                run_config, combined_artifacts, cal, withheld, probes,
                banks, bank_order, RESULTS_DIR, device,
            )

            all_results[ki][arm] = summary["final_withheld_acc"]
            print(f"  Withheld acc: {summary['final_withheld_acc']:.4f}")
            print(f"  Probe: {summary['final_probe_correct']}/{summary['final_probe_total']}")

    print("\n" + "=" * 60)
    print("STAGE B: SELECTION")
    print("=" * 60)

    decision = stage_b_selection(all_results)
    decision["all_results"] = {
        str(k): {arm: float(acc) for arm, acc in arms.items()}
        for k, arms in all_results.items()
    }

    with open(RESULTS_DIR / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    print(json.dumps(decision, indent=2))

    if decision["verdict"] == "PASS":
        print(f"\nSTAGE B: PASS — Winner: {decision['winner']}")
        print("Proceed to Stage C with sealed keys.")
    else:
        print(f"\nSTAGE B: FAIL — {decision.get('reason', 'No eligible candidate')}")
        print("GAT experiment terminates.")


if __name__ == "__main__":
    main()
