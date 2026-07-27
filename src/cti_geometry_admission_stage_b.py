"""Geometry Admission Test: Stage B structural screen orchestrator.

Protocol: OCF_GAT_STAGE_B_STRUCTURAL_SCREEN_V1

12 runs: 3 seeds (201, 202, 203) x 2 candidates (raw, obs) x 2 conditions
(correct artifact, Haar-matched artifact). Adjacent correct/Haar pairs.

No partner teachers. No deranged teachers. Haar is the structural null.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cti_geometry_admission_automaton import (
    DEVELOPMENT_KEY_JSON,
    key_from_json,
    generate_anchors,
    partition_anchors_into_banks,
    generate_bank_order_permutation,
    generate_calibration_set,
    generate_withheld_eval_set,
    generate_direct_edges,
    hash_eval_set,
    materialize_development_key,
    collate_fn,
    simulate_automaton,
)
from cti_geometry_admission_models import create_transformer_student, count_parameters
from cti_geometry_admission_installer import (
    calibrate_coefficient,
    train_installer_run,
    evaluate_withheld,
    centroid_probe,
    load_teacher_artifacts,
)
from cti_geometry_admission_geometry import (
    generate_haar_rotation_raw,
    generate_haar_rotation_obs,
    apply_haar_to_raw_targets,
    apply_haar_to_obs_targets,
)
from cti_geometry_admission_statistics import (
    stage_b_structural_screen,
    STAGE_B_SEEDS,
    STAGE_B_CANDIDATES,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_b"
STAGE_A_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_a"

KEY_SLOT = 0
COEFFICIENT_SEED = 400
COOLDOWN_S = 60

RUN_MANIFEST = []
for seed in STAGE_B_SEEDS:
    for cand in STAGE_B_CANDIDATES:
        RUN_MANIFEST.append({"seed": seed, "candidate": cand, "condition": "correct"})
        RUN_MANIFEST.append({"seed": seed, "candidate": cand, "condition": "haar"})


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _run_id(run: dict) -> str:
    return f"b_s{run['seed']}_{run['candidate']}_{run['condition']}"


def prepare(device: torch.device) -> dict:
    """Prepare phase: validate Stage A, generate Haar, freeze coefficients."""
    print("=" * 60)
    print("STAGE B PREPARE")
    print("=" * 60)

    key_hash = materialize_development_key(STAGE_A_DIR)
    print(f"  Development key hash: {key_hash}")

    manifest_path = STAGE_A_DIR / "anchor_manifest.json"
    with open(manifest_path) as f:
        anchor_manifest = json.load(f)

    capacity_path = STAGE_A_DIR / "capacity_summary.json"
    with open(capacity_path) as f:
        capacity = json.load(f)
    if not capacity.get("stage_a_pass"):
        raise RuntimeError("Stage A did not pass capacity gates")

    numerical_path = STAGE_A_DIR / "numerical_audit.json"
    with open(numerical_path) as f:
        numerical = json.load(f)
    if not numerical.get("all_pass"):
        raise RuntimeError("Stage A numerical audit did not pass")
    if not numerical.get("repeat_match_raw") or not numerical.get("repeat_match_obs"):
        raise RuntimeError("Stage A repeat match failed")

    raw_manifest_path = STAGE_A_DIR / "raw_trace_manifest.json"
    obs_manifest_path = STAGE_A_DIR / "observable_trace_manifest.json"
    with open(raw_manifest_path) as f:
        raw_manifest = json.load(f)
    with open(obs_manifest_path) as f:
        obs_manifest = json.load(f)

    raw_manifest_hashes = raw_manifest.get("bank_hashes", {})
    obs_manifest_hashes = obs_manifest.get("bank_hashes", {})
    for bank_idx in range(32):
        raw_path = STAGE_A_DIR / f"bank_{bank_idx:03d}" / "raw_trace.json"
        obs_path = STAGE_A_DIR / f"bank_{bank_idx:03d}" / "observable_trace.json"
        if not raw_path.exists() or not obs_path.exists():
            raise RuntimeError(f"Missing artifact for bank {bank_idx}")
        expected_raw = raw_manifest_hashes.get(str(bank_idx))
        if expected_raw:
            actual_raw = _sha256_file(raw_path)
            if actual_raw != expected_raw:
                raise RuntimeError(
                    f"Raw artifact hash mismatch bank {bank_idx}: "
                    f"expected={expected_raw[:16]}, actual={actual_raw[:16]}"
                )
        expected_obs = obs_manifest_hashes.get(str(bank_idx))
        if expected_obs:
            actual_obs = _sha256_file(obs_path)
            if actual_obs != expected_obs:
                raise RuntimeError(
                    f"Observable artifact hash mismatch bank {bank_idx}: "
                    f"expected={expected_obs[:16]}, actual={actual_obs[:16]}"
                )
    print("  Stage A artifacts: validated (32 raw + 32 observable, hashes checked)")

    raw_targets, obs_targets, _ = load_teacher_artifacts(STAGE_A_DIR)
    for bank_idx in range(32):
        for j in range(6):
            r = raw_targets[bank_idx][j]
            if r.dtype != np.float32:
                raise RuntimeError(f"Raw bank {bank_idx} depth {j}: dtype={r.dtype}")
            if not np.isfinite(r).all():
                raise RuntimeError(f"Raw bank {bank_idx} depth {j}: non-finite values")
    for bank_idx in range(32):
        for j in range(6):
            r_obs = obs_targets[bank_idx][j]["R_obs"]
            u_basis = obs_targets[bank_idx][j]["U_basis"]
            if r_obs.dtype != np.float32 or u_basis.dtype != np.float32:
                raise RuntimeError(f"Obs bank {bank_idx} depth {j}: wrong dtype")
            if not np.isfinite(r_obs).all() or not np.isfinite(u_basis).all():
                raise RuntimeError(f"Obs bank {bank_idx} depth {j}: non-finite values")
    print("  Artifact shapes and dtypes: validated")

    print("\n  Generating Haar artifacts...")
    haar_raw = {}
    haar_obs = {}
    haar_Q_hashes = {"raw": {}, "obs": {}}
    for bank_idx in range(32):
        Q_raw = generate_haar_rotation_raw(64, bank_idx)
        teacher_R_list = [raw_targets[bank_idx][j] for j in range(6)]
        haar_raw[bank_idx] = {
            j: r for j, r in enumerate(apply_haar_to_raw_targets(teacher_R_list, Q_raw))
        }
        haar_Q_hashes["raw"][bank_idx] = hashlib.sha256(Q_raw.tobytes()).hexdigest()

        Q_obs = generate_haar_rotation_obs(8, bank_idx)
        teacher_R_obs_list = [obs_targets[bank_idx][j]["R_obs"] for j in range(6)]
        haar_obs[bank_idx] = {}
        rotated_obs = apply_haar_to_obs_targets(teacher_R_obs_list, Q_obs)
        for j in range(6):
            haar_obs[bank_idx][j] = {
                "R_obs": rotated_obs[j],
                "U_basis": obs_targets[bank_idx][j]["U_basis"],
            }
        haar_Q_hashes["obs"][bank_idx] = hashlib.sha256(Q_obs.tobytes()).hexdigest()
    print(f"  Haar artifacts generated: 32 raw + 32 observable")

    print("\n  Freezing step-0 initializations...")
    init_hashes = {}
    init_dir = RESULTS_DIR / "initializations"
    init_dir.mkdir(parents=True, exist_ok=True)
    for seed in STAGE_B_SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = create_transformer_student()
        state = model.state_dict()
        init_path = init_dir / f"init_s{seed}.pt"
        torch.save(state, init_path)
        init_hashes[seed] = _sha256_file(init_path)
        del model
        print(f"    Seed {seed}: {init_hashes[seed][:16]}...")

    print("\n  Calibrating coefficients...")
    key = key_from_json(DEVELOPMENT_KEY_JSON)
    cal_examples = generate_calibration_set(key, key_slot=KEY_SLOT)
    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)

    torch.manual_seed(COEFFICIENT_SEED)
    torch.cuda.manual_seed_all(COEFFICIENT_SEED)
    ref_model = create_transformer_student().to(device)

    correct_artifacts = {"raw": raw_targets, "obs": obs_targets}
    frozen_coefficients = {}
    for cand in STAGE_B_CANDIDATES:
        arm = f"{cand}_correct"
        coeff = calibrate_coefficient(
            ref_model, cal_examples, banks, arm, correct_artifacts, device,
        )
        if not np.isfinite(coeff) or coeff <= 0:
            raise RuntimeError(f"Invalid coefficient for {cand}: {coeff}")
        frozen_coefficients[cand] = float(coeff)
        print(f"    {cand}: lambda = {coeff:.6f}")
    del ref_model
    torch.cuda.empty_cache()

    print("\n  Generating evaluation data...")
    withheld_examples, direct_probes = generate_withheld_eval_set(key, KEY_SLOT, 0)
    direct_edges = generate_direct_edges(key)
    bank_order = generate_bank_order_permutation()

    withheld_hash = hash_eval_set(withheld_examples)
    probe_hash = hash_eval_set(direct_probes)
    cal_hash = hash_eval_set(cal_examples)
    bank_order_hash = hashlib.sha256(json.dumps(bank_order).encode()).hexdigest()

    precommit = {
        "protocol": "OCF_GAT_STAGE_B_STRUCTURAL_SCREEN_V1",
        "manifest": [_run_id(r) for r in RUN_MANIFEST],
        "seeds": STAGE_B_SEEDS,
        "candidates": STAGE_B_CANDIDATES,
        "key_slot": KEY_SLOT,
        "key_hash": key_hash,
        "init_hashes": init_hashes,
        "frozen_coefficients": frozen_coefficients,
        "haar_Q_hashes": haar_Q_hashes,
        "withheld_hash": withheld_hash,
        "probe_hash": probe_hash,
        "calibration_hash": cal_hash,
        "bank_order_hash": bank_order_hash,
        "anchor_manifest_hash": _sha256_file(manifest_path),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "precommit.json", "w") as f:
        json.dump(precommit, f, indent=2)
    print(f"\n  Precommit written: {RESULTS_DIR / 'precommit.json'}")

    return {
        "raw_targets": raw_targets,
        "obs_targets": obs_targets,
        "haar_raw": haar_raw,
        "haar_obs": haar_obs,
        "frozen_coefficients": frozen_coefficients,
        "init_hashes": init_hashes,
        "cal_examples": cal_examples,
        "withheld_examples": withheld_examples,
        "direct_probes": direct_probes,
        "direct_edges": direct_edges,
        "banks": banks,
        "bank_order": bank_order,
        "precommit": precommit,
    }


def install(prep: dict, device: torch.device) -> list[dict]:
    """Install phase: run the 12 installer runs."""
    print("\n" + "=" * 60)
    print("STAGE B INSTALL (12 runs)")
    print("=" * 60)

    raw_targets = prep["raw_targets"]
    obs_targets = prep["obs_targets"]
    haar_raw = prep["haar_raw"]
    haar_obs = prep["haar_obs"]
    coefficients = prep["frozen_coefficients"]
    cal_examples = prep["cal_examples"]
    banks = prep["banks"]
    bank_order = prep["bank_order"]
    init_dir = RESULTS_DIR / "initializations"

    run_summaries = []
    for i, run in enumerate(RUN_MANIFEST):
        run_name = _run_id(run)
        seed = run["seed"]
        cand = run["candidate"]
        condition = run["condition"]
        arm = f"{cand}_correct"
        coeff = coefficients[cand]

        if condition == "correct":
            artifacts = {"raw": raw_targets, "obs": obs_targets}
        else:
            artifacts = {"raw": haar_raw, "obs": haar_obs}

        init_path = init_dir / f"init_s{seed}.pt"

        if i > 0:
            temp_str = ""
            try:
                from cti_geometry_admission_trainer import get_gpu_temp_c
                temp_str = f" (GPU: {get_gpu_temp_c()}C)"
            except Exception:
                pass
            print(f"\n[Cooldown] {COOLDOWN_S}s{temp_str}...")
            time.sleep(COOLDOWN_S)
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\n[{i+1}/12] {run_name}")

        summary = train_installer_run(
            run_config={
                "name": run_name,
                "arm": arm,
                "seed": seed,
                "lr": 5e-4,
                "arch": "transformer",
                "coefficient": coeff,
                "init_checkpoint": str(init_path),
            },
            teacher_artifacts=artifacts,
            calibration_examples=cal_examples,
            withheld_examples=[],
            direct_probes=[],
            anchor_banks=banks,
            bank_order=bank_order,
            output_dir=RESULTS_DIR,
            device=device,
        )
        run_summaries.append({"run": run, "summary": summary})

    return run_summaries


def adjudicate(prep: dict, device: torch.device) -> dict:
    """Adjudicate phase: evaluate all runs and apply structural screen."""
    print("\n" + "=" * 60)
    print("STAGE B ADJUDICATE")
    print("=" * 60)

    key = key_from_json(DEVELOPMENT_KEY_JSON)
    withheld_examples = prep["withheld_examples"]
    direct_probes = prep["direct_probes"]
    cal_examples = prep["cal_examples"]

    withheld_accuracies = {}
    probe_results = {}

    for seed in STAGE_B_SEEDS:
        withheld_accuracies[seed] = {}
        probe_results[seed] = {}

        for cand in STAGE_B_CANDIDATES:
            withheld_accuracies[seed][cand] = {}
            probe_results[seed][cand] = {}

            for condition in ["correct", "haar"]:
                run_name = f"b_s{seed}_{cand}_{condition}"
                run_dir = RESULTS_DIR / run_name
                model_path = run_dir / "model_final.pt"

                if not model_path.exists():
                    raise RuntimeError(f"Missing model: {model_path}")

                model = create_transformer_student().to(device)
                model.load_state_dict(torch.load(
                    model_path, map_location=device, weights_only=True,
                ))

                w_acc = evaluate_withheld(model, withheld_examples, device)
                p_correct, p_total = centroid_probe(
                    model, cal_examples, direct_probes, device,
                )
                p_acc = p_correct / p_total if p_total > 0 else 0.0

                ckpt_hash = _sha256_file(model_path)

                withheld_accuracies[seed][cand][condition] = w_acc
                probe_results[seed][cand][condition] = {
                    "withheld_acc": float(w_acc),
                    "probe_correct": p_correct,
                    "probe_total": p_total,
                    "probe_acc": float(p_acc),
                    "checkpoint_hash": ckpt_hash,
                }

                print(f"  {run_name}: withheld={w_acc:.4f}, probe={p_correct}/{p_total}")
                del model
                torch.cuda.empty_cache()

    precommit_path = RESULTS_DIR / "precommit.json"
    if not precommit_path.exists():
        raise RuntimeError("Missing precommit.json -- prepare() was not run")
    with open(precommit_path) as f:
        precommit = json.load(f)

    protocol_checks = {
        "stage_a_artifacts_valid": True,
        "all_runs_complete": True,
        "initialization_hashes_paired": True,
        "coefficient_frozen": True,
        "artifact_hashes_valid": True,
        "no_forbidden_info": True,
        "all_losses_finite": True,
    }

    init_dir = RESULTS_DIR / "initializations"
    for seed in STAGE_B_SEEDS:
        init_path = init_dir / f"init_s{seed}.pt"
        if not init_path.exists():
            protocol_checks["initialization_hashes_paired"] = False
        else:
            actual_hash = _sha256_file(init_path)
            expected = precommit.get("init_hashes", {}).get(str(seed))
            if actual_hash != expected:
                protocol_checks["initialization_hashes_paired"] = False

    for seed in STAGE_B_SEEDS:
        for cand in STAGE_B_CANDIDATES:
            for condition in ["correct", "haar"]:
                run_name = f"b_s{seed}_{cand}_{condition}"
                run_dir = RESULTS_DIR / run_name
                summary_path = run_dir / "summary.json"
                if not summary_path.exists():
                    protocol_checks["all_runs_complete"] = False
                    continue
                with open(summary_path) as f:
                    summary = json.load(f)
                if summary.get("status") != "complete":
                    protocol_checks["all_runs_complete"] = False
                    continue

                init_hash = summary.get("init_hash", "")
                expected_init = precommit.get("init_hashes", {}).get(str(seed))
                if init_hash != expected_init:
                    protocol_checks["initialization_hashes_paired"] = False

                expected_coeff = precommit.get("frozen_coefficients", {}).get(cand)
                actual_coeff = summary.get("coefficient")
                if expected_coeff is not None and actual_coeff != expected_coeff:
                    protocol_checks["coefficient_frozen"] = False

                log_path = run_dir / "training_log.jsonl"
                if log_path.exists():
                    with open(log_path) as lf:
                        for line in lf:
                            entry = json.loads(line)
                            if not np.isfinite(entry.get("task_loss", 0)):
                                protocol_checks["all_losses_finite"] = False
                            if not np.isfinite(entry.get("aux_loss", 0)):
                                protocol_checks["all_losses_finite"] = False

    screen = stage_b_structural_screen(withheld_accuracies, protocol_checks)

    result = {
        "protocol": "OCF_GAT_STAGE_B_STRUCTURAL_SCREEN_V1",
        "verdict": screen["verdict"],
        "winner": screen.get("winner"),
        "selection": screen.get("selection"),
        "withheld_accuracies": {
            str(seed): {
                cand: {
                    cond: float(withheld_accuracies[seed][cand][cond])
                    for cond in ["correct", "haar"]
                }
                for cand in STAGE_B_CANDIDATES
            }
            for seed in STAGE_B_SEEDS
        },
        "probe_results": {
            str(seed): {
                cand: {
                    cond: probe_results[seed][cand][cond]
                    for cond in ["correct", "haar"]
                }
                for cand in STAGE_B_CANDIDATES
            }
            for seed in STAGE_B_SEEDS
        },
        "frozen_coefficients": prep["frozen_coefficients"],
        "precommit_hash": _sha256_json(prep["precommit"]),
    }

    with open(RESULTS_DIR / "decision.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Verdict: {result['verdict']}")
    if result["winner"]:
        print(f"  Winner: {result['winner']}")
    print(json.dumps(result, indent=2))

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prep = prepare(device)
    install(prep, device)
    result = adjudicate(prep, device)

    if result["verdict"] == "STRUCTURAL_SCREEN_PASS":
        print(f"\nSTAGE B: STRUCTURAL_SCREEN_PASS -- Winner: {result['winner']}")
        print("Proceed to Stage C with sealed CM-CKS pairs.")
    elif result["verdict"] == "STRUCTURAL_SCREEN_FAIL":
        print("\nSTAGE B: STRUCTURAL_SCREEN_FAIL")
        print("No candidate shows structural transfer above Haar null.")
    else:
        print(f"\nSTAGE B: {result['verdict']}")
        print("Protocol violated. Cannot interpret results.")


if __name__ == "__main__":
    main()
