"""Geometry Admission Test: Stage B-P orchestrator (CM-CKS paired development screen).

Reuses Stage A teacher as base. Generates 2 transposition partners.
Trains 2 partner teachers + extracts artifacts. Runs 8 installer runs
to select raw vs observable by changed-edge crossover + stability.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cti_geometry_admission_automaton import (
    DEVELOPMENT_KEY_JSON,
    OP_NAMES,
    NUM_STATES,
    NUM_OPS,
    key_from_json,
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
from cti_geometry_admission_trainer import train_one_run as train_teacher_run
from cti_geometry_admission_extraction import (
    extract_hidden_states,
    extract_raw_trace,
    extract_observable_connection,
    generate_perturbations,
    center_and_normalize,
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
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_b"
STAGE_A_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission" / "stage_a"

DEV_PAIRED_SEED = 401
KEY_SLOT = 0


def edge_index(state: int, op_name: str) -> int:
    """Convert (state, op_name) to 0-47 direct edge index."""
    return state * NUM_OPS + OP_NAMES.index(op_name)


def derive_dev_partners(base_key_json: dict, key_slot: int) -> list[dict]:
    """Derive 2 deterministic transposition partners from the base key."""
    calibrated_op = OP_NAMES[key_slot % NUM_OPS]
    other_ops = [op for op in OP_NAMES if op != calibrated_op]

    partners = []
    for pidx in range(2):
        withheld_op = other_ops[pidx % len(other_ops)]
        h = hashlib.sha256(
            f"GAT_STAGE_B_DEV_PARTNER_{pidx}_{withheld_op}".encode()
        ).digest()
        u = int(h[0]) % NUM_STATES
        v = (u + 1 + int(h[1]) % (NUM_STATES - 1)) % NUM_STATES
        partner_key, metadata = paired_key_from_transposition(
            base_key_json, calibrated_op, withheld_op, u, v,
        )
        partners.append({
            "partner_index": pidx,
            "partner_key_json": partner_key,
            "metadata": metadata,
        })
    return partners


def extract_teacher_artifacts(key_json: dict, teacher_dir: Path, device: torch.device) -> dict:
    """Extract raw R and observable R artifacts from a trained teacher."""
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


def load_student_and_get_logits(
    run_dir: Path, direct_edges: list[dict], device: torch.device,
) -> np.ndarray:
    """Load a trained student and return direct-edge logits (48, 12)."""
    model = create_transformer_student().to(device)
    model.load_state_dict(torch.load(
        run_dir / "model_final.pt", map_location=device, weights_only=True,
    ))
    logits = evaluate_direct_edge_logits(model, direct_edges, device)
    del model
    torch.cuda.empty_cache()
    return logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    key_path = STAGE_A_DIR / "development_key.json"
    if not key_path.exists():
        print("ERROR: Stage A development key not found. Run Stage A first.")
        return
    with open(key_path) as f:
        base_key_json = json.load(f)

    teacher_dir = STAGE_A_DIR / "teacher"
    summary_path = teacher_dir / "summary.json"
    if not summary_path.exists():
        print("ERROR: Stage A teacher not trained. Run Stage A first.")
        return
    with open(summary_path) as f:
        teacher_summary = json.load(f)
    if teacher_summary.get("status") != "complete":
        print("ERROR: Stage A teacher training not complete.")
        return
    base_gate_ok = (
        teacher_summary.get("final_in_range", 0) >= 0.995
        and teacher_summary.get("final_extrapolation", 0) >= 0.990
        and teacher_summary.get("final_direct_edges", 0) == 48
    )
    if not base_gate_ok:
        print("ERROR: Stage A teacher below capacity gates.")
        print(f"  in_range={teacher_summary.get('final_in_range')}, "
              f"extrap={teacher_summary.get('final_extrapolation')}, "
              f"edges={teacher_summary.get('final_direct_edges')}/48")
        return

    base_key = key_from_json(base_key_json)

    print("\n" + "=" * 60)
    print("STAGE B-P: GENERATING DEVELOPMENT PARTNERS")
    print("=" * 60)
    partners = derive_dev_partners(base_key_json, KEY_SLOT)
    for p in partners:
        m = p["metadata"]
        print(f"  Partner {p['partner_index']}: "
              f"withheld={m['withheld_op']}, "
              f"transposition=({m['transposition'][0]},{m['transposition'][1]}), "
              f"changed edges: {[edge_index(e['state'], e['op']) for e in m['changed_edges']]}")

    with open(RESULTS_DIR / "dev_partners.json", "w") as f:
        json.dump(partners, f, indent=2)

    print("\n" + "=" * 60)
    print("STAGE B-P: TRAINING PARTNER + REPLAY TEACHERS")
    print("=" * 60)

    import cti_geometry_admission_trainer as trainer_mod
    original_results_dir = trainer_mod.RESULTS_DIR
    trainer_mod.RESULTS_DIR = RESULTS_DIR

    for p in partners:
        pidx = p["partner_index"]
        pkey = key_from_json(p["partner_key_json"])
        eval_sets = generate_all_eval_sets(pkey, seed=42)

        run_cfg = {
            "name": f"partner_{pidx}_teacher",
            "arch": "teacher",
            "seed": 101,
            "lr": 3e-4,
        }
        print(f"\nTraining partner {pidx} teacher...")
        summary = train_teacher_run(run_cfg, pkey, eval_sets, device, allow_resume=True)

        gate_ok = (
            summary["final_in_range"] >= 0.995
            and summary["final_extrapolation"] >= 0.990
            and summary["final_direct_edges"] == 48
        )
        if not gate_ok:
            print(f"  ERROR: Partner {pidx} teacher below capacity!")
            print(f"    in_range={summary['final_in_range']:.4f} "
                  f"extrap={summary['final_extrapolation']:.4f} "
                  f"edges={summary['final_direct_edges']}/48")
            print("Stage B-P: VOID -- Partner teacher capacity failure.")
            return

    print("\nTraining same-key replay teacher...")
    replay_eval = generate_all_eval_sets(base_key, seed=42)
    replay_summary = train_teacher_run(
        {"name": "replay_teacher", "arch": "teacher", "seed": 101, "lr": 3e-4},
        base_key, replay_eval, device, allow_resume=True,
    )
    print(f"  Replay teacher: in_range={replay_summary['final_in_range']:.4f} "
          f"extrap={replay_summary['final_extrapolation']:.4f}")

    trainer_mod.RESULTS_DIR = original_results_dir

    print("\n" + "=" * 60)
    print("STAGE B-P: EXTRACTING ARTIFACTS")
    print("=" * 60)

    print("  Extracting base teacher artifacts...")
    base_artifacts = extract_teacher_artifacts(base_key_json, teacher_dir, device)

    partner_artifacts = []
    for p in partners:
        pidx = p["partner_index"]
        print(f"  Extracting partner {pidx} artifacts...")
        part_dir = RESULTS_DIR / f"partner_{pidx}_teacher"
        art = extract_teacher_artifacts(p["partner_key_json"], part_dir, device)
        partner_artifacts.append(art)

    print("  Extracting replay teacher artifacts...")
    replay_artifacts = extract_teacher_artifacts(
        base_key_json, RESULTS_DIR / "replay_teacher", device,
    )

    print("\n" + "=" * 60)
    print("STAGE B-P: COEFFICIENT FREEZING")
    print("=" * 60)

    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)
    bank_order = generate_bank_order_permutation()

    cal_base = generate_calibration_set(base_key, key_slot=KEY_SLOT)

    torch.manual_seed(400)
    torch.cuda.manual_seed_all(400)
    ref_model = create_transformer_student().to(device)
    frozen_coefficients = {}
    for arm in ["raw_correct", "obs_correct"]:
        frozen_coefficients[arm] = calibrate_coefficient(
            ref_model, cal_base, banks, arm,
            {"raw": base_artifacts["raw"], "obs": base_artifacts["obs"]},
            device,
        )
        print(f"  Frozen coefficient for {arm}: {frozen_coefficients[arm]:.6f}")
    del ref_model
    torch.cuda.empty_cache()

    with open(RESULTS_DIR / "frozen_coefficients.json", "w") as f:
        json.dump(frozen_coefficients, f, indent=2)

    print("\n" + "=" * 60)
    print("STAGE B-P: INSTALLER RUNS (8 total)")
    print("=" * 60)

    base_direct = generate_direct_edges(base_key)
    run_count = 0
    partner_evaluations = []

    for pidx, p in enumerate(partners):
        pkey = key_from_json(p["partner_key_json"])
        partner_direct = generate_direct_edges(pkey)
        changed = p["metadata"]["changed_edges"]
        changed_indices = [edge_index(e["state"], e["op"]) for e in changed]
        student_seed = DEV_PAIRED_SEED

        cal_partner = generate_calibration_set(pkey, key_slot=KEY_SLOT)

        candidate_results = {}

        for candidate in ["raw", "obs"]:
            arm = f"{candidate}_correct"
            coeff = frozen_coefficients[arm]

            run_a_name = f"p{pidx}_{candidate}_A_s{student_seed}"
            run_b_name = f"p{pidx}_{candidate}_B_s{student_seed}"

            print(f"\n[{run_count+1}/8] {run_a_name}")
            run_count += 1
            train_installer_run(
                {"name": run_a_name, "arm": arm, "seed": student_seed,
                 "lr": 5e-4, "arch": "transformer", "coefficient": coeff},
                {"raw": base_artifacts["raw"], "obs": base_artifacts["obs"]},
                cal_base, [], base_direct, banks, bank_order, RESULTS_DIR, device,
            )

            print(f"\n[{run_count+1}/8] {run_b_name}")
            run_count += 1
            train_installer_run(
                {"name": run_b_name, "arm": arm, "seed": student_seed,
                 "lr": 5e-4, "arch": "transformer", "coefficient": coeff},
                {"raw": partner_artifacts[pidx]["raw"],
                 "obs": partner_artifacts[pidx]["obs"]},
                cal_partner, [], partner_direct, banks, bank_order, RESULTS_DIR, device,
            )

            logits_a = load_student_and_get_logits(
                RESULTS_DIR / run_a_name, base_direct, device,
            )
            logits_b = load_student_and_get_logits(
                RESULTS_DIR / run_b_name, partner_direct, device,
            )

            crossover = counterfactual_edge_crossover(logits_a, logits_b, changed)
            stability = unchanged_edge_stability(logits_a, logits_b, changed_indices)
            pair = cm_pair_success(crossover, stability)

            candidate_results[candidate] = {
                "crossover": crossover,
                "stability": stability,
                "pair_success": pair,
            }

            print(f"  {candidate}: crossover={crossover['all_crossover']}, "
                  f"mean_d={crossover['mean_d']:.4f}, "
                  f"stable={stability['stable']}, "
                  f"success={pair['success']}")

        partner_evaluations.append(candidate_results)

    print("\n" + "=" * 60)
    print("STAGE B-P: SAME-KEY REPLAY NOISE CEILING")
    print("=" * 60)

    replay_seed = 403
    replay_drift = {}
    for candidate in ["raw", "obs"]:
        arm = f"{candidate}_correct"
        coeff = frozen_coefficients[arm]

        orig_name = f"replay_{candidate}_orig_s{replay_seed}"
        repl_name = f"replay_{candidate}_repl_s{replay_seed}"

        print(f"\n  {orig_name}")
        train_installer_run(
            {"name": orig_name, "arm": arm, "seed": replay_seed,
             "lr": 5e-4, "arch": "transformer", "coefficient": coeff},
            {"raw": base_artifacts["raw"], "obs": base_artifacts["obs"]},
            cal_base, [], base_direct, banks, bank_order, RESULTS_DIR, device,
        )

        print(f"  {repl_name}")
        train_installer_run(
            {"name": repl_name, "arm": arm, "seed": replay_seed,
             "lr": 5e-4, "arch": "transformer", "coefficient": coeff},
            {"raw": replay_artifacts["raw"], "obs": replay_artifacts["obs"]},
            cal_base, [], base_direct, banks, bank_order, RESULTS_DIR, device,
        )

        logits_orig = load_student_and_get_logits(
            RESULTS_DIR / orig_name, base_direct, device,
        )
        logits_repl = load_student_and_get_logits(
            RESULTS_DIR / repl_name, base_direct, device,
        )

        replay_stability = unchanged_edge_stability(
            logits_orig, logits_repl, [], drift_ceiling=1.0,
        )
        replay_drift[candidate] = {
            "mean_tv": replay_stability["mean_tv"],
            "max_tv": replay_stability["max_tv"],
            "flip_count": replay_stability["flip_count"],
        }
        print(f"  {candidate} replay drift: mean_tv={replay_stability['mean_tv']:.4f}, "
              f"max_tv={replay_stability['max_tv']:.4f}, flips={replay_stability['flip_count']}")

    print("\n" + "=" * 60)
    print("STAGE B-P: CANDIDATE SELECTION")
    print("=" * 60)

    raw_score = sum(
        1 for ev in partner_evaluations if ev["raw"]["pair_success"]["success"]
    )
    obs_score = sum(
        1 for ev in partner_evaluations if ev["obs"]["pair_success"]["success"]
    )
    raw_mean_d = np.mean([ev["raw"]["crossover"]["mean_d"] for ev in partner_evaluations])
    obs_mean_d = np.mean([ev["obs"]["crossover"]["mean_d"] for ev in partner_evaluations])

    replay_noise_exceeds = {}
    for candidate in ["raw", "obs"]:
        cand_mean_d = float(np.mean([ev[candidate]["crossover"]["mean_d"]
                                     for ev in partner_evaluations]))
        noise_tv = replay_drift[candidate]["mean_tv"]
        exceeds = cand_mean_d > 2.0 * noise_tv if noise_tv > 0 else cand_mean_d > 0
        replay_noise_exceeds[candidate] = exceeds
        print(f"  {candidate}: CM mean_d={cand_mean_d:.4f}, replay_noise={noise_tv:.4f}, "
              f"exceeds_2x={exceeds}")

    print(f"  Raw:  {raw_score}/2 partners pass, mean_d={raw_mean_d:.4f}")
    print(f"  Obs:  {obs_score}/2 partners pass, mean_d={obs_mean_d:.4f}")

    raw_viable = raw_score > 0 and replay_noise_exceeds.get("raw", False)
    obs_viable = obs_score > 0 and replay_noise_exceeds.get("obs", False)

    if raw_viable and obs_viable:
        if raw_score > obs_score:
            winner = "raw"
        elif obs_score > raw_score:
            winner = "obs"
        elif raw_mean_d >= obs_mean_d:
            winner = "raw"
        else:
            winner = "obs"
    elif raw_viable:
        winner = "raw"
    elif obs_viable:
        winner = "obs"
    else:
        winner = None

    neither_pass = winner is None

    decision = {
        "winner": winner,
        "verdict": "FAIL" if neither_pass else "PASS",
        "raw_score": raw_score,
        "obs_score": obs_score,
        "raw_mean_d": float(raw_mean_d),
        "obs_mean_d": float(obs_mean_d),
        "frozen_coefficients": frozen_coefficients,
        "replay_noise_ceiling": replay_drift,
        "replay_noise_exceeds": replay_noise_exceeds,
        "partner_evaluations": [
            {cand: {
                "crossover_success": ev[cand]["crossover"]["all_crossover"],
                "mean_d": ev[cand]["crossover"]["mean_d"],
                "stable": ev[cand]["stability"]["stable"],
                "mean_tv": ev[cand]["stability"]["mean_tv"],
                "pair_success": ev[cand]["pair_success"]["success"],
            } for cand in ["raw", "obs"]}
            for ev in partner_evaluations
        ],
    }

    if neither_pass:
        fail_reasons = []
        if raw_score == 0 and obs_score == 0:
            fail_reasons.append("Neither candidate produced crossover on development partners")
        if not replay_noise_exceeds.get("raw", False) and raw_score > 0:
            fail_reasons.append("Raw CM effect within replay noise ceiling")
        if not replay_noise_exceeds.get("obs", False) and obs_score > 0:
            fail_reasons.append("Obs CM effect within replay noise ceiling")
        decision["reason"] = "; ".join(fail_reasons) if fail_reasons else "No viable candidate"

    with open(RESULTS_DIR / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    print(json.dumps(decision, indent=2))

    if decision["verdict"] == "PASS":
        print(f"\nSTAGE B-P: PASS -- Winner: {decision['winner']}")
        print("Proceed to Stage C-I with sealed CM-CKS pairs.")
    else:
        print("\nSTAGE B-P: FAIL -- No candidate shows crossover effect.")
        print("GAT experiment terminates.")


if __name__ == "__main__":
    main()
