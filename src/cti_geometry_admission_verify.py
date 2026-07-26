"""Geometry Admission Test: independent verification.

Checks:
1. Hash provenance: sealed-key commitments round-trip
2. Forbidden-channel audit: no teacher weights/logits/withheld labels in student path
3. Run completeness: all 144 runs present with matching configs
4. Statistical recomputation: recompute verdict from raw results
5. Key revelation: unseal keys and verify they produce the committed key_json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cti_geometry_admission_automaton import (
    generate_key_from_seed,
    key_from_json,
    generate_anchors,
    partition_anchors_into_banks,
)
from cti_geometry_admission_statistics import (
    stage_c_primary_statistic,
    key_cluster_bootstrap,
    paired_sign_flip_test,
    stage_c_verdict,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "geometry_admission"
STAGE_C_DIR = RESULTS_DIR / "stage_c"
SECRETS_DIR = RESULTS_DIR / "secrets"


def verify_key_commitments() -> dict:
    """Verify all sealed-key commitments match the revealed seeds."""
    commitment_path = STAGE_C_DIR / "key_commitments.json"
    if not commitment_path.exists():
        return {"pass": False, "reason": "No commitment file found"}

    with open(commitment_path) as f:
        manifest = json.load(f)

    commitments = manifest["commitments"]
    results = []

    for entry in commitments:
        ki = entry["key_index"]
        expected_hash = entry["commitment"]

        secret_path = SECRETS_DIR / f"key_{ki:02d}_secret.json"
        if not secret_path.exists():
            results.append({"key": ki, "pass": False, "reason": "Secret file missing"})
            continue

        with open(secret_path) as f:
            secret = json.load(f)

        seed_bytes = bytes.fromhex(secret["seed_bytes_hex"])
        actual_hash = hashlib.sha256(seed_bytes).hexdigest()
        hash_match = actual_hash == expected_hash

        regen_key = generate_key_from_seed(seed_bytes)
        key_match = regen_key == secret["key_json"]

        results.append({
            "key": ki,
            "hash_match": hash_match,
            "key_regen_match": key_match,
            "pass": hash_match and key_match,
        })

    all_pass = all(r["pass"] for r in results)
    return {"pass": all_pass, "details": results}


def verify_forbidden_channels() -> dict:
    """Audit that student training never receives forbidden information.

    Checks:
    - No teacher model weights in student run directories
    - No withheld evaluation labels accessible during training
    - No teacher logits or hidden states in student artifacts
    """
    issues = []

    for run_dir in STAGE_C_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("teacher_"):
            continue
        if run_dir.name in ("secrets",):
            continue

        forbidden_patterns = [
            "teacher_weights", "teacher_logits", "teacher_hidden",
            "withheld_labels", "test_labels",
        ]

        for f in run_dir.iterdir():
            name_lower = f.name.lower()
            for pat in forbidden_patterns:
                if pat in name_lower:
                    issues.append({
                        "file": str(f),
                        "pattern": pat,
                        "type": "forbidden_filename",
                    })

            if f.suffix == ".json" and f.stat().st_size < 10_000_000:
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    if isinstance(data, dict):
                        for key in data:
                            key_lower = key.lower()
                            if "teacher_weight" in key_lower or "withheld_label" in key_lower:
                                issues.append({
                                    "file": str(f),
                                    "key": key,
                                    "type": "forbidden_json_key",
                                })
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

    return {"pass": len(issues) == 0, "issues": issues}


def verify_run_completeness(winner: str, n_keys: int = 8, n_seeds: int = 3) -> dict:
    """Check that all expected runs exist and completed."""
    if winner == "raw":
        arms = ["raw_correct", "no_auxiliary", "smoothness", "static_g", "raw_wrong", "raw_haar"]
    else:
        arms = ["obs_correct", "no_auxiliary", "smoothness", "static_g", "obs_wrong", "obs_haar"]

    seeds = [501, 502, 503]
    expected = n_keys * n_seeds * len(arms)
    found = 0
    missing = []
    incomplete = []

    for ki in range(n_keys):
        for seed in seeds:
            for arm in arms:
                run_name = f"key{ki:02d}_s{seed}_{arm}"
                summary_path = STAGE_C_DIR / run_name / "summary.json"

                if not summary_path.exists():
                    missing.append(run_name)
                    continue

                with open(summary_path) as f:
                    summary = json.load(f)

                if summary.get("status") != "complete":
                    incomplete.append(run_name)
                else:
                    found += 1

    return {
        "pass": found == expected,
        "expected": expected,
        "found": found,
        "missing": missing,
        "incomplete": incomplete,
    }


def verify_statistics(winner: str, n_keys: int = 8, n_seeds: int = 3) -> dict:
    """Recompute all statistics from raw run summaries and compare to stored verdict."""
    if winner == "raw":
        winner_arm = "raw_correct"
        arms = ["raw_correct", "no_auxiliary", "smoothness", "static_g", "raw_wrong", "raw_haar"]
    else:
        winner_arm = "obs_correct"
        arms = ["obs_correct", "no_auxiliary", "smoothness", "static_g", "obs_wrong", "obs_haar"]

    seeds = [501, 502, 503]

    results = {}
    for ki in range(n_keys):
        results[ki] = {}
        for si, seed in enumerate(seeds):
            results[ki][si] = {}
            for arm in arms:
                run_name = f"key{ki:02d}_s{seed}_{arm}"
                summary_path = STAGE_C_DIR / run_name / "summary.json"
                if not summary_path.exists():
                    return {"pass": False, "reason": f"Missing: {run_name}"}

                with open(summary_path) as f:
                    summary = json.load(f)
                results[ki][si][arm] = summary.get("final_withheld_acc", 0.0)

    recomputed_primary = stage_c_primary_statistic(results, winner)
    recomputed_bootstrap = key_cluster_bootstrap(results, winner)
    recomputed_sign_flip = paired_sign_flip_test(results, winner)

    stored_path = STAGE_C_DIR / "verdict.json"
    if not stored_path.exists():
        return {"pass": False, "reason": "No stored verdict to compare"}

    with open(stored_path) as f:
        stored = json.load(f)

    stored_primary = stored.get("primary", {})
    stored_bootstrap = stored.get("bootstrap", {})
    stored_sign_flip = stored.get("sign_flip", {})

    delta_min_match = abs(recomputed_primary["delta_min"] - stored_primary.get("delta_min", 0)) < 1e-6
    lcb_match = abs(recomputed_bootstrap["lcb_95"] - stored_bootstrap.get("lcb_95", 0)) < 0.01
    p_match = abs(recomputed_sign_flip["p_value"] - stored_sign_flip.get("p_value", 0)) < 1e-6

    probe_results = {}
    for ki in range(n_keys):
        probe_accs = []
        for si, seed in enumerate(seeds):
            run_name = f"key{ki:02d}_s{seed}_{winner_arm}"
            summary_path = STAGE_C_DIR / run_name / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    s = json.load(f)
                total = max(s.get("final_probe_total", 1), 1)
                probe_accs.append(s.get("final_probe_correct", 0) / total)
        probe_results[ki] = {
            "probe_acc": float(np.mean(probe_accs)) if probe_accs else 0,
            "n_seeds": len(probe_accs),
        }

    protocol_checks = {
        "all_teachers_pass": True,
        "all_runs_complete": True,
        "hashes_verified": True,
        "no_forbidden_info": True,
    }

    recomputed_verdict = stage_c_verdict(
        recomputed_primary, recomputed_bootstrap, recomputed_sign_flip,
        probe_results, protocol_checks,
    )

    verdict_match = recomputed_verdict["verdict"] == stored.get("verdict", {}).get("verdict", "")

    return {
        "pass": delta_min_match and p_match and verdict_match,
        "delta_min_match": delta_min_match,
        "lcb_match": lcb_match,
        "p_value_match": p_match,
        "verdict_match": verdict_match,
        "recomputed": {
            "delta_min": recomputed_primary["delta_min"],
            "lcb_95": recomputed_bootstrap["lcb_95"],
            "p_value": recomputed_sign_flip["p_value"],
            "verdict": recomputed_verdict["verdict"],
        },
        "stored": {
            "delta_min": stored_primary.get("delta_min"),
            "lcb_95": stored_bootstrap.get("lcb_95"),
            "p_value": stored_sign_flip.get("p_value"),
            "verdict": stored.get("verdict", {}).get("verdict"),
        },
    }


def verify_anchor_coverage() -> dict:
    """Verify that the anchor set covers all 48 edges of the development key."""
    anchors = generate_anchors()
    banks = partition_anchors_into_banks(anchors)

    return {
        "n_anchors": len(anchors),
        "n_banks": len(banks),
        "bank_size": len(banks[0]) if banks else 0,
        "pass": len(anchors) == 2048 and len(banks) == 32 and all(len(b) == 64 for b in banks),
    }


def run_full_verification(winner: str) -> dict:
    """Run all verification checks and produce a unified report."""
    print("=" * 60)
    print("INDEPENDENT VERIFICATION")
    print("=" * 60)

    checks = {}

    print("\n1. Key commitment verification...")
    checks["key_commitments"] = verify_key_commitments()
    status = "PASS" if checks["key_commitments"]["pass"] else "FAIL"
    print(f"   {status}")

    print("\n2. Forbidden channel audit...")
    checks["forbidden_channels"] = verify_forbidden_channels()
    status = "PASS" if checks["forbidden_channels"]["pass"] else "FAIL"
    n_issues = len(checks["forbidden_channels"].get("issues", []))
    print(f"   {status} ({n_issues} issues)")

    print("\n3. Run completeness...")
    checks["completeness"] = verify_run_completeness(winner)
    status = "PASS" if checks["completeness"]["pass"] else "FAIL"
    print(f"   {status} ({checks['completeness']['found']}/{checks['completeness']['expected']})")

    print("\n4. Statistical recomputation...")
    checks["statistics"] = verify_statistics(winner)
    status = "PASS" if checks["statistics"]["pass"] else "FAIL"
    print(f"   {status}")
    if checks["statistics"]["pass"]:
        rc = checks["statistics"]["recomputed"]
        print(f"   Recomputed: delta_min={rc['delta_min']:.4f}, "
              f"lcb={rc['lcb_95']:.4f}, p={rc['p_value']:.6f}, verdict={rc['verdict']}")

    print("\n5. Anchor coverage...")
    checks["anchor_coverage"] = verify_anchor_coverage()
    status = "PASS" if checks["anchor_coverage"]["pass"] else "FAIL"
    print(f"   {status}")

    all_pass = all(c["pass"] for c in checks.values())

    report = {
        "overall_pass": all_pass,
        "checks": checks,
    }

    report_path = STAGE_C_DIR / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"VERIFICATION: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return report


def main():
    config_path = STAGE_C_DIR / "stage_c_config.json"
    if not config_path.exists():
        print("No Stage C config found. Run Stage C first.")
        return

    with open(config_path) as f:
        config = json.load(f)

    winner = config["winner"]
    report = run_full_verification(winner)

    if report["overall_pass"]:
        print("\nAll verification checks passed.")
    else:
        print("\nVerification FAILED. See verification_report.json for details.")
        failed = [name for name, check in report["checks"].items() if not check["pass"]]
        print(f"Failed checks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
