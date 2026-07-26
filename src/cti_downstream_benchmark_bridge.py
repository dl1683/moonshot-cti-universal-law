"""Preregister the CTI downstream public-benchmark bridge.

This script does not run public LLM benchmarks. It freezes the external panel,
model/family split rules, success thresholds, and lm-eval command templates
needed to test whether CTI geometry predicts public downstream benchmark
behavior rather than only split-safe representation quality.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "cti_downstream_benchmark_bridge_preregistration_20260609.json"


def load_result(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def preflight() -> dict[str, Any]:
    packages = {
        name: module_available(name)
        for name in ("lm_eval", "transformers", "torch", "datasets", "scipy", "sklearn")
    }
    lm_eval_cli = shutil.which("lm-eval")
    cli_smoke = cli_smoke_check(lm_eval_cli)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "lm_eval_cli": lm_eval_cli,
        "lm_eval_cli_smoke": cli_smoke,
        "packages": packages,
        "can_run_public_panel_here": bool(
            lm_eval_cli and packages["torch"] and cli_smoke["returncode"] == 0
        ),
        "reason_if_false": (
            None
            if lm_eval_cli and packages["torch"] and cli_smoke["returncode"] == 0
            else "Public panel not runnable in this environment without a passing lm-eval CLI smoke test and model backend."
        ),
    }


def cli_smoke_check(lm_eval_cli: str | None) -> dict[str, Any]:
    if not lm_eval_cli:
        return {
            "command": "lm-eval --help",
            "returncode": None,
            "stdout_head": "",
            "stderr_head": "lm-eval executable not found",
        }
    completed = subprocess.run(
        [lm_eval_cli, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    blocker = None
    if "HybridCache" in completed.stderr and "transformers" in completed.stderr:
        blocker = "peft_transformers_HybridCache_import_mismatch"
    return {
        "command": f"{lm_eval_cli} --help",
        "returncode": completed.returncode,
        "detected_blocker": blocker,
        "stdout_head": completed.stdout[:600],
        "stderr_head": completed.stderr[:2400],
    }


def external_task_panel() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "ifeval",
            "official_name": "IFEval",
            "shots": 0,
            "measure": [
                "inst_level_strict_acc,none",
                "prompt_level_strict_acc,none",
            ],
            "capability": "instruction following",
        },
        {
            "task_id": "bbh",
            "official_name": "Big Bench Hard",
            "shots": 3,
            "measure": ["acc_norm,none"],
            "capability": "multi-step reasoning and hard objective tasks",
        },
        {
            "task_id": "math_lvl_5",
            "official_name": "MATH Level 5",
            "shots": 4,
            "measure": ["exact_match,none"],
            "capability": "competition math",
        },
        {
            "task_id": "gpqa",
            "official_name": "GPQA",
            "shots": 0,
            "measure": ["acc_norm,none"],
            "capability": "graduate-level science QA",
            "contamination_guard": "GPQA text access is gated; do not copy examples into CTI artifacts.",
        },
        {
            "task_id": "musr",
            "official_name": "MuSR",
            "shots": 0,
            "measure": ["acc_norm,none"],
            "capability": "long-context multi-step reasoning",
        },
        {
            "task_id": "mmlu_pro",
            "official_name": "MMLU-PRO",
            "shots": 5,
            "measure": ["acc,none"],
            "capability": "hard multitask knowledge with 10 choices",
        },
    ]


def model_family_blocks() -> list[dict[str, Any]]:
    return [
        {
            "family": "pythia_dense",
            "role": "source-like dense scaling baseline",
            "candidate_model_ids": [
                "EleutherAI/pythia-160m",
                "EleutherAI/pythia-410m",
                "EleutherAI/pythia-1b",
                "EleutherAI/pythia-1.4b",
            ],
        },
        {
            "family": "qwen_dense",
            "role": "modern dense decoder family",
            "candidate_model_ids": [
                "Qwen/Qwen2.5-0.5B",
                "Qwen/Qwen3-0.6B-Base",
                "Qwen/Qwen3-1.7B-Base",
            ],
        },
        {
            "family": "olmo_tinyllama_dense",
            "role": "non-Pythia dense training mixtures",
            "candidate_model_ids": [
                "allenai/OLMo-1B-hf",
                "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            ],
        },
        {
            "family": "state_space_or_liquid_hybrid",
            "role": "architecture-shift pressure block",
            "candidate_model_ids": [
                "state-spaces/mamba-130m-hf",
                "LiquidAI/LFM2-350M",
            ],
        },
        {
            "family": "moe_or_hybrid_attention",
            "role": "known CTI scope-stress block",
            "candidate_model_ids": [
                "Zyphra/Zamba2-1.2B",
                "ibm-granite/granite-3.0-1b-a400m-base",
            ],
        },
    ]


def command_templates() -> dict[str, Any]:
    output_root = "results/downstream_bridge/<run_id>/<model_slug>"
    model_args = "pretrained=<model_id>,revision=<revision>,dtype=auto"
    return {
        "harness_task_listing": "lm-eval ls groups",
        "leaderboard_group_run": (
            "lm-eval run --model hf "
            f"--model_args {model_args} "
            "--tasks leaderboard "
            "--batch_size auto "
            f"--output_path {output_root}"
        ),
        "hf_docs_compatibility_run": (
            "lm-eval "
            f'--model_args="{model_args}" '
            "--tasks=leaderboard "
            "--batch_size=auto "
            f"--output_path={output_root}"
        ),
        "task_level_template": (
            "lm-eval run --model hf "
            f"--model_args {model_args} "
            "--tasks <task_id> "
            "--batch_size auto "
            f"--output_path {output_root}/<task_id>"
        ),
        "instruction_model_addendum": (
            "For instruction-tuned models only, add --apply_chat_template and "
            "record whether fewshot_as_multiturn was used."
        ),
    }


def build_bridge() -> dict[str, Any]:
    h3 = load_result("cti_downstream_h3_n9.json")
    contract = load_result("cti_invariance_contract_20260609.json")
    h3_stats = h3["H3_extended"]
    downstream_claim = contract["claim_ladder"][4]

    internal_signal = {
        "source": "results/cti_downstream_h3_n9.json",
        "dataset": "Banking77",
        "n_models": h3_stats["n_models"],
        "spearman_rho": h3_stats["spearman_rho"],
        "p_value": h3_stats["p_value"],
        "status": "internal_downstream_signal_only",
        "scope_guard": (
            "This is a supervised retrieval/ranking signal on Banking77; it is not "
            "evidence that CTI predicts public LLM benchmark aggregates."
        ),
    }

    return {
        "experiment": "cti_downstream_public_benchmark_bridge_preregistration",
        "date": "2026-06-09",
        "status": "preregistered_not_run",
        "question": (
            "Can CTI geometry, computed before public benchmark outputs are read, "
            "predict current public LLM downstream benchmark performance under "
            "held-out model families?"
        ),
        "claim_boundary": {
            "invariance_contract_status": downstream_claim["status"],
            "current_verified_bridge": downstream_claim["strict_metric"][
                "current_verified_bridge"
            ],
            "public_claim_allowed_now": False,
            "allowed_sentence": (
                "CTI has an internal downstream signal and a preregistered public "
                "benchmark bridge; public benchmark prediction remains unproven "
                "until the bridge is run and passes."
            ),
            "forbidden_sentence": (
                "CTI predicts MMLU/GPQA/Open LLM Leaderboard performance."
            ),
        },
        "source_grounding": {
            "huggingface_open_llm_leaderboard_about": "https://huggingface.co/docs/leaderboards/main/open_llm_leaderboard/about",
            "eleutherai_lm_eval_interface": "https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md",
            "eleutherai_leaderboard_task_group": "https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/leaderboard",
            "apple_downstream_metrics": "https://machinelearning.apple.com/research/downstream-metrics",
            "downstream_scaling_arxiv": "https://arxiv.org/abs/2512.08894",
        },
        "internal_signal": internal_signal,
        "external_panel": {
            "name": "Open LLM Leaderboard v2 style six-task panel",
            "task_group": "leaderboard",
            "tasks": external_task_panel(),
            "normalization_rule": (
                "Use the exact official leaderboard-normalized aggregate if available; "
                "otherwise record raw task metrics and use a predeclared macro average "
                "after min/max normalization computed only on the frozen evaluated panel."
            ),
        },
        "geometry_protocol": {
            "probe_suite": [
                "banking77",
                "agnews",
                "dbpedia",
                "20newsgroups",
                "clinc_oos",
            ],
            "features_frozen_before_benchmark_outputs": [
                "final_layer_kappa_nearest_macro",
                "ref_local_safe_rate_macro",
                "ref_local_margin_log_q25_macro",
                "ref_dist_ratio_macro",
                "ref_local_lid_proxy_mean_macro",
                "log_parameter_count",
            ],
            "primary_cti_score": (
                "Fit only on non-held-out families: normalized linear panel over "
                "logK, kappa_nearest, local safe rate, lower-tail margin, and density. "
                "For the no-fit diagnostic, rank models by final_layer_kappa_nearest_macro."
            ),
            "forbidden_inputs": [
                "public benchmark accuracy",
                "public benchmark leaderboard aggregate",
                "benchmark labels or answers used as CTI geometry probes",
                "post-hoc task dropping after seeing results",
            ],
        },
        "model_family_blocks": model_family_blocks(),
        "primary_success_criteria": {
            "minimum_complete_models": 12,
            "minimum_complete_families": 4,
            "total_panel_spearman_rho": ">= 0.60",
            "total_panel_permutation_p": "<= 0.05",
            "leave_one_family_out_macro_spearman": ">= 0.35",
            "leave_one_family_out_negative_folds": "0 when held-out family has at least 3 models",
            "partial_spearman_after_log_params": ">= 0.30",
            "baseline_guard": (
                "Primary CTI score must beat log-parameter-count and generic spectral "
                "geometry baselines by at least +0.10 Spearman on the complete panel."
            ),
        },
        "failure_rule": {
            "if_primary_fails": (
                "Keep CTI as a representation-quality/scope law and remove public "
                "downstream benchmark prediction from paper-facing language."
            ),
            "if_only_total_correlation_passes": (
                "Report as scale-correlated diagnostic only; do not claim family-transfer "
                "prediction."
            ),
            "if_lm_eval_task_set_changes": (
                "Freeze a new dated bridge before running; do not mix task panels."
            ),
        },
        "commands": command_templates(),
        "preflight": preflight(),
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    artifact = build_bridge()
    OUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"status: {artifact['status']}")
    print(f"can_run_public_panel_here: {artifact['preflight']['can_run_public_panel_here']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
