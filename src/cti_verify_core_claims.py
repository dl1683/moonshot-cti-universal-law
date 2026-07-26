"""Verify CTI's central public claims against canonical result JSON files.

This is a lightweight replication gate. It does not rerun expensive experiments;
it checks that the paper-facing numbers still match the current canonical
artifacts and that honest negative results remain represented as negatives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_result(name: str) -> dict[str, Any]:
    path = RESULTS / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_path(data: dict[str, Any], dotted_path: str) -> Any:
    value: Any = data
    for part in dotted_path.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        else:
            value = value[part]
    return value


def as_float(value: Any) -> float:
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


class ClaimVerifier:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self,
        claim: str,
        source_file: str,
        observed: Any,
        expected: str,
        passed: bool,
    ) -> None:
        self.checks.append(
            {
                "status": "PASS" if passed else "FAIL",
                "claim": claim,
                "source": f"results/{source_file}",
                "observed": observed,
                "expected": expected,
            }
        )

    def threshold(
        self,
        claim: str,
        source_file: str,
        data: dict[str, Any],
        path: str,
        op: str,
        threshold: float,
    ) -> None:
        observed = as_float(get_path(data, path))
        if op == ">=":
            passed = observed >= threshold
        elif op == ">":
            passed = observed > threshold
        elif op == "<=":
            passed = observed <= threshold
        elif op == "<":
            passed = observed < threshold
        else:
            raise ValueError(f"Unsupported operator: {op}")
        self.add(claim, source_file, observed, f"{op} {threshold}", passed)

    def near(
        self,
        claim: str,
        source_file: str,
        data: dict[str, Any],
        path: str,
        expected_value: float,
        tolerance: float,
    ) -> None:
        observed = as_float(get_path(data, path))
        passed = abs(observed - expected_value) <= tolerance
        self.add(
            claim,
            source_file,
            observed,
            f"{expected_value} +/- {tolerance}",
            passed,
        )

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [check for check in self.checks if check["status"] != "PASS"]


def build_verification() -> ClaimVerifier:
    verifier = ClaimVerifier()

    loao_file = "cti_kappa_loao_per_dataset.json"
    loao = load_result(loao_file)
    verifier.near("LOAO alpha is the canonical NLP-decoder slope", loao_file, loao, "global_fit.alpha", 1.477, 0.005)
    verifier.threshold("LOAO fit quality remains paper-grade", loao_file, loao, "global_fit.r2", ">=", 0.95)
    verifier.threshold("LOAO alpha CV remains below 2.5%", loao_file, loao, "loao_alpha_cv", "<=", 0.025)
    verifier.threshold("LOAO covers at least 12 architectures", loao_file, loao, "n_architectures", ">=", 12)

    utility_file = "cti_utility_revised.json"
    utility = load_result(utility_file)
    verifier.threshold("H8+ blind holdout has 77 predictions", utility_file, utility, "h8_prospective_blind.n_predictions", ">=", 77)
    verifier.threshold("H8+ blind holdout correlation remains high", utility_file, utility, "h8_prospective_blind.logit_pearson_r", ">=", 0.87)
    verifier.threshold("H8+ blind holdout MAE remains below 0.08", utility_file, utility, "h8_prospective_blind.mae_full_model", "<=", 0.08)
    verifier.threshold("Layer-selection beats random baseline by at least 40pp", utility_file, utility, "summary.h1_lift_over_random", ">=", 40.0)

    confusion_file = "cti_confusion_causal_prediction.json"
    confusion = load_result(confusion_file)
    for result in confusion["results"]:
        delta = result["delta"]
        source = f"{confusion_file} delta={delta}"
        verifier.add(
            "Causal confusion prediction keeps r_tau_star above 0.75",
            source,
            result["r_tau_star"],
            ">= 0.75",
            result["r_tau_star"] >= 0.75,
        )
        verifier.add(
            "Causal confusion prediction keeps sign accuracy above 90%",
            source,
            result["sign_accuracy"],
            ">= 0.90",
            result["sign_accuracy"] >= 0.90,
        )
        verifier.add(
            "Causal confusion preregistered hypotheses pass",
            source,
            {
                "H1": result["pass_H1"],
                "H2": result["pass_H2"],
                "H3": result["pass_H3"],
            },
            "all true",
            bool(result["pass_H1"] and result["pass_H2"] and result["pass_H3"]),
        )

    allen_file = "cti_allen_all_sessions_complete.json"
    allen = load_result(allen_file)
    verifier.threshold("Mouse V1 mean r_kappa remains above 0.70", allen_file, allen, "summary.mean_r_kappa", ">=", 0.70)
    verifier.add(
        "Mouse V1 session pass rate remains at least 30/32",
        allen_file,
        allen["summary"]["H1_pass_rate"],
        ">= 30/32",
        as_float(allen["summary"]["H1_pass_rate"]) >= 30 / 32,
    )

    multiarea_file = "cti_allen_multiarea_batch.json"
    multiarea = load_result(multiarea_file)
    for key in ("H_area1", "H_area2", "H_area3"):
        verifier.add(
            f"Allen multi-area hypothesis {key} remains pass",
            multiarea_file,
            multiarea["hypothesis_results"][key],
            "PASS true",
            bool(multiarea["hypothesis_results"][key]["PASS"]),
        )

    equicorr_file = "cti_allen_equicorr_multiarea.json"
    equicorr = load_result(equicorr_file)
    verifier.add(
        "Allen cross-area equicorrelation hypothesis remains pass",
        equicorr_file,
        equicorr["hypothesis_results"]["H_equicorr1"],
        "PASS true",
        bool(equicorr["hypothesis_results"]["H_equicorr1"]["PASS"]),
    )

    cross_rho_file = "cti_cross_modal_rho.json"
    cross_rho = load_result(cross_rho_file)
    verifier.threshold("Cross-modal rho mean remains near-simplex", cross_rho_file, cross_rho, "key_finding.rho_mean_all", ">=", 0.45)
    verifier.threshold("Cross-modal rho range upper bound remains tight", cross_rho_file, cross_rho, "key_finding.rho_range.1", "<=", 0.47)

    smollm_file = "cti_smollm2_ood_prediction.json"
    smollm = load_result(smollm_file)
    verifier.threshold("Blind SmolLM2 OOD r remains above 0.80", smollm_file, smollm, "blind_prediction.pearson_r", ">=", 0.80)
    verifier.add("Blind SmolLM2 OOD verdict remains PASS", smollm_file, smollm["verdict"], "PASS", smollm["verdict"] == "PASS")

    audio_file = "cti_audio_speech.json"
    audio = load_result(audio_file)
    verifier.threshold("Audio speech law r remains above 0.89", audio_file, audio, "law_fit.r_pearson", ">=", 0.89)
    verifier.threshold("Audio speech law p remains significant", audio_file, audio, "law_fit.p_pearson", "<=", 0.01)

    alpha_rho_file = "cti_alpha_rho_multidataset.json"
    alpha_rho = load_result(alpha_rho_file)
    verifier.threshold("Zero-parameter alpha-rho MAE remains below 0.07", alpha_rho_file, alpha_rho, "aggregate.raw_mae", "<=", 0.07)
    verifier.threshold("Zero-parameter alpha-rho mean relative error remains below 5%", alpha_rho_file, alpha_rho, "aggregate.mean_rel_error", "<=", 0.05)

    h3_file = "cti_downstream_h3_n9.json"
    h3 = load_result(h3_file)
    verifier.threshold("H3 model-ranking rho remains above 0.80", h3_file, h3, "H3_extended.spearman_rho", ">=", 0.80)
    verifier.threshold("H3 model-ranking p remains significant", h3_file, h3, "H3_extended.p_value", "<=", 0.01)

    protein_file = "cti_protein_esm2.json"
    protein = load_result(protein_file)
    verifier.add(
        "Protein cross-architecture negative remains negative-alpha failure",
        protein_file,
        protein["cross_architecture"],
        "alpha < 0 and |r| < 0.30 and p > 0.50",
        protein["cross_architecture"]["alpha"] < 0
        and abs(protein["cross_architecture"]["r_pearson"]) < 0.30
        and protein["cross_architecture"]["p_pearson"] > 0.50,
    )

    nsd_file = "cti_nsd_human_fmri.json"
    nsd = load_result(nsd_file)
    verifier.add(
        "NSD human fMRI remains null under pooled balanced analysis",
        nsd_file,
        nsd["pooled_fit"],
        "|r| < 0.20 and p > 0.05",
        abs(nsd["pooled_fit"]["r"]) < 0.20 and nsd["pooled_fit"]["p_value"] > 0.05,
    )

    generation_file = "cti_generation_law_analysis.json"
    generation = load_result(generation_file)
    verifier.threshold("Generation law no-LFM top1K r remains strong negative", generation_file, generation, "cross_architecture_no_LFM.kappa_top1K_vs_logCE.r", "<=", -0.69)
    verifier.threshold("Generation law no-LFM p remains significant", generation_file, generation, "cross_architecture_no_LFM.kappa_top1K_vs_logCE.p", "<=", 0.001)
    verifier.threshold("Generation fixed-V r remains strong negative", generation_file, generation, "fixed_V_CE20K.kappa_bar_vs_logCE.r", "<=", -0.83)
    verifier.threshold("Generation architecture interaction remains non-significant", generation_file, generation, "fixed_V_CE20K.architecture_independence_F_test.p", ">=", 0.10)

    field_before_q_file = "cti_field_before_q_audit.json"
    field_before_q = load_result(field_before_q_file)
    verifier.threshold(
        "Field-before-q audit has full filtered split-safe coverage",
        field_before_q_file,
        field_before_q,
        "scope.n_rows",
        ">=",
        380,
    )
    verifier.threshold(
        "Reference local safe rate predicts disjoint holdout q under LOMO",
        field_before_q_file,
        field_before_q,
        "best.reference_only_leave_one_model_out_metric_plus_logK.r2",
        ">=",
        0.90,
    )
    verifier.threshold(
        "Reference local margin has positive disjoint holdout q transfer under LODO",
        field_before_q_file,
        field_before_q,
        "best.reference_only_leave_one_dataset_out_metric_plus_logK.r2",
        ">=",
        0.45,
    )
    verifier.threshold(
        "Reference local safe rate beats shuffled-label safe rate under LOMO",
        field_before_q_file,
        field_before_q,
        "answer_so_far.critical_comparison.reference_safe_minus_shuffle_safe_r2",
        ">=",
        0.50,
    )

    field_scope_file = "cti_field_scope_transfer_audit.json"
    field_scope = load_result(field_scope_file)
    verifier.threshold(
        "Coupled reference field repairs leave-one-dataset transfer",
        field_scope_file,
        field_scope,
        "answer_so_far.minimal_transfer_panel_lodo.r2",
        ">=",
        0.90,
    )
    verifier.threshold(
        "Coupled reference field has positive macro fold transfer",
        field_scope_file,
        field_scope,
        "answer_so_far.minimal_transfer_panel_lodo.macro_fold_r2",
        ">=",
        0.50,
    )
    verifier.threshold(
        "Coupled reference field beats best single reference metric under LODO",
        field_scope_file,
        field_scope,
        "answer_so_far.critical_deltas.minimal_over_best_single_lodo_r2",
        ">=",
        0.40,
    )
    verifier.threshold(
        "Field transfer collapses without reference safe rate",
        field_scope_file,
        field_scope,
        "answer_so_far.critical_deltas.full_panel_without_safe_rate_lodo_r2",
        "<",
        0.0,
    )
    verifier.threshold(
        "Field transfer remains weak without centroid kappa",
        field_scope_file,
        field_scope,
        "answer_so_far.critical_deltas.full_panel_without_centroid_kappa_lodo_r2",
        "<=",
        0.35,
    )

    family_transfer_file = "cti_field_family_transfer_audit.json"
    family_transfer = load_result(family_transfer_file)
    verifier.threshold(
        "Coupled reference field survives leave-one-family-out transfer",
        family_transfer_file,
        family_transfer,
        "answer_so_far.family_minimal_panel_lofo.r2",
        ">=",
        0.94,
    )
    verifier.threshold(
        "Coupled reference field has strong macro family-fold transfer",
        family_transfer_file,
        family_transfer,
        "answer_so_far.family_minimal_panel_lofo.macro_fold_r2",
        ">=",
        0.90,
    )
    verifier.threshold(
        "Family transfer beats centroid-only control",
        family_transfer_file,
        family_transfer,
        "answer_so_far.critical_deltas.family_minimal_over_centroid_lofo_r2",
        ">=",
        0.30,
    )
    verifier.threshold(
        "Family transfer beats shuffled-label control",
        family_transfer_file,
        family_transfer,
        "answer_so_far.critical_deltas.family_minimal_over_shuffle_lofo_r2",
        ">=",
        0.20,
    )

    arch_shift_file = "cti_field_arch_shift_mamba130m_audit.json"
    arch_shift = load_result(arch_shift_file)
    verifier.threshold(
        "Coupled reference field predicts held-out Mamba aggregate",
        arch_shift_file,
        arch_shift,
        "answer_so_far.family_minimal_panel_external_mamba.r2",
        ">=",
        0.90,
    )
    verifier.threshold(
        "Held-out Mamba aggregate MAE remains low",
        arch_shift_file,
        arch_shift,
        "answer_so_far.family_minimal_panel_external_mamba.mae",
        "<=",
        0.20,
    )
    verifier.threshold(
        "Held-out Mamba coupled panel has at least two positive dataset folds",
        arch_shift_file,
        arch_shift,
        "answer_so_far.fold_diagnostics.family_minimal_positive_dataset_folds",
        ">=",
        2,
    )
    verifier.threshold(
        "Held-out Mamba safe-rate scalar is positive on every dataset fold",
        arch_shift_file,
        arch_shift,
        "answer_so_far.fold_diagnostics.safe_scalar_positive_dataset_folds",
        ">=",
        3,
    )
    verifier.threshold(
        "Held-out Mamba coupled panel beats shuffled-label control",
        arch_shift_file,
        arch_shift,
        "answer_so_far.critical_deltas.family_minimal_over_shuffle_r2",
        ">=",
        0.75,
    )

    falcon_shift_file = "cti_field_arch_shift_falconh1_05b_audit.json"
    falcon_shift = load_result(falcon_shift_file)
    verifier.add(
        "Held-out Falcon-H1 hybrid is recorded as a scope break",
        falcon_shift_file,
        falcon_shift["answer_so_far"]["verdict"],
        "falconh1_architecture_shift_breaks_or_weakens_scope_law",
        falcon_shift["answer_so_far"]["verdict"]
        == "falconh1_architecture_shift_breaks_or_weakens_scope_law",
    )
    verifier.threshold(
        "Held-out Falcon-H1 target has full split-safe coverage",
        falcon_shift_file,
        falcon_shift,
        "scope.test.n",
        ">=",
        36,
    )
    verifier.threshold(
        "Held-out Falcon-H1 does not meet clean architecture-shift R2 threshold",
        falcon_shift_file,
        falcon_shift,
        "answer_so_far.family_minimal_panel_external_target.r2",
        "<",
        0.90,
    )
    verifier.threshold(
        "Held-out Falcon-H1 has at most one positive dataset fold",
        falcon_shift_file,
        falcon_shift,
        "answer_so_far.fold_diagnostics.family_minimal_positive_dataset_folds",
        "<=",
        1,
    )
    verifier.threshold(
        "Held-out Falcon-H1 coupled panel still beats shuffled-label control",
        falcon_shift_file,
        falcon_shift,
        "answer_so_far.critical_deltas.family_minimal_over_shuffle_r2",
        ">=",
        1.0,
    )

    falcon_diag_file = "cti_arch_shift_falconh1_05b_diagnostics.json"
    falcon_diag = load_result(falcon_diag_file)
    verifier.add(
        "Falcon-H1 diagnostic identifies centroid-field discordance",
        falcon_diag_file,
        falcon_diag["answer_so_far"]["diagnosis"],
        "centroid_field_discordance_breaks_uncalibrated_architecture_transfer",
        falcon_diag["answer_so_far"]["diagnosis"]
        == "centroid_field_discordance_breaks_uncalibrated_architecture_transfer",
    )
    verifier.threshold(
        "Falcon-H1 safe scalar beats the coupled panel under source-trained transfer",
        falcon_diag_file,
        falcon_diag,
        "answer_so_far.safe_scalar_beats_family_minimal_r2_by",
        ">=",
        0.05,
    )
    verifier.threshold(
        "Falcon-H1 target kappa-safety discordance is material",
        falcon_diag_file,
        falcon_diag,
        "answer_so_far.centroid_safe_margin_z_signature.kappa_minus_safe_z",
        ">=",
        1.0,
    )
    verifier.threshold(
        "Falcon-H1 target kappa-margin discordance is material",
        falcon_diag_file,
        falcon_diag,
        "answer_so_far.centroid_safe_margin_z_signature.kappa_minus_margin_z",
        ">=",
        1.25,
    )

    scope_gate_file = "cti_arch_shift_scope_gate.json"
    scope_gate = load_result(scope_gate_file)
    verifier.add(
        "Q-free architecture scope gate separates current Mamba/Falcon targets",
        scope_gate_file,
        scope_gate["answer_so_far"]["verdict"],
        "candidate_scope_gate_separates_mamba_pass_from_falcon_break_current_n2",
        scope_gate["answer_so_far"]["verdict"]
        == "candidate_scope_gate_separates_mamba_pass_from_falcon_break_current_n2",
    )
    verifier.threshold(
        "Q-free architecture scope gate uses full source split-safe rows",
        scope_gate_file,
        scope_gate,
        "source_scope.n_rows",
        ">=",
        387,
    )
    verifier.threshold(
        "Candidate scope gate is perfect on current two architecture targets",
        scope_gate_file,
        scope_gate,
        "answer_so_far.candidate_gate_accuracy_on_current_targets",
        ">=",
        1.0,
    )
    verifier.add(
        "Candidate scope gate is explicitly not yet prospective",
        scope_gate_file,
        scope_gate["answer_so_far"]["candidate_gate_is_prospective"],
        "false until tested on a new held-out target",
        scope_gate["answer_so_far"]["candidate_gate_is_prospective"] is False,
    )
    verifier.add(
        "Candidate scope gate triggers only on Falcon-H1 in current target set",
        scope_gate_file,
        scope_gate["answer_so_far"]["candidate_gate_triggered_targets"],
        "['Falcon-H1-0.5B']",
        scope_gate["answer_so_far"]["candidate_gate_triggered_targets"]
        == ["Falcon-H1-0.5B"],
    )
    verifier.threshold(
        "Candidate scope gate beats coarse source-percentile gate on current targets",
        scope_gate_file,
        {
            "delta": scope_gate["answer_so_far"]["candidate_gate_accuracy_on_current_targets"]
            - scope_gate["answer_so_far"]["source_percentile_gate_accuracy_on_current_targets"]
        },
        "delta",
        ">=",
        0.5,
    )

    zamba_arch_file = "cti_field_arch_shift_zamba2_12b_audit.json"
    zamba_arch = load_result(zamba_arch_file)
    verifier.add(
        "Held-out Zamba2 hybrid is recorded as a scope break",
        zamba_arch_file,
        zamba_arch["answer_so_far"]["verdict"],
        "zamba2_12b_architecture_shift_breaks_or_weakens_scope_law",
        zamba_arch["answer_so_far"]["verdict"]
        == "zamba2_12b_architecture_shift_breaks_or_weakens_scope_law",
    )
    verifier.threshold(
        "Held-out Zamba2 target has full split-safe coverage",
        zamba_arch_file,
        zamba_arch,
        "answer_so_far.family_minimal_panel_external_target.n_test",
        ">=",
        36,
    )
    verifier.threshold(
        "Held-out Zamba2 does not meet clean architecture-shift R2 threshold",
        zamba_arch_file,
        zamba_arch,
        "answer_so_far.family_minimal_panel_external_target.r2",
        "<",
        0.9,
    )
    verifier.threshold(
        "Held-out Zamba2 safe scalar beats coupled panel",
        zamba_arch_file,
        {
            "delta": zamba_arch["answer_so_far"]["critical_controls"][
                "reference_safe_scalar_external_target"
            ]["r2"]
            - zamba_arch["answer_so_far"]["family_minimal_panel_external_target"]["r2"]
        },
        "delta",
        ">=",
        0.1,
    )

    zamba_diag_file = "cti_arch_shift_zamba2_12b_diagnostics.json"
    zamba_diag = load_result(zamba_diag_file)
    verifier.add(
        "Zamba2 diagnostic identifies centroid-field discordance",
        zamba_diag_file,
        zamba_diag["answer_so_far"]["diagnosis"],
        "centroid_field_discordance_breaks_uncalibrated_architecture_transfer",
        zamba_diag["answer_so_far"]["diagnosis"]
        == "centroid_field_discordance_breaks_uncalibrated_architecture_transfer",
    )
    verifier.threshold(
        "Zamba2 target kappa-margin discordance is material",
        zamba_diag_file,
        zamba_diag,
        "answer_so_far.centroid_safe_margin_z_signature.kappa_minus_margin_z",
        ">=",
        1.5,
    )

    zamba_gate_file = "cti_arch_shift_scope_gate_with_zamba2_audit.json"
    zamba_gate = load_result(zamba_gate_file)
    verifier.threshold(
        "Original candidate scope gate fails after adding Zamba2",
        zamba_gate_file,
        zamba_gate,
        "answer_so_far.candidate_gate_accuracy_on_current_targets",
        "<",
        1.0,
    )
    verifier.threshold(
        "Post-Zamba v2 gate separates current three architecture targets",
        zamba_gate_file,
        zamba_gate,
        "answer_so_far.candidate_gate_v2_post_zamba_accuracy_on_completed_targets",
        ">=",
        1.0,
    )
    verifier.add(
        "Post-Zamba v2 gate triggers on Falcon-H1 and Zamba2 only",
        zamba_gate_file,
        zamba_gate["answer_so_far"]["candidate_gate_v2_post_zamba_triggered_targets"],
        "['Falcon-H1-0.5B', 'Zamba2-1.2B']",
        zamba_gate["answer_so_far"]["candidate_gate_v2_post_zamba_triggered_targets"]
        == ["Falcon-H1-0.5B", "Zamba2-1.2B"],
    )
    verifier.add(
        "Post-Zamba v2 gate is explicitly not prospective",
        zamba_gate_file,
        zamba_gate["answer_so_far"]["candidate_gate_v2_post_zamba_is_prospective"],
        "false until tested on a new held-out target",
        zamba_gate["answer_so_far"]["candidate_gate_v2_post_zamba_is_prospective"] is False,
    )

    headwise_arch_file = "cti_field_arch_shift_gated_attention_headwise_audit.json"
    headwise_arch = load_result(headwise_arch_file)
    verifier.add(
        "Held-out headwise gated attention is recorded as a scope break",
        headwise_arch_file,
        headwise_arch["answer_so_far"]["verdict"],
        "gated_attention_headwise_architecture_shift_breaks_or_weakens_scope_law",
        headwise_arch["answer_so_far"]["verdict"]
        == "gated_attention_headwise_architecture_shift_breaks_or_weakens_scope_law",
    )
    verifier.threshold(
        "Held-out headwise gated attention has full split-safe coverage",
        headwise_arch_file,
        headwise_arch,
        "answer_so_far.family_minimal_panel_external_target.n_test",
        ">=",
        36,
    )
    verifier.threshold(
        "Held-out headwise gated attention has negative architecture-transfer R2",
        headwise_arch_file,
        headwise_arch,
        "answer_so_far.family_minimal_panel_external_target.r2",
        "<",
        0.0,
    )

    headwise_diag_file = "cti_arch_shift_gated_attention_headwise_diagnostics.json"
    headwise_diag = load_result(headwise_diag_file)
    verifier.add(
        "Headwise gated attention diagnostic identifies local-field collapse",
        headwise_diag_file,
        headwise_diag["answer_so_far"]["diagnosis"],
        "local_field_collapse_breaks_uncalibrated_architecture_transfer",
        headwise_diag["answer_so_far"]["diagnosis"]
        == "local_field_collapse_breaks_uncalibrated_architecture_transfer",
    )

    elementwise_arch_file = "cti_field_arch_shift_gated_attention_elementwise_audit.json"
    elementwise_arch = load_result(elementwise_arch_file)
    verifier.add(
        "Held-out elementwise gated attention is recorded as a scope break",
        elementwise_arch_file,
        elementwise_arch["answer_so_far"]["verdict"],
        "gated_attention_elementwise_architecture_shift_breaks_or_weakens_scope_law",
        elementwise_arch["answer_so_far"]["verdict"]
        == "gated_attention_elementwise_architecture_shift_breaks_or_weakens_scope_law",
    )
    verifier.threshold(
        "Held-out elementwise gated attention has full split-safe coverage",
        elementwise_arch_file,
        elementwise_arch,
        "answer_so_far.family_minimal_panel_external_target.n_test",
        ">=",
        36,
    )
    verifier.threshold(
        "Held-out elementwise gated attention has negative architecture-transfer R2",
        elementwise_arch_file,
        elementwise_arch,
        "answer_so_far.family_minimal_panel_external_target.r2",
        "<",
        0.0,
    )

    elementwise_diag_file = "cti_arch_shift_gated_attention_elementwise_diagnostics.json"
    elementwise_diag = load_result(elementwise_diag_file)
    verifier.add(
        "Elementwise gated attention diagnostic identifies local safe/dist collapse",
        elementwise_diag_file,
        elementwise_diag["answer_so_far"]["diagnosis"],
        "local_safe_dist_collapse_breaks_uncalibrated_architecture_transfer",
        elementwise_diag["answer_so_far"]["diagnosis"]
        == "local_safe_dist_collapse_breaks_uncalibrated_architecture_transfer",
    )

    gated_gate_file = "cti_arch_shift_scope_gate_with_gated_attention_elementwise_audit.json"
    gated_gate = load_result(gated_gate_file)
    verifier.threshold(
        "Post-Zamba v2 gate fails after gated-attention targets",
        gated_gate_file,
        gated_gate,
        "answer_so_far.candidate_gate_v2_post_zamba_accuracy_on_completed_targets",
        "<",
        0.7,
    )
    verifier.threshold(
        "Post-headwise v3 gate still misses elementwise gated attention",
        gated_gate_file,
        gated_gate,
        "answer_so_far.candidate_gate_v3_post_headwise_accuracy_on_completed_targets",
        "<",
        1.0,
    )
    verifier.threshold(
        "Post-elementwise v4 gate separates the pre-OLMoE five architecture targets",
        gated_gate_file,
        gated_gate,
        "answer_so_far.candidate_gate_v4_post_elementwise_accuracy_on_completed_targets",
        ">=",
        1.0,
    )
    verifier.add(
        "Post-elementwise v4 gate triggers on all current scope breaks only",
        gated_gate_file,
        gated_gate["answer_so_far"]["candidate_gate_v4_post_elementwise_triggered_targets"],
        "['Falcon-H1-0.5B', 'GatedAttention-1B-Headwise', 'Zamba2-1.2B', 'GatedAttention-1B-Elementwise']",
        gated_gate["answer_so_far"]["candidate_gate_v4_post_elementwise_triggered_targets"]
        == [
            "Falcon-H1-0.5B",
            "GatedAttention-1B-Headwise",
            "Zamba2-1.2B",
            "GatedAttention-1B-Elementwise",
        ],
    )
    verifier.add(
        "Post-elementwise v4 gate is explicitly not prospective",
        gated_gate_file,
        gated_gate["answer_so_far"]["candidate_gate_v4_post_elementwise_is_prospective"],
        "false until tested on a new held-out target",
        gated_gate["answer_so_far"]["candidate_gate_v4_post_elementwise_is_prospective"] is False,
    )

    olmoe_prereg_file = "cti_arch_scope_gate_v4_preregistration_20260608.json"
    olmoe_prereg = load_result(olmoe_prereg_file)
    verifier.add(
        "OLMoE v4 preregistration was frozen before OLMoE q",
        olmoe_prereg_file,
        olmoe_prereg["status"],
        "frozen_after_elementwise_gated_attention_before_olmoe_q",
        olmoe_prereg["status"] == "frozen_after_elementwise_gated_attention_before_olmoe_q",
    )

    olmoe_preq_file = "cti_field_before_q_olmoe_1b7b_preq.json"
    olmoe_preq = load_result(olmoe_preq_file)
    olmoe_preq_rows = olmoe_preq["rows"]
    olmoe_preq_bad_q_fields = [
        (idx, key)
        for idx, row in enumerate(olmoe_preq_rows)
        for key in row
        if "q_holdout" in key or key in {"acc_holdout", "logit_q_holdout"}
    ]
    verifier.add(
        "OLMoE pre-q target rows withhold q and task accuracy fields",
        olmoe_preq_file,
        {"n_rows": len(olmoe_preq_rows), "bad_q_fields": olmoe_preq_bad_q_fields},
        "36 rows and no q_holdout/acc_holdout/logit_q_holdout fields",
        len(olmoe_preq_rows) == 36 and not olmoe_preq_bad_q_fields,
    )

    olmoe_arch_file = "cti_field_arch_shift_olmoe_1b7b_audit.json"
    olmoe_arch = load_result(olmoe_arch_file)
    verifier.add(
        "Held-out OLMoE survives the source-trained architecture-shift audit",
        olmoe_arch_file,
        olmoe_arch["answer_so_far"]["verdict"],
        "coupled_reference_field_survives_olmoe_1b7b_architecture_shift",
        olmoe_arch["answer_so_far"]["verdict"]
        == "coupled_reference_field_survives_olmoe_1b7b_architecture_shift",
    )
    verifier.threshold(
        "Held-out OLMoE family-minimal transfer R2 remains strict-pass",
        olmoe_arch_file,
        olmoe_arch,
        "answer_so_far.family_minimal_panel_external_target.r2",
        ">=",
        0.9,
    )
    verifier.threshold(
        "Held-out OLMoE family-minimal transfer MAE remains below strict threshold",
        olmoe_arch_file,
        olmoe_arch,
        "answer_so_far.family_minimal_panel_external_target.mae",
        "<=",
        0.25,
    )
    verifier.threshold(
        "Held-out OLMoE passes on all dataset folds",
        olmoe_arch_file,
        olmoe_arch,
        "answer_so_far.fold_diagnostics.family_minimal_positive_dataset_folds",
        ">=",
        3,
    )

    olmoe_diag_file = "cti_arch_shift_olmoe_1b7b_diagnostics.json"
    olmoe_diag = load_result(olmoe_diag_file)
    verifier.add(
        "OLMoE diagnostic records no extra architecture failure mode",
        olmoe_diag_file,
        olmoe_diag["answer_so_far"]["diagnosis"],
        "architecture_transfer_needs_no_extra_failure_diagnosis",
        olmoe_diag["answer_so_far"]["diagnosis"]
        == "architecture_transfer_needs_no_extra_failure_diagnosis",
    )

    olmoe_gate_file = "cti_arch_shift_scope_gate_with_olmoe_1b7b_audit.json"
    olmoe_gate = load_result(olmoe_gate_file)
    verifier.add(
        "Post-elementwise v4 gate has a completed prospective OLMoE result",
        olmoe_gate_file,
        olmoe_gate["answer_so_far"]["latest_candidate_gate"]["prospective_targets"],
        "['OLMoE-1B-7B-0125']",
        olmoe_gate["answer_so_far"]["latest_candidate_gate"]["prospective_targets"]
        == ["OLMoE-1B-7B-0125"],
    )
    verifier.add(
        "Post-elementwise v4 gate separates all completed architecture targets after OLMoE",
        olmoe_gate_file,
        olmoe_gate["answer_so_far"]["latest_candidate_gate"]["verdict"],
        "candidate_scope_gate_v4_separates_completed_architecture_targets",
        olmoe_gate["answer_so_far"]["latest_candidate_gate"]["verdict"]
        == "candidate_scope_gate_v4_separates_completed_architecture_targets",
    )
    verifier.threshold(
        "Post-elementwise v4 gate is correct on completed architecture targets after OLMoE",
        olmoe_gate_file,
        olmoe_gate,
        "answer_so_far.latest_candidate_gate.accuracy_on_completed_targets",
        ">=",
        1.0,
    )
    verifier.add(
        "Post-elementwise v4 gate correctly did not warn on OLMoE",
        olmoe_gate_file,
        {
            "not_triggered": olmoe_gate["answer_so_far"]["latest_candidate_gate"]["not_triggered_targets"],
            "triggered": olmoe_gate["answer_so_far"]["latest_candidate_gate"]["triggered_targets"],
        },
        "OLMoE in not_triggered and absent from triggered",
        "OLMoE-1B-7B-0125"
        in olmoe_gate["answer_so_far"]["latest_candidate_gate"]["not_triggered_targets"]
        and "OLMoE-1B-7B-0125"
        not in olmoe_gate["answer_so_far"]["latest_candidate_gate"]["triggered_targets"],
    )

    phi_prereg_file = "cti_arch_scope_gate_v4_phi_tiny_moe_preregistration_20260608.json"
    phi_prereg = load_result(phi_prereg_file)
    verifier.add(
        "Phi-tiny-MoE v4 preregistration was frozen before Phi q",
        phi_prereg_file,
        phi_prereg["status"],
        "frozen_after_olmoe_v4_pass_before_phi_tiny_moe_q",
        phi_prereg["status"] == "frozen_after_olmoe_v4_pass_before_phi_tiny_moe_q",
    )

    phi_preq_file = "cti_field_before_q_phi_tiny_moe_preq.json"
    phi_preq = load_result(phi_preq_file)
    phi_preq_rows = phi_preq["rows"]
    phi_preq_bad_q_fields = [
        (idx, key)
        for idx, row in enumerate(phi_preq_rows)
        for key in row
        if "q_holdout" in key or key in {"acc_holdout", "logit_q_holdout"}
    ]
    verifier.add(
        "Phi-tiny-MoE pre-q target rows withhold q and task accuracy fields",
        phi_preq_file,
        {"n_rows": len(phi_preq_rows), "bad_q_fields": phi_preq_bad_q_fields},
        "36 rows and no q_holdout/acc_holdout/logit_q_holdout fields",
        len(phi_preq_rows) == 36 and not phi_preq_bad_q_fields,
    )

    phi_arch_file = "cti_field_arch_shift_phi_tiny_moe_audit.json"
    phi_arch = load_result(phi_arch_file)
    verifier.add(
        "Held-out Phi-tiny-MoE is recorded as a scope break",
        phi_arch_file,
        phi_arch["answer_so_far"]["verdict"],
        "phi_tiny_moe_architecture_shift_breaks_or_weakens_scope_law",
        phi_arch["answer_so_far"]["verdict"]
        == "phi_tiny_moe_architecture_shift_breaks_or_weakens_scope_law",
    )
    verifier.threshold(
        "Held-out Phi-tiny-MoE does not meet clean architecture-shift R2 threshold",
        phi_arch_file,
        phi_arch,
        "answer_so_far.family_minimal_panel_external_target.r2",
        "<",
        0.9,
    )
    verifier.threshold(
        "Held-out Phi-tiny-MoE misses strict MAE threshold",
        phi_arch_file,
        phi_arch,
        "answer_so_far.family_minimal_panel_external_target.mae",
        ">",
        0.25,
    )
    verifier.threshold(
        "Held-out Phi-tiny-MoE remains directionally positive on all folds",
        phi_arch_file,
        phi_arch,
        "answer_so_far.fold_diagnostics.family_minimal_positive_dataset_folds",
        ">=",
        3,
    )

    phi_diag_file = "cti_arch_shift_phi_tiny_moe_diagnostics.json"
    phi_diag = load_result(phi_diag_file)
    verifier.add(
        "Phi-tiny-MoE diagnostic identifies positive but miscalibrated transfer",
        phi_diag_file,
        phi_diag["answer_so_far"]["diagnosis"],
        "positive_but_miscalibrated_architecture_transfer",
        phi_diag["answer_so_far"]["diagnosis"]
        == "positive_but_miscalibrated_architecture_transfer",
    )

    phi_gate_file = "cti_arch_shift_scope_gate_with_phi_tiny_moe_audit.json"
    phi_gate = load_result(phi_gate_file)
    phi_gate_target = [
        target for target in phi_gate["target_results"] if target["display"] == "Phi-tiny-MoE-instruct"
    ][0]
    verifier.add(
        "Post-elementwise v4 gate is prospectively falsified by Phi-tiny-MoE",
        phi_gate_file,
        phi_gate["answer_so_far"]["latest_candidate_gate"]["verdict"],
        "candidate_scope_gate_v4_does_not_separate_completed_architecture_targets",
        phi_gate["answer_so_far"]["latest_candidate_gate"]["verdict"]
        == "candidate_scope_gate_v4_does_not_separate_completed_architecture_targets",
    )
    verifier.threshold(
        "Post-elementwise v4 gate accuracy drops after Phi-tiny-MoE",
        phi_gate_file,
        phi_gate,
        "answer_so_far.latest_candidate_gate.accuracy_on_completed_targets",
        "<",
        1.0,
    )
    verifier.add(
        "Post-elementwise v4 gate records both OLMoE and Phi as prospective targets",
        phi_gate_file,
        phi_gate["answer_so_far"]["latest_candidate_gate"]["prospective_targets"],
        "['OLMoE-1B-7B-0125', 'Phi-tiny-MoE-instruct']",
        phi_gate["answer_so_far"]["latest_candidate_gate"]["prospective_targets"]
        == ["OLMoE-1B-7B-0125", "Phi-tiny-MoE-instruct"],
    )
    verifier.add(
        "Post-elementwise v4 gate missed the Phi-tiny-MoE scope break",
        phi_gate_file,
        phi_gate_target["evaluation_against_outcome"],
        "candidate_scope_gate_v4_correct false and should_warn true",
        phi_gate_target["evaluation_against_outcome"]["candidate_scope_gate_v4_correct"] is False
        and phi_gate_target["evaluation_against_outcome"]["should_warn_given_outcome"] is True,
    )

    v5_post_phi_file = "cti_arch_scope_calibrated_detector_v5_post_phi.json"
    v5_post_phi = load_result(v5_post_phi_file)
    verifier.add(
        "Post-Phi v5 calibrated detector is explicitly post-hoc",
        v5_post_phi_file,
        v5_post_phi["status"],
        "post_hoc_detector_candidate_not_prospective",
        v5_post_phi["status"] == "post_hoc_detector_candidate_not_prospective",
    )
    verifier.threshold(
        "Post-Phi v5 calibrated detector fits the completed pre-Granite targets",
        v5_post_phi_file,
        v5_post_phi,
        "answer_so_far.detector_v5_accuracy_on_completed_targets",
        ">=",
        1.0,
    )

    granite_prereg_file = "cti_arch_scope_detector_v5_granite_preregistration_20260608.json"
    granite_prereg = load_result(granite_prereg_file)
    verifier.add(
        "Granite v5 preregistration was frozen before Granite q",
        granite_prereg_file,
        granite_prereg["status"],
        "frozen_after_phi_v4_miss_before_granite_q",
        granite_prereg["status"] == "frozen_after_phi_v4_miss_before_granite_q",
    )

    granite_preq_file = "cti_field_before_q_granite_3_0_1b_a400m_preq.json"
    granite_preq = load_result(granite_preq_file)
    granite_preq_rows = granite_preq["rows"]
    granite_preq_bad_q_fields = [
        (idx, key)
        for idx, row in enumerate(granite_preq_rows)
        for key in row
        if "q_holdout" in key or key in {"acc_holdout", "logit_q_holdout"}
    ]
    verifier.add(
        "Granite pre-q target rows withhold q and task accuracy fields",
        granite_preq_file,
        {"n_rows": len(granite_preq_rows), "bad_q_fields": granite_preq_bad_q_fields},
        "36 rows and no q_holdout/acc_holdout/logit_q_holdout fields",
        len(granite_preq_rows) == 36 and not granite_preq_bad_q_fields,
    )

    granite_arch_file = "cti_field_arch_shift_granite_3_0_1b_a400m_audit.json"
    granite_arch = load_result(granite_arch_file)
    verifier.add(
        "Held-out Granite MoE is recorded as a scope break",
        granite_arch_file,
        granite_arch["answer_so_far"]["verdict"],
        "granite_3_0_1b_a400m_architecture_shift_breaks_or_weakens_scope_law",
        granite_arch["answer_so_far"]["verdict"]
        == "granite_3_0_1b_a400m_architecture_shift_breaks_or_weakens_scope_law",
    )
    verifier.threshold(
        "Held-out Granite MoE does not meet clean architecture-shift R2 threshold",
        granite_arch_file,
        granite_arch,
        "answer_so_far.family_minimal_panel_external_target.r2",
        "<",
        0.9,
    )
    verifier.threshold(
        "Held-out Granite MoE misses strict MAE threshold",
        granite_arch_file,
        granite_arch,
        "answer_so_far.family_minimal_panel_external_target.mae",
        ">",
        0.25,
    )
    verifier.threshold(
        "Held-out Granite MoE remains directionally positive on all folds",
        granite_arch_file,
        granite_arch,
        "answer_so_far.fold_diagnostics.family_minimal_positive_dataset_folds",
        ">=",
        3,
    )

    granite_diag_file = "cti_arch_shift_granite_3_0_1b_a400m_diagnostics.json"
    granite_diag = load_result(granite_diag_file)
    verifier.add(
        "Granite MoE diagnostic identifies positive but miscalibrated transfer",
        granite_diag_file,
        granite_diag["answer_so_far"]["diagnosis"],
        "positive_but_miscalibrated_architecture_transfer",
        granite_diag["answer_so_far"]["diagnosis"]
        == "positive_but_miscalibrated_architecture_transfer",
    )

    granite_gate_file = "cti_arch_shift_scope_detector_v5_with_granite_audit.json"
    granite_gate = load_result(granite_gate_file)
    granite_gate_target = [
        target
        for target in granite_gate["completed_targets"]
        if target["display"] == "Granite-3.0-1B-A400M-Base"
    ][0]
    verifier.add(
        "Post-Phi v5 detector is prospectively falsified by Granite",
        granite_gate_file,
        granite_gate["answer_so_far"]["detector_v5_incorrect_targets"],
        "['Granite-3.0-1B-A400M-Base']",
        granite_gate["answer_so_far"]["detector_v5_incorrect_targets"]
        == ["Granite-3.0-1B-A400M-Base"],
    )
    verifier.threshold(
        "Post-Phi v5 detector accuracy drops after Granite",
        granite_gate_file,
        granite_gate,
        "answer_so_far.detector_v5_accuracy_on_completed_targets",
        "<",
        1.0,
    )
    verifier.add(
        "Post-Phi v5 detector missed the Granite scope break",
        granite_gate_file,
        {
            "detector_triggered": granite_gate_target["detector_v5"]["triggered"],
            "should_warn": granite_gate_target["should_warn"],
        },
        "detector_triggered false and should_warn true",
        granite_gate_target["detector_v5"]["triggered"] is False
        and granite_gate_target["should_warn"] is True,
    )

    v6_file = "cti_arch_scope_v6_selective_abstention.json"
    v6 = load_result(v6_file)
    verifier.add(
        "Post-Granite v6 scope overlay is explicitly not prospective",
        v6_file,
        v6["status"],
        "post_granite_selective_abstention_overlay_not_prospective",
        v6["status"] == "post_granite_selective_abstention_overlay_not_prospective",
    )
    verifier.add(
        "V6 records Granite as the v5 false clear it repairs",
        v6_file,
        v6["answer_so_far"]["v5_false_clear_targets"],
        "['Granite-3.0-1B-A400M-Base']",
        v6["answer_so_far"]["v5_false_clear_targets"]
        == ["Granite-3.0-1B-A400M-Base"],
    )
    verifier.add(
        "V6 selective abstention has no false clears on the current table",
        v6_file,
        v6["answer_so_far"]["v6_false_clear_targets"],
        "[]",
        v6["answer_so_far"]["v6_false_clear_targets"] == [],
    )
    verifier.add(
        "V6 abstains on the ambiguous OLMoE/Granite basin",
        v6_file,
        v6["answer_so_far"]["v6_abstain_targets"],
        "['OLMoE-1B-7B-0125', 'Granite-3.0-1B-A400M-Base']",
        v6["answer_so_far"]["v6_abstain_targets"]
        == ["OLMoE-1B-7B-0125", "Granite-3.0-1B-A400M-Base"],
    )
    verifier.threshold(
        "V6 keeps decisive accuracy perfect on the current post-hoc table",
        v6_file,
        v6,
        "answer_so_far.v6_decisive_accuracy",
        ">=",
        1.0,
    )
    verifier.threshold(
        "V6 remains a selective overlay rather than full coverage",
        v6_file,
        v6,
        "answer_so_far.v6_coverage",
        "<",
        1.0,
    )
    verifier.add(
        "V6 turns Granite from false clear into abstain_calibrate",
        v6_file,
        v6["answer_so_far"]["granite_action"],
        "abstain_calibrate",
        v6["answer_so_far"]["granite_action"] == "abstain_calibrate",
    )
    granite_neighbors = v6["answer_so_far"]["granite_nearest_neighbors"]
    verifier.add(
        "V6 records Granite's ambiguous MoE neighborhood",
        v6_file,
        [
            {
                "display": row["display"],
                "outcome_class": row["outcome_class"],
                "distance": row["distance"],
            }
            for row in granite_neighbors[:2]
        ],
        "OLMoE strict pass and Phi weak positive miscalibration within radius",
        granite_neighbors[0]["display"] == "OLMoE-1B-7B-0125"
        and granite_neighbors[0]["distance"] < 0.7
        and granite_neighbors[1]["display"] == "Phi-tiny-MoE-instruct"
        and granite_neighbors[1]["distance"] < 0.9,
    )
    verifier.add(
        "V6 leave-one-target-out overlay also avoids false clears",
        v6_file,
        v6["answer_so_far"]["leave_one_target_out_false_clears"],
        "[]",
        v6["answer_so_far"]["leave_one_target_out_false_clears"] == [],
    )

    lfm2_prereg_file = "cti_arch_scope_v7_lfm2_350m_preregistration_20260608.json"
    lfm2_prereg = load_result(lfm2_prereg_file)
    verifier.add(
        "LFM2 v7 preregistration was frozen before LFM2 q",
        lfm2_prereg_file,
        lfm2_prereg["status"],
        "frozen_after_granite_v5_miss_and_v6_abstention_before_lfm2_q",
        lfm2_prereg["status"]
        == "frozen_after_granite_v5_miss_and_v6_abstention_before_lfm2_q",
    )

    lfm2_preq_file = "cti_field_before_q_lfm2_350m_preq.json"
    lfm2_preq = load_result(lfm2_preq_file)
    lfm2_preq_bad_q_fields = []
    for row in lfm2_preq["rows"]:
        for field in ("q_holdout", "acc_holdout", "logit_q_holdout"):
            if field in row:
                lfm2_preq_bad_q_fields.append(field)
    verifier.add(
        "LFM2 pre-q target rows withhold q and task accuracy fields",
        lfm2_preq_file,
        {
            "n_rows": len(lfm2_preq["rows"]),
            "bad_q_fields": sorted(set(lfm2_preq_bad_q_fields)),
        },
        "36 rows and no q_holdout/acc_holdout/logit_q_holdout fields",
        len(lfm2_preq["rows"]) == 36 and not lfm2_preq_bad_q_fields,
    )

    lfm2_v7_preq_file = "cti_arch_scope_v7_with_lfm2_350m_preq.json"
    lfm2_v7_preq = load_result(lfm2_v7_preq_file)
    lfm2_pending = [
        target
        for target in lfm2_v7_preq["answer_so_far"]["pending_fresh_targets"]
        if target["display"] == "LFM2-350M"
    ][0]
    verifier.add(
        "V7 predicted a singleton pass set for LFM2 before q",
        lfm2_v7_preq_file,
        {
            "labels": lfm2_pending["prediction_set"]["labels"],
            "action": lfm2_pending["prediction_set"]["action"],
            "rows_are_q_withheld": lfm2_pending["rows_are_q_withheld"],
        },
        "{pass_or_partial_pass} and q-withheld rows",
        lfm2_pending["prediction_set"]["labels"] == ["pass_or_partial_pass"]
        and lfm2_pending["prediction_set"]["action"] == "clear_with_current_evidence"
        and lfm2_pending["rows_are_q_withheld"] is True,
    )

    lfm2_arch_file = "cti_field_arch_shift_lfm2_350m_audit.json"
    lfm2_arch = load_result(lfm2_arch_file)
    verifier.add(
        "Held-out LFM2 survives the source-trained architecture-shift audit",
        lfm2_arch_file,
        lfm2_arch["answer_so_far"]["verdict"],
        "coupled_reference_field_survives_lfm2_350m_architecture_shift",
        lfm2_arch["answer_so_far"]["verdict"]
        == "coupled_reference_field_survives_lfm2_350m_architecture_shift",
    )
    verifier.threshold(
        "Held-out LFM2 family-minimal transfer R2 remains strict-pass",
        lfm2_arch_file,
        lfm2_arch,
        "answer_so_far.family_minimal_panel_external_target.r2",
        ">=",
        0.9,
    )
    verifier.threshold(
        "Held-out LFM2 family-minimal transfer MAE remains below strict threshold",
        lfm2_arch_file,
        lfm2_arch,
        "answer_so_far.family_minimal_panel_external_target.mae",
        "<=",
        0.25,
    )
    verifier.threshold(
        "Held-out LFM2 passes on all dataset folds",
        lfm2_arch_file,
        lfm2_arch,
        "answer_so_far.fold_diagnostics.family_minimal_positive_dataset_folds",
        ">=",
        3,
    )

    lfm2_diag_file = "cti_arch_shift_lfm2_350m_diagnostics.json"
    lfm2_diag = load_result(lfm2_diag_file)
    verifier.add(
        "LFM2 diagnostic records no extra architecture failure mode",
        lfm2_diag_file,
        lfm2_diag["answer_so_far"]["diagnosis"],
        "architecture_transfer_needs_no_extra_failure_diagnosis",
        lfm2_diag["answer_so_far"]["diagnosis"]
        == "architecture_transfer_needs_no_extra_failure_diagnosis",
    )

    lfm2_v7_file = "cti_arch_scope_v7_with_lfm2_350m_audit.json"
    lfm2_v7 = load_result(lfm2_v7_file)
    lfm2_v7_target = [
        target for target in lfm2_v7["targets"] if target["display"] == "LFM2-350M"
    ][0]
    verifier.add(
        "V7 final table has no completed prediction-set false exclusions",
        lfm2_v7_file,
        lfm2_v7["answer_so_far"]["completed_prediction_set_false_exclusions"],
        "[]",
        lfm2_v7["answer_so_far"]["completed_prediction_set_false_exclusions"] == [],
    )
    verifier.threshold(
        "V7 final singleton predictions remain correct on completed targets",
        lfm2_v7_file,
        lfm2_v7,
        "answer_so_far.completed_singleton_accuracy",
        ">=",
        1.0,
    )
    verifier.add(
        "V7 LFM2 singleton prediction contains the audited truth",
        lfm2_v7_file,
        {
            "truth_label": lfm2_v7_target["truth_label"],
            "prediction_set": lfm2_v7_target["prediction_set"]["labels"],
            "contains_truth": lfm2_v7_target["prediction_set"]["contains_truth"],
            "singleton_correct": lfm2_v7_target["prediction_set"]["singleton_correct"],
        },
        "truth pass_or_partial_pass inside singleton prediction set",
        lfm2_v7_target["truth_label"] == "pass_or_partial_pass"
        and lfm2_v7_target["prediction_set"]["labels"] == ["pass_or_partial_pass"]
        and lfm2_v7_target["prediction_set"]["contains_truth"] is True
        and lfm2_v7_target["prediction_set"]["singleton_correct"] is True,
    )

    invariance_contract_file = "cti_invariance_contract_20260609.json"
    invariance_contract = load_result(invariance_contract_file)
    invariance_claims = {
        row["claim"]: row for row in invariance_contract["claim_ladder"]
    }
    verifier.add(
        "CTI invariance contract is active but not final validation",
        invariance_contract_file,
        invariance_contract["status"],
        "scope_contract_active_not_final_validation",
        invariance_contract["status"] == "scope_contract_active_not_final_validation",
    )
    verifier.add(
        "CTI invariance contract preserves the claim ladder statuses",
        invariance_contract_file,
        {row["claim"]: row["status"] for row in invariance_contract["claim_ladder"]},
        "source law supported, local field supported, decision-margin mechanism scoped, invariance rejected, v7 provisional, downstream unproven",
        invariance_claims[
            "Within the source NLP-decoder scope, the EVT-derived functional law is strong."
        ]["status"]
        == "supported"
        and invariance_claims[
            "A coupled local competition field repairs dataset and family transfer better than centroid-only geometry."
        ]["status"]
        == "supported_with_scope"
        and invariance_claims[
            "Exact held-out decision-margin surgery passes source-family and fresh-family fixed-dose local mechanism gates."
        ]["status"]
        == "mechanism_candidate_supported_with_fresh_family_replication"
        and invariance_claims["Uncalibrated q-free architecture invariance is false."][
            "status"
        ]
        == "rejected"
        and invariance_claims[
            "The frozen v7 q-free prediction-set protocol is a promising scope contract."
        ]["status"]
        == "provisional"
        and invariance_claims[
            "CTI predicts public LLM downstream benchmark generalization."
        ]["status"]
        == "unproven",
    )
    verifier.add(
        "CTI invariance contract marks downstream benchmark prediction unproven",
        invariance_contract_file,
        invariance_claims["CTI predicts public LLM downstream benchmark generalization."],
        "downstream benchmark generalization has no current verified bridge",
        invariance_claims["CTI predicts public LLM downstream benchmark generalization."][
            "status"
        ]
        == "unproven"
        and invariance_claims["CTI predicts public LLM downstream benchmark generalization."][
            "strict_metric"
        ]["current_verified_bridge"]
        is None,
    )
    verifier.add(
        "CTI invariance contract keeps v7 provisional after LFM2",
        invariance_contract_file,
        invariance_claims[
            "The frozen v7 q-free prediction-set protocol is a promising scope contract."
        ]["strict_metric"],
        "LFM2 is one fresh singleton success; v7 remains provisional",
        invariance_claims[
            "The frozen v7 q-free prediction-set protocol is a promising scope contract."
        ]["status"]
        == "provisional"
        and invariance_claims[
            "The frozen v7 q-free prediction-set protocol is a promising scope contract."
        ]["strict_metric"][
            "fresh_singleton_successes"
        ]
        == ["LFM2-350M"],
    )
    verifier.add(
        "CTI invariance contract blocks known overclaims",
        invariance_contract_file,
        invariance_contract["ship_readiness"]["do_not_cite"],
        "full architecture transfer, downstream prediction, v7 validation, local-field replacement, out-of-scope causality",
        invariance_contract["ship_readiness"]["do_not_cite"]
        == [
            "full architecture-invariant transfer",
            "downstream benchmark prediction",
            "v7 validation",
            "local field replaces kappa",
            "local field causality outside the fixed-dose decision-margin scope",
        ],
    )

    benchmark_bridge_file = "cti_downstream_benchmark_bridge_preregistration_20260609.json"
    benchmark_bridge = load_result(benchmark_bridge_file)
    verifier.add(
        "CTI downstream benchmark bridge is preregistered but not run",
        benchmark_bridge_file,
        benchmark_bridge["status"],
        "preregistered_not_run",
        benchmark_bridge["status"] == "preregistered_not_run",
    )
    verifier.add(
        "CTI downstream benchmark bridge forbids a public claim before validation",
        benchmark_bridge_file,
        benchmark_bridge["claim_boundary"],
        "public_claim_allowed_now false and current_verified_bridge null",
        benchmark_bridge["claim_boundary"]["public_claim_allowed_now"] is False
        and benchmark_bridge["claim_boundary"]["current_verified_bridge"] is None,
    )
    verifier.add(
        "CTI downstream benchmark bridge scopes H3 as internal-only",
        benchmark_bridge_file,
        benchmark_bridge["internal_signal"],
        "Banking77 internal_downstream_signal_only with rho>=0.80 and p<=0.01",
        benchmark_bridge["internal_signal"]["dataset"] == "Banking77"
        and benchmark_bridge["internal_signal"]["status"]
        == "internal_downstream_signal_only"
        and benchmark_bridge["internal_signal"]["spearman_rho"] >= 0.80
        and benchmark_bridge["internal_signal"]["p_value"] <= 0.01,
    )
    verifier.add(
        "CTI downstream benchmark bridge freezes the six-task public panel",
        benchmark_bridge_file,
        [task["task_id"] for task in benchmark_bridge["external_panel"]["tasks"]],
        "ifeval, bbh, math_lvl_5, gpqa, musr, mmlu_pro",
        [task["task_id"] for task in benchmark_bridge["external_panel"]["tasks"]]
        == ["ifeval", "bbh", "math_lvl_5", "gpqa", "musr", "mmlu_pro"],
    )
    verifier.add(
        "CTI downstream benchmark bridge uses the lm-eval leaderboard task group",
        benchmark_bridge_file,
        {
            "task_group": benchmark_bridge["external_panel"]["task_group"],
            "command": benchmark_bridge["commands"]["leaderboard_group_run"],
        },
        "leaderboard task group in command template",
        benchmark_bridge["external_panel"]["task_group"] == "leaderboard"
        and "--tasks leaderboard"
        in benchmark_bridge["commands"]["leaderboard_group_run"],
    )
    verifier.add(
        "CTI downstream benchmark bridge freezes strict family-transfer thresholds",
        benchmark_bridge_file,
        benchmark_bridge["primary_success_criteria"],
        "12 models, 4 families, rho>=0.60, LOFO macro>=0.35, partial>=0.30",
        benchmark_bridge["primary_success_criteria"]["minimum_complete_models"] >= 12
        and benchmark_bridge["primary_success_criteria"]["minimum_complete_families"] >= 4
        and benchmark_bridge["primary_success_criteria"]["total_panel_spearman_rho"]
        == ">= 0.60"
        and benchmark_bridge["primary_success_criteria"][
            "leave_one_family_out_macro_spearman"
        ]
        == ">= 0.35"
        and benchmark_bridge["primary_success_criteria"][
            "partial_spearman_after_log_params"
        ]
        == ">= 0.30",
    )
    verifier.add(
        "CTI downstream benchmark bridge preflight records runnable status",
        benchmark_bridge_file,
        benchmark_bridge["preflight"],
        "can_run_public_panel_here is boolean and cli smoke result is recorded",
        isinstance(benchmark_bridge["preflight"]["can_run_public_panel_here"], bool)
        and "lm_eval_cli_smoke" in benchmark_bridge["preflight"],
    )

    nobel_ladder_file = "cti_nobel_trial_ladder_20260609.json"
    nobel_ladder = load_result(nobel_ladder_file)
    verifier.add(
        "CTI Nobel trial ladder is active but not validation",
        nobel_ladder_file,
        nobel_ladder["status"],
        "active_question_first_trial_plan_not_validation",
        nobel_ladder["status"] == "active_question_first_trial_plan_not_validation",
    )
    verifier.add(
        "CTI Nobel trial ladder keeps a substantial question-first budget",
        nobel_ladder_file,
        nobel_ladder["question_first_budget"],
        "at least 6 clusters and 25 questions",
        nobel_ladder["question_first_budget"]["question_clusters"] >= 6
        and nobel_ladder["question_first_budget"]["total_questions"] >= 25,
    )
    verifier.add(
        "CTI Nobel trial ladder freezes the six decisive trial IDs",
        nobel_ladder_file,
        [trial["trial_id"] for trial in nobel_ladder["trial_ladder"]],
        "public downstream, causal local field, Aristotelian null, theory, scope, natural shift",
        [trial["trial_id"] for trial in nobel_ladder["trial_ladder"]]
        == [
            "public_downstream_family_transfer",
            "local_field_causal_surgery",
            "aristotelian_permutation_null_alignment",
            "point_process_evt_derivation",
            "fresh_scope_abstention_or_warning",
            "natural_shift_confusion_forecast",
        ],
    )
    first_trial = nobel_ladder["trial_ladder"][0]
    verifier.add(
        "CTI Nobel trial ladder anchors the first trial to the public benchmark bridge",
        nobel_ladder_file,
        first_trial,
        "bridge preregistered, public claim false, eval environment blocker recorded",
        first_trial["trial_id"] == "public_downstream_family_transfer"
        and first_trial["current_evidence"]["bridge_status"] == "preregistered_not_run"
        and first_trial["current_evidence"]["public_claim_allowed_now"] is False
        and first_trial["current_evidence"]["preflight_blocker"]
        == "peft_transformers_HybridCache_import_mismatch",
    )
    causal_trial = nobel_ladder["trial_ladder"][1]
    verifier.add(
        "CTI Nobel trial ladder scopes the fixed-dose mechanism pass",
        nobel_ladder_file,
        {
            "trial_id": causal_trial["trial_id"],
            "pass_rule": causal_trial["pass_rule"],
            "kill_rule": causal_trial["kill_rule"],
        },
        "current and fresh fixed-1.5 gates passed, source and fresh 1.25 gates passed, threshold bracket retained",
        causal_trial["trial_id"] == "local_field_causal_surgery"
        and causal_trial["pass_rule"]["current_fixed_1p5_gate"]["passed"] is True
        and causal_trial["pass_rule"]["current_fixed_1p5_gate"][
            "target_arm_direction_accuracy"
        ]
        == 1.0
        and causal_trial["pass_rule"]["fresh_family_fixed_1p5_gate"]["passed"] is True
        and causal_trial["pass_rule"]["fresh_family_fixed_1p5_gate"][
            "fixed_strength"
        ]
        == 1.5
        and causal_trial["pass_rule"]["fresh_family_fixed_1p5_gate"][
            "target_arm_direction_accuracy"
        ]
        == 1.0
        and causal_trial["pass_rule"]["fresh_family_fixed_1p5_gate"][
            "matched_null_compete_rate"
        ]
        == 0.0
        and causal_trial["pass_rule"]["source_fixed_1p25_gate"]["passed"] is True
        and causal_trial["pass_rule"]["source_fixed_1p25_gate"]["fixed_strength"]
        == 1.25
        and causal_trial["pass_rule"]["fresh_family_fixed_1p25_gate"]["passed"]
        is True
        and causal_trial["pass_rule"]["fresh_family_fixed_1p25_gate"][
            "fixed_strength"
        ]
        == 1.25
        and causal_trial["current_evidence"]["dose_transition_bracket"]
        == "(1.0, 1.25]"
        and "threshold trial inside (1.0, 1.25]" in causal_trial["kill_rule"],
    )
    verifier.add(
        "CTI Nobel trial ladder preserves public overclaim boundaries",
        nobel_ladder_file,
        nobel_ladder["decision_policy"]["do_not_ship"],
        "no full architecture transfer, public downstream prediction, v7 validation, or universal local-field causality",
        nobel_ladder["decision_policy"]["do_not_ship"]
        == [
            "full architecture-invariant transfer",
            "public downstream benchmark prediction",
            "v7 validation",
            "local field as universal causal mechanism outside the replicated fixed-dose decision-margin scope",
        ],
    )
    external_urls = [
        source["url"]
        for source in nobel_ladder["sources"]["external_literature"]
    ]
    verifier.add(
        "CTI Nobel trial ladder includes current external geometry and benchmark pressure sources",
        nobel_ladder_file,
        external_urls,
        "Aristotelian, LID, Representation Gap, downstream metrics, leaderboard, lm-eval, gated attention",
        external_urls
        == [
            "https://arxiv.org/abs/2602.14486",
            "https://arxiv.org/abs/2601.22722",
            "https://arxiv.org/abs/2605.21692",
            "https://arxiv.org/abs/2512.08894",
            "https://huggingface.co/docs/leaderboards/main/open_llm_leaderboard/about",
            "https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/leaderboard",
            "https://arxiv.org/abs/2505.06708",
        ],
    )

    local_surgery_file = "cti_local_field_causal_surgery_preregistration_20260609.json"
    local_surgery = load_result(local_surgery_file)
    verifier.add(
        "CTI local-field causal surgery is preregistered but not causal validation",
        local_surgery_file,
        local_surgery["status"],
        "preregistered_with_observational_feasibility_not_causal_validation",
        local_surgery["status"]
        == "preregistered_with_observational_feasibility_not_causal_validation",
    )
    verifier.add(
        "CTI local-field causal surgery forbids causal overclaim",
        local_surgery_file,
        local_surgery["claim_boundary"],
        "causal_claim_allowed_now false",
        local_surgery["claim_boundary"]["causal_claim_allowed_now"] is False
        and local_surgery["claim_boundary"]["forbidden_sentence"]
        == "The local field is proven causal.",
    )
    intervention = local_surgery["preregistered_intervention"]
    verifier.add(
        "CTI local-field causal surgery freezes three intervention arms",
        local_surgery_file,
        [arm["name"] for arm in intervention["arms"]],
        "safe raise, safe lower, matched-norm null",
        [arm["name"] for arm in intervention["arms"]]
        == [
            "safe_raise_zero_sum_boundary_surgery",
            "safe_lower_zero_sum_boundary_surgery",
            "matched_norm_null_surgery",
        ],
    )
    verifier.add(
        "CTI local-field causal surgery preserves kappa matching and local-field movement filters",
        local_surgery_file,
        intervention["acceptance_filters"],
        "kappa <=0.02, local movement >=0.10, at least 30 contrasts",
        intervention["acceptance_filters"]["abs_delta_kappa_nearest"] == "<= 0.02"
        and intervention["acceptance_filters"][
            "abs_delta_local_safe_rate_or_margin_z"
        ]
        == ">= 0.10"
        and intervention["acceptance_filters"]["minimum_paired_contrasts"] >= 30,
    )
    feasibility = local_surgery["observational_feasibility_audit"]
    primary_pairs = feasibility["primary_matched_signal"]["independent_greedy_pairs"]
    verifier.add(
        "CTI local-field causal surgery has enough kappa-matched safe-rate feasibility pairs",
        local_surgery_file,
        primary_pairs,
        "n>=30, sign accuracy>=0.95, one-sided p<=1e-10",
        primary_pairs["n"] >= 30
        and primary_pairs["sign_accuracy"] >= 0.95
        and primary_pairs["binomial_one_sided_p"] <= 1e-10,
    )
    secondary_pairs = feasibility["secondary_matched_signal"][
        "independent_greedy_pairs"
    ]
    verifier.add(
        "CTI local-field causal surgery has a lower-tail margin feasibility signal",
        local_surgery_file,
        secondary_pairs,
        "n>=30, sign accuracy>=0.90, one-sided p<=1e-7",
        secondary_pairs["n"] >= 30
        and secondary_pairs["sign_accuracy"] >= 0.90
        and secondary_pairs["binomial_one_sided_p"] <= 1e-7,
    )
    lid_pairs = feasibility["lid_proxy_control"]["independent_greedy_pairs"]
    verifier.add(
        "CTI local-field causal surgery LID proxy control does not explain the direction",
        local_surgery_file,
        lid_pairs,
        "LID proxy sign accuracy<=0.60",
        lid_pairs["sign_accuracy"] <= 0.60,
    )
    verifier.add(
        "CTI local-field causal surgery decision names forbidden claims",
        local_surgery_file,
        local_surgery["decision"]["do_not_claim"],
        "do not claim local field is causal before surgery passes",
        "local field is causal" in local_surgery["decision"]["do_not_claim"]
        and "observational matched pairs are independent causal evidence"
        in local_surgery["decision"]["do_not_claim"],
    )

    zero_sum_summary_file = "cti_local_field_zero_sum_surgery_pilot_summary_20260609.json"
    zero_sum_summary = load_result(zero_sum_summary_file)
    verifier.add(
        "CTI zero-sum local-field surgery pilot is negative or ambiguous",
        zero_sum_summary_file,
        zero_sum_summary["status"],
        "pilot_negative_or_ambiguous_not_trial_pass",
        zero_sum_summary["status"] == "pilot_negative_or_ambiguous_not_trial_pass",
    )
    verifier.add(
        "CTI zero-sum local-field surgery pilot blocks causal promotion",
        zero_sum_summary_file,
        zero_sum_summary["claim_boundary"],
        "trial_2_passed false and causal_claim_allowed_now false",
        zero_sum_summary["claim_boundary"]["trial_2_passed"] is False
        and zero_sum_summary["claim_boundary"]["causal_claim_allowed_now"] is False,
    )
    zero_sum_aggregate = zero_sum_summary["aggregate"]
    verifier.add(
        "CTI zero-sum local-field surgery pilot ran three real pilots",
        zero_sum_summary_file,
        zero_sum_aggregate,
        "3 pilots, centroid preserved in all target arms, kappa window held in all target arms",
        zero_sum_aggregate["n_pilots"] == 3
        and zero_sum_aggregate["target_centroid_preserved_count"] == 3
        and zero_sum_aggregate["target_kappa_window_count"] == 3,
    )
    verifier.add(
        "CTI zero-sum local-field surgery pilot did not achieve bidirectionality",
        zero_sum_summary_file,
        zero_sum_aggregate,
        "bidirectional target count is 0 and safe-lower negative q count is 0",
        zero_sum_aggregate["bidirectional_target_count"] == 0
        and zero_sum_aggregate["safe_lower_negative_q_count"] == 0,
    )
    verifier.add(
        "CTI zero-sum local-field surgery pilot records null competition",
        zero_sum_summary_file,
        zero_sum_aggregate,
        "matched-norm null competes in at least two pilots",
        zero_sum_aggregate["matched_norm_null_competes_count"] >= 2,
    )
    verifier.add(
        "CTI zero-sum local-field surgery pilot decision forbids Trial 2 promotion",
        zero_sum_summary_file,
        zero_sum_summary["decision"],
        "do_not_promote_trial_2 true and bad causal claims forbidden",
        zero_sum_summary["decision"]["do_not_promote_trial_2"] is True
        and "local field causal surgery passed"
        in zero_sum_summary["decision"]["do_not_claim"],
    )

    decision_margin_file = "cti_decision_margin_surgery_batch_20260609.json"
    decision_margin = load_result(decision_margin_file)
    verifier.add(
        "CTI decision-margin surgery batch passes only the single-model gate",
        decision_margin_file,
        decision_margin["status"],
        "single_model_batch_pass_not_cross_family_causal_validation",
        decision_margin["status"]
        == "single_model_batch_pass_not_cross_family_causal_validation",
    )
    verifier.add(
        "CTI decision-margin surgery batch blocks cross-family causal overclaim",
        decision_margin_file,
        decision_margin["claim_boundary"],
        "trial_2_passed false and causal_claim_allowed_now false",
        decision_margin["claim_boundary"]["trial_2_passed"] is False
        and decision_margin["claim_boundary"]["causal_claim_allowed_now"] is False,
    )
    decision_margin_aggregate = decision_margin["aggregate"]
    verifier.add(
        "CTI decision-margin surgery batch has at least 30 accepted contrasts",
        decision_margin_file,
        decision_margin_aggregate,
        "36 attempted, 35 accepted, 1 rejected by acceptance filter, 0 failures",
        decision_margin_aggregate["n_attempted_contrasts"] == 36
        and decision_margin_aggregate["n_accepted_contrasts"] >= 30
        and decision_margin_aggregate["n_rejected_by_acceptance_filter"] == 1
        and decision_margin_aggregate["n_failures"] == 0,
    )
    verifier.add(
        "CTI decision-margin surgery batch is bidirectional on accepted contrasts",
        decision_margin_file,
        decision_margin_aggregate,
        "direction accuracy 1.0 and bidirectional success rate 1.0",
        decision_margin_aggregate["target_arm_direction_accuracy"] == 1.0
        and decision_margin_aggregate["bidirectional_success_rate"] == 1.0,
    )
    verifier.add(
        "CTI decision-margin surgery batch beats matched decision-row nulls",
        decision_margin_file,
        decision_margin_aggregate,
        "matched-null compete rate 0.0",
        decision_margin_aggregate["matched_null_compete_rate"] == 0.0,
    )
    verifier.add(
        "CTI decision-margin surgery batch preserves single-model claim scope",
        decision_margin_file,
        decision_margin["decision"],
        "promote to cross-family trial but do not claim cross-family causality",
        decision_margin["decision"]["promote_to_cross_family_trial"] is True
        and "cross-family CTI local-field causality is proven"
        in decision_margin["decision"]["do_not_claim"],
    )

    cross_family_grid_file = "cti_decision_margin_cross_family_trial_20260609.json"
    cross_family_grid = load_result(cross_family_grid_file)
    grid_aggregate = cross_family_grid["aggregate"]
    grid_uses_global_strength_1p5 = all(
        row["best_arms"]["decision_margin_raise_surgery"]["strength"] == 1.5
        and row["best_arms"]["decision_margin_lower_surgery"]["strength"] == 1.5
        for row in cross_family_grid["contrasts"]
    )
    verifier.add(
        "CTI cross-family decision-margin grid trial passes with a global strength",
        cross_family_grid_file,
        {
            "status": cross_family_grid["status"],
            "aggregate": grid_aggregate,
            "grid_uses_global_strength_1p5": grid_uses_global_strength_1p5,
        },
        "108 accepted, 4 families, direction accuracy 1.0, null compete 0.0, all best strengths 1.5",
        cross_family_grid["status"]
        == "cross_family_decision_margin_trial_pass_mechanism_candidate"
        and grid_aggregate["n_attempted_contrasts"] == 108
        and grid_aggregate["n_accepted_contrasts"] == 108
        and grid_aggregate["n_model_families_accepted"] == 4
        and grid_aggregate["target_arm_direction_accuracy"] == 1.0
        and grid_aggregate["matched_null_compete_rate"] == 0.0
        and grid_uses_global_strength_1p5,
    )

    fixed_1p0_file = "cti_decision_margin_cross_family_fixed_strength_20260609.json"
    fixed_1p0 = load_result(fixed_1p0_file)
    fixed_1p0_aggregate = fixed_1p0["aggregate"]
    verifier.add(
        "CTI fixed-strength 1.0 cross-family audit remains negative",
        fixed_1p0_file,
        fixed_1p0_aggregate,
        "accepted 108 but bidirectional rate <0.70 and null compete rate >0.20",
        fixed_1p0["status"] == "cross_family_decision_margin_trial_negative_or_ambiguous"
        and fixed_1p0_aggregate["n_accepted_contrasts"] == 108
        and fixed_1p0_aggregate["target_arm_direction_accuracy"] >= 0.80
        and fixed_1p0_aggregate["bidirectional_success_rate"] < 0.70
        and fixed_1p0_aggregate["matched_null_compete_rate"] > 0.20
        and fixed_1p0_aggregate["cross_family_trial_pass"] is False,
    )

    fixed_1p5_file = "cti_decision_margin_cross_family_fixed_strength_1p5_20260609.json"
    fixed_1p5 = load_result(fixed_1p5_file)
    fixed_1p5_aggregate = fixed_1p5["aggregate"]
    verifier.add(
        "CTI fixed-strength 1.5 cross-family mechanism gate passes",
        fixed_1p5_file,
        fixed_1p5["status"],
        "cross_family_decision_margin_fixed_strength_trial_pass_mechanism_candidate",
        fixed_1p5["status"]
        == "cross_family_decision_margin_fixed_strength_trial_pass_mechanism_candidate",
    )
    verifier.add(
        "CTI fixed-strength 1.5 cross-family trial has balanced accepted coverage",
        fixed_1p5_file,
        fixed_1p5_aggregate,
        "108 accepted, 0 failures, 4 accepted families, max family share 0.25",
        fixed_1p5_aggregate["n_attempted_contrasts"] == 108
        and fixed_1p5_aggregate["n_accepted_contrasts"] == 108
        and fixed_1p5_aggregate["n_failures"] == 0
        and fixed_1p5_aggregate["n_model_families_accepted"] == 4
        and fixed_1p5_aggregate["max_accepted_family_share"] == 0.25,
    )
    verifier.add(
        "CTI fixed-strength 1.5 cross-family trial is bidirectional with no null competition",
        fixed_1p5_file,
        fixed_1p5_aggregate,
        "direction accuracy 1.0, bidirectional rate 1.0, null compete rate 0.0",
        fixed_1p5_aggregate["raise_direction_matches"] == 108
        and fixed_1p5_aggregate["lower_direction_matches"] == 108
        and fixed_1p5_aggregate["target_arm_direction_accuracy"] == 1.0
        and fixed_1p5_aggregate["bidirectional_success_rate"] == 1.0
        and fixed_1p5_aggregate["matched_null_compete_rate"] == 0.0
        and fixed_1p5_aggregate["cross_family_trial_pass"] is True,
    )
    verifier.add(
        "CTI fixed-strength 1.5 result preserves broader claim limits",
        fixed_1p5_file,
        fixed_1p5["claim_boundary"],
        "local mechanism candidate allowed, universal law and public downstream proof false",
        fixed_1p5["claim_boundary"]["trial_2_decision_margin_passed"] is True
        and fixed_1p5["claim_boundary"][
            "local_field_causal_mechanism_candidate_allowed"
        ]
        is True
        and fixed_1p5["claim_boundary"]["universal_cti_law_proven"] is False
        and fixed_1p5["claim_boundary"]["public_downstream_transfer_proven"] is False,
    )

    fresh_1p5_file = "cti_decision_margin_fresh_family_fixed_strength_1p5_20260609.json"
    fresh_1p5 = load_result(fresh_1p5_file)
    fresh_1p5_aggregate = fresh_1p5["aggregate"]
    fresh_families = sorted(fresh_1p5_aggregate["accepted_family_counts"])
    verifier.add(
        "CTI fresh-family fixed-strength 1.5 mechanism replication passes",
        fresh_1p5_file,
        fresh_1p5["status"],
        "cross_family_decision_margin_fixed_strength_trial_pass_mechanism_candidate",
        fresh_1p5["status"]
        == "cross_family_decision_margin_fixed_strength_trial_pass_mechanism_candidate",
    )
    verifier.add(
        "CTI fresh-family fixed-strength 1.5 replication uses the intended frozen scope",
        fresh_1p5_file,
        {
            "scope": fresh_1p5["scope"],
            "accepted_families": fresh_families,
        },
        "fixed strength 1.5 over qwen3, tinyllama, mistral, and rwkv",
        fresh_1p5["scope"]["selection_mode"] == "fixed_strength"
        and fresh_1p5["scope"]["fixed_strength"] == 1.5
        and fresh_1p5["scope"]["n_sample_requested"] == 600
        and fresh_1p5["scope"]["n_select"] == 40
        and fresh_families == ["mistral", "qwen3", "rwkv", "tinyllama"],
    )
    verifier.add(
        "CTI fresh-family fixed-strength 1.5 replication has balanced accepted coverage",
        fresh_1p5_file,
        fresh_1p5_aggregate,
        "108 accepted, 0 failures, 4 accepted families, max family share 0.25",
        fresh_1p5_aggregate["n_attempted_contrasts"] == 108
        and fresh_1p5_aggregate["n_accepted_contrasts"] == 108
        and fresh_1p5_aggregate["n_failures"] == 0
        and fresh_1p5_aggregate["n_model_families_accepted"] == 4
        and fresh_1p5_aggregate["max_accepted_family_share"] == 0.25,
    )
    verifier.add(
        "CTI fresh-family fixed-strength 1.5 replication is bidirectional with no null competition",
        fresh_1p5_file,
        fresh_1p5_aggregate,
        "direction accuracy 1.0, bidirectional rate 1.0, null compete rate 0.0",
        fresh_1p5_aggregate["raise_direction_matches"] == 108
        and fresh_1p5_aggregate["lower_direction_matches"] == 108
        and fresh_1p5_aggregate["target_arm_direction_accuracy"] == 1.0
        and fresh_1p5_aggregate["bidirectional_success_rate"] == 1.0
        and fresh_1p5_aggregate["matched_null_compete_rate"] == 0.0
        and fresh_1p5_aggregate["cross_family_trial_pass"] is True,
    )
    verifier.add(
        "CTI fresh-family fixed-strength 1.5 result preserves broader claim limits",
        fresh_1p5_file,
        fresh_1p5["claim_boundary"],
        "local mechanism candidate allowed, universal law and public downstream proof false",
        fresh_1p5["claim_boundary"]["trial_2_decision_margin_passed"] is True
        and fresh_1p5["claim_boundary"][
            "local_field_causal_mechanism_candidate_allowed"
        ]
        is True
        and fresh_1p5["claim_boundary"]["universal_cti_law_proven"] is False
        and fresh_1p5["claim_boundary"]["public_downstream_transfer_proven"] is False,
    )

    fixed_1p25_file = "cti_decision_margin_cross_family_fixed_strength_1p25_20260609.json"
    fixed_1p25 = load_result(fixed_1p25_file)
    fixed_1p25_aggregate = fixed_1p25["aggregate"]
    verifier.add(
        "CTI fixed-strength 1.25 source-family dose-boundary audit passes",
        fixed_1p25_file,
        fixed_1p25_aggregate,
        "108 accepted, 4 families, direction accuracy 1.0, null compete rate 0.0",
        fixed_1p25["status"]
        == "cross_family_decision_margin_fixed_strength_trial_pass_mechanism_candidate"
        and fixed_1p25["scope"]["fixed_strength"] == 1.25
        and fixed_1p25_aggregate["n_accepted_contrasts"] == 108
        and fixed_1p25_aggregate["n_failures"] == 0
        and fixed_1p25_aggregate["n_model_families_accepted"] == 4
        and fixed_1p25_aggregate["target_arm_direction_accuracy"] == 1.0
        and fixed_1p25_aggregate["bidirectional_success_rate"] == 1.0
        and fixed_1p25_aggregate["matched_null_compete_rate"] == 0.0
        and fixed_1p25_aggregate["cross_family_trial_pass"] is True,
    )

    fresh_1p25_file = "cti_decision_margin_fresh_family_fixed_strength_1p25_20260609.json"
    fresh_1p25 = load_result(fresh_1p25_file)
    fresh_1p25_aggregate = fresh_1p25["aggregate"]
    fresh_1p25_families = sorted(fresh_1p25_aggregate["accepted_family_counts"])
    verifier.add(
        "CTI fixed-strength 1.25 fresh-family dose-boundary audit passes",
        fresh_1p25_file,
        {
            "aggregate": fresh_1p25_aggregate,
            "accepted_families": fresh_1p25_families,
        },
        "108 accepted across qwen3, tinyllama, mistral, rwkv with no null competition",
        fresh_1p25["status"]
        == "cross_family_decision_margin_fixed_strength_trial_pass_mechanism_candidate"
        and fresh_1p25["scope"]["fixed_strength"] == 1.25
        and fresh_1p25_aggregate["n_accepted_contrasts"] == 108
        and fresh_1p25_aggregate["n_failures"] == 0
        and fresh_1p25_aggregate["n_model_families_accepted"] == 4
        and fresh_1p25_families == ["mistral", "qwen3", "rwkv", "tinyllama"]
        and fresh_1p25_aggregate["target_arm_direction_accuracy"] == 1.0
        and fresh_1p25_aggregate["bidirectional_success_rate"] == 1.0
        and fresh_1p25_aggregate["matched_null_compete_rate"] == 0.0
        and fresh_1p25_aggregate["cross_family_trial_pass"] is True,
    )

    return verifier


def print_text_report(verifier: ClaimVerifier) -> None:
    print("CTI core-claim verification")
    print(f"checks: {len(verifier.checks)}")
    print(f"failed: {len(verifier.failed)}")
    print()
    for check in verifier.checks:
        print(f"[{check['status']}] {check['claim']}")
        print(f"  source: {check['source']}")
        print(f"  observed: {check['observed']}")
        print(f"  expected: {check['expected']}")
    print()
    if verifier.failed:
        print("VERDICT: FAIL")
    else:
        print("VERDICT: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    verifier = build_verification()
    if args.json:
        print(
            json.dumps(
                {
                    "status": "fail" if verifier.failed else "pass",
                    "n_checks": len(verifier.checks),
                    "n_failed": len(verifier.failed),
                    "checks": verifier.checks,
                },
                indent=2,
            )
        )
    else:
        print_text_report(verifier)
    return 1 if verifier.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
