# CTI Universal Law -- Claim-to-Evidence Checklist

## Status: Updated Mar 3, 2026

Every claim in the paper abstract and main text mapped to supporting result files.

---

## Abstract Claims

| # | Claim | Evidence File | Status |
|---|-------|---------------|--------|
| 1 | Functional form derived from EVT/Gumbel-race (conditional theorem) | `research/OBSERVABLE_ORDER_PARAMETER_THEOREM.md` | DERIVED |
| 2 | LOAO 12 NLP archs: alpha=1.477, CV=0.023, R^2=0.955 | `results/cti_kappa_loao_per_dataset.json` | VERIFIED |
| 3 | RWKV-4 boundary: alpha=2.887 in [2.43, 3.29] | `results/cti_kappa_loao_per_dataset.json` (RWKV entry) | VERIFIED |
| 4 | ViT-Large R^2=0.964 cross-modal | `results/cti_vit_cross_modality.json` | VERIFIED |
| 5 | Causal confusion-matrix: r=0.842-0.776, sign acc 93-100%, n=182, p<10^-35 | `results/cti_confusion_causal_prediction.json` | VERIFIED |
| 6 | Blind OOD: r=0.817, p=0.013 (SmolLM2-1.7B) | `results/cti_smollm2_ood_prediction.json` | VERIFIED |
| 7 | Biological 30/32 V1 sessions, mean r=0.736 | `results/cti_allen_all_sessions_complete.json` | VERIFIED |
| 8 | Multi-area batch: VISl 22/22, VISam 24/25 | `results/cti_allen_multiarea_batch.json` | VERIFIED |
| 9 | Near-simplex rho~0.46, CV=1.0% across 6 modalities (NLP, audio, vision, cortex) | `results/cti_cross_modal_rho.json`, `results/cti_allen_equicorr_multiarea.json`, `results/cti_cross_family_equicorr.json` | VERIFIED |
| 10 | Alpha-rho prediction: 1.540 vs 1.477 (+4.3%, zero params) | `results/cti_alpha_rho_multidataset.json` | VERIFIED |
| 11 | Per-model alpha CV (2.3%) consistent with estimation noise (expected 2.8%) | `results/cti_alpha_noise_analysis.json` | VERIFIED |
| 12 | Encoder alpha 4-5x higher for same rho | `results/cti_cross_family_equicorr.json`, `results/cti_encoder_loao.json` | VERIFIED |
| 13 | Honest scope: alpha varies by signal type (NLP 1.48, ViT 4.5, CNN 4.0, audio 4.7) | `results/cti_extended_family_loao.json`, `results/cti_alpha_family_law.json` | VERIFIED |
| 14 | 4 probe calibration reduces MAE by 86% | `results/cti_one_point_calibration.json` | VERIFIED |
| 14b | ResNet-50 CIFAR-100 same functional form | `results/cti_resnet50_cifar100.json` | VERIFIED |
| 15 | Generation law: kappa from W_U predicts logCE, r=-0.55 (n=22), r=-0.70 (n=21 excl. LFM outlier) | `results/cti_generation_law_analysis.json`, `results/cti_generation_hypothesis_scorecard.json` | VERIFIED |
| 16 | Fixed-V Transformer vs SSM: r=-0.84, architecture-independent (F p=0.147) | `results/cti_generation_law_analysis.json` | VERIFIED |
| 17 | beta_gen~0: vocabulary size drops out (K_eff~2-3) | `results/cti_generation_keff.json` | VERIFIED |
| 18 | Partial r controlling for log(N_params) = -0.546, p=0.013 | `results/cti_generation_law_analysis.json` | VERIFIED |
| 19 | Audio: 7 speech models, r=0.898, p=0.006, alpha_audio=4.669 | `results/cti_audio_speech.json` | VERIFIED |
| 20 | Audio-vision alpha convergence: alpha_audio (4.67) ~ alpha_CNN (4.4) | `results/cti_audio_speech.json` | VERIFIED |

## Main Text Claims

| # | Claim | Evidence File | Status |
|---|-------|---------------|--------|
| 15 | H8+ holdout: r=0.879, MAE=0.077, n=77, all 6 criteria PASS | `results/cti_utility_revised.json` | VERIFIED |
| 16 | LOMFO all 4 families: r>=0.84 each | `results/cti_lomfo_lodo_stress_test.json` | VERIFIED |
| 17 | LODO cross-dataset: mean r=0.125 (honest scope limit) | `results/cti_lomfo_lodo_stress_test.json` | VERIFIED |
| 18 | Do-interventions confirm causal direction | `results/cti_do_intervention_multi_arch.json`, `results/cti_do_intervention_text.json` | VERIFIED |
| 19 | Orthogonal factorial: kappa is causal driver | `results/cti_orthogonal_factorial.json` | VERIFIED |
| 20 | H3 ranking: 9 models, rho=0.833, p=0.005 (kappa ranks by MAP@10) | `results/cti_downstream_h3_n9.json` | VERIFIED |
| 21 | Multi-area: VISp 30/30, all areas >=87% pass | `results/cti_allen_multiarea_batch.json` | VERIFIED |
| 22 | Equicorrelation: rho area-invariant (0.43-0.46) | `results/cti_allen_equicorr_multiarea.json` | VERIFIED |
| 23 | Three-level universality (form/constant/intercept) | `results/cti_lomfo_lodo_stress_test.json`, `results/cti_extended_family_loao.json` | VERIFIED |
| 24 | Alpha-rho: MAE=0.068 PASS, bootstrap reliability=0.998 | `results/cti_alpha_rho_multidataset.json` | VERIFIED |
| 25 | Encoder LOAO: CV=0.42 (confirms decoder-only universality) | `results/cti_encoder_loao.json` | VERIFIED |
| 26 | kappa not a model-size proxy (gamma=0.003, p=0.91) | `results/cti_scaling_dynamics.json` | VERIFIED |
| 27 | Simplex ETF gives rho=1/2, alpha_simplex=1.595 | `research/OBSERVABLE_ORDER_PARAMETER_THEOREM.md` (analytic) | DERIVED |
| 28 | Beta_infinity=1.0 renormalized to 0.746 | `results/cti_kappa_loao_per_dataset.json` | VERIFIED |

## Biological Claims

| # | Claim | Evidence File | Status |
|---|-------|---------------|--------|
| 29 | Allen Neuropixels: K=118, 32 sessions, 50-250ms window | `results/cti_allen_all_sessions_complete.json` | VERIFIED |
| 30 | Multi-area: VISp, VISl, VISal, VISam, VISrl all PASS | `results/cti_allen_multiarea_batch.json` | VERIFIED |
| 31 | Rho~0.466 in 5 cortical areas (CV=1.65%) | `results/cti_allen_equicorr_multiarea.json` | VERIFIED |
| 32 | Cadieu V4 < IT hierarchy gradient | `results/cti_neuroscience_cadieu2014.json` | VERIFIED |

## Theory Claims

| # | Claim | Evidence File | Status |
|---|-------|---------------|--------|
| 33 | Gumbel-race competition gives logit-linear form | `research/OBSERVABLE_ORDER_PARAMETER_THEOREM.md` (Theorem 1) | DERIVED |
| 34 | d_eff_comp = 1/(1-rho) from whitened cosine structure | `research/OBSERVABLE_ORDER_PARAMETER_THEOREM.md` | DERIVED |
| 35 | KS test: Gumbel fit fails at d=200, but logit-linear margin passes (p=0.265) | `results/cti_gumbel_theory.json` | VERIFIED |
| 35b | Synthetic Gumbel validation: logit-linear form R^2=0.982 under controlled Gaussian data | `results/cti_synthetic_gumbel_validation.json` | VERIFIED |
| 35c | Synthetic: alpha(rho) scaling requires anisotropic noise (not derivable from isotropic Gaussians) | `results/cti_synthetic_gumbel_validation.json` | VERIFIED |

## Scope Limits (Honest Negatives)

| # | Claim | Evidence File | Status |
|---|-------|---------------|--------|
| 36 | Protein LMs: law FAILS (alpha=-1.17, r=-0.15, p=0.76, 7 models, 3 families) | `results/cti_protein_esm2.json` | VERIFIED |
| 37 | Human fMRI (NSD): null result (chance-level 1-NN, pooled r=0.12, p=0.18) | `results/cti_nsd_human_fmri.json` | VERIFIED |
| 38 | Encoder LOAO: CV_alpha=0.42 (decoders universal, encoders NOT) | `results/cti_encoder_loao.json` | VERIFIED |
| 39 | LODO cross-dataset: mean r=0.125 (intercept is task-specific) | `results/cti_lomfo_lodo_stress_test.json` | VERIFIED |

---

## Pre-Submission Checks

- [x] All alpha/CV/R^2 values verified against `cti_kappa_loao_per_dataset.json`
- [x] Causal r and p-values verified against `cti_confusion_causal_prediction.json`
- [x] Biological pass rates verified against `cti_allen_all_sessions_complete.json`
- [x] H8+ holdout all 6 criteria verified against `cti_utility_revised.json`
- [x] Alpha-rho numbers verified against `cti_alpha_rho_multidataset.json`
- [x] H3 ranking rho/p verified against `cti_downstream_h3_n9.json`
- [x] Audio speech results verified against `cti_audio_speech.json` (r=0.898, alpha=4.669)
- [x] Protein negative verified against `cti_protein_esm2.json` (alpha=-1.17, r=-0.15)
- [x] NSD fMRI null verified against `cti_nsd_human_fmri.json` (pooled r=0.12, p=0.18)
- [x] Generation law verified: r=-0.55 (n=22) / r=-0.70 (n=21 excl. LFM)
- [x] Synthetic Gumbel validation: R^2=0.982 logit-linear form confirmed
- [x] arXiv PDF compiles clean (33 pages, 0 overfull boxes)
- [x] COLM PDF compiles clean (14 pages, 0 overfull boxes)
