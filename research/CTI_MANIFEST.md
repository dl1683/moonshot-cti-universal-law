# CTI Universal Law -- Canonical File Manifest

## Purpose

This repo contains ~310 experiment scripts and ~324 result JSONs from iterative development.
This manifest lists only the canonical files that support the paper's claims.

---

## Canonical Result Files

| File | What It Proves | Paper Section |
|------|---------------|---------------|
| `results/cti_kappa_loao_per_dataset.json` | LOAO alpha=1.477, CV=2.3%, R^2=0.955 | Primary result |
| `results/cti_utility_revised.json` | H8+ holdout: r=0.879, MAE=0.077 | Holdout validation |
| `results/cti_allen_all_sessions_complete.json` | 30/32 mouse V1 sessions PASS | Biological |
| `results/cti_allen_multiarea_batch.json` | 30 mice, 5 cortical areas all PASS | Biological multi-area |
| `results/cti_allen_equicorr_multiarea.json` | rho=0.43-0.46 across 5 areas | Equicorrelation |
| `results/cti_confusion_causal_prediction.json` | r=0.842, sign acc 93%, p<10^-35 | Causal evidence |
| `results/cti_do_intervention_multi_arch.json` | Do-intervention confirms direction | Causal evidence |
| `results/cti_orthogonal_factorial.json` | Factorial design confirms kappa | Causal evidence |
| `results/cti_vit_cross_modality.json` | ViT-Large R^2=0.964 | Cross-modal |
| `results/cti_expanded_blind_ood.json` | Blind OOD r=0.817, p=0.013 | OOD validation |
| `results/cti_lomfo_lodo_stress_test.json` | LOMFO all 4 families r>=0.84 | Stress test |
| `results/cti_downstream_h3_n9.json` | H3 ranking rho=0.833, p=0.005 | Practical utility |
| `results/cti_downstream_protocol_v3.json` | H1/H2 layer-selection PASS | Practical utility |
| `results/cti_extended_family_loao.json` | Alpha-family law (NLP/ViT/CNN) | Family analysis |
| `results/cti_alpha_rho_multidataset.json` | Zero-param prediction +4.7% error | Theory connection |
| `results/cti_alpha_rho_derivation.json` | Single-dataset alpha-rho baseline | Theory connection |
| `results/cti_cross_family_equicorr.json` | Encoder rho same as decoder | Equicorrelation |
| `results/cti_encoder_loao.json` | Encoder CV=0.42 (not universal) | Scope limit |
| `results/cti_scaling_dynamics.json` | kappa not model-size proxy | E1/E2 controls |
| `results/cti_neuroscience_cadieu2014.json` | Cadieu V4 hierarchy gradient | Biological |

## Canonical Scripts

| Script | What It Runs |
|--------|-------------|
| `src/cti_kappa_nearest_universal.py` | Core LOAO fit |
| `src/cti_utility_revised.py` | H8+ holdout validation |
| `src/cti_allen_batch_remaining.py` | Allen Neuropixels pipeline |
| `src/cti_allen_multiarea_batch.py` | Multi-area biological + equicorrelation |
| `src/cti_confusion_causal_prediction.py` | Causal confusion prediction |
| `src/cti_generate_figures.py` | All paper figures |
| `src/cti_generate_new_figures.py` | Additional figures (confusion, H3, three-level) |
| `src/cti_downstream_h3_extension.py` | H3 n=9 ranking |
| `src/cti_alpha_rho_multidataset.py` | Alpha-rho multi-dataset bootstrap |
| `src/cti_encoder_loao.py` | Encoder universality test |
| `src/cti_scaling_dynamics.py` | E1/E2 scaling dynamics |

## Canonical Research Docs

| Document | Purpose |
|----------|---------|
| `research/OBSERVABLE_ORDER_PARAMETER_THEOREM.md` | Master derivation chain |
| `research/CTI_SUCCESS_CRITERIA.md` | Pass/fail criteria (current) |
| `research/CODEX_NOBEL_REVIEW_BRIEF.md` | Evidence summary for review |
| `research/CLAIM_EVIDENCE_CHECKLIST.md` | Paper claim -> JSON mapping |
| `research/THEORETICAL_NARRATIVE.md` | Theory exposition |
| `research/NEUROSCIENCE_DATASETS.md` | Biological data resource guide |
| `experiments/EXPERIMENTS.md` | Experiment log (reverse chronological) |

## Paper

| File | Purpose |
|------|---------|
| `paper/cti_universal_law.tex` | Main paper source (28 pages) |
| `paper/cti_universal_law.pdf` | Compiled PDF |
| `paper/cti_universal_law_anonymous.tex` | Anonymous version for review |
| `paper/cti_universal_law_anonymous.pdf` | Anonymous compiled PDF |
| `paper/references.bib` | Bibliography |

## Figures

All in `results/figures/`:
- `fig_cti_universal_law.png` -- Main law fit
- `fig_cti_multimodal_summary.png` -- Cross-modal validation
- `fig_cti_allen_biological.png` -- Biological validation
- `fig_cti_h8plus_holdout.png` -- H8+ holdout
- `fig_cti_spread_vs_K.png` -- Spread vs K
- `fig_cti_confusion_causal.png` -- Causal confusion prediction
- `fig_cti_h3_ranking.png` -- H3 architecture ranking
- `fig_cti_three_level.png` -- Three-level universality
- `fig_cti_evidence_overview.png` -- Evidence overview

## Non-Canonical Files

The remaining ~290 scripts and ~300 JSONs are from iterative development, ablations, dead ends, and intermediate experiments. They are preserved for reproducibility but are not required to verify the paper's claims.
