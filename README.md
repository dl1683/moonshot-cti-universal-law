# CTI Universal Law

**A first-principles derivation of a universal law governing the quality of learned representations.**

## The Law

```
logit(q_norm) = alpha * kappa_nearest - beta * log(K-1) + C_dataset
```

Where:
- `q_norm` = normalized 1-NN accuracy: `(acc_1NN - 1/K) / (1 - 1/K)`
- `kappa_nearest` = nearest-class separation SNR: `min_{j!=k} ||mu_j - mu_k|| / (sigma_W * sqrt(d))`
- `alpha` = 1.477 (NLP decoders, CV=2.3% across 12 architectures, R^2=0.955)
- `beta` = 0.746 (log-K scaling)
- `C_dataset` = per-dataset intercept

The **functional form is derived** from extreme value theory (Gumbel race competition among K classes) before fitting any constants. This is a conditional theorem -- the shape is proven, only the constants are estimated.

## Key Results

### Core Universality

| Test | Result | Status |
|------|--------|--------|
| LOAO across 12 NLP architectures (192 pts) | alpha=1.477, CV=2.3%, R^2=0.955 | **PASS** |
| Pre-registered RWKV-4 boundary test | alpha=2.887 in [2.43, 3.29] | **PASS** |
| Blind OOD (new arch + new datasets) | r=0.817, p=0.013 | **PASS** |
| H8+ expanded holdout (11 models x 8 datasets, n=77) | r=0.879, MAE=0.077 | **PASS** |
| LOMFO (leave-one-family-out, all 4 families) | r>=0.84 each family | **PASS** |

### Causal Evidence

| Test | Result | Status |
|------|--------|--------|
| Confusion-matrix causal prediction (3 shift levels) | r=0.842-0.776, sign acc 93-100%, n=182, p<10^-35 | **PASS** |
| Frozen do-interventions (multi-architecture) | Predicted direction confirmed | **PASS** |
| Orthogonal factorial design | kappa_nearest is causal driver | **PASS** |

### Cross-Modal Validation

| Test | Result | Status |
|------|--------|--------|
| ViT-Large (CIFAR-10) | R^2=0.964 | **PASS** |
| ResNet-50 (CIFAR-100) | Same functional form | **PASS** |
| Alpha-family law | NLP decoders: 1.48, ViT: 0.63, CNN: 4.4 | **PASS** |

### Biological Generalization (Mouse Visual Cortex)

| Test | Result | Status |
|------|--------|--------|
| 32 mouse V1 Neuropixels sessions | 30/32 PASS, mean r=0.736 | **PASS** |
| Multi-area batch (30 mice, 5 cortical areas) | VISp 30/30, VISl 22/22, all areas >=87% | **PASS** |
| Equicorrelation across areas | rho=0.43-0.46 (CV=1.65%) | **PASS** |

### Practical Utility

| Test | Result | Status |
|------|--------|--------|
| H3 cross-model ranking (9 architectures) | rho=0.833, p=0.005 (kappa ranks by MAP@10) | **PASS** |
| Alpha-rho zero-parameter prediction | Mean error +4.7%, MAE=0.068 (zero free params) | **PASS** |

### Three-Level Universality Structure

The law exhibits universality at three levels:
1. **Functional form**: `logit(q_norm) = alpha * kappa_nearest - beta * log(K-1) + C` holds across all tested architectures, modalities, and biological systems
2. **Slope constant**: `alpha` is universal within a modality family (NLP decoders: CV=2.3%)
3. **Intercept**: `C_dataset` varies by task (expected -- different datasets have different baseline difficulties)

Cross-dataset prediction (LODO CV=0.42) is an expected scope limit under this three-level structure, not a failure.

## Honest Scope

- `alpha` varies by architecture family (NLP decoders: 1.48, ViT: 0.63, CNN: 4.4) -- universality is of **functional form**, not constants
- Absolute prediction requires per-family calibration (4 probe measurements reduce MAE by 86%)
- Within-dataset architecture ranking is the primary validated use case
- Encoder `alpha` is not universal within-family (CV=0.42 for encoders vs 0.023 for decoders)
- Per-model `rho` does not monotonically predict per-model `alpha` (r=-0.55) -- `rho` captures the universal geometric baseline but not architecture-specific residuals

## Paper

`paper/cti_universal_law.pdf` -- 28 pages, targeting COLM 2026

## Repository Structure

```
src/           CTI experiment scripts (cti_*.py)
results/       Canonical result JSONs + figures
paper/         LaTeX source + compiled PDF
research/      Theory docs, pre-registrations, literature synthesis
experiments/   Experiment ledger (EXPERIMENTS.md)
```

### Canonical Files

| Category | File | Description |
|----------|------|-------------|
| **Core law** | `results/cti_kappa_loao_per_dataset.json` | LOAO fit: alpha=1.477, R^2=0.955 |
| **Holdout** | `results/cti_utility_revised.json` | H8+ expanded holdout (n=77) |
| **Biological** | `results/cti_allen_all_sessions_complete.json` | 32 mouse V1 sessions |
| **Multi-area** | `results/cti_allen_multiarea_batch.json` | 30 mice, 5 cortical areas |
| **Equicorrelation** | `results/cti_allen_equicorr_multiarea.json` | rho across 5 areas |
| **Causal** | `results/cti_confusion_causal_prediction.json` | Confusion-matrix causal test |
| **Downstream** | `results/cti_downstream_h3_n9.json` | H3 ranking (9 models, rho=0.833) |
| **Alpha-rho** | `results/cti_alpha_rho_multidataset.json` | Zero-parameter alpha prediction |
| **Cross-modal** | `results/cti_vit_cross_modality.json` | ViT validation (R^2=0.964) |
| **Family law** | `results/cti_extended_family_loao.json` | Alpha by architecture family |
| **Theory** | `research/OBSERVABLE_ORDER_PARAMETER_THEOREM.md` | Master derivation chain |
| **Figures** | `src/cti_generate_figures.py` | All paper figures |

## Running Experiments

```bash
# Refit the universal law (12 NLP architectures, 4 datasets)
python src/cti_kappa_nearest_universal.py

# Run H8+ expanded holdout validation
python src/cti_utility_revised.py

# Biological validation (Allen Neuropixels, 32 sessions)
python src/cti_allen_batch_remaining.py

# Reproduce all paper figures
python src/cti_generate_figures.py
```

## Citation

If you use this work, please cite:

```
@article{cti2026,
  title={A Universal Law for Learned Representation Quality from Extreme Value Theory},
  year={2026}
}
```
