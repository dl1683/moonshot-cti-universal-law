# CTI Universal Law

**A first-principles derivation of a universal law governing the quality of learned representations.**

## The Law

```
logit(q_norm) = α · κ_nearest − β · log(K−1) + C₀
```

Where:
- `q_norm` = normalized 1-NN accuracy: `(acc_1NN − 1/K) / (1 − 1/K)`
- `κ_nearest` = nearest-class separation SNR: `min_{j≠k} ‖μⱼ − μₖ‖ / (σ_W · √d)`
- `α` ≈ 1.477 (NLP decoders, CV=2.3% across 12 architectures)

The functional form is derived from **extreme value theory** (Gumbel race) before fitting any constants. This is a conditional theorem — the shape is proven, only the constants are estimated.

## Key Results

| Test | Result | Status |
|------|--------|--------|
| LOAO across 12 NLP architectures | α=1.477, CV=2.3%, R²=0.955 | PASS |
| Pre-registered RWKV-4 boundary test | α=2.887 ∈ [2.43, 3.29] | PASS |
| Blind OOD (new arch + new datasets) | r=0.817, p=0.013 | PASS |
| ViT-Large cross-modal | R²=0.964 | PASS |
| Biological: 32 mouse V1 sessions | 30/32 PASS, mean r=0.736 | PASS |
| H8+ expanded holdout (11 models × 8 datasets, n=77) | r=0.879, MAE=0.077 | PASS |
| LOMFO (leave-one-family-out, all 4 families) | r≥0.84 each family | PASS |

## Honest Scope

- `α` varies by architecture family (NLP decoders: 1.48, ViT: 0.63, CNN: 4.4) — universality is of **functional form**, not constants
- Absolute prediction requires per-family calibration (4 probe measurements reduce MAE by 86%)
- Within-dataset architecture ranking is the primary validated use case

## Paper

`paper/cti_universal_law.pdf` — 25 pages

## Structure

```
src/          CTI experiment scripts (cti_*.py)
results/      Canonical result JSONs + figures
paper/        LaTeX source + compiled PDF
research/     Theory docs, pre-registrations, literature synthesis
```

## Running Experiments

```bash
# Refit the universal law
python src/cti_kappa_nearest_universal.py

# Run H8+ expanded holdout validation
python src/cti_utility_revised.py

# Reproduce all paper figures
python src/cti_generate_figures.py
```
