# CTI Universal Law -- Success Criteria

## Updated Jul 25, 2026 (equicorrelation adjudicated)

### The Law

```
logit(q_norm) = alpha * kappa_nearest - beta * log(K-1) + C_dataset
```

---

## Validated Criteria (ALL PASS)

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| LOAO alpha stability | CV < 0.25 | CV=0.023 (10x below threshold) | **PASS** |
| LOAO fit quality | R^2 > 0.90 | R^2=0.955 | **PASS** |
| Pre-registered RWKV boundary | alpha in [2.43, 3.29] | alpha=2.887 | **PASS** |
| Blind OOD prediction | r > 0.70 | r=0.817, p=0.013 | **PASS** |
| H8+ holdout (6 criteria) | All 6 pass | All 6 pass (r=0.879, MAE=0.077) | **PASS** |
| Biological generalization | >70% sessions with r>0.50 | 30/32 (93.75%), mean r=0.736 | **PASS** |
| Multi-area biological | r>0.70 in >=2 non-V1 areas | VISl 22/22, VISam 24/25 | **PASS** |
| Causal confusion prediction | r>0.50, sign acc>80% | r=0.842, sign=93% | **PASS** |
| Cross-model ranking (Spearman rho) | rho>0.50, p<0.05 | rho=0.833, p=0.005 | **PASS** |

## Retracted Criteria (Equicorrelation Program)

All criteria below depend on the legacy cross-modal equicorrelation estimator and are retracted, regardless of whether the historical row passed or failed. They are not retained as approximate-universality evidence.

| Criterion | Historical Threshold | Historical Result | Status |
|-----------|----------------------|-------------------|--------|
| Cross-modal equicorrelation constancy | Low cross-modal CV | mean rho=0.462, range [0.455, 0.467], CV=1.0% | **RETRACTED** |
| Alpha-rho per-model | r > 0.70 | r=-0.546 | **RETRACTED** |
| Alpha-rho disattenuated | r > 0.70 | r=-0.519 | **RETRACTED** |
| Alpha-rho MAE | < 0.15 | 0.068 | **RETRACTED** |
| Alpha-rho mean error | < 10% | +4.7% | **RETRACTED** |

**Retraction basis (Jul 25, 2026).** `src/cti_cross_modal_rho.py:89` multiplies projected directions by `sqrt_Lambda`, implementing `Sigma_W^(1/2)` (covariance amplification), not the claimed `Sigma_W^(-1/2)` whitening. WL1 confirmed exact parity between this bug and the legacy measurements. WL2 found the historical [0.455, 0.467] band inside matched-null central 99% intervals (approximately [0.453, 0.482]), and QL1 derived rho tending to 1/2 for isotropic centroids solely from the shared anchor. WL3 could not run a corrected real-data audit because the retained caches contain no aligned embeddings and labels. There is therefore no survival evidence for universal equicorrelation, near-simplex universality, or an alpha(rho) zero-parameter explanation.

## Honest Failures / Scope Limits

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| LODO cross-dataset | r > 0.50 | mean r=0.125 | **EXPECTED FAIL** |
| Encoder universality | CV < 0.20 | CV=0.42 | **EXPECTED FAIL** |

## 9/10 Nobel-Track Requirements (Current: 7.5/10)

1. [x] Derived law form from first principles (EVT/Gumbel)
2. [x] Cross-architecture universality (12 NLP decoders, CV=2.3%)
3. [x] Causal evidence (do-interventions + confusion prediction + factorial)
4. [x] Biological validation (mouse visual cortex, 5 areas)
5. [x] Cross-modal validation (ViT, ResNet, same form)
6. [x] Practical utility (H3 ranking: rho=0.833, p=0.005)
7. [ ] arXiv publication and visibility
8. [ ] External replication by independent lab
9. [ ] Second species biological data (human fMRI)
10. [ ] Per-model alpha prediction (beyond mean-level)

---

## Historical Note

Phase 1 (Feb 2026) explored a compute-distortion power law D(C) = D_inf + k*C^(-alpha).
That hypothesis was falsified and the project pivoted to the kappa_nearest law.
Phase 1 documents are archived in `research/archive/`.
