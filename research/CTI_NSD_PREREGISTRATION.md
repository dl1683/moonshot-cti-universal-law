# CTI NSD Human fMRI Pre-Registration

## Date: March 3, 2026
## Status: LOCKED (pre-data-collection)

---

## Experiment: CTI Universal Law Validation on Human fMRI (NSD)

### Dataset
- Natural Scenes Dataset (NSD), Allen, St-Yves et al. 2022
- Subject 01, first 10 sessions (4,489 unique images)
- Kastner2015 visual ROI atlas (V1-IPS)
- K = 12 COCO supercategories (primary-area labeling)

### Primary Hypothesis (H_human1)
The CTI law `logit(q_norm) = alpha * kappa_nearest + C` holds across
human visual cortical ROIs with:
- **Direction**: alpha > 0 (higher kappa -> higher accuracy)
- **Correlation**: |r| > 0.50 across 11 merged ROIs (V1, V2, V3, hV4, VO, PHC, TO, LO, V3AB, IPS01, IPS23)

### Secondary Hypotheses

**H_human2 (Hierarchy)**:
Higher visual areas (TO, LO, PHC) have higher kappa_nearest than lower areas (V1, V2, V3) for object category classification, consistent with the known representational hierarchy.

**H_human3 (Equicorrelation)**:
Mean pairwise centroid cosine similarity (rho) is approximately constant across ROIs, with CV(rho) < 20%.

**H_human4 (Cross-Species)**:
The functional form (logit-linear in kappa) matches Allen Neuropixels mouse data. Alpha may differ (different species, different K, different stimuli), but the FORM is the same.

### Analysis Plan
1. Download betas (GLMdenoise_RR) sessions 1-10, func1pt8mm space
2. Apply Kastner2015 ROI mask, merge dorsal+ventral (V1v+V1d -> V1, etc.)
3. Average across repetitions of same image
4. PCA reduce to 100 dimensions per ROI
5. Compute kappa_nearest, 1-NN LOO accuracy per ROI
6. Fit linear regression: logit(q_norm) ~ kappa_nearest across ROIs
7. Report r, R^2, alpha, Spearman rho, equicorrelation

### Success Criteria
- H_human1: |r| > 0.50 across ROIs (same threshold as Allen bio validation)
- H_human2: mean(kappa_higher) > mean(kappa_lower), one-sided t-test p < 0.05
- H_human3: CV(rho) < 20%
- H_human4: Both mouse and human show positive alpha with |r| > 0.50

### Abort Criteria
- If 1-NN accuracy is at chance (1/K = 8.3%) for all ROIs: data quality issue
- If fewer than 5 ROIs have sufficient data: insufficient coverage
