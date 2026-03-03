# Pre-Registered Hypotheses: CTI Generation Law Extension

## Date: 2026-03-02
## Status: LOCKED — do not modify after first experiment begins
## Author: Devansh, with Codex design gate review

---

## Motivation

The CTI Universal Law governs classification quality:

    logit(q_norm) = alpha * kappa_nearest - beta * log(K-1) + C_dataset

We extend this to next-token prediction (NTP) by observing that NTP is
classification with K = V (vocabulary size). Each token prediction is a
V-way Gumbel race in the unembedding space W_U.

The generation law predicts:

    log(PPL) = beta * log(V-1) - alpha_gen * kappa_bar_token + C_model

where kappa_bar_token is the mean nearest-token separation measured from
the unembedding matrix W_U alone (no forward passes required).

## Key Definitions

- **W_U**: Unembedding matrix, shape (V, d_model). Columns map hidden
  states to logits. W_U[v,:] is the "centroid" for token v.
- **kappa_y**: For token y, kappa_y = min_{j != y} ||w_y - w_j||_whitened
  / sigma_ref, where whitening is by the within-token hidden-state scatter.
  Practically: kappa_y = min_{j != y} ||Sigma_W^{-1/2}(w_y - w_j)|| / sqrt(d_eff).
- **kappa_bar_uniform**: Unweighted mean kappa_y across all V tokens.
- **kappa_bar_freq**: Frequency-weighted mean, using token frequencies
  from the evaluation corpus: kappa_bar_freq = sum_y f_y * kappa_y.
  NOTE: Perplexity = exp(E_freq[CE_y]), so the THEORETICALLY CORRECT
  average is frequency-weighted. Test both; frequency-weighted should
  have higher correlation with log(PPL). Zipf's law means rare tokens
  (potentially lower kappa due to clustering) get less weight.
- **alpha_gen**: Slope of log(PPL) vs kappa_bar across models.
- **PPL**: Perplexity on WikiText-103 validation split (LOCKED corpus).
  We use WikiText-103 because it is the standard benchmark for LM
  evaluation and all Pythia models have published PPL on it.

## Practical Measurement Protocol

Since we cannot compute the full within-token scatter Sigma_W without
extensive forward passes, we use two proxy measurements:

### Proxy A: Raw W_U geometry (no forward passes)
- Normalize W_U rows to unit norm
- kappa_y^{raw} = min_{j != y} ||w_y/||w_y|| - w_j/||w_j||||
- kappa_bar^{raw} = mean(kappa_y^{raw})

### Proxy B: Whitened W_U geometry (requires one forward pass batch)
- Run ~10K tokens through the model, collect final-layer hidden states
- Compute empirical Sigma_W from residuals h(x) - w_y (NC decomposition)
- Whiten W_U columns by Sigma_W^{-1/2}
- Compute kappa_bar^{whitened} as above

Both proxies are tested. Proxy B is theoretically correct; Proxy A is
the zero-cost version.

---

## Models

### Primary suite: Pythia family (same tokenizer V=50280)
- Pythia-160m
- Pythia-410m
- Pythia-1b
- Pythia-1.4b
- Pythia-2.8b

### Extension suite (different tokenizers — tests beta * log(V-1))
- GPT-2 (V=50257)
- Qwen3-0.6B (V=151936)
- Llama-3.2-1B (V=128256)

---

## Pre-Registered Hypotheses

### H_gen1: kappa_bar correlates with log(PPL) across Pythia + extension models
**Prediction**: Pearson r(kappa_bar^{raw}, log(PPL)) < -0.80 across ALL 8 models
(5 Pythia + GPT-2 + Qwen3-0.6B + Llama-3.2-1B). Use Pearson r on continuous
values for better statistical power than Spearman with small n.
**Success criterion**: r < -0.80 (n=8, p < 0.02 for r=-0.80)
**Rationale**: n=5 (Pythia only) has poor statistical power (p=0.13 for rho=-0.80).
Including extension suite gives n=8 with adequate power. Cross-tokenizer models
test whether kappa captures genuine quality variation, not just V artifacts.
Li et al. 2025 found Spearman >0.95 for representation dispersion.
NOTE: This is a FEASIBILITY test, not a definitive significance study.

### H_gen2: alpha_gen is in the range predicted by classification
**Prediction**: |d(log PPL)/d(kappa_bar)| in [0.5, 3.5]
**Success criterion**: Slope magnitude in [0.5, 3.5]
**Rationale**: Classification alpha = 1.477 for NLP decoders. Generation alpha may differ
because the equicorrelation structure differs (V >> d vs K << d), but should be O(1).
The range [0.5, 3.5] spans alpha(rho) for rho in [0, 0.8].

### H_gen3: Random W_U produces no signal (null check)
**Prediction**: For each Pythia model, replace W_U with a random matrix of the
same dimensions (i.i.d. N(0, 1/d) entries, then normalize rows). Compute kappa_bar^{random}.
rho(kappa_bar^{random}, log(PPL)) should be < 0.30 in absolute value.
**Success criterion**: |rho| < 0.30 for random W_U
**Rationale**: Golechha et al. ICLR 2025 showed random W_U has similar orthogonality
to trained. But kappa_nearest requires alignment between hidden states and W_U columns
(NC), which random matrices cannot provide. This is the critical null check.

### H_gen4: alpha_gen is constant within the Pythia family
**Prediction**: LOAO within Pythia: fit alpha on 4 models, predict the 5th.
Residual < 0.15 nats for all 5 leave-one-out folds AND kappa-based residual
is smaller than the naive baseline (predicting mean(log PPL) of training set).
**Success criterion**: (a) Mean residual < 0.15 nats across 5 folds, AND
(b) kappa-based residual < mean-baseline residual in at least 4/5 folds.
**Rationale**: Criterion (b) prevents vacuous pass — ensures kappa adds
information beyond a trivial mean prediction. In classification, alpha is
constant within architecture families (CV=2.3%).

### H_gen5: Whitened kappa outperforms raw kappa
**Prediction**: Proxy B (whitened) achieves higher |rho| than Proxy A (raw)
by at least 0.05 absolute.
**Success criterion**: |rho_B| - |rho_A| > 0.05
**Rationale**: Theory predicts that whitening by Sigma_W is necessary to get
the correct geometric separation. Raw cosine distances are confounded by
anisotropic hidden-state distributions.

### H_gen6 (Exploratory): Alpha_gen reflects token clustering
**Prediction**: alpha_gen from fitting log(PPL) vs kappa_bar should be
in [1.0, 1.5] for NLP decoders. Theoretical prediction: alpha_gen = 1.128
if W_U columns are perfectly uniform (rho_whitened = 0). Higher alpha indicates
residual token clustering (positive equicorrelation in whitened space).
**Success criterion**: alpha_gen in [0.8, 2.0] (broad but above zero)
**Rationale**: Derived from alpha(rho) = sqrt(4/pi)/sqrt(1-rho). Uniform
tokens give rho=0 → alpha=1.128. Token clustering gives rho>0 → alpha>1.128.

### H_gen7 (Exploratory): Beta captures vocabulary size effect
**Prediction**: For models with different tokenizers (GPT-2 V=50K, Qwen3
V=152K, Llama V=128K), the residual log(PPL) - alpha*kappa_bar correlates
positively with log(V-1).
**Success criterion**: Direction correct (positive correlation). No
quantitative threshold (exploratory, confounded by model quality).

### H_gen8: kappa predicts PPL beyond model size (confound control)
**Prediction**: Partial correlation r(kappa_bar, log(PPL) | log(N_params))
remains negative and |r_partial| > 0.50.
**Success criterion**: |r_partial| > 0.50
**Rationale**: Without this, a skeptic argues kappa is a proxy for model size.
Our classification result E1 showed kappa NOT correlated with model size
(p=0.91), but generation may differ. This is the critical confound control.

### H_gen9: kappa outperforms simpler geometric baselines
**Prediction**: kappa_bar achieves higher |r| with log(PPL) than:
  (a) effective rank of W_U (exp(entropy of normalized singular values))
  (b) mean cosine similarity of W_U rows
  (c) condition number of W_U
**Success criterion**: |r_kappa| > max(|r_effrank|, |r_cossim|, |r_condnum|)
**Rationale**: If a simpler metric works equally well, the Gumbel-race
derivation adds no value. kappa should win because it captures the CORRECT
geometric quantity for the competition mechanism.

---

## Step 0: NC Degree Measurement (MUST PASS before proceeding)

Before running any generation law experiment, measure the degree of Neural
Collapse in Pythia models:

For each model, run ~10K tokens through the model:
1. Get h(x) (final hidden state before unembedding)
2. For each token y, project onto w_y: gamma = h(x) @ w_y / ||w_y||^2
3. Residual: eps = h(x) - gamma * w_y
4. R^2_NC = 1 - mean(||eps||^2) / mean(||h(x)||^2)

**Pass criterion**: R^2_NC > 0.3 for at least 3/5 Pythia models.
If R^2_NC < 0.3 for all models, the NC assumption is too weak and the
generation law's derivation does not apply. Abort the experiment.

This costs ~30 minutes and prevents wasting 4 hours on a doomed experiment.

---

## Decision Matrix

| H_gen1 | H_gen3 | Interpretation |
|--------|--------|----------------|
| PASS   | PASS   | Generation law has signal. Proceed to full validation. |
| PASS   | FAIL   | Signal may be artifact of W_U structure, not NC alignment. Investigate. |
| FAIL   | PASS   | kappa_bar is not the right order parameter. Try alternatives. |
| FAIL   | FAIL   | Generation extension may not work. Return to classification focus. |

| H_gen2 | Interpretation |
|--------|----------------|
| PASS   | alpha_gen is O(1), consistent with Gumbel-race theory |
| FAIL (slope too small) | The Gumbel-race mechanism may not govern NTP |
| FAIL (slope too large) | Possible amplification effect beyond Gumbel race |

| H_gen4 | Interpretation |
|--------|----------------|
| PASS   | alpha_gen is a family constant, like classification |
| FAIL   | alpha_gen varies with model size — more complex than classification |

---

## Threats and Mitigations

1. **Golechha et al. ICLR 2025**: Random W_U has same orthogonality.
   Mitigation: H_gen3 null check. kappa requires NC alignment, not just orthogonality.

2. **Kulkarni et al. 2026**: Effective rank doesn't predict performance across OLMo models.
   Mitigation: kappa_nearest != effective rank. kappa captures separation/noise, not spectral spread.
   If OLMo checkpoints are available, test kappa on their suite (secondary experiment).

3. **Zhao et al. ICLR 2026**: NC requires SGD, not AdamW.
   Mitigation: Pythia uses AdamW. If NC is weak, Proxy B (whitened) should still capture
   whatever geometric structure exists. H_gen5 tests whether whitening matters.

4. **Context-dependence**: Static W_U ignores context-dependent token difficulty.
   Mitigation: If H_gen1 passes with static kappa, context effects are second-order.
   If H_gen1 fails, try context-conditioned kappa (forward-pass dependent).

5. **Confound: Model size**: Larger models have both lower PPL and potentially
   different kappa. Control: include model size as a covariate. Partial correlation
   rho(kappa, log PPL | model_size) should remain significant.

---

## Implementation Plan

0. **NC degree measurement** — run ~10K tokens through each Pythia model,
   compute R^2_NC (NC alignment quality). If R^2_NC < 0.3 for all models,
   ABORT. (~30 minutes)
1. Extract W_U from all 8 model checkpoints
2. Compute kappa_bar^{raw} for each (Proxy A) — ~30 minutes total
3. Compute simpler baselines (effective rank, mean cossim, condition number)
4. Evaluate PPL on WikiText-103 validation for each — ~1 hour total
   (use published PPL values where available to save compute)
5. Compute random-W_U kappa (H_gen3 null check)
6. Test H_gen1, H_gen2, H_gen3, H_gen8, H_gen9 immediately
7. Run one forward-pass batch for Proxy B (whitened kappa) — ~2 hours total
8. Test H_gen4, H_gen5, H_gen6, H_gen7
9. Report ALL results honestly, including failures

Total estimated compute: ~4.5 hours on RTX 5090.
Step 0 can save ~4 hours if NC is too weak.

---

## This Document is LOCKED

No modifications after the first forward pass is computed.
All results will be reported regardless of outcome.
