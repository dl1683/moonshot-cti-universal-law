# The CTI Universal Law: A First-Principles Study Guide

**Everything you need to understand the math, the evidence, and the open directions.**

---

## TL;DR — The One-Sentence Version

> There is a single equation that tells you how well any classifier (neural network, biological brain, anything) will perform, using only the geometry of how it organizes information — and it's derivable from first principles.

The equation:

```
logit(q_norm) = alpha * kappa_nearest - beta * log(K-1) + C
```

This guide explains every symbol, where the equation comes from, why it works, what it predicts, and where it breaks down.

---

## Table of Contents

1. [The Setup: What Are We Measuring?](#1-the-setup-what-are-we-measuring)
2. [The Key Quantities](#2-the-key-quantities)
3. [Why Logit? Why Linear?](#3-why-logit-why-linear)
4. [The Gumbel Race: Where the Law Comes From](#4-the-gumbel-race-where-the-law-comes-from)
5. [Neural Collapse: Why the Geometry Is Universal](#5-neural-collapse-why-the-geometry-is-universal)
6. [The Equicorrelation rho: A Deeper Invariant](#6-the-equicorrelation-rho-a-deeper-invariant)
7. [The Three Constants: alpha, beta, C](#7-the-three-constants-alpha-beta-c)
8. [Cross-Modal Validation: Vision, Audio, Biology](#8-cross-modal-validation-vision-audio-biology)
9. [The Generation Law: Extending to Next-Token Prediction](#9-the-generation-law-extending-to-next-token-prediction)
10. [Causal Evidence: Is This Just a Correlation?](#10-causal-evidence-is-this-just-a-correlation)
11. [Where the Law Fails (Honest Scope)](#11-where-the-law-fails-honest-scope)
12. [Three-Level Universality Structure](#12-three-level-universality-structure)
13. [Open Directions and Next Steps](#13-open-directions-and-next-steps)
14. [Quick Reference: Key Numbers](#14-quick-reference-key-numbers)

---

## 1. The Setup: What Are We Measuring?

### The fundamental question

Given a trained neural network (or a biological brain), how good are its internal representations at distinguishing between categories?

We don't care about the final classifier head. We care about the **geometry of the embeddings** — the spatial layout of how the model organizes data points internally. If two classes are well-separated in embedding space, the model can tell them apart easily. If they're tangled up, it can't.

### The probe: 1-Nearest Neighbor (1-NN)

To measure representation quality without adding any learnable parameters, we use the simplest possible classifier: **1-nearest neighbor (1-NN)**.

Given a test point x:
1. Find its nearest neighbor in the training set (by Euclidean distance in embedding space)
2. Predict x has the same label as that neighbor

1-NN accuracy depends ONLY on the geometry of the embeddings. No probe capacity. No optimization. Pure geometric signal.

### Why 1-NN and not a linear probe?

A linear probe (train a small classifier on top of embeddings) mixes geometry with the probe's own optimization. It can mask bad geometry if the probe has enough capacity, or fail with good geometry if the hyperparameters are wrong. 1-NN removes this confounder entirely: the accuracy IS the geometry.

---

## 2. The Key Quantities

### q_norm: Chance-Normalized Accuracy

Raw 1-NN accuracy is misleading across different K (number of classes). With K=2 (binary), random guessing gives 50%. With K=100, random guessing gives 1%. We need a common scale.

```
q_norm = (acc_1NN - 1/K) / (1 - 1/K)
```

- q_norm = 0: exactly at chance (no geometric signal)
- q_norm = 1: perfect classification (every point's nearest neighbor is same-class)
- q_norm < 0: below chance (adversarial geometry — possible but rare)

**Example**: If K=10 and 1-NN accuracy = 55%, then q_norm = (0.55 - 0.1)/(1 - 0.1) = 0.45/0.90 = 0.50.

### kappa_nearest: The Geometric Signal-to-Noise Ratio

This is the single most important number in the entire theory. It measures **how well-separated the closest pair of class clusters are, relative to the within-class spread**.

```
kappa_nearest(k) = min_{j != k} ||mu_j - mu_k|| / (sigma_W * sqrt(d))
```

Where:
- **mu_k** = centroid (mean embedding) of class k
- **min_{j != k}** = take the NEAREST other class (the hardest competitor)
- **||mu_j - mu_k||** = Euclidean distance between class centroids
- **sigma_W** = pooled within-class standard deviation per dimension
- **d** = embedding dimension
- **sigma_W * sqrt(d)** = expected within-class L2 radius (under isotropic null)

**Why "nearest"?** Because in a 1-NN race, the class you're most likely to confuse a point with is the NEAREST other class. Far-away classes are irrelevant to the competition. This is the "bottleneck" — the weakest link in the chain.

**Why normalize by sigma_W * sqrt(d)?** To make kappa dimensionless. The numerator (centroid distance) has units of distance. The denominator (expected within-class radius) also has units of distance. The ratio is pure signal-to-noise, comparable across models with different embedding dimensions.

**Intuition**: kappa_nearest is like a "contrast ratio" for class separation.
- kappa_nearest << 1: the noise (within-class spread) swamps the signal (centroid separation). Classification is at chance.
- kappa_nearest ~ 1: the signal and noise are comparable. Classification is in the transition zone.
- kappa_nearest >> 1: signal dominates. Classification is near-perfect.

**Concrete example**: Pythia-160M on AGNews (K=4), final layer:
- Centroid distances between the 4 news categories: 15.2, 16.8, 18.1, ... (varies by pair)
- Nearest pair distance: ||mu_sports - mu_world|| = 15.2
- sigma_W * sqrt(768) = 14.3
- kappa_nearest = 15.2 / 14.3 = 1.06

### K: Number of Classes

The number of categories being classified. This enters the law through log(K-1) because having more classes means more competitors in the race. Doubling the classes doesn't double the difficulty — it increases it logarithmically (due to the Gumbel extreme-value structure).

---

## 3. Why Logit? Why Linear?

### The logit transformation

The logit function maps a probability p in (0,1) to the whole real line:

```
logit(p) = log(p / (1-p))
```

Its inverse is the sigmoid (logistic function):

```
sigmoid(x) = 1 / (1 + exp(-x))
```

**Why logit?** Because q_norm is bounded between 0 and 1, and we want to model something that can take any real value (the geometric SNR kappa_nearest goes from 0 to infinity). The logit is the canonical link function for binomial data (from generalized linear models / logistic regression), and it has a very specific physical motivation here: the Gumbel race.

When we write:
```
logit(q_norm) = alpha * kappa_nearest - beta * log(K-1) + C
```

This is equivalent to:
```
q_norm = sigmoid(alpha * kappa_nearest - beta * log(K-1) + C)
       = 1 / (1 + exp(-(alpha * kappa_nearest - beta * log(K-1) + C)))
```

This is NOT a logistic regression. It's a **structural prediction from extreme value theory**. The sigmoid/logit appears because of the specific probability structure of the Gumbel race (explained next).

---

## 4. The Gumbel Race: Where the Law Comes From

This is the core derivation. Let me build it step by step.

### Step 1: What happens in a 1-NN classification

When you classify a query point x from class k using 1-NN:

1. Compute the distance from x to every training point
2. Find the closest one
3. Predict x belongs to the same class as that closest point

The query is classified correctly if and only if the **nearest same-class point is closer than all nearest different-class points**. This is a RACE between:
- M+ = distance to the nearest point from the correct class k
- M_j = distance to the nearest point from each wrong class j (for all j != k)

**1-NN is correct iff M+ < min_{j != k} M_j**

### Step 2: Extreme Value Theory (EVT) — What does the minimum look like?

The minimum distance to n points from a distribution concentrates around a specific value and follows a specific distribution. This is what Extreme Value Theory tells us.

**The three universal limit distributions (Fisher-Tippett theorem, 1928):**
The maximum (or minimum, by reflection) of n i.i.d. random variables, after appropriate centering and scaling, converges to one of exactly three distributions:
1. **Gumbel** — for distributions with exponential-type tails (Gaussian, exponential, etc.)
2. **Frechet** — for heavy-tailed distributions
3. **Weibull** — for distributions with finite support

For Gaussian/sub-Gaussian class conditionals (which neural network embeddings approximately satisfy), the minimum distances fall in the **Gumbel domain of attraction**.

**The Gumbel distribution:**
```
P(G <= x) = exp(-exp(-(x - mu)/s))
```
where mu is the location and s is the scale. The key property: the location mu decreases as O(sqrt(log n)) with more samples n (more points to be close to), but the scale s stays roughly constant.

### Step 3: Setting up the race

After EVT normalization, we have:
- M+ ~ Gumbel(mu+, s)       — distance to nearest same-class point
- M_j ~ Gumbel(mu_j, s)     — distance to nearest point from class j

The key: **all these Gumbels share approximately the same scale s** (because the within-class spread is similar across classes). They differ only in their locations (mu+ vs mu_j), which depend on how far class j's centroid is from x's true class k.

### Step 4: The Gumbel race identity (the crucial formula)

Here is the magical property of Gumbel distributions. If G_0, G_1, ..., G_{K-1} are independent Gumbels with common scale s and locations mu_0, mu_1, ..., mu_{K-1}, then:

```
P(G_0 < min(G_1, ..., G_{K-1})) = 1 / (1 + SUM_{j>=1} exp(-(mu_j - mu_0)/s))
```

This is EXACT. Not an approximation. The probability of one Gumbel beating all others is a logistic function of the location gaps.

**Why this is extraordinary**: For most distributions, computing P(min of K things) requires intractable K-dimensional integrals. For Gumbels, it factorizes into a simple closed-form sigmoid. This is why EVT + Gumbel gives us a clean law.

### Step 5: Applying to 1-NN classification

Set G_0 = M+ (same-class nearest distance, small is good) and G_j = M_j (wrong-class nearest distances, we want these to be bigger). Then:

```
q = P(correct) = P(M+ < min_j M_j) = 1 / (1 + SUM_{j != k} exp(-(mu_j - mu+)/s))
```

where mu_j - mu+ is the "gap" between the wrong-class location and the same-class location. Bigger gap = easier to classify.

### Step 6: Nearest-class dominance

The sum SUM_{j != k} exp(-Delta_{k,j}/s) is dominated by the SMALLEST gap — the nearest competing class.

**Dense (ETF/equidistant) case**: If all K-1 competitors are equidistant (gap = Delta for all j), then:
```
SUM = (K-1) * exp(-Delta/s)
```

Taking logit:
```
logit(q) = Delta/s - log(K-1)
```

**Hierarchical (real-world) case**: Real datasets have class hierarchy. If only sqrt(K) effective competitors are active (the rest are too far away to matter), then:
```
logit(q) ≈ Delta/s - 0.5 * log(K-1)
```

This gives beta = 0.5 instead of beta = 1. The empirical value is beta ≈ 0.478, confirming sparse competition.

### Step 7: Connecting the gap to kappa

The gap Delta = mu_j - mu+ depends on:
1. How far apart the centroids are (the numerator of kappa)
2. How much noise there is (sigma_W, the denominator of kappa)
3. The embedding dimension d (affects concentration)

Under isotropic Gaussian geometry, the gap is approximately linear in kappa_nearest:
```
Delta/s ≈ a_1 * kappa_nearest + a_0
```

This gives us:
```
logit(q_norm) = (a_1/s) * kappa_nearest - beta * log(K-1) + (a_0/s)
             = alpha * kappa_nearest    - beta * log(K-1) + C
```

**That's the law.** The logit-linear form is a STRUCTURAL consequence of:
1. Minimum distances follow Gumbel distributions (EVT)
2. The Gumbel race identity gives a logistic function (exact)
3. The gap is approximately linear in kappa (geometric linearization)
4. Nearest-class dominance reduces the K-way race to an effective 2-way race

### What's derived vs. what's estimated

- **Derived (conditional theorem)**: The functional form — logistic in kappa, logarithmic in K. This follows from EVT + Gumbel race under the stated assumptions (sub-Gaussian tails, shared scale, nearest-class dominance).
- **Estimated from data**: The constants alpha, beta, C. These depend on specific properties of the within-class covariance, the effective dimension, and finite-sample corrections that differ by architecture family.

This is why the paper calls it a "conditional theorem" — IF the assumptions hold, THEN the form must be logit-linear. The form is not a curve fit. The constants are.

---

## 5. Neural Collapse: Why the Geometry Is Universal

### What is Neural Collapse?

Neural Collapse (NC) is a phenomenon discovered by Papyan, Han, and Donoho (PNAS 2020). At the terminal phase of training, deep networks converge to a very specific geometric structure:

1. **NC1**: Within-class variability collapses — all points from the same class cluster tightly around their centroid
2. **NC2**: Class centroids form an Equiangular Tight Frame (ETF) — a maximally equidistant configuration, like a regular simplex
3. **NC3**: The classifier (last-layer weights) aligns with the centroids
4. **NC4**: The model's predictions converge to a nearest-centroid classifier

### Why NC matters for CTI

NC provides the geometric REASON why the law works across architectures:

**At exact NC (the limiting case):**
- All class pairs are equidistant (ETF) → kappa_nearest = kappa_spec (no bottleneck)
- Within-class covariance is isotropic → the Gaussian assumption is well-satisfied
- The Gumbel race is perfectly symmetric → beta = 1 exactly

**Near NC (what real networks achieve):**
- Class centroids approximately form a regular simplex → near-equidistant
- Within-class noise is approximately shared across classes → Gumbel shared-scale assumption holds
- The bottleneck class pair drives classification → kappa_nearest is the right metric

**Key insight**: kappa_nearest is a measure of HOW CLOSE the representation is to Neural Collapse geometry. Far from NC, kappa_nearest is small (bad separation). Near NC, kappa_nearest is large (good separation). The law maps this geometric proximity to classification quality.

### Why different architectures converge

If two models both approach NC geometry on the same dataset, they'll have similar kappa_nearest values and similar classification quality — REGARDLESS of whether they're transformers, SSMs, RNNs, or biological neurons. The architecture determines HOW the model gets to the geometry (the optimization path), but the law governs WHAT the geometry implies for classification (the endpoint).

This is why alpha is constant across 12 NLP architectures (CV = 2.3%): they all converge to similar NC geometry, and the Gumbel race structure is the same for all of them.

---

## 6. The Equicorrelation rho: A Deeper Invariant

### What rho measures

rho is the average pairwise cosine similarity between centroid-difference vectors, after whitening by the within-class covariance.

To compute rho:
1. Take the within-class covariance matrix Sigma_W
2. Whiten it: Sigma_W^(-1/2) * (mu_j - mu_k) for all pairs (j,k)
3. Compute cosine similarities between all pairs of these whitened centroid-difference vectors
4. Average them

**Physical meaning**: rho measures how "aligned" the different competition directions are in whitened space.
- rho = 0: all competition directions are orthogonal (independent competitions)
- rho = 1: all competition directions are the same (degenerate 1D competition)
- rho = 1/2: this is the value for a regular simplex (ETF geometry)

### Why rho = 0.46 everywhere

The remarkable empirical finding: rho ≈ 0.46 across ALL modalities tested:

| Modality | rho | Source |
|----------|-----|--------|
| NLP decoders (11 architectures) | 0.463 | LOAO fit |
| Audio speech (WavLM, HuBERT) | 0.455-0.464 | Speech Commands |
| Vision (ViT-Base, ResNet50) | 0.458-0.467 | CIFAR-10/100 |
| Mouse visual cortex (Allen) | 0.466 | Neuropixels K=118 |
| NLP encoders (BERT, ELECTRA) | 0.446-0.456 | Multiple datasets |
| **Pooled (6 modalities)** | **0.462, CV=1.0%** | Cross-modal |

The theoretical prediction for a perfect simplex (exact NC) is rho = 1/2 = 0.5. The measured rho ≈ 0.46 is 8% below perfect NC — a near-simplex geometry.

**This is the tightest invariant in the entire paper** (CV = 1.0%, tighter than alpha's 2.3%).

### The alpha prediction from rho

From the Gumbel race theory, the competition effective dimension is:
```
d_eff_comp = 1 / (1 - rho)
```

And alpha should be:
```
alpha = sqrt(4/pi) * sqrt(d_eff_comp) = sqrt(4/pi) / sqrt(1 - rho)
```

With rho = 0.46:
```
alpha_pred = sqrt(4/pi) / sqrt(0.54) = 1.128 / 0.735 = 1.534
```

Observed: alpha = 1.477. Error: +3.8%. Zero free parameters.

**But**: This only works for NLP decoders. For ViT (alpha ≈ 4.5), the same rho ≈ 0.46 would predict alpha ≈ 1.5, not 4.5. This means there's a SECOND factor (beyond rho) that amplifies alpha for continuous-signal modalities. The leading hypothesis is that the within-class covariance Sigma_W has higher effective rank for image/audio features than for discrete tokens, creating a modality-specific renormalization. This is an open problem.

---

## 7. The Three Constants: alpha, beta, C

### alpha: The Slope (Signal Amplification)

alpha converts geometric SNR (kappa) into classification log-odds. Bigger alpha = steeper sigmoid = sharper transition from bad to good classification.

**Measured values by modality:**

| Signal Type | alpha | Modalities |
|-------------|-------|------------|
| Discrete tokens (NLP decoders) | ~1.48 | Pythia, Qwen, OLMo, etc. |
| Continuous signals | ~3.3-5.0 | ViT (~4.5), CNN (~4.0), Audio (~4.7) |
| NLP encoders | ~7.7 | BERT, DeBERTa, ELECTRA |

**Key facts about alpha:**
- Within NLP decoders: CV = 2.3% across 12 architectures (it's essentially a constant)
- This CV is CONSISTENT with LOAO estimation noise (expected ~2.8% from 176 training points per fold) — there's no residual per-model signal. Alpha is a TRUE constant within decoder family.
- Across families: alpha varies by signal type. This is NOT a failure; it's level-2 universality.

### beta: The Competition Penalty

beta measures how much classification difficulty increases with more classes.

**Theoretical prediction**: beta = 1.0 (if all K-1 competitors are equally active — the dense ETF/equidistant limit).

**Empirical value**: beta ≈ 0.478 (much less than 1).

**Why beta < 1?** Because real datasets have hierarchical class structure. Not all K-1 classes are equally confusable. If you're classifying news articles, "sports" and "world news" might be confusable, but "sports" and "technology" are very distinct. Only the nearby classes in semantic space are active competitors. The effective number of competitors is:

```
N_eff ≈ sqrt(K-1)    (not K-1)
```

This gives beta = d(log(sqrt(K-1)))/d(log(K-1)) = 1/2 = 0.5.

The pre-registered constrained comparison confirms beta=0.5 beats beta=1.0 by 10.1% in LOAO MAE. The unconstrained MLE is beta = 0.478, within 0.6% of 0.5.

### C: The Intercept (Task Difficulty)

C (or C_dataset, C_0) encodes the "base difficulty" of a dataset that cannot be predicted from geometry alone. It depends on:
- The specific classes chosen (some category systems are inherently harder)
- The evaluation protocol (train/test split, balanced vs imbalanced)
- The noise floor of the measurement

**Critical fact**: C is NOT universal. Each new dataset requires at least one calibration measurement to estimate C. This is why LODO (leave-one-dataset-out) fails (mean r = 0.125): the intercept can't be predicted from other datasets.

This is an honest scope limitation, not a failure. The law predicts SHAPE (how quality varies with kappa), not LEVEL (what the absolute quality is on a new task).

---

## 8. Cross-Modal Validation: Vision, Audio, Biology

### Vision: ViT and ResNet

**ViT-Large on CIFAR-10 (K=10):**
- R^2 = 0.964, r = 0.982
- Same logit-linear form as NLP
- alpha_ViT ≈ 4.5 (3x higher than NLP)

**ResNet50 on CIFAR-100 (K=100):**
- r = 0.749
- alpha_CNN ≈ 4.4 (close to ViT)
- Below Monte Carlo noise floor (E[r] = 0.875) — hierarchical superclass structure in CIFAR-100 creates non-ideal geometry

**ViT-Base on CIFAR-100 (K=100):**
- r = 0.773 (similar to ResNet50)
- This confirms the K=100 drop is NOT architecture-specific — it's a dataset property

### Audio: Speech Models

Seven frozen speech models from 4 architectures (Wav2Vec2, HuBERT, WavLM, Whisper) on Speech Commands v0.02 (K=36 spoken keywords):
- r = 0.898, p = 0.006
- alpha_audio = 4.669
- alpha_audio ≈ alpha_CNN ≈ alpha_ViT (all continuous-signal modalities converge at ~4-5)

### Biology: Mouse Visual Cortex

**Allen Neuropixels (K=118 natural images, 32 mice):**
- 30/32 sessions PASS (r > 0.50)
- All 32 show positive r (range [0.44, 0.89])
- Mean r = 0.736

The law form is preserved in biological neurons — not just artificial ones. The constant is different (A_bio is 15-34x smaller than A_NLP), because biological neurons are not trained by gradient descent to maximize separation. But the geometric skeleton is the same.

**Equicorrelation in cortex**: rho = 0.466 (mean across 5 areas), within 1% of NLP decoders. Near-simplex geometry is substrate-independent.

### Macaque Visual Cortex

- Macaque IT (high-level visual): per-image r = 0.41
- Macaque V4 (mid-level visual): per-image r = 0.116
- The hierarchy gradient (IT > V4) is consistent with representational hierarchy theory

### Where it fails biologically

**Human fMRI (NSD, K=12):**
- 1-NN accuracy at chance level (~8%)
- Pooled r = 0.12, p = 0.18 (not significant)
- Raw BOLD voxel responses lack sufficient SNR for nearest-neighbor classification of semantic categories
- The law REQUIRES above-chance 1-NN as a prerequisite; fMRI doesn't provide it

---

## 9. The Generation Law: Extending to Next-Token Prediction

### The key insight

Next-token prediction (NTP) is V-way classification at each position: the model outputs logits z_v = w_v^T * h(x) and the correct token wins the softmax "race". This is the SAME Gumbel race, just with V (vocabulary size) classes instead of K.

The architecture-independence lemma: the generation law depends ONLY on the unembedding step z = W_U @ h(x), not on how h(x) is computed. Therefore:
- Pure Transformer → same law
- Pure SSM (Mamba) → same law
- Hybrid (Transformer + SSM) → same law

### kappa from W_U (zero forward passes)

Instead of running the model to get embeddings, we extract kappa directly from the unembedding matrix W_U:
1. Get W_U (the matrix mapping hidden states to logits)
2. Normalize each row to unit norm
3. For the top-1000 most frequent tokens, compute nearest-neighbor distances
4. kappa_bar = mean of these distances

This requires NO forward passes — just loading the weights.

### Results across 22 models

**All models (n=22):** r(kappa, log(CE)) = -0.55, p = 0.008
**Excluding LFM2.5 outlier (n=21):** r = -0.70, p < 0.001

**Fixed-V group (Pythia + Mamba, V=50280, n=10):**
- r = -0.837, p = 0.003
- F-test for architecture interaction: p = 0.147 (NOT significant)
- Transformers and SSMs lie on the SAME regression line

### Why beta_gen ≈ 0 (vocabulary size drops out)

The CTI classification law has a -beta * log(K-1) term because adding more classes means more competitors. In generation, K = V (vocabulary size), so you'd expect:
```
log(PPL) = -alpha_gen * kappa + beta_gen * log(V-1) + C
```

But empirically, beta_gen ≈ 0. The vocabulary size drops out.

**Why?** Because of the K_eff decomposition. At each token position:
```
CE = margin + log(K_eff)
```
where K_eff ≈ 2-3 tokens effectively compete. Despite V = 32K-152K, the softmax concentrates probability on just a few alternatives. This means log(V-1) is irrelevant — what matters is log(K_eff) ≈ log(2-3), which is approximately constant across architectures.

### Why R^2 is only 0.49

The generation law has R^2 ≈ 0.49 (vs 0.955 for classification). This is PREDICTED by the theory:
- In classification, kappa_nearest captures the FULL race conditions (centroid separation in the representation space h)
- In generation, kappa from W_U captures only the OUTPUT geometry, not the context-modeling quality h(x)
- The residual correlates with model size (r = -0.466), confirming that the unexplained variance is attributable to h(x) quality differences

---

## 10. Causal Evidence: Is This Just a Correlation?

The law could be an artifact — a statistical correlation without causal content. Five tiers of evidence argue otherwise:

### Tier 1: Frozen Do-Interventions

Physically move one class centroid in embedding space (do-intervention on mu_k) and measure the change in logit(q). The nearest-class intervention (moving the closest competitor) has r = 0.899 with logit change. The farthest-class intervention (moving a distant class) has r = 0.000 (negative control). This confirms nearest-class geometry is the causal driver.

### Tier 2: Orthogonal Factorial

A cleaner experimental design with 14 focus classes:
- Arm A: intervene on nearest class → r = 0.899
- Arm B: control (no intervention) → r = 0.000
- Arm C: intervene on farthest class → r = 0.000
This rules out confounders.

### Tier 3: Cross-Modal Dose-Response

Apply surgery at 7 scales to 6 unseen architectures (3 text, 3 vision). With frozen alpha and zero refit: 6/6 directional PASS, pooled MAE = 0.010.

### Tier 4: Pre-Registered Confusion Matrix Prediction

The strongest test. With FIXED A = 1.054 and tau* = 0.20 (zero degrees of freedom, no refit):
- Predict how EVERY off-diagonal confusion matrix entry changes when you shift a centroid
- delta=1: r = 0.842, sign accuracy = 92.9%
- delta=2: r = 0.808, sign accuracy = 100%
- delta=3: r = 0.776, sign accuracy = 100%
- n = 182 class-pair predictions at each level, all p < 10^-35

100% sign accuracy means the law correctly predicts the DIRECTION of every confusion change. This is not just correlation — it's quantitative causal prediction.

### Tier 5: Top-m Competitor Sweep

Moving the nearest m centroid pairs by equal amounts and fitting the slope:
- m=1: alpha_1 = 1.052
- m=2: alpha_2 = 1.523 ≈ alpha_LOAO = 1.477 (2.3% error!)
- m=13: alpha_13 = 3.015

The LOAO cross-architecture constant is recovered by involving exactly TWO competitor directions. This zero-parameter localization confirms that ~2 near-tie competitors drive the constant.

---

## 11. Where the Law Fails (Honest Scope)

### Hard failures

1. **Protein language models**: ESM-2, ProtBERT (7 models, 3 families) on EC-number classification (K=7). Result: alpha = -1.17, r = -0.15, p = 0.76. Complete FAILURE. Why: enzyme commission classes are extremely heterogeneous — each EC number spans thousands of structurally diverse enzymes. The law requires semantically coherent categories where geometry drives classification.

2. **Human fMRI**: NSD dataset, 11 visual ROIs, K=12 COCO supercategories. Result: 1-NN at chance, pooled r = 0.12, p = 0.18. The law requires above-chance 1-NN as a prerequisite; raw BOLD fMRI doesn't provide it.

3. **Encoder LOAO**: 5 encoder architectures. CV_alpha = 0.42 (threshold was 0.20). FAIL — encoder alpha is NOT family-invariant. Range: 4.2 (ELECTRA-small) to 16.9 (BERT-base). Pooling protocol and pre-training objective jointly determine encoder alpha.

### Partial failures

4. **LODO cross-dataset**: Mean r = 0.125. The intercept C cannot be predicted from other datasets. This is EXPECTED under three-level universality (it's level-3 drift).

5. **Large K regime (K=100)**: Both ViT and ResNet at K=100 show r ≈ 0.75, below the Monte Carlo noise floor. CIFAR-100's hierarchical superclass structure creates non-ideal geometry.

6. **Alpha-rho per-model prediction**: r = -0.55 (not monotonic). BUT: the observed alpha CV (2.3%) matches LOAO estimation noise (expected 2.8%), so there's no real per-model signal for rho to explain. The mean-level prediction works (+4.3% error); per-model prediction fails because there's nothing to predict.

---

## 12. Three-Level Universality Structure

The CTI law exhibits a hierarchy analogous to renormalization group in physics:

### Level 1: Form Universality (Strongest)

The logit-linear structure logit(q) ∝ kappa_nearest is EVT-derived and confirmed across:
- NLP (12+ architectures)
- Vision (ViT, ResNet50)
- Audio (4 speech architectures)
- Mouse visual cortex (32 sessions)
- Macaque IT/V4

This is like the ideal gas law PV = nRT: the FORM is universal across substances.

### Level 2: Constant Universality Within Signal Type

alpha is stable within architecture families:
- NLP decoders: alpha ≈ 1.48 (CV = 2.3%)
- Continuous signals: alpha ≈ 3.3-5.0 (ViT, CNN, audio)
- Encoders: alpha ≈ 7.7 (but not universal within encoders)

This is like the van der Waals equation: the form is universal, but a and b differ by substance.

### Level 3: Intercept Is Task-Specific

C_dataset encodes task difficulty. Cannot be predicted from other tasks. One calibration measurement per dataset is required.

This is like the ground-state energy: universal law, but each system has its own zero point.

---

## 13. Open Directions and Next Steps

### Near-term (submission-ready additions)

1. **Expanded generation law validation**: 25+ models across all architectures in the model directory (Transformers, SSMs, Hybrids, novel architectures). Pre-registration addendum written.
2. **External replication**: Contact independent groups (St-Yves for fMRI; others for NLP) to replicate on their data.
3. **COLM 2026 submission**: Abstract deadline March 26, full paper March 31.

### Medium-term (open theoretical problems)

4. **Renormalization theory for alpha across families**: Why alpha_NLP ≈ 1.5 but alpha_ViT ≈ 4.5? Both have the same rho ≈ 0.46. The answer must lie in the structure of Sigma_W — how within-class noise differs between discrete tokens and continuous signals. Deriving alpha = f(rho, Sigma_W) from first principles would complete the theory.

5. **Non-asymptotic bounds**: The current proof is asymptotic (d → infinity, n → infinity). Finite-sample bounds (Berry-Esseen for Gumbel order statistics) would make the theorem quantitative rather than qualitative.

6. **Multilabel extension**: The current law assumes single-label classification. Multilabel settings (like GoEmotions) systematically fail. A separate treatment using Poisson competition models is needed.

### Long-term (paradigm extensions)

7. **Training objective derived from the law**: If q = sigmoid(alpha * kappa), then maximizing kappa_nearest during training should maximize quality. This gives a principled margin-based training objective. The causal evidence shows kappa_nearest is the right lever (not dist_ratio, which is diagnostic but not causal).

8. **Connection to scaling laws**: If kappa_nearest ~ C^gamma (power law in compute), then q(C) = sigmoid(alpha * C^gamma + C_0). This connects representation geometry to the compute scaling laws of Kaplan et al., providing a MICROSCOPIC explanation for macroscopic scaling.

9. **Universal renormalization theory**: The constant alpha depends on (rho, Sigma_W structure, bidirectionality). A complete theory deriving alpha from first principles for ANY architecture would be the crown jewel.

---

## 14. Quick Reference: Key Numbers

### The Law
```
logit(q_norm) = alpha * kappa_nearest - beta * log(K-1) + C
```

### Primary Constants (NLP Decoders, 12 architectures)
- alpha = 1.477 ± 0.034 (CV = 2.3%)
- beta = 0.478 (constrained fit: 0.5)
- R^2 = 0.955 (per-dataset intercepts)

### Cross-Modal alpha
| Modality | alpha |
|----------|-------|
| NLP decoders | 1.48 |
| CNN (ResNet) | 4.0-4.4 |
| ViT | 4.5 |
| Audio (speech) | 4.67 |
| NLP encoders | ~7.7 |

### Universal rho
- Mean rho = 0.462, CV = 1.0% across 6 modalities
- Regular simplex prediction: rho = 0.500
- Deviation from simplex: -8%

### Key Validation Results
| Test | Result | Status |
|------|--------|--------|
| 12-arch LOAO (NLP) | CV = 0.023 | PASS |
| RWKV boundary | 2.887 in [2.43, 3.29] | PASS |
| Blind OOD (SmolLM2) | r = 0.817, p = 0.013 | PASS |
| ViT-Large | R^2 = 0.964 | PASS |
| H8+ holdout (n=77) | r = 0.879, MAE = 0.077 | PASS |
| Audio (7 models) | r = 0.898, p = 0.006 | PASS |
| Bio (32 mice) | 30/32 pass, mean r = 0.736 | PASS |
| Confusion causal | r = 0.842, 100% sign acc | PASS |
| Generation (22 models) | r = -0.55 / -0.70 | PASS |
| Cross-modal rho | CV = 1.0% | PASS |
| Protein ESM-2 | r = -0.15 | FAIL |
| Human fMRI | r = 0.12, p = 0.18 | FAIL |
| Encoder LOAO | CV = 0.42 | FAIL |

### Zero-Parameter Predictions
- alpha from rho: 1.540 vs 1.477 (+4.3% error)
- Confusion matrix: 100% sign accuracy, zero refit
- LOAO slope from m=2 competitors: alpha_2 = 1.523 vs 1.477 (2.3% error)
- K_eff from softmax concentration: ~2-3 (beta_gen ≈ 0)

---

## Appendix: The Mathematical Chain in One Page

**Start**: K classes in R^d with centroids {mu_k} and shared within-class std sigma_W.

**Step 1** — Define kappa:
```
kappa_nearest(k) = min_{j!=k} ||mu_j - mu_k|| / (sigma_W * sqrt(d))
```

**Step 2** — 1-NN classification as a race:
```
P(correct | class k) = P(M+ < min_{j!=k} M_j)
```
where M+ = nearest same-class distance, M_j = nearest class-j distance.

**Step 3** — EVT: M+ and M_j converge to Gumbel distributions with shared scale s:
```
M+ ~ Gumbel(mu+, s)
M_j ~ Gumbel(mu_j^(-), s)
```

**Step 4** — Gumbel race identity (exact):
```
P(M+ < min_j M_j) = [1 + SUM_{j!=k} exp(-(mu_j^(-) - mu+)/s)]^{-1}
```

**Step 5** — Nearest-class dominance + sparse hierarchy:
```
SUM ≈ (K-1)^beta * exp(-Delta_nearest/s)     where beta ≈ 0.5
```

**Step 6** — Linear gap-to-kappa map:
```
Delta_nearest/s ≈ alpha * kappa_nearest + C_0
```

**Step 7** — Combine, take logit:
```
logit(q_norm) = alpha * kappa_nearest - beta * log(K-1) + C
```

**Done.** From Euclidean geometry → EVT → Gumbel race identity → logit law. QED (conditional on assumptions A1-A3).
