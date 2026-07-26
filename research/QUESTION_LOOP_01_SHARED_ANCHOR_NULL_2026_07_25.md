# Question Loop 1: Shared-Anchor Null

**Date:** July 25, 2026  
**Decision:** Retire scalar cross-modal rho as a universality claim.

## Is any residual cross-modal geometric quantity distinguishable from a generic labeled high-dimensional cloud?

### Decision

**Not in the current evidence.** The historical result does not merely have a possible null explanation: its central value and tight range are quantitatively predicted by the shared-anchor null. Correct inverse whitening will not rescue the scalar. Under a matched isotropic null it drives the effective dimension upward and moves rho *closer to 1/2*, where a regular simplex and a generic high-dimensional cloud are hardest to distinguish.

The surviving question is:

> After conditioning on the exact shared-anchor/Wishart null, is there a reproducible residual spectrum or higher-order semantic structure that predicts behavior outside the geometry used to construct it?

## Shared-anchor derivation

Let the K centroids be independent draws

$$
\mu_0,\ldots,\mu_{K-1}\overset{\mathrm{iid}}{\sim}\mathcal N(0,\tau^2 I_\nu),
$$

where nu is the dimension after correct whitening. Choose `mu_0` as anchor and define `X_i = mu_i-mu_0`. For `i != j`, coordinatewise,

$$
\operatorname{Var}(X_i)=\operatorname{Var}(X_j)=2\tau^2,
\qquad
\operatorname{Cov}(X_i,X_j)=\tau^2.
$$

Therefore

$$
\boxed{\operatorname{Corr}(X_{i\ell},X_{j\ell})=1/2.}
$$

The 1/2 is caused solely by subtracting the same random anchor. It contains no class structure, learning, neural collapse, modality, or biology.

The measured cosine is the uncentered sample correlation of nu independent bivariate-normal coordinate pairs with population correlation `c=1/2`. Its exact expectation for integer isotropic dimension nu is

$$
m_\nu(c)=c\,
\frac{\Gamma((\nu+1)/2)^2}
{\Gamma(\nu/2)\Gamma((\nu+2)/2)}
\,{}_2F_1\!\left(\frac12,\frac12;\frac{\nu+2}{2};c^2\right).
$$

Thus

$$
\boxed{\mathbb E[\widehat\rho]=m_\nu(1/2),}
$$

independent of K. K changes uncertainty, not the expected value.

| nu | null mean `m_nu(1/2)` |
|---:|---:|
| 2 | 0.4063 |
| 4 | 0.4517 |
| 5 | 0.4614 |
| 6 | 0.4678 |
| 8 | 0.4760 |
| 16 | 0.4881 |
| 32 | 0.4941 |
| 256 | 0.4993 |

The large-dimension expansion is

$$
m_\nu(1/2)=\frac12-\frac{3}{16\nu}+O(\nu^{-2}).
$$

The six values in `results/cti_cross_modal_rho.json` imply, under this null, isotropic dimensions from about 4.31 to 5.91. The cross-modal mean 0.462 corresponds to `nu ~= 5.1`: almost an exact fingerprint of a low-rank shared-anchor null.

For anisotropic Gaussians, substituting the participation ratio

$$
\nu_{\mathrm{eff}}=\frac{\operatorname{tr}(\Sigma)^2}{\operatorname{tr}(\Sigma^2)}
$$

is only a moment-matched approximation. The exact distribution depends on the full spectrum. A valid empirical null must reproduce K, class counts, covariance spectrum, shrinkage, truncation, and reuse of the same data to estimate whitening.

## Where K enters, and why the result looked tight

For one anchor, let `m=K-1` and average its `choose(m,2)` cosines. A Wishart/delta-method calculation at `c=1/2` gives

$$
\operatorname{Var}(\widehat\rho_{\mathrm{anchor}})
=\frac{(m+1)^2}{8m(m-1)\nu}+O(\nu^{-2})
=\boxed{\frac{K^2}{8(K-1)(K-2)\nu}+O(\nu^{-2})}.
$$

It does not vanish as K grows because all comparisons share the anchor. More importantly, when first-order influence terms are averaged over every anchor and pair, they cancel by permutation and translation symmetry. The all-anchor mean is unusually stable under the null. A tight cross-model CV is an expected estimator property, not evidence for a biological constant.

## Why the whitening bug produces the observed number

The code projects into within-class covariance eigenvectors and executes:

```text
whitened = proj * sqrt_Lambda
```

Correct whitening divides by `sqrt_Lambda`, with shrinkage or a floor. Under a matched labeled-cloud null, centroid-estimation noise has covariance proportional to within-class covariance. In its eigenbasis:

- before transformation, coordinate variance is proportional to `lambda_a`;
- inverse whitening makes it proportional to 1;
- the implemented multiplication makes it proportional to `lambda_a^2`.

The bug concentrates the angle into leading directions, with approximate effective dimension

$$
\nu_{\mathrm{wrong}}\approx
\frac{(\sum_a\lambda_a^2)^2}{\sum_a\lambda_a^4}.
$$

That can easily be near five despite retaining 256 components. It explains both rho near 0.462 and the apparent stability across steep representation spectra.

## What could genuinely deviate?

Real representations can depart through non-exchangeable semantic centroids, between/within covariance mismatch, unequal class uncertainty, higher-order non-Gaussian structure, feature-classifier alignment, or task-aligned dynamics. Raw rho is a poor detector: exact ETF geometry and a generic high-dimensional shared-anchor cloud both approach 1/2. For fixed K and large dimension, even centered iid isotropic points approach an ETF-like Gram by concentration.

The strongest viable static target is a **null-normalized semantic residual operator**: the centroid Gram minus its full matched-null expectation, whitened by its null covariance. Surprising findings would be a stable residual spectrum, higher-order simplex defects, semantic-confusion eigenvectors, or feature-classifier coupling that:

1. survives matched Gaussian and label-permutation controls;
2. predicts held-out behavior after task, family, and scale controls; and
3. changes capability under a targeted fixed-compute intervention.

The more plausible universal is dynamic: a law for how semantic neighborhoods split, merge, or rotate across depth. Static terminal symmetry is where concentration and training objectives erase provenance.

## Recent literature

Recent work makes a free-standing ETF universality claim less credible:

- Jacot et al. prove collapse under specific balance, conditioning, weight-decay, and optimization assumptions ([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d33d31a87eb276ffe47fa7324648098b-Abstract-Conference.html)).
- Zhao et al. show optimizer choice can determine whether collapse emerges and argue against it under AdamW-style decoupled weight decay ([arXiv:2602.16642](https://arxiv.org/abs/2602.16642)).
- Hong and Ling show dependence on architecture and data properties, and that collapse can coexist with poor generalization ([JMLR 2026](https://jmlr.org/beta/papers/v27/24-1429.html)).
- Alcala et al. derive orthoplex/softmax-code geometries once class count exceeds the simplex regime ([arXiv:2603.20587](https://arxiv.org/abs/2603.20587)).
- Tan et al. find task-intrinsic cyclic rank-2 geometry, not simplex ETF, for modular addition ([arXiv:2606.08985](https://arxiv.org/abs/2606.08985)).
- Wang et al. show ETF-like feature and classifier spaces can still be poor when misaligned under long-tailed data ([arXiv:2512.07844](https://arxiv.org/abs/2512.07844)).

## NARRATIVE ATTACK

**“That is obvious.”** Shared-anchor correlation is obvious; the damaging fact is quantitative. The exact finite-dimension null lands on 0.462, and all-anchor averaging suppresses first-order variation. The `CV=1%` headline is something the null tends to exhibit.

**“That is trivial.”** Corrected rho near 0.5 would be even more trivial. ETF edge cosines and generic high-dimensional shared-anchor cosines coincide. Fixing whitening cannot make this scalar discriminating.

## MISSION TEST

Killing rho prevents scarce compute from validating a geometric inevitability. A residual serves democratization only if it predicts which transformations can be copied into a cheaper substrate. Static beauty without compression leverage fails the mission.

## What would the result need to BE for the narrative to be unkillable?

A residual must survive full-spectrum matched clouds, random labels, imbalance, covariance-estimation reuse, shrinkage variants, and corrected multiple testing across modalities. It must predict held-out cross-family behavior after scale controls, and an intervention on it must improve a student at fixed compute. A universal raw rho can never meet this bar.
