# Question Loop 3: Minimal Dynamic Object

**Date:** July 25, 2026  
**Decision:** Admit one ablation-complete trial of normalized update-to-state transport.

## What is the smallest architecture-independent object beyond standard relational distillation?

### Decision

**The irreducible first-order addition is a relational connection, represented by the skew part of update-to-state alignment. The practical trial object should be the full normalized update-to-state matrix.**

For centered, scale-normalized probe states `H_r in R^(B x d)`, define

$$
G_r=H_rH_r^\top,
\qquad
A_r=(H_{r+1}-H_r)H_r^\top,
$$

and, on the support of G,

$$
\mathcal R_r=G_r^{\dagger/2}A_rG_r^{\dagger/2}.
$$

R removes state scale and expresses the projected update in relational coordinates. Its symmetric component is strain; its skew component is connection/circulation. Given G, it contains the same projected information as A but is more comparable across layers and models.

The minimal new information beyond a known Gram trajectory is

$$
\boxed{\Omega_r=\frac12(\mathcal R_r-\mathcal R_r^\top).}
$$

The first experiment should match full R and ablate symmetric and skew parts. Declaring skew-only success before establishing stability would optimize elegance over evidence.

## Candidate comparison

Assume B probes, hidden width d, and R sampled transitions.

| Candidate | Information beyond one static G | Architecture independence | Cost versus Gram KD | Verdict |
|---|---|---|---|---|
| **Gram velocity** `Delta G` | Pairwise expansion/contraction | Strong: sample-space, width-independent | Adjacent Grams plus `O(B^2)` subtraction | No new information if layerwise Grams are matched |
| **Update-to-state** `A=Delta H H^T` | Gram velocity plus directed skew/circulation | Conditional: common adjacent gauge/state required | Extra `Bxd` by `dxB` multiply: `O(RB^2d)`, storage `O(RB^2)` | Best minimal trial |
| **Normalized generator** R | Same projected information as A, state-normalized | Conditional, with rank/regularization choices | A plus low-rank inverse/eigendecomposition of G | Best eventual compiler representation if A works |
| **Finite-difference Jacobian** | Perturbation response and off-trajectory local behavior | Weak-to-medium; input-space relational response is most portable | p extra evaluations/JVPs for p directions | More identifying, not minimal; prior art exists |
| **Attention-transition operator** | Explicit token routing | Poor: excludes SSMs/RNNs; tokenizer/head dependent | Often `O(T^2)` per head/layer and large storage | Architecture-specific and already distilled |
| **FSP matrix** `H_r^T H_(r+1)` | Feature-channel flow | Poor across widths without adapters | `O(B d_r d_(r+1))` | Established prior art |
| **Perturbation-response kernel** | Local operator on relational state | Potentially strong in input/sample space | Prohibitively high without sketches | Long-term fallback if A loses needed off-span information |

## Why Gram velocity is insufficient

If relational KD matches G at several layers, then

$$
\Delta G_r=G_{r+1}-G_r
$$

is algebraically fixed. A Gram-velocity loss might ease optimization or continuous-depth alignment, but it transfers no information absent from the static Gram sequence.

A dynamic arm that beats one terminal Gram but not a matched multi-layer Gram trajectory has demonstrated denser static supervision, not transfer of a transformation.

## Why A is the smallest plausible addition

Because

$$
\operatorname{sym}(A_r)\approx\Delta G_r/2,
$$

the only first-order information not in the Gram path is `skew(A_r)`. It distinguishes transitions with identical relational strain but different directed transport. It is a `B x B` sample-space object, so teacher and student may have different hidden widths while producing comparable matrices. It works across transformers, SSMs, and RNNs when each exposes a persistent hidden state and aligned probes.

The independence is conditional:

- adjacent widths or token sets require adapters/pooling;
- depth needs a shared clock such as normalized compute or relational arc length;
- independent layer gauges change the update;
- A ignores update components orthogonal to the current probe span; and
- batch matching need not generalize off-distribution.

These are falsifiers, not engineering footnotes.

## Prior art and 2025-2026 literature

Yim et al.'s Flow of Solution Procedure (FSP) matrix already transfers a cross-layer Gram-like object intended to encode the direction of a network's solution process ([CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html)). FSP resembles `H_r^T H_(r+1)` in feature-channel space. Proposed A uses the dual sample-space residual product `Delta H H^T`, permitting different teacher/student widths and isolating the update. That is a real distinction, but the program cannot claim “first transfer of transformation flow.”

Jacobian matching also predates this proposal and transfers local response information ([Srinivas and Fleuret, 2018](https://arxiv.org/abs/1803.00443)). It is more informative than A but costlier and less architecture-neutral.

A targeted search found no 2025-2026 paper using the exact sample-space object `Delta H H^T`, its skew part, or normalized R as a general cross-architecture target. That is not a novelty proof. The closest work sets a demanding baseline:

- Bhattarai et al. show zero projection-MSE or CKA need not preserve feature structure and report gains from Procrustes/feature-Gram alignment ([arXiv:2509.25253](https://arxiv.org/abs/2509.25253)). This remains static even at multiple layers.
- Yu et al. find reverse and otherwise nonsensical teacher-student layer matching can work surprisingly well ([arXiv:2502.04499](https://arxiv.org/abs/2502.04499)). This attacks the idea that ordinary intermediate matching transfers an ordered program.
- Guigon et al.'s compute-controlled decoder study finds hidden-layer distillation does not consistently beat standard KD downstream, despite systematic perplexity gains in shared settings ([arXiv:2605.11513](https://arxiv.org/abs/2605.11513)).
- Lutz et al. extract an explicit depth recursion driven by mixed feature-label Gram structure in constrained in-context classifiers ([arXiv:2604.11613](https://arxiv.org/abs/2604.11613)). This supports the premise while showing identification needs strong symmetries.
- Distribution matching remains an active 2025 feature-KD baseline ([KD2M, arXiv:2504.01757](https://arxiv.org/abs/2504.01757)). A new method must beat distributional, Procrustes, Gram, and logit objectives, not only RKD.

## Minimum decisive experiment

The current DG-0 dynamic arm is too aggregated to establish novelty. The decision-grade comparison is:

1. logits/standard KD;
2. one terminal Gram;
3. **static path:** the same teacher layers as the dynamic arm, matching every G;
4. **strain:** static path plus `sym(R)`, expected to be informationally redundant;
5. **connection:** static path plus `skew(R)`;
6. **full generator:** static path plus full R;
7. closest affordable FSP/Jacobian-strength baseline; and
8. depth-permuted, sample-permuted, skew-sign-flipped, and compute-matched controls.

The decisive contrast is

$$
\text{connection/full generator}
\quad\text{versus}\quad
\text{static path + strain + compute match}.
$$

If only skew-containing arms win, the method has isolated transformation information unavailable to Gram trajectories. If static path or strain matches the gain, “dynamic geometry” is denser relational KD.

## Cost and democratization constraints

The extra A multiply has the same asymptotic order as Gram computation, `O(B^2 d)`, but approximately doubles large matrix work per transition. Storage is `O(RB^2)`. It is tolerable only for modest batches or low-rank sketches; token-level B over long contexts is not.

A viable compiler must show low effective rank of R, enabling `O(Bsd)` sketches with `s << B`; one-time reusable teacher traces; no inference overhead; training overhead small relative to saved compute; and robustness to layer resampling and student depth. If the target requires giant batches, full Jacobians, or a custom adapter for every student, it centralizes intelligence rather than democratizing it.

## NARRATIVE ATTACK

**“That is obvious.”** “Use changes, not states” is obvious. Gram velocity does exactly that and remains redundant. The only new first-order content is the directed connection term, after declaring a gauge and clock.

**“That is trivial.”** Full A is uncomfortably close to the 2017 FSP idea in dual coordinates. The story is defensible only if the sample-space residual/skew component transfers capability across incompatible widths and substrates where FSP, static path matching, CKA, and logits fail.

## MISSION TEST

This serves cheap intelligence only if a small, reusable, low-rank relational generator closes a meaningful large-teacher/small-student gap across architectures. It fails if it is a costly auxiliary loss for same-family compression or if the trace is as expensive and proprietary as the weights.

## What would the result need to BE for the narrative to be unkillable?

At least 10x parameter compression and two genuinely different teacher-student substrate pairs, including transformer-to-SSM/RNN, on two frozen benchmarks. Connection/full transport must beat logits, Procrustes, static multi-layer Grams, Gram velocity/strain, FSP, and compute-matched controls by 3+ points and close at least 20% of the teacher-student gap. The skew ablation must contribute reproducibly, permuted/skew-flipped controls must fail, trace overhead must be modest and amortizable, and inference cost must remain unchanged. Anything weaker is useful KD engineering, not a geometry compiler.
