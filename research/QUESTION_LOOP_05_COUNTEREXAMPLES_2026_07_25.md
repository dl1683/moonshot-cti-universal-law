# Question Loop 05 — Can \(A_r\) Match While Capability Does Not?

**Date:** 2026-07-25
**Decision:** Yes. Perfect update-to-state matching is neither sufficient nor necessary for capability, and current DG-0 controls do not distinguish information transfer from teacher-shaped regularization. The transfer interpretation remains unearned until it passes teacher-identity, causal-use, and structure-matched-random controls.

## The exact non-identifiability

Let \(H\in\mathbb{R}^{B\times d}\) have full row rank and let the desired target be \(A\in\mathbb{R}^{B\times B}\). A student can realize that target with

\[
\Delta H=A(HH^\top)^{-1}H+Z,\qquad ZH^\top=0.
\]

Then \(\Delta H H^\top=A\) exactly. The arbitrary \(Z\) component is invisible to \(A\), and neither term has to encode a correct answer. Even if \(G=HH^\top\) is also matched, the hidden states remain identifiable only up to an isometry on the probed span; a wrong or ignored readout can still destroy task performance.

There is also an exact relationship often blurred by the “Gram velocity” language:

\[
G_{r+1}-G_r=A_r+A_r^\top+\Delta H_r\Delta H_r^\top.
\]

Thus \(\operatorname{sym}(A_r)\) is only a first-order Gram change when updates are small. The skew component is genuinely absent from the Gram change, but that does not make it semantic or causally used.

## Three counterexample families

| Scenario | Construction and plausibility | Negative control that catches it | Does skew/connection rescue the claim? |
|---|---|---|---|
| **Cosmetic sidecar** | Give the student a hidden subspace or auxiliary branch that fits \(A_r\) while the answer head ignores it. More trivially, match every \(A_r\) and then permute answer-token rows of the output head. Capability collapses while \(A_r\) is unchanged. This is algebraically certain and practically **highly plausible** with excess width, adapters, or a fixed probe set. | Measure \(A_r\) on held-out examples and interventions, then causally ablate/rotate the matched subspace. Train an **A-only witness** with no labels/logits and a frozen random or label-permuted head. If it reaches the target \(A\)-error while staying at chance or below SFT, \(A\) is not sufficient. Add an output-Jacobian/readout-use assay rather than another correlation. | No. A sidecar can realize an arbitrary skew target too, and \(Z H^\top=0\) remains invisible. Whitening may make probe-space matching tighter but can amplify low-rank/noisy directions. |
| **Different mechanism, good answers** | A functionally strong student can use different coordinates, depth timing, features, or an output-only/on-policy route and therefore ignore the teacher \(A_r\). Adjusted downstream weights can preserve the function under hidden reparameterization while changing the observed update-to-state matrices. Yu et al.’s reverse-layer result and strong output-KD systems make this **highly plausible**. | Include TSD/RG-style output KD and FDD as first-class arms. Test necessity directly: penalize or randomize \(A_r\) while preserving logits and see whether capability survives. Require improvements in \(A\)-match to mediate capability across seeds, teachers, and tasks; a good non-\(A\) arm falsifies necessity but not possible usefulness. | It narrows what static Gram matching misses, but a useful computation can follow another connection, gauge, or depth clock. “Different skew” is not “wrong reasoning.” |
| **Teacher-shaped regularizer** | The loss may stop collapse, damp excessive residual updates, smooth optimization, or select a stable basin without conveying teacher-specific knowledge. Guigon et al.’s warm-start interpretation, FDD’s degradation when too many layers are constrained, and TSD’s benefit from selecting only some tokens make this the **default plausible explanation**. | Compare against update-norm, orthogonality, Jacobian/Lipschitz, temporal-smoothness, and gradient-norm-matched regularizers. More importantly, use wrong-task, weak, and untrained teachers plus Haar-conjugated/spectrum- and smoothness-matched \(A\) targets. Cross teacher identity and skill: the gain must follow the teacher’s differential competence. | Skew could simply enforce rotation/anti-collapse. A sign flip is insufficient because it preserves norm and spectrum. A spectrum-matched random skew and a wrong-teacher skew are the relevant controls. |

There is already an empirical warning, though not yet for \(A_r\): [Bhattarai et al.](https://arxiv.org/abs/2509.25253) report that feature-only alignment can coexist with collapsed task performance. On their single-layer BERT test, CKA-only scores 10.66 on CoLA and 47.29 on RTE, versus 51.02 and 61.73 for fine-tuning; Procrustes becomes competitive only when combined with output supervision from teacher logits and/or labels. Geometric similarity does not tell the output head which function is correct.

There is a further **batch-composition shortcut**. Because \(A_r\) lives in sample space, its off-diagonal entries depend on which examples are co-batched and on how token states become rows of \(H\). A length, template, or answer-format partition can dominate the apparent geometry. Evaluate the same held-out examples under multiple independently sampled co-batches, length/format-matched adversaries, and a fixed unseen anchor bank. If the loss or its predictive relationship to accuracy is not stable under re-batching, the alleged dynamic object is a minibatch statistic.

## The central confound: preventing bad states versus transferring good ones

Yes: matching \(A_r\) could merely prevent the student from learning bad representations. That is still potentially useful engineering, but it is **regularization**, not evidence that the teacher’s computational geometry was transferred.

The discriminating experiment is teacher-specific counterfactual transfer. Use two teachers with crossed competence—for example, one stronger on algebraic composition and one stronger on distractor resistance—while holding architecture, trace volume, target norms/spectra, and student compute fixed. Correct-teacher \(A_r\) should selectively improve the capability on which that teacher is better. Wrong-teacher and structure-matched-random targets should not. If all three stabilize training similarly, DG-0 found a generic geometric prior.

A second test is timing. Apply correct versus randomized \(A_r\) only as an early warm start, then remove it. If final performance converges, the mechanism is initialization/optimization. Persistent separation after matched training, coupled with teacher-skill specificity and causal use of the matched subspace, is stronger transfer evidence.

## GSM8K is too easy to carry this claim

GSM8K is suitable for catching implementation failures. It is poor as the sole discriminator because it is fixed, template-rich, answer-scored, contamination-prone, and already familiar to Qwen-family models. A student can improve through memorized templates, answer priors, rationale imitation, or generic stabilization without acquiring the proposed dynamics.

The next gate should use generated, held-out problem families:

- [GSM-SEM](https://arxiv.org/abs/2605.07053) makes fresh stochastic semantic variants and reports large declines under stricter shifts. Use maximum strictness with seeds and templates unavailable during training.
- [GSM-Infinite](https://proceedings.mlr.press/v267/zhou25m.html) generates controllable computational graphs and tests length/complexity extrapolation. It is harder to game with surface templates.
- [L0-Bench](https://arxiv.org/abs/2503.22832) offers exact executable process traces with controllable step depth. It can test whether dynamic matching transfers multi-step execution rather than only final-answer style.

At minimum, cross a fresh semantic shift with an unseen reasoning-depth range. The decisive metric is family-level exact accuracy with bootstrap over problem families, not repeated samples from one public test file.

## What a strong negative looks like

The trivial mathematical negative is decisive about sufficiency: match \(A_r\) perfectly, scramble the output head, and fail. The scientifically important negative is harder:

> On unseen GSM-SEM/GSM-Infinite families, an A-only or wrong-teacher student reaches the same held-out normalized \(A\)-error as the correct-teacher arm, yet performs no better—or worse—than SFT; ablating the matched subspace leaves its predictions unchanged.

That result would kill the claim that low \(A\)-error demonstrates transferred computation. It would not kill \(A_r\) as a regularizer. Conversely, if low error is attainable only on training probes but not fresh families, DG-0 has learned a trace discriminator rather than a portable dynamic law.

## NARRATIVE ATTACK

“\(A_r\) is an auxiliary statistic with a huge null space. The student can satisfy it in a decorative subspace, ignore it when solving the task, or benefit only because it limits unstable updates. The skew term sounds connection-like, but it is gauge-sensitive and can regularize rotations without carrying teacher-specific information.”

The attack stands until DG-0 demonstrates teacher-specific, causally used transfer—not merely correlation between a lower loss and a better model.

## MISSION TEST

Would the method transfer scarce teacher competence into a smaller, different substrate, or merely make ordinary fine-tuning more stable? The democratization moonshot requires the former. If a cheap generic regularizer matches the result, use the cheap regularizer and retire the universal-law narrative.

## Next-gate specification

Before the main pilot, run a **counterexample mini-gate** on the cached traces:

1. verify numerically that an A-only witness can achieve near-zero \(A\)-loss with a frozen wrong head;
2. compare correct-teacher, wrong-teacher, and spectrum/smoothness-matched random \(A\) and skew targets;
3. match gradient norm, training FLOPs, and early-warm-start duration against generic stability regularizers;
4. pre-register causal matched-subspace ablation and output-Jacobian use tests;
5. require held-out re-batching/anchor-bank stability and length/format adversaries;
6. reserve unseen GSM-SEM families and a GSM-Infinite depth-extrapolation split.

If correct-teacher geometry does not outperform every structure-matched target in teacher-skill-specific and causally mediated fashion, classify DG-0 as regularization and stop the transfer-law claim.
