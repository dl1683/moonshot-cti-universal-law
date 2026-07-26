# Question Loop 08 — Generative Directions for DG-0

**Date:** 2026-07-25  
**Decision:** Do not position DG-0 as a richer hidden-state loss. The strongest program is to turn teacher-specific, causally observable transport into a compact capability patch. Raw skew is the discovery object; **observable connection** is the likely transferable object.

## Executive decision

The mathematical distinction established in QL2–3 is real:

\[
\Omega_r=\operatorname{skew}(\mathcal R_r)
\]

contains directed first-order transport absent from a Gram trajectory. But QL5–6 also established that arbitrary skew can live in a decorative subspace, depend on gauge or batch composition, and regularize without transferring competence.

The generative move is therefore not “use more skew.” It is:

> **Find the smallest part of the teacher's directed transport that is teacher-specific, reachable from task perturbations, used by the readout, and reusable across students.**

The strongest one-sentence story for the whole direction is:

> **Package a large model's missing skill as a tiny geometric patch and install it in any cheap model—without copying its weights or rationales.**

That story is paradigm-level only if the patch is differential, causally used, architecture-neutral, compact, and cheaper than existing distillation or parameter-delta skill packs.

## DIRECTIONS

### Ranked portfolio

| Rank | Direction | Impact | Feasibility | Why it is above ordinary KD |
|---:|---|---:|---:|---|
| 1 | **Observable Connection Codec** | 10/10 | 7/10 | Replaces “match all geometry” with the minimal controllable-and-observable transport program and opens a capability bound |
| 2 | **Differential Capability Patches** | 10/10 | 6/10 | Turns the QL6 crossed-teacher falsifier into a product: transferable skill deltas rather than whole-model imitation |
| 3 | **Closed-Loop Intelligence Transfer** | 9/10 | 6/10 | Tests skew where direction, memory, and compounding errors matter most: long-horizon control and algorithmic execution |
| 4 | **Gauge-Invariant Geometric Phase** | 10/10 | 4/10 | Moves beyond a gauge-sensitive depth statistic to order-sensitive loop structure that static geometry cannot encode |

The ranking is by impact multiplied by feasibility, not by mathematical elegance. Direction 1 is the best immediate pivot. Direction 4 may be the deepest eventual result, but it should not delay the cheap tests.

### 1. Observable Connection Codec

#### Proposal

Raw \(\Omega_r\) counts every directed mode, including modes the output ignores. Import the controllability/observability distinction from model reduction.

Let \(z_r\) denote a low-dimensional relational coordinate for the probe state. Estimate:

- an empirical controllability Gramian \(W_c\) from how task-relevant input or state perturbations move \(z_r\);
- an empirical observability Gramian \(W_o\) from output Jacobians, answer-margin changes, or intervention effects downstream of \(z_r\);
- a balanced basis \(U_s\) spanning modes with the largest joint controllability/observability values.

The candidate transfer object is then not all of \(\Omega_r\), but

\[
\boxed{\Omega^{\mathrm{obs}}_r=U_s^\top \Omega_r U_s,}
\]

plus the minimal state/readout anchors required to interpret it. This is a **behaviorally balanced connection**.

Balanced truncation is the relevant intellectual bridge: it keeps dynamical modes that are both reachable and visible at the output, and classical error bounds depend on the discarded Hankel singular values. Koopman balancing extends the idea to nonlinear dynamics in a lifted observable space. This does not automatically make the theorem valid for transformers, but it gives DG-0 the right question and failure metric ([Yeung, Liu, and Hodas](https://arxiv.org/abs/1709.08712); [Fujimoto and Scherpen](https://epubs.siam.org/doi/10.1137/070695332); [Corbin and Kramer, 2026](https://arxiv.org/abs/2604.23044)).

#### The devastating theorem target

Prove a local realization bound of the following form:

> For a stable, locally linearizable teacher and student on a covered probe tube, if their readout anchors and balanced observable connections agree to error \(\epsilon\), then their output-trajectory discrepancy is bounded by a term in \(\epsilon\) plus the sum of discarded behavioral singular values. If this bound stays below the teacher's decision margin, the student preserves the teacher's decision on the covered family.

A useful bound would look schematically like

\[
\sup_{t\le T}\|y_T(t)-y_S(t)\|
\le
C_T\left(
\epsilon_{\mathrm{anchor}}
+\epsilon_{\Omega}
+\sum_{i>s}\sigma_i
\right),
\]

with every constant and coverage assumption explicit.

This would not prove universal intelligence. It would prove something more valuable than a correlation: **which skew modes may be discarded without losing a capability and why**.

#### Why this would make people reconsider

It simultaneously answers three fatal objections:

1. **Cosmetic sidecar:** output-invisible modes receive low observability weight.
2. **Cost:** a fast singular-value tail yields a small reusable trace.
3. **Causal use:** the object is selected by intervention-to-output influence, not by geometric energy alone.

The conceptual shift is from representation matching to **behavior-preserving model reduction of computation**.

#### Cheapest decisive experiment

Do not train the full baseline sweep first. On cached teacher traces:

1. choose 128–512 fixed anchors from two generated reasoning families;
2. estimate a cheap observability proxy using answer-margin gradients or a small number of JVP/VJP probes;
3. estimate controllability from semantic perturbations and hidden-state interventions;
4. compare rank-\(s\) reconstructions selected by raw skew energy, balanced observability, output Jacobian alone, and random subspaces;
5. test which subspace best predicts teacher-specific margin changes under held-out interventions;
6. in one small student run, compare full skew with balanced rank-\(s\) skew at equal gradient norm and FLOPs.

Decisive positive evidence is that 5–10% of the modes preserve nearly all teacher-specific intervention prediction and match or beat full-skew transfer. This is a trace-analysis gate plus two small training arms, not a 40–80 GPU-hour sweep.

#### Gossip-magazine test

> “Scientists found the handful of moving directions an AI actually uses to think—and copied only those.”

#### Kill condition

Kill this direction if the behavioral singular spectrum has no useful decay; energy-only or random low-rank skew performs as well; selected modes do not causally affect outputs; observability estimation costs as much as ordinary KD; or the basis must be retuned for every student.

### 2. Differential Capability Patches

#### Proposal

Stop transferring an entire teacher. Transfer the **difference made by acquiring one capability**.

Given a base teacher \(T_0\) and a specialist \(T_k\) derived from the same base, define

\[
\Delta\Omega_k^{\mathrm{obs}}
=
\Omega^{\mathrm{obs}}(T_k)
-
\Omega^{\mathrm{obs}}(T_0).
\]

Install that delta into a student while preserving its existing behavior:

\[
\Omega_S'\approx\Omega_S+\alpha_k\Delta\Omega_k^{\mathrm{obs}}.
\]

This turns QL6's crossed-competence teacher test from a defensive control into the central artifact. The desired unit of intelligence is no longer a checkpoint. It is a portable, capability-specific transport delta.

Compact parameter-delta SkillPacks already perform heterogeneous capability fusion, so “skill patch” alone is not novel. GraftLLM is an ICLR 2026 comparator and narrative threat. DG-0 must win because its patch is defined in a shared relational/behavioral space rather than the source model's parameter coordinates, and because the *same extracted patch* can teach materially different substrates ([GraftLLM](https://openreview.net/forum?id=wJtD28wGV1)).

#### Why this would make people reconsider

Successful differential patches would change the unit of model distribution:

- one expert can publish a small skill artifact instead of a full checkpoint;
- students need not share the teacher's width, layers, or parameter names;
- a patch can be audited, priced, revoked, combined, and tested independently;
- crossed improvements directly show that the signal follows teacher competence rather than generic regularization.

#### Cheapest decisive experiment

Create two specialist teachers from one backbone with deliberately crossed competence: specialist A for algebraic composition and specialist B for distractor resistance or state tracking. Use generated train and held-out families. Extract both deltas relative to the shared base and train identical small students with:

1. the correct differential patch;
2. the wrong differential patch;
3. the negated patch;
4. a spectrum/smoothness-matched random patch;
5. specialist logits with the same examples and teacher-call budget;
6. a compact parameter-delta/LoRA SkillPack comparator.

The decisive result is a crossed interaction: patch A selectively improves A, patch B selectively improves B, wrong and random patches do not, and general capability is preserved. The strongest cheap extra test is subtraction: applying \(-\Delta\Omega_A\) should selectively remove A competence from a student that has it.

#### Gossip-magazine test

> “They copied one skill out of a giant AI and emailed it as a patch to a tiny one.”

#### Kill condition

Kill the patch thesis if absolute teacher geometry is required; wrong-teacher or matched-random patches give the same gains; patches only work within one architecture family; composition causes ordinary model-merging interference; or output KD/parameter SkillPacks win at equal artifact size and compute.

### 3. Closed-Loop Intelligence Transfer

#### Proposal

Move the flagship application from one-shot math answers to systems whose capability is a transition law: partially observed agents, tool-use policies, stateful algorithms, and long-horizon execution.

This is where skew should matter most. Static states and endpoint logits can agree while the direction of memory update is wrong. Small local errors then move the student onto unseen states and compound. Current agentic distillation explicitly tries to densify sparse long-horizon outcomes with on-policy skill signals, making it a stronger frontier than offline GSM8K alone ([SEED](https://arxiv.org/abs/2607.14777)).

The cross-field claim is not that a transformer “is a physical control system.” It is that the same mathematical problem appears in both:

> preserve an input-output behavior while replacing a high-order internal realization with a cheaper one.

The neuroscience analogy is suggestive but secondary. Rotational population dynamics have been linked to motor generation and to protecting memory from interference, yet rotational diagnostics also admit simple sequence-based null explanations. That history is a warning to demand causal interventions, not a source of proof ([Churchland et al.](https://www.nature.com/articles/nature11129); [Libby and Buschman](https://www.nature.com/articles/s41593-021-00821-9); [Michaels et al. critique](https://www.nature.com/articles/s41598-019-54760-4)).

#### The devastating theorem target

Prove a horizon-dependent simulation bound:

> Under a declared common probe interface, incremental stability, and observable-subspace coverage, connection error \(\epsilon\) implies bounded policy/output error over horizon \(T\); below a predeclared action margin, the student takes the same actions as the teacher.

The bound should expose failure, for example:

\[
\|y_T-y_S\|_{[0,T]}
\lesssim
 e^{LT}\left(\delta_0+T\epsilon_{\Omega}+\epsilon_{\mathrm{tail}}\right).
\]

The scientific test is whether observable skew reduces the term that grows with horizon relative to logits and static geometry.

#### Why this would make people reconsider

A transformer teacher transferring a closed-loop strategy to a tiny GRU or SSM would show cross-substrate realization, memory-update transfer, horizon extrapolation, and a direct route to cheap local agents.

#### Cheapest decisive experiment

Start below language-agent scale:

1. use a deterministic partially observed environment or exact algorithmic state machine with hidden memory requirements;
2. train a modest transformer teacher and a 10× smaller GRU/SSM student;
3. reserve horizons 2–4× longer than training and unseen transition compositions;
4. compare behavior cloning/logit KD, on-policy imitation, static path Gram, FDD, full skew, and observable skew;
5. intervene on the matched recurrent subspace and re-run closed-loop rollouts.

A MiniGrid-style memory task, Sokoban subset, or exact executable process benchmark can make the first decision. Do not begin with expensive web agents.

#### Gossip-magazine test

> “A tiny recurrent agent inherited a giant transformer's strategy—not its answers—and kept working when the task became four times longer.”

#### Kill condition

Kill this application if the advantage disappears under on-policy imitation; the student fails longer horizons; the connection needs privileged deployment state; every substrate needs a bespoke gauge; or causal ablation does not selectively remove the behavior.

### 4. Gauge-Invariant Geometric Phase

#### Proposal

Take the gauge objection seriously enough to change the object.

An open one-dimensional depth path has no intrinsic curvature, and a local connection can be changed—or locally gauged away—by layer-dependent coordinates. Potentially invariant content appears around **closed loops** in an input, perturbation, or program-composition space. The path-ordered product around such a loop is a holonomy; its conjugacy invariants survive changes of frame.

\[
\mathcal H(\gamma)
=
\mathcal P\prod_{e\in\gamma}\exp(\Omega_e).
\]

Noncommuting operations produce different geometric phases even when endpoint states or static Grams are indistinguishable.

Representation holonomy is already an ICLR 2026 diagnostic. It is gauge-invariant after whitening, separates models that CKA cannot, and correlates with robustness. DG-0's opportunity is **not** to claim discovery of holonomy. It is to test whether loop transport is a transferable and causally necessary capability object ([Sevetlidis and Pavlidis](https://arxiv.org/abs/2601.21653)).

#### The devastating theorem target

Construct a separation theorem:

> There exists a family of order-sensitive tasks for which two systems have identical static Gram trajectories and identical symmetric strain on every observed edge, yet differ in loop holonomy and task behavior. Any distillation rule measurable only from Gram paths cannot distinguish them; a holonomy-aware rule can.

An even stronger result would connect the commutator term in the Baker–Campbell–Hausdorff expansion to a lower bound on order-sensitive error.

#### Why this would make people reconsider

It would identify a concrete computation that endpoints and all static layerwise geometry provably miss: noncommutative order. “Reasoning is path-dependent” would become a separation result rather than a metaphor.

#### Cheapest decisive experiment

Use exact noncommutative programs: compare \(AB\) with \(BA\), include commutator loops \(ABA^{-1}B^{-1}\), balance superficial endpoints and static Gram spectra, and test longer unseen compositions. Measure raw skew, gauge-randomized skew, and loop holonomy. Compare a holonomy loss against static path, local skew, Jacobian, and output KD. The first result should use a small exact sequence model, not an LLM.

#### Gossip-magazine test

> “Two AIs looked identical in every snapshot. Only the twist around a reasoning loop revealed which one understood order.”

#### Kill condition

Kill this direction if holonomy is unstable under probes/whitening/gauge; Jacobians or ordinary sequence supervision are cheaper and equal; loop invariants diagnose but do not transfer; the synthetic separation fails on a natural family; or the loop construction leaks the solution.

## NARRATIVE ATTACK

The strongest dismissal of DG-0 is:

> “DG-0 is jPCA for transformers wrapped in an expensive distillation loss. Its skew matrix depends on gauge, layer clock, whitening, minibatch composition, and a chosen probe bank. A student can fit it in a decorative subspace or benefit from it as an anti-collapse regularizer. Once you add output Jacobians and controllability tests, you have reinvented system identification or Jacobian matching. Once you add holonomy, you are following an existing ICLR 2026 diagnostic. Once you call the result a skill patch, you compete with parameter-delta SkillPacks that already work. There is no paradigm here until one compact artifact, extracted once, transfers one teacher's differential competence into multiple cheaper and incompatible students better than logits, on-policy KD, and parameter patches.”

This dismissal is currently correct.

The adversary becomes persuadable only with a chained result:

1. **specificity:** a differential patch carries the source teacher's crossed competence;
2. **causality:** the student's output depends on the matched observable modes;
3. **compression:** a small behavioral singular subspace retains the effect;
4. **portability:** the same artifact works in transformer, SSM/GRU, or another incompatible realization;
5. **economics:** extraction and training cost amortize below ordinary KD or distributing the teacher.

No isolated GSM8K gain wins this argument.

## MISSION TEST

DG-0 serves democratization only if the transferable artifact is more open and cheaper than the model it summarizes.

### Non-negotiable mission gates

- **At least 10× parameter compression.**
- **No student inference overhead.**
- **One extraction, multiple students:** no teacher re-querying for each substrate.
- **Compactness:** trace storage must be a small fraction of teacher weights; report the exact rate-distortion curve.
- **Low training burden:** charge added FLOPs, memory, and teacher calls.
- **Open inspectable object:** publish the probe distribution, sketch, gauge/clock contract, and validator even if weights are closed.
- **Differential competence:** transfer a scarce skill, not generic stability.
- **No privileged deployment channel:** the student runs independently after training.

| Direction | Mission value | Main risk |
|---|---|---|
| Observable Connection Codec | **Highest** — creates the compact artifact and a principled cost/error curve | Observability measurement may erase the cost advantage |
| Differential Capability Patches | **Highest** — directly distributes scarce competence | Existing parameter SkillPacks may be simpler and stronger |
| Closed-Loop Intelligence Transfer | **High** — cheap local agents are economically consequential | On-policy coverage may explain all gains |
| Gauge-Invariant Geometric Phase | **Conditional** — potentially foundational, not automatically cheap | Could remain a beautiful diagnostic with no compression leverage |

The Qwen3-4B-to-0.6B pilot remains useful as a Rung 1 implementation smoke test. It should not consume the full Rung 2 sweep before two cheap checks are added: observable-rank decay and gauge/rebatch stability. If those fail, pivot before spending 40–80 GPU-hours.

## NEXT DIRECTIONS

### QL9 — Observable connection formalism

Define \(W_c\), \(W_o\), the balanced basis, output/readout anchor, and discrete-time connection exactly. Determine what is gauge covariant or invariant. State the narrowest valid capability-bound theorem and construct counterexamples to every dropped assumption.

### QL10 — Differential skill-patch algebra

Specify base/specialist subtraction, patch scaling, addition, negation, and interference tests. Design the crossed-competence microbenchmark with GraftLLM/LoRA, logits, wrong patch, random patch, and removal comparators.

### QL11 — Gauge and holonomy adjudication

Decide whether raw depth skew is a legitimate target after legal gauge changes. Reproduce representation-holonomy nulls, separate input-space loops from depth transport, and locate novelty beyond causal transfer of an existing diagnostic.

### QL12 — Long-horizon flagship selection

Choose one cheap exact sequential environment and one natural agentic/process benchmark. Freeze the probe interface, horizon split, on-policy comparator, recurrent/SSM student, and intervention protocol.

### QL13 — Capability rate-distortion and economics

Precommit curves over rank, sketch bits, probes, JVP/VJP count, teacher calls, student FLOPs, wall time, artifact size, and teacher-gap closure. Compare against logits, rationales, parameter deltas, and ordinary low-rank compression.

### QL14 — Single-bet adjudication

Have an adversary choose the strongest remaining dismissal after QL9–13. Select exactly one flagship claim, theorem target, and decisive experiment. Freeze kill criteria before the expensive pilot. Continue the loop unless the adversary explicitly concedes that the proposed result, if observed under the frozen controls, would force a change of view.