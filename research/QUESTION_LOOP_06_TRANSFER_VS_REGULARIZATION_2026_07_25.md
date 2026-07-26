# Question Loop 06 — Transfer vs Regularization: The Decisive Distinction

**Date:** 2026-07-25
**Decision:** "Transfer" is not a synonym for "improved accuracy." It is an operational claim requiring teacher-identity-specific, causally mediated evidence. Until DG-0 passes all four tests below, classify any A_r gain as regularization.

## The problem

Every geometry-matching distillation loss in the literature (FDD, MTA, Procrustes/Gram, LoRi) implicitly claims to transfer teacher knowledge. But three independent lines of evidence (QL2-QL5) show that:

1. A_r matching has a huge null space — the student can satisfy it in a decorative subspace (QL5, cosmetic sidecar).
2. Reverse and out-of-order layer matching can rival forward matching (Yu et al.), attacking the "ordered program" narrative.
3. Hidden-layer distillation's most stable benefit looks like a warm start, not knowledge transfer (Guigon et al.).

The default explanation for any DG-0 gain is therefore **regularization**: the loss prevents collapse, damps unstable updates, or selects a smoother basin — without conveying teacher-specific computational structure.

## What "transfer" means operationally

Transfer is the claim that the student acquires capability X **because** the teacher had capability X and the geometric loss conveyed it. This requires all four of:

### Test 1: Teacher identity specificity

Use two teachers with **crossed competence** — e.g., Teacher-A stronger on algebraic composition, Teacher-B stronger on distractor resistance — at identical architecture, trace volume, and compute.

- Correct-teacher A_r must selectively improve the capability the teacher is better at.
- Wrong-teacher A_r must NOT improve that capability (or must improve the wrong one).
- Structure-matched random A_r (same spectrum, smoothness, norm) must fail on both.

If all three stabilize training similarly, the mechanism is regularization.

### Test 2: Causal use of matched subspace

After training with A_r matching:

- Identify the subspace where teacher and student A_r agree most.
- Causally ablate that subspace (project it out or rotate it randomly).
- If capability survives ablation, A_r matching didn't cause the capability.

Additionally, compute the output Jacobian through the matched subspace. If the readout head doesn't route through the matched features, A_r is decorative.

### Test 3: Timing separation

Apply correct vs randomized A_r only as an **early warm start** (first 20% of training), then remove the loss.

- If final performance **converges** (warm-start and full-training arms match), the mechanism is initialization/optimization, not persistent transfer.
- **Persistent separation** after matched total training, coupled with teacher-skill specificity, is stronger transfer evidence.

### Test 4: Matched-strength alternative regularizers

Compare A_r matching against:

- Update-norm regularization (||Delta H|| target matched to teacher)
- Orthogonality regularization (prevent representation collapse)
- Jacobian/Lipschitz smoothness constraints
- Gradient-norm matching
- Temporal smoothness (low-pass filter on updates)

All matched to the same compute and gradient-norm budget. If any generic regularizer matches DG-0's gain, the claim reduces to "geometry-shaped regularization" — useful engineering, not a universal law.

## The A-only witness experiment

This is the cheapest decisive test and should run first:

1. Freeze a random or label-permuted output head.
2. Train the student backbone to minimize A_r error only (no labels, no logits, no task loss).
3. Measure: can the student achieve near-zero A_r error while staying at chance accuracy?

If yes: A_r matching is **not sufficient** for capability. The null space is exploitable in practice, not just in theory.

If no (A_r error cannot be minimized without also learning the task): that's weak evidence of entanglement, but still not transfer — the student might be learning A_r as a byproduct of learning the task, not vice versa.

## What each verdict means for the program

| Verdict | Implication | Action |
|---------|-------------|--------|
| All 4 tests pass | A_r transfers teacher-specific computation | Proceed to cross-substrate (transformer-to-SSM) transfer |
| Tests 1-2 pass, 3-4 fail | Transfer exists but is equivalent to generic stabilization | Useful KD method; retire the "universal law" narrative |
| Test 1 fails | No teacher specificity — pure regularization | Kill DG-0 transfer claim; evaluate as a regularizer only |
| A-only witness succeeds at chance | Null space is exploitable | Must add readout-use constraints before any transfer claim |

## NARRATIVE ATTACK

"You showed that matching a matrix makes training better. That's what every KD paper since Hinton 2015 has shown. The word 'transfer' is a narrative addition, not an empirical finding. Show me that the student computes differently — in a teacher-specific, causally verified way — or stop calling it transfer."

This attack wins until Test 1 + Test 2 both pass with large effect sizes.

## MISSION TEST

Does DG-0 deliver **teacher-specific knowledge** to the student, or does it deliver **generic training stability** that any well-tuned regularizer provides? The democratization moonshot needs the former: scarce, expensive teacher competence flowing to cheap students. If the mechanism is regularization, use the cheapest regularizer and redirect the moonshot.
