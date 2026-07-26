The right answer is a modified version of your counter-proposal.

Keep training to length 32 and keep anchors at 8–32. Do not require 99% on arbitrary mixed sequences of length 17–32. But gating only on 1–16 is insufficient: add a capacity gate on the exact Stage B/C input family.

## Why your core idea is right

Reducing everything to length 16 would discard a useful source of representational variation. I overstated the case for doing that.

The extraction’s six-transition “depth clock” comes from model layers, not token count ([spec](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/OPEN_CAPABILITY_FILE_GEOMETRY_ADMISSION_STAGE_A_2026_07_25.md:290>)). Still, longer sequences give:

- More varied internal computations.
- More possible perturbation locations.
- More heterogeneous final-token representations.
- Better odds that the artifact captures composition rather than only lookup-table geometry.

So retain:

- Training lengths 1–32.
- Anchors 8–32.
- The current 12-state/four-operation task.
- The current teacher and 10× student architectures.
- Stress 33–64 as diagnostic only.

## Where your counter-proposal needs one addition

The Stage B/C endpoint is not arbitrary random composition. It is the structured family:

\[
q^p\,x\,q^r,\qquad p,r\in[0,15],
\]

with exactly one withheld edge and total length 1–31 ([design](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/STEERING_DIALOGUE_R4_ROUND4_STAGE_BC_2026_07_26.md:98>)).

Therefore the proper capacity question is:

> Can this architecture solve the eventual evaluation family when given full supervision?

That is much more relevant than whether it reaches 99% on arbitrary mixed 17–32 sequences.

Adopt three separate measurements:

1. `covered_core`, lengths 1–16 — blocking capacity gate.
2. `target_family`, the exact fully supervised \(q^p x q^r\) distribution — blocking capacity gate.
3. `covered_long_mixed`, random lengths 17–32 — non-blocking representation diagnostic.

Do not call the third one extrapolation.

## Frozen R9 gates

Teacher:

- ≥99.5% on `covered_core`.
- ≥95% on fully supervised `target_family`.
- 48/48 direct edges.
- Two consecutive evaluations.
- All extraction numerical gates pass.

Transformer student:

- At least two of three seeds reach ≥95% on `covered_core`.
- Those seeds reach ≥95% on fully supervised `target_family`.
- Those seeds score 48/48 direct edges.
- No seed below 90% on `covered_core`.

Keep `covered_long_mixed` accuracy and accuracy by individual length in the report, but do not block extraction or transfer on it.

Why 95%? Because the Stage C scientific effect must be at least 20 points with a lower confidence bound above 10 points. A full-supervision ceiling error of at most five points is comfortably smaller than the effect being adjudicated. At 61.8%, the ceiling error is 38.2 points and overwhelms the experiment; at 95%, it does not.

## Is an easy capacity gate meaningful?

Yes. A capacity gate is not supposed to be the scientific achievement.

Its purpose is to remove this alternative explanation:

> “The artifact failed only because the student architecture could not solve the evaluated task even with all labels.”

The meaningful test remains:

- Only 12 labeled edges.
- Thirty-six withheld edges.
- Correct artifact versus no auxiliary, smoothness, static geometry, wrong-key, and Haar controls.
- A ≥20-point advantage.
- Replication across sealed keys.
- No teacher access during installation.

Making the capacity preflight difficult does not strengthen that causal test. It adds an unrelated failure mode.

## Are incorrect long anchors fatal?

No, but they narrow the claim.

The observable basis uses the teacher’s predicted top-vs-runner margin ([extraction](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_extraction.py:204>)). On an incorrectly answered anchor, it therefore measures the geometry supporting the teacher’s actual decision, not the correct answer.

That is acceptable for testing:

> “Does this artifact contain key-specific information about withheld transitions?”

It is not sufficient for claiming:

> “The artifact faithfully transfers the teacher’s long-horizon competence.”

The wrong-key and Haar controls will determine whether the mixed-quality long-anchor artifact nevertheless carries usable transition information. If it does not, Stage B legitimately fails.

Do not filter anchors based on teacher correctness; that would introduce key- and outcome-dependent anchor selection.

## What to do with Student 1

Let the current clean 15,000-step run finish.

At the time I inspected it, its restarted full-15K segment was around 84.0% at step 6,250 and had gained approximately three points over the preceding 1,000 steps. Reaching 95% is plausible, but not assured; simple curve projections are too unstable to adjudicate now.

Use this stop rule:

- If Student 1 reaches ≥95% on both `covered_core` and `target_family`, run seeds 202 and 203.
- If it reaches ≥95% core but fails `target_family`, the current Stage B evaluation family is incompatible with the student and the protocol needs redesign.
- If it remains below 95% core, stop. Do not launch the other seeds or enlarge the model.
- Do not require Student 1 to learn arbitrary mixed length-17–32 sequences before extrapolation learning begins; that is no longer the relevant gate.

## Final steering decision

Proceed with R9 as follows:

- Keep max training length 32.
- Keep anchors 8–32.
- Keep the current teacher.
- Finish Student 1.
- Replace the 99% random-long gate with a 95% exact-target-family gate.
- Keep random-long and stress-long as diagnostics.
- Run extraction once teacher target-family performance and numerical gates pass.
- Launch the remaining students only after Student 1 clears the revised gate.

This preserves the rich-representation hypothesis without allowing unrelated long-horizon Transformer weakness to veto the actual withheld-edge transfer experiment.

