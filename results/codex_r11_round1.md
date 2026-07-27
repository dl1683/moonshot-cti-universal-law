Verdict: accept the clean kill. Do not run Stage C, GRU confirmation, extra seeds, coefficient sweeps, or a skew-\(R\) rescue. This is a valid scientific `FAIL`, not a protocol `VOID`.

## 1. What failed

The experiment falsified this operational claim:

> Matching a teacher’s normalized anchor-space relational dynamics transfers teacher-specific competence unavailable in the calibration labels.

It failed decisively:

- Correct \(R\) lost to Haar for every seed.
- Correct observable \(R\) also lost for every seed.
- Correct artifacts were 0.65 and 1.41 percentage points worse on average.
- All models memorized calibration perfectly.
- All models remained near chance on the three unlabeled operations.
- The diagnostic probes were also near chance.

The sign-test \(p=1.0\) does not prove exact equality, but it is the worst possible result for the preregistered positive direction. The correct artifact never won once. The authoritative outcome is in [decision.json](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/geometry_admission/stage_b/decision.json:1>).

### Geometry thesis or experiment design?

Both—but in different senses.

The thesis failed operationally. Raw \(R\) and observable \(R\) are not portable capability files. They can be installed without installing the missing capability. That kills “Intelligence = Geometry” in the strong form represented by this program.

The design did not fail procedurally. The Haar comparison, paired initialization, frozen coefficients, final-only withheld evaluation, and fail-closed adjudication did exactly what they were supposed to do: expose a nonfunctional artifact.

But the design was never capable of adjudicating every possible meaning of “geometry.” It tested two particular finite-probe statistics. The repo’s earlier identifiability analysis had already established that finite anchor geometry leaves off-anchor behavior unconstrained. Stage B has now demonstrated that problem empirically, not merely theoretically.

So the honest conclusion is:

> This is a successful experiment that killed an overbroad thesis by revealing that its chosen geometric object was not functionally identifying.

It would be dishonest to downgrade this to “an optimization problem” or “we need more seeds.”

## 2. Why the loss converged without transfer

The key mistake is equating “low auxiliary loss” with “the student learned the teacher’s computation.”

The implemented object is

\[
R_j=G_j^{-1/2}\bigl(X_{j+1}-X_j\bigr)X_j^\top G_j^{-1/2}.
\]

Its rows and columns index the 64 examples in an anchor bank—not automaton states, symbols, or reusable computational variables. The implementation is explicit in [cti_geometry_admission_geometry.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_geometry.py:20>).

That creates five failure mechanisms.

1. **It describes sample flow, not the transition law.**

   \(R\) says how the relational configuration of 64 fixed sequences changes between two depths. It does not represent the reusable operator \(s\mapsto\pi_x(s)\). A new sequence has no row or column in the stored matrix.

2. **The same anchors were optimized repeatedly.**

   The student saw 32 fixed banks totaling 2,048 anchors for 5,000 updates. It could satisfy bank-specific relational constraints without discovering a shared rule for each operation token.

3. **There was no compositional factorization constraint.**

   Nothing required every occurrence of symbol \(b\), for example, to implement one common state permutation across banks and contexts. Aggregate batch geometry can be correct while individual transitions are wrong.

4. **The geometry could live in a sidecar subspace.**

   A sufficiently wide student can use one set of dimensions to drive the auxiliary loss and another to memorize the calibration labels. The classifier was never forced to causally read the \(R\)-matching subspace. The near-chance centroid probes support that interpretation.

5. **Normalization removes information.**

   Centering, global normalization, whitening, and the observable rank-eight projection discard absolute scale, offsets, readout orientation, and off-span behavior. These are legitimate invariances for comparing representations, but they are dangerous when the goal is to recover a function.

The calibration set is particularly diagnostic. It includes every one-step transition for one operation, so that entire permutation is revealed. The other three permutations—about 86.5 bits of missing information—remain unlabeled. The exact split is visible in [cti_geometry_admission_automaton.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:379>).

Therefore:

- 100% calibration accuracy proves memorization capacity.
- Low correct-\(R\) loss proves geometric plasticity on the anchors.
- Near-chance withheld accuracy proves the missing operator information was not converted into usable behavior.

One further nuance: the converged loss does not establish that the student matched “the teacher’s geometry” generally. It matched this particular normalized statistic on these fixed banks.

The fact that correct \(R\) became substantially easier to fit than Haar while producing worse withheld accuracy is especially damaging. Correct coordinate alignment was recognizable in loss space but irrelevant—or mildly harmful—in function space.

## 3. Best next directions

| Direction | Narrative | Feasibility | Distance from kill | Manifesto | Total |
|---|---:|---:|---:|---:|---:|
| Minimum Sufficient Capability Compiler | 9 | 9 | 10 | 10 | **38** |
| Universal Compute Governor | 9 | 7 | 9 | 10 | **35** |
| Geometry-Without-Capability theorem and benchmark | 8 | 10 | 7 | 6 | **31** |

### 1. Minimum Sufficient Capability Compiler

Corrected thesis:

> A capability is a compact executable program or sufficient statistic—not a representation that merely resembles the teacher’s.

Concrete experiment:

- First establish the honest denominator on the current automaton.
- Calibration reveals one permutation; encode the remaining three using an 87-bit enumerative permutation code.
- Build a frozen, teacher-free interpreter/installer that consumes that file with no per-key optimization.
- Compare against \(R\), KD, direct labels, and a transition-table baseline.
- Then move to structured affine automata over \(GF(2)^n\), where the underlying algorithm is far smaller than the full state-transition table.
- Withhold states and composition depths, not merely table entries.

Success requires:

- At least 99.9% accuracy and exact direct transitions on 100 sealed programs.
- Generalization to sequences at least four times longer than extraction queries.
- Artifact size no more than \(1.25\times\) the true program description length.
- No per-key 5,000-step training; one-shot installation or at most 50 fixed updates.
- The identical file must execute through two implementations, such as a Transformer-facing adapter and a GRU-facing adapter.
- Extraction/query cost must beat ordinary example-by-example distillation.

This preserves the “download a skill” story while replacing an insufficient proxy with an executable object.

### 2. Universal Compute Governor

New thesis:

> Intelligence becomes cheaper by allocating computation causally, not by copying internal geometry.

Concrete experiment:

- Add hard early exits or recurrent depth to the existing automaton students.
- Train a small controller to choose how much computation each example receives.
- Optimize accuracy under a frozen FLOP or energy budget.
- Evaluate on sealed keys, unseen sequence lengths, and deliberately mixed difficulty.
- Then replicate on two real tasks using models from the canonical registry.

Success requires:

- At least 50% reduction in measured energy or FLOPs at no more than 0.5 percentage-point accuracy loss.
- Or at least five points higher hard-slice accuracy at identical average compute.
- Controller overhead below 5%.
- Replication across Transformer and recurrent substrates and five seeds.
- Pareto dominance over every static-depth model, not merely one chosen baseline.

This is farther from capability transfer, but it serves the manifesto most directly: intelligence as allocation rather than scale.

### 3. Geometry Without Capability

Turn the kill into a rigorous negative result:

> Neural geometry can be copied almost perfectly without copying the computation it appears to describe.

Concrete package:

- Prove a finite-probe non-identifiability theorem: a model can satisfy arbitrary anchor \(R\), fit calibration perfectly, and implement arbitrary behavior off the anchors.
- Construct an explicit two-branch student with a geometry sidecar and an independent functional branch.
- Run paired-key swaps where only one withheld permutation changes.
- Add a frozen cross-key decoder to determine whether the artifact contains recoverable key information that the installer cannot use, or whether \(R\) discarded it entirely.
- Lesion the geometry branch after training.

Success requires:

- Auxiliary loss below \(10^{-3}\) and 100% calibration.
- Withheld behaviors ranging from chance to perfect while anchor-\(R\) remains matched within tolerance.
- Lesioning the geometry branch destroys at least 90% of measured alignment while changing accuracy by less than one point.
- Swapping correct-key \(R\) fails to produce edge-specific signed behavioral changes.

This could be a strong negative paper and would prevent the field from confusing representation matching with capability transfer. It is less directly constructive for the manifesto, so it should be tightly scoped.

## 4. Recommendation

Pursue the **Minimum Sufficient Capability Compiler**.

It is the cleanest intellectual correction:

- The kill says resemblance is not sufficiency.
- Random permutation automata have essentially no compressible algorithm beyond their transition tables.
- Therefore the next test should use structured programs and judge files by description length, query complexity, exact execution, and compositional generalization.
- The cheapest realization is allowed to be symbolic, neural, or hybrid. Requiring everything to remain a neural representation would repeat the same category error.

Immediate steering decision:

1. Permanently close GAT Stage C and conditional GRU confirmation.
2. Preserve Stage B as the canonical clean kill.
3. Perform the Geometry-Without-Capability analysis only as a bounded closeout package.
4. Lock `MINIMUM_SUFFICIENT_CAPABILITY_FILE_V1` before implementation.
5. Make the 87-bit explicit automaton file the baseline every learned artifact must beat or justify.
6. Stop the new direction if it requires per-key optimization or cannot outperform direct program transmission.

The manifesto should change from **“Intelligence = Geometry”** to:

> **Intelligence = the cheapest executable structure sufficient for the behavior.**

That survives this kill. The strong geometry thesis does not.

