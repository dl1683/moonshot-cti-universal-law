# Question Loop 07 — Evidence Ladder to a Portable Geometric Program

**Date:** 2026-07-25
**Decision:** “Portable geometric program” is not one empirical threshold. It is the conjunction of teacher-specific information, real utility, causal identification, cross-substrate portability, frozen reuse, program-like composition, and favorable economics. Eight ordered rungs are the minimum honest ladder from the present zero-result state. Earlier rungs can be publishable; only the last two support the moonshot narrative.

## Claim boundary

Even the complete ladder cannot prove the metaphysical universal “intelligence is a portable geometric program.” It can establish a strong, falsifiable scientific claim:

> A compact relational generator extracted from one model can causally transmit, compose, edit, and reuse nontrivial competence across substantially different model substrates at lower cost than existing transfer channels.

That result would make the larger manifesto credible. Calling a hidden-state loss a “program” before it composes, edits, relays, or runs across substrates is category inflation.

Compute estimates below are incremental planning ranges for one 32 GB RTX 5090, not benchmarked promises. They assume cached teacher traces, 512-token training sequences, mixed precision, full tuning at 0.5–0.6B, parameter-efficient tuning at 4–8B where needed, and automated overnight sweeps.

## The ladder

### Rung 0 — Controlled information channel

**Claim.** \(R\), and specifically its skew-containing component, can carry task-specific information that a teacher-invariant regularizer cannot possess.

**Demonstration.** Run the eight-key, 12-state automaton experiment from QL6. The correct-key trace must beat wrong-key, Haar/spectrum-matched, static-\(G\), smoothness, and no-auxiliary controls by at least 20 points; the key-cluster 95% lower bound must exceed 10 points; a frozen student-state probe must recover at least 70% of withheld transitions.

**Kill.** Correct and structure-matched traces tie, or \(R\)-loss falls without recovery of unseen transitions. Stop using \(R\)-match as evidence of transferred computation.

**Cost / time.** 10–20 GPU-hours; 3–5 days.

**Must survive.** Nothing empirical; QL3's algebra, QL4's controls, and QL5's sidecar attack become executable.

**If it fails.** Test a perturbation-response kernel or sketched Jacobian on the same locked task. Do not escalate the current object to language models.

### Rung 1 — Useful real-task auxiliary loss

**Claim.** On natural-language reasoning, full/skew-\(R\) adds capability beyond the complete July 2026 distillation Pareto set at matched resources.

**Demonstration.** Qwen3-4B→Qwen3-0.6B on two frozen generated families: strict semantic variants and unseen operation-depth extrapolation. Compare strong output/on-policy KD, FDD, multilayer Gram/Procrustes, LoRi-style moments, static path, strain, full/skew \(R\), four structured controls, and compute match over five seeds. Require at least 3 absolute points over the strongest baseline on each family with a family-bootstrap 95% lower bound above zero, at least 20% incremental teacher-gap closure, and at least 1.5 points from a skew-containing arm beyond path plus strain.

**Kill.** The best \(R\) arm misses any gate, or the gain is matched by more output KD, static path, generic regularization, or extra compute. Kill DG-0 as a privileged training method; retain only a cheaper regularizer if it wins economically.

**Cost / time.** 120–220 GPU-hours; 2–3 weeks.

**Must survive.** Rung 0's correct-key specificity, including held-out and structure-matched controls.

**If it fails.** The strong geometric-program path collapses for \(R\). A different dynamic object must restart at Rung 0 rather than inherit credit.

### Rung 2 — Teacher-specific transfer on real competence

**Claim.** The language-model gain follows what the teacher knows, not merely the shape of its trace.

**Demonstration.** Run QL6's isogenic \(T_D/T_R\) crossed-teacher experiment. Require both teacher competence gaps \(\geq15\) points; both student source effects \(\geq5\) points with 95% lower bounds above 2; oracle-select \(>\) swapped by \(\geq5\) points; correct source \(>\) SFT, matched random, and generic regularization by \(\geq3\) points per specialty; and competence-effect correlation \(\rho\geq0.70\).

**Kill.** All teacher targets stabilize similarly; correct only hurts less than wrong; the effect follows trace norm/domain ID; or it disappears under re-batching. Reclassify Rung 1 as regularization, not transfer.

**Cost / time.** 120–200 GPU-hours; 2–3 weeks, partly shareable with Rung 1.

**Must survive.** The same baseline advantage and controlled-channel result. Teacher-specificity may not trade away real utility.

**If it fails.** A useful-loss paper remains possible if Rung 1 passed. The portable-program narrative stops.

### Rung 3 — Minimal object is identified and causally used

**Claim.** The incremental signal is a compact directed relational generator, not denser supervision, a decorative subspace, or a fragile batch statistic.

**Demonstration.** Across both real task families: skew/full \(R\) must beat path-\(G\), exact strain, FDD derivative, random skew, sign flip, depth permutation, and equal-rank low-rank controls. Five re-batchings change effects by \(<1\) point. A predeclared geometry-bearing-subspace intervention removes \(\geq50\%\) of the teacher-selective gain while an orthogonal norm-matched intervention removes \(\leq20\%\). A rank-\(s\) trace using \(\leq10\%\) of full storage retains \(\geq90\%\) of the gain.

**Kill.** Symmetric/path supervision matches the gain, output predictions ignore the matched subspace, or success needs the original batch/layer map. Then \(R\) is not an identified portable object.

**Cost / time.** 160–280 GPU-hours; 3–5 weeks.

**Must survive.** Rungs 0–2's information, accuracy, and teacher-specific effects under the compressed representation.

**If it fails.** Narrow the claim to empirical feature KD or replace \(R\) with a perturbation-response object. Do not call the representation a compiler IR.

### Rung 4 — Cross-substrate transfer at moonshot compression

**Claim.** One teacher trace can improve students with different computation substrates at at least 10× parameter compression.

**Demonstration.** Use one 7–8B teacher and the same frozen trace artifact for both a 0.5–0.6B transformer student and [`tiiuae/Falcon-H1-0.5B-Instruct`](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct), a hybrid Transformer–Mamba model. This is roughly 14–16× teacher/student compression. Freeze the clock, pooling, rank, loss coefficient rule, anchors, and evaluation before the second substrate. On two fresh reasoning families, require \(R\) to beat each substrate's strongest output/static baseline by at least 3 points with positive family-bootstrap lower bounds and close at least 20% of each teacher gap. Inference cost must be unchanged.

**Kill.** The effect survives only in Qwen→Qwen, requires a hand-built semantic adapter for Falcon-H1, or one substrate has a material regression. The method is architecture-specific KD.

**Cost / time.** 220–380 GPU-hours; 4–6 weeks.

**Must survive.** The same teacher specificity, skew contribution, causal-use test, and low-rank trace from Rungs 0–3.

**If it fails.** Publish the same-family result honestly if strong, but retire “substrate-neutral” and “portable.”

### Rung 5 — Frozen compiler generalizes

**Claim.** A fixed extractor, trace schema, alignment clock, rank budget, and coefficient-selection rule work on unseen teacher/student/task combinations without pair-specific tuning.

**Demonstration.** Pretrain/freeze the compiler on the Qwen math pair. Blindly apply it to: one unseen teacher family; two unseen students, including the hybrid substrate; and three task families spanning arithmetic composition, executable code/process reasoning, and symbolic state tracking. Permit only a global token/state pooling rule declared before evaluation. Require a win over the strongest matched baseline on at least five of six unseen pair-task cells, mean advantage \(\geq2\) points with hierarchical-bootstrap lower bound \(>0\), no cell worse than baseline by more than 1 point, and \(\geq15\%\) gap closure in each task family.

**Kill.** Every pair needs its own layer map, adapter, rank, or loss schedule; success is confined to the tuning domain; or the new teacher requires a new trace grammar. That is a recipe, not a portable compiler.

**Cost / time.** 350–650 GPU-hours; 6–10 weeks.

**Must survive.** Cross-substrate, causal, low-rank, teacher-specific gains with the entire method frozen.

**If it fails.** Search for explicit invariants that predict adapter choice from architecture metadata. Any repaired compiler must be re-tested on a newly sealed family.

### Rung 6 — Program behavior: compose, edit, and relay

**Claim.** The artifact behaves like a program rather than a training hint: independent competence modules can be combined, selectively removed, and passed through a student.

**Demonstration.**

- **Compose:** extract disjoint low-rank modules from depth and robustness specialists, combine them without a joint teacher or joint-teacher rationales, and reach within 3 points of a joint-distillation oracle on both skills while beating either single module by at least 5 points on its missing skill.
- **Edit:** remove one module and selectively reduce its target skill by at least 5 points while changing the retained skill by at most 1 point.
- **Relay:** extract the artifact from the 0.6B student and train a 0.2–0.3B grandstudent; retain at least 80% of the student's incremental gain over the grandstudent baseline.

All three must work with the frozen Rung 5 compiler and without outcome-based module selection.

**Kill.** Modules interfere like ordinary multitask losses, editing is nonselective, or the student cannot relay the signal. Then the artifact is supervision, not a portable program.

**Cost / time.** 300–550 GPU-hours; 6–10 weeks.

**Must survive.** Rung 5's unseen-pair generalization and every earlier causal/control gate.

**If it fails.** The scientific result may still be portable distillation, but the noun “program” must be removed.

### Rung 7 — Democratizing portability and independent survival

**Claim.** The program is reusable enough to change who can access competence.

**Demonstration.** Freeze and release traces, extractor, schema, training recipe, accounting harness, and negative controls. Across at least three teacher families, three student substrates, and three capability families, require: \(\geq10\times\) compression; median advantage \(\geq3\) points over the strongest matched baseline; \(\geq20\%\) gap closure; trace artifact \(\leq5\%\) of teacher weight bytes; one-time teacher extraction; student-training overhead \(\leq25\%\) versus SFT; no inference overhead; and at least 2× lower total cost than the cheapest baseline reaching equal accuracy. A genuinely independent group must reproduce the predeclared primary effect with a positive lower confidence bound.

**Kill.** The gains require online teacher access, proprietary traces, giant anchor banks, more tuning cost than output KD, or fail independent reproduction. Scientific transfer may remain; democratization and paradigm claims do not.

**Cost / time.** 600–1,200 local GPU-hours plus external compute; 3–6 calendar months because independent replication, documentation, and coordination dominate.

**Must survive.** Every earlier rung without relaxed thresholds.

**If it fails.** Optimize the artifact only if the scientific claim remains intact. Never substitute parameter compression for end-to-end cost reduction.

## Can the ladder be shorter?

The experiments can be bundled, but the logical ladder cannot. A single factorial study could combine Rungs 1–3, and a carefully frozen cross-substrate suite could combine Rungs 4–5. That yields five execution packages:

1. keyed controlled channel;
2. baseline-complete real crossed transfer plus causal ablations;
3. frozen \(\geq10\times\) cross-substrate transfer;
4. composition/edit/relay;
5. economics plus independent replication.

Removing any package leaves a fatal alternative: regularization, weak-baseline KD, architecture-specific alignment, non-programmatic supervision, or uneconomic centralization. The shortest path is therefore better experimental reuse, not lower evidentiary height.

## Feasibility and significance thresholds

The estimated local total is roughly **1,900–3,500 GPU-hours** if every rung works on its first serious attempt: 80–146 days of nonstop GPU time. Realistically, one person with one 5090 could reach Rung 4 in **3–5 months** and Rung 6 in **8–14 months**, with automation and no major redesign. Rung 7 is not achievable by one person alone because independent replication is part of its definition. Human debugging, dataset sealing, and analysis—not theoretical peak FLOPs—are likely the bottleneck.

- **Publishable:** Rung 1, if two benchmarks, five seeds, the full Pareto baseline set, and resource accounting all survive. This is a distillation paper, not a manifesto.
- **Attention-worthy:** Rung 2 is mechanistically unusual; Rung 4 is a strong headline if the same trace works at \(\geq10\times\) across transformer and hybrid/SSM substrates.
- **Paradigm-shifting:** not before Rung 6. Rung 7, including independent survival and a real cost advantage, is what could win over an informed adversary.

The loop invariant therefore remains active through Rung 5. A sympathetic reviewer may be excited earlier; an adversary is not rationally won over until the artifact demonstrates program behavior and democratic economics.

## NARRATIVE ATTACK

“This is a long ladder designed so that any positive result can be narrated upward. Hidden-state distillation is already known. Cross-architecture gains can come from shared language and data. Low-rank modules are ordinary adapters. Composition is multitask training. Compression does not democratize anything if extracting and tuning the traces costs more than generating rationales.”

This attack is correct until the same frozen artifact carries post-committed information, beats the baseline frontier, crosses a substrate boundary, composes without a joint teacher, edits selectively, relays through a student, and wins end-to-end cost accounting. The ladder is designed so no lower rung borrows the language of a higher one.

## MISSION TEST

The moonshot is served only if a small lab can download a compact competence artifact and make a cheap, architecturally different model materially better without teacher weights, online teacher calls, proprietary labels, or pair-specific engineering. If the final method is merely a better auxiliary loss for Qwen-family fine-tuning, use it—but close the universal-program project.

## NEXT DIRECTIONS

1. **Information-rate law:** portable performance may scale with recoverable teacher-specific bits per trace byte, not raw \(R\)-loss. Estimate this exactly on automata, then test whether it predicts language-transfer efficiency.
2. **Modular curvature hypothesis:** disjoint competence may occupy approximately commuting low-rank skew modules. Module commutator norm should predict composition interference before student training.
3. **Relay fixed-point hypothesis:** if the object is program-like, teacher→student→grandstudent extraction should converge to a stable low-rank generator while cosmetic teacher idiosyncrasies decay.
4. **Automatic clock hypothesis:** relational arc length, chosen without outcomes, should outperform normalized layer number across transformer and Falcon-H1. Failure would localize portability to the alignment clock rather than \(R\).
5. **Economics-first anchor hypothesis:** disagreement-selected anchor banks may retain 90% of the gain with 10% of trace bytes, making the last rung attainable on consumer hardware.
6. **Negative universality hypothesis:** some capabilities will not admit low-rank relational transport. Mapping failure as a function of task branching, required external memory, and teacher-student capacity may reveal the actual scope law even if the universal slogan fails.
