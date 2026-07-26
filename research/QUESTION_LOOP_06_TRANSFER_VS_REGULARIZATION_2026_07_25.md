# Question Loop 06 — Transfer Versus Regularization

**Date:** 2026-07-25
**Decision:** Do not let an early-warm-start result or a correct-teacher win over an unmatched random target carry the transfer claim. The decisive test is a randomized teacher-identity intervention with crossed competence, structure-matched trace controls, and held-out capabilities whose direction must follow the teacher. Run a keyed synthetic version first; it is cheaper and has known ground truth.

## 1. Operational meaning of information transfer

“The student resembles the teacher” is not transfer. “The auxiliary loss improves accuracy” is not transfer. For DG-0, **information transfer** should mean:

> Holding student initialization distribution, examples, labels, output supervision, optimizer, auxiliary-loss strength, compute, and all declared trace statistics fixed, changing only the identity of the teacher trace causes a predictable change in the student's held-out competence that follows information uniquely possessed by that teacher.

This is a causal definition. Let \(T_D\) and \(T_R\) be teachers respectively stronger on depth composition and distractor resistance. For identically trained students \(S_D\) and \(S_R\) receiving only the corresponding teacher's \(R\)-trace, define

\[
\delta_D=q_D(S_D)-q_D(S_R),\qquad
\delta_R=q_R(S_R)-q_R(S_D),
\]

\[
\delta_{\mathrm{cross}}=\frac{\delta_D+\delta_R}{2}.
\]

A teacher-invariant regularizer can improve both students, but it does not predict the crossed sign pattern \(\delta_D>0,\delta_R>0\) after norm, spectrum, rank, depth smoothness, batch composition, and gradient magnitude have been matched. The operational claim is therefore not that \(R\) is sufficient for intelligence. It is that \(R\) is an information-bearing causal channel beyond a specified class of generic stability priors.

No finite experiment proves the absence of every possible regularization account. The experiment below can, however, reject the important class in which the gain depends only on trace structure or optimization stabilization rather than teacher-specific competence.

## 2. The decisive real-task experiment

### Use isogenic crossed teachers, not a convenient pair assumed to be crossed

The primary pair should be two LoRA-specialized copies of [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), initialized from the same checkpoint:

- **\(T_D\), depth specialist:** rank-64 LoRA, trained on 12,000 generated clean arithmetic programs at operation depth 6–12, no irrelevant clauses, two epochs, maximum 512 tokens.
- **\(T_R\), robustness specialist:** the same LoRA rank, token budget, optimizer, and number of updates, trained on 12,000 depth-3–5 programs with 6–10 irrelevant quantities, variable renaming, clause reordering, paraphrase shifts, and adversarially plausible distractors.

Use the same generator grammar and answer format for both. Hold out operation templates, lexical templates, and generator seeds. The pre-screen is 800 frozen problems per skill. Do not proceed unless:

- \(q_D(T_D)-q_D(T_R)\geq15\) points;
- \(q_R(T_R)-q_R(T_D)\geq15\) points;
- each specialist scores at least 70% on its specialty; and
- their mean accuracy across the two skills differs by at most 3 points.

This construction is deliberately less glamorous than selecting two unrelated frontier models. It makes teacher identity close to randomized: architecture, base weights, width, tokenizer, training tokens, and LoRA capacity are held fixed.

An off-the-shelf corroboration pair is [`Qwen/Qwen2.5-Math-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) and [`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct), tested on fresh GSM-Infinite-style arithmetic graphs versus generated executable program-synthesis tasks. Both are 7B-class Qwen2.5 specialists and fit one at a time on a 32 GB 5090. This is not the primary causal test: domain, post-training, and output-format differences are nuisance variables, and “crossed” must still be established locally by the same 15-point rule.

### Student, data, and arms

Use [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) as the student. Full-fine-tune it on the same 8,000 gold-labeled, balanced easy/mid-depth examples in every arm. Cache teacher states on the same 4,096 unlabeled hard anchor problems. Give no teacher-specific logits, rationales, sampled outputs, confidence, or verifier scores to any identity arm.

Evaluate on 2,000 frozen problems per skill, clustered into at least 100 independently generated problem families. Use five student seeds. The minimum eight-arm design is:

1. gold SFT only;
2. SFT plus update-norm/Jacobian-smoothness regularization;
3. SFT plus spectrum-, rank-, Frobenius-norm-, and depth-autocorrelation-matched random \(R\);
4. all anchors from \(T_D\);
5. all anchors from \(T_R\);
6. **oracle-select:** \(T_D\) trace on depth examples and \(T_R\) trace on robustness examples;
7. **swapped:** \(T_R\) on depth and \(T_D\) on robustness;
8. oracle-select static path \(G_r\) plus symmetric strain, with the same layers and gradient budget.

For the random control, use a fixed Haar conjugation per anchor bank, \(R'_r=Q R_r Q^\top\), plus a depth-consistent phase randomization if necessary to match autocorrelation. Pre-calibrate auxiliary coefficients on a disjoint pilot so median auxiliary-gradient norm is matched within 5% across arms. Charge trace extraction, storage, student FLOPs, wall time, and hyperparameter searches.

### Adjudication: what would establish transfer

The transfer verdict requires **all** of the following:

1. The teacher pre-screen satisfies the crossed 15-point competence gate.
2. \(\delta_D\geq5\) points and \(\delta_R\geq5\) points; each family-cluster bootstrap 95% lower bound exceeds 2 points.
3. Oracle-select beats swapped by at least 5 points averaged over the two skills and beats SFT, matched random, generic regularization, and static-path/strain by at least 3 points on each corresponding specialty.
4. The sign of the teacher advantage is reproduced in at least four of five seeds. Across at least eight subfamilies, the student teacher-source effect tracks the teacher competence difference with Spearman \(\rho\geq0.70\), 95% lower bound above 0.40.
5. A skew-containing target contributes at least 1.5 points beyond static path plus strain. Wrong, Haar-conjugated, sign-flipped, and depth-permuted targets do not reproduce the selective gain.
6. Re-batching the held-out anchors five ways changes the selective effect by less than 1 point.
7. A predeclared causal intervention on the geometry-bearing residual subspace removes at least half of \(\delta_{\mathrm{cross}}\), while a norm-matched orthogonal intervention removes at most 20%. This is supporting evidence that the matched channel is read, rather than a decorative sidecar.

The key result is the difference-in-differences, not a high final accuracy. If every structured target improves both skills equally, the result is regularization. If the correct target only hurts less than the wrong target, it is incompatibility, not positive transfer. If teacher-specific logits or rationales vary between arms, the experiment is invalid.

### Fatal failure modes

- **Manufactured crossing:** one specialist is merely worse overall, or its specialty sits at a different ceiling. The pre-screen and mean-accuracy match are mandatory.
- **Trace-statistic leakage:** teacher identity remains decodable from magnitude, effective rank, layerwise smoothness, sequence length, or loss scale. Match these statistics and report an identity classifier before looking at outcomes.
- **Domain-ID shortcut:** the oracle arm tells the student whether an example is “depth” or “robustness.” Every arm already receives the same task text and sampling mixture; evaluate within each domain, not only pooled.
- **Output leakage:** teacher answers, rationales, logits, stopping length, or correctness gates contaminate the trace path. Cache hidden targets from fixed forced prefixes and audit files for forbidden channels.
- **Negative-transfer asymmetry:** the wrong teacher may be actively incompatible. Require correct-teacher improvement over SFT and matched regularizers, not merely correct \(>\) wrong.
- **Student-capacity interference:** a 0.6B student may be unable to retain both skills. Include single-skill ceiling runs; an oracle-select failure below those ceilings is not evidence against all transfer objects.
- **Batch shortcut:** \(R\) may encode length, template, or answer-format partitions. Use a fixed unseen anchor bank, length/format adversaries, and repeated co-batching.
- **Hyperparameter favoritism:** tuning the correct arm more heavily can create the effect. Freeze one coefficient-selection rule and count all pilot runs.
- **Gauge/clock mismatch:** a result may depend on a hand-tuned layer map. Freeze normalized-depth or relational-arc-length mapping before outcome evaluation and report sensitivity.

Planning estimate: two specialist LoRAs, trace extraction, eight arms, five seeds, and evaluation require roughly **120–200 RTX 5090 GPU-hours** and **12–18 calendar days**, assuming 512-token sequences and full 0.6B student fine-tuning. This is an estimate, not a measured throughput claim; one smoke run must calibrate it. The 5090 has 32 GB GDDR7 according to [NVIDIA's specification](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/).

## 3. The simpler, cheaper separation: a secret-program channel

Before spending on Qwen, run a keyed finite-state-transducer experiment:

- Generate eight random 12-state automata. Each of four symbols applies an independently sampled state permutation, creating about 115 bits of transition-table entropy.
- Train an identical 20M-parameter, 12-layer transformer teacher to at least 99% final-state accuracy for each key.
- Give a 5M-parameter, 6-layer student only 64 labeled calibration sequences covering 12 of 48 state-symbol edges but all 12 output-state labels, so readout calibration is not the bottleneck. Give it \(R\)-targets on 2,048 unlabeled anchor sequences that cover all edges.
- Test on 4,000 sequences forced to traverse the 36 unlabelled edges. Chance final-state accuracy is \(1/12=8.3\%\).
- Compare correct-key trace, wrong-key trace, Haar/spectrum-matched trace, static \(G\), generic smoothness, and no auxiliary loss over eight keys and three seeds.

The key is sampled and hash-committed after the control generator is frozen. A generic regularizer has no access to which arbitrary transition table is correct. Require correct-key \(R\) to beat every control by at least 20 accuracy points with a key-cluster bootstrap 95% lower bound above 10 points, and require a frozen probe on student states to recover at least 70% of the 36 withheld transitions versus 8.3% chance. If it does, teacher-specific information crossed the channel. If matched random performs equally, \(R\) is a prior. If correct \(R\) lowers its own loss but does not recover withheld transitions, it is a sidecar.

Estimated cost is **10–20 GPU-hours and 3–5 calendar days**, including eight teachers, six student conditions, three seeds, and debugging. This achieves a cleaner transfer-versus-regularization separation than the full language experiment, but only in a controlled world. It licenses the real trial; it cannot establish natural-language portability.

The QL5 **A-only witness** is cheaper still and should remain a preflight: near-zero \(A\)-loss with a frozen wrong head proves practical nonsufficiency and catches sidecars. It cannot achieve the same separation, because both information transfer and regularization can coexist with an exploitable null space.

## 4. Can early warm-start then removal carry the claim?

No. Equal final performance after removal supports an optimization explanation. Persistent separation does not rescue the claim: early regularization can select a basin, prevent irreversible collapse, change margins, or alter the data curriculum long after the loss disappears.

Use timing as a factorial diagnostic—correct versus wrong/random target, applied during 0–10%, 10–50%, 50–100%, or 0–100% of updates. Transfer predicts teacher-specific crossed effects that can arise after the initial basin is established and should scale with informative exposure. Pure warm-start regularization predicts concentration in the first window and weak dependence on teacher identity. Timing is useful mechanism evidence only when nested inside the identity experiment.

## NARRATIVE ATTACK

“The authors trained two specialists, then used the specialist matching the task. Of course a domain-matched hidden loss wins; it may encode style, difficulty, or layer scale. Their persistent warm-start effect is path dependence. Nothing shows that the proposed matrix carries a computation.”

The real-task design weakens this attack through isogenic teachers, crossed differences, structure matching, no output channel, and causal use. The secret-key automaton test breaks its strongest form: a teacher-invariant stabilizer cannot select an arbitrary, post-committed transition table. Until both tests pass, the adversary is not won over.

## MISSION TEST

The democratization mission requires scarce competence to move through a reusable artifact, not a bespoke regularizer requiring the original teacher online. Pass only if the same cached, compact trace selectively transfers held-out teacher knowledge, survives compression and re-batching, and costs materially less than obtaining equivalent labels or rationales. A same-family accuracy bump is useful engineering, not the moonshot.

## NEXT DIRECTIONS

1. **Key-capacity hypothesis:** the number of recoverable automaton transition bits should grow with the effective rank of skew-\(R\), while matched symmetric strain saturates. Measure a transfer-rate curve in recovered bits per stored trace byte.
2. **Late-injection hypothesis:** if \(R\) carries content rather than only initializing a basin, correct-key trace introduced after 50% of training should still produce selective recovery; generic smoothness should not.
3. **Trace-splicing hypothesis:** composing low-rank \(R\) modules from two automata should transfer the union of disjoint symbol operators without a jointly trained teacher. This is the first cheap test of “program” rather than “signal.”
4. **Teacher-dose hypothesis:** across deliberately degraded specialists, the student differential should be monotone in the teacher's held-out competence differential after trace statistics are matched.
5. **Active-anchor hypothesis:** anchors selected to maximize disagreement between correct and wrong teacher \(R\) should transmit more withheld transition bits per example than random anchors, making the method more democratic if selection is label-free.
