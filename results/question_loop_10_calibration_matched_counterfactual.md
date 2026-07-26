# Question Loop 10: Calibration-Matched Counterfactual Identity

## Verdict

The R7 fix repairs the capacity prerequisite. It does not repair the transfer identification problem.

The single most dangerous question is:

> **Would the claimed correct-artifact advantage survive if the competing teacher artifact agreed exactly on everything revealed by calibration and differed only in two outputs of one withheld permutation?**

The present wrong-key control cannot answer this. Calibration reveals one complete permutation q, because it includes all 12 length-one examples for that operator ([automaton](../src/cti_geometry_admission_automaton.py#L285)). Stage C then uses an artifact from an independently random next key as the wrong-key control ([stage C](../src/cti_geometry_admission_stage_c.py#L384)). That artifact normally disagrees not only on the 36 withheld edges, but also on the 12 already-labeled q edges.

Consequently, `correct > wrong` can be produced by **calibration compatibility**: the correct artifact cooperates with the supervised loss on q, while the wrong artifact fights it. Because every withheld sequence has the form q^p x q^r, better learning of the known q can also improve the aggregate withheld-sequence score without installing the unknown edge x.

If the advantage disappears under a calibration-matched counterfactual, the current direction is killed at its narrowest claim. There would be no evidence that the artifact specifies withheld transition identity.

## 1. Steelman after R7

R7 closes the old positional-support failure:

- capacity training now samples lengths 1-32 ([trainer](../src/cti_geometry_admission_trainer.py#L188));
- the ceiling is fixed at 7,000 steps ([trainer](../src/cti_geometry_admission_trainer.py#L48));
- the legacy `dev_extrapolation` split is explicitly described as covered-long, not extrapolation ([trainer](../src/cti_geometry_admission_trainer.py#L114));
- anchors now span lengths 8-32 ([automaton](../src/cti_geometry_admission_automaton.py#L193)), covering the positional support used by the longest restricted-supervision probes.

The strongest defensible prospective claim is therefore:

> Given a teacher competent over the full operational support, a frozen teacher-derived relational artifact may cause a smaller Transformer, trained on one fully revealed operator, to perform better on sealed sequences containing three unrevealed operators than five declared controls.

If Stage C passes its frozen thresholds, that would be nontrivial evidence that the artifact is useful teacher-conditioned supervision. The pass requires a minimum 0.20 mean advantage over each declared control, a bootstrap lower bound above 0.10, an exact sign-flip p <= 0.05, and at least 70% centroid-probe accuracy on every key ([statistics](../src/cti_geometry_admission_statistics.py#L215)).

That is the steelman. It is still not evidence for a compact capability file, architecture-independent installation, an algorithm rather than a table, or mission-level economics.

## 2. Attack QL9 before accepting any of it

QL9's nominal 290,000x capacity ratio is not a measurement of transmitted information. Float count is an upper bound on channel capacity, not mutual information. Highly redundant matrices can occupy megabytes while carrying only a few effective bits. Therefore artifact size alone does **not** show memorization and cannot kill transfer.

The 86.5-bit conditional key entropy does survive re-derivation:

```text
H(K | C) = 3 log2(12!) = 86.51 bits.
```

The length-one calibration rows reveal q exactly and the other three permutations are independently random. But this is a lower bound on the irreducible description of this random task, not proof that the artifact uses those bits efficiently.

QL9 also overcalled its proposed bundle "decisive." A paired-key flip can establish counterfactual teacher identity. It cannot by itself establish:

- compactness;
- algorithm transfer rather than table transmission;
- cross-substrate portability;
- causal reliance at inference time;
- economic advantage over sending the key directly.

The proposed late-injection and subspace-lesion additions are separate questions. A "geometry-bearing subspace" is circular unless frozen without withheld outcomes, and timing interventions add failure modes unrelated to identity. The first experiment should isolate one causal variable cleanly.

What remains valid after that attack is more specific and more dangerous than QL9's headline: **the current wrong-key arm is not calibration-matched, so the present protocol can mistake compatibility on known facts for transfer of unknown facts.**

## 3. The decisive control: Calibration-Matched Counterfactual Key Swap

Name: **CM-CKS (Calibration-Matched Counterfactual Key Swap).**

For each precommitted base key K_A:

1. Select the calibrated operator q, a withheld operator x, and two source states u,v by a committed hash.
2. Construct K_B by swapping only pi_x(u) and pi_x(v). Keep q, the other two withheld permutations, all tokens, anchors, and calibration examples bit-identical.
3. Train teachers T_A,T_B from the same initialization, token stream, optimizer schedule, and deterministic execution. The key intervention is the only intended difference.
4. Extract Z_A,Z_B, then install each into students with identical initial weights, identical calibration batches, identical bank order, and the same frozen coefficient.
5. Evaluate both installed students against both counterfactual labelings. Do not expose direct or wrapped withheld results until every run and hash is complete.

Use the actual classifier logits on the two altered direct edges as the primary endpoint, not the centroid probe. For altered edge e, let y_A(e) and y_B(e) be its two counterfactual outputs and define:

```text
D_e = [log p_S(Z_A)(y_A | e) - log p_S(Z_A)(y_B | e)]
    - [log p_S(Z_B)(y_A | e) - log p_S(Z_B)(y_B | e)].
```

The artifact-specific prediction is D_e > 0 for both changed edges. At the same time:

- predictions on the other 34 withheld direct edges must remain stable;
- wrapped examples traversing the changed edges must switch toward their corresponding key;
- wrapped examples not traversing the changed edges must remain stable;
- same-key technical replicates must bound numerical/training nondeterminism.

Precommit at least 16 sealed pairs. Require positive pair-level signed effects for at least 14/16 pairs, a nonzero key-cluster lower confidence bound, and a strict unchanged-edge drift ceiling. Freeze exact effect and drift floors from development pairs before generating sealed pairs.

| Result | Meaning |
|---|---|
| No signed switch | No evidence that the artifact specifies withheld identity; kill the transfer claim. |
| Global degradation only | Optimization incompatibility, not targeted information transfer. |
| Signed switch plus large unchanged-edge drift | High-bandwidth key perturbation, still not selective installation. |
| Signed switch confined to altered edges | Counterfactual evidence that teacher-specific withheld information crosses the artifact channel. |

Even the fourth outcome establishes only an information-bearing supervision channel. It does not establish a compact or portable program.

## 4. Feed the work loop

Do **not** stop the repaired R7 Stage A capacity run. Its result remains a valid prerequisite and useful cost measurement.

Do **not** launch the current Stage B/C protocol unchanged after Stage A. Insert CM-CKS as a blocking identity gate:

1. Preserve Stage A as capacity/extraction qualification only.
2. Add paired-key generation that holds the calibrated permutation exactly fixed and applies one committed withheld transposition.
3. Replace the independently random wrong-key arm with the calibration-matched partner artifact for the primary identity comparison. The old wrong-key arm may remain only as a diagnostic stress control.
4. Add edge-level provenance to every wrapped example: whether it traverses either altered edge and the counterfactual labels under both keys.
5. Record final classifier logits for direct probes and disaggregate changed versus unchanged withheld edges.
6. Remove intermediate withheld centroid-probe disclosure. The installer currently computes and logs it every 250 steps ([installer](../src/cti_geometry_admission_installer.py#L387)); sealed confirmation should reveal it once, after the fixed final checkpoint.
7. Require deterministic paired execution or same-key technical replicates.
8. Precommit the pair generator, signed statistic, unchanged-edge drift bound, pair count, exclusions, and all hashes before any sealed artifact is extracted.

Only after CM-CKS passes should the work loop spend on bitrate, timing, lesion, GRU, or language-scale experiments.

## 5. Narrative attack

### Strongest "that's obvious" dismissal

> Of course a multi-megabyte trace from a 19.5M-parameter teacher trained on the secret table can improve a student. The trace is an indirect label channel. You have shown that supervision can contain supervised information.

The current protocol cannot defeat this dismissal because it never makes a minimal change to the secret and asks whether the installed behavior changes in the same direction.

### Strongest "that's trivial" dismissal

> This is an 11-byte random key wrapped in thousands of gradient steps and megabytes of matrices. Send the permutation table and use a tiny interpreter. Calling the wrapper a capability file does not make the payload intelligence.

This dismissal remains correct even if CM-CKS passes.

### Mission test

The current synthetic experiment does not yet make intelligence cheap or accessible. It spends teacher training, artifact extraction, megabytes of storage, and 5,000 installation steps to communicate a task whose irreducible unknown is about 86.5 bits.

Its mission value is only as a cheap falsification gate: can the proposed geometry channel transmit causally specific information at all? If it cannot, stop. If it can, immediately move to an experiment where a reusable algorithm, not a random table, is available to transfer.

### What the result must be for the narrative to be unkillable

No result on this random-permutation automaton can make the broad narrative unkillable. The task contains no shorter latent algorithm than its random key.

The eventual result would need all of the following:

- counterfactual signed specificity under calibration-matched keys;
- a frozen codec and frozen installer trained only on development tasks;
- a payload near the task's measured rate-distortion frontier, compared with direct labels, logits, table coding, and ordinary distillation;
- installation into unseen Transformer and recurrent substrates with no pair-specific tuning;
- generalization to withheld states, operators, and longer compositions on a structured task whose rule is shorter than its table;
- lower total cost and latency than fine-tuning, teacher querying, and direct program distribution;
- independent replication.

The defensible headline before that evidence is: **"A teacher trace may be a causally specific supervision channel."** It is not yet "download a skill."

## 6. Next directions

### H1: Calibration-compatibility confound

**Hypothesis:** The present correct-versus-wrong advantage is substantially explained by agreement on the revealed operator q.

**Test:** Compare independently random wrong-key artifacts with CM-CKS partner artifacts while keeping all student runs paired.

**Prediction:** The large wrong-key margin shrinks sharply when q is held fixed. If the correct artifact carries withheld identity, a smaller but signed edge-specific margin remains.

### H2: Local counterfactual geometry

**Hypothesis:** Switching one withheld transposition causes a reproducible, localized change in the artifact that installs the corresponding two edge outputs.

**Test:** CM-CKS across at least 16 sealed pairs, with changed-edge logit contrasts and unchanged-edge stability as joint endpoints.

**Prediction:** Observable geometry will be more selective than raw R, even if raw R has greater aggregate accuracy, because the observable bottleneck suppresses key-irrelevant teacher variation.

### H3: Information, not float count, controls success

**Hypothesis:** Withheld performance follows recoverable I(K;Z | C), not serialized artifact bytes or raw geometry loss.

**Test:** Train one codec on development keys, freeze it, quantize artifacts across 32-1,024 bits, and decode on sealed keys. Compare against optimal permutation coding and direct edge labels at the same bit budgets.

**Prediction:** If geometry is only a bloated table codec, its frontier is strictly dominated by direct key coding. If it exposes reusable structure on a structured task, it can dominate raw label transmission at matched distortion.

### H4: Structured-rule phase transition

**Hypothesis:** Geometry becomes program-like only when the automaton family is generated by a compact latent rule rather than independent random permutations.

**Test:** Use group actions or parameterized state machines with 128-256 states. Withhold entire states, operators, and composition depths while capping the artifact below full-table entropy.

**Prediction:** Random tables fail beyond the transmitted bitrate; structured families extrapolate from a compact artifact.

### H5: Amortized cross-substrate installation

**Hypothesis:** A genuine capability file can be consumed by one frozen installer across unseen keys and architectures.

**Test:** Meta-train the installer only on development automata, freeze it, and measure one-shot or few-step installation into unseen Transformer and GRU students.

**Prediction:** If 5,000 per-key gradient steps remain necessary, the artifact is bespoke training supervision rather than a portable file.

## Bottom line

R7 makes a future negative or positive transfer result interpretable with respect to teacher capacity. It does not make the current controls identify withheld information.

The next gate is not "does correct beat a totally different key?" It is:

> **When calibration is identical and only two secret outputs change, does changing only the artifact change only those outputs in the correct signed direction?**

If no, kill the direction. If yes, admit only the narrow information-channel claim and proceed to the separate compression and program-transfer gates.
