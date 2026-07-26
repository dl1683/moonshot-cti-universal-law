Your scope objection is correct. My Round 1 design over-hardened a first kill gate.

The budget objection is also correct. But two proposed shortcuts are scientifically invalid:

- You cannot manufacture \(Z_B\) by transposing the key after extracting \(Z_A\). A partner artifact requires a separately trained \(T_B\).
- Multiple partner transpositions sharing one base teacher are correlated contrasts, not independent pairs. They are useful in development but cannot inflate sealed \(n\).

## Final R8 Round 2 ruling

Adopt a minimal CM-CKS implementation:

- Zero new Python modules.
- Modify five source files.
- Leave extraction and trainer unchanged.
- Use Stage A’s teacher as the development base.
- Use multiple partners per base only in development.
- Run eight independent sealed base/partner pairs.
- Use two paired student seeds.
- Require 7/8 pair successes.
- Accept high false-negative risk because this is an economic kill gate, not a definitive publication-scale confirmation.

Do not run the current Stage B/C unchanged.

## 1. Scope: adopt the minimal version

The blinded evaluator, artifact module, and complete process isolation can be deferred until CM passes.

The minimum touched surface is:

1. [automaton.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:39>)
2. [stage_b.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:185>)
3. [stage_c.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:245>)
4. [installer.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:286>)
5. [statistics.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_statistics.py:57>)

Leave these unchanged:

- extraction;
- geometry;
- models;
- trainer;
- independent verifier, temporarily.

Use a fresh `results/geometry_admission/stage_c_cm/` directory and pair-specific run names to avoid stale completion reuse without changing the trainer.

### `automaton.py`

Add:

```python
paired_key_from_transposition(
    base_key_json,
    calibrated_op,
    withheld_op,
    source_u,
    source_v,
) -> tuple[partner_key_json, pair_metadata]
```

It must assert:

- `withheld_op != calibrated_op`;
- `u != v`;
- both keys contain four valid permutations;
- the calibrated permutation is identical;
- exactly two table entries differ;
- both differing entries belong to the declared withheld operator;
- calibration-set hashes are identical.

No broader counterfactual module is needed.

### `stage_b.py`

Replace the current unrelated-two-key candidate screen.

Minimal Stage B:

- Base: reuse the completed Stage A development teacher.
- Generate two deterministic transposition partners of that same base key.
- Train only the two partner teachers.
- Extract raw and observable artifacts using existing extraction functions.
- Use one paired student seed.
- Run raw-A/raw-B/observable-A/observable-B for each partner.

That is eight development installer runs.

Select raw or observable using changed-edge crossover subject to unchanged-edge stability. Multiple partners share the development base teacher; that is acceptable because Stage B is selection, not confirmation.

The old random-wrong/Haar/static-\(G\)/smoothness arms can remain available but should not be run unless the CM effect appears.

### `stage_c.py`

Replace the eight independent keys and cyclic wrong-key derangement with:

- eight independent sealed base seeds;
- one hash-derived transposition partner per base;
- eight base teachers plus eight partner teachers;
- winner artifact A versus winner artifact B;
- two paired student seeds.

Run matrix:

\[
8\times2\times2=32
\]

student installations.

Use the same winner arm and the same frozen coefficient for A and B. Do not route B through `raw_wrong` or `obs_wrong`, because those currently have separately calibrated coefficients. Artifact identity must be the only student-treatment difference.

Stage C can load the final checkpoints and compute native classifier logits itself or call a small reusable evaluation helper added to the installer. No new evaluator module is needed.

### `statistics.py`

Add only:

- pair-level direct crossover;
- pair-level affected-wrapped crossover;
- unchanged-edge drift;
- exact pair sign test;
- CM PASS/FAIL/VOID verdict.

No bootstrap should be presented as reliable uncertainty estimation with only eight pairs. Use exact signs plus frozen absolute effect floors.

## 2. Budget: use eight independent pairs

The one-teacher proposal needs clarification.

### Invalid version

This is invalid:

1. Train \(T_A\).
2. Extract \(Z_A\).
3. Modify the key.
4. Algebraically “transpose” \(Z_A\) to obtain \(Z_B\).

There is no known equivariant map from a key-table transposition to the raw or observable artifact. Such a construction would test a hand-designed artifact editor, not teacher-conditioned geometry.

### Valid version

This is valid:

1. Train \(T_A\) on \(K_A\).
2. Train \(T_B\) on transposed \(K_B\).
3. Extract \(Z_A,Z_B\).
4. Cross the artifacts through otherwise paired student runs.

That still requires two teachers per independent pair.

### Reusing one base teacher

Reuse is acceptable in Stage B:

```text
one Stage A base teacher
+ two partner teachers
= two development contrasts
```

It is not acceptable to count those as two independent sealed pairs. Their effects share \(T_A\), \(Z_A\), calibration, and base key. The scientific cluster remains the base key.

Therefore Stage C uses eight independent bases, each with one partner.

### Revised compute

New teacher runs:

- 2 Stage B partner teachers;
- 1 same-key replay teacher;
- 16 sealed teachers.

Total: **19 new teachers**.

At 70–90 minutes each:

- approximately 22–29 teacher GPU-hours.

Student runs:

- 8 Stage B candidate runs;
- 2 same-key noise-control runs;
- 32 sealed runs.

Total: **42 installer runs**.

This may fit 45 hours, but only if the first complete CM installer timing supports it. Before sealed generation, require:

```text
projected teachers
+ extraction
+ 42 installer runs
+ 20% retry reserve
<= 45 GPU-hours
```

If it does not fit, stop. Do not reduce below eight independent bases.

## 3. Eight-pair statistics

“12/16-equivalent” would mean 6/8 positive. That is not enough:

\[
P(X\ge6\mid p=0.5)=37/256=0.1445.
\]

The smallest conventional exact-sign threshold is:

\[
P(X\ge7\mid p=0.5)=9/256=0.0352.
\]

Therefore require **7/8 successful pairs**.

This has only about:

- 50% power when true pair success probability is 0.80;
- 81% power when true pair success probability is 0.90.

That is acceptable only because the project is looking for a strong, reliable effect worth scaling. We explicitly accept false negatives.

Define pair success conjunctively:

- A-artifact student prefers \(y_A\) over \(y_B\);
- B-artifact student prefers \(y_B\) over \(y_A\);
- direct signed effect exceeds a frozen practical floor;
- affected wrapped sequences switch correctly;
- unchanged direct and wrapped outputs stay within frozen drift ceilings.

Average the two student seeds and two changed edges inside each pair. They do not increase \(n\).

Verdicts:

- **PASS:** at least 7/8 pair successes and all aggregate effect/stability floors pass.
- **FAIL:** six or fewer successes, or any practical/localization gate fails. This is an operational kill, not proof that the population effect is exactly zero.
- **VOID:** commitment, pair-construction, teacher-capacity, run-completeness, or data-integrity failure.

## 4. Intermediate probes: keep them, but actually seal them

I accept this simplification. Because the probes do not feed gradients, their presence does not destroy the causal contrast.

The current behavior is still unsuitable for sealed confirmation: probe results are printed, written into the ordinary training log, and copied into `summary.json` every 250 steps ([installer.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:386>)).

Minimal change:

- add `seal_probes: bool`;
- continue computing the diagnostic centroid probe;
- under sealed mode, do not print it;
- write it only to `sealed_probe_log.jsonl`;
- exclude probe fields from ordinary `training_log.jsonl` and `summary.json`;
- hash-chain or final-hash the sealed file;
- do not open any sealed probe log until all 32 student summaries exist.

This is an honor-system embargo, not cryptographic isolation. That is acceptable for the internal kill gate. A PASS must be hardened before an external claim.

The primary CM endpoint remains final native classifier logits, not the centroid learning curve.

## 5. Teacher determinism: exact equality is unnecessary

I withdraw the exact deterministic-replay requirement.

The causal treatment is the complete pathway:

\[
K\rightarrow T\rightarrow Z\rightarrow S.
\]

Teacher weights need not match exactly. All teachers must use the same initialization seed, training stream, schedule, hardware class, and capacity gates. Training nondeterminism then becomes nuisance variation.

But matching final accuracy within 0.1% is not enough. Two equally accurate teachers can have substantially different relational artifacts.

Minimal control:

- train one same-key replay of the Stage A development teacher using the same seed and protocol;
- extract the winning artifact from the original and replay teachers;
- install both into the same student initialization;
- measure final direct-logit and wrapped-output drift.

Use this replay to set a teacher-noise ceiling. The sealed CM changed-edge effect must materially exceed same-key replay drift, while unchanged-edge drift must remain near it.

This costs one teacher plus two student installations—not a full determinism infrastructure.

If same-key replay produces artifact-induced behavioral changes comparable to CM, Stage C is blocked: teacher training noise is too large for an eight-pair design.

## 6. Timing

Finalize and implement CM now. Do not wait for Stage A to finish before editing and CPU-testing the protocol.

The live log currently shows:

- step 2250;
- in-range 0.9543;
- covered-long accuracy 0.1012.

The latter remains far below the 0.990 gate, so teacher completion is not yet teacher qualification. Still, the remaining wall-time estimate is compatible with implementing the minimal CM changes in parallel.

Sequence:

1. Implement the five-file minimal patch now.
2. Let Stage A continue untouched.
3. Reuse the Stage A teacher only if its final two evaluations pass.
4. Run the two-partner Stage B screen and one same-key replay.
5. Measure real CM installer and extraction cost.
6. Freeze winner, effect floors, drift ceilings, pair derivation, seeds, code hashes, and 45-hour projection.
7. Generate the eight sealed base keys.
8. Run Stage C once.

## Bottom line

Adopted simplifications:

- no new modules;
- no process isolation;
- no trainer or extraction rewrite;
- intermediate probes retained under embargo;
- approximate teacher matching accepted;
- eight independent sealed pairs;
- one base with multiple partners allowed in development.

Rejected shortcuts:

- synthetic post-extraction artifact transposition;
- counting multiple partners of one base as independent pairs;
- treating 6/8 as confirmatory;
- using final teacher accuracy alone to bound artifact noise;
- allowing separately calibrated A/B coefficients.

This is the smallest experiment I would sign off as a legitimate CM-CKS kill gate.

