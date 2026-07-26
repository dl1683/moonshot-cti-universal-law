CM-CKS is correct and mandatory. The present Stage B/C cannot identify transfer of withheld transition identity.

My decision:

- Keep Stage A unchanged and let R7 finish.
- Replace the active Stage B/C admission logic with CM-CKS.
- Rewrite Stage B as a paired-key development/candidate screen.
- Rewrite Stage C as a sealed paired-key identity confirmation.
- Retain the independently random wrong-key arm only as a non-adjudicating development diagnostic.
- Do not launch the existing Stage B/C code.
- If desired later, move the old aggregate six-arm comparison to a separate post-CM utility gate. It must not be allowed to rescue a CM-CKS failure.

No repository files or running outputs were modified.

## Why QL10 is right

Calibration includes every length-one transition of \(q\), revealing its full permutation ([automaton.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:282>)). The withheld inputs then consist of \(q^p x q^r\), so every score depends on both the known \(q\) and one unknown edge ([automaton.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:321>)).

Current Stage C takes the artifact from the next independently random key ([stage_c.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:382>)). That artifact normally disagrees on:

- all 12 calibration-revealed \(q\) edges;
- all 36 withheld edges.

Therefore correct-artifact superiority can reflect less conflict with calibration, not transmission of the unknown transitions. The current aggregate accuracy statistic cannot separate those explanations.

QL10’s minimal intervention—keep \(q\) fixed and transpose two outputs in one withheld permutation—is exactly the right causal contrast.

## Modifications CM-CKS still needs

QL10 is directionally sufficient, but its current success rule is not.

### 1. Require a genuine crossover, not merely \(D_e>0\)

Its difference-in-differences can be positive even if both students prefer the same label or both are wrong.

For each altered edge require, separately:

\[
m_A=z_{S(Z_A)}(y_A)-z_{S(Z_A)}(y_B)>0
\]

and

\[
m_B=z_{S(Z_B)}(y_B)-z_{S(Z_B)}(y_A)>0.
\]

Then the signed effect is \(D=m_A+m_B\). Using logits is numerically simpler and exactly equivalent to the proposed log-probability odds.

A PASS must require:

- positive pair-level \(D\);
- positive A-side and B-side margins independently;
- absolute recovery/top-1 floors, so tiny relative shifts in two bad models cannot pass;
- the same bidirectional crossover on affected wrapped sequences.

### 2. Make localization conjunctive

The two changed direct edges are primary. Also require:

- affected wrapped sequences switch toward their corresponding key;
- the other 34 direct edges remain stable;
- wrapped inputs not traversing either changed edge remain stable;
- unchanged-edge top-1 flip rate and predictive-distribution drift stay below frozen ceilings.

Use total-variation distance or Jensen–Shannon divergence for unchanged-output drift. Freeze the ceilings using development-pair and deterministic-replay measurements before sealed generation.

### 3. Treat pairs—not seeds, edges, or examples—as scientific replicates

Student seeds are technical replication. Two changed edges and hundreds of wrapped examples within one pair are correlated measurements, not additional \(n\).

## Is 16 pairs enough?

Sixteen is enough for a budget-constrained kill gate, but QL10’s proposed `14/16` rule is unnecessarily underpowered.

| Rule | Null one-sided probability | Power if pair sign probability is 0.80 |
|---|---:|---:|
| ≥14/16 positive | 0.0021 | 0.352 |
| ≥12/16 positive | 0.0384 | 0.798 |
| ≥17/24 positive | 0.0320 | 0.911 |

My recommendation is:

- Use 16 sealed pairs for this first admission test.
- Require at least 12/16 positive pair-level signed effects, not 14/16.
- Also require a pair-clustered confidence lower bound above a precommitted practical-effect floor—not merely above zero.
- Require the component-wise crossover and unchanged-edge stability gates above.
- Run a prospective power calculation from public development-pair variance before generating sealed pairs.
- If that calculation gives less than 80% power at the declared minimum meaningful effect, increase to 24 pairs or do not run. Do not silently shrink below 16.

At the current observed teacher pace, even 16 pairs means 32 teachers and may consume most of the reserve. The post-Stage-A timing audit must include teacher training, artifact extraction, and all installer runs. If 16 cannot fit, the correct response is more budget or termination—not an eight-pair underpowered confirmation.

## Revised stage architecture

### Stage A — capacity and extraction qualification

Keep the running R7 experiment. Its outcome cannot repair or invalidate QL10’s identification argument, but it determines whether the teacher/student substrate and cost prerequisite pass.

### Stage B-P — public paired development screen

Use four public deterministic pairs, balanced across the four calibration operators.

For each pair:

- freeze \(q\);
- choose withheld \(x\) and states \(u,v\) from a committed balanced schedule plus hash;
- train \(T_A,T_B\) from identical initialization and token stream;
- install raw-A/raw-B and observable-A/observable-B into paired students;
- use two student seeds;
- calculate direct crossover, wrapped crossover, and unchanged drift.

Select raw versus observable only among candidates that pass the identity/localization conditions. Select by the larger standardized pair-level signed effect; break a stored-precision tie in favor of observable.

Run no-auxiliary, smoothness, static-\(G\), Haar, and independently random wrong-key on a small public diagnostic subset. They do not determine the CM identity verdict.

### Stage C-I — sealed identity confirmation

Use 16 sealed pairs, three paired student seeds, and only the Stage B winner:

\[
16\text{ pairs}\times2\text{ artifact assignments}\times3\text{ seeds}=96
\]

installer runs.

The Stage C PASS is the intersection of:

- direct-edge bidirectional crossover;
- affected-wrapped-sequence crossover;
- pair-level exact sign test;
- practical-effect lower confidence bound;
- unchanged direct-edge stability;
- unaffected wrapped-sequence stability;
- all capacity, hash, process-boundary, and completeness checks.

A CM failure ends the teacher-information-transfer direction. Old controls cannot override it.

### Later Stage D-U — optional aggregate utility

Only after CM passes, the old no-auxiliary/static-\(G\)/smoothness/Haar/random-wrong comparison may answer a different question:

> Does the information-bearing artifact improve aggregate task performance relative to useful regularizers and baselines?

That is useful but downstream of identity.

## Exact source changes

### New modules

- `src/cti_geometry_admission_counterfactual.py`

  Own pair-plan derivation, partner construction, canonical pair manifests, balanced \(q/x\) scheduling, invariant checks, changed-edge provenance, dual labels, and pair hashes. Keep all of this out of the generic automaton module.

- `src/cti_geometry_admission_artifacts.py`

  Centralize artifact extraction, canonical serialization, hashes, schema validation, and pair-side provenance. Stage B/C currently duplicate extraction/control construction.

- `src/cti_geometry_admission_evaluator.py`

  Load completed student checkpoints and produce sealed direct/wrapped logits. The installer must never receive sealed labels or compute outcome metrics.

### Existing modules

- [automaton.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:39>)

  Leave generic key generation and simulation here. Do not add CM orchestration. The new counterfactual module should import its constants and simulation primitives.

- [installer.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:286>)

  Remove `withheld_examples` and `direct_probes` from `train_installer_run`. Remove intermediate centroid probing at lines 386–405 and final withheld evaluation. Return only training-safe losses, timing, checkpoint, and identity hashes.

  Replace `correct/wrong` artifact semantics with a blinded artifact reference. A and B must use the same candidate coefficient; separately calibrated “wrong” coefficients would introduce another treatment difference.

- [stage_b.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:124>)

  Replace the two-unrelated-key 18-run screen with Stage B-P. The old controls remain diagnostic-only. Write pair, calibration, artifact, run, and outcome manifests separately.

- [stage_c.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:245>)

  Replace the eight-key/six-arm matrix with 16 sealed pairs and blinded A/B artifacts. Split preparation, installation, and adjudication into separate process phases. Include pair-plan and analysis hashes in the precommit.

  The current implementation also does not perform the commitment-order sorting promised by the R4 specification; the CM redesign should use canonical pair IDs rather than implicit generation indices.

- [statistics.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_statistics.py:57>)

  Add or replace the active Stage C functions with:

  - `counterfactual_edge_effects`
  - `counterfactual_pair_scores`
  - `exact_pair_sign_test`
  - `pair_cluster_effect_ci`
  - `unchanged_edge_stability`
  - `counterfactual_stage_c_verdict`

  Preserve pair clustering. Do not count seeds or edges as independent observations.

- [trainer.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:152>)

  Add deterministic-execution configuration and a same-key replay test. Completed runs may only be reused when the complete run identity matches; the current name/status-only skip at lines 164–169 can reuse stale outputs after a key or config change.

  Required identity:

  ```text
  stage, pair_id, side, key_commitment, architecture, seed,
  config_hash, data_hash, artifact_hash, code_hash, environment_hash
  ```

  If same-key teacher replay is not identical—or bounded tightly enough under a frozen tolerance—the sealed design needs teacher-level technical replication and must be recosted.

- [verify.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_verify.py:80>)

  Replace filename/source heuristics with manifest-bound verification. It must prove:

  - pair seeds and partner derivations match commitments;
  - \(q\) and calibration hashes are identical;
  - exactly two entries of exactly one withheld permutation differ;
  - artifact and run provenance paths and hashes match;
  - installer input manifests contain no sealed labels;
  - no direct/withheld evaluation occurred before fixed final checkpoints;
  - blinded assignment and run order match the commitment;
  - raw logits reproduce every statistic and verdict.

  The present verifier misses the intermediate direct-probe disclosure because it only searches for `evaluate_withheld` inside the training loop.

- Documentation

  Create a locked CM-CKS preregistration and update `STATUS.md` so existing Stage B/C are marked blocked/superseded—not “DONE.” Update the R4/R5 design surface through a successor document rather than silently editing the historical record.

## Original wrong-key arm

Keep it, but only as a diagnostic.

It answers whether a completely incompatible artifact causes negative transfer and lets you quantify how much the old effect was calibration conflict. It does not answer withheld identity. A large `correct > random-wrong` effect alongside a small or absent CM effect would confirm QL10’s confound.

It should not consume sealed primary multiplicity or be part of the CM PASS rule.

## Timing decision

Redesign now while Stage A runs. Do not wait for Stage A results to settle the causal design; Stage A cannot make the current wrong-key contrast valid.

Execution sequence:

1. Finish and adjudicate R7 Stage A.
2. In parallel, implement and CPU-test CM generation, process isolation, statistics, and verification.
3. After Stage A, perform deterministic replay and measured cost/power gates.
4. Freeze code, thresholds, pair count, pair schedule, and hashes.
5. Only then generate sealed pairs.

## What a CM-CKS PASS would prove

A clean PASS would establish:

> Under matched calibration and paired training conditions, changing only a teacher-derived relational artifact causes a localized, correctly signed change in student behavior corresponding to post-committed withheld transition identity.

That is genuine causal evidence that teacher-specific information crossed the artifact-mediated training channel.

It would not prove:

- compactness;
- an algorithm rather than table transmission;
- uniqueness or superiority of geometry as the encoding;
- cross-substrate portability;
- inference-time artifact use;
- a frozen general installer;
- economic advantage over shipping the 11-byte key;
- language capability transfer;
- “download a skill.”

Score after a strong sealed CM PASS:

- Nobel: **2.5/10**, up from 2.0.
- Turing: **4.0/10**, up from 3.0.

The update is real but deliberately modest. Reaching Turing 6+ requires bitrate competition against direct key/label coding, structured-rule rather than random-table transfer, a frozen amortized installer, and cross-substrate success. Nobel-scale relevance begins only when the phenomenon becomes a general law or mechanism beyond this synthetic channel.

