The frozen design should be an 18-run Stage B screen followed by a 144-run Transformer Stage C confirmation. Stage C here means the third stage of the Geometry Admission Test; it is unrelated to the permanently killed legacy Gate C.

One correction to the accepted [Stage A specification](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/OPEN_CAPABILITY_FILE_GEOMETRY_ADMISSION_STAGE_A_2026_07_25.md:1>): the sentence saying “Stage B has 144 confirmation runs” should read “Stage C has 144 confirmation runs.”

# Common Stage B/C contract

Let:

- anchor bank \(b\in\{0,\ldots,31\}\);
- bank size \(n=64\);
- checkpoints \(j\in\{0,\ldots,6\}\);
- transitions \(j\in\{0,\ldots,5\}\);
- observable rank \(r=8\).

The same ordered 2,048 Stage A anchors and the same 32 banks are used for every key, arm, and seed.

For student checkpoint states \(H^S_{b,j}\in\mathbb R^{64\times d_S}\), recompute the Stage A objects:

\[
X^S_{b,j}
=
\frac{\sqrt{64}\,CH^S_{b,j}}
{\|CH^S_{b,j}\|_F+10^{-12}},
\]

\[
G^S_{b,j}=X^S_{b,j}X^{S\top}_{b,j},
\]

\[
A^S_{b,j}
=
(X^S_{b,j+1}-X^S_{b,j})X^{S\top}_{b,j},
\]

\[
\lambda^S_{b,j}
=
10^{-3}\frac{\operatorname{tr}(G^S_{b,j})}{63},
\]

\[
\mathcal R^S_{b,j}
=
G_{b,j,\lambda}^{S,-1/2}
A^S_{b,j}
G_{b,j,\lambda}^{S,-1/2}.
\]

Use the student’s own \(G^S\) and ridge. Do not reuse the teacher whitening matrix.

Matrix construction, eigendecomposition, and auxiliary losses run in float32 with autocast disabled. Teacher targets and observable bases are detached constants. Gradients pass through the entire student construction, including its inverse square root; no stop-gradient whitening or learned adapter is permitted.

Numerically symmetrize \(G^S\) before `eigh`. Clamp only negative eigenvalues in \([-10^{-6},0]\) to zero. An eigenvalue below \(-10^{-6}\), nonfinite loss, or failed eigendecomposition is a numerical failure, not an invitation to modify the formula.

# Labeled calibration and withheld transitions

Stage B/C students do not receive the Stage A online supervision stream.

## Declared 12 labeled edges

Each key slot is assigned one labeled operation:

\[
q_k = k\bmod 4,
\]

with slot order corresponding to \(a,b,c,d\).

- Stage B development key 0 uses \(a\).
- Stage B development key 1 uses \(b\).
- The eight Stage C keys, sorted by published seed commitment, use
  \(a,b,c,d,a,b,c,d\).
- The Stage A coefficient-calibration key uses \(a\).

The 12 edges \((s,q_k)\), \(s=0,\ldots,11\), are the labeled edges. The other 36 state-operation edges are withheld.

## Exactly 64 labeled sequences

For lengths

\[
L\in\{1,2,4,8,16\}
\]

and every \(s_0\in\{0,\ldots,11\}\), create the sequence \(q_k^L\). This gives 60 examples.

Add:

\[
(s_0,L)\in\{(0,3),(3,5),(6,7),(9,11)\}.
\]

Every operation in all 64 sequences is \(q_k\). Therefore the set reveals exactly 12 transition edges and no others. The \(L=1\) block guarantees that all 12 output classes occur.

The full set of 64 examples is used as the task batch on every optimizer step. No new labeled examples are generated.

## Withheld evaluation set

Create exactly 4,000 examples per key.

For each withheld edge \(e=(s,x)\), where \(x\neq q_k\):

1. Allocate 111 examples, giving 3,996 total.
2. Sort the 36 edge identifiers by

\[
\operatorname{SHA256}(
\texttt{"GAT\_WITHHELD\_EXTRA\_V1"}\parallel k\parallel e
).
\]

3. Give the first four edges one additional example.

For each example, select a distinct pair

\[
(p,r)\in\{0,\ldots,15\}^2
\]

from a hash-seeded permutation of the 256 possible pairs. Set:

\[
s_0=\pi_{q_k}^{-p}(s),
\]

and construct:

\[
q_k^p\,x\,q_k^r.
\]

Each input has length 1–31 and contains exactly one withheld-operation edge; every other transition uses the labeled operation. This makes final-state accuracy directly attributable to recovery of the targeted withheld transition.

Hash inputs and labels before training. Stage C labels must not be exposed to the student runner or used for checkpoint selection.

Also store the 36 direct withheld examples:

\[
[\mathrm{STATE}_s,\mathrm{OP}_x],
\qquad x\neq q_k,
\]

for the frozen probe assay.

# 1. Candidate transfer losses

All six transitions receive equal weight. MSE here is exactly squared Frobenius distance divided by the number of matrix entries.

## Raw \(R\)

For teacher target \(\mathcal R^T_{b,j}\in\mathbb R^{64\times64}\):

\[
\boxed{
\mathcal L_{\mathrm{raw}}(b)
=
\frac1{6}
\sum_{j=0}^{5}
\frac{
\|\mathcal R^S_{b,j}-\mathcal R^T_{b,j}\|_F^2
}{64^2}
}
\]

This is full-\(R\), not skew-only. Do not add separate symmetric/skew weights, relative-error denominators, cosine loss, or Procrustes alignment.

## Observable \(R\)

Use the exact teacher basis \(U^T_{b,j}\in\mathbb R^{64\times8}\) stored in the artifact. Do not calculate a student-balanced basis.

\[
\widehat{\mathcal R}^{S,\mathrm{obs}}_{b,j}
=
U_{b,j}^{T\top}
\mathcal R^S_{b,j}
U^T_{b,j}.
\]

Then:

\[
\boxed{
\mathcal L_{\mathrm{obs}}(b)
=
\frac1{6}
\sum_{j=0}^{5}
\frac{
\|
\widehat{\mathcal R}^{S,\mathrm{obs}}_{b,j}
-
\mathcal R^{T,\mathrm{obs}}_{b,j}
\|_F^2
}{8^2}
}
\]

The same \(U\) and target bytes are used for every seed and, conditionally, the later GRU replication.

# 2. Control arms

## No auxiliary

\[
\mathcal L_{\mathrm{none}}=0.
\]

The run receives only the 64-example task loss. It uses the same optimizer steps and task order. Its lower compute cost is recorded rather than hidden with artificial ballast.

## Generic smoothness

This is hidden-depth smoothness, not parameter-update regularization:

\[
\boxed{
\mathcal L_{\mathrm{smooth}}(b)
=
\frac1{6}
\sum_{j=0}^{5}
\frac{
\|X^S_{b,j+1}-X^S_{b,j}\|_F^2
}{64}
}
\]

Because every \(X^S\) has squared Frobenius norm approximately 64, the loss is width-independent. It receives no teacher artifact.

Do not use L2 distance from initialization or optimizer-update regularization; those test a different mechanism and have architecture-dependent scale.

## Static \(G\)

Stage A extraction must additionally serialize the teacher \(G^T_{b,j}\) for all seven checkpoints. This is a control-artifact extension, not a third candidate.

\[
\boxed{
\mathcal L_G(b)
=
\frac1{7}
\sum_{j=0}^{6}
\frac{
\|G^S_{b,j}-G^T_{b,j}\|_F^2
}{64^2}
}
\]

Use centered, globally normalized \(G\) exactly as in Stage A. No cross-layer generator is included.

## Wrong-key

For target key \(k\), train on its 64 calibration labels and evaluate on its withheld transitions, but replace the candidate artifact with one from \(\kappa(k)\).

Stage B swaps its two keys:

\[
\kappa(0)=1,\qquad \kappa(1)=0.
\]

For Stage C, sort keys lexicographically by published seed commitment and use the cyclic derangement:

\[
\kappa(k)=(k+1)\bmod 8.
\]

The wrong-key artifact uses the same ordered anchors and bank boundaries. Apply the winner’s normal loss formula against the wrong artifact:

\[
\mathcal L_{\mathrm{wrong\text{-}raw}}
=
\mathcal L_{\mathrm{raw}}(
\mathcal R^{T,\kappa(k)}
),
\]

or:

\[
\mathcal L_{\mathrm{wrong\text{-}obs}}
=
\mathcal L_{\mathrm{obs}}(
U^{T,\kappa(k)},
\mathcal R^{T,\mathrm{obs},\kappa(k)}
).
\]

No target-key teacher value may be mixed into this arm.

## Haar/spectrum-matched

This is an orthogonal similarity transform, not a fresh Gaussian matrix.

### Raw candidate

Let:

\[
e=\frac1{\sqrt{64}}\mathbf 1.
\]

Use the fixed 64-dimensional Helmert basis \(E\in\mathbb R^{64\times63}\) for the centered subspace. For each bank, draw \(Z_b\in\mathbb R^{63\times63}\) with IID standard-normal entries from a domain-separated PCG64DXSM seed. Compute:

\[
Z_b=Q_b^{c}R_b
\]

by QR and flip columns so the diagonal of \(R_b\) is positive. Extend it to anchor space:

\[
Q_b=ee^\top+E Q_b^cE^\top.
\]

Use the same \(Q_b\) for all six depths:

\[
\widetilde{\mathcal R}_{b,j}
=
Q_b\mathcal R^T_{b,j}Q_b^\top.
\]

The loss is raw MSE against \(\widetilde{\mathcal R}\).

This exactly preserves:

- eigenvalues;
- singular values;
- Frobenius norm;
- symmetric and skew Frobenius norms;
- rank;
- every pairwise inter-depth Frobenius inner product.

It destroys the correct anchor-to-anchor association.

### Observable candidate

Draw one fixed \(Q_b^{(8)}\in O(8)\) per bank with the same QR rule:

\[
\widetilde{\mathcal R}^{\mathrm{obs}}_{b,j}
=
Q_b^{(8)}
\mathcal R^{T,\mathrm{obs}}_{b,j}
Q_b^{(8)\top}.
\]

Use the same \(Q_b^{(8)}\) for all six depths. The student side remains:

\[
U_{b,j}^{T\top}\mathcal R^S_{b,j}U^T_{b,j}.
\]

Store and hash all \(Q\) matrices. There is no outcome-dependent phase randomization because using one \(Q\) across depth already preserves depth autocorrelation exactly.

# 3. Training and coefficient protocol

## Teacher preparation

Train one fully supervised Stage A teacher for each Stage B/C key using the exact Stage A teacher protocol and capacity gates.

Teacher seed:

```text
101 for every key
```

Using the same initialization and input stream across keys is intentional.

A teacher that misses the capacity gate does not yield a usable artifact.

## Student optimizer

Use the Transformer student configuration from Stage A.

```text
AdamW
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.01
peak lr = 5e-4
warmup = 250 steps
cosine decay to 10% of peak lr
steps = exactly 5000
gradient clipping = 1.0
bf16 forward/backward
fp32 optimizer states
```

There is no early stopping or best-checkpoint selection. Step 5,000 is the adjudicated checkpoint.

The batch differs from Stage A because supervision is deliberately restricted:

- task forward: all 64 labeled calibration examples;
- auxiliary forward: one 64-anchor bank.

The total loss at step \(t\) is:

\[
\boxed{
\mathcal L_t
=
\mathcal L_{\mathrm{task}}(\mathcal C_k)
+
\lambda_a\mathcal L_a(B_{b_t})
}
\]

where \(a\) is the arm.

Generate one frozen permutation of banks using:

```text
SHA256("GAT_INSTALLER_BANK_ORDER_V1")
```

with PCG64DXSM. Cycle through that order for all 5,000 steps. Every arm, key, and seed gets the identical bank sequence.

The teacher is absent during student training. The student process may read only:

- the 64 calibration examples;
- anchor tokens and bank manifest;
- the selected artifact;
- its frozen coefficient;
- the training configuration.

It must not receive the key, teacher weights, teacher hidden states, logits, or withheld labels.

## One-shot auxiliary coefficient calibration

Do not use one numerical \(\lambda\) for all losses because raw \(64^2\), observable \(8^2\), \(G\), and smoothness have different gradient scales.

Use one frozen rule that produces one coefficient per auxiliary arm.

Reference conditions:

```text
key: Stage A development key
student seed: 400
student: Transformer student
checkpoint: initialization, before optimizer step 1
task data: the Stage B/C 64-example calibration set
banks: all 32 Stage A anchor banks
target auxiliary/task gradient ratio: 1.0
```

Let \(\theta_{\mathrm{trunk}}\) contain every student parameter except the final classifier.

Compute:

\[
g_{\mathrm{task}}
=
\left\|
\nabla_{\theta_{\mathrm{trunk}}}
\mathcal L_{\mathrm{task}}
\right\|_2,
\]

and, for each auxiliary arm \(a\):

\[
g_{a,b}
=
\left\|
\nabla_{\theta_{\mathrm{trunk}}}
\mathcal L_a(B_b)
\right\|_2,
\]

\[
\widetilde g_a
=
\operatorname{median}_{b=0}^{31}g_{a,b}.
\]

Freeze:

\[
\boxed{
\lambda_a
=
\frac{g_{\mathrm{task}}}{\widetilde g_a}
}
\]

This produces a 1:1 median auxiliary-to-task trunk-gradient ratio at the reference point.

Calibrate separately for:

- raw correct;
- raw wrong;
- raw Haar;
- observable correct;
- observable wrong;
- observable Haar;
- static \(G\);
- generic smoothness.

For wrong-key coefficient calibration, use the first Stage B development-key artifact as the wrong source applied to the Stage A reference student. This occurs before any Stage B student outcome is evaluated.

Requirements:

- all gradient norms finite and nonzero;
- achieved ratios within 5% of 1.0;
- no clipping of \(\lambda\);
- no coefficient sweep;
- no per-key or per-seed recalibration;
- no online gradient balancing.

Stage C uses the exact coefficients frozen before Stage B outcomes. A later GRU confirmation reruns this same formula once on the Stage A key with a dedicated GRU reference seed; it does not tune on GRU outcomes.

# 4. Stage B candidate screen

## Development keys

Generate two public deterministic development seeds:

```text
SHA256("GAT_STAGE_B_DEV_KEY_V1|0")
SHA256("GAT_STAGE_B_DEV_KEY_V1|1")
```

Use the full 32 digest bytes as seed material for the Stage A key derivation procedure. These keys can never appear among the eight sealed keys.

Student seed:

```text
401
```

Use identical initialization for all arms within a key.

## Exact 18-run matrix

For each of two keys:

1. correct raw;
2. raw wrong-key;
3. raw Haar;
4. correct observable;
5. observable wrong-key;
6. observable Haar;
7. static \(G\);
8. generic smoothness;
9. no auxiliary.

Total:

\[
2\times9=18\text{ runs}.
\]

## Selection metric

The primary metric is final-state accuracy on the frozen 4,000-example withheld set at step 5,000.

For candidate \(m\in\{\mathrm{raw},\mathrm{obs}\}\), define its applicable controls:

\[
C_m=
\{
\mathrm{none},
\mathrm{smooth},
G,
\mathrm{wrong}_m,
\mathrm{Haar}_m
\}.
\]

For development key \(k\):

\[
D_{m,k}
=
A_{m,k}
-
\max_{c\in C_m}A_{c,k}.
\]

Candidate \(m\) is eligible only if:

\[
D_{m,0}>0,\qquad D_{m,1}>0,
\]

and:

\[
\frac{D_{m,0}+D_{m,1}}2\ge0.10.
\]

Thus a candidate must beat its strongest applicable control on both keys and by at least 10 percentage points on average.

Among eligible candidates, select the one with larger:

\[
S_m=\frac{A_{m,0}+A_{m,1}}2.
\]

If \(S_{\mathrm{raw}}=S_{\mathrm{obs}}\) to the stored accuracy precision, choose observable because it has the smaller payload. Training loss, \(R\)-loss, artifact rank, and wall time are not winner-selection metrics.

If neither candidate is eligible, Stage B is a valid FAIL and Stage C does not launch.

# Frozen probe assay

After student training, freeze every weight.

Using final checkpoint \(S_6\):

1. Extract the 64 calibration hidden vectors.
2. L2-normalize each vector.
3. For each output class \(y\), average its calibration vectors and normalize the resulting centroid \(\mu_y\).
4. For each of the 36 direct withheld inputs, predict:

\[
\hat y
=
\arg\max_y
\frac{h^\top\mu_y}{\|h\|_2\|\mu_y\|_2}.
\]

This probe uses only the original 64 labels, has no fitted optimizer or hyperparameter, and receives no withheld label during fitting.

Chance is \(1/12\). At least 26/36 correct is required to exceed 70% for an individual table.

The native student classifier remains the source of primary withheld-sequence accuracy; the centroid probe is the sidecar check.

# 5. Stage C sealed confirmation

## Freeze before keys

Before generating any sealed key, commit and hash:

- source code;
- Stage B winner;
- loser exclusion;
- all five control definitions;
- coefficient values;
- student seeds;
- calibration split generator;
- withheld-test generator;
- bank order;
- wrong-key derangement;
- Haar seeds and construction;
- statistical analysis code;
- PASS/FAIL/VOID rules.

Then generate eight 32-byte seeds with the OS CSPRNG and publish each seed hash. Store seeds in the ignored `secrets/` surface until reveal. Derive and serialize canonical key JSON exactly as Stage A specifies.

The eight keys may not be replaced because of anchor coverage, teacher difficulty, training results, or apparent outlier behavior.

## Run matrix

Student seeds:

```text
501
502
503
```

The six arms are:

1. Stage B winner, correct key;
2. no auxiliary;
3. generic smoothness;
4. static \(G\);
5. winner with wrong-key artifact;
6. winner with Haar/spectrum-matched artifact.

Therefore:

\[
8\times3\times6=144
\]

Transformer student runs.

The losing Stage B candidate is excluded. Testing it after observing sealed results would turn Stage C into another candidate search.

The conditional GRU experiment is a separate additional 144-run matrix and launches only after the Transformer Stage C PASS.

## Primary statistic

Let:

\[
A_{k,s,a}
\]

be withheld-set accuracy for key \(k\), seed \(s\), and arm \(a\). For control \(c\):

\[
d_{k,s,c}
=
A_{k,s,\mathrm{winner}}
-
A_{k,s,c}.
\]

Average paired seeds within key:

\[
\bar d_{k,c}
=
\frac13\sum_{s=1}^{3}d_{k,s,c}.
\]

The control-specific effect is:

\[
\Delta_c
=
\frac18\sum_{k=1}^{8}\bar d_{k,c}.
\]

The primary statistic is the weakest controlled advantage:

\[
\boxed{
\Delta_{\min}
=
\min_{c\in C}\Delta_c
}
\]

This is equivalent to adjudicating against the globally strongest control while requiring the result against every control.

## Key-cluster bootstrap

Use 100,000 bootstrap replicates with:

```text
SHA256("GAT_STAGE_C_KEY_BOOTSTRAP_V1")
```

as the PCG64DXSM seed.

For each replicate:

1. resample eight key indices with replacement;
2. keep all three paired seeds and all arms together within each sampled key;
3. recompute every \(\Delta_c\);
4. store \(\Delta_{\min}\).

Use the 2.5th percentile as the lower endpoint of a two-sided percentile 95% interval. Do not resample individual examples or treat the 4,000 deterministic test sequences as independent scientific replicates. Keys are the primary generalization units.

## Paired sign-flip test

As a confirmatory test, enumerate all \(2^8=256\) key-level sign vectors \(\epsilon_k\in\{-1,+1\}\):

\[
T_\epsilon
=
\min_c
\frac18
\sum_{k=1}^{8}\epsilon_k\bar d_{k,c}.
\]

The one-sided p-value is:

\[
p
=
\frac{
\#\{\epsilon:T_\epsilon\ge\Delta_{\min}\}
}{256}.
\]

This joint minimum statistic controls the five-control comparison without selecting a favorable control after the fact.

# PASS, FAIL, and VOID

## PASS

Stage C passes only if all of the following hold:

1. All eight teachers pass the Stage A teacher capacity and extraction gates.
2. All 144 student runs complete under the frozen protocol.
3. The winner beats every control by at least 20 points in the key-averaged paired comparison:

\[
\Delta_{\min}\ge0.20.
\]

4. The key-cluster bootstrap lower bound satisfies:

\[
\operatorname{LCB}_{95\%}(\Delta_{\min})>0.10.
\]

5. The joint paired sign-flip test satisfies:

\[
p\le0.05.
\]

6. For every key, the correct-winner arm’s three-seed mean centroid-probe recovery is at least 70% over its 36 withheld direct edges.
7. All artifact hashes, key commitments, data hashes, coefficient hashes, and run identities verify.
8. No forbidden information entered a student process.

This PASS supports the narrow claim that the frozen artifact carried post-committed transition information beyond the five declared controls. It does not establish language transfer or resurrect AMCL.

## FAIL

FAIL means the protocol was valid and interpretable, but one or more scientific thresholds were missed.

Examples:

- Stage B has no eligible candidate.
- \(\Delta_{\min}<0.20\).
- Bootstrap lower bound is at most 0.10.
- Probe recovery misses 70%.
- Static \(G\), wrong-key, Haar, smoothness, or no-auxiliary matches the winner.
- Correct-artifact loss falls but withheld accuracy does not improve.
- A frozen candidate deterministically diverges despite passing the pre-launch smoke test.

After a Stage C FAIL:

- do not test the Stage B loser on these keys;
- do not change \(\lambda\);
- do not replace failed keys;
- do not relax the 20/10-point thresholds;
- do not launch GRU confirmation.

A repaired or different candidate must restart with newly committed sealed keys.

## VOID

VOID is reserved for loss of experimental interpretability, not an unfavorable result.

VOID conditions include:

- sealed keys generated before configuration/code commitment;
- seed commitment mismatch;
- accidental reuse of a Stage A/B development key;
- student access to full key data, teacher weights, hidden states, logits, or withheld labels;
- incorrect labeled/withheld edge partition;
- wrong anchor ordering or bank membership;
- artifact hash mismatch;
- per-arm or post-outcome coefficient changes;
- outcome-dependent checkpoint selection;
- missing runs after the frozen hardware-retry rule;
- teacher capacity or extraction failure that prevents a valid source artifact;
- anchor minimum edge coverage below 400 for a sealed key;
- analysis-code discrepancy that changes the primary statistic.

A transient hardware/process crash permits one exact retry with the same run identity, seed, configuration, and artifact hash. A repeated deterministic model failure is FAIL, not VOID.

A void key cannot be silently replaced. The entire sealed confirmation remains VOID pending adjudication and a fresh commitment.

# 6. Implementation architecture

Use a modular library with thin, checkpointed stage orchestrators. Do not build a monolith.

The current [automaton module](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:1>) should be extended rather than duplicated.

Recommended source surface:

```text
src/cti_geometry_admission_automaton.py
src/cti_geometry_admission_models.py
src/cti_geometry_admission_geometry.py
src/cti_geometry_admission_artifacts.py
src/cti_geometry_admission_teacher.py
src/cti_geometry_admission_installer.py
src/cti_geometry_admission_statistics.py
src/cti_geometry_admission_verify.py
src/cti_geometry_admission_stage_a.py
src/cti_geometry_admission_stage_b.py
src/cti_geometry_admission_stage_c.py
```

Responsibilities:

- `automaton.py`: keys, simulation, anchors, 64-example calibration split, 4,000-example withheld set, direct probe set, hashes, coverage audits.
- `models.py`: Transformer/GRU definitions and the exact seven-checkpoint interface.
- `geometry.py`: differentiable \(X,G,A,R\), observable projection, static-\(G\), smoothness, Haar construction, numerical checks.
- `artifacts.py`: canonical schemas, base64 little-endian float32 array encoding, stable SHA-256, atomic writes, artifact validation.
- `teacher.py`: full-supervision teacher training and artifact-safe extraction handoff.
- `installer.py`: restricted-supervision student training and arm registry.
- `statistics.py`: centroid probe, Stage B selection, key-cluster bootstrap, sign-flip test, verdict calculation.
- `verify.py`: independent hash, provenance, forbidden-channel, run-completeness, and commitment verification.
- `stage_a/b/c.py`: orchestration only; no duplicated model, loss, or statistics implementations.

Teacher training, extraction, and student installation should run as separate process phases even though they share library code. This enforces the information boundary: the installer never imports a live teacher object or reads its checkpoint.

## Checkpointing

Every run identity should be:

```text
(stage, key_commitment, architecture, arm, seed,
 config_hash, data_hash, artifact_hash, code_hash)
```

Use append-only JSONL for run records and atomic JSON writes for summaries. A restart skips only an already completed record with an exact matching identity.

Recommended Stage B outputs:

```text
results/geometry_admission/stage_b/config.json
results/geometry_admission/stage_b/config.sha256
results/geometry_admission/stage_b/development_keys.json
results/geometry_admission/stage_b/calibration_manifest.json
results/geometry_admission/stage_b/withheld_eval_manifest.json
results/geometry_admission/stage_b/coefficient_calibration.json
results/geometry_admission/stage_b/teacher_runs.jsonl
results/geometry_admission/stage_b/artifact_manifest.json
results/geometry_admission/stage_b/student_runs.jsonl
results/geometry_admission/stage_b/screen_summary.json
results/geometry_admission/stage_b/decision.json
```

Recommended Stage C outputs:

```text
results/geometry_admission/stage_c/precommit.json
results/geometry_admission/stage_c/precommit.sha256
results/geometry_admission/stage_c/key_commitments.json
results/geometry_admission/stage_c/key_manifest.json
results/geometry_admission/stage_c/calibration_manifest.json
results/geometry_admission/stage_c/withheld_eval_manifest.json
results/geometry_admission/stage_c/teacher_runs.jsonl
results/geometry_admission/stage_c/artifact_manifest.json
results/geometry_admission/stage_c/student_runs.jsonl
results/geometry_admission/stage_c/probe_results.jsonl
results/geometry_admission/stage_c/statistical_test.json
results/geometry_admission/stage_c/verification.json
results/geometry_admission/stage_c/verdict.json
```

One live-file correction is required before sealing: the current `generate_sealed_key()` returns the key and seed hash but discards the original seed bytes. The commitment must remain revealable and independently verifiable, so the seed must be returned to the preparation command and stored only under the ignored `secrets/` surface.

