## Design verdict

**Design GO; implementation remains NO-GO.**

Canonical Stage B is a **12-run, one-key, three-initialization paired effect-size screen**. It supersedes R4’s 18-run Stage B and the current transposition/crossover implementation. R4’s loss definitions and broad-control Stage C remain conceptually valid, subject to the corrections below.

### Settled decisions

- **12 installer runs:** 3 student initializations × 2 candidates × same/wrong artifact.
- **Wrong artifact:** one fully trained, calibration-matched, fully deranged counterfactual teacher—not Haar and not a two-edge transposition.
- **Student initialization:** fresh step-0 weights using seeds 201, 202, 203. Never load the trained Stage A student checkpoints.
- **Stage B inference:** effect-size screen only. No significance claim.
- **Development key:** keep `DEVELOPMENT_KEY_JSON` as source, but atomically materialize and hash-check `stage_a/development_key.json`.
- **Capacity:** R9 `target_family` gates are canonical; extrapolation is diagnostic only.

The live evidence supports this: Stage A passes, all three student seeds pass R9, and the teacher’s `target_family=0.99989` despite diagnostic extrapolation `0.6184` ([capacity summary](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/geometry_admission/stage_a/capacity_summary.json:1>)).

## 1. Canonical Stage B manifest

Protocol identifier:

```text
OCF_GAT_STAGE_B_PAIRED_SCREEN_V1
```

Target:

```text
target key        = DEVELOPMENT_KEY_JSON
key slot          = 0
calibrated op     = a
calibration set   = exact 64-example set
withheld endpoint = exact 4,000 target-key examples
direct probe      = exact 36 target-key withheld edges
student seeds     = 201, 202, 203
```

Exact execution order:

| Order | Run ID | Init | Candidate | Artifact |
|---:|---|---:|---|---|
| 1 | `b_s201_raw_same` | 201 | raw R | target teacher |
| 2 | `b_s201_raw_wrong` | 201 | raw R | matched wrong teacher |
| 3 | `b_s201_obs_same` | 201 | observable | target teacher |
| 4 | `b_s201_obs_wrong` | 201 | observable | matched wrong teacher |
| 5 | `b_s202_raw_wrong` | 202 | raw R | matched wrong teacher |
| 6 | `b_s202_raw_same` | 202 | raw R | target teacher |
| 7 | `b_s202_obs_wrong` | 202 | observable | matched wrong teacher |
| 8 | `b_s202_obs_same` | 202 | observable | target teacher |
| 9 | `b_s203_obs_same` | 203 | observable | target teacher |
| 10 | `b_s203_obs_wrong` | 203 | observable | matched wrong teacher |
| 11 | `b_s203_raw_same` | 203 | raw R | target teacher |
| 12 | `b_s203_raw_wrong` | 203 | raw R | matched wrong teacher |

This counterbalances candidate and condition order while keeping each comparison adjacent.

Every run uses:

```text
steps              = 5000 exactly
optimizer          = AdamW
betas              = (0.9, 0.95)
eps                = 1e-8
weight_decay       = 0.01
peak lr            = 5e-4
warmup             = 250
cosine minimum     = 10% of peak
gradient clipping  = 1.0
task batch         = all 64 calibration examples
auxiliary batch    = one frozen 64-anchor bank
bank order         = GAT_INSTALLER_BANK_ORDER_V1
adjudicated model  = step 5000 only
```

No withheld accuracy or centroid-probe result is computed during training. Final evaluation occurs in a separate evaluator phase after the final checkpoint is hashed.

## 2. Wrong-key artifact

Use a **calibration-matched full derangement**:

1. Begin with the target development key.
2. Copy operation `a` exactly.
3. Deterministically generate replacement permutations for `b`, `c`, and `d`.
4. Reject each permutation until it differs from the target mapping at all 12 states.
5. Assert:

```text
calibration examples identical     = true
calibration labels identical       = true
withheld mappings differing        = 36/36
ordered anchors and banks identical = true
```

Train one wrong-key teacher with teacher seed 101 under the full Stage A protocol, apply the R9 teacher gates, and extract/hash raw and observable artifacts through the same pipeline as the target teacher.

Why:

- A completely random key mismatches the labeled calibration operation and creates a task-conflict confound.
- Haar is a structural null, not a semantic wrong-key control. Retain it for Stage C.
- A two-edge transposition changes only about 222–224 of the 4,000 withheld examples—at most roughly **5.6 percentage points** of directly attributable endpoint mass. It is unsuitable for a 10-point screen and tests a different localized-crossover question.

For observable artifacts, the entire artifact changes: both the wrong teacher’s `U_basis` and `R_obs`.

## 3. Student initialization

**Start from scratch.**

For seeds 201, 202, and 203:

1. Construct the untrained Transformer student.
2. Serialize its step-0 state once.
3. Hash it.
4. Reload those exact bytes for all four runs associated with that seed.

Do not use `transformer_s1/s2/s3/model_final.pt`. Those models already know essentially the whole automaton. Loading them would create ceiling contamination and turn Stage B into a fine-tuning assay.

Stage A established that the architecture and all three preregistered initialization seeds possess sufficient capacity. It did not produce installation starting points.

## 4. Stage B statistics

For candidate \(m\) and seed \(s\):

\[
\Delta_{s,m}=A^{same}_{s,m}-A^{wrong}_{s,m}.
\]

Candidate eligibility requires:

\[
\min_s\Delta_{s,m}>0,
\]

\[
\operatorname{median}_s(\Delta_{s,m})\ge0.10,
\]

\[
\operatorname{mean}_s(\Delta_{s,m})\ge0.10.
\]

The median requirement prevents one exceptional seed from carrying the screen.

If both candidates qualify, select the candidate with larger mean paired difference. An exact tie selects observable because of its smaller payload. If neither qualifies, Stage B is `SCREEN_FAIL`; Stage C does not launch.

Report the one-sided sign-test p-value descriptively. With 3/3 positive:

\[
p=1/8=0.125.
\]

Therefore this is explicitly **not statistical confirmation**.

### Stage C qualification

Stage C is the inferential stage, with keys—not seeds or evaluation examples—as the independent units.

The proposed “7/8 gives \(p=0.035\)” is correct only for the signs of one preregistered scalar contrast:

\[
P(X\ge7\mid n=8,p=0.5)=9/256=0.03515625.
\]

It is not valid for the current `cm_exact_sign_test()` applied to a thresholded conjunction of crossover, effect, and stability.

Canonical Stage C should retain the R4 **8 keys × 3 seeds × 6 arms = 144 runs**, key-cluster bootstrap, and joint minimum sign-flip statistic ([R4 Stage C](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/STEERING_DIALOGUE_R4_ROUND4_STAGE_BC_2026_07_26.md:625>)). Each sealed target key should receive its own calibration-matched deranged companion rather than an unmatched cyclic-key artifact.

## 5. Coefficients

Within a candidate pair, the coefficient must be identical; otherwise more than the artifact changes.

For candidate \(m\), use seed 400 and calculate one coefficient from the pooled 64 auxiliary gradients:

\[
\widetilde g_m
=
\operatorname{median}
\{g^{target}_{m,b},g^{wrong}_{m,b}:b=0,\ldots,31\},
\qquad
\lambda_m=g_{\mathrm{task}}/\widetilde g_m.
\]

Use that same \(\lambda_m\) for same-key and wrong-key runs. Freeze separate coefficients for raw and observable.

All gradients must be finite and nonzero. Also require the target/wrong median auxiliary-gradient ratio to lie within `[0.8, 1.25]`; otherwise the control is not strength-matched and Stage B preparation fails before outcomes.

## 6. File-level implementation plan

- `cti_geometry_admission_stage_b.py`: **full rewrite**. Remove partner/replay training, crossover logic, and in-memory extraction. Implement `prepare`, `install`, and `adjudicate` phases around the exact manifest. The current stale extrapolation check is visible at [line 164](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:164>).

- `cti_geometry_admission_installer.py`: retain optimizer schedule, loss dispatch, withheld evaluator, and centroid probe. Rewrite coefficient calibration, initialization loading, run identities, restart handling, final-only evaluation, and missing-artifact behavior. It currently constructs a fresh model without checkpoint input and skips runs by summary status alone ([line 335](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:335>)).

- `cti_geometry_admission_statistics.py`: replace `stage_b_selection()` with the paired screen above. Remove the CM conjunction sign test from the canonical path. Keep Stage C primary statistic/bootstrap/sign-flip functions after adding strict shape/finiteness validation and PCG64DXSM. Make every protocol check fail closed; current missing checks default to true ([line 225](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_statistics.py:225>)).

- `cti_geometry_admission_automaton.py`: retain simulation, calibration, withheld-set, anchor, and bank-order generators. Add deterministic calibration-matched full-derangement generation and key materialization. Retire `paired_key_from_transposition()` from the canonical experiment.

- `cti_geometry_admission_trainer.py`: keep training logic. Expose one canonical R9 teacher-gate function and call it everywhere; do not duplicate thresholds in orchestrators.

- `cti_geometry_admission_extraction.py`: retain raw and observable extraction. Generalize serialization/provenance for arbitrary teacher keys and add seven-checkpoint static-\(G\) serialization.

- `cti_geometry_admission_geometry.py`: retain the declared losses. Fix the eigenspectrum contract: clamp only values in `[-1e-6, 0]`; fail below `-1e-6`, on nonfinite values, or failed `eigh`.

- `cti_geometry_admission_stage_c.py`: full rewrite before sealed execution. The current 32-run transposition/crossover program is not R4 Stage C.

- `cti_geometry_admission_verify.py`: extend to verify the Stage B 12-run identity matrix, initialization equality, paired dataset hashes, artifact-only differences, coefficients, and fail-closed protocol fields.

- `cti_geometry_admission_models.py`: keep unchanged.

## 7. Required pre-launch gates

1. Atomically write `stage_a/development_key.json` from `DEVELOPMENT_KEY_JSON`. Its parsed canonical hash must equal the live anchor-manifest value:

```text
19e929b139569f2c7bee4ed6089334a34cd0abf366351b3d98a585f8155244e1
```

2. Validate all 32 target raw/observable bank hashes, numerical PASS, and repeat identity.

3. Wrong teacher must pass:

```text
in_range >= 0.995
target_family >= 0.995
direct edges = 48/48
final two evaluations pass
anchor accuracy >= 0.95
perturbation accuracy >= 0.95
extraction numerical and repeat gates pass
```

4. Assert calibration hashes match between target and wrong keys and all 36 withheld mappings differ.

5. Verify every paired run shares initialization hash, calibration hash, withheld hash, optimizer/config hash, bank-order hash, and coefficient. Only condition and artifact hash may differ.

6. Missing artifact targets, zero/nonfinite gradient norms, nonfinite loss/logits, or materially negative Gram eigenvalues must fail—never silently become no-auxiliary.

7. Run identity must bind code, configuration, initialization, data, anchor, artifact, and coefficient hashes. A directory name plus `status=complete` is insufficient.

8. Extend Stage A artifacts with `static_gram_manifest.json` before Stage C; R5 already identified that missing output ([R5 audit](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/STEERING_DIALOGUE_R5_FINAL_2026_07_26.md:36>)).

**Signed off:** this 12-run paired screen is the canonical Stage B design. No Stage B launch should occur until the implementation and independent verifier conform exactly. No files were changed.