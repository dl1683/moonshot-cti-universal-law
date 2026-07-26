## Round 2 decisions

I accept the cost argument. **The deranged teacher is not required for Stage B.** Stage B becomes a structural candidate screen; semantic key selectivity moves entirely to Stage C.

### 1. Wrong-key teacher: defer to Stage C

Use the R4 Haar-matched artifact in Stage B.

For each candidate:

\[
\Delta_{s,m}=A^{correct}_{s,m}-A^{Haar}_{s,m}.
\]

Eligibility remains:

\[
\min_s\Delta_{s,m}>0,\qquad
\operatorname{median}_s\Delta_{s,m}\ge0.10,\qquad
\operatorname{mean}_s\Delta_{s,m}\ge0.10.
\]

This is cost-rational because Stage B asks only:

> Does the candidate contain useful anchor-specific organization beyond a spectrum- and depth-correlation-matched structural null?

It does **not** ask whether the artifact installs the correct key rather than another semantically valid key.

One qualification: “cannot beat Haar, cannot beat anything” is not logically guaranteed—Haar might accidentally regularize well—but it is an appropriate triage rule before a costly sealed experiment.

Consequences:

- No Stage B partner teacher.
- No transposition teacher.
- No replay teacher.
- No Stage B claim of wrong-key selectivity.
- Stage B verdict should be named `STRUCTURAL_SCREEN_PASS`, not scientific `PASS`.
- Stage C must include the calibration-matched deranged-teacher control and Haar as separate controls.

For observable Haar, retain the target teacher’s `U_basis` and rotate only `R_obs` exactly as R4 specifies. For raw Haar, use the centered-subspace orthogonal similarity transform with one fixed \(Q_b\) per bank across all depths.

### 2. Coefficient: target-only calibration

Agreed. Drop my pooled calibration and `[0.8, 1.25]` gate.

For each candidate \(m\):

\[
\widetilde g_m
=
\operatorname{median}_{b=0}^{31}
\left\|\nabla_{\theta_{\mathrm{trunk}}}
\mathcal L_m^{correct}(B_b)\right\|_2,
\]

\[
\lambda_m=\frac{g_{\mathrm{task}}}{\widetilde g_m}.
\]

Then use the exact same coefficient for:

```text
raw correct
raw Haar

observable correct
observable Haar
```

This is cleaner:

- coefficient is calibrated solely from the real artifact;
- it is frozen before outcomes;
- only artifact identity changes within a pair;
- Haar cannot receive favorable independent scaling.

Haar auxiliary gradients must still be finite and nonzero, but their magnitude is diagnostic—not a separate gate or recalibration trigger.

The same principle should govern Stage C: the winner’s correct-artifact coefficient is used unchanged for correct, wrong-key, and Haar candidate arms.

### 3. Fresh initialization and 5,000 steps

Keep **fresh initialization and exactly 5,000 steps**.

The student is not expected to infer the whole automaton from 64 labels alone. The intended division is:

- the 64 examples teach the labeled operation, class interface, and output alignment;
- the artifact supplies information about the withheld structure;
- Haar tests whether correct anchor organization matters.

The 64 examples are repeated as a full batch for 5,000 steps, while the student receives 5,000 auxiliary updates—about 156 complete cycles through the 32 artifact banks. Optimization exposure is therefore substantial even though labeled diversity is deliberately restricted.

Overfitting is not merely a nuisance here; it is part of the test. A student that memorizes the 64 labels but cannot generalize beyond Haar should fail.

Do not increase to 10,000:

- it is not justified by existing installer evidence;
- it doubles cost;
- it may amplify calibration memorization;
- it changes a frozen R4 parameter without a principled selection rule.

If 5,000 steps are insufficient, Stage B should reveal that. Any later 10,000-step protocol would be a separately declared second-generation screen, not an invisible extension.

### 4. Monitoring: losses yes, probe accuracy no

Accept active monitoring, but reject computing centroid-probe accuracy every 250 steps.

Safe monitoring:

```text
task loss
auxiliary loss
total loss
calibration accuracy
learning rate
gradient norm before clipping
clipped gradient norm
GradScaler scale
parameter/update norm
finite-value checks
wall time and temperature
```

Log the compact set every 100 steps; calibration accuracy may be computed every 250.

This catches silent divergence without exposing a withheld behavioral endpoint. Abort immediately on:

- nonfinite loss or gradient;
- missing auxiliary target;
- failed eigendecomposition;
- eigenvalue below `-1e-6`;
- nonfinite parameters or logits;
- repeated deterministic numerical failure.

Do **not** compute the 36-edge centroid-probe accuracy during training. Even if declared “non-decisional,” visible probe trajectories can influence aborts, retries, interpretation, and subsequent protocol changes.

The final sequence accuracy and centroid probe are computed only after the step-5000 checkpoint is saved and hashed.

### 5. Run order: simplify it

Agreed. Counterbalancing provides little value once every run:

- loads an immutable initialization checkpoint;
- has a fresh optimizer;
- uses frozen deterministic inputs;
- receives the same cooldown policy.

Use simple adjacent correct-then-Haar pairs.

## Revised exact 12-run manifest

| Order | Run ID | Init | Candidate | Condition |
|---:|---|---:|---|---|
| 1 | `b_s201_raw_correct` | 201 | raw | correct artifact |
| 2 | `b_s201_raw_haar` | 201 | raw | Haar artifact |
| 3 | `b_s201_obs_correct` | 201 | observable | correct artifact |
| 4 | `b_s201_obs_haar` | 201 | observable | Haar artifact |
| 5 | `b_s202_raw_correct` | 202 | raw | correct artifact |
| 6 | `b_s202_raw_haar` | 202 | raw | Haar artifact |
| 7 | `b_s202_obs_correct` | 202 | observable | correct artifact |
| 8 | `b_s202_obs_haar` | 202 | observable | Haar artifact |
| 9 | `b_s203_raw_correct` | 203 | raw | correct artifact |
| 10 | `b_s203_raw_haar` | 203 | raw | Haar artifact |
| 11 | `b_s203_obs_correct` | 203 | observable | correct artifact |
| 12 | `b_s203_obs_haar` | 203 | observable | Haar artifact |

If both candidates qualify, choose the larger mean correct-minus-Haar difference; exact tie goes to observable.

## Round 2 canonical position

Locked changes:

- **Stage B:** correct artifact versus Haar only.
- **Stage B cost:** zero additional teacher training.
- **Coefficient:** target-only calibration, identical within each pair.
- **Initialization:** fresh frozen step-0 weights for seeds 201–203.
- **Training:** exactly 5,000 steps.
- **Monitoring:** optimization and calibration health only.
- **Evaluation:** withheld sequence accuracy and centroid probe only after final checkpoint hashing.
- **Order:** simple adjacent correct-then-Haar pairs.
- **Stage C:** calibration-matched semantic wrong-key control remains mandatory.

This is a better Stage B: cheaper, narrower, and honest about what it establishes.