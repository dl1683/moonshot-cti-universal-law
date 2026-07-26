## Final sign-off: APPROVED WITH FIVE BINDING AMENDMENTS

The scientific design is locked as `OCF_GAT_STAGE_B_STRUCTURAL_SCREEN_V1`. Implementation may proceed after incorporating these details.

1. **Reuse and correct existing Haar functions.** Do not create duplicate implementations. Update `generate_haar_rotation_raw/obs()` and `apply_haar_to_*()`:

   - use `np.random.Generator(np.random.PCG64DXSM(...))`, not `default_rng`;
   - seed from the full domain-separated SHA-256 digest;
   - use an explicit fixed Helmert basis, not the current SVD-derived centered basis;
   - QR-sign-canonicalize so `diag(R) > 0`;
   - serialize and hash every \(Q_b\) and transformed artifact;
   - use one \(Q_b\) per bank across all six depths and all seeds;
   - observable Haar retains the target `U_basis`.

2. **Freeze coefficient calibration completely.**

   ```text
   reference seed = 400
   reference state = fresh step-0 Transformer
   artifact = target correct artifact only
   banks = all 32
   coefficients = one raw, one observable
   ```

   Require finite, nonzero task and auxiliary gradients and achieved target-artifact gradient ratio within 5% of 1.0. No clipping, sweep, or control-specific calibration.

3. **Centroid probe is diagnostic only.** It is computed after the final checkpoint hash, but it does not affect eligibility, candidate selection, or winner choice. The primary screen endpoint is step-5000 withheld-sequence accuracy.

4. **Add `STRUCTURAL_SCREEN_VOID`.** PASS/FAIL alone cannot safely represent protocol invalidity:

   - `PASS`: valid protocol and an eligible winner.
   - `FAIL`: valid protocol, but neither candidate meets the screen.
   - `VOID`: hash mismatch, invalid initialization pairing, missing/corrupt artifacts, forbidden information exposure, incorrect data partition, or other loss of interpretability.

   A reproducible candidate divergence after a valid preflight is scientific `FAIL`; a transient hardware crash permits one exact-identity retry.

5. **Record the supersession before execution.** Add a canonical locked protocol document for `OCF_GAT_STAGE_B_STRUCTURAL_SCREEN_V1` and mark R4’s 18-run Stage B and the CM/transposition implementation as superseded. Stage C remains separately NO-GO until rewritten.

## Approved implementation plan

### Prepare

- Atomically materialize `development_key.json`.
- Verify its logical canonical hash against `anchor_manifest.json`.
- Validate:

  ```text
  stage_a_pass = true
  32 raw banks and hashes
  32 observable banks and hashes
  numerical_audit.all_pass = true
  repeat_match_raw = true
  repeat_match_obs = true
  anchor ordering and bank membership
  target artifact shapes, float32 dtype, and finiteness
  ```

- Generate, serialize, and hash raw/observable Haar artifacts.
- Generate and hash step-0 initializations for seeds 201–203.
- Calibrate and freeze target-only raw/observable coefficients.
- Hash calibration, withheld, direct-probe, anchor, and bank-order datasets.
- Write the exact 12-run manifest and immutable precommit before installation.

### Install

Execute adjacent correct/Haar pairs exactly as declared. All four runs for a seed reload the same step-0 checkpoint bytes.

Monitor only:

```text
task/aux/total loss
calibration accuracy
learning rate
gradient and clipping norms
GradScaler state
parameter/update norms
finite-value and eigenspectrum checks
thermal and wall-time diagnostics
```

No withheld evaluation or centroid probe during optimization. The installer process must consume artifact files, not teacher checkpoints or live teacher objects.

### Adjudicate

After each step-5000 checkpoint is saved and hashed:

1. evaluate the frozen 4,000-example withheld set;
2. compute the frozen 36-edge centroid probe;
3. independently reconstruct paired differences;
4. apply the locked min/median/mean thresholds;
5. select by mean \(\Delta\), with observable winning an exact tie;
6. emit PASS/FAIL/VOID with explicit protocol checks.

## File decisions

- `automaton.py`: add atomic development-key materialization and verification.
- `geometry.py`: repair existing Haar construction and eigenspectrum handling.
- `statistics.py`: replace Stage B selection; make verdicts fail-closed.
- `stage_b.py`: complete rewrite into prepare/install/adjudicate.
- `installer.py`: checkpoint initialization, exact run identity, safe monitoring, numerical aborts, final-only evaluation, and `try/finally` cleanup.
- `models.py`: unchanged.
- `extraction.py`: unchanged for Stage B.
- Stage A source: unchanged.
- Stage C: not part of this implementation and remains blocked.

**Explicit sign-off:** proceed with implementation of `OCF_GAT_STAGE_B_STRUCTURAL_SCREEN_V1` under this locked contract.