[SEVERITY: CRITICAL] [FILE:src/cti_geometry_admission_installer.py:144] Coefficient calibration crashes before training. `trunk_params` includes the final normalization weight, but auxiliary losses use pre-normalization hidden states, so that parameter is absent from the auxiliary graph. Both raw and observable seed-400 calibration produced `RuntimeError: One of the differentiated Tensors appears to not have been used in the graph.` None of the 12 runs can currently start.

[SEVERITY: HIGH] [FILE:src/cti_geometry_admission_installer.py:141] Calibration remains fail-open after that crash is repaired. Zero/nonfinite auxiliary gradients are skipped, an empty set silently returns `1.0`, task-gradient finiteness/nonzero is not checked, the achieved 1:1 ratio is not verified within 5%, and Haar gradients receive no finite/nonzero diagnostic.

[SEVERITY: CRITICAL] [FILE:src/cti_geometry_admission_stage_b.py:252] The prepare/install information boundary is not real. `install()` receives the complete in-memory `prep`, which contains withheld labels and probes, and the same process retains the development key. Runs receive in-memory raw and observable artifacts instead of consuming only the selected serialized artifact file. This violates the restricted installer-process contract and requires `VOID`.

[SEVERITY: CRITICAL] [FILE:src/cti_geometry_admission_stage_b.py:108] Required Stage A validation is not implemented. Raw/observable manifests are loaded but never used; bank hashes, exact shapes, anchor ordering, and bank membership are not checked. Only file existence, dtype, and finiteness are tested. The current live artifacts independently pass these checks, but tampering would not be detected by this orchestrator.

[SEVERITY: CRITICAL] [FILE:src/cti_geometry_admission_stage_b.py:140] Haar matrices and transformed artifacts are kept only in memory. Neither is serialized, and only each `Q` byte string is hashed. There are no transformed-artifact hashes, shape/dtype-bound serialization, or correct-artifact bank hashes in the precommit. Thus the Haar input actually used by training is not committed.

[SEVERITY: CRITICAL] [FILE:src/cti_geometry_admission_stage_b.py:213] The precommit is not an immutable hash chain. It omits code/configuration hashes, ordered anchor data, correct/Haar artifact files, and per-run identities. It has no separately recorded pre-execution hash, is overwritten on rerun, and adjudication hashes the in-memory dictionary rather than verifying the committed file.

[SEVERITY: CRITICAL] [FILE:src/cti_geometry_admission_installer.py:333] Completed runs are reused solely from directory name plus `status=="complete"`. Artifact, initialization, coefficient, data, configuration, and code changes can therefore reuse stale results. Additionally, `summary.json` is written before `model_final.pt`; a failed final save leaves a “complete” run that will be skipped.

[SEVERITY: CRITICAL] [FILE:src/cti_geometry_admission_stage_b.py:378] Adjudication fabricates protocol validity. Six checks are hard-coded `True`; only summary presence/status is examined. Initialization pairing, artifact hashes, finite logs/models, coefficients, forbidden channels, final step, and checkpoint identity are never recomputed. Missing or corrupt inputs generally raise and terminate instead of emitting `STRUCTURAL_SCREEN_VOID`. There is no independent reconstruction.

[SEVERITY: HIGH] [FILE:src/cti_geometry_admission_stage_b.py:357] Checkpoints are evaluated before being hashed, contrary to the binding order. The resulting checkpoint hash is not compared with any training summary or committed identity.

[SEVERITY: CRITICAL] [FILE:src/cti_geometry_admission_statistics.py:57] The screen does not validate structure, completeness, types, ranges, or finiteness. A probe with one candidate containing a NaN endpoint still returned `STRUCTURAL_SCREEN_PASS` because the other candidate qualified. Any nonfinite endpoint must make the entire protocol `VOID`.

[SEVERITY: HIGH] [FILE:src/cti_geometry_admission_statistics.py:71] The exact 0.10 boundary is numerically wrong. Three mathematical deltas of `0.6 - 0.5 = 0.10` produced `0.09999999999999998`, causing `STRUCTURAL_SCREEN_FAIL`. Thresholds should be adjudicated from integer correct counts—400 examples for 0.10—not binary floating subtraction.

[SEVERITY: HIGH] [FILE:src/cti_geometry_admission_geometry.py:45] Eigenspectrum handling is incomplete. A zero-trace Gram produces zero ridge and nonfinite `R` without an exception; ridge positivity and final `R` finiteness are not checked. Moreover, intended step-0 seeds already produced computed minima of `-1.20e-6` and `-1.07e-6` on legitimate PSD Grams, below the locked abort threshold. The compliant numerical formulation and exact-seed preflight must be stabilized; the threshold must not simply be loosened.

[SEVERITY: HIGH] [FILE:src/cti_geometry_admission_installer.py:397] Required monitoring is incomplete. There is no explicit finite check for total loss or logits, no post-step parameter check, update norm, clipped norm, GradScaler scale, eigenspectrum diagnostic, or thermal record. Consequently `all_losses_finite=True` cannot be substantiated.

[SEVERITY: HIGH] [FILE:results/codex_r10_round3.md:37] Binding amendment 5 is absent. No canonical locked protocol document exists, R4 still declares its 18-run design frozen, and `stage_b.py:458` tells users to proceed to Stage C even though Stage C is explicitly NO-GO pending rewrite.

[SEVERITY: LOW] [FILE:src/cti_geometry_admission_installer.py:340] The checkpoint branch is ordered correctly: the model is created before loading, and no manual seed is set in that branch. However, the no-checkpoint fallback seeds only after constructing the model, making that fallback seed ineffective.

Confirmed positives: the Helmert basis is `(n,n-1)`, orthonormal, and mean-orthogonal; Haar structures match the correct target dictionaries, use one rotation per bank, retain observable `U_basis`, and preserve spectra to float32 precision. File handles are properly closed. GPU capacity is not a blocker: the 1.92M-parameter model peaked around 204 MiB allocated/226 MiB reserved for a representative full raw step on the 24,463 MiB RTX 5090, with no evident cross-run retention risk.

Required fixes before re-review:

1. Repair coefficient autograd handling and enforce every calibration/Haar-gradient invariant.
2. Serialize and validate all correct/Haar artifacts and rotations.
3. Build a complete, atomic, immutable input precommit and per-run identity chain.
4. Separate prepare, install, and adjudicate into restricted file-mediated processes.
5. Permit restart only for an exact identity; save/hash the final model before atomically marking completion.
6. Replace hard-coded checks with an independent fail-closed verifier and guaranteed `VOID` emission.
7. Validate endpoint schemas/finiteness and adjudicate thresholds from integer counts.
8. Complete and preflight the eigenspectrum and finite-value contract on seeds 201–203.
9. Add every mandatory monitoring field and derive validity from the logs.
10. Publish the locked Stage B protocol, mark R4 superseded, and keep Stage C explicitly blocked.
11. Move fallback seeding before model construction.

VERDICT: NO-GO for launching the 12 training runs.

