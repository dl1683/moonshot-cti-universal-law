**NO-GO** — donor GPU training must not proceed.

The current artifact passes the current verifier, and all three source hashes match. However, the gate does not yet guarantee that the executed experiment is the committed experiment.

## Blockers

1. **CRITICAL — The precommit is not independently tamper-evident**

   [`integrity_sha256`](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:296>) is a self-hash stored inside the same mutable JSON. Anyone changing the artifact and source files can recompute the source hashes and self-hash, after which verification passes. The protocol specification itself is not hashed. The local Git commit is unsigned, and the verifier does not require a trusted commit or clean checkout.

   Exact fix:

   - Publish the canonical precommit digest outside the mutable repository—signed release/tag, append-only registry, timestamped transparency log, or another trusted channel.
   - Make the verifier require that trusted digest as an input and compare it against the single-read canonical payload.
   - Freeze `CAUSAL_SKILL_ORGAN_ADMISSION_V1.md` by hash.
   - Bind the exact Git tree/commit and reject dirty or unexpected source trees.
   - Do not allow `create_precommit()` to overwrite an already anchored V1 artifact.

2. **HIGH — `training_config` is incomplete and is not the configuration actually executed**

   [`training_config()`](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_train_donor.py:62>) omits outcome-affecting values including:

   - Training RNG seed `42` and evaluation seed `9999`
   - Gradient clipping threshold `1.0`
   - Optimizer class and AdamW `betas`, `eps`, `amsgrad`, `foreach`/`fused` behavior
   - LR schedule identity and exact endpoint convention
   - AMP/autocast dtype and GradScaler settings
   - Evaluation batch count, evaluation batch size, and final evaluation count
   - Loss reduction
   - Full versus smoke run mode
   - Checkpoint resume/selection policy
   - Generator retry limits and effective train/evaluation length ranges
   - CUDA deterministic, TF32, matmul, and threading settings

   Source hashes indirectly bind literals in the files, but that is not equivalent to a complete, typed experimental configuration.

   There is also a check/use gap: [`train_donor()`](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_train_donor.py:253>) verifies a newly constructed dictionary, then continues reading mutable globals such as `TORCH_SEED`, `LR`, `WEIGHT_DECAY`, `BATCH_SIZE`, `WARMUP_STEPS`, `CAPACITY_GATES`, and `COOLDOWN_SECONDS`. Direct mutation before verification is now caught, but mutation after verification remains possible. The mutable `partitions` dictionary has the same problem.

   Exact fix: construct one immutable, exhaustive configuration object, verify it, and use only that validated object for model construction, RNG creation, optimizer setup, batching, scheduling, clipping, evaluation, gates, and checkpoint policy. Return the validated configuration from the verifier rather than rereading globals.

3. **HIGH — Strict type verification is demonstrably incomplete**

   [`_strict_config_eq()`](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:352>) accepts a frozen integer for a live float:

   ```text
   frozen=0, live=0.0 → ACCEPT
   ```

   Lists are not recursively checked:

   ```text
   frozen=[1], live=[True] → ACCEPT
   frozen=[{"x": 1}], live=[{"x": True}] → ACCEPT
   ```

   Top-level fields use ordinary Python equality, so values such as `49062.0` can satisfy the live integer `49062`. `None` round-trips correctly when both sides are `None`; string/bytes mismatches reject, but unsupported containers are not fail-closed.

   Exact fix:

   - Require exact type identity for scalar leaves: `type(frozen) is type(live)`.
   - Recurse into both dictionaries and lists, checking exact lengths, key types, and element types.
   - Handle `None` explicitly.
   - Reject unsupported live/frozen types rather than falling back to Python equality.
   - Apply this single validator to every precommit field, including counts, seeds, encoding, partition structures, runtime environment, hashes, model configuration, and training configuration.

4. **HIGH — Checkpoint resume is neither authenticated nor safely loaded**

   [`load_checkpoint()`](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_train_donor.py:208>) calls `torch.load(..., weights_only=False)` before checking the embedded precommit hash. A malicious or corrupted pickle can execute code before validation. The embedded hash only establishes that the checkpoint claims a particular precommit; it does not authenticate the model, optimizer, RNG states, step, or lineage.

   `find_latest_checkpoint()` also trusts filename ordering and does not verify that the saved step matches the filename or lies within the committed run.

   Exact fix:

   - Use a non-executable format or `weights_only=True`, separating JSON-safe metadata and RNG state where necessary.
   - Atomically save checkpoints.
   - Hash the complete checkpoint and record its digest in an external append-only run manifest chained to the precommit.
   - Before restoring state, validate schema, exact types, tensor names/shapes/dtypes, step bounds, filename/step agreement, optimizer configuration, and parent-checkpoint digest.
   - Reject rather than silently accepting missing RNG/scaler fields for resumable committed runs.

5. **HIGH — Bit-identical determinism is not guaranteed**

   The current environment reported:

   ```text
   deterministic_algorithms=false
   cudnn.deterministic=false
   cudnn.allow_tf32=true
   CUBLAS_WORKSPACE_CONFIG unset
   ```

   The artifact’s [`runtime_env`](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/causal_organ/precommit.json:442>) does not freeze GPU model/capability, driver, OS/architecture, backend flags, matmul precision, AMP dtype, CUDA-visible-device mapping, or thread counts.

   Exact fix: configure determinism before model creation, then freeze and verify it:

   - `torch.use_deterministic_algorithms(True)`
   - `torch.backends.cudnn.deterministic = True`
   - `torch.backends.cudnn.benchmark = False`
   - Disable TF32 or explicitly commit its use
   - Set the required cuBLAS workspace configuration before process startup
   - Freeze GPU identity/capability, driver, OS, device index, thread counts, AMP/scaler configuration, and matmul precision
   - Add fresh-run and checkpoint-resume tests requiring identical model, optimizer, scaler, and RNG-state hashes

6. **HIGH — Evaluation strata are heavily confounded and withheld results are exposed during training**

   Training generation itself respected the intended boundaries in a 5,000-sample probe: zero held-out-state, excluded-bigram, or withheld-trigram violations.

   The evaluation strata do not isolate the dimensions named in the protocol. In 5,000 samples:

   - 4,970/5,000 “length extrapolation” examples also contained excluded bigrams.
   - 3,489/5,000 also contained withheld trigrams.
   - Every length example used a held-out initial state.
   - 1,291/5,000 “held-out state” examples contained withheld trigrams.
   - 3,300/5,000 “withheld trigram” examples also used long lengths, and all used held-out states.

   This follows directly from the generators at [transducer:684](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:684>). Consequently, the reported `length_extrapolation` gate is actually a multi-factor intersection despite the protocol separately defining a full withheld intersection.

   In addition, [`evaluate()`](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_train_donor.py:135>) exposes every withheld stratum every 500 steps. It does not backpropagate or automatically select by withheld accuracy, but it exposes results to the operator before training concludes.

   Exact fix:

   - Define single-factor strata that hold all other dimensions to training support.
   - Reject accidental excluded bigrams/withheld trigrams in isolated strata.
   - If the existing withheld trigrams intrinsically contain excluded bigrams, either define them explicitly as a composite split or create a properly isolated precommitted set before training.
   - Reserve the existing generator for the named full-intersection split.
   - During training, evaluate only training-distribution capacity; run sealed withheld evaluation once against a frozen final checkpoint.

7. **MEDIUM — Architecture binding is only partially explicit**

   Donor `d_model`, layer count, heads, state dimension, dropout, parameter count, and estimated MACs are frozen correctly. However, [`model_config`](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:1324>) omits architecture-affecting fields such as feed-forward dimension/multiplier, activation, normalization placement and epsilon, biases, embedding vocabulary sizes, output-head structure, socket dimensions, host `d_organ`, and organ `d_hidden`.

   The model source hash catches ordinary source edits, but the actual donor is still created later from defaults rather than from the verified model configuration.

   Exact fix: serialize the complete constructor and block configuration for every model and instantiate the actual model from that validated configuration. Verify the resulting module schema, parameter names/shapes/dtypes, frozen flags, socket bytes, parameter count, and architecture fingerprint before constructing the optimizer.

8. **MEDIUM — The protocol’s eight-GPU-hour donor gate is not enforced**

   The locked protocol requires donor capacity within eight GPU-hours, but the training loop only records wall time and [`check_gates()`](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_train_donor.py:168>) can return success regardless of budget. A model finishing after the cap would therefore be reported as passed.

   Exact fix: commit the budget and timing method, synchronize CUDA around measurements, record cumulative fresh-plus-resumed GPU time in the checkpoint chain, stop at the cap, and make budget compliance mandatory for `final_pass`.

## Checks that passed

- The checked-in artifact currently passes `verify_precommit()`.
- All three frozen source hashes match the live files.
- Duplicate JSON keys and non-finite constants are rejected.
- The precommit itself is read once and its verified hash is returned; the earlier second-read issue is fixed.
- Loss is correctly averaged across four register heads, and exact-state accuracy is computed correctly.
- AMP unscaling occurs before gradient clipping.
- Resume bookkeeping is nominally off-by-one correct: checkpoints store `step + 1`, and the resumed range begins at that step.
- Training partitions are disjoint, cover all 65,536 states, and sampled training examples obey the current exclusions.
- The checkout remained clean at `ba719b25725a82f04bf90c04337143ea5c4e29dc`; no training was launched and no repository files were changed.

