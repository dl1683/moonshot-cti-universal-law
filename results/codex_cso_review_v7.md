# CSO precommit verification review (post-fe8f437)

## Verdict: NO-GO

Donor capacity training must not proceed. The requested v7 fix checks all pass, but the precommit boundary still accepts configuration drift and does not freeze the actual runtime/reproducibility envelope.

Review target: live `HEAD` `902593cb7ca9c8d09fb7bce457fb67b395a6a148`. This is `fe8f4376cc0ef210ea490f76d888d0d6cdfe5675` plus one `STATUS.md`-only commit, so the reviewed implementation files are identical to commit `fe8f437`.

## Blocking findings

### 1. HIGH - Full-run `max_steps` and the resulting LR schedule are not committed

The trainer source hash freezes the default source text, but not the runtime call. `train_donor(max_steps=MAX_STEPS, smoke=False)` exposes `max_steps` as a public argument (`src/cti_causal_organ_train_donor.py:242`). That value controls both the loop horizon (`:310`) and every cosine-schedule value (`:312`, via `lr_schedule()` at `:110-114`). A caller can run, for example, `train_donor(max_steps=1000)` with the committed generator, models, and trainer source hashes unchanged. `verify_precommit(partitions)` has no runtime training-config argument and therefore still passes.

The artifact has no `training_config` or `max_steps` field; its required top-level schema ends with partition data, source hashes, socket commitments, hashes, and `model_config` (`src/cti_causal_register_transducer.py:303-312`). `max_steps` is only recorded after training in the result (`src/cti_causal_organ_train_donor.py:397-407`), which is too late for a precommit.

Required fix: commit a normalized `training_config` and require the trainer to pass its effective runtime config into verification before model construction. At minimum bind run mode, `max_steps`, batch size, optimizer/LR/weight decay, warmup/schedule, evaluation/checkpoint cadence, all RNG seeds, AMP policy, and gate thresholds. Alternatively, remove runtime overrides from the locked full-run entry point.

### 2. HIGH - Device, precision path, and software environment are selected after the commitment boundary

`DEVICE` is selected from CUDA availability at import time (`src/cti_causal_organ_train_donor.py:34`). GradScaler and autocast are then enabled only on CUDA (`:284`, `:319`), so the same accepted precommit permits materially different CPU/full-precision and CUDA/AMP experiments. The chosen device is recorded only after training (`:404`). No Python, NumPy, PyTorch, CUDA/cuDNN version, hardware identity, or deterministic-algorithm policy is frozen or verified, and no `torch.use_deterministic_algorithms(...)`/cuDNN determinism configuration exists in this CSO path.

Source hashes do not close this gap because dependency implementations and the selected precision/device branch are outside those source files. This is a reproducibility-envelope inconsistency that the verifier cannot reject.

Required fix: precommit and verify the supported runtime envelope before training (at least device type, AMP policy, Python/NumPy/PyTorch/CUDA versions, and deterministic-algorithm settings), or explicitly lock execution to one environment and enforce that lock. Record hardware/driver metadata as well if exact reruns are required across machines.

### 3. MEDIUM - JSON booleans pass as frozen numeric zeroes

For live float fields, `verify_precommit()` accepts any `isinstance(frozen_val, (int, float))` and then compares `float.hex()` (`src/cti_causal_register_transducer.py:493-503`). In Python, `bool` is a subclass of `int`, and `float(False).hex()` equals `float(0.0).hex()`.

Adversarial test: I changed only `model_config.donor.dropout` from JSON `0.0` to JSON `false`, recomputed the full `integrity_sha256`, and supplied the document in memory. `verify_precommit()` printed `Precommit verification: PASS`. The artifact is type-inconsistent but accepted, contrary to the fail-closed/strict-schema requirement. The same bypass applies to the other committed zero-valued dropout fields.

Required fix: reject booleans explicitly and validate exact JSON types for every schema field before semantic comparison (for example, `type(value) is float` for committed float fields and `type(value) is int` for integer fields). Apply the same rule recursively across `model_config`, not only its current float fields.

## Requested verification matrix

| # | Check | Result | Evidence |
|---:|---|---|---|
| 1 | `_json_strict` rejects NaN and duplicate keys | PASS | `NaN` raised `ValueError: Non-finite JSON constant`; duplicate `x` raised `ValueError: Precommit contains duplicate keys`. |
| 2 | `integrity_sha256` covers all fields | PASS | Changing only `model_config.donor.n_heads` changed the recomputed digest from `8b9ad9447bed...` to `d12262f06000...`. The untouched recomputation exactly equals frozen `8b9ad9447beda5ba3019aed2d0cd2c19394994b459d06723ad006305a1f47784`. |
| 3 | Float comparison uses `float.hex()` | PASS | Source uses exact hex comparison at `src/cti_causal_register_transducer.py:499`; a one-ULP `host_t.compute_ratio` change (`0.06686903729401561` to `0.06686903729401562`) was rejected. |
| 4 | Derived-set consistency | PASS | Replacing `included_bigram_set` with an empty frozenset raised `included_bigram_set inconsistent with included_bigrams`. |
| 5 | `verify_data_generation` reconstructs canonical sets | PASS | An injected training example used excluded bigram `(0, 0)` while the tampered derived set called it included; verification rejected it as `Excluded bigram in train: (0, 0)`, demonstrating checks use canonical lists (`src/cti_causal_register_transducer.py:1049-1065`). |
| 6 | Socket hashes ignore ambient thread count | PASS | Both oracle and actual socket hashes were identical at 1, 2, and 4 threads. Transformer: `e57d7d7de57bb4540ea2adf73f47126dba1ed5b8e8771b9ca5ae4a2dbf611ddd`; GRU: `eb9669024b85321ed77de2a6dd4b74e9f90723353644d7724bfcc0d2a34ec15d`. Actual `_make_frozen_socket()` bytes matched the commitments. |
| 7 | `n_heads` and `dropout` are frozen | PASS | Donor `n_heads` 6 to 3 and dropout 0.0 to 0.25 were independently rejected. Donor and Transformer host store both as instance attributes; GRU host stores dropout (`src/cti_causal_organ_models.py:49-52`, `:191-195`, `:321-325`). |
| 8 | Models/trainer source hashes are verified | PASS | Replacing either live hash oracle with 64 zeroes independently raised the corresponding source-change error. Frozen hashes equal live files: models `0b252b18bc5fb87b472f927fef61826bbb199c17dce93efe643490ee509ef8c7`; trainer `494e836fe918653e93017840f06cabcaa31f6894a0ff5b73bef560b79976c4bb`. |
| 9 | `create_precommit(..., model_config=None)` fails | PASS | Raised `ValueError: model_config is required for precommit creation` before any write. |
| 10 | Remaining correctness issues | FAIL | The three blockers above remain. |

## Artifact and checkpoint inspection

The locked artifact contains the required additions:

- donor: `n_heads: 6`, `dropout: 0.0`;
- Transformer host: `n_heads: 4`, `dropout: 0.0`;
- GRU host: `dropout: 0.0` (GRU has no `n_heads` parameter);
- `models_source_sha256` and `trainer_source_sha256` are present and match live bytes.

Normal `verify_precommit(init_partitions())` passes on the untouched artifact. Checkpoint saves include model, optimizer, Torch RNG, CUDA RNG when applicable, NumPy RNG, scaler, best accuracy, eval RNG, and the precommit integrity hash (`src/cti_causal_organ_train_donor.py:162-186`). The official resume path supplies the current hash (`:290-296`). In-memory probes confirmed that both a mismatched checkpoint hash and a missing checkpoint hash are rejected before model state loading (`:194-208`). Smoke/full checkpoint directories remain separate.

The admission document was checked against the implementation. The reviewed boundary is especially relevant to its locked pre-training partitions (`research/CAUSAL_SKILL_ORGAN_ADMISSION_V1.md:59-75`), donor/host gates (`:179-204`), and protocol-VOID rules for partition/hash drift and withheld-result influence (`:226-236`).

## Final disposition

All fixes claimed for `fe8f437` are present and their direct adversarial checks pass. They are not yet complete enough for GO because the verifier does not bind the effective training invocation or execution environment and still has a strict-JSON type-confusion acceptance path. No training was launched. The locked `results/causal_organ/precommit.json` and all source/research files remained untouched during review.
