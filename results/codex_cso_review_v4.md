# Verdict: NO-GO

At commit `432069e`, donor training is now mechanically runnable, but the frozen-contract and resume guarantees remain insufficient for protocol-valid capacity training.

Tally: **8 FIXED, 5 NOT_FIXED** across the original findings.

## Original 13 findings

| # | Status | Verification |
|---:|---|---|
| 1 | **FIXED** | Zero excluded-bigram or withheld-trigram violations in 300 training examples. |
| 2 | **NOT_FIXED** | All 300 full-intersection examples correctly contained a held-out state, long length, excluded bigram, and withheld trigram. However, the protocol-required counterfactual-suffix stratum is still absent from both evaluation registries. Full intersection is not a substitute for counterfactual suffixes. |
| 3 | **NOT_FIXED** | Normal verification passes and a changed live model parameter is rejected. But verification accepted a changed explicit included-bigram list, changed socket seed, and an extra model-config field. The integrity digest also excludes model configuration and much top-level contract metadata. |
| 4 | **FIXED** | The million-element lower bound is frozen, called by full verification, and asserted before success. I did not rerun the million-element BFS under the bounded-test constraint. |
| 5 | **FIXED** | Live configuration matches the precommit: donor `35,420,160` step MACs, Host T `2,368,512`, ratio `6.6869%`. |
| 6 | **FIXED** | Two length-32 CUDA examples produced finite states/logits/gradients; state RMS was `0.999999`. |
| 7 | **FIXED** | Both hosts consume `organ.read(organ_state)`. |
| 8 | **NOT_FIXED** | Socket construction is deterministic, RNG-preserving, frozen, and the Transformer socket matches the committed hash. However, verification accepts a changed seed, never recomputes the live hash, and the GRU `32→256` socket has no committed hash. |
| 9 | **FIXED** | Generation uses immutable sets from the supplied partitions and bounded retries; the 300-example probe passed. |
| 10 | **NOT_FIXED** | TransformerHost now rejects holey masks. But `CausalOrgan.forward()` still accepts no mask, and an all-false GRU mask processes a fake padded operation: changing only padding changed the output by up to `0.596`. |
| 11 | **FIXED** | The probabilistic split is documented and hashes use fixed-width big-endian encodings. The inaccurate encoding label is reported below. |
| 12 | **FIXED** | All live dimensions, parameter counts, MACs, compute ratio, and the `61,110`-byte organ size match the precommit. |
| 13 | **NOT_FIXED** | Direct `CausalOrgan(d_hidden=54)` construction correctly rejects its `70,326`-byte serialization. Canonical organ artifact hashing and identical-byte verification at both installations still do not exist. |

## Three new v3 issues

| Issue | Status | Verification |
|---|---|---|
| `make_batch()` TypeError | **FIXED** | Training generation returns the expected dictionary; an eight-example batch constructed successfully. |
| Smoke/full checkpoint contamination | **FIXED** | Smoke and full runs use distinct checkpoint directories. |
| Fresh/resumed reproducibility | **NOT_FIXED** | Torch CPU, CUDA, and NumPy RNG streams restored exactly, and `TORCH_SEED=2026` is frozen. But the checkpoint identity is never checked during load, AMP `GradScaler` state is omitted, and evaluation RNG/best-run state resets on resume. |

## New issues found

- **HIGH:** `load_checkpoint()` accepts a checkpoint carrying an arbitrary precommit identity because it has no expected-hash comparison.
- **HIGH:** AMP scaler state is not checkpointed. A resumed CUDA run can therefore use a different loss scale and diverge from an uninterrupted run.
- **MEDIUM:** Evaluation RNG, `best_acc`, and related evaluation state restart after resume, so evaluation history and `donor_best.pt` are not reproducible.
- **MEDIUM:** The precommit declares `state_dtype: int64_be`, but state indices are actually hashed with `struct.pack(">I")`, i.e. unsigned 32-bit big-endian.
- **MEDIUM:** `verify_group_order()` uses `list.pop(0)` inside a million-element BFS, producing linear-time queue removals. A `deque` is needed before treating this as a practical repeatable gate.
- **MEDIUM:** The built-in data-generation verifier checks execution correctness and stratum labels, but not each evaluation stratum’s semantic constraints.

Relevant surfaces: [transducer](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py>), [models](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py>), [trainer](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_train_donor.py>), [precommit](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/causal_organ/precommit.json>), and [locked protocol](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/CAUSAL_SKILL_ORGAN_ADMISSION_V1.md>).

No files were changed by this review. During testing, uncommitted edits appeared concurrently in the transducer, donor trainer, and v3 review. They were not part of commit `432069e` and were not credited in this verdict.