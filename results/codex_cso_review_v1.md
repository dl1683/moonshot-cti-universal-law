# Tier 1 verdict: NO-GO

Do not start donor capacity training yet. The simulator’s primitive operations are correct and GPU memory is safe, but the current data generator violates withheld-trigram requirements, the required evaluation distributions do not exist, the partition precommit is absent, and the mandatory group-order property is not actually verified.

## Confirmed correct

- All eight scalar and vectorized operations match the locked rules.
- All eight inverses recover all 65,536 states exactly.
- All operations are bijective.
- Noncommutativity meets the protocol’s stated minimum: 23/28 operation pairs are noncommuting for at least one state.
- State encoding/decoding passed all 65,536 round trips.
- The donor processes operations sequentially through one recurrent state.
- No intermediate or final ground-truth register state enters a model input. Initial registers are legitimate task inputs.
- Default organ: 32-dimensional state, 13,936 total parameters, 60,284-byte FP32 `state_dict`; it currently fits the 64 KiB limit.

## Findings

1. **CRITICAL — Withheld trigrams are present in training**  
   [transducer.py:157](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:157>), [transducer.py:188](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:188>)

   Training rejects excluded bigrams but never rejects the 32 withheld trigrams. Of those 32 trigrams, **19 have both constituent bigrams included**, so they are fully legal training sequences and will eventually be sampled.

   **Fix:** Pass a frozen withheld-trigram set into the generator and reject any candidate that completes one. Add a high-volume test asserting zero withheld-trigram occurrences in training.

2. **CRITICAL — Required evaluation splits are not generated**  
   [transducer.py:203](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:203>), [protocol.md:67](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/CAUSAL_SKILL_ORGAN_ADMISSION_V1.md:67>)

   `generate_eval_example()` samples unrestricted random programs. It does not guarantee:

   - An excluded bigram.
   - A withheld trigram.
   - The full withheld intersection.
   - Counterfactual suffix pairs.

   Random examples cannot support separate excluded-composition, unseen-trigram, or full-intersection accuracy claims.

   **Fix:** Implement explicit generators for length extrapolation, excluded-bigram, withheld-trigram, full intersection, and counterfactual suffix strata. Each example should carry auditable split metadata and have constraint assertions.

3. **CRITICAL — Required partition precommit is absent and unenforced**  
   [transducer.py:240](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:240>), [protocol.md:74](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/CAUSAL_SKILL_ORGAN_ADMISSION_V1.md:74>)

   The code calculates hashes, but the current checkout has no `results/causal_organ/precommit.json`. Training also has no load-and-verify barrier comparing live partitions against frozen hashes.

   **Fix:** Write an atomic precommit containing canonical partition hashes, exact lists/configuration, generator code hash, dtype/encoding, seed, and model configuration. Training must refuse to start unless it verifies this immutable artifact.

4. **CRITICAL — Group order is not verified, despite “ALL PASS”**  
   [transducer.py:399](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:399>), [transducer.py:514](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:514>)

   The protocol requires group order `>> 65,536`. The code instead measures the orbit of `(1,2,3,4)`, which is not group order. It reaches 61,440 states and accepts that without an assertion. `run_full_verification()` does not call this check before printing `ALL VERIFICATION CHECKS PASSED`.

   The transformations are linear and fix zero, so the claimed transitive action on all `Z_16^4` cannot hold.

   **Fix:** Represent generators as `4×4` matrices over `Z_16` and compute or rigorously lower-bound the generated matrix-group order. Assert the locked threshold inside `run_full_verification()` before printing success.

5. **CRITICAL — Host T cannot meet the ≤10% inference-compute gate as configured**  
   [models.py:164](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:164>), [protocol.md:203](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/CAUSAL_SKILL_ORGAN_ADMISSION_V1.md:203>)

   Approximate per-step core MACs:

   - Donor: about 35.4M.
   - Host T with three tokens—host state, operation, organ: about 6.7M.
   - Ratio: approximately **19%**, before miscellaneous overhead.
   - Even without the organ token, the Host T core is approximately 12.5%.

   Therefore one mandatory admission criterion is structurally unreachable.

   **Fix:** Shrink and freeze Host T before training. For example, `d_model=128`, four layers and `d_state≈64` should move the core below 10%; verify with a precommitted FLOP counter.

6. **HIGH — Donor recurrent state is numerically unstable at initialization**  
   [models.py:101](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:101>)

   The unnormalized `state + new_state` recurrence grows rapidly:

   | Length | State RMS | Logit RMS |
   |---:|---:|---:|
   | 0 | 0.20 | 0.05 |
   | 12 | 9.26 | 2.25 |
   | 32 | 75.07 | 18.47 |

   This creates a serious length-extrapolation and gradient-stability risk.

   **Fix:** Use a normalized or gated recurrent update, such as `LayerNorm(state + α·update)` or a GRU-style interpolation. Add finite-value, state-norm, logit-scale, and gradient-norm tests through length 32.

7. **HIGH — Organ readout is dead code; hosts consume raw state**  
   [models.py:251](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:251>), [models.py:339](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:339>), [models.py:408](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:408>)

   The protocol defines `m_t = Gφ(z_t)`, but neither host calls `organ.read()`. Both socket raw `z_t`, leaving `readout` unused and violating the specified transition/message interface.

   **Fix:** Define one explicit lifecycle—initialize, transition, read message—and make both hosts consume `organ.read(organ_state)` at the same point in every step.

8. **HIGH — Socket-training policy is unresolved**  
   [models.py:189](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:189>), [models.py:300](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:300>)

   Both socket projections are trainable host-specific layers. Hosts trained without an organ cannot train these paths; training them after installation could trigger the protocol’s “task-specific socket training” kill criterion.

   **Fix:** Predeclare a compliant socket strategy before any capacity work: fixed parameter-free projections, or a clearly task-independent generic socket-training phase. Freeze and hash the resulting socket contract.

9. **HIGH — Training generator ignores its `included_bigrams` argument**  
   [transducer.py:171](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:171>), [transducer.py:218](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:218>)

   The argument is unused; generation relies on mutable global `_INCLUDED_BIGRAM_SET`. Passing an empty list still produced a six-operation example using the globally initialized partition. Before initialization, generation can loop forever.

   **Fix:** Remove the global and use an immutable set derived from the supplied partition. Validate that every preceding operation has at least one permitted successor.

10. **MEDIUM — Mask handling is inconsistent and assumes contiguous right padding**  
    [models.py:255](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:255>), [models.py:339](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:339>), [models.py:412](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:412>)

    Hosts advance organ state on masked instructions. GRU final-state selection uses `mask.sum()-1`, which is wrong for masks containing holes. `CausalOrgan.forward()` has no mask at all.

    **Fix:** Mask organ transitions, require and assert contiguous right-padding, and use packed GRU sequences or the actual last-true index.

11. **MEDIUM — The “75%” split is not exactly 75%, and hashes are platform-encoded**  
    [transducer.py:124](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:124>), [transducer.py:244](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_register_transducer.py:244>)

    The threshold split produces 49,062 training states rather than 49,152: 74.86%, not exactly 75%. Array hashes use native-endian `int64.tobytes()`, making the committed hash representation platform-dependent.

    **Fix:** Sort states by canonical SHA-256 rank and take exactly 49,152, or explicitly lock the probabilistic-threshold interpretation. Hash a fixed-width, fixed-endian representation.

12. **MEDIUM — Actual model sizes materially differ from the locked descriptions**  
    [models.py:43](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:43>), [models.py:164](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:164>), [models.py:277](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:277>)

    Actual defaults:

    - Donor: 18,862,784, described as 19.5M.
    - Host T: 2,562,496, described as 1.9M.
    - Host G: 1,342,976, described as 1.85M.

    These are too divergent for matched-size/compute comparisons to remain informal.

    **Fix:** Retune defaults and freeze exact parameter counts, dimensions and FLOPs in the precommit.

13. **MEDIUM — Organ limits are descriptive, not enforced**  
    [models.py:373](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:373>), [models.py:449](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_causal_organ_models.py:449>)

    Defaults pass, but callers may construct `d_state > 32` or enlarge `d_hidden`; no assertion prevents an invalid artifact. Quantization, canonical serialization, byte hashing and identical-host-byte verification are not implemented.

    **Fix:** Enforce state/parameter limits in construction and extraction, serialize canonically, measure the actual artifact file, and verify its SHA-256 before both installations.

## Performance assessment

OOM is not a concern.

Measured locally on the available RTX 5090 Laptop GPU with 24,463 MiB VRAM, batch 128, length 32, four-head cross-entropy, backward, AdamW state creation and optimizer step:

- BF16 autocast: 1,323 MiB peak allocated; 1,618 MiB peak reserved.
- FP32: 2,259 MiB peak allocated; 2,310 MiB peak reserved.

Actual training lengths stop at 12, so training memory should be lower. Gradient checkpointing is unnecessary.

The main GPU concern is throughput and kernel efficiency: length 32 invokes 320 tiny Transformer-layer applications through a Python recurrence. Mixed-length padded batches also execute every step for every example. Bucket batches by length or process only active rows.

CPU generation is not currently a bottleneck: measured approximately 50,000 training examples/s and 40,000 evaluation examples/s, far above observed GPU consumption. Batched transition tables could optimize it later, but are unnecessary before the correctness repairs.

## Gate to GO

At minimum, before donor training:

1. Exclude withheld trigrams from training.
2. Implement auditable evaluation strata.
3. Freeze and enforce the immutable precommit.
4. Replace the false group-order check with a real asserted verification.
5. Stabilize the donor recurrence through length 32.
6. Resolve the Host T compute budget and socket contract now, before spending donor GPU hours against an admission design that cannot pass.

