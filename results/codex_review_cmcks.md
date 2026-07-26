The CM-CKS implementation is not runnable or scientifically admissible as written. Stage B and Stage C both hit a deterministic runtime failure, and several later defects could produce an invalid PASS.

## Ranked findings

1. **CRITICAL — Both stages crash because `changed_edges` lacks `edge_index`.**

   `paired_key_from_transposition()` emits `{op, state, base_output, partner_output}` ([automaton.py:95](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:95>)), but `counterfactual_edge_crossover()` unconditionally reads `edge["edge_index"]` ([statistics.py:290](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_statistics.py:290>)). Both stages pass the metadata unchanged ([stage_b.py:322](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:322>), [stage_c.py:360](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:360>)).

   I reproduced `KeyError: 'edge_index'`. Stage B fails after its first two expensive installer runs; Stage C fails after training/extracting all 16 teachers and completing the first A/B installation.

2. **CRITICAL — The required wrapped-sequence endpoints do not exist.**

   R8 makes pair success conjunctive over changed direct edges, affected wrapped sequences, and unchanged direct and wrapped outputs ([R8:222](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/steering_dialogue_r8_round2.md:222>)). The implementation evaluates only 48 direct-edge logits. `cm_pair_success()` contains only direct crossover, a direct effect floor, and direct stability ([statistics.py:353](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_statistics.py:353>)).

   Consequently, Stage C could PASS even when the artifact cannot control any affected wrapped sequence or causes arbitrary wrapped-output damage.

3. **CRITICAL — The same-key replay ceiling is recorded but never used as a gate.**

   Stage B calculates only direct-edge TV/argmax drift ([stage_b.py:343](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:343>)). It does not measure wrapped-output drift, compare CM changed-edge effects against replay drift, block Stage C when replay noise is comparable, or freeze replay-derived thresholds. Stage C reads only the winner and coefficient ([stage_c.py:176](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:176>)).

   This directly violates R8 §5 ([R8:272](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/steering_dialogue_r8_round2.md:272>)) and permits a teacher-noise-driven PASS.

4. **HIGH — Stage B proceeds with unqualified teachers.**

   The reused Stage A teacher is checked only for `status == "complete"`, not its capacity gates ([stage_b.py:152](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:152>)). Failed partner gates merely print warnings and execution continues ([stage_b.py:201](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:201>)); the replay teacher is not gated at all.

   Stage C does eventually VOID on teacher failure, but unnecessarily trains every remaining teacher first ([stage_c.py:243](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:243>)), potentially wasting most of the sealed compute budget.

5. **HIGH — Non-finite/degenerate logits can generate an affirmative pair success.**

   `evaluate_direct_edge_logits()` has no shape or finiteness validation ([installer.py:286](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:286>)). The statistics functions likewise accept `NaN`/`Inf`.

   I constructed changed-edge logits containing `+Inf`; the result was `all_crossover=True`, `mean_d=inf`, stable unchanged edges, and `success=True`. NaNs elsewhere become ordinary FAILs rather than the required data-integrity VOID. JSON output may also contain nonstandard `NaN`/`Infinity`.

6. **HIGH — Two-seed aggregation and aggregate gates do not follow R8.**

   R8 says to average two student seeds and two changed edges inside each pair ([R8:230](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/steering_dialogue_r8_round2.md:230>)). Stage C instead requires both seeds to pass individually ([stage_c.py:376](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:376>)), creating an unintended stricter rule and false-negative risk.

   Conversely, `cm_cks_verdict()` never applies R8’s aggregate effect/stability floors—it merely reports their means ([statistics.py:414](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_statistics.py:414>)). Seven acceptable pairs plus one arbitrarily catastrophic pair can therefore PASS.

7. **HIGH — The verdict does not enforce exactly eight independent pairs.**

   The 7/8 calculation itself is correct: 7 successes gives `9/256 = 0.03515625`, while 6 gives `37/256`. However, for other sample sizes the threshold uses floor truncation, `int(n * .875)` ([statistics.py:388](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_statistics.py:388>)), and `cm_cks_verdict()` never requires `n == 8`.

   I confirmed that seven successful pair records produce a CM verdict of PASS. Run completeness is also based on a counter incremented before each call, rather than verified summaries and artifacts ([stage_c.py:321](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:321>)).

8. **HIGH — Pair construction, commitments, and calibration hashes are not actually verified.**

   The transposition helper does not validate that the base lists have exactly 12 entries drawn from `0..11`, nor calculate/assert identical calibration-set hashes as R8 requires. It only checks that the partner has 12 unique values ([automaton.py:91](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:91>)).

   Stage C hardcodes `"calibration_hashes_match": True` ([stage_c.py:408](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:408>)). On resume, it trusts secret files and stored metadata without regenerating keys, recomputing hashes, or comparing them with the commitment manifest ([stage_c.py:219](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:219>)). Corruption or stale files can therefore evade the intended VOID checks.

9. **MEDIUM — Empty-list interfaces are inconsistent and fail late.**

   `withheld_examples=[]` is safe: unsealed runs report `final_withheld_acc=0.0`; sealed runs do not evaluate it. Stage B deliberately relies on this behavior.

   A genuinely empty `direct_probes=[]` is not supported. At the first probe interval—after 250 training steps—`centroid_probe()` calls `collate_fn([])`, whose `max(...)` raises `ValueError: max() iterable argument is empty` ([automaton.py:230](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:230>)). The current stages positionally pass nonempty direct-edge lists, so they avoid this particular crash.

10. **MEDIUM — Sealed-probe handling is only partially compliant.**

    Stage C correctly passes `seal_probes=True` for both arms ([stage_c.py:323](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:323>)), and probe fields are excluded from ordinary summaries.

    However, R8 requires a hash chain or final hash. No hash is produced. Logs are opened in append mode ([installer.py:358](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:358>)), so a retry after partial failure mixes old and new sealed entries without integrity metadata.

11. **MEDIUM — Stage B exceeds the frozen compute design and does not use one development seed.**

    R8 budgets two same-key noise-control installations. Stage B runs original/replay for both candidates—four installations ([stage_b.py:343](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:343>)). It also assigns different student seeds to the two partners ([stage_b.py:283](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:283>)), despite the specification calling for one paired development seed.

12. **LOW — Exception cleanup is not robust.**

    Installer and trainer logs use manual `open()`/`close()` without `try/finally`; CUDA OOM, empty-probe failure, or another exception can leave file handles open and the trainer’s monkey-patched `RESULTS_DIR` unrestored. I found no persistent GPU leak on the normal success path: models fall out of scope and explicit evaluation/extraction models are deleted. CUDA allocator caching should not be mistaken for a leak.

## Confirmed correct

- Direct-edge indexing is correct: generation is state-major/op-minor and `state * 4 + op_index` matches all 48 entries.
- The exact 7/8 tail probability is implemented correctly for exactly eight inputs.
- A/B use the same winner arm, paired seed, frozen coefficient, bank order, and identical calibration data.
- Stage C does set `seal_probes=True`.

No files were changed.

