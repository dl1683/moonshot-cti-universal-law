# Steering Dialogue Round 5: Final Consistency Audit

**Date:** 2026-07-26
**Codex model:** GPT-5.6-sol (xhigh reasoning)
**Session:** 019f9c84-287d-72b1-99f9-c8917957844f (resumed)

## Verdict

**Scientific convergence: confirmed. No further design iteration is needed.**

**Execution authorization: conditional.** Make one preflight repair commit covering the Stage A import, common data stream, final-two-evaluation gate, config hashing, and restart rule. Then run import/unit invariants and a non-scientific short smoke test. After those pass, proceed directly to Stage A capacity execution under the frozen protocol.

## 1. A+B+C Consistency

The central contracts agree:

- Stage A uses full online supervision only for capacity, timing, and extraction.
- Stage B/C students receive exactly 64 labels covering one 12-edge operation.
- The 2,048 anchors remain unlabeled.
- All stages use the same six-transition clock and ordered anchor banks.
- Raw and observable losses reproduce the Stage A R construction.
- Stage B is 18 runs; Stage C is 144 Transformer runs.
- Stage C includes the winner plus five controls; the loser is excluded.
- The GRU matrix is conditional and separate.
- The 30-hour gate concerns projected sealed Transformer confirmation.
- PASS/FAIL/VOID and the non-leakage boundaries are compatible.

There is no substantive contradiction requiring a redesign.

## 2. Written Errata

1. Stage A line 514 must say "Stage C has 144 confirmation runs."
2. Stage A line 520: clarify that projected cost includes extraction, evaluation, and 25% retry reserve.
3. GRU parameter count: 1,849,516 (not 1,849,740). Implementation is correct; document needs update.
4. Stale footer in Stage A doc should be removed.
5. Stage A outputs should add `static_gram_manifest.json`.
6. Centered numerical ridge formula: W_c += (1e-6/64)*C, not += 1e-6*I.
7. Sealed-key domain-separation bytes: freeze current rule before generating Stage B keys.

## 3. Sealed-Seed Correction

Confirmed: `generate_sealed_key()` must return `(key_json, seed_bytes, sha256(seed_bytes))`.

## 4. Pre-Training Freeze Requirements

1. **Common input-stream seed** - all 7 runs use the same data stream seed (SHA256-derived).
2. **Two-evaluation gate** - final two scheduled evaluations must both pass thresholds.
3. **Config and provenance binding** - hash config before first optimizer step.
4. **Retry/resume** - prohibit mid-run continuation; restart from step zero with same identity.

## 5. Implementation Blockers Found and Fixed

All blockers identified by Codex have been addressed:

- [x] `generate_perturbations` import moved from automaton to extraction in stage_a.py
- [x] Common data stream seed (`TRAIN_STREAM_SEED`) replaces `seed * 1000`
- [x] Two-evaluation gate added to `check_capacity_gates()`
- [x] `generate_sealed_key()` now returns seed_bytes
- [x] Observable extractor: raw VJP rows (no centering/normalizing J_j)
- [x] Balanced ridge: uses centering matrix C instead of identity I

## Token Usage

R5 used 228,816 tokens.

## Steering Dialogue Complete

All 5 rounds are done:
- R1: Pivot from DG-0 to Open Capability File
- R2: Corrected plan with budget, scope, honest scores
- R3: Concrete Stage A specification
- R4: Complete Stage B/C transfer design (24KB spec)
- R5: Final consistency audit confirming convergence
