# Atlas R2.4 Protocol Amendment

Effective: 2026-07-28. Amends R2.2 and carries forward R2.3.

## Trigger

The R2.2 runner design gate (results/codex_r2_2_runner_design_gate.md)
identified three protocol-blocking implementation conflicts:

1. Invalid-output harm polarity is reversed in the scorer.
2. Sealed panels contain only integer references; the protocol states
   scoring uses unrounded PolicyEngine references.
3. `generate()` cannot distinguish EOS from cap hit and provides no
   hard outer watchdog.

The R2.4 amendment ruling (results/codex_r2_4_amendment_ruling.md)
specifies the exact resolution for each conflict.

## Change 1 --- Scorer bug fix

In `score_r2_2_household()` (src/cti_atlas_workloads.py), the
invalid-output branch (parsed_array is None):

- `harm` changes from `not zero_correct` to `zero_correct`.
  Section 5.2 of R2.2 states: "A schema-invalid household makes every
  field candidate-incorrect, so it can create harms and cannot create
  rescues." For invalid output, candidate is always wrong; harm is
  true when zero_baseline_correct is true.

- Invalid amount fields receive `normalized_magnitude_error = 10`.
  This is the NME cap used in `score_r2_2_field()`.

- Invalid eligibility fields receive
  `normalized_magnitude_error = null`.

- `R2_2_SCORER_VERSION` changes from `"r2.2.0"` to `"r2.4.0"`.

No other scorer logic changes. `score_r2_2_field()`,
`parse_r2_2_output()`, `format_r2_2_prompt()`, and
`_strip_think_block()` are unchanged.

## Change 2 --- Reference clarification (protocol text only)

R2.2 Section 1.1 states: "The scorer still compares the integer
prediction to the unrounded PolicyEngine reference."

This is replaced by: "The scorer compares the integer prediction to
the stored integer reference in the sealed panel. Amount references
are PolicyEngine floats rounded to the nearest integer with half
values away from zero; eligibility references are stored as integer
0 or 1."

Verified: all 6590 gold values across both sealed panels are native
JSON integers. No `refs_dict` field exists in any household. No
unrounded references are available in any sealed artifact.

No code changes. This corrects the protocol to match the sealed
artifact reality.

## Change 3 --- Hard watchdog and stop semantics

### 3a. Stop reason fields

`generate()` returns three new fields:

- `eos_reached`: bool --- last output token matches EOS token ID.
- `cap_hit`: bool --- output token count >= max_new_tokens AND not
  eos_reached.
- `stop_reason`: str --- exactly one of `"eos"`, `"cap_hit"`,
  `"watchdog_abort"`, `"empty"`.

These are mutually exclusive. The existing `timed_out` and `is_empty`
fields are preserved for backward compatibility.

### 3b. Cooperative timer fix

The cooperative `_TimeLimit` stopping criterion now starts its timer
BEFORE tokenization (at `_TimeLimit` creation time), not lazily on
first call. This ensures tokenization time counts toward the deadline.

### 3c. Supervised worker process

`SupervisedWorker` in `src/cti_atlas_inference.py` provides
process-isolated inference with hard watchdog kill:

- Model loads in a child process via `multiprocessing` (spawn).
- Supervisor timer starts BEFORE the generation request is sent
  (covers tokenization).
- If no response within `watchdog_seconds`, the child process is
  killed via `Process.kill()`.
- Synthesized result: `watchdog_abort=true`, charged at exactly
  `watchdog_seconds`.
- Worker can be respawned after a kill (model reloads).
- The cooperative `_TimeLimit` remains as an optimization within
  the child process but is not authoritative.

The R2.4 runner MUST use `SupervisedWorker` for all W-D3 execution.

## Propagation

- Protocol revision in task records: `r2.4`
- Panel revision: `r2.2` (unchanged)
- W-D2 records: `r2.1` (unchanged)
- Scorer version: `r2.4.0`
- Task record paths: `cti_atlas_r2_r2.4_P1_W-D3_{system_id}.json`
- Contract fingerprint binds: R2.4 protocol hash, scorer hash,
  inference hash, runner hash, and unchanged panel hashes.

## Frozen boundary

Under R2.4, all items frozen by R2.3 remain frozen:

- P/C panel bytes, hashes, identities, household selection
- Smoke salts and selected households
- `format_r2_2_prompt`, `_strip_think_block`, `parse_r2_2_output`
- `score_r2_2_field` (only the invalid-output branch of
  `score_r2_2_household` is corrected)
- Scoring thresholds ($50/5%), bootstrap seed 2204, bootstrap
  count 10000, strata definitions, ESS rules
- Token cap 448, watchdog 120s, latency floor 30s
- Per-system budget 3.00 GPU-hours, aggregate 27.00 GPU-hours
- Temperature 0, no format-repair retry, no W-D3 API execution

## Carries forward from R2.3

- Local generation cap: 448 max_new_tokens
- Preflight limit: 416 tokens
- M_eval: 353 tokens (argmax case)
