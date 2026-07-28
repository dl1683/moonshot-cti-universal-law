# Atlas R2 Protocol Amendment R2.2

Date frozen: 2026-07-28

Status: FROZEN DESIGN GATE. No R2.2 model or API execution is authorized
until the implementation, manifests, and fail-closed verifier conform to this
document.

Supersedes only:

- Section 3 (PolicyBench design), the W-D3 fields in Section 4, the short-task
  timeout interpretation in Section 7, and the Gate A aggregation sentence in
  Section 8 of `precommit/atlas_r2_protocol_r2_1.md`;
- the W-D3 explanation-validity phrase in Section 6 of
  `results/codex_scale_inversion_atlas_design_gate_r2.md`; and
- the immediate W-D2 API execution contract, which R2/R2.1 rostered but did not
  specify tightly enough to execute.

Everything else in R2 and R2.1 remains binding.

## 0. Binding disposition

1. The completed R2.1 Gate A output is immutable:
   `results/atlas_r2_gate_a_output.json`, SHA-256
   `41283d943c69693b8ce3623692b1005fdad644250e06aafce54375ae4939b401`.
2. Its six anchors and three exploratory systems are not reselected or
   relabeled by R2.2.
3. All R2.1 W-D3 records remain
   `diagnostic_r2.1_deviation_384_120`, with 0.00 Gate A weight.
4. No R2.1 W-D3 output may be merged with an R2.2 cell. R2.2 task records use
   protocol revision `r2.2`, new task IDs, and separate files.
5. W-D2 data, scoring, Gate A selection, and local-system quality anchors are
   unchanged.
6. R2.2 W-D3 is the structured calculation task
   `structured household JSON -> complete PolicyEngine output vector`. It does
   not test unstructured intake or natural-language explanation. Those require
   a separate workload and protocol.

The calibration inputs frozen from R2.1 are:

| Artifact | SHA-256 |
|---|---|
| `data/policybench/scenarios.csv` | `f3d5e4d8c80949f50e639e7e80696c6c9c64a2122ab050aca2ee93c21b2747fb` |
| `data/policybench/reference_outputs.csv` | `da51998841f41a7794c23d818a276aee53cd94767fbd08030d9ee293041e7aac` |

They contain 100 households and are calibration-only under R2.2. They receive
no R2.2 evaluation weight.

## 1. Compact W-D3 output contract

### 1.1 Canonical format

For each household, sort the required field names by ascending UTF-8 byte
order. The prompt lists the resulting zero-based mapping:

```text
0=child1_chip_eligible
1=child1_early_head_start_eligible
...
n-1=tanf
```

The answer is exactly one JSON array with exactly `n` entries in that order:

```json
[0,1,0,12434,0,2163]
```

The prompt says:

```text
Return exactly one minified JSON array and nothing else.
The array must contain exactly <n> entries in the indexed order above.
Eligibility entries must be integer 0 or 1.
Dollar entries must be signed integer dollars, rounded to the nearest dollar.
Do not emit keys, prose, Markdown, null, NaN, Infinity, or an explanation.
```

Rounding is nearest integer with half values away from zero. The scorer still
compares the integer prediction to the unrounded PolicyEngine reference.

The parser is fail-closed:

- strip only the model-family think block already removed by R2.1;
- parse the entire remaining string as one JSON value;
- allow RFC 8259 whitespace, but no fence, prefix, suffix, or regex extraction;
- require a JSON array, exact length, and native integer entries;
- reject booleans, floats, strings, nulls, NaN, Infinity, and out-of-domain
  eligibility values;
- do not salvage a prefix or score returned fields when the household schema
  is invalid.

An invalid household is incomplete and every field is scored incorrect.

### 1.2 Token cap

The R2.2 W-D3 generation cap is exactly `256 max_new_tokens` for every local
and API model. Temperature remains 0. No best-of-n or format-repair retry is
allowed in the raw W-D3 cell.

This cap is feasible, not guessed. A deterministic audit on the 100 frozen
calibration households found:

| Quantity | Result |
|---|---:|
| Required fields per household | 16 to 56 |
| Canonical minified-array tokens, median | 42 |
| Canonical minified-array tokens, p95 | 78 |
| Canonical minified-array tokens, maximum | 131 |
| JSON array with spaces, maximum | 186 |
| R2.2 cap | 256 |

The maximum was 131 tokens under each of the nine pinned Qwen3, Gemma3, and
Falcon-H1 tokenizers. Thus 256 is 1.95 times the observed canonical maximum and
1.38 times the observed whitespace-JSON maximum.

Before model execution, the sealed R2.2 prevalence and challenge gold arrays
must be tokenized by all nine pinned tokenizers. The preflight passes only if
every canonical gold array is at most 224 tokens, leaving at least 32 tokens of
headroom. If this fails, do not raise the cap: mark `W-D3_R2.2_SCHEMA_INVALID`,
keep W-D3 at zero downstream weight, and require R2.3.

### 1.3 R2.2 W-D3 task records

In addition to the unchanged R2.1 provenance, model, energy, and cost fields,
every household record must contain:

- `protocol_revision="r2.2"`, `panel_id` in `{P,C}`, and challenge stratum or
  null;
- canonical household-identity hash, task ID, scenario ID, and gold hash;
- exact ordered field-name list, its SHA-256, and expected array length;
- raw output, think-stripped output, output-token count, EOS flag, and
  `cap_hit` (generation stopped by 256 rather than EOS);
- `user_wall_seconds`, `latency_floor_met`, `watchdog_abort`, retry count, and
  retry reason;
- `schema_valid`, one enumerated schema-error code, and
  `qualifying_completion`;
- for every field: type, reference, parsed prediction or null, correctness,
  zero-baseline correctness, rescue flag, harm flag, and normalized magnitude
  error when applicable; and
- scorer version, exact model/package revision, run ID, segment ID, billed
  usage, dollars, GPU seconds, and allocated energy as applicable.

Allowed schema-error codes are exactly `NOT_JSON`, `NOT_ARRAY`,
`WRONG_LENGTH`, `NON_INTEGER`, `NONFINITE`, `BAD_ELIGIBILITY`, `EXTRA_TEXT`,
and `EMPTY`. Summary files remain derived; resume reads these task records.

## 2. Watchdog and user-facing latency

R2.1 conflated an execution stop with a procurement constraint. R2.2 freezes
two independent quantities:

- Outer watchdog: 120.000 wall seconds per attempt.
- User-facing latency floor: nearest-rank p95 end-to-end wall time at or below
  30.000 seconds.

At 30 seconds, the request is not killed. It is marked late and may continue
until completion, the 256-token cap, or the 120-second watchdog. A late valid
answer is scored for capability, but it does not count as an on-time
completion. A watchdog abort is right-censored at 120 seconds and is not
reported as a 30-second timeout.

`user_wall_seconds` starts immediately before request submission or local
generation and ends after strict schema validation. It includes queueing,
backoff, and any retry. Model/engine load is recorded separately and amortized,
as in R2.1.

Exactly one retry is allowed only for an explicit infrastructure error:
provider HTTP 408, 429, or 5xx; connection reset; CUDA driver failure; or host
process crash. The retry uses identical prompt bytes and generation settings.
Provider `Retry-After` is obeyed with a minimum of 2 seconds and maximum of 60
seconds. A malformed, empty, cap-hit, slow, or low-quality answer is not
retryable. The failed attempt, wait, and retry all remain in cost, reliability,
and end-to-end latency.

For local R2.2 W-D3, each system has a 2.50 GPU-hour cumulative generation cap
over its 180 households. Before every original or retry attempt, the runner
reserves the full 120-second watchdog allowance and starts only if
`spent + 120 seconds <= 2.50 hours`. If it cannot reserve an attempt, the cell
is `INCOMPLETE_BUDGET`, hence ineligible. The 120-second per-attempt watchdog
remains binding. The nine-system R2.2 generation maximum is therefore exactly
22.50 GPU-hours. The freeze-time P1 ledger contains 6.1442 GPU-hours, leaving
25.3558 of the 31.50-hour P1 allocation, so this cap requires no reserve
borrowing. Smoke tasks are a subset of the 180 and add no separate cap.

## 3. Baseline candidates and system-neutral cost

The only R2.2 W-D3 baseline candidates are:

### 3.1 `all_zero_r2_2`

- Emit an integer zero for every required field.
- No calibration data.
- Candidate boundary: indexed-field adapter, deterministic vector generator,
  strict validator, monitoring, and deployment wrapper.

### 3.2 `field_prior_r2_2`

- Fit only on the 100 frozen R2.1 calibration households in Section 0.
- For each exact field name, binary prior is the observed mode; ties resolve to
  0.
- Numeric prior is the observed median among calibration households containing
  that field, rounded half away from zero to integer dollars.
- An unseen field falls back to 0.
- Freeze the resulting field-to-value map and its SHA-256 before either R2.2
  evaluation panel is generated.
- Candidate boundary includes recomputing/acquiring the 100 PolicyEngine
  calibration labels, prior compilation, indexed-field adapter, strict
  validator, monitoring, and deployment wrapper.

### 3.3 `policyengine_us_1_723_0`

- `policybench==2.0.0`
- `policyengine-us==1.723.0`
- `policyengine-core==3.30.3`
- Python 3.13.7
- Execute the pinned US 2026 policy calculation from the structured scenario
  JSON and serialize the requested fields in the same indexed array.
- No LLM, retrieval, explanation generator, or human fallback.
- Any input-mapping, engine, or serialization failure is an incomplete task.
- Candidate boundary includes package acquisition/license, version pinning,
  scenario-to-PolicyEngine input mapping, validation, deployment, monitoring,
  and maintenance. Historical PolicyEngine R&D is excluded, just as model
  pretraining is excluded.
- A license audit is mandatory. Unknown or incompatible deployment license
  makes the candidate ineligible; it is not silently costed at zero.

### 3.4 Cost equation

Neural and non-neural candidates use the same equation. For a target of `V`
qualifying completed tasks:

```text
C_per_completed(s,V) =
  [F_common + F_candidate(s)] / V
  + E(c_run + c_retry + c_ops + c_review)
    / p_qualifying_completion(s)
```

If a system fails any hard quality, safety, completion, or latency floor,
selector cost is infinity even though its observed accounting cost is still
reported.

Frozen accounting rules:

- Primary volumes: 1,000, 10,000, and 100,000 qualifying completed tasks over
  one year.
- Primary engineering labor rate: USD 100/hour. Sensitivity rates: USD
  50/hour and USD 200/hour.
- Active deployment labor is logged prospectively in 15-minute increments
  under acquisition/license, integration, input mapping, version pinning,
  validation, deployment, compliance, or maintenance. Benchmark research,
  paper writing, and upstream R&D are excluded.
- `F_common` is the measured workload ingestion/schema/validator labor any
  deployer must incur. Add the same full amount to every hypothetical
  single-system deployment.
- `F_candidate` is candidate-specific measured labor, license/acquisition
  charge, and calibration compute. Field-prior calibration is charged;
  all-zero has none.
- One-year maintenance is exactly 20% of the deployment, integration,
  validation, and version-pinning labor dollars.
- CPU runtime is charged at USD 0.5078/hour: USD 0.50/hour capital plus a
  frozen 65 W at USD 0.12/kWh.
- `c_ops` is exactly 10% of `c_run + c_retry`.
- `c_review=0` for these three candidates. Abstention or manual escalation is
  an incomplete task, not free review.
- All attempts, failures, and allowed retries are charged.
- Report actual invoices and measured runtime as well as the frozen cost model.
- A headline cost ordering that reverses between the USD 50 and USD 200 labor
  sensitivities is labeled `LABOR_SENSITIVE`.

## 4. R2.2 prevalence and challenge panels

### 4.1 Calibration, prevalence, and challenge separation

The three data roles are disjoint:

1. `C0`: the existing 100 R2.1 households, calibration only.
2. `P`: 100 new natural-prevalence households.
3. `C`: 80 new challenge households, exactly 20 in each of four strata.

Use `policybench==2.0.0`, country `us`, program set `headline`, and tax year
2026.

The required implementation entry point is:

```text
python scripts/cti_build_atlas_r2_2_panels.py \
  --calibration-manifest data/policybench/scenarios.csv \
  --calibration-references data/policybench/reference_outputs.csv \
  --prevalence-n 100 --prevalence-seed 2201 \
  --challenge-pool-n 1000 --challenge-fallback-n 2000 \
  --challenge-seed 2202 --challenge-per-stratum 20 \
  --country us --program-set headline --year 2026
```

The script does not exist at freeze time; the implementation gate in Section 8
must add it and bind all arguments above into the sealed manifest.

Generate `P` with seed 2201 while excluding `C0`. Generate a 1,000-household
challenge candidate pool with seed 2202 while excluding `C0` and `P`. Identity
is the canonical tuple `(source_dataset, dataset_year, household_id,
tax_unit_id, country)`, not the generated `scenario_id`.

If the 1,000-household pool has fewer than 20 eligible unique households in
any stratum, regenerate once at 2,000 with the same seed and exclusions. If
2,000 is still insufficient, set `W-D3_R2.2_PANEL_INVALID`; do not change the
seed, weaken a stratum, reuse a household, or run a model.

Within each stratum, select the first 20 by ascending:

```text
SHA256("atlas-r2.2-d3-challenge" || canonical_household_identity)
```

Seal the calibration map, both manifests, all references, exact prompts,
software lock, tokenizer audit, and SHA-256 digests before the smoke.

### 4.2 Exact challenge strata

For a household define:

- `T=1` if at least one of
  `federal_income_tax_before_refundable_credits`,
  `state_income_tax_before_refundable_credits`, `local_income_tax`,
  `payroll_tax`, or `self_employment_tax` has absolute reference value at least
  USD 50.
- `CREDIT=1` if `federal_refundable_credits` or
  `state_refundable_credits` has absolute reference value at least USD 50.
- `B=1` if `snap`, `ssi`, or `tanf` has absolute reference value at least USD
  50, or any `*_eligible` reference is 1.

The mutually exclusive strata, applied in this order, are:

1. `REFUNDABLE_CREDIT`: `CREDIT=1`, irrespective of `T` or `B`.
2. `TAX_ONLY`: `T=1`, `CREDIT=0`, `B=0`.
3. `BENEFIT_ONLY`: `B=1`, `T=0`, `CREDIT=0`.
4. `TAX_AND_BENEFIT`: `T=1`, `B=1`, `CREDIT=0`.

Every selected challenge household retains every required negative and zero
field. No field is removed after stratum assignment.

## 5. Metrics, rescue/harm, and thresholds

### 5.1 Unchanged field correctness

For a schema-valid household:

- eligibility is correct only when the predicted integer 0/1 equals the
  reference 0/1;
- a reference amount equal to 0 is correct when absolute prediction is less
  than USD 50;
- a nonzero reference amount is correct when relative error is strictly less
  than 5% or absolute error is strictly less than USD 50.

Household agreement is correct fields divided by required fields. Panel
agreement is the unweighted macro-mean of household agreement. Missing or
invalid households have agreement 0.

All confidence intervals use 10,000 household-cluster bootstrap replicates
with PCG64 seed 2204. The challenge bootstrap resamples 20 households with
replacement inside each of the four strata on every replicate.

### 5.2 NRI-inspired rescue and harm

The immutable NRI reference is `all_zero_r2_2`, even if field-prior or
PolicyEngine has a higher mean score.

For panel `p` and field type `k` in `{eligibility, amount}`:

```text
rescue_rate[p,k] =
  count(zero baseline wrong AND candidate correct)
  / count(zero baseline wrong)

harm_rate[p,k] =
  count(zero baseline correct AND candidate wrong)
  / count(zero baseline correct)

NRI[p,k] = rescue_rate[p,k] - harm_rate[p,k]

NRI_macro[p] = 0.5 * NRI[p,eligibility] + 0.5 * NRI[p,amount]
```

A schema-invalid household makes every field candidate-incorrect, so it can
create harms and cannot create rescues. A zero denominator makes the panel
invalid; it is not replaced by zero or omitted.

For each challenge household stratum, also compute `rescue_rate_all` over all
fields the zero baseline gets wrong.

### 5.3 Prevalence panel gates

A candidate passes the prevalence panel only if all hold:

1. The lower 95% paired-bootstrap bound of
   `candidate agreement - PolicyEngine agreement` is greater than `-0.05`.
2. The lower 95% paired-bootstrap bound versus all-zero is greater than
   `-0.02`.
3. The lower 95% paired-bootstrap bound versus field-prior is greater than
   `-0.02`.
4. `NRI[P,eligibility] > 0` and `NRI[P,amount] > 0`.
5. Every critical floor in Section 5.5 passes on `P`.

### 5.4 Challenge panel gates

A candidate passes the challenge panel only if all hold:

1. `rescue_rate[C,eligibility] >= 0.80`.
2. `rescue_rate[C,amount] >= 0.80`.
3. `rescue_rate_all >= 0.70` separately in each of the four challenge strata.
4. The lower 95% bootstrap bound of `NRI_macro[C]` is greater than 0.
5. `NRI[C,eligibility] > 0` and `NRI[C,amount] > 0`.
6. Every critical floor in Section 5.5 passes on `C`.

No prevalence/challenge average can compensate for a failed panel or stratum.

### 5.5 Critical floors

These are hard constraints, applied separately to `P` and `C`:

| Floor | Exact definition | Threshold |
|---|---|---:|
| Eligibility false-positive rate | predicted 1 among reference-0 eligibility fields | `<= 0.05` |
| Amount false-positive rate | `abs(prediction) >= 50` among amount fields with `abs(reference) < 50` | `<= 0.05` |
| Normalized magnitude error | mean of `min(abs(pred-ref) / max(abs(ref),50), 10)` over amount fields where the zero baseline is wrong | `<= 0.25` |
| Catastrophic magnitude rate | fraction of the same material amount fields with normalized error `> 1.0` | `<= 0.05` |
| Qualifying completion | exact schema-valid answer returned within 30.000 user-wall seconds | `>= 0.95` |
| p95 user latency | nearest-rank p95 over all attempted households; incomplete/watchdog is `+infinity` | `<= 30.000 s` |

For normalized magnitude metrics, an invalid or missing amount prediction has
error 10. For the two false-positive rates, an invalid/missing prediction is
not an asserted positive; its penalty instead enters completion, agreement,
and NRI harm. Eligibility predictions other than exact integer 0/1 are schema
invalid, not thresholded.

For `P`, qualifying completion requires at least 95 of 100 households. For
`C`, it requires at least 76 of 80. Point thresholds are binding; Wilson and
bootstrap intervals are reported but do not replace these thresholds.

Threshold provenance is frozen here, before P/C model outputs: the 5-point
PolicyEngine non-inferiority and 95% completion floors inherit R2; the 2-point
trivial-baseline margin, 5% false-positive ceilings, 80% type rescue, 70%
per-stratum rescue, and 25% magnitude ceiling are W-D3-specific critical
floors. None may be tuned on the smoke or full panels.

## 6. W-D3 weight and execution gate

W-D3 Gate A weight under R2.2 is exactly `0.00`, unconditionally. Gate A is
already complete and cannot be reopened prospectively.

R2.2 W-D3 can enter Gate B as one binary workload-floor result, never as a
percentage averaged with W-D2, only if:

1. panel construction, hashes, software pins, schema token audit, all three
   baselines, and verifier pass before model execution;
2. the smoke below passes without a contract change; and
3. the candidate's full R2.2 cell passes both panels and every critical floor.

The smoke systems are:

- `falcon_h1_0.5b`, the cheapest local exploratory system; and
- `gemma3_12b`, the highest-W-D2-quality local system.

Each runs 24 sealed households: the first 8 prevalence households by
`SHA256("atlas-r2.2-d3-smoke-p" || identity)` and the first 4 households from
each challenge stratum by
`SHA256("atlas-r2.2-d3-smoke-c" || identity)`. These tasks count toward the
full cells if and only if no prompt, parser, cap, timeout, or scoring rule
changes afterward.

The smoke passes only if:

- both systems execute all 24 attempts without a harness or field-order defect;
- PolicyEngine produces exact references for all 24;
- Gemma3-12B has at least 23 of 24 qualifying completions; and
- neither system produces a schema-valid output that hits the 256-token cap.

There is no within-R2.2 prompt edit or cap increase after a smoke failure.
Failure sets W-D3 to diagnostic-only, gives it no Gate B workload vote, and
requires a new R2.3 amendment. If the smoke passes, all nine Gate A local
systems run the full 100+80 R2.2 panel. R2.2 authorizes no W-D3 API run.

## 7. W-D2 API ladder

### 7.1 Common contract

All six configured APIs run the unchanged 320 W-D2 episodes. The order within
each system is the existing query hash order, with languages ordered
`[en, es, fr, de, ja, zh_cn, ar, ko]`.

Common settings:

- unchanged R2.1 MKQA prompt and scorer;
- temperature 0;
- one answer, `max_output_tokens=64`, no tools, retrieval, or fine-tuning;
- 120-second outer watchdog and 30-second p95 user latency floor;
- at most one retry per transient task and at most 7 retry attempts per
  system, so at most 327 billed canonical/retry calls; Gemini Pro's separate
  16-call drift audit raises only its total reservation to 343;
- all provider-billed tokens, hidden/reasoning tokens if billed, retries,
  list-price cost, invoice cost, wall time, model response ID, and provider
  fingerprint are recorded.

The first eight canonical episodes, one query in all eight languages, are the
canary and count toward the 320 if the contract remains unchanged. The canary
requires at least 7 nonempty scored responses, usage/cost metadata on every
billed call, and no model-ID, prompt, or endpoint mismatch. Canary quality does
not trigger early stopping.

### 7.2 Exact order and sub-budgets

The reservation calculation uses at most 512 billed input tokens and 64 billed
output tokens per call at the frozen 2026-07-27 list prices. The five immutable
systems reserve 327 calls. Gemini Pro reserves 343 calls: 320 canonical calls,
at most 7 retries, and 16 no-retry drift-audit calls. The dollar ceiling, not
the reservation estimate, is binding.

The 320 frozen W-D2 prompts are at most 189 characters and at most 125 tokens
under the nine pinned local tokenizers, so the 512-token input reservation has
more than 4x observed tokenizer headroom. Provider-billed usage remains the
authoritative cost even when its tokenization differs.

| Order | System ID | Exact configured model ID | Role | Call reservation | Cost reservation | Hard W-D2 ceiling |
|---:|---|---|---|---:|---:|---:|
| 1 | `gemini_31_flash_lite` | `gemini-3.1-flash-lite` | value | 327 | USD 0.0732 | USD 0.50 |
| 2 | `gpt56_luna` | `gpt-5.6-luna` | value | 327 | USD 0.2930 | USD 1.50 |
| 3 | `gemini_31_pro` | `gemini-3.1-pro-preview` | frontier | 343 | USD 0.6147 | USD 3.00 |
| 4 | `claude_sonnet5` | `claude-sonnet-5` | value | 327 | USD 0.8162 | USD 4.00 |
| 5 | `gpt56_sol` | `gpt-5.6-sol` | frontier | 327 | USD 1.4650 | USD 7.50 |
| 6 | `claude_fable5` | `claude-fable-5` | frontier | 327 | USD 2.7206 | USD 14.00 |

The W-D2 ladder sub-budget is exactly USD 30.50. The pre-existing USD 1,200
global API ceiling remains binding; the effective ceiling is the smaller
remaining limit.

No API is skipped because a cheaper API scores well. All six are required to
establish the value-control and three-frontier ceiling.

### 7.3 Abort and drift rules

Abort only the affected system cell, preserve all costs and records, and
continue to the next rostered system when any of these occurs:

1. the exact configured model ID cannot be resolved before its canary; no alias
   or neighboring model may substitute;
2. the canary fails its contract;
3. the next reserved request would cross the system dollar ceiling;
4. 17 terminal task failures occur, because 304/320 (95%) completion has
   become impossible;
5. more than 7 transient retry attempts are requested;
6. billed usage or actual cost is missing after the one allowed retry;
7. response model ID or immutable provider fingerprint changes inside the
   cell;
8. duplicate task keys, mixed protocol revisions, or scorer mismatch occurs;
9. the USD 30.50 ladder sub-budget or USD 1,200 global budget is exhausted.

There is no quality-based sequential abort.

For abort rule 3, per-request reservation is the larger of (a) the configured
512-input/64-output list-price amount and (b) the highest actual billed cost of
any earlier call in that cell. The runner checks `spent + reservation` before
submitting the next call.

`gemini_31_pro` is a mutable preview. Complete its 320-task block inside seven
calendar days. At the end, repeat exactly 16 tasks selected by ascending
`SHA256("atlas-r2.2-gemini-drift" || task_id)`. Its cell is
`DRIFT_INVALID` if the provider model/fingerprint changes, the absolute paired
mean-F1 shift exceeds 0.05, or at least 3 of 16 pass/fail statuses flip. Drift
audit calls have no retry. A missing drift response is `DRIFT_INVALID`. Calls
and costs count against its USD 3.00 ceiling; therefore its implementation
reservation must leave room for them.

## 8. Required pre-execution implementation

Before any R2.2 model or API call, the verifier must reject:

- missing or changed Section 0 hashes;
- overlapping calibration, prevalence, or challenge household identities;
- wrong panel sizes or stratum counts;
- an unsealed field-prior map;
- tokenizer maximum above 224 on either evaluation panel;
- a parser that performs object/substring salvage or accepts non-integers;
- any W-D3 cap other than 256, watchdog other than 120 seconds, or latency
  floor other than 30 seconds;
- missing baselines or incomplete cost fields;
- any attempt to rerun/reweight Gate A;
- wrong API order, IDs, output cap, retry ceiling, or dollar ceiling; and
- any R2.1/R2.2 record merge.

Implementation changes made to satisfy this amendment are infrastructure, not
permission to run. The verifier output, manifest hashes, protocol file hash,
and clean dry runs must be committed before the first smoke or API canary.

## 9. Design score and 7/10 boundary

Protocol quality: 8/10. This is sufficient as the narrow R2.2 measurement
repair needed to preserve a credible route to a 7/10 Atlas. It is not itself
7/10 evidence.

The Atlas remains below 7/10 until the R2 headline conditions are actually
met: two independently sourced workload families with a complete sub-1.7B
local system non-inferior to all three frontiers, the same safety/reliability
floors, a lower-bound cost ratio above 10x at 10,000 tasks, prospective
selector success on eligible confirmation tracks, a correct cloud-required
negative case, and the future-model test.

This amendment is insufficient if any of the following happens:

- W-D3 is given positive retrospective Gate A weight or reorders the anchors;
- the seen R2.1 households are reused as R2.2 evaluation rather than
  calibration;
- the challenge strata, cap, prompt, parser, or thresholds change after smoke;
- rescue is reported without harm, negative fields are dropped, or panel
  failures are averaged away;
- PolicyEngine is omitted, treated as free while integration is omitted, or
  its upstream R&D is charged asymmetrically;
- an incomplete or mutable API cell is called a frontier comparison;
- W-D3's task-native PolicyEngine win is counted as a local-model
  scale-inversion workload; or
- the paper stops at discovery/API ceilings without dual prospective
  confirmation and the future-model test.
