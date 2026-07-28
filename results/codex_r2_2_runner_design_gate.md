Gate decision: **REJECT for model execution under R2.3 as currently frozen.**

The panels, hashes, tokenizer audit, roster, and smoke selection are sound. The precommit verifier passes, all 300 households have valid field ordering/hashes, and I independently reproduced all 592 fields in the 24-household smoke set with the pinned PolicyEngine stack with zero mismatches.

However, three protocol-blocking implementation conflicts remain. They cannot be silently repaired inside the runner because R2.3 permits cap-only changes.

## Blocking findings

1. **Invalid-output harm is reversed.**  
   For an invalid household, every field is candidate-incorrect, so harm must be true when `zero_baseline_correct` is true. The current invalid branch sets `"harm": not zero_correct`, the opposite. It also omits the required normalized magnitude error of `10` for invalid amount predictions. See [cti_atlas_workloads.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_atlas_workloads.py:607>) versus the frozen definition in [Section 5.2](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/precommit/atlas_r2_protocol_r2_2.md:456>).

2. **The sealed panels retain rounded integers, not unrounded PolicyEngine references.**  
   The builder constructs the unrounded array, rounds it into `gold_ints`, stores only those integers, then removes `refs_dict`; see [cti_build_atlas_r2_2_panels.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/scripts/cti_build_atlas_r2_2_panels.py:289>). But Section 1.1 says predictions are scored against unrounded references. For example, `scenario_000` stores `21923` and `8251`, while the pinned engine returns `21922.517578125` and `8250.8623046875`. This can change correctness at strict `$50`/`5%` boundaries.

3. **`generate()` does not provide or strictly enforce the required stop semantics.**  
   It returns no EOS or cap-hit flag. Its timer starts only when the stopping criterion is first called—after generation has begun—and is cooperative at token boundaries, not a hard outer watchdog. See [cti_atlas_inference.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_atlas_inference.py:109>). A stalled first token, CUDA hang, or host crash can exceed 120 seconds without authoritative recovery.

The current verifier checks the R2.3 audit values but does not test these scorer, reference, watchdog, or runner invariants. Therefore its current “ALL CHECKS PASSED” result is necessary but insufficient.

## Exact runner design

Once the blockers are prospectively resolved and resealed, use these constants and signatures:

```python
R2_3_PROTOCOL_REVISION = "r2.3"
R2_3_PANEL_REVISION = "r2.2"
R2_3_MAX_NEW_TOKENS = 448
R2_3_WATCHDOG_SECONDS = 120.0
R2_3_LATENCY_FLOOR_SECONDS = 30.0
R2_3_SYSTEM_GPU_CAP_SECONDS = 10_800.0
R2_3_TOTAL_GPU_CAP_SECONDS = 97_200.0

R2_3_PREVALENCE_PATH = (
    REPO / "data" / "policybench" / "r2_2_prevalence.json"
)
R2_3_CHALLENGE_PATH = (
    REPO / "data" / "policybench" / "r2_2_challenge.json"
)
R2_3_MANIFEST_PATH = (
    REPO / "data" / "policybench" / "r2_2_panel_manifest.json"
)

def run_p1_policybench_r2_3(budget, system_filter=None):
    ...

def run_smoke_r2_3(budget, system_filter=None):
    ...

def _run_policybench_r2_3_cells(
        budget, systems_cfg, system_ids, panel_rows, run_mode):
    ...

def _recompute_policybench_r2_3_summary(
        records, spec, panel_rows, expected_task_ids):
    ...
```

Parameterize the existing record helpers without changing W-D2 behavior:

```python
def _task_records_path(
        phase, workload, system_id,
        protocol_revision=PROTOCOL_REVISION):
    ...

def _load_task_records(
        phase, workload, system_id,
        protocol_revision=PROTOCOL_REVISION):
    ...

def _save_task_records(
        records, phase, workload, system_id,
        protocol_revision=PROTOCOL_REVISION):
    ...
```

Do **not** change global `PROTOCOL_REVISION = "r2.1"`: that would relabel W-D2 records. Every R2.3 call must pass `protocol_revision="r2.3"` explicitly. The existing global/path coupling is at [run_atlas_r2.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/scripts/run_atlas_r2.py:37>).

The exact per-system path is:

```text
results/cti_atlas_r2_task_records/
  cti_atlas_r2_r2.3_P1_W-D3_{system_id}.json
```

### Panel representation and IDs

Load both files with `load_r2_2_panel()` and wrap each household without modifying the sealed object:

```python
panel_rows = [
    {"panel_id": "P", "stratum": None, "household": hh}
    for hh in prevalence
] + [
    {"panel_id": "C", "stratum": hh["stratum"], "household": hh}
    for hh in challenge
]
```

Preflight every household:

```python
fields == sorted(fields, key=lambda x: x.encode("utf-8"))
n_fields == len(fields) == len(gold_array)
field_order_hash == sha256("|".join(fields).encode("utf-8")).hexdigest()
gold_minified == json.dumps(gold_array, separators=(",", ":"))
```

Define:

```python
identity_hash = sha256(
    household["identity_string"].encode("utf-8")
).hexdigest()

task_id = f"W-D3:{panel_id}:{identity_hash}"

gold_answer_hash = sha256(
    household["gold_minified"].encode("utf-8")
).hexdigest()
```

Full 64-character hashes should be stored—no 16-character truncation.

### Household execution flow

For each selected system and household:

1. Load existing R2.3 records.
2. Skip only records with `execution_state == "terminal"` and matching protocol, model revision, panel hash, field hash, and contract fingerprint.
3. Call `_thermal_gate(system_id)` before every original or retry attempt.
4. Atomically reserve 120 GPU-seconds under both caps.
5. Construct `prompt = format_r2_2_prompt(household)`.
6. Start `user_wall_seconds` immediately before the first `generate()` call.
7. Call:

```python
result = generate(
    model,
    tok,
    prompt,
    max_new_tokens=448,
    temperature=0.0,
    timeout_seconds=120,
)
```

8. Compute:

```python
clean = _strip_think_block(result["text"])
parsed, schema_valid, error_code = parse_r2_2_output(
    clean,
    expected_length=household["n_fields"],
    fields=household["fields"],
)
```

9. Stop the user-wall timer immediately after strict parsing. Then call:

```python
scores = score_r2_2_household(parsed, scoring_household)
```

10. Atomically save the task map after every terminal household.
11. Log the attempt with deterministic ledger ID `f"{task_id}#a{attempt_index}"`.
12. Recompute summaries only from task records plus the sealed expected panel—not from a prior summary.

Exactly one retry is allowed only for a classified CUDA-driver failure or supervised host-process crash. The retry must use identical prompt bytes and settings. Empty, malformed, late, cap-hit, watchdog, or low-quality output is never retried.

## Task-record schema

Each non-`__` map entry must contain:

```json
{
  "task_id": "W-D3:P:<64-hex>",
  "phase": "P1",
  "workload": "W-D3",
  "system_id": "falcon_h1_0.5b",
  "protocol_revision": "r2.3",
  "panel_protocol_revision": "r2.2",
  "panel_id": "P",
  "stratum": null,

  "canonical_identity": [
    "populace_us_2024", "2024", "21760", "102773", "us"
  ],
  "canonical_identity_hash": "<64-hex>",
  "scenario_id": "scenario_074",
  "gold_answer_hash": "<64-hex>",
  "fields": ["..."],
  "field_order_hash": "<64-hex>",
  "expected_length": 20,
  "prompt_sha256": "<64-hex>",
  "contract_fingerprint": "<64-hex>",

  "raw_output": "...",
  "cleaned_prediction": "...",
  "parsed_prediction": [0, 1],
  "input_tokens": 1234,
  "output_tokens": 87,
  "eos_reached": true,
  "cap_hit": false,
  "generation_stop_reason": "eos",
  "wall_seconds": 12.34,
  "user_wall_seconds": 12.36,
  "latency_floor_met": true,
  "watchdog_abort": false,
  "retry_count": 0,
  "retry_reason": null,

  "schema_valid": true,
  "schema_error_code": null,
  "qualifying_completion": true,
  "execution_state": "terminal",
  "status": "QUALIFYING_COMPLETION",

  "fields_correct": 17,
  "fields_total": 20,
  "household_agreement": 0.85,
  "all_correct": false,
  "field_results": [
    {
      "field": "head_medicaid_eligible",
      "type": "eligibility",
      "reference": 1,
      "predicted": 1,
      "correct": true,
      "zero_baseline_correct": false,
      "rescue": true,
      "harm": false,
      "normalized_magnitude_error": null
    }
  ],

  "scorer_version": "r2.2.0",
  "scorer_sha256": "<64-hex>",
  "model_id": "tiiuae/Falcon-H1-0.5B-Instruct",
  "model_revision": "8f2587ca...",
  "tokenizer_revision": "8f2587ca...",
  "package_revisions": {
    "python": "3.13.7",
    "torch": "2.11.0+cu128",
    "transformers": "5.12.1",
    "bitsandbytes": "0.49.0",
    "accelerate": "1.11.0",
    "tokenizers": "0.22.2",
    "nvidia-ml-py": "13.610.43"
  },

  "run_id": "run_<id>",
  "segment_id": "seg_<id>",
  "gpu_seconds": 12.34,
  "allocated_energy_joules": 1234.5,
  "billed_usage": {
    "input_tokens": 1234,
    "output_tokens": 87,
    "usage_complete": true
  },
  "dollars": {
    "api_usd": 0.0,
    "local_generation_usd": 0.001775
  },

  "attempts": [
    {
      "attempt_index": 0,
      "attempt_id": "W-D3:P:<hash>#a0",
      "state": "completed",
      "max_new_tokens": 448,
      "temperature": 0.0,
      "timeout_seconds": 120.0,
      "input_tokens": 1234,
      "output_tokens": 87,
      "gpu_seconds": 12.34,
      "eos_reached": true,
      "cap_hit": false,
      "watchdog_abort": false,
      "infrastructure_error_code": null
    }
  ]
}
```

`schema_error_code` must be null when valid; otherwise exactly one of:

```text
NOT_JSON, NOT_ARRAY, WRONG_LENGTH, NON_INTEGER, NONFINITE,
BAD_ELIGIBILITY, EXTRA_TEXT, EMPTY
```

Store raw Unicode strings losslessly. Do not use the existing ASCII replacement, which destroys exact raw output provenance.

## Resume and budget semantics

Before starting an attempt, write a provisional `attempt_reserved` entry atomically. Resume behavior:

- Terminal matching task: skip.
- Reserved/stale original attempt: conservatively charge 120 seconds, classify it as `HOST_PROCESS_CRASH`, and permit its single retry if budget remains.
- Mismatched revision, model, panel, or contract fingerprint: abort; never overwrite or merge.
- Budget exhaustion: mark the cell `INCOMPLETE_BUDGET`; do not keep retrying on future invocations.

Under an execution lock:

```python
system_spent + system_reserved + 120 <= 10_800
total_spent + total_reserved + 120 <= 97_200
```

After completion, release the unused reservation and charge actual synchronized generation time. A watchdog/stale reservation is charged exactly 120 seconds. Smoke tasks use the same files and cap—they receive no separate allowance.

## Smoke mode

The exact current hash-ranked smoke selection is:

- P: `scenario_074`, `071`, `049`, `056`, `006`, `081`, `087`, `057`
- `REFUNDABLE_CREDIT`: `937`, `393`, `1164`, `1294`
- `TAX_ONLY`: `1772`, `602`, `1309`, `1534`
- `BENEFIT_ONLY`: `1985`, `871`, `1579`, `1751`
- `TAX_AND_BENEFIT`: `420`, `866`, `1342`, `777`

Selection remains algorithmic using the two frozen salts; the explicit list is only an audit assertion.

`run_smoke_r2_3()` writes into the same Falcon/Gemma full-cell files. It passes iff:

```python
smoke_pass = (
    both_systems_have_exactly_24_started_and_terminal_households
    and harness_defect_count == 0
    and field_order_defect_count == 0
    and policyengine_reference_mismatch_count == 0
    and gemma3_12b_qualifying_completion_count >= 23
    and all_attempt_cap_hit_count == 0
)
```

Model schema failures are not automatically harness defects. `INCOMPLETE_BUDGET`, an unstarted household, or a contract mismatch fails the smoke.

A failure is immutable under R2.3: no rerun, prompt edit, parser change, scorer change, timeout change, or cap change. Full execution must remain blocked pending a new amendment.

## Summary structure

Use one derived system map:

```json
{
  "protocol_revision": "r2.3",
  "contract_fingerprint": "<64-hex>",
  "panel_hashes": {"P": "<hash>", "C": "<hash>"},
  "systems": {
    "<system_id>": {
      "cell_state": "COMPLETE",
      "expected_households": 300,
      "terminal_households": 300,
      "attempt_count": 301,
      "retry_count": 1,
      "cap_hit_count": 0,
      "watchdog_abort_count": 0,
      "budget": {
        "gpu_seconds": 7200.0,
        "gpu_hours": 2.0,
        "cap_hours": 3.0,
        "pass": true
      },
      "panels": {
        "P": {
          "expected": 100,
          "agreement": 0.81,
          "schema_valid_rate": 0.99,
          "qualifying_completion": {"count": 97, "rate": 0.97},
          "p95_user_wall_seconds": 28.4,
          "nri": {
            "eligibility": {
              "rescue_numerator": 0,
              "rescue_denominator": 0,
              "rescue_rate": null,
              "harm_numerator": 0,
              "harm_denominator": 0,
              "harm_rate": null,
              "nri": null
            },
            "amount": {},
            "macro": null
          },
          "baseline_comparisons": {},
          "critical_floors": {},
          "panel_valid": true,
          "panel_pass": false
        },
        "C": {
          "expected": 200,
          "nri": {},
          "strata": {
            "REFUNDABLE_CREDIT": {
              "rescue_rate_all": 0.0,
              "harm_rate_all": 0.0,
              "nri_all": 0.0,
              "nri_ci95_lower": 0.0
            }
          },
          "critical_floors": {},
          "panel_valid": true,
          "panel_pass": false
        }
      },
      "w_d3_floor_pass": 0,
      "standardized_shortfall": {
        "value": null,
        "is_infinite": true
      }
    }
  }
}
```

For each panel/type:

```python
rescue_rate = sum(rescue) / sum(not zero_baseline_correct)
harm_rate = sum(harm) / sum(zero_baseline_correct)
nri = rescue_rate - harm_rate
nri_macro = 0.5 * eligibility_nri + 0.5 * amount_nri
```

Retain numerators and denominators. A zero denominator yields `null`, `panel_valid=false`, and infinite shortfall—never zero.

Bootstrap all required bounds using 10,000 household-cluster replicates with a shared `np.random.Generator(np.random.PCG64(2204))`. P resamples 100 households; C resamples 50 independently within each stratum. Missing/invalid households score zero, and invalid amount NME is 10. The p95 is nearest-rank; incomplete/watchdog latency is infinity.

## CLI integration

Add `--smoke` and route before the generic unimplemented fallback:

```python
if args.phase == "P1" and args.workload == "W-D2":
    return run_p1_mkqa(budget, system_filter=args.system)

if args.phase == "P1" and args.workload == "W-D3":
    if args.smoke:
        return run_smoke_r2_3(
            budget,
            system_filter=args.system,
        )
    return run_p1_policybench_r2_3(
        budget,
        system_filter=args.system,
    )
```

Rules:

- `--smoke` is valid only for `P1/W-D3`.
- In smoke mode, `--system` may name only the two smoke systems; the overall result remains `PENDING` until both are complete.
- Full mode refuses execution unless smoke PASS is recomputed from task records under the identical contract fingerprint.
- W-D3 banners and summaries say `r2.3`; W-D2 remains `r2.1`.
- W-D3 dry-run must load and verify panels, audit, selections, record paths, resume state, and budgets—not merely run the generic budget check.

## Required module changes

- **`cti_atlas_workloads.py`: no R2.3 changes are permitted.** The four frozen functions must not be patched or wrapped to change their effective semantics. Because of the identified scorer/reference conflicts, resolution requires a new prospective amendment and newly sealed scorer/reference artifacts.
- **`cti_atlas_inference.py`: required change.** Extend `generate()` compatibly with `eos_reached`, `cap_hit`, `watchdog_abort`, and `stop_reason`; preserve unrounded timing; start its deadline before tokenization. A persistent supervised worker process is needed for a true Windows hard watchdog/host-crash recovery. Merely improving `_TimeLimit` remains cooperative and does not satisfy “outer watchdog.”
- **`verify_atlas_r2_precommit.py`: required change.** Add scorer invariants, unrounded-reference availability, parser fixtures, exact runner constants, task-path revision, contract fingerprint, budget reservation, and hard-watchdog tests.

## Frozen boundary

Under R2.3, none of these may change:

- The P/C panel bytes, hashes, identities, or household selection
- Smoke salts or selected households
- `format_r2_2_prompt`
- `_strip_think_block`
- `parse_r2_2_output`
- `score_r2_2_field`
- `score_r2_2_household`
- Scoring thresholds, bootstrap seed/count, strata, ESS rules, or field prior
- `448`, `120.000`, `30.000`, `3.00`, or `27.00`
- Temperature 0, one-array output, no repair retry, and no W-D3 API execution

Because the current scorer cannot satisfy that frozen boundary, the correct gate action is: **do not implement around it and do not launch the smoke; issue an R2.4 scorer/reference amendment first.**