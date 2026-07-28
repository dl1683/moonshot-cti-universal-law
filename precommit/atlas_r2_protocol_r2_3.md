# Atlas R2.3 — W-D3 Token-Cap Amendment

**Amendment date:** 2026-07-28
**Amends:** `atlas_r2_protocol_r2_2.md`
**Scope:** Local W-D3 token cap and associated cap references only.

## Binding language

R2.3 supersedes only the R2.2 local W-D3 token-cap and associated cap
references. The sealed prevalence and challenge panels, their hashes,
household selections, prompts, parser, scorer, metrics, thresholds,
budgets, watchdog, and latency floor are inherited unchanged. No R2.2
W-D3 model attempt occurred.

## Trigger

R2.2 Section 1.2 required a deterministic tokenizer preflight on the
sealed evaluation panels. The preflight found `M_eval=353` (Falcon-H1
tokenizer, scenario_1966, `indent=2`, 68 fields), which exceeds the
R2.2 preflight limit of 352. Per R2.2: "mark `W-D3_R2.2_SCHEMA_INVALID`,
exclude W-D3 from Gate B, and require R2.3."

## Replacement cap clause (supersedes R2.2 Section 1.2, paragraphs 1-4)

The R2.3 local W-D3 generation cap is exactly `448 max_new_tokens` for
every local model. Temperature remains 0. No best-of-n or format-repair
retry is allowed in the raw W-D3 cell. R2.3 authorizes no W-D3 API
execution, so this cap does not apply to an API model.

On the sealed evaluation panels, `M_local=353` across the nine pinned
tokenizers and the minified, `indent=2`, and `indent=4` serializers. The
cap is the smallest multiple of 64 at least `1.25 * M_local`:
`ceil_multiple_64(441.25) = 448`.

| Quantity | Result |
|---|---:|
| Required fields per household (eval panels) | 16 to 68 |
| Canonical minified-array tokens, median | ~42 |
| Canonical minified-array tokens, p95 | ~78 |
| Two- or four-space pretty JSON, Falcon-H1 maximum | 353 |
| R2.3 local cap | 448 |

Every audited serialization must be at most 416 tokens, preserving at
least 32 tokens of headroom below the 448-token cap. A maximum above
416 invalidates W-D3 and cannot trigger another silent cap increase.

The API cap clause (R2.2 Section 1.2, paragraphs 5-6) is unchanged.
The minimum API cap remains 384 and maximum 512.

## Operative reference changes

All operative references to the R2.2 local cap change as follows:

| R2.2 reference | R2.2 value | R2.3 value | Section |
|---|---|---|---|
| Local generation cap | 384 | 448 | 1.2 |
| `cap_hit` definition | "stopped by 384" | "stopped by 448" | 1.3 |
| Latency continuation cap | 384-token cap | 448-token cap | 2 |
| Smoke cap-hit check | "hits the 384-token cap" | "hits the 448-token cap" | 6 |
| Verifier local cap | 384 | 448 | 8 |
| Verifier tokenizer ceiling | 352 | 416 | 8 |
| Protocol revision in task records | `r2.2` | `r2.3` | 1.3 |

## Preserved without alteration

- R2.2 panel files, hashes, and manifest
- Smoke-household selection salts
- Parser and scorer logic
- 30-second latency floor
- 120-second watchdog
- 3.00 GPU-hour per-system and 27.00 GPU-hour aggregate caps
- API contract (R2.3 still authorizes no W-D3 API run)
- ESS thresholds, strata definitions, field-prior map
- Gate B formula, critical floors, bootstrap parameters

## Tokenizer audit

The sealed tokenizer-audit artifact `atlas_r2_3_tokenizer_audit.json`
records all tokenizer revisions, serializers, panel hashes, the
353-token argmax case, and the audit-code hash. This artifact must be
committed and verified before any smoke or model execution.

## Approval conditions

This approval becomes REJECT if:
- panels are regenerated;
- model outputs have already been inspected;
- tokenizer revisions change without a new audit; or
- any non-cap protocol term is altered under this amendment.
