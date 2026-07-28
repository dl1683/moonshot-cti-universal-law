## Ruling: APPROVE WITH CONDITIONS

Raising the local cap from 384 to 448 is the correct rule-preserving fix. The failure exposes a calibration-envelope mismatch—56 fields in C0 versus 68 in the real panels—not a defective output contract.

### Verification

I independently swept all `300 × 9 × 3 = 8,100` household/tokenizer/serializer combinations:

- Maximum: 353 tokens.
- Argmax: `scenario_1966`, Falcon-H1, 68 fields, `indent=2`; `indent=4` is also 353.
- `1.25 × 353 = 441.25`.
- Smallest multiple of 64 at least 441.25: 448.
- New preflight ceiling: `448 − 32 = 416`.
- Actual headroom from observed maximum to cap: `448 − 353 = 95`.
- Margin below the preflight ceiling: `416 − 353 = 63`.

All 300 households pass. The arithmetic is unambiguous.

### Legitimacy

This is legitimate because R2.2 explicitly required tokenizing the sealed evaluation gold arrays before execution and prescribed R2.3 if the 352 ceiling failed. It therefore pre-authorized a model-independent schema-sizing check, provided no model output had been inspected and the panels remained sealed and unchanged. See [R2.2 §1.2](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/precommit/atlas_r2_protocol_r2_2.md:102>).

Keeping 384 by relaxing the reserve from 32 to 31 would be post-hoc threshold tuning. Removing the household would corrupt the sealed panel. Rejecting pretty JSON would change parser semantics. Under the frozen formula, 448 is the smallest defensible cap.

### Second-order consequences

There is no blocking consequence:

- The worst observed prompt plus the 448-token allowance is 2,335 tokens; the smallest pinned model context is 16,384.
- The extra 64-token worst-case runtime can increase latency or GPU use, but the unchanged 30-second floor, 120-second watchdog, and three-hour system budget already measure and constrain that.
- Strict whole-output parsing means the extra allowance does not permit prose or repair attempts.
- The API cap clause remains separate and unchanged.
- Smoke must be rerun under 448, and any 448-token cap hit must still fail.

Two audit-documentation defects should be corrected before treating R2.3 as fully sealed:

1. [`r2_3_headroom=63`](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/precommit/atlas_r2_3_tokenizer_audit.json:52>) is mislabeled. Actual cap headroom is 95; 63 is the margin below the preflight ceiling.
2. The current R2.3 table’s approximate minified median/p95 values are inherited from C0. For the evaluation sweep, under the all-tokenizer aggregation, they are 46 and 88. Remove those nonbinding rows or define and correct them.

The exact audit code, tokenizer/package revisions, serializer bytes, special-token policy, panel hashes, and audit-artifact hash should also be sealed. The verifier should recompute token counts rather than trust the JSON’s declared maximum.

### Exact protocol amendment

Do not overwrite frozen R2.2. Add a separately hashed R2.3 delta containing:

> R2.3 supersedes only the R2.2 local W-D3 token-cap clause and its associated operative references. R2.2 remains `W-D3_R2.2_SCHEMA_INVALID`. The sealed calibration, prevalence, and challenge artifacts and hashes, household selections, prompts, parser, scorer, metrics, thresholds, budgets, watchdog, latency floor, smoke-selection salts, and API contract remain unchanged. No R2.2 W-D3 model attempt occurred before this amendment was frozen.

Replace the local portion of §1.2 with:

> The R2.3 W-D3 generation cap is exactly `448 max_new_tokens` for every local model. Temperature remains 0. No best-of-n or format-repair retry is allowed. R2.3 authorizes no W-D3 API execution.
>
> Let `M_local` be the maximum token count over all nine pinned local tokenizers, all sealed C0, P, and C gold arrays, and the canonical minified, `json.dumps(array, indent=2)`, and `json.dumps(array, indent=4)` serializers. Here `M_local=353`. The cap is
> `64 * ceil((1.25 * 353) / 64) = 448`.
>
> Before execution, every audited serialization must be at most `416` tokens, preserving at least 32 tokens below the generation cap. Failure sets `W-D3_R2.3_SCHEMA_INVALID`, excludes W-D3 from Gate B, and requires a new numbered amendment. No silent cap increase is permitted.

Mechanically supersede these references:

- §0 and §1.3: task-record revision `r2.2` → `r2.3`.
- §1.3: `cap_hit` means stopped by 448 rather than EOS.
- §2: continuation may run until the 448-token cap or 120-second watchdog.
- §6: smoke fails on any 448-token cap hit; no within-R2.3 cap change is allowed.
- §8: reject tokenizer maxima above 416 or any local cap other than 448.
- Preserve the API minimum/maximum of 384/512 unchanged.

That is a cap-envelope repair, not a redesign. The current checkout has subsequently advanced to R2.4, so this ruling validates the R2.3 amendment boundary; it is not fresh authorization to execute under revision `r2.3`.