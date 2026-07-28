The original R2.2 freeze was not safe to execute. Objections 1–3 exposed material defects; 4–5 exposed under-specified controls. The protocol has been prospectively refrozen before any R2.2 execution.

1. Token cap — amendment required

No, GPT/Claude/Gemini W-D3 tokenizers were not verified. Also, R2.2 authorizes those APIs only for W-D2—not W-D3—so the original “256 for every local and API model” sentence exceeded scope.

More importantly, a new local audit found:

- Minified maximum: 131 tokens
- Single-space JSON maximum: 186
- Pretty JSON maximum: 244 for Qwen/Gemma, 300 for Falcon

Therefore 256 can clip a parser-valid local output.

Amendment:

- Local W-D3 cap: exactly 384 tokens, derived as the smallest multiple of 64 above `1.25 × 300`.
- Preflight must test minified, `indent=2`, and `indent=4` arrays; every sealed array must be ≤352 tokens.
- Any cap hit fails the smoke.
- Future W-D3 API execution requires exact provider-tokenizer audits. Its common cap would use the same 1.25 rule, bounded to 384–512; unavailable tokenization or a result above 512 means no API run.

2. Challenge-panel size — fatal to stratum inference as written

The protocol previously computed global type NRI and per-stratum rescue, not per-stratum type NRI. But your central objection is correct: 360 fields are not 360 independent observations.

The frozen calibration proxy gives all-field rescue ESS:

| Stratum | Raw opportunities | ESS |
|---|---:|---:|
| Refundable credit | 123 | 15.45 |
| Tax only | 54 | 18.94 |
| Benefit only | 31 | 16.29 |
| Tax plus benefit | 64 | 18.79 |

Eligibility-rescue ESS is 0 in tax-only by construction and 8.29 in refundable-credit. Benefit-only amount-rescue ESS is only 4.50.

Amendment:

- Challenge panel: 200 households, exactly 50 per stratum.
- Stratum rescue, harm, and NRI are household-macro quantities.
- Each stratum requires rescue ≥0.70 and lower-95% NRI >0.
- Pre-execution opportunity ESS must be ≥35 per stratum and ≥60 for each global eligibility/amount rescue/harm cell.
- Candidate pool is 2,000 with one frozen fallback to 4,000; a second ESS failure invalidates W-D3.
- Full nine-system budget becomes 27 GPU-hours, using exactly 1.6442 hours of the existing global reserve.

Fifty households still do not support precision effect-size claims: worst-case normal 95% half-width is about 13.9 points. It is sufficient for a conservative gate with bootstrap lower bounds, not for claiming finely estimated stratum effects. Such a claim would need roughly 100+ households per stratum.

3. PolicyEngine — fatal if presented as an independent winner

PolicyEngine is a legitimate deployable calculator, so measuring its integration and operating cost is useful. But its 1.0 agreement is tautological because it generates the references.

Amendment:

- Statistical baselines are exactly `all_zero_r2_2` and `field_prior_r2_2`.
- PolicyEngine is the `REFERENCE_BOUND` implementation and operational cost control.
- Its licensing, integration, maintenance, runtime, latency, and failures remain fully costed.
- It is excluded from scientific leaderboards, Pareto-frontier claims, Gate A, Gate B, inversion counts, and independent workload wins.
- The selector may return `REFERENCE_BOUND_ONLY`, explicitly meaning an operational recommendation without independent accuracy validation.

Making PolicyEngine a scientific candidate would require a later holdout whose labels were neither generated nor adjudicated by PolicyEngine.

4. API order — partially handled, but amendment required

The first freeze already required all six systems and prohibited quality-based stopping. It lacked atomic funding and a sufficiently explicit no-edit rule.

The new hash-randomized order is:

1. `claude_fable5`
2. `gemini_31_flash_lite`
3. `claude_sonnet5`
4. `gemini_31_pro`
5. `gpt56_sol`
6. `gpt56_luna`

Additional controls:

- Reserve the entire $30.50 ladder budget before call one or run zero APIs.
- System ceilings are non-fungible.
- Seal prompt bytes, rendered requests, adapters, settings, normalizers, scorer, and retry rules in one contract hash.
- No prompt, schema, parser, interpretation, or scorer change between systems.
- Any such change invalidates the entire ladder and requires R2.3.
- Complete all six systems and the drift audit within seven calendar days.

5. W-D3 weight — amendment required, but positive Gate A weight rejected

Giving W-D3 positive Gate A weight now would retroactively reopen a completed selection after its failure mode was observed. That would weaken the preregistration.

But the prior binary description was under-specified. W-D3 now has two explicit Gate B consequences:

- Passing adds one discovery-workload floor to Gate B’s first ranking coordinate.
- A frozen continuous worst-constraint shortfall enters its second coordinate.

Thus W-D3 can reorder finalists. Candidate-specific missing or invalid results receive infinite shortfall; workload-level invalidation removes W-D3 equally from every candidate. Full rescue/harm/NRI, critical-floor, reliability, latency, and ESS vectors remain published.

It still cannot count as a headline inversion because R2.2 has no W-D3 API comparison and PolicyEngine is reference-bound.

Score

- First R2.2 freeze: 6/10; insufficient to execute.
- Refrozen design: 8/10; sufficient as a protocol supporting a route to a 7/10 Atlas, but not itself 7/10 evidence.

It becomes insufficient if fields are treated as independent, ESS gates fail, PolicyEngine is marketed as an independent win, an unaudited W-D3 API cap is used, the API ladder is partial or modified between systems, W-D3 is retrospectively added to Gate A, or W-D3 substitutes for the still-missing prospective frontier-relative inversions and confirmation tests.

The binding [R2.2 protocol](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/precommit/atlas_r2_protocol_r2_2.md:102>) and [round-2 adjudication](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/codex_r2_2_round2_adjudication.md:12>) are committed and pushed at `355a8b1`. Protocol SHA-256:

`0d1b33ec35c1f897a941a1525042b2926ebdb70433a84a702d2804be1e80002e`

No R2.2 experiment was run. Execution remains blocked because the existing verifier passes its older checks but does not yet implement the new R2.2 fail-closed validations.