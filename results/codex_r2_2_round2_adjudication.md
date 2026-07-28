# Atlas R2.2 Round-2 Adjudication

Date: 2026-07-28

Status: DESIGN REFROZEN BEFORE EXECUTION

No R2.2 model or API call occurred under the first R2.2 hash. The first freeze
at git commit `2f48de4` is retained as an audit record but is prospectively
withdrawn. The binding replacement is
`precommit/atlas_r2_protocol_r2_2.md` plus its updated SHA-256 sidecar.

## Rulings

| Objection | Ruling | Consequence |
|---|---|---|
| 1. 256-token W-D3 cap | Amendment required; not fatal to the Atlas | 384 for local W-D3; no W-D3 API cap is authorized without exact provider-tokenizer preflight |
| 2. 20 households per challenge stratum | Fatal to stratum inference as written | 50 households per stratum, household-macro rescue/harm/NRI, and model-independent ESS gates |
| 3. PolicyEngine as baseline | Fatal to a scientific-win narrative; valid only as a reference-bound operational control | Two statistical baselines remain; PolicyEngine moves to a separate `REFERENCE_BOUND` table and cannot win a scientific leaderboard |
| 4. Cheapest-first API order | Partially handled, but amendment required | Hash-randomized order, atomic full-ladder funding, immutable contract hash, and no inter-system edits |
| 5. W-D3 at 0% Gate A | Positive Gate A weight is rejected; prior Gate B role was under-specified | W-D3 remains 0% Gate A but gets an explicit Gate B floor and continuous standardized shortfall that can reorder finalists |

## 1. Token cap

The objection is correct that GPT, Claude, and Gemini W-D3 tokenization was not
verified. The original sentence claiming one 256-token cap for local and API
models exceeded the authorized experiment: R2.2 authorizes APIs only on W-D2,
not W-D3.

The objection also exposed a current local defect. The prior audit measured a
minified array and a single-space array, but the strict parser accepts RFC 8259
whitespace. Re-auditing the frozen 100 calibration arrays gave:

| Serializer | Qwen3/Gemma3 max | Falcon-H1 max |
|---|---:|---:|
| Minified | 131 | 131 |
| Default spaces | 186 | 186 |
| `indent=2` | 244 | 300 |
| `indent=4` | 244 | 300 |

Thus 256 can clip an accepted pretty-printed Falcon output. The local cap is
now derived as the smallest multiple of 64 at least 1.25 times the observed
maximum: `ceil_multiple_64(1.25 * 300) = 384`. New panel preflight requires
all minified, `indent=2`, and `indent=4` gold arrays to be at most 352 tokens
under all nine pinned tokenizers. Any cap hit fails the smoke.

There is no guessed API fallback. A later W-D3 API protocol must obtain exact
provider-native or pinned-tokenizer counts for every exact API model and every
sealed gold array. It derives one common cap by the same 1.25 rule, bounded to
384-512. Missing exact tokenization or a derived cap above 512 means no API
W-D3 run.

## 2. Challenge panel sample size

The original protocol did not compute a separate type-specific NRI inside
each stratum; it computed global type NRI and a stratum rescue floor. That
technical distinction does not rescue the design. Twenty household clusters
are not 360 independent observations.

Using the frozen calibration households as a model-output-free proxy and the
R2.2 strata, the opportunity-weight ESS
`(sum m_i)^2 / sum(m_i^2)` is:

| Stratum, 20 households | All-field rescue opportunities | ESS |
|---|---:|---:|
| Refundable credit | 123 | 15.45 |
| Tax only | 54 | 18.94 |
| Benefit only | 31 | 16.29 |
| Tax plus benefit | 64 | 18.79 |

Eligibility-rescue ESS is 0 in tax-only by construction and 8.29 in the
refundable-credit proxy. Amount-rescue ESS is 4.50 in benefit-only. Globally
over the 80 proxy households, eligibility-rescue ESS is 27.76 and
amount-rescue ESS is 54.52. The objection is therefore sustained.

The amended challenge panel has 200 households, 50 per stratum. Stratum
rescue and harm are first computed within each household and then macro-meaned
over 50 household clusters. Each stratum must have rescue at least 0.70 and a
positive lower 95% bootstrap bound on household-macro NRI. Before any model
run, each stratum must have all-field rescue-opportunity ESS at least 35, and
each global eligibility/amount rescue/harm ESS must be at least 60. The only
fallback is a predeclared pool expansion from 2,000 to 4,000; a second failure
invalidates W-D3.

Under the same-mix planning approximation, the C0 values scale to stratum ESS
38.63-47.35, global eligibility-rescue ESS 69.40, and amount-rescue ESS
136.30. Fifty clusters still do not support a precision claim: the worst-case
Bernoulli normal 95% half-width is 0.139. This is enough for a guarded gate
with lower-bound tests, not for a fine-grained stratum effect-size claim.

The nine full local cells now contain 300 households each and receive 3.00
GPU-hours each. The 27.00-hour maximum uses the 25.3558 hours remaining in P1
plus exactly 1.6442 hours of the already frozen global retry reserve.

## 3. PolicyEngine

PolicyEngine is an executable policy calculator, so costing it as a possible
deployment path is operationally honest. Calling its 1.0 agreement a measured
accuracy result is not. The same pinned engine generates the answer key, so
that value is tautological.

The amended role split is exact:

- `all_zero_r2_2` and `field_prior_r2_2` are the only statistical baselines;
- `policyengine_us_1_723_0` is the reference implementation and operational
  cost control;
- PolicyEngine integration, runtime, maintenance, completion, latency, and
  licensing are still measured under the common cost equation;
- PolicyEngine is excluded from model/baseline leaderboards, Pareto claims,
  Gate A, Gate B, scale-inversion counts, and independent workload wins; and
- the selector may report `REFERENCE_BOUND_ONLY`, explicitly meaning an
  operational recommendation without independent accuracy validation.

An independent scientific PolicyEngine win would require a later sealed
holdout whose labels were neither generated nor adjudicated by PolicyEngine.
R2.2 does not have that evidence and will not imply it.

## 4. API order and sequential bias

The first freeze already required all six systems and prohibited
quality-based stopping, so running out of budget was not intended to select a
cheap-only result. It did not atomically reserve the total budget or state the
no-inter-system-edit rule strongly enough. The optics and the control surface
were weak.

The amended order is derived from the immutable Gate A artifact hash, not
price or expected quality:

1. `claude_fable5`
2. `gemini_31_flash_lite`
3. `claude_sonnet5`
4. `gemini_31_pro`
5. `gpt56_sol`
6. `gpt56_luna`

Before the first call, the ledger must atomically reserve USD 30.50. The six
system ceilings are non-fungible, and all six reach a terminal state. One
manifest seals exact prompt bytes, rendered provider requests, adapters,
normalizers, scorer, settings, retry rules, and task order into an
`api_ladder_contract_hash`. No prompt, schema, parser, interpretation, or
scoring edit is allowed between systems. A desired edit after call one
invalidates the whole ladder and requires R2.3. The entire ladder must finish
within seven calendar days.

## 5. W-D3 decision value

Giving W-D3 positive Gate A weight now would be retrospective reselection
after Gate A and after the W-D3 failure mode was seen. That is less defensible
than zero weight, not more. It remains prohibited.

The objection is right that a vaguely described binary vote is too lossy.
However, proportional averaging is also the wrong remedy: quality, false
positives, magnitude error, completion, and latency are hard procurement
constraints and should not compensate one another.

The amendment gives W-D3 two exact Gate B effects for every local candidate:

1. a pass adds one discovery-workload floor to Gate B's first ranking
   coordinate; and
2. a frozen normalized worst-constraint shortfall enters Gate B's second
   ranking coordinate.

Those coordinates can reorder the four finalists. The full metric vector and
confidence intervals remain published. A workload-level invalidation removes
W-D3 from every candidate equally; a candidate-specific failure is an
infinite shortfall. PolicyEngine never enters that candidate rank.

W-D3 still cannot count as a headline inversion because R2.2 has no W-D3 API
comparison and its reference implementation is answer-key-bound. Its valid
scientific role is to test baseline rescue, reliability, and whether the
constraint selector promotes or rejects local models on a structured policy
calculation workload.

## Score

The first freeze is 6/10 and was not safe to execute. The refrozen design is
8/10: sufficient as an R2.2 measurement contract and sufficient to preserve a
credible route to a 7/10 Atlas, but not itself 7/10 evidence.

It becomes insufficient if an unaudited W-D3 API cap is used; the sealed ESS
floors fail; fields are treated as independent; PolicyEngine is marketed as
an independent winner; the API ladder is partial or contract-mutated; W-D3 is
retroactively added to Gate A; W-D3 is omitted from its frozen Gate B
consequence; or the program uses W-D3 to substitute for the still-missing two
prospective frontier-relative inversions, 10x lower-bound cost advantage,
confirmation tracks, cloud-required case, and future-model test.
