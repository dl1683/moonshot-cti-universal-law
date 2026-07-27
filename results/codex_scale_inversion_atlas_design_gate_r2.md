# Scale-Inversion Atlas: R2 Design-Gate Revision

**Date:** 2026-07-27  
**Verdict:** **CONDITIONAL GO, AFTER MATERIAL REDESIGN**  
**Supersedes:** `results/codex_scale_inversion_atlas_design_gate.md`  
**Binding compute ceiling:** 360 RTX 5090 GPU-hours, including reserve  
**Binding external API ceiling:** USD 1,200 at frozen list prices  

## Executive decision

Four of the five objections are correct without qualification. The third is
also correct about protocol robustness, although a failed confirmation source
must still reduce the strength of the scientific claim.

The original protocol is not executable as written. Its 320-hour budget is
withdrawn. It combined task timeouts with unstated expected runtimes, did not
fix survivor counts, and treated 51 templates as though the full factorial
would somehow prune itself. The revised Atlas:

1. reduces the discovery samples;
2. fixes every survivor count in advance;
3. replaces the 20-minute GPU occupancy assumption with enforceable cumulative
   generation caps while retaining a 20-minute user wall-clock floor;
4. adds six inference-only API systems, including three frontier "Goliaths";
5. adds a second, temporal confirmation track;
6. replaces one frozen adaptation recipe with equal-budget tuning and
   0.5x/1x/2x sensitivity;
7. removes RLVR and distillation from the core four-to-eight-week experiment.

The Atlas is a moonshot only if it produces and prospectively validates a
frontier-relative scale inversion. The selector and benchmark machinery are
supporting infrastructure, not the headline result.

## The repeatable one-sentence story

> **Can a pocket AI on one consumer GPU match GPT-5.6, Claude Fable 5, or
> Gemini 3.1 Pro Preview on your exact verified job for one-tenth the cost—and can we
> know before deploying it?**

The result is allowed to answer "no, pay for the cloud" for some workloads.
That boundary is part of the claim. The exciting positive result is not that
one local model wins a composite score; it is that the Atlas prospectively
identifies where a tiny complete system replaces a frontier API without
violating the user's quality, reliability, latency, or memory constraints.

## 1. Arithmetic: withdrawal and replacement

### 1.1 What was wrong

The objection's arithmetic is correct:

- `36 tasks x 48 systems x 20 minutes = 576 GPU-hours` for the W-D1 pilot
  alone under worst-case occupancy.
- `281 x 12 x 20 minutes = 1,124 GPU-hours` for W-C1.
- Nine RLVR cells at a six-hour ceiling consume 54 hours before QLoRA,
  distillation, evaluation, metrology, or confirmation.

Successive halving is not a budget unless the number retained at every rung
and the maximum resource consumed before elimination are precommitted. R1 did
neither. The 320-hour estimate was therefore fiction, not merely optimistic.

### 1.2 Binding task counts and resource caps

The revised experiment uses:

| Family | R1 size | R2 binding size | Per-system GPU cap |
|---|---:|---:|---:|
| W-D1 SWE-bench-Live MultiLang | 144 | 72: 24 pilot + 48 remainder | 8 cumulative generation minutes/task |
| W-D2 MKQA real queries | 1,600 episodes | 320: 40 queries x 8 languages | 30 seconds/task |
| W-D3 PolicyBench households | 100 | 100 | 30 seconds/task |
| W-C1 RealClawBench | 281 | 72 selected before reveal by hash | 8 cumulative generation minutes/task |
| W-C2 future SWE-bench-Live batch | absent | first 48 valid post-freeze tasks | 8 cumulative generation minutes/task |

For agentic tasks, the 8-minute cap is cumulative local-model generation time,
not the end-to-end timeout. CPU tests and sandbox work may continue up to the
20-minute user latency floor and run in parallel with the GPU queue. A task
that exhausts its generation, action, token, or wall-clock allowance is a
failure, not an unmetered extension. The harness must release the GPU while
CPU-only tests run.

For W-D2 and W-D3, the cap is one 128-token answer after a bounded prompt. For
W-D1, W-C1, and W-C2, the common cap is 16 tool actions, 8,192 cumulative
output tokens, 64,000 cumulative billed input tokens, eight GPU generation
minutes, and 20 wall-clock minutes.

The API systems share the action, token, and 20-minute user wall-clock caps.
They do not receive an invented GPU-time cap because provider compute time is
unobservable; their full latency and billed usage are recorded instead.

These are hard scheduler limits. The budget below is consequently a maximum,
not an estimate based on hoped-for average runtimes.

### 1.3 Fixed survivor counts

There are two and only two pruning gates:

**Gate A: nine raw local checkpoints to six anchors.** Retain exactly two
checkpoints per architecture family:

1. that family's highest-quality checkpoint on the frozen W-D2/W-D3 aggregate;
2. that family's cheapest checkpoint within 10 percentage points of its
   family best.

If these identify the same checkpoint, retain the next-cheapest checkpoint.
This yields exactly six anchors and preserves both scale and architectural
coverage.

**Gate B: augmented discovery pool to four local finalists.** Rank complete
systems lexicographically by:

1. number of discovery workload floors passed;
2. worst standardized quality shortfall;
3. all-in projected cost at 10,000 tasks;
4. peak memory;
5. frozen system ID.

Retain exactly four subject to at least two architecture families and at least
one checkpoint at or below 1.7B parameters. If a constraint is impossible,
fill by the ranking and record the failed diversity constraint. No additional
system may be rescued after viewing confirmation outcomes.

The choice of six preserves a small and a strong scale point in each family.
The choice of four is the maximum confirmation roster compatible with the
360-hour hard ceiling and still permits a meaningful constrained choice.

### 1.4 Exact 360-hour ledger

All formulas below are worst-case GPU occupancy under the new caps.

| Phase | Exact calculation | GPU-hours |
|---|---|---:|
| P0 preflight, harness, power calibration | fixed allocation | 24.0 |
| P1 raw W-D2/W-D3 screen | `9 x (320+100) x 0.5 / 60` | 31.5 |
| P2 raw W-D1 pilot | `6 x 24 x 8 / 60` | 19.2 |
| P3 RAG on six anchors | `6 x 420 x 0.5 / 60` | 21.0 |
| P3 best-of-4 on two small anchors | `2 x 420 x 4 x 0.5 / 60` | 28.0 |
| P3 QLoRA on four anchors | `4 x 420 x 0.5 / 60` | 14.0 |
| P3 QLoRA+RAG on two small anchors | `2 x 420 x 0.5 / 60` | 7.0 |
| P4 QLoRA recipe sensitivity | `4 anchors x 2 workloads x (0.5+1+2)` | 28.0 |
| P5 W-D1 best-of-4 pilot | `2 x 24 x 4 x 8 / 60` | 25.6 |
| P5 W-D1 remainder | `4 finalists x 48 x 8 / 60` | 25.6 |
| P6 W-C1 RealClaw confirmation | `4 x 72 x 8 / 60` | 38.4 |
| P6 W-C2 future SWE confirmation reserve | `4 x 48 x 8 / 60` | 25.6 |
| P7 energy repeat block | `4 x 3 x (16x0.5 + 16x8) / 60` | 27.2 |
| P7 alternative-engine audit | `4 x (16x0.5 + 16x8) / 60` | 9.1 |
| **Scheduled subtotal** |  | **324.2** |
| **Failure/retry reserve** | `360 - 324.2` | **35.8** |
| **Binding maximum** |  | **360.0** |

No phase may borrow from another except through the 35.8-hour reserve. If a
preflight or workload exceeds its row, scope is reduced according to the
precommitted task hash order; the ceiling is not silently raised.

At 45 scheduled GPU-hours per week this is eight weeks. At 60 hours per week
it is six weeks. CPU adjudication, API calls, data preparation, and analysis
parallelize with the single GPU queue. Model training and local inference do
not.

This is a drastic scope reduction. It is the only honest way to retain the
four-to-eight-week constraint on one RTX 5090.

## 2. The Goliath: frontier and value API systems

The local-only study would still answer a real procurement question, but the
objection is right about the moonshot and the user's actual decision. It would
not justify "stop paying for the biggest AI," because it never measured the
biggest AI.

### 2.1 Frozen API ladder

At P0, resolve and freeze the exact provider snapshot or dated model ID. The
roster is:

| Role | Provider system at design time | Adaptation |
|---|---|---|
| Frontier | OpenAI GPT-5.6 Sol | none |
| Value control | OpenAI GPT-5.6 Luna | none |
| Frontier | Anthropic Claude Fable 5 | none |
| Value control | Anthropic Claude Sonnet 5 | none |
| Frontier | Google Gemini 3.1 Pro Preview | none |
| Value control | Google Gemini 3.1 Flash-Lite | none |

The current official pages list GPT-5.6 Sol at $5/M input and $30/M output
tokens and Luna at $1/M and $6/M; Claude Fable 5 at $10/M and $50/M and
Sonnet 5 at $3/M and $15/M standard list prices; and Gemini 3.1 Pro Preview below
200k tokens at $2/M and $12/M and Flash-Lite at $0.25/M and $1.50/M.
Because provider aliases and prices change, the manifest stores the dated
snapshot, region, service tier, price table, and retrieval date.

If a provider exposes only a mutable preview ID, as Google currently does for
Gemini 3.1 Pro Preview, complete that API's discovery and confirmation block
inside one seven-day window and repeat a hash-selected 5% drift audit at the
end. A statistically material drift makes that API block non-confirmatory; an
alias string is not treated as an immutable model snapshot.

Sources:

- OpenAI model catalog: https://developers.openai.com/api/docs/models
- Anthropic model catalog: https://platform.claude.com/docs/es/about-claude/models/overview
- Google Gemini 3.1 guide: https://ai.google.dev/gemini-api/docs/gemini-3
- Google pricing: https://ai.google.dev/gemini-api/docs/pricing

The value controls are essential. Beating an expensive flagship is not an
avoided-cost result if a cheaper hosted model also meets the floor.

### 2.2 Comparison contract

API systems receive the same task-visible documents, retrieval corpus, tools,
action limit, cumulative input/output token caps, and adjudicator as local
systems. They receive no fine-tuning. Provider-native hidden reasoning is
permitted because the unit is the deployable system, not a mechanistically
matched model.

Record:

- provider-billed input, cached input, output, search, and tool charges;
- retry and rate-limit charges;
- wall-clock latency and failure rate;
- the frozen public list price and actual invoice charge.

Provider energy is **unknown**, not estimated from marketing claims. Local
energy is measured. The primary common cost axis is dollars per completed
qualifying task; energy is an additional local deployment constraint.

All six APIs run on the 420 short discovery episodes and the 192 possible
agentic episodes. Under the stated token caps and the listed prices, the
theoretical token bill is approximately $485 if every cap is exhausted.
The binding $1,200 gate covers retries, provider tool charges, and price drift.
Crossing it requires a new design gate, not an accounting footnote.

### 2.3 Revised system roster

The 51-template claim is deleted. The core has 32 deployable templates:

- 9 raw local W4 checkpoints;
- 6 API systems;
- 6 local RAG systems after Gate A;
- 4 QLoRA systems from the two strongest architecture families;
- 2 QLoRA+RAG small systems;
- 2 best-of-4 small systems;
- 3 task-native non-neural baselines.

Four high-precision variants of the finalists are an audit, not additional
selector candidates. The alternative-engine runs are also audits. Templates
are run only where applicable; this is deliberately not a full factorial.

The local checkpoints remain:

- Qwen3: 0.6B, 4B, 14B;
- Gemma 3: 1B, 4B, 12B;
- Falcon-H1: 0.5B, 3B, 7B.

The core local format is W4A16. The four finalist quantization audits use
W8A16 under the same engine and context limits; this is the only quantization
axis in R2. A different format is allowed only when a checkpoint cannot load,
and then the substitution is frozen before any quality run.

The three families retain at least two scale points through Gate A. The
headline "David" subset is at most 1.7B. A 14B local result is useful, but it
does not by itself establish the manifesto-level inversion.

## 3. Confirmation without a single point of failure

The objection is correct: "kill the program if RealClawBench fails" is a
falsifier, not a robust data plan.

### 3.1 Two confirmation tracks, frozen now

**W-C1: cross-family confirmation.** Before any label or outcome reveal,
select 72 RealClawBench tasks using ascending
`SHA256("atlas-r2-c1" || immutable_task_id)`. Preserve the task mix through
stratified hash selection if the source provides authoritative task
categories. The release must pass environment reconstruction, deterministic
verifier, immutable ID, and license checks in P0.

**W-C2: temporal prospective confirmation.** Use the first 48 executable,
license-valid tasks in the first SWE-bench-Live MultiLang release whose issue
or merge cutoff is strictly later than the signed selector freeze. Select by
ascending `SHA256("atlas-r2-c2" || immutable_task_id)`. No W-C2 task may enter
retrieval, tuning, threshold selection, prompt editing, or model selection.

SWE-bench-Live is explicitly maintained as an automatically updated benchmark,
its repository states a monthly update plan, and it provides testable
containerized environments. That makes the post-freeze batch a genuine future
data test:
https://github.com/microsoft/SWE-bench-Live

If W-C1 fails reproducibility before outcomes are viewed, W-C2 remains the
prospective confirmation. The experiment is not killed. However, without
W-C1 it has no untouched cross-family confirmation and the scientific score is
capped below 7/10. That is the part on which the protocol should hold its
ground: a backup can preserve learning, but it cannot manufacture independent
cross-family evidence.

If both are available, both run. A 7/10 result requires the selector to pass
on W-C1 and W-C2; it may not choose the more flattering one after the fact.

### 3.2 Freeze mechanics

Before either confirmation:

1. write `precommit/atlas_r2_selector.json`;
2. include source hashes, task-hash rules, system IDs, API snapshots, prices,
   prompts, retrieval corpora, quality/reliability floors, tie-breaks,
   amortization formulae, and abstention rules;
3. compute a SHA-256 digest;
4. run a duplicate-key-rejecting verifier;
5. tag the commit `atlas-r2-selector-freeze`;
6. export the commit hash and manifest digest to an external timestamped
   record.

The confirmation runner accepts only that digest. It emits predictions before
executing a candidate: selected system, predicted feasibility, expected cost,
and an ordered fallback.

### 3.3 Future-model test

The future-model test is not an alias update. It is the first eligible
post-freeze model release satisfying all of:

- checkpoint or API snapshot first became available after the signed freeze;
- it fits one of the declared local deployment envelopes or is a generally
  accessible API;
- no Atlas rule or threshold changes after its identity is known;
- only the predeclared 12-task calibration allowance may estimate hardware
  speed and memory; no quality labels are visible;
- the frozen selector predicts feasibility and the cheapest qualifying system
  before full execution.

Passing requires correct feasibility classification and no more than 20%
realized cost regret relative to the cheapest qualifying evaluated candidate.
If no eligible release appears within eight weeks, the core paper remains
unfinished rather than substituting an old model.

## 4. Adaptation fairness and recipe sensitivity

The objection is correct. Identical learning rate, rank, and step count across
0.6B and 14B models and unrelated tasks confounds model capability with recipe
misfit. Conversely, unrestricted per-model tuning would make cost and researcher
attention unbounded. The correct object is the best observed system under the
same declared adaptation-resource envelope.

### 4.1 Equal-budget, model-specific search

After Gate A, choose the two strongest architecture families on the frozen
W-D2/W-D3 aggregate. For each, take its small and large anchor: four models.
For each model and each of W-D2 and W-D3, run three resource envelopes:

- 0.5x: 0.5 GPU-hour;
- 1x: 1.0 GPU-hour;
- 2x: 2.0 GPU-hours.

That is `4 models x 2 workloads x 3.5 hours = 28 GPU-hours`.

Inside each envelope, asynchronous successive halving searches the same
predeclared grid:

- LoRA rank: 8 or 32;
- learning rate: `5e-5` or `2e-4`;
- target modules: attention-only or attention+MLP;
- early stopping: frozen validation verifier.

Every cell receives the same GPU-hour and energy ceiling, not the same number
of optimizer steps. Training tokens, joules, wall time, examples seen, and all
failed trials count as adaptation cost. The winning recipe is chosen only on
the discovery validation split.

The 0.5/1/2-hour envelope includes training **and** its bounded validation
inference; the timer does not pause for evaluation. The separate 14-hour P3
row is the one full 420-episode evaluation of each selected QLoRA system, not
an uncounted evaluation of every searched recipe.

### 4.2 Robustness rule

A hard Atlas recommendation involving adaptation is allowed only if:

1. its feasibility verdict is unchanged between 1x and 2x;
2. the selected system is unchanged, or the 1x choice has at most 20% cost
   regret under the 2x results;
3. quality changes monotonically within its bootstrap uncertainty or any
   non-monotonicity is explicitly modeled;
4. the recommendation survives the high-precision finalist audit.

Otherwise the selector returns `RECIPE_SENSITIVE` and recommends a calibration
run instead of claiming a stable optimum.

This does not prove robustness to 10x or 100x training. It establishes the
bounded surface the project can afford. RLVR and distillation are therefore
removed from the core Atlas. Their original 512-prompt/500-step treatment was
too small to support a production claim and too expensive to expand honestly
inside this gate. They may enter a later, separately budgeted mechanism study
only after the core frontier-relative result exists.

## 5. The dual-loop success condition

The objection is correct: "we built a selector" is a methodology result.
Methodology alone does not pass the moonshot gate.

### 5.1 Headline scale-inversion criterion

The Atlas reaches the 7/10 scientific bar only if all of the following occur:

1. on at least two independently sourced workload families, a complete local
   system at or below 1.7B is non-inferior to all three frontier APIs within a
   predeclared five-percentage-point quality margin;
2. it passes the same reliability and safety floors;
3. at 10,000 completed tasks it costs at least 10x less than the cheapest
   qualifying hosted API, including hardware, energy, adaptation, engineering,
   and failure/retry cost;
4. the frozen selector predicts those inversions on W-C1/W-C2 rather than
   merely explaining them afterward;
5. the selector also correctly identifies at least one workload where the
   local system fails and the cloud should be purchased;
6. the future-model test passes.

Use paired task bootstrap confidence intervals. "Non-inferior" requires the
lower 95% confidence bound on local-minus-frontier quality to exceed -5 points.
The cost ratio's lower 95% bound must exceed 10x.

If the selector works but no sub-1.7B system beats the cheapest qualifying API
by 10x at frontier-equivalent quality, the output may be a useful procurement
tool or benchmark paper, but it is not the Scale-Inversion moonshot and is
capped at 5.5/10.

### 5.2 The specific anti-benchmark-paper kill

Kill the Atlas as the primary program if, after discovery and both eligible
confirmation tracks, any of these holds:

- fewer than two independently sourced workload families show a sub-1.7B
  local system at frontier non-inferiority;
- no verified local-vs-hosted avoided-cost ratio reaches 10x at 10,000 tasks;
- selector cost regret exceeds 20% on either confirmation track;
- recommendation stability is below 80% across bootstrap resamples;
- more than 25% of candidate conclusions are `RECIPE_SENSITIVE`;
- the only positive result depends on a composite score, omitted adaptation
  labor, unverifiable cloud-energy estimates, or a hand-selected quality floor;
- the future-model test fails;
- the work's strongest defensible sentence is "we evaluated many systems and
  built a framework."

RealClawBench alone no longer kills execution. Failure of both confirmation
tracks, contamination of the seal, or inability to reproduce authoritative
verifiers does.

## 6. Measurement consequences

The R1 measurement contract otherwise survives, with these clarifications:

- **Quality:** executable pass rate for W-D1/W-C2, canonical/deterministic
  answer accuracy for W-D2, PolicyEngine agreement and explanation-field
  validity for W-D3, and official deterministic task completion for W-C1.
- **Reliability:** invalid-action rate, timeout rate, crash rate, verifier
  disagreement, abstention calibration, and paired retry success. A system
  must complete at least 95% of attempted tasks without infrastructure-caused
  failure and may not exceed the frozen unsafe-action threshold.
- **Latency:** end-to-end user wall time including retrieval, tools, tests,
  retries, and queueing; report median and p95. GPU accounting is separate.
- **Energy:** NVML power samples at 10 Hz integrated above an idle baseline,
  cross-checked against wall power on the metrology subset; report joules per
  successful task and include failed attempts.
- **Memory:** NVML peak allocated plus reserved GPU memory, sampled during each
  run and cross-checked with the inference engine's allocator logs.
- **Adaptation:** actual GPU-hours, joules, training tokens, API teacher cost
  if any, data preparation labor, and evaluation failures.
- **Amortization:** report 1K, 10K, and 100K successful tasks. Divide fixed
  hardware and adaptation cost by successful—not attempted—tasks and add
  energy, API, operations, and expected retry cost.
- **Safety:** workload-specific prohibited-action and unsupported-claim
  checks are hard feasibility constraints, never score offsets.

The primary user-level object remains:

`cheapest complete system that satisfies this workload's quality, safety,
reliability, p95 latency, and memory constraints at the user's volume`.

## 7. Binding artifact changes

Create or revise these files before GPU execution:

| File | Purpose |
|---|---|
| `results/codex_scale_inversion_atlas_design_gate_r2.md` | This binding R2 decision |
| `precommit/atlas_r2_selector.json` | Machine-readable selector, candidate, floor, and tie-break contract |
| `precommit/atlas_r2_selector.schema.json` | Strict schema; rejects unknown and duplicate-key-equivalent fields |
| `precommit/atlas_r2_task_seal.json` | W-C1 hash sample and W-C2 prospective selection rule |
| `precommit/atlas_r2_budget.json` | Phase GPU-hour, API-dollar, token, action, and retry ceilings |
| `configs/atlas_r2_systems.yaml` | Exact local revisions, quantization, engines, API snapshots, prompts, and tools |
| `configs/atlas_r2_adaptation.yaml` | QLoRA grid, ASHA rule, 0.5x/1x/2x budgets, and robustness decision |
| `configs/atlas_r2_workloads.yaml` | Source revisions, licenses, task counts, split hashes, and adjudicators |
| `scripts/verify_atlas_r2_precommit.py` | Fail-closed commitment verifier |
| `scripts/run_atlas_r2.py` | Budget-enforcing runner that refuses unsealed confirmation |
| `scripts/score_atlas_r2.py` | Paired metrics, floors, confidence intervals, cost regret, and kill tests |
| `results/atlas_r2_preflight.md` | Reproducibility and environment gate |
| `results/atlas_r2_discovery.md` | Discovery-only results and Gate A/B decisions |
| `results/atlas_r2_confirmation.md` | Prediction-first W-C1/W-C2 and future-model results |
| `results/atlas_r2_cost_ledger.csv` | Task-level tokens, dollars, joules, latency, memory, failures, and amortization |

Until the precommit, budget enforcer, and confirmation seal validate, the
correct action is not to launch a large run.

## Final architectural position

The original protocol tried to preserve too many models, augmentations, and
tasks and then asked successive halving to make the arithmetic disappear. R2
does not.

The Atlas remains worth pursuing only as a **prospective frontier-replacement
test**:

- the Goliaths are explicit;
- the David threshold is explicit;
- the 10x avoided-cost bar is explicit;
- the cases where cloud remains necessary are part of the result;
- the selector is frozen before independent and temporal confirmation;
- and the compute budget is enforced by the runner rather than narrated after
  the fact.

That is a falsifiable moonshot. A local-only leaderboard with a cost model is
not.
