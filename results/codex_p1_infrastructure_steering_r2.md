Your four-phase execution model is correct. My ten steps were dependency-ordered, not a demand for ten serial work blocks. Parallelize after freezing the shared schema.

The decisive changes:

- Fix Gemma now in the implementation batch, then validate it in preflight.
- Allocate segment energy proportional to synchronized generation time.
- Treat the “predictive law” as a prospective feasibility frontier with a decision guarantee—not a post-hoc scaling curve.
- A single spectacular David/Goliath result is insufficient for 9/10; a broad, prospective, replicated inversion could reach it without a closed-form law.

## Start now: the actual execution plan

### Minute zero: freeze R2.1

Before editing, commit the short protocol amendment defining:

- First 40 scorable MKQA queries by frozen hash order.
- Official MKQA scoring and language macro-average.
- One structured PolicyBench generation per household.
- Authoritative task-record and segment-record schemas.
- Margin `0.05` and cost assumptions.
- Actual numerical format for every checkpoint.
- Existing P1 data labeled `pilot_superseded`; it remains in the budget ledger but not canonical analysis.
- The prospective feasibility-frontier hypothesis described below.

Then execute four phases.

### Phase 1 — Batch implementation

Run four module-owned workstreams in parallel after the schema is fixed:

1. **Runner, records, resume, and energy**

   - Authoritative task-ID-keyed result records.
   - Protocol revision and run ID in every record.
   - Resume from task records, never summaries.
   - Append-only segment records.
   - Atomic writes and duplicate rejection.
   - Recompute all summaries from task-record unions.
   - Enforce time/token caps.

2. **Workloads and scientific scoring**

   - First 40 scorable MKQA queries.
   - Official Apple normalization, segmentation, `Counter` F1, EM, language macro-average, query-clustered uncertainty.
   - W-D3 as 100 compact JSON household generations.
   - Field-level scoring, household macro-score, parse-valid rate, and all-fields-correct rate.

   Keep MKQA and W-D3 with one owner because both modify [cti_atlas_workloads.py](C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\src\cti_atlas_workloads.py).

3. **Downstream scorer and Gate A**

   - Margin `0.05`.
   - Frozen nonzero capital, electricity, adaptation, retrieval, and API costing.
   - 50/50 workload-level Gate A aggregate.
   - Correct next-cheapest fallback.
   - Paired/query-clustered bootstrap.
   - Hard failure on incomplete cells, mixed protocol revisions, missing costs, or unknown scorer versions.

4. **Inference compatibility, including Gemma**

   Fix the double BOS in [cti_atlas_inference.py](C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\src\cti_atlas_inference.py) now. Use:

   - `apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")`;
   - no text-then-retokenize path;
   - exactly one BOS assertion in tests;
   - model-specific fallback only for tokenizers that lack compatible arguments.

   Use BF16 for Gemma 3, including `bnb_4bit_compute_dtype=torch.bfloat16` when quantized. Google’s official Gemma example uses tokenized chat-template output and BF16. [Gemma 3 model card](https://huggingface.co/google/gemma-3-4b-it#usage)

   Also resolve the adjacent format discrepancy now: [atlas_r2_systems.yaml](C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\configs\atlas_r2_systems.yaml) declares every checkpoint W4A16, while the loader quantizes only models above 7B. My recommendation is to honor the frozen W4A16 core and make the loader follow the configuration for all nine systems. Any incompatible substitution must be named explicitly in R2.1 before rerunning.

Use separate logical commits for these workstreams even if implementation is concurrent. One monolithic commit would make the protocol repair unnecessarily difficult to audit.

### Phase 2 — Integrated validation gate

Do not load full models until these checks pass:

- 40 query IDs and exactly 320 MKQA episodes.
- Selected original 25 queries remain a subset.
- Official evaluator fixtures match exactly for all eight languages.
- Japanese/Chinese character segmentation; Korean whitespace scoring.
- Exactly 100 W-D3 prompts covering all 1,970 reference fields.
- An interrupted run and uninterrupted run produce identical scientific summaries.
- No duplicate `(revision, workload, system, task)` keys.
- Segment energy allocations sum exactly to measured segment energy.
- `0.05`, not `5.0`, reaches the non-inferiority calculation.
- Local cost is positive under the frozen assumptions.
- Gate A refuses missing or mixed-revision cells.
- Gemma input has one BOS.
- Empty output receives `empty_output`, not ordinary `fail`.
- The 30-second generation cap is enforced.

### Phase 3 — Model preflight

The Gemma fix is already implemented; this phase proves it.

Run:

- One deterministic prompt through all nine checkpoints.
- One eight-language MKQA query plus two households through Qwen-0.6B and Gemma-4B.
- One Gemma-12B smoke prompt to verify W4/BF16 compute.
- One forced interruption/resume.
- One deliberately malformed W-D3 output.
- One energy segment.

Required exit conditions:

- Nonempty, non-pad Gemma output.
- Exactly one BOS.
- Expected dtype and quantization confirmed from the loaded model.
- Raw prediction survives resume.
- Correct score version appears in every record.
- Energy conservation holds.
- No canonical P1 records are written; use a preflight run ID.

### Phase 4 — Canonical P1-R2.1

1. Run all nine local systems on W-D2, 320 episodes each.
2. Run the frozen API ladder on W-D2 to establish the ceiling.
3. Run all nine local systems on W-D3, 100 households each.
4. Run the API ladder on W-D3.
5. Generate derived summaries from authoritative task records.
6. Audit completeness and protocol hashes.
7. Apply Gate A.
8. Do not enter P2 unless the Gate A audit passes.

## Energy allocation

Use proportional allocation:

\[
E_i = E_{\text{segment}}
      \frac{t_i}{\sum_j t_j}
\]

where \(t_i\) is synchronized generation time:

1. `torch.cuda.synchronize()`
2. Start timer.
3. Generate.
4. `torch.cuda.synchronize()`
5. Stop timer.

Do not divide equally. Output lengths and generation times vary materially, so equal allocation systematically overcharges short tasks and undercharges long ones.

Keep `segment_energy_joules` as the primary measured quantity and `allocated_energy_joules` as a clearly labeled derived task field. Require allocations to sum to the segment measurement within floating-point tolerance.

Do not complicate P1 by integrating individual 10 Hz sample windows; short generations have too few samples for stable direct estimates. P7 repeated metrology is the appropriate place for finer task-level energy measurement. Model-loading energy should be recorded separately and amortized according to the frozen deployment-volume assumption, not charged repeatedly because an experimental run was interrupted.

## The 9/10 route

I revise my earlier wording: **8/10 is the ceiling of the current benchmarking design, not the direction’s intrinsic ceiling.**

### The right predictive law

The right object is a multidimensional **feasibility frontier**, not “quality is a function of parameter count.”

A concrete form is:

\[
P(\text{success}_{s,t})
=
\sigma(\theta_s^\top x_t-b_t)
\]

where:

- \(\theta_s\) is a system capability vector estimated from a small frozen calibration panel;
- \(x_t\) is an outcome-blind task-demand vector—knowledge freshness, language, constraint count, arithmetic demand, tool horizon, context burden, and required reliability;
- \(b_t\) is task difficulty;
- cost, memory, latency, and safety remain explicit constraints rather than being folded into quality.

The Atlas then predicts:

\[
s^*
=
\arg\min_s \operatorname{Cost}(s,h,v)
\]

subject to lower-confidence quality, reliability, and safety floors.

“Scale inversion” occurs when a smaller system already lies on the quality-saturation side of the task-demand boundary, so added capacity increases cost without materially increasing qualifying performance.

The scientific claim is not that this equation fits discovery data. It must predict, before labels:

- Whether a new model is feasible.
- The cheapest qualifying model.
- Which workloads admit sub-1.7B systems.
- Realized cost regret on unseen workloads and a future model.

That predictive compression—plus a finite-sample selection or regret guarantee—is the strongest route to Turing-level significance.

### Could David versus Goliath alone reach 9?

A single 0.6B-versus-GPT-5.6 victory cannot. Even at 100× lower cost, one workload remains vulnerable to contamination, prompt fit, tool asymmetry, benchmark specialization, and coincidental task structure. Its ceiling is approximately 7/10.

A law is not strictly necessary if the empirical result becomes a broad regularity. A 9/10 empirical result would require the inversion to hold across:

- At least three genuinely independent deployment-grade workload families.
- At least one agentic and one consequence-bearing workload.
- At least two small-model architecture families.
- Multiple frontier and value APIs.
- Untouched prospective tasks.
- A future model released after selector freeze.
- Alternate engine/precision and recipe-sensitivity audits.

At that point it is no longer merely a David/Goliath anecdote; it is a discovered systems phenomenon.

### Minimum paradigm-shift result

The minimum credible 9/10 package is:

- A sub-1.7B complete system exceeds a meaningful absolute quality and safety floor.
- Its paired lower 95% confidence bound is above the −5-point frontier margin.
- It achieves at least **100× all-in cost reduction on two independent workloads**.
- The result replicates in a second small-model family.
- The frozen Atlas predicts both inversions before target outcomes.
- It makes no feasibility mistakes and realizes at most 10% cost regret on W-C1, W-C2, and the future-model test.
- The conclusion survives quantization, engine, prompt, recipe, API-drift, and energy audits.

That is the target. R2.1 can produce the empirical shock and the first prospective frontier test. A genuinely general law will probably require a broader R3 workload matrix, because five workload families are not enough to justify “universal.” Do not weaken that epistemic boundary merely to print 9/10 sooner.