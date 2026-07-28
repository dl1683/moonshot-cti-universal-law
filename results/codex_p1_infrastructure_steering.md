## Verdict

Choose **(a): stop all model runs and fix the complete P1 measurement path first.** The 5a5c853 patch repairs only pass-count display; it does not make resumed results scientifically mergeable.

Treat every existing W-D2 result as **pilot/superseded data**. Preserve it, but rerun canonical P1 under a new protocol revision. The live ledger contains about 1.29 generation-hours—roughly 4% of the allocation—so compute conservation is not a credible reason to retain invalid measurements.

## Decisions A–G

### A. Run 320 episodes

Keep the frozen **40 scorable queries × 8 languages = 320**.

The current “200 episodes” are only 25 independent questions repeated across languages. Expanding from 25 to 40 raises the independent query clusters by 60%, which is far more valuable than the episode count suggests.

Do not merely add 120 episodes. Because raw predictions were not retained and scoring is changing, rerun all 320 episodes for all nine systems under the corrected instrumentation. Select the first 40 answerable queries by the existing hash order, using answerability metadata only—not observed model performance—and document this as protocol revision R2.1.

### B. Fix MKQA scoring now

Adopt Apple’s official scoring semantics exactly:

- Character segmentation for `ja` and `zh_cn`.
- Whitespace tokenization for `ko`; Korean is not character-segmented by the official implementation.
- Language-specific normalization.
- Token multiplicities using `Counter`, not sets.
- Continuous EM and F1 as primary metrics.
- Macro-average across the eight languages.
- Confidence intervals clustered by query ID, because translations are correlated.

Do not invent a blanket character-level “CJK” scorer. [Apple’s official evaluator](https://github.com/apple/ml-mkqa/blob/main/mkqa_eval_util.py) defines Japanese and Chinese as mixed-segmentation languages but not Korean; its [evaluation contract](https://github.com/apple/ml-mkqa#evaluation) also reports per-language results and language macro-averages.

Keep `F1 >= 0.5` only as an explicitly frozen secondary procurement threshold, never as the primary scientific metric.

### C. Fix the downstream scorer now

Change the non-inferiority margin to **0.05 immediately**.

Also replace the zero-cost placeholders before collecting more data. Freeze:

- Local system acquisition cost.
- Useful lifetime and utilized inference hours.
- Electricity price.
- Measured kWh.
- Fixed adaptation/RAG costs amortized at the declared volume.
- API list prices and actual billed charges.
- Retry and failure costs.

Local inference cost should be:

`capital amortization + electricity + workload-specific fixed costs`

Do not count both an owned-hardware hourly rate and capital amortization. Provider energy remains unknown.

This cannot wait until P5 because the runner must capture the inputs now. The exact final cost analysis can mature later; the measurement contract cannot.

### D. Redesign W-D3 as 100 generations

Run **one structured generation per household**, not one generation per target:

- Prompt for a compact JSON object containing every required output variable.
- Use the frozen 128-token allowance; the current 32-token limit is incorrect.
- Score every returned field with the existing binary or dollar tolerance.
- Compute each household’s score as its fraction of correct fields.
- Define the system’s primary W-D3 score as the macro-mean over 100 households.
- Separately report JSON parse validity and strict all-fields-correct household rate.

This preserves 100 independent experimental units, matches the 900-generation budget, and tests a realistic complete household decision. The 1,970-target design would create pseudo-replication and expose each requested variable in a separate simplified prompt. Also fix `load_policybench(n_households=100)`, which currently ignores its argument.

### E. Exact execution order

1. **Stop Falcon-3B now after its current durable checkpoint.** Do not continue Gemma-4B, Falcon-7B, or Gemma-12B.

2. **Write and freeze protocol revision R2.1** covering the 320-episode rule, official MKQA scorer, 100-household W-D3 design, cost assumptions, artifact schema, and superseded pilot designation.

3. **Replace the resume architecture.** Resume and summaries must read authoritative task records, not infer scientific results from the cost CSV.

4. **Implement official MKQA scoring and corrected workload selection.**

5. **Implement household-level W-D3**, including structured-output parsing and 128-token generation.

6. **Fix downstream scoring and Gate A**, including:

   - `0.05` margin;
   - real cost accounting;
   - query-clustered uncertainty;
   - language macro-averaging;
   - the incorrect “next-best” fallback, which must be next-cheapest;
   - a frozen 50/50 workload-level Gate A aggregate so 320 MKQA translations do not outweigh 100 households merely through row count.

7. **Fix metering and cap enforcement.** The current budget checker counts generation seconds rather than full GPU occupancy, the 30-second cap is not actually enforced, and ledger energy is zero. These are part of the same infrastructure gate.

8. **Add interruption tests.** A clean uninterrupted run and an interrupted/resumed run over identical fake tasks must produce byte-equivalent task sets and equivalent summaries. Also test duplicate prevention, empty output, malformed JSON, Japanese/Chinese segmentation, Korean whitespace scoring, and energy-segment aggregation.

9. **Run two-system preflight:** one known-good Qwen plus Gemma-4B. Resolve Gemma’s chat-template path before treating empty outputs as model failures. Record empty generation as `inference_error` or `empty_output`, not an ordinary wrong answer.

10. **Start canonical P1-R2.1:**

    - Run all nine local systems on W-D2, 320 episodes each.
    - Run the frozen API ladder on W-D2 to establish the workload ceiling.
    - Run all nine local systems on W-D3, 100 households each.
    - Run the API ladder on the same W-D3 households.
    - Derive all summaries from authoritative records.
    - Apply Gate A only after every required system/workload cell is complete.

### F. Retain raw predictions

Yes. Store authoritative per-system records under a structure such as:

`results/cti_atlas_r2_task_records/cti_atlas_r2_r2_1_p1_wd2_<system_id>.json`

Use a task-ID-keyed map and atomic replacement after every completed task. Each task needs:

- Raw decoded output before `<think>` removal.
- Cleaned prediction used by the scorer.
- EM, F1, status, language, and scorer version.
- Gold-answer hash.
- Model and tokenizer revision.
- Chat-template and generation-configuration hashes.
- Input/output tokens, latency, GPU seconds, and segment ID.
- Explicit empty-output or inference-error classification.

The summary files should be disposable derived artifacts, never the resume source.

### G. Reconstruct mean F1 from task records

Yes—but **do not add workload-specific F1 columns to the cost ledger**.

Instead:

- Persist per-task EM/F1 in the authoritative result records.
- Add `protocol_revision`, `run_id`, `record_id`, and `segment_id` to the cost ledger.
- Recompute mean F1, language metrics, pass counts, and subgroup metrics from the union of task records on every resume.
- Store each energy measurement as an append-only segment containing its task IDs; sum segments across resumes. Do not pretend 10 Hz per-task energy estimates are precise.
- Allocate segment energy to tasks only for costing, clearly marked as allocated rather than directly measured.

The old mean F1 and overwritten energy segments cannot be reconstructed from the current ledger. That is another reason the existing W-D2 runs cannot remain canonical.

## Score after these fixes

- **Nobel: 4/10**
- **Turing: 6/10**
- **Combined direction: 5/10**

The repaired Atlas would become a credible, prospective systems-science program. It would not yet be a paradigm-level result: that requires successful frontier non-inferiority, meaningful quality floors, verified cost inversion, W-C1/W-C2 prospective confirmation, and the future-model test. If those all succeed cleanly, the ceiling rises to approximately **Nobel 6/10, Turing 8/10**—still not 9/10 without a general predictive law explaining when scale inversion occurs.

