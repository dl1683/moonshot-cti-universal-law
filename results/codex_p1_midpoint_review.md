## Verdict

Pause P1 after the current inference task/checkpoint. Do **not** start W-D3 or skip the remaining models. The strongest finding is currently a measurement-integrity failure, not a scale-inversion result.

At the 22:42 ET audit snapshot, the ledger had advanced to Qwen-14B at 37/200 = 18.5%, but several headline JSON metrics are corrupted by resume behavior. Gate A cannot be trusted until this is repaired.

**Nobel/Turing score: 3/10 today.** The research question remains potentially 7/10 with prospective confirmation, but current execution integrity is about 2/10.

## Critical blockers

1. **MKQA has 200 episodes, not the frozen 320.** The code selects 40 queries, then removes 15 answerless queries, leaving only 25 queries × 8 translations. The protocol binds 40 × 8. See [workload selection](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\src\cti_atlas_workloads.py:43>) and [post-selection filtering](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\scripts\run_atlas_r2.py:246>).

2. **Resume corrupts summaries.** The runner skips ledger-completed tasks but builds the new summary from only the remaining tasks, then overwrites that system’s JSON entry. Consequently:

   - Qwen-4B JSON contains only its last 80 tasks: 9/80, not ledger-total 25/200.
   - Qwen-14B now contains only its final seven tasks: 0/7, despite ledger-total 37/200.
   - Energy and F1 totals are segment-only.
   - Gate A would currently mis-rank Qwen-14B.

   The defect is at [remaining-task selection](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\scripts\run_atlas_r2.py:260>) through [summary overwrite](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\scripts\run_atlas_r2.py:335>).

3. **W-D3 is budget-incompatible.** The binding design budgets 100 PolicyBench household episodes, but the loader produces **1,970 separate generations** per model. That turns the planned 900 runs into 17,730. The protocol explicitly says 100 households and budgets P1 accordingly at [design lines 69–70](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\results\codex_scale_inversion_atlas_design_gate_r2.md:69>), while the loader emits one episode per reference variable at [workload loader](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\src\cti_atlas_workloads.py:178>).

4. **The MKQA metric is noncanonical and linguistically asymmetric.** It uses whitespace-separated sets, so Japanese and Chinese partial-credit F1 largely degenerates into exact match. It also discards token multiplicity and applies an invented `F1 ≥ 0.5` pass threshold. Official MKQA reports continuous EM/F1 using language-specific normalization and macro-averaging by language; it also explicitly supports answerable/unanswerable scoring. [Official MKQA evaluation](https://github.com/apple/ml-mkqa#evaluation), [MKQA paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00433/108607/MKQA-A-Linguistically-Diverse-Benchmark-for).

5. **The downstream scorer is not operationally valid.** It compares 0–1 rates against a margin of `5.0`, making non-inferiority effectively automatic, while local GPU and energy costs are multiplied by zero. See [scorer margin](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\scripts\score_atlas_r2.py:27>) and [cost calculation](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\scripts\score_atlas_r2.py:114>).

## Answers A–G

**A. Is scoring too strict?**  
Not simply “too strict”—it is the wrong abstraction. Do not loosen the 0.5 threshold after seeing outcomes. Make continuous, official-style F1 and EM primary; report macro-F1 by language and the full distribution. Keep thresholded pass/fail only as a predeclared secondary procurement metric.

The available distributions are extremely zero-inflated:

- Gemma-1B: 175/200 F1=0, median=0; 17 exact matches, 19 passes.
- Falcon-0.5B: 188/200 F1=0, median=0.
- Qwen-4B’s stored distribution covers only 80 tasks and is not representative of its full run.

Raw predictions were not retained, so the missing runs cannot be properly rescored without rerunning or finding external logs.

**B. Do the results support the Atlas thesis?**  
They support two narrower hypotheses:

- Family/training/implementation can matter as much as parameter count.
- Qwen shows a real scale signal: 14B beat 4B by six points on the paired 200 episodes.

They do **not** yet support Scale Inversion. There are no API comparisons, meaningful quality-floor passes, all-in costs, selector predictions, or untouched confirmation results. At 9.5% accuracy, “cheap” is irrelevant unless that accuracy satisfies a real user floor—and it almost certainly does not.

**C. Add early stopping?**  
No. Gate A is frozen at two systems per family on the combined W-D2/W-D3 aggregate. Skipping systems would violate the architecture-coverage commitment at [Gate A](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\results\codex_scale_inversion_atlas_design_gate_r2.md:97>). Moreover, Falcon-3B has only 43 observations, W-D3 is absent, and five local checkpoints remain unresolved. The pattern is nowhere near clear enough.

**D. Required Goliath pass rates?**  
There is no magic absolute Goliath rate. The narrative requires:

- David above a meaningful, frozen quality floor;
- paired lower 95% CI for David minus every frontier API above −5 points;
- at least 10× lower all-in cost than the cheapest qualifying hosted API.

With Gemma-1B currently at 9.5%, a Goliath point estimate above 14.5% already defeats pointwise non-inferiority; uncertainty makes the practical ceiling lower, probably around 10–12% unless their errors are highly paired. But if both systems score around 10%, the result is still not a credible deployment inversion. It is merely parity at unusably low quality.

**E. Likely Gate A selection?**  
Gate A is not presently computable because W-D3 is missing and the JSON is corrupted. If forced to project from the ledger’s MKQA results:

- Qwen: **Qwen-14B + Qwen-4B**. Qwen-4B is within ten points of the 14B best; 0.6B is not.
- Gemma: probably **Gemma-12B + Gemma-1B**, provided 1B remains within ten points after W-D3; otherwise 12B + 4B.
- Falcon: probably **best of 7B/3B + 0.5B**, because the cheapest model will likely remain within ten points in this low-performing family.

The code also mishandles the “same model satisfies both roles” fallback: it can select the next-best model instead of the protocol’s next-cheapest model at [Gate A fallback](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\src\cti_atlas_analysis.py:74>).

**F. Is 9.5% versus 12.5% noise?**  
Yes—entirely plausible noise.

On paired outcomes:

- Difference: Qwen-4B +3 points.
- Discordant outcomes: 17 favor Qwen, 11 favor Gemma.
- McNemar exact p≈0.345.
- Query-cluster bootstrap 95% CI: approximately **−4.5 to +10.5 points**.

There are only 25 independent questions; the eight language translations are correlated. “4× parameter efficiency” is therefore premature. Call it a **candidate family-level inversion**, not an established efficiency law.

**G. Best use of time now?**  
Repair P1 before spending more compute.

1. Stop launching models.
2. Amend W-D2 selection to the first 40 **scorable** queries. The existing 25 can remain a subset, adding 15 queries × 8 languages.
3. Retain raw predictions, score version, per-task EM/F1, language, gold-answer hash, latency, and energy.
4. Replace the scorer with official-style multilingual normalization and macro-F1; predeclare any binary pass rule.
5. Make resume merge task-level records and recompute summaries from the union.
6. Redesign W-D3 as 100 household-level generations or explicitly rebudget 1,970 target-level generations. Do not run its current form.
7. Fix the 5-vs-0.05 non-inferiority units and real all-in cost accounting.
8. Then finish all nine W-D2 systems, run API calibration, and only afterward execute the corrected W-D3.

## Observation adjudication

- Gemma-1B ≈ Qwen-4B: **interesting but statistically unresolved**.
- Falcon-H1 underperforms transformers: **not established**; 0.6B Qwen versus 0.5B Falcon differs by only two points, p≈0.39. The family and implementation are confounded with architecture.
- MKQA too hard versus models too small: **cannot distinguish yet** because scoring and sampling are defective; APIs will provide the necessary ceiling calibration.
- Falcon energy anomaly: **validated for this exact backend**, not for Mamba/SSMs generally. Falcon-0.5B consumed 3.84× Gemma-1B’s energy and took 4.18× as long over the same 200 episodes. That supports “the current Transformers Falcon-H1 path is inefficient,” not “SSMs are intrinsically inefficient.”

