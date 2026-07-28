## Verdict

Do not kill W-D3. Kill its current metric as a Gate A signal.

The workload has useful Atlas value as a **baseline-dominance / no-LLM regime**, but the apparent W-D2/W-D3 scale divergence is not capability evidence. Give current W-D3 **0% Gate A weight**, stop further canonical W-D3 runs at the next durable task boundary, and supersede the measurement contract before proceeding.

Scores below are recommendation strength.

| Recommendation | Score |
|---|---:|
| Freeze current W-D3 as diagnostic-only | **10/10** |
| Give current W-D3 zero Gate A weight | **10/10** |
| Redesign W-D3 into prevalence and challenge panels | **9/10** |
| Make task-native baselines first-class selector candidates | **10/10** |
| Publish this as metric-collapse evidence, not “regex beats AI” | **9/10** |
| Continue Atlas with a mandatory baseline-first preflight | **9/10** |

## What the records actually show

The official scorer gives the all-zero predictor **0.848549 macro score**, not merely because 83.9% of references are exactly zero: its `$50` tolerance also credits nine small nonzero amounts. Every household scores at least 0.625, so every household crosses the implementation’s arbitrary 0.5 “pass” threshold. Yet only **4/100 households are completely correct**, and the baseline misses 309 fields that matter.

Baseline-relative field accounting is more revealing:

| System | Tasks | Model score | Zero baseline on same tasks | Baseline errors rescued | Baseline-correct fields broken |
|---|---:|---:|---:|---:|---:|
| [Qwen3-0.6B](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/cti_atlas_r2_task_records/cti_atlas_r2_r2.1_P1_W-D3_qwen3_0.6b.json>) | 100 | 0.7654 | 0.8485 | **0** | **231** |
| [Qwen3-4B](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/cti_atlas_r2_task_records/cti_atlas_r2_r2.1_P1_W-D3_qwen3_4b.json>) | 100 | 0.5096 | 0.8485 | 47 | 796 |
| [Qwen3-14B](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/cti_atlas_r2_task_records/cti_atlas_r2_r2.1_P1_W-D3_qwen3_14b.json>) | 23 | 0.7312 | **0.8790** | 15 | 127 |

The 0.6B system is therefore strictly dominated by the constant predictor: it rescues **nothing** the baseline gets wrong and only introduces failures.

The committed 14B subset is easier and more zero-heavy than the complete panel. Its correct same-task deficit is **14.8 points**, not the notes’ 11.7-point comparison against the full-panel baseline. The worktree record was advancing during my review, so I used the committed 23-task snapshot requested in your prompt.

There is a faint capability signal under balanced scoring: four-stratum macro accuracy is approximately 0.442 for 0.6B, 0.384 for 4B, and 0.484 for the partial 14B. That supports a U-shaped behavioral story, but none is remotely competent on nonzero numeric calculations.

## A harder blocker: these runs are not canonical R2.1

The current results violate the frozen protocol:

- R2.1 freezes a **128-token** allowance and defines macro household score as primary [in the protocol](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/precommit/atlas_r2_protocol_r2_1.md:44>).
- The runner actually uses **384 tokens and a 120-second timeout** [here](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/scripts/run_atlas_r2.py:612>), despite the 30-second cap.
- Every examined W-D3 output exceeded 128 tokens. All five 0.6B parse failures and all nine 4B parse failures hit exactly 384 tokens; three of four committed 14B parse failures did likewise.
- Gate A uses W-D3’s 0.5-threshold pass rate—not its frozen macro score—and also uses W-D2’s secondary pass threshold instead of primary macro F1 [in the scorer](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/scripts/score_atlas_r2.py:202>).
- The design promises “explanation-field validity,” but the implemented prompt requests only numeric JSON [here](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_atlas_workloads.py:375>).
- Gate A presently accepts partial cells and can return fewer than six anchors.

Thus H-WD3-3 is preliminarily true but mostly mechanical: complex households require longer JSON and hit an undeclared output ceiling. H-WD3-1 and H-WD3-4 describe metric gaming, not capability. H-WD3-2 remains a useful diagnostic hypothesis, but should not justify six more full canonical runs under this contract.

## Answers to your five questions

### 1. Does W-D3 still contribute signal? **Yes, but not the signal in the leaderboard. — 9/10**

The divergence between W-D2 and W-D3 supports one Atlas principle: **there is no workload-independent model ranking**. It does not show that smaller models understand tax policy better.

Do not simply delete zero-heavy households. These appear drawn from a population-based source, so the prevalence regime may be operationally real. Instead create:

- A **prevalence panel** preserving the natural distribution and measuring deployment utility.
- A new, sealed **challenge panel** stratified across nonzero tax liability, refundable credits, benefit eligibility, and mixed complex households.

Do not remove negative fields within challenge households; false positives remain consequential. Score household-clustered:

- positive and negative eligibility recall separately;
- zero and nonzero numeric accuracy separately;
- magnitude error on nonzero amounts;
- parse/completion reliability;
- baseline-relative rescues versus harms;
- hard critical-error floors.

More fundamentally, raw LLM imitation of PolicyEngine is the wrong complete system. PolicyEngine generates the reference and is itself a task-native computation engine. Either W-D3 should validate that the selector chooses PolicyEngine/no LLM, or the AI-bearing task should become **unstructured household intake → validated structured inputs → PolicyEngine execution → grounded explanation**.

### 2. Gate A weighting — **Current W-D3 weight: exactly 0%. — 10/10**

Any positive raw weight rewards models for approximating the constant predictor.

Supersede the 50/50 rule transparently in R2.2 before running Gate A:

1. Select each family’s quality anchor using **W-D2 macro F1**.
2. Select its cheapest checkpoint as the explicit cost/scale anchor, labeled exploratory if it is more than ten points behind.
3. Record W-D3 as `NO_RAW_LLM_QUALIFIES`; do not let it reorder LLM anchors.

After W-D3 redesign, do not return to averaging two incomparable percentages. Rank by:

1. number of workload-specific hard floors passed;
2. worst standardized shortfall relative to the strongest trivial/task-native baseline;
3. cost;
4. memory and stable ID.

That mirrors Gate B’s sounder constrained-selection logic.

### 3. Is the all-zero result publishable? — **Atlas case study 8/10; standalone result 3/10**

“The cheapest AI for tax policy is a regex predicting zeros” is wrong twice:

- It is a constant predictor, not a regex.
- It is not a qualifying complete system once positive-event and critical-error floors exist.

Used as a triumph, it undermines the manifesto. Used as a benchmark-audit result, it strengthens the Atlas:

> The first apparent scale inversion vanished under baseline audit: doing nothing beat every model because the metric rewarded class prevalence rather than policy competence.

Or, more sharply:

> Sometimes the cheapest correct system is no AI—but only after rare, consequential cases are protected by hard floors.

This finding must not count as one of the manifesto’s required frontier-equivalent scale inversions. It is a selector-validity result.

### 4. What should the selector do when a trivial baseline dominates? — **10/10**

Make baselines actual candidate systems, not decorative references:

1. Evaluate constant, field-prior, task-native engine, local model, API, and hybrid candidates.
2. Require every candidate to pass rare-event, safety, reliability, and completion floors before mean quality matters.
3. Eliminate any candidate dominated in both quality and cost.
4. If a baseline qualifies, select it.
5. If a baseline qualifies only on an outcome-blind subset, permit a calibrated cascade: baseline for safe cases, specialist/model/API for uncertain cases.
6. If nothing qualifies, return `NO_QUALIFYING_SYSTEM`; never force an AI recommendation.

For current W-D3, all-zero does not qualify under meaningful positive-recall floors. Qwen3-0.6B is dominated by it anyway. PolicyEngine is the likely qualifying low-cost system.

The selector’s output taxonomy should become:

`NON_NEURAL | LOCAL_AI | HYBRID | CLOUD_AI | NO_QUALIFYING_SYSTEM`

### 5. Momentum — **honest signal 8/10; execution discipline 4/10**

You are gaining honest signal, but spending too much compute to learn cheap truths.

This is not a twelfth killed research direction. It is a preventable measurement-gate failure: label prevalence, an all-zero baseline, output-length feasibility, and task-native software should have been tested before any nine-model screen.

My steering decision:

- Stop treating further current-format W-D3 execution as canonical at the next durable boundary.
- Preserve existing records as `diagnostic_superseded`.
- Freeze an R2.2 amendment.
- Install a mandatory P0 baseline audit for every future workload: prevalence, trivial baseline, task-native baseline, metric-gaming probes, output-cap feasibility, and two-system smoke.
- Continue Atlas. Do not pivot programs again.

Current scientific score remains approximately **Nobel 4/10, Turing 6/10, Fields 0/10**. This finding does not raise it. A corrected baseline-first Atlas with prospective confirmation still has a credible **combined 7/10** route.

The blunt version: **the Atlas survived because you found the flaw before confirmation. Now prove that it can refuse a seductive but invalid “cheap win.”**

