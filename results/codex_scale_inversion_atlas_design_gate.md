# Scale-Inversion Atlas: Exact Experimental Protocol

**Design-gate date:** 2026-07-27  
**Architectural verdict:** **CONDITIONAL GO**  
**Primary unit of analysis:** the user's completed task under a declared workload, hardware, volume, quality floor, and reliability floor  
**Primary endpoint:** avoided all-in cost relative to the cheapest qualifying comparator, never a composite efficiency score

This protocol implements the R13 decision in `results/codex_steering_r13_r2.md`.
It is not a static leaderboard, a universal scaling law, or a router. It is a
prospective constrained selector over complete deployable systems:

\[
s^\star(w,h,V,\tau)=\arg\min_{s\in\mathcal S} C_{\mathrm{all-in}}(s,w,h,V)
\]

subject to

\[
Q(s,w)\geq Q_{\min},\quad
R(s,w)\geq R_{\min},\quad
L_{95}(s,h)\leq L_{\max},\quad
M_{\mathrm{peak}}(s,h)\leq M_{\max}.
\]

The Atlas may return **NO FEASIBLE LOCAL SYSTEM**. An abstention is a correct
selection when every candidate violates a hard floor.

## 0. Gate conditions

No GPU sweep starts until all five conditions pass:

1. Every workload has a pinned upstream revision, license record, SHA-256
   manifest, deterministic adjudicator, and task-level leakage boundary.
2. The RealClawBench package can be reproduced from its released environments
   and deterministic verifiers. Its current early-release status makes this a
   real preflight risk.
3. The nine primary checkpoints load from the local Hugging Face cache at the
   pinned revisions below, and every declared quantization mode either passes a
   32-case smoke test or is recorded as `INFEASIBLE`. No silent substitution.
4. Hidden adjudicators are unavailable to generation, retrieval, training,
   best-of-\(n\) selection, and RLVR. Only train/public verifiers may be used
   before final scoring.
5. The manifest generator and an independent checker agree on source paths,
   hashes, split membership, model revisions, recipes, seeds, price
   assumptions, and the sealed confirmation boundary.

If condition 1, 2, or 4 cannot be made true, the Atlas does not run. Those are
scientific-validity failures, not engineering delays.

## 1. Workload families

The experiment uses benchmark packaging only when the underlying instances are
real queries, real households, real issue reports, or real deployed-agent
sessions. It does not use MMLU, GSM8K, HumanEval, synthetic conflict sets, or
other canonical model-centric benchmarks as Atlas outcomes.

### W-D1: Fresh repository repair — discovery

- **Source:** `SWE-bench-Live/MultiLang`, frozen at the upstream release that
  includes the 2026-05-16 update.
- **Underlying user unit:** a real GitHub issue and repository state.
- **Languages:** C/C++, C#, Java, TypeScript/JavaScript, Go, and Rust.
- **Sample:** 144 tasks, exactly 24 per language. Within each language, sort
  eligible task IDs by `SHA256(upstream_revision || task_id)` and take the
  first 24. Do not filter by observed model success or issue difficulty.
- **Adjudicator:** the upstream container plus fail-to-pass and
  pass-to-pass executable tests.
- **Primary quality:** repository issue resolution rate.
- **Secondary quality:** pass-to-pass preservation rate and compile rate.
- **Reliability floor:** at least 98% of attempted tasks leave the sandbox
  intact, produce a parseable patch, and do not regress pass-to-pass tests.
- **Why it qualifies:** the tasks come from real issue/PR histories and the
  May 2026 release provides executable Linux containers across six languages.
- **Why SWE-bench Verified is rejected:** it is old, contamination-prone, and
  OpenAI's 2026 audit reports that flawed tests now reject many functionally
  correct solutions. It can appear only as a compatibility diagnostic, never
  as an Atlas claim surface.

### W-D2: Multilingual knowledge access — discovery

- **Source:** Apple `ml-mkqa`, pinned upstream dataset and official evaluator.
- **Underlying user unit:** a Google Natural Questions search query; MKQA
  supplies human translations and language-independent answer annotations.
- **Sample:** 200 unique query IDs across eight predeclared languages:
  `en`, `ar`, `km`, `ms`, `th`, `tr`, `vi`, and `zh_cn`, for 1,600
  query-language tasks. Select 25 query IDs from each official answer-type
  stratum by ascending `SHA256(dataset_revision || example_id)`; if a stratum
  has fewer than 25, allocate the shortfall in fixed answer-type order.
- **Adjudicator:** the official MKQA normalization and alias evaluator, with
  Wikidata QID exactness where a QID exists.
- **Primary quality:** macro-average answerable exact match across languages.
- **Secondary quality:** token F1, unanswerable accuracy, entity-QID accuracy,
  and worst-language exact match.
- **Reliability floor:** worst-language exact match at least 50% of the
  macro-average, unanswerable false-answer rate no more than 10%, and at least
  99% parseable responses.
- **RAG corpus:** a pinned 2026-06 Wikipedia/Wikidata snapshot. The gold answer
  table is never indexed.
- **Why it qualifies:** the source queries were issued by real search users;
  translation and adjudication are independently produced, and the official
  evaluator is deterministic.

### W-D3: Household tax and benefit decisions — discovery

- **Source:** PolicyBench US snapshot for tax year 2026, PolicyEngine-US, and
  its pinned populace-US household sample.
- **Underlying user unit:** one representative US household seeking tax and
  benefit outputs.
- **Sample:** all 100 published PolicyBench households, 18 scored outputs, and
  1,984 household-output targets. No subsampling.
- **Adjudicator:** pinned PolicyEngine-US reference outputs. Amounts are
  correct within 1% with a one-dollar tolerance at zero; binary eligibility is
  exact 0/1.
- **Primary quality:** household-impact-weighted exact match.
- **Secondary quality:** within-1% rate, amount accuracy, participation
  accuracy, output coverage, and worst-group accuracy by income quartile,
  household composition, and state.
- **Reliability floor:** at least 95% schema coverage, no more than 1% false
  positive eligibility on high-cost programs, and worst-group exact match no
  more than 10 points below the overall rate.
- **RAG corpus:** only pinned public program guidance and tax-year documents.
  PolicyEngine outputs and source formulas are excluded from the neural RAG
  index.
- **Required non-LLM baseline:** direct PolicyEngine execution. If it satisfies
  the user's interface and latency constraints, it should win. Hiding this
  baseline would turn the Atlas into model advertising.
- **Why it qualifies:** the cases are sampled from representative household
  microdata and adjudicated by an independently maintained executable
  tax-benefit model.

### W-C1: Real deployed-assistant work — sealed confirmation

- **Source:** RealClawBench frozen main release, 281 tasks reconstructed from
  real OpenClaw developer-agent sessions.
- **Underlying user unit:** one real deployed-agent request.
- **Scope:** all 281 released tasks; no sampling after seal.
- **Adjudicator:** case-specific deterministic Python verifiers in the
  reconstructed environments.
- **Primary quality:** mean task success.
- **Secondary quality:** subtask success, pass@3 diagnostic, and category
  success for file, code, data, command, and project work.
- **Reliability floor:** at least 95% valid terminations, zero forbidden
  external side effects, no verifier tampering, and worst-category success no
  more than 15 points below overall success.
- **Seal:** task contents, verifier code, reference artifacts, and per-category
  outcomes are encrypted or moved outside the working tree. Before selector
  freeze, the team may see only the dataset card, aggregate category counts,
  runtime requirements, and the manifest commitment.
- **Confirmation candidates:** exactly 12, defined in Section 2. The selector,
  six system predictions, and three volume-specific choices are signed before
  the seal is opened.
- **Why it qualifies:** its tasks come from a large real-session pool and use
  reconstructed environments with deterministic verifiers.

### Source independence and contamination policy

- W-D1, W-D2, W-D3, and W-C1 come from four different organizations and four
  different collection processes.
- Public discovery workloads may be contaminated in pretrained models; this is
  measured as a threat and is not called prospective.
- W-C1 postdates every primary checkpoint in the initial roster. Exact-match
  n-gram and issue/answer leakage scans are run against all adaptation corpora.
- Any task whose gold answer, gold patch, verifier assertion, or reference
  artifact appears in training or retrieval is removed by a rule fixed before
  model outputs are inspected. If more than 5% of any family is removed, the
  family is invalidated rather than silently repaired.

## 2. System configurations

### 2.1 Primary checkpoint roster

The roster has three deployable instruction-tuned families and three scale
points per family. All revisions are the revisions currently present in the
local model cache.

| Family | Role | Checkpoint | Local revision |
|---|---|---|---|
| Qwen3 dense Transformer | small | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen3 dense Transformer | middle | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Qwen3 dense Transformer | large | `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Gemma 3 dense Transformer | small | `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| Gemma 3 dense Transformer | middle | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` |
| Gemma 3 dense Transformer | large | `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` |
| Falcon-H1 Transformer-Mamba hybrid | small | `tiiuae/Falcon-H1-0.5B-Instruct` | `8f2587ca06bff78d8fa1adfccbe8c24d5f86b368` |
| Falcon-H1 Transformer-Mamba hybrid | middle | `tiiuae/Falcon-H1-3B-Instruct` | `01087ec4c132d7f186908716b3530ea187f062a1` |
| Falcon-H1 Transformer-Mamba hybrid | large | `tiiuae/Falcon-H1-7B-Instruct` | `41e72f27effbab80cd45b6e884688452253a3686` |

The roster intentionally uses families with instruction-tuned scale ladders
that can run as complete systems on a 24 GB card.

### 2.2 Explicit exclusions

- **Llama 4 Scout and Maverick:** local cache entries contain metadata but no
  weights; total MoE weights do not fit one 24 GB GPU, and the directory does
  not provide two locally deployable Llama 4 scale points. Including an API or
  CPU-offloaded system would change the declared hardware. Exclude.
- **Pure state-spaces Mamba/Mamba2:** the cached ladders are base language
  models, not comparable instruction systems. Falcon-H1 supplies the runnable
  Transformer-Mamba ladder. Pure Mamba is allowed only in a separately labeled
  architecture diagnostic.
- **RWKV7 1.5B and 2.9B:** cached weights are base/world checkpoints without a
  matched instruct ladder or a proven common QLoRA/RLVR path. Keep a 32-task
  smoke diagnostic, but do not train the selector on it.
- **Qwen3 30B-A3B/32B, Falcon-H1 34B, OLMo 32B:** they either exceed the
  all-in 24 GB constraint at the required context or are incomplete locally.

An excluded family may enter a later Atlas version only through a new protocol
addendum signed before its outputs are observed.

### 2.3 Quantization and inference contract

- **High precision:** BF16 weights and BF16 activations for the six small and
  middle checkpoints.
- **Eight bit:** bitsandbytes LLM.int8 weights, BF16 activations for the three
  large checkpoints.
- **Four bit:** bitsandbytes NF4, double quantization enabled, BF16 compute for
  all nine checkpoints. QLoRA adapters use this same base representation.
- **Engine:** one pinned Hugging Face Transformers/Accelerate/bitsandbytes
  environment, SDPA attention, batch size 1, deterministic greedy generation
  except the explicitly declared best-of-\(4\) arm.
- **Engine sensitivity audit:** the six small/large anchors are rerun with the
  best supported vLLM or llama.cpp deployment engine. These are audit runs, not
  selector-training rows. If the cheapest feasible system changes in more than
  20% of user profiles, the single-engine Atlas is invalid and must expand the
  candidate definition.

### 2.4 Exact 51-entry configuration roster

There are **48 neural configuration templates plus three task-native
baselines, 51 total roster entries**.

#### Raw/quantized systems: 18

1. Nine checkpoints in NF4 four bit.
2. Six small/middle checkpoints in BF16.
3. Three large checkpoints in eight bit.

#### Small/large anchor augmentations: 24

For the small and large NF4 checkpoint in each family, six anchors total:

1. Six RAG systems.
2. Six QLoRA systems.
3. Six QLoRA-plus-RAG systems.
4. Six best-of-\(4\)-plus-public-verifier systems.

The best-of-\(4\) arm uses temperature 0.7, top-p 0.95, four independent fixed
seeds, and a workload-specific public verifier. It may use public tests,
schema checks, citation support, type checks, and self-consistency. It may not
use the hidden adjudicator or gold answer.

#### Student elicitation systems: 6

For the small NF4 checkpoint in each family:

1. Three teacher-distilled students. The teacher is the large checkpoint from
   the same family. Teacher generation, failed candidates, public verification,
   and student training are all charged.
2. Three RLVR students. Reward comes only from the public/train deterministic
   verifier; all rollouts and verifier calls are charged.

#### Task-native baselines: 3

1. Universal abstain/no-op.
2. MKQA retrieval-only BM25/Wikidata answer baseline.
3. Direct PolicyEngine execution for W-D3.

The three baselines are evaluated only where their interface is meaningful.
They remain roster entries so the Atlas cannot manufacture an LLM advantage by
omitting a cheaper deterministic solution.

### 2.5 Adaptation recipes

All recipes are frozen once, not tuned independently per model.

- **QLoRA:** rank 16, alpha 32, dropout 0.05, all linear projections, AdamW,
  learning rate \(2\times10^{-4}\), cosine decay with 3% warmup, effective
  batch 32, three epochs, maximum 2,000 optimizer steps, one discovery seed.
  The final selected small and large system is repeated with seeds 17, 29, and
  41.
- **Distillation:** 2,000 training tasks per discovery family, four teacher
  candidates per task, accept only public-verifier-valid candidates, train the
  small student with 0.5 supervised gold loss plus 0.5 accepted teacher-output
  loss, same optimizer budget as QLoRA.
- **RLVR:** LoRA policy on the small NF4 model, 512 prompts, eight rollouts per
  prompt, GRPO, KL coefficient 0.05, maximum 500 optimizer steps, and a hard
  ceiling of six GPU-hours per family/model cell.
- **RAG:** hybrid BM25 plus `Qwen/Qwen3-Embedding-0.6B`, top 20 lexical/dense
  union, rerank to top 6, fixed 8,192-token context ceiling. Index construction
  and embedding energy are charged.
- **Generation ceilings:** 8,192 input tokens and 2,048 output tokens for W-D1
  and W-C1; 4,096 input and 256 output for W-D2 and W-D3. Timeouts score zero
  and keep their consumed cost.

The initial discovery matrix therefore contains 48 neural templates across
three discovery workloads, plus the applicable task-native baselines. W-C1
confirmation evaluates exactly 12 candidates: the six small/large raw NF4
anchors and one discovery-locked augmentation for each anchor. The augmentation
choice is frozen before the W-C1 seal opens. Any adapted confirmation candidate
uses an adapter trained only on the pooled discovery corpora and is applied
unchanged to W-C1; no W-C1 example is used for adaptation.

For count clarity:

- the reusable roster contains **51 configuration templates**;
- the unpruned discovery design contains 144 neural
  configuration-by-workload cells plus five applicable task-native baseline
  cells, **149 discovery cells**;
- confirmation adds 12 frozen cells;
- the engine sensitivity audit adds six mixed-suite cells;
- the initial protocol therefore schedules at most **167 executed cells**
  before the external future-model event. Successive halving reduces task
  executions inside those cells but does not redefine the roster.

## 3. Measurement protocol

### 3.1 End-to-end task boundary

For one assigned task, timing and energy begin immediately before retrieval,
prompt construction, or agent startup and end after parsing, public
verification, retries, and final artifact write. Hidden adjudication runs
outside this boundary and is reported separately. Model load is measured as a
separate cold-start cost and amortized under a declared session length.

Crashes, invalid outputs, and timeouts:

- consume their actual energy and time;
- receive quality zero;
- count in the assigned-task denominator;
- are never removed as outliers.

The primary energy statistic is gross GPU-board joules per assigned task.
Dynamic joules above idle and joules per verified success are secondary.

### 3.2 Quality

- **W-D1:** resolved / 144, pass-to-pass preservation, compile success.
- **W-D2:** macro answerable exact match across eight languages; official F1,
  unanswerable accuracy, QID exactness, and worst-language exact match.
- **W-D3:** household-impact-weighted exact match; within-1%, amount,
  participation, coverage, and worst-group rates.
- **W-C1:** mean deterministic task success, subtask success, valid
  termination, and worst-category success.

Report 10,000-replicate cluster bootstrap 95% confidence intervals. Clusters
are repository for W-D1, query ID for W-D2, household for W-D3, and source
session/task for W-C1. Noninferiority uses a precommitted three-percentage-point
margin. Hypothesis-level multiple tests use Benjamini-Hochberg FDR 0.05.

### 3.3 Energy

- Sample `nvmlDeviceGetPowerUsage` through `pynvml` every 50 ms in a dedicated
  process and integrate power by the trapezoid rule.
- Record raw timestamped milliwatts, GPU utilization, clock, P-state,
  temperature, and power limit.
- Measure idle board power for five minutes before each block. Gross joules are
  primary; idle-subtracted joules are secondary.
- Run five warmups, then three measurement blocks in counterbalanced system
  order. Cool to within 3 C of the block-start temperature before the next
  block.
- No other GPU process may run during an energy block.
- Adaptation energy includes teacher generation, rejected samples, retrieval
  index construction if GPU-backed, optimizer steps, evaluation used for early
  stopping, and checkpoint serialization.

### 3.4 Latency and throughput

- Use `time.perf_counter_ns()` around the end-to-end boundary.
- Report median, p95, p99, time to first token, inter-token latency, and tasks
  per hour.
- Batch size is one for the primary user-facing result. A batch-8 throughput
  diagnostic is reported separately and never substituted for latency.
- A task timeout is 20 minutes for W-D1/W-C1 and 120 seconds for W-D2/W-D3.

### 3.5 Peak memory

- Reset and record `torch.cuda.max_memory_allocated()` and
  `torch.cuda.max_memory_reserved()` for PyTorch runs.
- Independently sample per-process and device-used memory from NVML every
  50 ms. NVML peak process memory is the primary cross-engine measure.
- Record host RAM and page-file peak through `psutil`.
- A system is locally feasible only if peak GPU memory is no more than 22.0 GB,
  leaving 2.0 GB operational headroom on the 24 GB card.

### 3.6 Reliability and safety

Reliability is a hard constraint, not a term in a score:

- schema/parse success;
- timeout and crash rate;
- worst-group or worst-category quality;
- forbidden tool-call and external-side-effect count;
- sandbox escape, verifier tampering, and gold-artifact access;
- false positive high-cost eligibility for W-D3;
- unanswerable hallucination for W-D2;
- pass-to-pass regression for W-D1.

Any sandbox escape, gold/verifier access, or undeclared external side effect
makes that system configuration infeasible regardless of mean quality.

### 3.7 All-in cost and amortization

For deployment volume \(V\):

\[
C_{\mathrm{all-in}}(s,w,h,V)=
\frac{C_{\mathrm{adapt}}+C_{\mathrm{teacher}}+C_{\mathrm{index}}+
C_{\mathrm{tune}}+C_{\mathrm{deploy}}}{V}
+C_{\mathrm{inference/task}}+C_{\mathrm{retrieval/task}}+
C_{\mathrm{verifier/task}}.
\]

Report this at \(V\in\{1{,}000,10{,}000,100{,}000\}\).

Every cost component has three parallel units:

1. physical: joules and GPU-seconds;
2. operational: wall seconds and peak memory;
3. monetary: USD from a frozen electricity price, actual or replacement GPU
   cost, 10,000-hour base lifetime, and measured paid API/data costs.

The monetary conclusion must survive sensitivity analyses at 5,000, 10,000,
and 20,000 GPU-hour lifetimes and at 0.5x, 1x, and 2x the frozen electricity
price. Human research labor is reported separately and is not hidden inside
GPU cost.

**Avoided cost** is:

\[
1-\frac{C_{\mathrm{all-in}}(\hat s)}
{C_{\mathrm{all-in}}(s_{\mathrm{reference}})}
\]

where both \(\hat s\) and the reference meet every identical hard floor. No
savings claim is permitted against a reference that fails quality or safety.

### 3.8 Constraint profiles

For each workload, evaluate low, operational, and high quality floors at all
three volumes, for 36 precommitted user profiles:

| Workload | Low | Operational | High |
|---|---:|---:|---:|
| W-D1 issue resolution | 10% | 20% | 35% |
| W-D2 multilingual exact match | 45% | 60% | 75% |
| W-D3 weighted exact match | 70% | 85% | 95% |
| W-C1 task success | 25% | 40% | 55% |

The reliability floors in Section 1 apply at every quality level. Primary
latency is p95 no more than 120 seconds for W-D2/W-D3 and 20 minutes for
W-D1/W-C1. Peak memory is always 22.0 GB or less.

These thresholds are frozen before the raw sweep. If no system reaches a high
floor, the correct answer is no feasible local system.

## 4. Frozen hypothesis ledger

### ATLAS-H0: Prospective constrained selection

**Claim.** A discovery-trained selector can identify the cheapest feasible
system under absolute constraints on an untouched workload family.

**Confirmation success.**

1. Feasibility classification is correct for at least 8 of the 9 W-C1
   quality-volume profiles.
2. There are zero false-feasible choices with a hard safety violation.
3. Among profiles with a feasible oracle, median cost regret is at most 10%
   and maximum regret at most 25%.
4. Quality forecast mean absolute error is at most 3 percentage points;
   p95-latency and energy forecasts are each within 25%.
5. The selector beats all three preregistered rules by at least 20% in median
   cost regret: largest model, smallest model passing the calibration sample,
   and PER-style composite rank.

Fail any item and the prospective-selector claim fails.

### PC-H1: Prior-conflict crossover window

PC-H1 is evaluated only on tasks with a closed structured action space or a
canonical correct output sequence. Free-form patches and tasks without a
well-defined competing action are ineligible rather than assigned an invented
conflict value. Define pre-adaptation conflict \(C\) on eligible calibration
tasks as the mean positive length-normalized log-odds assigned to the modal
wrong canonical action over the correct canonical action. Define latent support
\(S\) as base verified pass@32.

**Claim.** Conditional on \(S\) exceeding the operational quality floor,
small-minus-large adapted quality has a concave conflict response:

\[
\Delta Q=\beta_0+\beta_1 C+\beta_2 C^2+\beta_3 S+\beta_4\log B+\epsilon,
\quad \beta_1>0,\ \beta_2<0.
\]

**Success.**

1. Both signs hold with bootstrap 95% intervals excluding zero in discovery.
2. The estimated positive crossover window contains at least 20% of eligible
   discovery cells and appears in at least two workload families.
3. Frozen window membership predicts the sign of the small-large gap on at
   least 75% of eligible confirmation cells. If W-C1 exposes no eligible
   structured cells, PC-H1 remains unconfirmed and cannot be called a success.
4. The predicted adaptation-data crossover is within 2x and final quality
   within 3 points without workload-specific intercepts.

Otherwise PC-H1 fails. A monotone conflict claim is prohibited.

### LS-H1: Latent solvability predicts RLVR success

**Claim.** Base-model verified pass@32 predicts whether RLVR will cross the
operational pass@1 quality floor.

**Success.**

1. Leave-one-workload-out discovery AUC is at least 0.80.
2. Spearman correlation between base pass@32 and RLVR pass@1 gain is at least
   0.60 with a 95% interval above zero.
3. The frozen threshold achieves balanced accuracy at least 0.75 on eligible
   W-C1 tasks or categories.
4. No model with pass@32 below half the floor is incorrectly predicted to
   become feasible.

If confirmation has no valid RLVR interface, LS-H1 remains unconfirmed rather
than being scored as success.

### EC-H1: Elicitation can be cheaper than direct scale

**Claim.** When small-model base pass@32 already exceeds the quality floor,
compressing coverage into pass@1 with RLVR is cheaper than either best-of-\(n\)
or deploying the large family member at sufficient volume.

**Success.**

1. RLVR reaches within 3 points of base pass@32.
2. It satisfies the same reliability floor.
3. It has lower all-in cost than both comparators at \(V=10{,}000\) and
   \(100{,}000\).
4. The median saving is at least 2x in two discovery families and the direction
   is correct on confirmation.

Training energy, rejected rollouts, and verifier cost are mandatory. Omit any
one and the hypothesis is void.

### VA-H1: Volume changes the cheapest feasible system

**Claim.** Adaptation cost creates predictable deployment-volume crossovers.

**Success.**

1. The oracle system changes between 1K and 100K volume in at least two
   discovery families.
2. The frozen selector predicts the W-C1 break-even volume within 2x.
3. The same direction survives all declared hardware/electricity sensitivity
   cases.

### Q-H1: Four-bit deployment preserves the decision

**Claim.** NF4 deployment is noninferior in quality to the family's
higher-precision comparator while reducing energy or memory enough to alter
user feasibility.

**Success.**

1. Quality loss is no more than 2 points with the lower 95% confidence bound
   above -3 points.
2. Peak memory falls at least 35%.
3. Gross joules per task fall at least 20% or the system becomes feasible under
   the memory constraint.
4. The result holds at two scale points in at least two families.

No hypothesis may be added after confirmation is opened. Post-hoc findings go
in an explicitly exploratory ledger.

## 5. Prospective confirmation and the 7/10 bar

### 5.1 Selector freeze

The freeze object contains:

- canonical JSON manifest of every source path and hash;
- 51-entry roster and 12-entry confirmation shortlist;
- feature definitions and transformations;
- model coefficients or executable selector byte hash;
- missing-data and abstention rules;
- all 36 constraint profiles;
- hypotheses, thresholds, and kill criteria;
- environment, model, prompt, adapter, retrieval, and engine hashes;
- price and hardware-lifetime assumptions;
- a random 256-bit salt committed before W-C1 reveal.

Use RFC 8785 canonical JSON and SHA-256. Store the preimage privately, publish
the commitment, and have the independent checker reject duplicate keys,
path/hash mismatches, unbound inputs, or any W-C1 task/verifier leakage.

Opening W-C1 is irreversible. After opening, no selector, feature, candidate,
prompt, recipe, or threshold change is a confirmation result.

### 5.2 What counts as the sealed-family result

W-C1 is the untouched confirmation family. The result is successful only if
ATLAS-H0 passes. A good average score, a visually plausible Pareto plot, or a
correct small-versus-large direction does not substitute for ATLAS-H0.

### 5.3 What counts as a future-model test

The 7/10 claim additionally requires a model that did not exist when the
selector commitment was published.

The eligible future event is the **first** public open-weight instruction
family after the freeze that:

1. publishes at least two dense or fixed-active-parameter sizes between 0.5B
   and 14B;
2. permits local evaluation;
3. fits the 22.0 GB runtime ceiling in a declared quantization;
4. is supported by the frozen interface or a purely syntactic adapter whose
   code is written without seeing task outcomes.

No provider or family may be skipped because its result looks inconvenient.

Before full evaluation, the selector receives only model-card metadata, the
same 32-task calibration budget used in discovery, and hardware measurements
from those 32 tasks. It predicts:

- quality, p95 latency, joules, and peak memory;
- the cheapest feasible future-model configuration at each volume;
- whether adding the new family changes the incumbent Atlas choice.

Evaluate two sizes in raw NF4 and the single augmentation predicted by the
selector: four new candidates. Compare them with the frozen 12, for a
16-candidate oracle.

The workload is the first RealClawBench live/later-window batch collected after
the future model's release, or an equivalently committed real-session batch.
This prevents the future checkpoint from training on the confirmation tasks.

**Future-model success:** all five ATLAS-H0 criteria pass on the future batch,
and the selector correctly predicts whether the future family displaces the
incumbent in at least 8 of 9 quality-volume profiles.

If no eligible family or later-window batch appears during the initial eight
weeks, Atlas v1 can earn 5.5-6/10, not 7/10. A future release cannot be
scheduled into existence.

## 6. Timeline and compute

The initial program is capped at **320 measured GPU-hours** and eight calendar
weeks. The future-model event is outside this cap.

| Phase | Calendar | GPU hours | Work |
|---|---:|---:|---|
| P0 provenance and preflight | week 1 | 12 | hashes, licenses, containers, 32-case model/quant smoke tests |
| P1 harness and metrology | week 1-2 | 18 | task adapters, NVML sampler, cold/warm controls, failure injection |
| P2 raw/quant discovery matrix | week 2-3 | 55 | 18 raw systems, three discovery families, successive-halving only after fixed pilot |
| P3 RAG and best-of-4 | week 3-4 | 42 | retrieval indices, six anchor ladders, public-verifier accounting |
| P4 QLoRA, distillation, RLVR | week 4-5 | 105 | 18 QLoRA adapters, nine distillation cells, nine capped RLVR cells |
| P5 replication and selector fit | week 5-6 | 28 | three-seed finalists, bootstrap, baselines, profile oracle |
| P6 freeze and W-C1 confirmation | week 6-7 | 45 | 12 frozen candidates on 281 sealed tasks |
| P7 engine sensitivity and audit | week 7-8 | 15 | six anchor engine checks, checker, final decision |
| **Total** | **8 weeks** | **320** | |

### Successive-halving rule

P2 runs every raw system first on 25% of each discovery sample. A system is
pruned only if the upper 95% quality bound is below the lowest quality floor or
it violates memory/safety. All systems within 5 points of a floor continue.
Pruning uses no composite score and cannot remove the highest-quality member of
any model family.

### Parallel work

Parallelizable without contaminating power measurements:

- dataset acquisition, license review, hashing, and container builds;
- CPU test execution for the previous W-D1 generation batch while the GPU
  generates the next batch;
- retrieval indexing on CPU;
- verifier unit tests, leakage scans, and bootstrap analysis;
- documentation and independent-checker work.

Serial on the single GPU:

- all energy blocks;
- adapter training;
- teacher generation;
- RLVR rollouts;
- final confirmation generation.

No two GPU workloads overlap during metered runs. Calendar feasibility assumes
roughly 40-50 measured GPU-hours per week, leaving thermal cooldown, failed
jobs, and human audit time. Exceeding 320 hours requires a new signed spend
gate, not a quiet extension.

## 7. Program kill criteria

Kill the **scientific Atlas program**, not merely one hypothesis, if any one of
these occurs:

1. W-C1 cannot be kept untouched or its deterministic environments cannot be
   reproduced.
2. Hidden gold, hidden tests, verifier assertions, or reference artifacts leak
   into generation, retrieval, training, reward, or best-of-\(n\) selection.
3. More than 5% of a workload is invalidated by broken adjudication,
   contamination, or missing provenance.
4. Quality or reliability floors are changed after any model output is seen.
5. ATLAS-H0 fails on W-C1 and the selector does not beat the best naive rule by
   at least 20% median cost regret.
6. The selector requires workload-name intercepts, model-family IDs, or manual
   case labels that are unavailable to a new user.
7. No qualifying system saves at least 2x all-in cost against the large-model
   reference at an identical operational floor in at least two of four
   workload families.
8. Claimed savings disappear at all three amortization volumes or under the
   declared electricity/hardware-lifetime sensitivity grid.
9. Small-system wins occur only because the large system lacked the same
   documents, adaptation data, verifier access, context limit, or optimization
   opportunity.
10. A task-native deterministic system is omitted despite being cheaper and
    feasible.
11. Rankings change enough under the engine audit to alter more than 20% of
    user-profile decisions and the program does not expand the engine axis.
12. Energy differences are smaller than measurement repeatability: block-level
    coefficient of variation exceeds 10% after thermal control, or the claimed
    saving is below twice the metrology error.
13. The 320 GPU-hour cap is reached before selector freeze with fewer than
    three valid discovery families.
14. A novelty audit identifies an existing system that already performs
    absolute-constraint selection, includes adaptation amortization, and passes
    prospective unseen-family and future-model tests.
15. The main deliverable reduces to a matrix, Pareto plot, or PER-like score
    and cannot make a falsifiable pre-run choice for W-C1.

Criterion 15 is the explicit **“just a benchmark paper”** kill. A useful public
matrix may still be released as a 5/10 resource, but the scientific program is
closed rather than redescribed as successful.

Do **not** kill the Atlas merely because the oracle prefers a large model in
some or most regimes. “Use the large system here” is a valid user result.

## 8. Exact files to create

The current design-gate artifact is:

- `results/codex_scale_inversion_atlas_design_gate.md` — architectural
  decision, exact experiment, success criteria, and implementation map.

Create these human source-of-truth files next:

1. `research/SCALE_INVERSION_ATLAS_PROTOCOL.md` — locked canonical protocol;
   copied from this gate only after preflight facts are resolved.
2. `research/SCALE_INVERSION_ATLAS_HYPOTHESIS_LEDGER.md` — immutable
   confirmatory hypotheses and exploratory appendix.
3. `research/SCALE_INVERSION_ATLAS_DATA_CARD.md` — upstream revisions,
   licenses, sampling, contamination scans, and adjudicator contracts.
4. `research/SCALE_INVERSION_ATLAS_SELECTOR_CARD.md` — features, calibration
   budget, constraints, abstention, oracle, regret, and limitations.
5. `research/SCALE_INVERSION_ATLAS_KILL_LOG.md` — append-only gate outcomes and
   stopped branches.

Create these scripts, preserving the repository's `cti_*.py` convention:

1. `src/cti_atlas_manifest.py` — canonical manifest and commitment generator.
2. `src/cti_atlas_workloads.py` — workload acquisition, frozen sampling, and
   adjudicator adapters.
3. `src/cti_atlas_systems.py` — 51-entry roster, prompt/runtime contracts,
   quantization, retrieval, and generation.
4. `src/cti_atlas_measure.py` — NVML energy, latency, memory, cold/warm, and
   failure accounting.
5. `src/cti_atlas_adapt.py` — QLoRA, distillation, and RLVR frozen recipes.
6. `src/cti_atlas_selector.py` — feature extraction, constraint feasibility,
   prediction, oracle, and regret.
7. `src/cti_atlas_confirm.py` — seal verification and one-way confirmation
   runner.
8. `src/cti_atlas_independent_checker.py` — duplicate-key rejection,
   path/hash validation, split/leak checks, and generator/checker parity.
9. `src/cti_atlas_report.py` — tables, Pareto surfaces, avoided-cost profiles,
   and claim-bounded report.

Create these canonical result files:

1. `results/cti_atlas_manifest.json` — complete frozen machine contract.
2. `results/cti_atlas_preflight.json` — gate-condition and negative-fixture
   evidence.
3. `results/cti_atlas_discovery.json` — discovery measurements and profile
   oracle.
4. `results/cti_atlas_hypotheses.json` — frozen hypothesis fits and decisions.
5. `results/cti_atlas_selector_freeze.json` — selector commitment and
   confirmation predictions.
6. `results/cti_atlas_confirmation.json` — W-C1 reveal receipt and outcomes.
7. `results/cti_atlas_future_model.json` — initially `PENDING_EXTERNAL_EVENT`;
   later the future-model precommit and result.
8. `results/cti_atlas_audit.json` — independent checker, leakage,
   repeatability, and engine-sensitivity evidence.
9. `results/cti_atlas_decision.json` — final machine-readable
   `PASS`, `RESOURCE_ONLY`, `KILL`, or `PENDING_FUTURE_MODEL` verdict.

Per-task raw generations, adapters, indexes, and power traces remain local
gitignored caches named `results/cti_atlas_cache_*`; only hashes, sufficient
audit statistics, and canonical summaries are committed.

## 9. Final sign-off

This design is a **CONDITIONAL GO** for implementation through P0 only.

P0 must return to a second gate with:

- upstream revisions and license receipts;
- exact W-D1/W-D2 sample manifests;
- a pinned PolicyBench/PolicyEngine snapshot;
- a reproducible RealClawBench seal;
- all nine model-load and quantization smoke results;
- measured idle/power-sampling repeatability;
- the generated manifest and an independently passing checker.

Only then is the 320-hour experiment authorized.

The Atlas earns its value by making expensive mistakes avoidable:

> choose the cheapest complete system that actually meets the user's floor,
> and say “none” when local affordability is not yet technically honest.

## Primary external sources

- [SWE-bench-Live repository](https://github.com/microsoft/SWE-bench-Live)
- [OpenAI 2026 SWE-bench Verified audit](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
- [Apple MKQA repository](https://github.com/apple/ml-mkqa)
- [PolicyBench methodology](https://policybench.org/)
- [PolicyEngine US documentation](https://policyengine.github.io/policyengine-us/)
- [RealClawBench paper](https://arxiv.org/abs/2606.03889)
- [RealClawBench deterministic leaderboard](https://realclawbench.github.io/leaderboard.html)
- [NVIDIA NVML](https://developer.nvidia.com/management-library-nvml)
