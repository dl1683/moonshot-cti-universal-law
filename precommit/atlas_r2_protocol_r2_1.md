# Atlas R2 Protocol Amendment R2.1

Date frozen: 2026-07-27
Supersedes: R2 (implicit in run_atlas_r2.py pre-amendment)
Codex steering: results/codex_p1_infrastructure_steering.md, results/codex_p1_infrastructure_steering_r2.md

## Rationale

Codex midpoint review of P1 pilot data identified 5 critical infrastructure blockers:
scoring bug (CJK whitespace tokenization), broken Gemma inference (fp16 pad-token output),
zero energy/cost tracking, non-resumable metrics, and insufficient episode count (25 vs 40
independent queries). All existing P1 data is designated `pilot_superseded` -- retained in
budget ledger but excluded from canonical analysis.

## 1. MKQA Task Selection (W-D2)

- Pre-filter MKQA queries to those with at least one non-empty answer text in ANY of the
  8 target languages BEFORE hash-rank selection.
- Select first 40 answerable queries by frozen hash order (seed: "atlas-r2-d2").
- 40 queries x 8 languages = 320 episodes per system.
- The pilot's 25-query subset is expected to remain a subset of the 40-query R2.1 selection
  (same hash ordering, larger answerable pool).

## 2. MKQA Scoring

Official Apple MKQA evaluation semantics (ml-mkqa/mkqa_eval_util.py):
- mixed_segmentation for ja and zh_cn: CJK characters split individually, non-CJK on whitespace
- Whitespace tokenization for ko (NOT character-segmented per Apple's official implementation)
- Whitespace tokenization for en, es, fr, de, ar
- Language-specific article removal: en (a/an/the), es (el/la/los/las/un/una/unos/unas),
  fr (le/la/les/l'/un/une/des), de (der/die/das/den/dem/des/ein/eine/einem/einen/eines/einer),
  ar (al-)
- Token multiplicities via collections.Counter (NOT set intersection)
- Continuous EM and token-F1 as primary metrics
- Macro-average across 8 languages
- Confidence intervals clustered by query_id (translations are correlated)
- F1 >= 0.5 pass threshold retained as frozen secondary procurement metric

## 3. PolicyBench Design (W-D3)

- One structured generation per household (100 households total, not 1970 per-variable).
- Hash-rank scenarios by seed "atlas-r2-d3", select first 100.
- Prompt for compact JSON object containing ALL required output variables per household.
- 128-token generation allowance (was 32 in pilot).
- Score every returned field: binary = exact 0/1 match, dollar = 5% relative or $50 absolute.
- Household score = fraction of correct fields.
- System W-D3 score = macro-mean over 100 households.
- Also report: JSON parse validity rate, strict all-fields-correct rate.

## 4. Task Record Schema

Authoritative per-system record files:
  results/cti_atlas_r2_task_records/cti_atlas_r2_r2_1_<phase>_<workload>_<system_id>.json

Task-ID-keyed map. Atomic write (write temp, rename) after each completed task.
Each record includes:
- task_id, protocol_revision ("r2.1"), run_id
- raw_output (before think-tag removal)
- cleaned_prediction (after think-tag removal, used by scorer)
- Scoring: exact_match, f1, status (W-D2) or fields_correct, household_score, parse_valid (W-D3)
- scorer_version (hash of scoring function)
- gold_answer_hash (SHA256 of sorted answer texts)
- model_revision (hf_revision from config)
- input_tokens, output_tokens, wall_seconds, gpu_seconds
- segment_id (links to energy segment)
- Classification: empty_output or inference_error (not ordinary fail)

Summary files are disposable derived artifacts. Resume reads task records, not summaries.

## 5. Energy Measurement

- Segment-level energy: one EnergyMeter segment per model load session.
- Proportional allocation to tasks by synchronized generation time:
  E_i = E_segment * (t_i / sum_j(t_j))
- torch.cuda.synchronize() called before start and after end of each generation.
- segment_energy_joules = primary measured quantity (stored in segment record).
- allocated_energy_joules = derived per-task field. Must sum to segment within FP tolerance.
- Model-loading energy recorded separately, amortized by frozen volume assumption.

## 6. Numerical Format

All 9 local checkpoints loaded with NF4 quantization (bitsandbytes):
- bnb_4bit_quant_type: "nf4"
- bnb_4bit_compute_dtype: torch.bfloat16 (all families)
- load_in_4bit: True

Model-specific torch_dtype for non-quantized operations:
- gemma3 family: torch.bfloat16 (required by architecture; fp16 produces pad-only output)
- qwen3 family: torch.bfloat16
- falcon_h1 family: torch.bfloat16

Substitutions from W4A16 roster: None. All 9 systems quantized per config.

## 7. Inference Path

- apply_chat_template(tokenize=True, return_dict=True, return_tensors="pt",
  add_generation_prompt=True). No text-then-retokenize path.
- Model-specific kwarg: enable_thinking=False for Qwen3 (TypeError fallback for others).
- BOS assertion: exactly one BOS token before generation (warning if violated).
- Generation cap: 30 seconds enforced via StoppingCriteria.
- Empty output (all pad tokens) classified as "empty_output", not ordinary "fail".

## 8. Scorer and Cost

- NON_INFERIORITY_MARGIN = 0.05 (was 5.0; scores are 0-1 proportions).
- Local system cost assumptions (frozen for R2):
  - RTX 5090 Laptop acquisition: $2,999
  - Useful life: 3 years, 2000 utilized hours/year
  - Capital amortization: $0.50/hour
  - Electricity: $0.12/kWh, 150W TDP envelope = $0.018/hour
  - Effective local hourly rate: $0.518/hour
  - Adaptation/RAG costs: $0 for P1 baseline
- API costs: per-token pricing from atlas_r2_systems.yaml (frozen 2026-07-27).
- 50/50 workload-level Gate A aggregate (W-D2 and W-D3 weighted equally).
- Next-cheapest fallback (not next-best).
- Query-clustered bootstrap confidence intervals.
- Hard failure on: incomplete cells, mixed protocol revisions, missing costs, unknown scorer.

## 9. Resume Architecture

- Resume reads task-record JSON files (Section 4), not cost ledger summaries.
- All metrics (mean F1, pass rate, language breakdown, energy) recomputed from task-record
  union on every resume.
- Duplicate (protocol_revision, workload, system_id, task_id) keys rejected at write time.
- Cost ledger remains append-only for budget tracking; scientific metrics come from records.

## 10. Pilot Data Disposition

- All P1 data collected before R2.1 labeled pilot_superseded.
- Retained in budget ledger (GPU-hours count toward total allocation).
- Excluded from canonical analysis.
- Canonical R2.1 records written to separate task-record files (Section 4).
- Pilot result JSON files (atlas_r2_p1_mkqa_raw.json) renamed with _pilot suffix.
