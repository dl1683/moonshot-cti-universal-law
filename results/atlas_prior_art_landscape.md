# Scale-Inversion Atlas: Prior Art Landscape

Claude internet research (Jul 27, 2026) for the Atlas design gate.

## Existing Benchmarking and Measurement Frameworks

### 1. Bench360 (arxiv 2511.16682, Nov 2025)
- Full-stack pipeline for LOCAL LLM inference benchmarking
- Evaluates latency, throughput, quality, cost across tasks, GPUs, engines
- Functional task quality (execution accuracy, F1, EM, ROUGE) not just perplexity
- 4 NLP tasks, 3 GPUs, 4 engines
- Key finding: "No universal best option" -- tradeoffs are substantial
- Open source: github.com/slinusc/bench360
- **Atlas relation:** Required baseline. Bench360 is descriptive; Atlas is prescriptive + prospective.

### 2. SweetSpot (arxiv 2602.05695, Feb 2026)
- Analytical model for PREDICTING energy efficiency of LLM inference
- Predictive, not just descriptive
- **Atlas relation:** Potential tool for energy prediction. Atlas needs measured energy, not predicted.

### 3. PER Metric (arxiv 2603.21389, ESANN 2025)
- Performance-Efficiency Ratio: geometric mean of accuracy, throughput, memory, latency
- 16 models, 5 standard NLP tasks
- Small models (0.5-3B) achieve superior PER across all tested tasks
- Limitations: relative (min-max normalized), no quality floor, pretrained only, no adaptation
- **Atlas relation:** Required baseline. Reject PER-style composite ranking as Atlas objective.

### 4. SEALing the Gap (arxiv 2603.02949, Mar 2026)
- Reference framework for LLM inference carbon estimation
- Multi-benchmark driven embodiment
- **Atlas relation:** Carbon/energy measurement methodology.

## Model Routing and Selection

### 5. RouteLLM
- Dynamic routing between small and large models based on query complexity
- Learns classifier: simple -> small model, complex -> large model
- **Atlas relation:** Router assumes the large model is always better on hard tasks. Atlas tests whether that's true.

### 6. FrugalGPT
- Cascades of models + caching to reduce API costs
- **Atlas relation:** API-oriented. Atlas is deployment-oriented (local hardware).

## Real-World Task Benchmarks

### 7. LiveClawBench (arxiv 2604.13072)
- Complex, real-world assistant tasks
- Long-horizon execution, cross-service coordination
- **Atlas relation:** Potential workload family for complex tasks.

### 8. WildBench (arxiv 2406.04770)
- Challenging tasks from real users in the wild
- **Atlas relation:** Source of natural workload distribution.

## Scale Inversion Evidence

### 9. MIT Meek Models (arxiv 2507.07931, Jul 2025)
- Diminishing returns to compute -> capability convergence
- Small models approach SOTA performance levels
- **Atlas relation:** Theoretical backing for why scale inversions should become more common.

### 10. RLVR Findings (NeurIPS 2025 oral, arxiv 2504.13837)
- RLVR does NOT inject new knowledge. Activates capabilities ALREADY LATENT in base model.
- Coverage bounded by base model. Distillation can introduce patterns from teacher.
- **Atlas relation:** Latent solvability hypothesis -- base pass@K predicts RLVR success.

### 11. Task-Specific Efficiency Analysis
- Forbes: "Small Language Models Outperform Frontier AI On Cost, Speed And Accuracy"
- 2.6B SLM vs 671B DeepSeek-R1 on domain tasks: SLM wins
- Qwen 3.5 9B matches 120B on key benchmarks
- **Atlas relation:** The empirical phenomenon the Atlas maps.

### 12. Fine-Tuning Dynamics
- PriFT (2606.09396): distribution shift scales with training data
- Midtraining (2510.14865): high-shift regimes cause gradient conflicts
- RFT vs SFT (2506.23508): SFT causes catastrophic forgetting proportional to divergence
- Overtrained Models Are Harder to Fine-Tune (ICML 2025)
- **Atlas relation:** PC-H1 hypothesis ingredients. Not sufficient for flagship direction alone.

## What Does NOT Exist (Atlas Gap)

1. **Constraint-based selection** -- no existing tool takes user constraints (quality floor, safety floor, latency, memory, budget) and returns the cheapest qualifying system
2. **Complete system comparison** -- existing work compares models, not complete systems (model + quantization + RAG + adaptation + verification)
3. **Prospective model prediction** -- no existing work freezes a selector and then tests it on a model released after the freeze
4. **All-in cost with adaptation amortization** -- no existing benchmark amortizes adaptation cost over realistic deployment volume
5. **Natural deployment workloads** -- most benchmarks use standard academic tasks, not production-representative workloads
6. **Discovery/confirmation split** -- no existing benchmark separates hypothesis-generating tasks from confirmation tasks

## Honest Assessment

The Atlas occupies a genuine gap. Existing tools are either:
- Descriptive (Bench360, PER) -- map what happened, not what WILL happen
- Routing-focused (RouteLLM, FrugalGPT) -- assume large model is better on hard tasks
- Standard-benchmark-only -- don't use natural deployment workloads
- Model-only -- don't compare complete systems with adaptation

The Atlas threat is that it becomes "just another benchmark paper" (5/10 ceiling). The 7/10 requires prospective confirmation: freeze the selector, then test on an unseen workload and a model released after the freeze.
