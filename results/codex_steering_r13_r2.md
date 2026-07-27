# R13 Round 2 decision

**PCSI-1 does not survive as the flagship.** It survives only as a falsifiable secondary hypothesis.

The **Scale-Inversion Atlas should become the primary program**, but it must be materially stronger than PER: an absolute, constraint-based, prospective model-selection system—not another composite efficiency score or static benchmark table.

## A. Does PCSI-1 survive the Day 3 novelty audit?

### As a standalone 7+ direction: no

The PER paper does not exactly duplicate PCSI, but it occupies too much of the surrounding territory.

PER evaluates 16 models across five standard tasks and combines accuracy, throughput, memory, and latency using a min–max-normalized geometric mean. It concludes that 0.5–3B models have the best PER across its task set. [Task-Specific Efficiency Analysis](https://arxiv.org/html/2603.21389v1)

However, PER does **not** demonstrate PCSI’s proposed result:

- It measures pretrained models, not matched post-adaptation crossovers.
- It does not study prior conflict.
- It provides a descriptive taxonomy, not a prospective predictor.
- It accepts 10–35% accuracy losses in exchange for efficiency.
- It does not demonstrate that a smaller model matches or beats a larger model in quality.
- It does not evaluate adaptation cost, workload volume, energy, safety floors, or natural deployment distributions.

Its metric also has important limitations:

1. Min–max normalization makes PER relative to the selected candidate set.
2. Throughput and latency are strongly coupled and effectively receive separate weight.
3. A single geometric mean hides user-specific constraints and trade-offs.
4. Different model scales use different numbers and types of GPUs.
5. Per-GPU throughput is not the same as deployability on one user device.
6. A model can rank highly despite an unacceptable absolute error rate.

So kill criterion 1 does **not** fire through exact duplication.

But the broader novelty verdict is still negative. “Map which task characteristics favor small models” is now established territory. PCSI would need its prior-conflict mechanism to carry the novelty—and that mechanism is heavily threatened.

### The mechanism audit

The cited fine-tuning literature establishes most of PCSI’s immediate conceptual ingredients:

- PriFT explicitly measures target-token support under the frozen pretrained distribution and uses that support to control SFT. [PriFT](https://arxiv.org/abs/2606.09396)
- Midtraining finds the largest benefits for target domains distant from general pretraining data and identifies a late-training plasticity limit. [Midtraining](https://arxiv.org/abs/2510.14865)
- The RFT/SFT study finds that SFT learns novel tasks quickly but forgets prior capabilities, while RFT follows rollouts better aligned with the base probability landscape. [RFT versus SFT](https://arxiv.org/abs/2506.23508)
- Existing work already shows that extended pretraining can reduce downstream plasticity. [Overtrained Models Are Harder to Fine-Tune](https://proceedings.mlr.press/v267/springer25a.html)

PCSI’s remaining novelty would be the interaction:

\[
\text{prior conflict}\times\text{model size}
\longrightarrow
\text{size-ordering reversal}.
\]

That interaction is not directly established by the cited papers. But proving it would now look like an extension of an occupied literature, not a new theory of affordable intelligence.

There is also no firm basis for the step:

> Larger model → more entrenched prior → harder adaptation.

Larger models can have stronger priors, but they may also have:

- greater redundant capacity;
- better sample efficiency;
- more task-relevant latent knowledge;
- better ability to represent old and new behavior simultaneously.

Indeed, the RFT/SFT experiments do not show a clean small-model adaptation advantage: the 7B model often learns the new task better than the 3B model, even if some forgetting is more severe. PriFT also works across both 1.5B and 7–8B scales without establishing an adverse size interaction.

Therefore:

> **PCSI survives as an unanswered empirical interaction, but not as a sufficiently novel flagship principle.**

## B. Should monotonicity become a threshold claim?

No. A single threshold is still too strong.

The original monotonic claim should be retracted:

\[
\frac{\partial(A_s-A_l)}{\partial C}>0.
\]

Your counterexample is decisive:

- Low conflict: the large model’s extra capability should usually win.
- Intermediate conflict: a small model may adapt cheaply enough to match or beat it.
- Extreme conflict or novelty: neither model may possess the necessary primitives; alternatively, the larger model’s capacity may again dominate.

Prior conflict also conflates two different conditions:

1. **Contradiction:** the model knows relevant primitives but favors the wrong behavior.
2. **Absence:** the model lacks the knowledge or capability entirely.

SFT, RLVR, and local adaptation behave very differently in those regimes.

The defensible secondary hypothesis is a **bounded crossover region**:

\[
A_s-A_l \geq 0
\quad\text{only for}\quad
C\in[C_1(S,B),C_2(S,B)],
\]

where:

- \(C\) is prior conflict;
- \(S\) is latent task support or solvability;
- \(B\) is the adaptation/deployment budget.

That is not a clean new law. It is a region to map.

Accordingly, rename PCSI-1 to something less presumptive:

> **PC-H1: Prior-conflict crossover-window hypothesis**

Its prediction is:

> Conditional on both model scales already having sufficient latent task support, moderate prior conflict may create a budget-dependent region where the small model reaches the quality floor more cheaply.

PC-H1 should be pre-registered as one Atlas hypothesis, not used to define the Atlas or its tasks.

## C. Should the Atlas be primary?

**Yes. This is the correct architectural decision.**

But the Atlas cannot lead with “small models have better composite efficiency.” PER already did that.

The Atlas must answer a different, user-facing question:

> Given my workload, hardware, volume, quality requirement, and safety floor, what is the cheapest complete AI system that meets them?

The core object should be a constrained decision, not a scalar score:

\[
s^\star
=
\arg\min_s C_{\text{all-in}}(s,w,h,V)
\]

subject to:

\[
Q(s,w)\geq Q_{\min},
\quad
R(s,w)\geq R_{\min},
\quad
L(s,h)\leq L_{\max},
\quad
M(s,h)\leq M_{\max}.
\]

Here:

- \(s\) is a complete system configuration;
- \(w\) is a natural user workload;
- \(h\) is hardware;
- \(V\) is deployment volume;
- \(Q\) is task quality;
- \(R\) is reliability or worst-group safety;
- \(L\) is latency;
- \(M\) is memory;
- \(C_{\text{all-in}}\) includes adaptation, inference, verifier, retrieval, and amortized training costs.

### What makes this different from PER

| PER | Scale-Inversion Atlas |
|---|---|
| Relative composite score | Absolute user constraints |
| Five standard benchmarks | Natural deployment workloads |
| Raw pretrained models | Complete systems |
| Static description | Prospective selection |
| Throughput, latency, memory | Energy, dollars, latency, memory, connectivity, adaptation |
| No quality floor | Required quality and safety floors |
| Candidate-set-dependent ranking | Pareto frontier and cheapest feasible system |
| No workload-volume accounting | Adaptation amortized over realistic volume |

The Atlas should compare systems such as:

- raw small and large models;
- quantized variants;
- RAG;
- QLoRA/SFT;
- distillation;
- RLVR;
- best-of-\(n\);
- verifier-guided search.

This matters because “small versus large” is often the wrong unit of comparison. The user chooses an operational system, not a parameter count.

### Scientific discipline

Do lead with the Atlas and let hypotheses emerge—but prevent post-hoc storytelling through a discovery/confirmation split:

1. **Discovery workloads:** map the frontier and generate hypotheses.
2. **Frozen hypothesis ledger:** specify predictors, signs, thresholds, and falsifiers.
3. **Confirmation workloads:** untouched natural task families.
4. **Prospective model test:** evaluate at least one model released after the selector is frozen.

Without prospective confirmation, the Atlas is an excellent 5/10 public resource. With successful prospective selection, it can become 6–7/10 science.

The viral narrative is stronger than PCSI:

> **“Stop paying for the biggest AI. Tell us your task and hardware; the Atlas tells you the smallest system that is actually good enough.”**

## D. Does the RLVR insight change the mechanism?

It changes the Atlas schema. It does not justify replacing prior conflict with “fewer competing capabilities.”

The NeurIPS 2025 oral finds that current RLVR primarily improves low-\(k\) performance while the base model recovers or exceeds it at large \(k\); the authors conclude that current RLVR is largely bounded by base-model reasoning coverage. Distillation, by contrast, can introduce patterns from a teacher. [Does RL Really Incentivize Reasoning Capacity?](https://arxiv.org/abs/2504.13837)

But the strongest defensible statement is narrower than “RLVR never creates capability.” Other work reports capability gain through self-distillation or prolonged RL beyond the sampled base boundary. [Adaptive Guidance](https://neurips.cc/virtual/2025/loc/san-diego/126691), [ProRL](https://arxiv.org/abs/2505.24864) The literature is not settled on how absolute that boundary is.

The “fewer competing latent capabilities” mechanism is currently speculative. More capabilities do not imply a harder optimization problem unless we can measure:

- how many viable solution modes exist;
- how reward mass is distributed across them;
- whether that distribution worsens with model size;
- whether restricting modes causally improves RLVR.

Otherwise “latent capability competition” risks becoming the next attractive essence.

### What should enter the Atlas

Add two distinct axes:

1. **Latent solvability:** base-model pass@\(K\) or verified coverage before training.
2. **Elicitation cost:** compute required to convert that coverage into reliable pass@1 behavior.

This gives a clean deployment question:

> When is it cheaper to elicit a capability already latent in a small model than to deploy a larger model that expresses it directly?

For every RLVR result, charge:

- training rollouts;
- verifier computation;
- failed samples;
- inference-time reasoning;
- final deployment energy.

A useful secondary hypothesis is:

> If a small model’s base pass@\(K\) already exceeds the required quality floor, RLVR may compress that latent coverage into pass@1 more cheaply than deploying a larger model.

That is useful, but it is not presently a new 7/10 mechanism. It belongs beside PC-H1 within the Atlas.

## E. Revised honest scores

### PCSI-1

- Current standalone hypothesis: **3/10**
- Reproducible crossover window on several tasks: **4.5–5/10**
- Prospective prediction across natural task families and two model families: **6–6.5/10**
- Honest present 7+ trajectory: **no**

The ceiling fell because:

- efficiency mapping is occupied by PER and related work;
- prior support and distribution shift are occupied;
- monotonicity is false;
- conflict alone cannot distinguish wrong prior from missing capability;
- the surviving interaction is scientifically legitimate but incremental.

### Scale-Inversion Atlas

- Proposal today: **4/10**
- Audited open matrix plus public constraint-based selector: **5.5–6/10**
- Prospectively selects the cheapest qualifying system on unseen workload families and future models: **7/10**
- Broad adoption plus repeated, surprising scale inversions: **7.5+**
- Current 9+ claim: **no**

## Final architectural decision

1. **Close PCSI-1 as the Phase 3 flagship.**
2. Retain it as **PC-H1**, a bounded crossover-window hypothesis.
3. Make the **Scale-Inversion Atlas** the primary program.
4. Reimplement PER as a required baseline, but reject PER-style composite ranking as the Atlas objective.
5. Treat prior conflict, latent solvability, adaptation data, verification, and test-time compute as competing candidate predictors.
6. Freeze a prospective selector before confirmation tasks and future model releases.
7. Judge success by avoided cost at a fixed quality and safety floor—not by whether the project discovers another universal equation.

The key correction is:

> Do not build an Atlas to prove why small models win. Build an Atlas that tells users which system wins. Then require any proposed explanation to predict an untouched part of the map.