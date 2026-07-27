# R13 verdict

The project should pivot its identity.

The right moonshot is no longer “find a universal object inside intelligence.” It is:

> **Build a predictive science of scale inversion: determine, before expensive deployment, when a small specialized model will beat a much larger general model on a real user workload—and when it will not.**

There is no honest 7+/10 direction already established. The best candidate below is approximately **4/10 today**, with a **conditional 7–7.5/10 ceiling** if its extreme, prospective result survives. A 9+/10 claim is not credible from the present evidence or a four-to-eight-week single-GPU program.

## 1. Failure synthesis

The 11 kills are not 11 unrelated disappointments. They form four layers of the same failure.

| Layer | Directions | What failed |
|---|---|---|
| Assay validity | CTI, equicorrelation, benchmark bridge | The measured object was adjacent to the target, created by preprocessing, or confounded by scale. |
| Artifact consequence | AMCL, GAT | Infrastructure or an installable object existed, but no independent capability consequence followed. |
| Mechanism acquisition | CSO | Once the object became genuinely causal, even the favorable synthetic donor could not learn it. |
| Scientific novelty | ACQ, CCL, PCI, VIL, CIF | The claimed principle reduced to established theory, existing systems work, or an information-access tautology. |

The repeated sequence is:

```text
Name an essence
→ construct an assay where it is visible
→ recover or manipulate it
→ infer a consequence the assay never independently demonstrated
```

The repository’s own postmortems now support that conclusion: [STATUS.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/STATUS.md>), [R12 R1](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/codex_steering_r12.md>), [R12 R2](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/codex_steering_r12_r2.md>), and the [CIF gate](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/codex_cif_theorem_gate.md>).

### What this project can do

It can:

- Design unusually hard precommitments and kill gates.
- Find confounds, circular metrics, baseline omissions, and claim inflation.
- Run strong inference and QLoRA-scale experiments on deployable open models.
- Measure exact outcome, energy, latency, memory, adaptation cost, and worst-group performance.
- Produce credible negative results and useful deployment maps.
- Test a sharply stated empirical relation prospectively.
- Turn one-GPU limits into a scientific constraint: the result must concern intelligence affordable on one GPU or less.

### What it cannot honestly do from this setup

It cannot:

- Establish a universal law of intelligence from one model family, synthetic environment, or internal observable.
- Infer economic value from parameter count, artifact size, FLOPs, or representation fidelity.
- claim cross-task universality using per-task intercepts or overlapping model-family evidence.
- Treat synthetic worlds that expose the desired state or interface as evidence about natural intelligence.
- Train a new foundation model large enough to make broad capability claims.
- Convert a conjunction of known methods into a new principle.
- Reach 9+/10 merely because a result has a dramatic name or an enormous model-size ratio.

The honest constraint map is therefore:

> **This project is capable of discovering and validating deployment regimes. It is not presently capable of deriving a universal essence of intelligence.**

That is not a small distinction. It should determine the project’s identity.

## 2. The meta-question

Yes: as an operating strategy, “find a universal law or theorem about intelligence” is fundamentally flawed.

A universal law might exist. The failure is making universality the entry condition rather than the end state earned by repeated prediction. That framing has Goodharted the project toward:

- compact scalars;
- grand ontological nouns;
- synthetic tasks that make those nouns legible;
- formal statements selected for headline value;
- evidence interpreted through the desired destination score.

The project should become something like:

> **Affordable Intelligence Science**  
> Find, predict, and explain regimes where limited hardware achieves equal or better real-world outcomes than frontier-scale systems.

A law may eventually emerge from that program. But the order must reverse:

```text
Natural user workload
→ independently measured scale inversion
→ replicated boundary
→ prospective predictor
→ mechanism
→ only then, perhaps, a law
```

The 9+/10 Nobel target should remain a distant aspiration, not a direction-admission threshold. At present:

- Current scientific position: **2/10**, consistent with [CLAUDE.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/CLAUDE.md>).
- A rigorous cross-domain deployment atlas: **5/10**.
- A prospective, family-independent predictor of 10× scale inversions: **7–7.5/10**.
- A 9+ principle would require years of replication, successful predictions on unseen model generations, and real deployment consequences. It cannot be responsibly scheduled for the next eight weeks.

## 3. What the field actually needs

The high-level opportunity is real, but “small models are beating large models and nobody understands why” is too strong.

The examples currently called “small beats large” mix several distinct causes:

- better or more relevant training data;
- later architectures and post-training recipes;
- task-specific fine-tuning or distillation;
- RLVR and exact verification;
- much more test-time sampling;
- sparse versus dense parameter counts;
- benchmark protocol differences;
- deployment hardware effects.

For example, Qwen’s provider-reported results show its 9B model beating GPT-OSS-120B on several knowledge and instruction benchmarks, while the larger model remains substantially ahead on other workloads such as code. That is evidence of **task-dependent ordering reversal**, not general 9B-over-120B dominance. [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)

Likewise, the official LFM2-2.6B card documents strong on-device efficiency and comparisons mainly against similarly sized models. The popular “2.6B beats 671B” line refers to much narrower experimental or instruction-following comparisons and should not be presented as general domain superiority. [Liquid AI LFM2-2.6B](https://huggingface.co/LiquidAI/LFM2-2.6B)

Small-model gains from RLVR are real—a recent 0.6B/1B code study reports gains of up to 13 pass@1 points—but reward design can also produce degenerate behavior. [Small-model RLVR study](https://arxiv.org/abs/2605.30478) Test-time search can let very small reasoning models beat vastly larger immediate-answer systems on mathematics, but that comparison includes extra search and a reward model. [Compute-optimal test-time scaling](https://arxiv.org/abs/2502.06703) Focused distillation can let 2–3B translation models match systems up to 70B on low-resource languages. [LoResMT 2026](https://aclanthology.org/2026.loresmt-1.1/)

There is also already relevant science:

- Fine-tuning performance and model selection have a rectified scaling law with a low-data “pre-power” phase. [Rectified Scaling Law](https://arxiv.org/abs/2402.02314)
- Domain-specific distillation has emerging scaling laws. [Task-specific distillation scaling](https://arxiv.org/abs/2606.24747)
- Inference-aware scaling explicitly predicts that high deployment volume can favor smaller, more heavily trained models. [Beyond Chinchilla-Optimal](https://proceedings.mlr.press/v235/sardana24a.html)
- “Meek Models” already argues that diminishing returns can drive capability convergence. [MIT FutureTech](https://futuretech.mit.edu/publication/meek-models-shall-inherit-the-earth)

So the unoccupied target is not:

> Why can a small model sometimes win?

It is:

> **Can we prospectively predict the workload, data, model, and cost boundary at which the ranking reverses—under matched information and serious baselines?**

That is a legitimate moonshot: predictive model–task matching, not another internal ontology.

## 4. One direction: Prior-Conflict Scale Inversion

### The user-first observation

Poor and local users frequently need a model to obey truths that are weakly represented—or contradicted—by the global internet distribution:

- current repository behavior versus obsolete API conventions;
- local administrative rules versus globally common defaults;
- low-resource language conventions versus dominant-language priors;
- organization-specific actions versus generic assistant habits.

A large model may possess more generic knowledge while also having a stronger, more confidently entrenched wrong prior for the local workload.

### Exact claim

Call the hypothesis **PCSI-1: Prior-Conflict Scale Inversion**.

> Holding local data access and adaptation budget fixed, the small-minus-large post-adaptation performance gap increases monotonically with pre-adaptation prior conflict. Above a prospectively estimated conflict threshold, a 0.6–1.7B model will match or beat a 14–32B model on the local workload while consuming at least 10× less inference energy.

The one new relation is:

\[
\frac{\partial}{\partial C}
\left(
A_{\text{small}}-A_{\text{large}}
\right)>0,
\qquad
\exists C^\star:
A_{\text{small}}\geq A_{\text{large}},
\]

where:

- \(C\) is measured before adaptation on a sealed calibration set: the frequency and confidence with which the pretrained model prefers a globally common but locally incorrect action;
- \(A\) is held-out exact task success, with a separate safety floor;
- the claimed relation must transfer without task-specific intercepts.

This does not claim “small is better.” It predicts one reason that size can become a liability: more strongly learned global priors may require more local evidence or adaptation compute to overwrite.

Prior conflict is already known to hinder fine-tuning, so that fact alone is not novel. Recent work identifies conflicts with pretrained knowledge as a major source of incomplete SFT learning; inverse-scaling and catastrophic-overtraining results are also direct threats. [Incomplete Learning Phenomenon](https://aclanthology.org/2026.acl-long.1393/), [Inverse Scaling](https://arxiv.org/abs/2306.09479), [Overtrained Models Are Harder to Fine-Tune](https://proceedings.mlr.press/v267/springer25a.html)

The novel result would have to be the **prospective size crossover under natural workloads and measured economics**.

### Exact comparison

Use pretrained deployment-scale models immediately:

- Qwen3 0.6B and 1.7B versus 14B and 32B.
- A confirmation ladder such as OLMo2 1B versus 7B/13B.
- Four-bit deployment where needed, with quantization controls.

Use three independently sourced natural workload families:

1. Post-cutoff repository/API tasks with executable tests.
2. Jurisdiction-specific public-service or eligibility action routing, derived from official versioned rules and checked by a deterministic rule engine.
3. Low-resource multilingual public-service intent/action tasks with human labels.

These should be action or structured-output tasks with authoritative adjudication—not subjective chat quality.

Every scale receives:

- zero/few-shot prompting;
- RAG or supplied local documentation;
- independently optimized QLoRA/SFT;
- equal-data and equal-adaptation-joule comparisons;
- compute-optimal inference within the device budget;
- task-specific distillation where applicable.

The large model must get the same local information and its best feasible adaptation. Otherwise the result is merely “fine-tuned small beats untuned large.”

Primary outcomes:

- exact held-out task success;
- worst-group and high-cost-error performance;
- GPU joules per completed task;
- latency and peak memory;
- adaptation energy amortized at 1,000, 10,000, and 100,000 deployments.

### Required 7+ result

PCSI-1 earns approximately 7–7.5 only if all of these hold:

- A threshold fitted on two workload families predicts the crossover direction on the untouched third family.
- Crossover data requirement is predicted within 2× and final accuracy within 3 points.
- No per-task intercepts are needed.
- The result holds in two model families.
- A 0.6–1.7B system is noninferior within 3 points overall and at least 5 points better on the precommitted high-conflict slice.
- Inference energy is at least 10× lower.
- Worst-group safety does not regress.
- Existing fine-tuning scaling laws, zero-shot accuracy, pilot fine-tuning loss, and simple task difficulty fail to explain the result equally well.

Viral story:

> **“When the internet is wrong for your world, the smaller AI wins.”**

That is stronger and more socially meaningful than “9B beats 120B on benchmark X.”

### Kill criteria

Kill PCSI-1 immediately if any of these occur:

1. A three-day novelty audit finds the same prior-conflict-by-model-size crossover relation and prospective predictor.
2. No measurable conflict-by-scale interaction appears in the 72-hour stock-baseline pilot.
3. High-conflict small models fail to close at least 70% of the small–large gap.
4. The large model, given matched local data and optimized adaptation, remains more accurate at every cost point.
5. Small-model wins occur only against untuned, poorly prompted, or compute-starved large models.
6. Rectified Scaling Law, base accuracy, or fine-tuning loss predicts the crossover within the same error.
7. The proposed relation needs workload-specific intercepts.
8. Results hold only on constructed conflict benchmarks, not natural workloads.
9. Results disappear on a second model family.
10. RAG or in-context local rules bring the large model within 3 points at acceptable cost.
11. The energy advantage is below 10×, or adaptation amortization removes it at realistic workload volume.
12. High-cost or worst-group errors worsen.
13. Benchmark contamination or provider-specific evaluation protocols explain the inversion.
14. Full evaluation requires privileged model internals, hidden-state interventions, or user-query oracles.

### Eight-constraint audit

| Constraint | PCSI-1 |
|---|---|
| Outcome first | Exact useful actions, safety, energy, latency |
| Independent outcome | Prior conflict measured before adaptation; success measured on sealed held-out tasks |
| Natural tasks | Live repositories, official rules, human-labeled multilingual workloads |
| No privileged interfaces | Standard inputs, documents, outputs, and training |
| Scale crossover | 0.6–1.7B versus 14–32B; ≥10× energy |
| Strongest baseline first | Stock RAG, QLoRA, distillation, and compute-optimal inference precede any new method |
| One new relation | Prior conflict predicts the size-ordering reversal |
| Deployment scale | Existing open-weight models runnable or adaptable on one RTX 5090 |

It avoids both deep traps:

- No compact internal artifact is proposed.
- No necessary information is hidden and later exposed through a privileged query.

## 5. If nothing survives

Nothing currently deserves a 7+ score. PCSI-1 merely has a conditional 7+ result ceiling, with a low probability of passing all gates.

If it dies, the realistic product is a **5/10 Scale-Inversion Atlas**:

- a reproducible model–task–cost matrix;
- measured energy, latency, memory, and adaptation break-even points;
- matched small/large comparisons;
- explicit regions where specialization, RLVR, RAG, or test-time compute work;
- explicit regions where larger models remain necessary;
- a lightweight selector that recommends the cheapest model meeting a user’s accuracy and safety floor.

That would not explain intelligence universally. It could nevertheless prevent thousands of users and small organizations from paying for models they do not need—and identify workloads that can genuinely run locally.

Yes: a verified 5/10 result that helps poor users is better than another rhetorically inflated 7/10 attempt. After 11 kills, that is not lowering the manifesto’s standard. It is finally applying it.

