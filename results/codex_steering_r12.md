I read [CLAUDE.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/CLAUDE.md>), [STATUS.md](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/STATUS.md>), and [QL11 R2](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/results/codex_cso_question_loop_ql11_r2.md>).

The brutal verdict:

> ACQ-1 should not become Phase 3. Its central result is largely known, and its benchmark manufactures the abstraction it claims to discover. The previous 6–7/10 ceiling was inflated; the exact proposed experiment is about 3–4/10.

## Part 1: Failure synthesis

### 1. The deep pattern connecting all six failures

All six fall into the same compactness–consequence trap:

> The project searches for a small, universal internal object after intelligence already exists. When that object is easy to identify, it is tautological, null, or supplied by the experiment. When it is causal enough to carry real capability, it is difficult to learn, architecture-dependent, and not cheaply portable.

That pattern explains every kill:

- Equicorrelation found a clean invariant because the preprocessing created a geometric null.
- The benchmark bridge found high agreement because \(q\) and \(\kappa\) were two views of essentially the same neighborhood structure.
- AMCL perfected commitment infrastructure before demonstrating that any substantive phenomenon existed.
- GAT installed geometry successfully—but the installed object was causally irrelevant to capability.
- CSO moved to a genuinely functional object, and the small donor could not even learn that object on a favorable synthetic task.
- CSO’s strategic ceiling remained low because the experiment supplied the mechanism boundary and packaged familiar ingredients into a conjunction.

So the deepest failure is not “geometry was wrong” or “the donor was too small.” It is this selection rule:

```text
Choose an essence of intelligence
→ construct a world where the essence is well-defined
→ measure recovery of the essence
→ extrapolate to cheap intelligence
```

That is backwards. The next program must start with a direct scale-defying consequence and remain agnostic about its mechanism.

### 2. What the solution space must now look like

The graveyard imposes hard constraints:

1. **Outcome first.** The primary dependent variable must be useful task success per measured joule, second, byte, or dollar—not geometry, state count, fidelity, representation quality, or artifact size alone.

2. **Independent outcome.** The success measure must not be mathematically adjacent to the predictor or training objective.

3. **Natural or sealed tasks.** The task generator cannot install the proposed ontology, causal boundary, decomposition, or compact program.

4. **No privileged interfaces.** Avoid simulator states, reset-to-snapshot access, paired rerenders, manually designated sockets, teacher internals, or intervention points unavailable in deployment.

5. **A scale crossover, not a small gain.** The moonshot result must look like “1–3B matches or beats 14–30B at at least 10× lower measured energy” or an equally sharp impossibility theorem. A 10–15% improvement over another small model is ordinary research.

6. **The strongest existing method runs first.** No new framework until the cheapest stock baseline has shown that the surprising result is not already known or impossible.

7. **One new relation.** Not “bisimulation + JEPA + active learning + MDL.” One phase transition, lower bound, or capability that remains surprising after every component is named.

8. **Deployment scale from day one.** Use pretrained 0.5–3B models already near the capability boundary. The CSO result argues strongly against expecting a newly trained 19M model to discover nontrivial algorithms in four weeks.

### 3. Assumptions still untested

The program is still assuming that:

- useful capabilities possess compact, stable, architecture-independent descriptions;
- a learner can discover those descriptions more cheaply than simply learning the task;
- small synthetic models reveal the same bottlenecks as pretrained models;
- representation or planning compute is the economic bottleneck rather than perception, data acquisition, memory bandwidth, and user context;
- reduced parameters or MACs translate into lower wall-clock energy on cheap hardware;
- simulator interventions approximate information obtainable in the real world;
- the same abstraction remains sufficient across new goals;
- a universal scalar law exists at all;
- skills can be separated from factual memory, language, prompting, and architecture;
- solving a controlled benchmark predicts value for a poor user;
- surprising internal structure is necessary for cheap intelligence.

The final assumption may be the biggest blind spot. Cheap intelligence may come from exploiting inexpensive external structure—verification, interaction, local data, execution, or collaboration—rather than discovering a better internal essence.

### 4. What the failures predict should work

They predict success in domains where:

- correctness can be checked more cheaply than an answer can be generated;
- the user or environment can supply missing information interactively;
- a small pretrained model already contains the primitive capabilities;
- the task has real external structure—compilers, tests, constraints, databases, or community-specific data;
- specialization matters more than broad generic knowledge.

This prediction is supported by existing work, but that is also the novelty problem. Small specialized models already beat much larger generic models in low-resource settings, including results where 2–3B models match or exceed systems up to 70B after focused adaptation. [ACL 2025](https://aclanthology.org/2025.acl-srw.24/), [LoResMT 2026](https://aclanthology.org/2026.loresmt-1.1/)

Likewise, smaller models with more inference-time computation can already outperform larger models on some reasoning workloads. [Inference Scaling Laws](https://arxiv.org/abs/2408.00724)

So these approaches are predicted to work, but “specialize,” “verify,” or “search more” alone are not new 7/10 directions.

## Part 2: ACQ-1 stress test

### Antipattern audit

| Antipattern | Verdict | Why |
|---|---|---|
| Begins with the answer | **Yes** | It defines actionable causal quotient as the desired object, then constructs environments possessing exactly that quotient. |
| Surrogates replace consequences | **Yes** | State count, latent MACs, quotient F1, and toy planning stand in for actual affordable utility. |
| Task contains the answer | **Strong yes** | Same-state rerenders label nuisance invariance; snapshot resets and alternative suffixes expose causal distinctions; the simulator already knows the partition. |
| Infrastructure before nontriviality | **Yes** | Three environments, ten baselines, six gates, and 60–80 GPU-hours were proposed before establishing novelty against block-MDP and exogenous-state work. |
| Conjunction novelty | **Yes** | It combines causal states, bisimulation, active system identification, paired invariance, quantization, and MDL. |
| Destination score leaks backward | **Yes** | “Million-times noisier world” and democratized intelligence inflate a controlled state-abstraction experiment. |

### The single most dangerous objection

> ACQ-1 constructs a block MDP with a small latent state, adds exogenous pixels, gives the learner rerender and reset access that identifies the latent partition, and then “discovers” that planning depends on the latent MDP rather than irrelevant pixels. That conclusion was true before the learner ran.

This is not merely an intuitive objection. Prior work already provides:

- efficient RL whose sample complexity depends on latent-state count with no dependence on potentially infinite observation-space size; [BRIEE](https://proceedings.mlr.press/v162/zhang22aa.html)
- latent-state decoding from rich observations; [Du et al.](https://proceedings.mlr.press/v97/du19b.html)
- explicit removal of exogenous variables to accelerate RL; [Dietterich et al.](https://proceedings.mlr.press/v80/dietterich18a.html)
- compact planning models that omit large exogenous processes; [Chitnis and Lozano-Pérez](https://proceedings.mlr.press/v100/chitnis20a.html)
- reward-free agent-centric representations designed for rich exogenous visual information. [ACRO](https://proceedings.mlr.press/v202/islam23a.html)

There is also a measurement problem: increasing pixel entropy does not necessarily increase computational difficulty. Randomizing a background can drastically raise Shannon entropy while leaving image dimensions and encoder FLOPs unchanged. Conversely, total system compute must still pay to read and encode every pixel. ACQ therefore oscillates between:

- latent planning compute, where nuisance-independence is almost definitional; and
- total compute, where sensory complexity cannot simply disappear.

### Was 6–7/10 inflated?

Yes.

My revised scores:

- **Exact ACQ-1 result ceiling:** 3–4/10.
- **If it beats every baseline on three synthetic worlds:** good representation-learning paper, perhaps 4/10.
- **If it generalizes without privileged rerenders/resets to several natural environments and predicts actual end-to-end energy:** 5–6/10.
- **To reach 7/10:** it would need a genuinely new theoretical limit or a broad natural-world scaling inversion—not a distractor benchmark.

### Does the central claim survive “that’s obvious”?

No, not in its current form.

“Planning should depend on control-relevant state rather than irrelevant observation detail” is standard knowledge in RL, control, state abstraction, bisimulation, predictive-state representations, and system identification. A practitioner may not use the phrase “intervention-distinguishable causal complexity,” but renaming the sufficient state does not create a new principle.

Counterexample-guided refinement might still be a useful algorithm. That would make ACQ an algorithmic contribution, not a new law of intelligence.

## Part 3: Three conditional 7+ directions

A proposal itself cannot honestly score 7+. The following are **7+ exact-result ceilings**: each earns that score only if the specified hard result occurs. They are deliberately high-risk.

### 1. CCL-1 — Capability Conservation Law

**Claim**

There is an irreducible communication price for skill transfer:

> The artifact bits plus adaptive teacher-response bits required to move a fixed student to error \(D\) are lower-bounded by a task-conditional capability rate–distortion function—and a constructive codec can approach that bound.

This would replace “find the essence of intelligence” with “measure exactly how much missing task information must cross the boundary.”

**Experiment**

Spend 4–6 weeks, primarily on theory:

1. Define the student prior, task family, allowable transfer channel, total transmitted bits, teacher queries, and end-task loss.
2. Prove matching or near-matching lower and upper bounds for finite stochastic task families.
3. Exhaustively verify the bound on small function classes.
4. Test whether the bound predicts unseen empirical transfer curves for logits, rationales, LoRA deltas, executable tools, and direct examples across several pretrained student/teacher pairs.
5. Charge every byte and every adaptive response—not only the final artifact.

**Kill criteria**

Kill immediately if:

- the theorem is merely conditional rate–distortion, Fano, teaching dimension, or MDL with renamed variables;
- the empirical bound is more than 2× loose;
- estimating it requires knowing the target function;
- it cannot predict unseen task or student ordering;
- the constructive method reduces to ordinary task-specific distillation.

**Narrative**

> “You cannot email an AI skill in fewer bits than the student is missing.”

**Ceiling**

- Exact-result ceiling: **7.5/10**
- Option value: **8/10**
- Cheap first kill: a rigorous novelty audit and nontrivial lemma must survive within five days.

This is the cleanest response to all six failures because a negative or conditional theorem would explain when compression is possible instead of presuming that it is.

### 2. PCI-1 — Proof-Carrying Pocket Intelligence

**Claim**

> Above a measurable proposal-coverage threshold, certified capability becomes largely independent of generator size: a 1–3B local model can match a 30B model’s usable correctness by emitting machine-checkable certificates, at at least 10× lower energy.

“Usable correctness” means an answer is accepted only when a tiny trusted checker validates it.

**Experiment**

Use sealed, natural, machine-checkable tasks in at least three unrelated domains:

- post-cutoff repository repairs with hidden tests;
- formal theorem holes checked by Lean;
- SQL/database transformations;
- constrained scheduling or configuration problems.

Compare 1–3B, 7–14B, and approximately 30B quantized models under:

- greedy generation;
- best-of-\(n\);
- ordinary verifier-guided beam search;
- iterative repair;
- the proposed proof-carrying protocol.

Every model receives the same checker feedback. Measure accepted coverage, false acceptance, joules, latency, memory, and total checker calls.

Required pass:

- the 1–3B system finishes within 3 points of the 30B system using the same protocol;
- zero false accepts in the primary matrix;
- at least 10× lower measured energy;
- passes in three domains;
- ordinary best-of-\(n\) and existing verifier search remain at least 10 points behind.

**Kill criteria**

Kill if:

- standard verifier-guided search comes within 3 points;
- the checker or tests reveal solution-specific information;
- false acceptance exceeds 0.1%;
- the 30B model retains a gap above 5 points;
- total search energy destroys the deployment advantage;
- success is limited to generated puzzles or one programming benchmark.

Verification is already crowded: current work studies compute-optimal solving versus verification, and verifier-guided search has documented scaling failures. [When to Solve, When to Verify](https://arxiv.org/abs/2504.01005), [Scaling Flaws of Verifier-Guided Search](https://arxiv.org/abs/2502.00271) Therefore only the extreme cross-size, cross-domain, lower-energy result earns novelty.

**Narrative**

> “A pocket AI can be trusted because every answer comes with a receipt.”

**Ceiling**

- Exact-result ceiling: **7.5/10**
- Option value: **7/10**
- Estimated feasibility: 4–8 weeks, mostly inference, under roughly 60 GPU-hours.

This is my preferred empirical bet, but only after a 72-hour stock-baseline gate.

### 3. VIL-1 — Village Intelligence Phase Transition

**Claim**

> A fixed network of heterogeneous sub-billion models can synthesize solutions that no member can produce alone, outperforming a much larger monolithic model at lower total energy.

The critical word is **synthesize**. Merely selecting the best member is an ensemble and does not count.

**Experiment**

Use pretrained Transformer, recurrent, and SSM small models. Freeze a single architecture-blind communication protocol before evaluation.

Test on sealed natural tasks with exact external adjudication but no checker feedback during solving:

- multi-file code changes;
- compositional planning;
- multilingual structured reasoning.

Compare against:

- same-model self-consistency;
- majority voting;
- heterogeneous best-of-\(n\);
- learned routing;
- a larger model at matched total joules;
- an oracle that may select any individual member’s answer.

Required pass:

- at least 10 points above the larger matched-energy model across three domains;
- at least 30 solved instances where every constituent fails in repeated individual trials;
- group outputs demonstrably absent from the union of individual final answers;
- at least 20% lower measured energy;
- no hand-assigned expert roles or post-hoc routing.

**Kill criteria**

Kill if:

- gains come only from selecting a member that was already correct;
- a same-model ensemble comes within 3 points;
- the protocol uses test outcomes or hidden judges during reasoning;
- communication cost erases the energy advantage;
- hand-designed roles are necessary;
- there are no robust “all individuals fail, collective succeeds” examples.

Small-model ensembles can already be more FLOP-efficient than single large models, so ordinary ensembling is not novel. [Google Research](https://research.google/pubs/when-ensembling-smaller-models-is-more-efficient-than-single-large-models/) Only genuine beyond-union synthesis would change the scientific picture.

**Narrative**

> “A village of pocket AIs outthought one data-center brain.”

**Ceiling**

- Exact-result ceiling: **8/10**
- Option value: **7/10**
- Probability of passing: low—but failure can be detected cheaply without training new models.

## Recommendation

1. **Do not preregister or implement ACQ-1 as Phase 3.** Preserve it as a possible 3/10 benchmarking exercise, not the flagship.

2. **Do not select another ontology.** “Capability is information,” “intelligence is verification,” and “intelligence is collective” must remain hypotheses, not doctrine.

3. Run a five-day elimination tournament:

   - CCL-1 must produce a nontrivial theorem target that is not a rate–distortion restatement.
   - PCI-1 must close at least 70% of the small/large gap on two domains using existing methods while retaining a projected 10× energy advantage.
   - VIL-1 must produce at least five beyond-union successes on a sealed pilot.

4. Build nothing larger until one survives.

If forced to choose today, choose **PCI-1**, because its primary outcome is direct, independently checkable, economically measurable, and deployable. But the baseline bar must be brutal: if ordinary verifier-guided repair already explains the effect—or cannot close the model-size gap—kill it within 72 hours.

The project does not need a seventh ambitious vocabulary. It needs one result where a poor user’s cheap machine does something a rich user’s large model cannot do as cheaply, and where nobody can dismiss the result by saying the benchmark supplied the trick.

