# Revised verdict

You are right on all three structural objections:

- Jumping from the register transducer to Qwen/Pythia is not a bridge. It is “solve mechanistic interpretability.”
- Composition should not precede credible extraction.
- CSO is not yet the direction. It is one surviving hypothesis after the previous direction died.

I retract “make unconstrained capability extraction the next experiment” and “make composition the immediate flagship.”

My recommendation is:

> Finish the frozen CSO admission test, preserve the result, and then temporarily fork away from transplantation. Run a 4–8 week Causal World Compression experiment testing whether planning cost scales with **action-distinguishable causal complexity rather than sensory complexity**.

If perfect, that experiment has a realistic **6–7/10 ceiling**. It can be done with small models and exact oracle evaluation on one RTX 5090. It does not require solving arbitrary-model interpretability.

# 1. The gap problem

There is no honest one-GPU path from the current organ to arbitrary capability extraction from Qwen3 or Pythia in one step.

The present experiment gives the extractor:

- a designated recurrent state boundary;
- a known action alphabet;
- deterministic dynamics;
- known task segmentation;
- interventions at the exact presumed causal boundary;
- hosts designed around a compatible socket.

A pretrained LLM gives none of these. Its capabilities may be:

- spread across layers and token positions;
- conditional on prompts and surrounding knowledge;
- implemented by multiple redundant pathways;
- created dynamically at inference;
- entangled with language and factual memory;
- nonstationary across examples.

Removing the designated slot alone changes the problem qualitatively.

## The only credible CSO bridge

If CSO passes and we insist on continuing its ladder, the achievable bridge is not Qwen. It is:

### CSO-WM0: unprivileged small-world-model extraction

Train a 10–30M recurrent Transformer world model on a partially observable, multi-goal pixel environment, but give it:

- no designated causal state slot;
- no register labels;
- no handcrafted organ boundary;
- no single fixed output task.

Then search over a constrained set of whole-layer/time-indexed subspaces for a causally sufficient predictive state, fit a standalone state-transition surrogate, and test it in two small hosts on unseen goals.

Preliminary success would require:

- at least 90% novel-goal planning success;
- at least 15 points over observational distillation;
- no manual layer selection after seeing results;
- no ground-truth state supervision;
- identical organ bytes in Transformer and GRU hosts;
- intervention fidelity above 90%;
- organ ablation removing at least 80% of the transferred advantage.

That is feasible on a 5090 because the donor is small and the causal search space is deliberately bounded.

But its ceiling is only approximately **5/10**. It remains a synthetic world model and still does not establish general capability compilation.

So: a bridge exists, but it does not solve the impact problem.

# 2. The chicken-and-egg

Agreed. Composition is premature.

The correct CSO ladder is:

```text
structured mechanism
        ↓
unprivileged world-model mechanism
        ↓
multiple independently extracted mechanisms
        ↓
composition
```

Skipping from the first to the fourth rung would create another inflated conjunction claim.

Composition should become a flagship only after:

1. At least two independently learned nontrivial capabilities have been extracted.
2. Each works alone in multiple hosts.
3. Each survives strong direct-training and distillation baselines.
4. Their interfaces were not jointly designed around the target composition.

Until then, “zero-shot composition” must remain an unearned future claim.

# 3. The mechanistic-interpretability trap

There is no secret laptop advantage that lets us beat Anthropic or DeepMind at arbitrary circuit localization.

We have only three defensible strategies:

## A. Change the problem from circuit identity to functional identification

Learn an architecture-independent causal model using active queries and interventions, without claiming it is materially the donor’s original circuit.

This competes with system identification, predictive-state learning, automata learning, and causal abstraction—not full mechanistic interpretability.

It is feasible, but the correct term is **causal emulator**, not extracted circuit.

## B. Train models to expose mechanisms

Design training so models form stable, standardized causal interfaces. This asks:

> Can extractability be made a learned architectural property?

That may be useful, but it is modular architecture research, not discovery inside arbitrary pretrained models.

## C. Restrict extraction to previously localized capabilities

Use known circuits such as entity tracking or algorithmic tasks and test whether their causal computations can be rendered standalone.

That is experimentally manageable but scientifically narrow. Circuit Distillation already transfers targeted mechanisms within a model family, so the remaining cross-architecture increment may not clear 6/10 ([Circuit Distillation](https://arxiv.org/abs/2509.25002)).

Therefore the honest strategic conclusion is:

> Do not compete head-on with frontier mechanistic interpretability. Either redefine CSO as active functional system identification, or leave CSO until a smaller scientific result gives us a genuine lever.

# 4. Alternative-direction tournament

The old top-five list is not binding. It was a February speculative ranking, before the current kill history. More importantly, the root manifesto still says every project should prove “Intelligence = Geometry” ([manifesto](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/CLAUDE.md:9>)). Phase 1 has now falsified that as a mandatory scientific answer. The two mission invariants survive; the geometry axiom should no longer participate in direction selection.

| Direction | 5090 feasibility | Perfect-result ceiling | Main problem |
|---|---:|---:|---|
| Arbitrary-model capability compiler | 1/10 | 9/10 | Requires unsolved mechanistic localization |
| CSO unprivileged world-model rung | 6/10 | 5/10 | Still synthetic; transplantation may be distillation |
| Energy-Native Intelligence | 8/10 | 4/10 | Risks being efficiency engineering |
| Compression-Driven Math Discovery | 4/10 | 5–6/10 | Crowded and evaluation is difficult |
| Causal World Compression, broadly stated | 6/10 | 5/10 | Existing causal-state/bisimulation/PSR literature |
| **Actionable Complexity Law** | **6/10** | **6–7/10** | Must show a scaling separation, not just distractor robustness |

## Energy-Native Intelligence

“Joules per correct answer” is a good evaluation discipline, but not itself a paradigm. Energy/accuracy metrics, early exits, conditional computation, hardware-aware search, and neuromorphic inference already exist; energy-precision ratios were proposed years ago ([NeuralPower](https://proceedings.mlr.press/v77/cai17a.html)).

To become moonshot science, ENI needs a new law or computational primitive. Merely training an early-exit controller would score approximately 3–4/10.

## Compression-Driven Math Discovery

The original thesis—useful lemmas are compression primitives—is attractive, but no longer open territory:

- REFACTOR extracts reusable theorems that shorten existing proofs and improve downstream proving ([REFACTOR](https://openreview.net/forum?id=827jG3ahxL)).
- Recent work explicitly synthesizes lemmas using DAG compression ([Wernhard 2026](https://arxiv.org/abs/2602.15511)).
- Current theorem-generation work already frames theorem usefulness as proof-library compression.
- Growing verified lemma libraries are established in systems such as LEGO-Prover ([LEGO-Prover](https://arxiv.org/abs/2310.00656)).

A laptop project could still contribute, but a simple “compression gain predicts useful lemmas” experiment is now unlikely to score 6.

## Causal World Compression

The broad idea also has substantial prior art:

- Causal states as minimal predictive representations are decades old ([Shalizi and Crutchfield](https://arxiv.org/abs/cond-mat/9907176)).
- Causal-state representations have already been learned in POMDPs and connected to bisimulation ([Zhang et al.](https://arxiv.org/abs/1906.10437)).
- Compressed predictive-state models have been used for planning ([Hamilton et al.](https://www.jmlr.org/papers/v15/hamilton14a.html)).
- Reward-free agent-centric representations already address rich exogenous visual information ([ACRO](https://proceedings.mlr.press/v202/islam23a.html)).
- Controlled JEPA identifiability is an active current research topic ([Zhang et al. 2026](https://arxiv.org/abs/2607.22430)).

So “JEPA plus causal states” is another Venn diagram.

The opening is narrower:

> Establish an empirical scaling law separating actionable causal complexity from observation entropy.

That is a scientific claim rather than a method bundle.

# 5. The pattern behind five dead ends

The kill discipline is good. Five falsifications are preferable to five inflated papers.

But there is a systematic pattern.

## A. The projects begin with an answer

The old manifesto said intelligence equals geometry. That encouraged searches for geometric confirmation instead of neutral searches for efficiency arbitrage.

After geometry died, the program immediately substituted another essence:

```text
intelligence is geometry
        ↓
capabilities are causal organs
```

The noun changed, but the cognitive habit remained: propose a deep ontology first, then build a synthetic experiment in which that ontology is present by construction.

## B. Surrogates replace consequences

Repeated substitutions include:

- representation quality for intelligence;
- nearest-neighbor geometry for capability;
- benchmark correlation for economic consequence;
- installed geometry for functional transfer;
- finite-state transplantation for portable real skills;
- parameter or byte reduction for democratization.

The experiment becomes rigorous about the surrogate while the gap to the mission remains unmeasured.

## C. The chosen tasks contain the desired answer

The register transducer genuinely is a compact causal state-transition mechanism. Therefore it is an easy environment in which to prove that causal mechanisms can be represented compactly.

But the scientific question is whether messy learned capabilities have that property. The task assumes the key thesis instead of making the learner discover it.

## D. Infrastructure arrives before nontriviality

Several projects built elaborate protocols before resolving the cheapest hostile objection:

- Is predictor nearly the target?
- Is the strongest result a null artifact?
- Is the benchmark appropriate for the model scale?
- Is the object vastly larger than the true program?
- Does the experiment supply the causal boundary?

The new rule should be:

> Before implementation, identify the one result that would remain surprising after the strongest known baseline and the shortest-description attack.

## E. “No one combines all five” substitutes for a new principle

Conjunction novelty has repeatedly made familiar ingredients appear moonshot-sized. A moonshot needs a surprising relation, law, impossibility result, or capability—not merely an unoccupied table cell.

## F. Direction scores have evaluated destination rather than experiment

The 9/10 vision leaks backward into the score of a 2–3/10 admission test.

Every experiment should now receive two independent scores:

- **result ceiling:** importance if this exact experiment passes;
- **option value:** how much it reduces uncertainty about a later 9/10 program.

CSO admission is approximately 3/10 result ceiling, perhaps 5/10 option value.

# 6. Minimum viable paradigm shift

## Proposed experiment: ACQ-1 — Actionable Causal Quotient

### Central claim

> The compute required for planning scales with the number of intervention-distinguishable world states, not with the amount of sensory data.

Informally:

> **AI should model what its actions can change, not every pixel it can see.**

This is more specific than “causal world compression” and more consequential than distractor robustness.

The experiment asks whether we can increase observation entropy by orders of magnitude while leaving:

- learned state count;
- world-model size;
- planning compute;
- training sample complexity;
- goal success

approximately unchanged.

At the same time, increasing actual causal/control complexity must increase those quantities predictably.

That crossed scaling test is the core result.

## Formal object

Let histories \(h\) and \(h'\) be equivalent when no permitted future action sequence can produce distinguishable endogenous outcome distributions:

\[
h \sim h'
\iff
\forall a_{1:k},\;
P(Y_{t+1:t+k}\mid h, do(a_{1:k}))
=
P(Y_{t+1:t+k}\mid h', do(a_{1:k}))
\]

The quotient \(H/{\sim}\) is the actionable causal state.

The learner does not receive simulator state labels. It receives:

- pixel histories;
- actions;
- ordinary trajectories;
- paired nuisance interventions that re-render or alter exogenous variables while preserving physical state;
- the ability to execute alternative action suffixes from saved snapshots.

Ground-truth quotient labels are available only to the final evaluator.

## Environments

Use three finite but visually rich, partially observable families:

1. **Key–Door Memory**
   - Hidden possession and door state.
   - Long partial-observation histories.
   - Procedural layouts and goal locations.

2. **Switch Maze**
   - Multiple switches with noncommuting effects.
   - Stochastic transitions.
   - Visually identical observations with different hidden histories.

3. **Sokoban-Lite**
   - Irreversible actions.
   - Planning failures cannot be repaired by reactive control.
   - Held-out layouts and goal states.

Each underlying environment is rendered as 64×64 RGB with controlled exogenous complexity:

- zero nuisance;
- static textures;
- independently moving sprites;
- natural-video backgrounds;
- action-correlated distractors;
- unseen renderer families at test time.

Vary nuisance entropy over at least a 16× range while holding the causal transition system fixed.

Separately vary the true causal quotient size, for example:

\[
|\mathcal S_c| \in \{32,64,128,256,512\}
\]

while holding renderer complexity fixed.

## Learning method

### Active Causal Quotient Refinement

Train:

\[
z_t = E(h_t), \qquad z_{t+1}=T(z_t,a_t)
\]

with a small discrete or strongly quantized bottleneck.

Use four losses:

1. **Nuisance-intervention invariance**  
   Paired re-renderings of the same physical snapshot must map to the same code.

2. **Action-conditioned predictive sufficiency**  
   The latent transition must predict future latent distributions under alternative action suffixes.

3. **Counterexample-guided refinement**  
   Search for two histories currently assigned the same code and an action suffix that makes their outcome distributions diverge. When found, split the representation.

4. **Description-length penalty**  
   Penalize active code count and transition-model complexity so the model cannot preserve all pixel information.

The distinctive component is #3. Paired augmentation alone is an obvious baseline. Active suffixes must discover precisely which apparently similar histories need separation for control.

## Downstream evaluation

Freeze the encoder and latent dynamics before revealing test goals.

Supply 100 held-out goal images or goal predicates. No representation updates and no policy training are allowed.

Plan entirely inside the learned quotient using a fixed planner. Compare against an oracle planner using the true causal quotient.

Include:

- unseen start states;
- unseen goals;
- horizons twice the training horizon;
- unseen layouts;
- unseen visual renderers;
- action-correlated nuisance reversal.

## Baselines

At matched trajectories, parameters, latent dimension, and planning budget:

1. Pixel next-frame world model.
2. RSSM/Dreamer-style reconstruction model.
3. Action-conditioned JEPA.
4. Passive causal-state representation learner.
5. Multi-step inverse-dynamics/ACRO-style representation.
6. Bisimulation representation.
7. RePo/information-bottleneck world model.
8. Paired nuisance invariance without active counterexamples.
9. Active prediction without the bottleneck.
10. Oracle causal quotient.

The most dangerous baselines are #4, #5, and #8.

# Frozen success criteria

All primary gates should be required.

## Gate A: causal-state recovery

Across all three environment families:

- pairwise quotient-equivalence F1 at least 0.95;
- learned active code count between 0.5× and 2× the oracle minimal quotient;
- no ground-truth latent variables used during learning.

## Gate B: arbitrary-goal planning

- At least 90% success over held-out start/goal pairs.
- Within 5 percentage points of the oracle causal-state planner.
- At least 85% success at twice the training horizon.
- No goal-specific representation or dynamics training.

## Gate C: nuisance invariance

When nuisance entropy increases by at least 16×:

- planning success drops no more than 5 points;
- active latent-state count changes by no more than 10%;
- planning MACs change by no more than 20%;
- required environment interactions increase by less than 25%.

## Gate D: causal-complexity scaling

When true causal quotient size increases 16×:

- learned code count tracks it monotonically;
- planning compute and sample requirements increase predictably;
- nuisance entropy explains less than 10% as much variance in compute as causal-state complexity.

The precommitted primary regression should be:

\[
\log C =
\alpha\log |\mathcal S_c|
+\beta H(N)+b
\]

Admission requires:

\[
\alpha > 0,\qquad |\beta| < 0.1\alpha
\]

with confidence intervals excluding \(\beta \ge 0.25\alpha\).

## Gate E: nontrivial active advantage

At equal data and model size:

- at least 15 points planning advantage over the best passive representation baseline under action-correlated nuisance;
- at least 10 points over paired-invariance training without counterexample refinement;
- at least 10× reduction in latent rollout compute or artifact size versus the best baseline reaching comparable success.

## Gate F: replication

- All three environment families pass.
- At least four of five seeds pass each primary threshold.
- One standard distractor benchmark must reproduce the direction of effect, even if exact quotient recovery is unavailable there.

# Kill criteria

Kill the distinctive claim if any occurs:

1. Paired invariance plus ordinary JEPA comes within 3 points.
2. ACRO, causal-state learning, or bisimulation comes within 3 points at matched size/data.
3. Active refinement improves representation metrics but not planning.
4. Learned code count or compute grows materially with nuisance entropy.
5. Goal-specific fine-tuning is required.
6. Oracle latent labels are needed.
7. Success is confined to one custom environment.
8. The standard distractor benchmark reverses the effect.
9. More than 80 GPU-hours are needed without all synthetic primary gates passing.
10. The active intervention protocol requires information unavailable from saved snapshots and rerendering.

# VOID criteria

- Oracle state or quotient labels enter training.
- Test goals influence representation selection.
- Different methods receive different trajectory coverage.
- Renderer seeds overlap across train and test.
- Baseline latent sizes are not matched or swept.
- Hyperparameters are selected on the final nuisance-shift split.

# Schedule and budget

## Weeks 1–2: freeze the scientific object

- Implement and exhaustively verify the three environment families.
- Compute oracle causal quotients.
- Implement renderer interventions.
- Freeze splits, entropy levels, goals, hashes, and regression.
- Reproduce two strongest external baselines before implementing the new method.

**Kill immediately** if the oracle quotient itself does not offer at least 10× planning compression over a pixel-state model. That checks whether the environment contains a meaningful efficiency arbitrage.

## Weeks 3–4: minimal algorithm

- Bottleneck encoder and latent transition.
- Paired-invariance baseline.
- Passive causal-state baseline.
- Counterexample-guided refinement.
- CPU and one-seed smoke tests only.

## Weeks 5–6: frozen primary matrix

- Three environment families.
- Five seeds.
- Nuisance and causal-complexity sweeps.
- All baselines.
- No post-result threshold changes.

## Weeks 7–8: externality test

- One Distracting Control Suite or similarly standard pixel-control benchmark.
- Hardware energy and wall-clock measurement.
- Final adjudication and claim audit.

Budget: approximately 60–80 GPU-hours, with models below roughly 5M parameters.

# Why this moves the needle

The register experiment asks:

> Can a compact mechanism be represented compactly?

ACQ-1 asks:

> Can a learner discover the amount of world that actually matters for action, and can its compute become independent of everything else it sees?

A perfect result would support a sharp alternative to scale:

\[
\boxed{\text{Cost of useful intelligence} \propto
\text{actionable causal complexity},
\quad\text{not sensory entropy}}
\]

The viral result would be:

> “We made the AI’s visual world a million times noisier. Its brain did not get larger, slower, or worse—because it learned to model only what actions could change.”

That is still not 9/10. The worlds are controlled, interventions are simulator-assisted, and the result would require external replication. But it is a legitimate 6–7/10 minimum viable paradigm shift.

# Final strategic recommendation

1. **Complete CSO admission unchanged.**
2. **Do not automatically proceed to organ composition or Qwen extraction.**
3. **Pre-register ACQ-1 during the final CSO phase.**
4. **Run ACQ-1 as an independent direction test.**
5. If ACQ-1 passes, Causal World Compression becomes the flagship and CSO becomes a possible deployment mechanism later.
6. If it fails against ACRO/JEPA/bisimulation, kill it quickly—without building another large doctrine.

CSO is not necessarily wrong. But it has not earned the right to monopolize the portfolio merely because it was the most recent pivot.

