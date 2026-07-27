Round 2 eliminates PCI-1 and CCL-1 as flagship directions. VIL-1 is not impossible—there is a real beyond-union example—but its basic novelty is already preempted. Your interaction theorem is the strongest conceptual pivot, but as currently stated it is a standard missing-information result, not yet a 7/10 theorem.

## 1. Does PCI-1 survive test-time compute?

**No, not in its current form.**

Proof-carrying search is a subtype of test-time computation. Its comparison class must include:

- self-consistency and majority voting;
- best-of-\(n\);
- revision and backtracking;
- tree search;
- outcome and process verifiers;
- difficulty-adaptive compute allocation;
- early stopping against overthinking.

The precise result in arXiv:2408.00724 is Llemma-7B outperforming Llemma-34B with standard majority voting—not a general 7B-versus-70B theorem—but the strategic threat is real: small-model inference already occupies the cost-performance frontier. [Inference Scaling Laws](https://arxiv.org/abs/2408.00724) Compute-optimal allocation has also beaten models 14× larger on suitable problems. [Snell et al.](https://arxiv.org/abs/2408.03314)

“Overthinking” strengthens the adaptive-allocation baseline rather than rescuing PCI. It shows that additional tokens can become harmful, with optimal budgets varying by difficulty and indicator-based stopping retaining 97% of peak accuracy using 60% of the compute. [When More Thinking Hurts](https://arxiv.org/abs/2604.10739)

PCI would survive only with this result:

> At equal certified accuracy, an interaction with a checker produces an abrupt reduction in required generation compute—at least 10× below the complete compute-optimal test-time-scaling frontier—and this transition replicates across unrelated verifier domains.

Required comparison:

\[
\text{certified success}
\quad\text{versus}\quad
\{\text{joules, latency, tokens, verifier calls, memory}\}
\]

for every baseline, not just greedy generation.

Even then, PCI has a mission ceiling. It helps where society has already paid to construct formal infrastructure: tests, compilers, proof assistants, schemas, or exact constraints. It does not naturally cover agricultural judgment, medical triage, legal interpretation, or local-language advice.

The poor user also cares about:

- whether the device solves their actual task;
- whether an incorrect answer can cause harm;
- whether the checker exists offline;
- how long and how much battery the process consumes.

So:

- **PCI-1 as previously stated:** 4/10.
- **PCI with a genuine 10× convergence transition:** 6–7/10.
- **Current recommendation:** demote, do not select.

“Pocket AI is 23× cheaper per certified software repair” would be a valid numbered headline. “Answers with receipts” alone is not enough.

## 2. One concrete beyond-union synthesis example

Yes. I found one that satisfies the narrow literal criterion.

In **Dynamic Collaboration of Multi-Language Models based on Minimal Complete Semantic Units** at EMNLP 2025, Qwen-2-7B, Llama-3-8B, and GLM-4-9B individually answered a book-page problem incorrectly with **150, 150, and 75**. Their token-level DDS integration produced the correct answer, **300**, which was absent from every model’s individual final output. The paper explicitly labels Table 8 as examples where all three individual models are wrong but the collaboration is correct. [Hao et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.651/)

That is genuine beyond-union-of-final-answers generation.

But it does not save VIL-1:

- It combines token distributions, not independent cheap devices communicating through language.
- All three 7–9B models must run.
- The authors acknowledge increased inference compute, simultaneous model-loading requirements, and poor suitability for resource-constrained deployment.
- Aggregate improvements are modest—for example, HumanEval rises from the best single model’s 79.9 to 81.1.
- A few qualitative cases do not establish a phase transition or systematic emergent capability.

This changes the verdict from “possibly impossible” to “already demonstrated weakly.”

The new VIL bar would be:

> At least 100 precommitted all-members-wrong cases, at least 20% systematic beyond-union recovery, and a 10× cost advantage over the large-model and compute-optimal single-model frontiers.

That is probably feasible to test but highly unlikely to pass. Since beyond-union token fusion is already published, even a pass below that extreme threshold is approximately 5/10.

## 3. Is CCL-1 genuinely new?

**No. I retract it.**

As formulated, CCL-1 is rate–distortion plus teaching/communication complexity:

- Rate–distortion lower bounds and achievability have already been applied directly to neural model compression. [Gao et al., ICML 2019](https://proceedings.mlr.press/v97/gao19c.html)
- Model-output distortion has already been used to characterize pruning limits. [Isik et al., AISTATS 2022](https://proceedings.mlr.press/v151/isik22a.html)
- Communication lower and upper bounds for distributed learning already depend on teaching dimension, mistake bounds, VC dimension, and related task complexity. [Balcan et al., COLT 2012](https://proceedings.mlr.press/v23/balcan12a.html)

“Artifact bits plus teacher-response bits lower-bound what the student can learn” follows the established framework too directly.

A genuinely nontrivial five-day gate would have required all of the following:

1. A formal separation between static transfer and interactive teaching while holding total communicated bits fixed.
2. A superconstant—ideally exponential—gap.
3. Matching lower and upper bounds.
4. A task family not constructed merely to reproduce a known communication-complexity separation.
5. A reduction audit demonstrating that the theorem is not generalized binary search, teaching dimension, active learning, interactive communication, or classical rate–distortion.
6. A prediction about actual model transfer costs that the existing quantities cannot make.

That is no longer “find a nontrivial lemma.” It is “solve a new problem in interactive learning theory.” Possible, but not a responsible four-week flagship bet.

## 4. The interaction-necessity theorem

### The good news

It avoids the exact internal-essence version of the compactness–consequence trap:

- no privileged neural representation;
- no geometry or causal-organ ontology;
- no cross-architecture transplant;
- no assumption that intelligence possesses a compact internal object.

It also attacks a real scale assumption:

> More computation cannot manufacture information absent from the input.

That is philosophically cleaner than ACQ.

### The bad news

As stated, the theorem is almost immediate.

Let:

- \(W\) be the hidden world or user state;
- \(X\) be the information initially given to the model;
- \(A^\star(W)\) be the correct decision.

The best possible noninteractive error, regardless of model size or compute, is the Bayes error:

\[
R_0^\star
=
1-
\mathbb E_X
\left[
\max_a P(A^\star=a\mid X)
\right].
\]

If two possible worlds produce the same \(X\) but require different actions, no noninteractive model can always succeed. An interaction transcript \(T\) changes the posterior:

\[
R_T^\star
=
1-
\mathbb E_{X,T}
\left[
\max_a P(A^\star=a\mid X,T)
\right].
\]

Once a model is powerful enough to implement the Bayes-optimal policy, adding parameters cannot lower \(R_0^\star\). That is the proposed “model-size independence above a floor.”

But it is independence by definition:

- missing information creates irreducible Bayes error;
- receiving information can reduce it;
- extra computation cannot beat Bayes optimality.

Active learning already gives exponential separations between passive and interactive learning for some classes. [Balcan and Long](https://proceedings.mlr.press/v30/Balcan13.html) Decision theory already studies which user questions maximize expected value of information. [Cohn et al.](https://proceedings.mlr.press/v33/cohn14.html) Recent work explicitly quantifies the tradeoff between query-language complexity and interaction rounds. [Active Learning with Simple Questions](https://proceedings.mlr.press/v247/vasilis24a.html)

### Antipattern audit

| Antipattern | Verdict |
|---|---|
| Begins with the answer | **Yes.** It begins by selecting tasks where relevant facts are unavailable without interaction. |
| Surrogates replace consequences | **Partial.** Decision error and cost can be direct, but an abstract task class may still be remote from poor-user utility. |
| Task contains the answer | **Yes.** A hidden variable and query oracle can make interaction necessary by construction. |
| Infrastructure before nontriviality | **Risk.** Four weeks of formalism could rediscover Bayes risk, active learning, or query complexity. |
| Conjunction novelty | **No.** It is at least one clean claim. |
| Destination score leaks | **Yes.** “Bigger AI cannot replace asking” makes a standard observability limit sound like a new intelligence theorem. |

So it avoids the compactness–consequence trap but enters a new one:

> **The observability–consequence tautology:** hide necessary information from the agent, permit a query that reveals it, then prove that querying is necessary.

There is also a manifesto complication: user interaction is not free. Poor users may lack time, literacy, connectivity, expert knowledge, or reliable observations. In medical, legal, and agricultural settings, the user may not know the fact the model needs. A theorem that prices compute but treats human attention and answer error as free would fail the mission.

## The salvageable version

The right question is not:

> Can we prove interaction is necessary?

It is:

> Under an equal total cost, when does adaptive interaction outperform additional model size and additional test-time computation—and by how much?

Call it the **Compute–Interaction Frontier**, but do not yet call it a law.

The potentially nontrivial object is the adaptive advantage:

\[
\Gamma(\epsilon)
=
\frac{
\text{minimum cost of any nonadaptive system reaching error }\epsilon
}{
\text{minimum cost of an adaptive system reaching error }\epsilon
}.
\]

The comparison must charge:

- inference joules;
- latency;
- user seconds;
- answer bits;
- user error probability;
- connectivity;
- model memory;
- number and complexity of questions.

Critically, compare adaptive interaction against:

- a larger model;
- 100× test-time compute;
- a one-shot request receiving the same number of answer bits;
- fixed questions;
- random questions;
- complete-context oracle;
- small and large interactive models.

The surprising result would be:

> Two adaptive questions let a 3B offline model beat a 30B model with 100× test-time compute, at 25× lower total cost—including the user’s time—across three natural task families.

That is a numbered, direct consequence. It is not a theorem yet.

### Five-day theorem gate

Continue only if, by day five, you can state a theorem satisfying all of these:

1. The result compares adaptive and nonadaptive access under the same external-information budget.
2. Internal computation is unrestricted or defined using a legitimate complexity measure—not neural parameter count.
3. The gap is superconstant, preferably exponential.
4. User responses are noisy and queries have heterogeneous costs.
5. The task class is independently motivated rather than manufactured around the query protocol.
6. The theorem cannot be obtained by substituting variables into Fano, value-of-information, generalized binary search, active learning, decision-tree complexity, or interactive communication results.
7. There is a matching or near-matching constructive policy.

If item 6 fails, the theorem is dead even if the proof is elegant.

## Final verdict

- **PCI-1:** does not survive ordinary compute-optimal inference as a flagship.
- **VIL-1:** beyond-union synthesis exists, but only weakly; basic novelty is preempted.
- **CCL-1:** retracted as rate–distortion/communication theory with new nouns.
- **Interaction necessity as stated:** 3/10; true but close to tautological.
- **Compute–Interaction Frontier with a new adaptive separation and real cost accounting:** conditional 7/10, but it has not earned that score.

Your direction is the best question remaining, but the scientifically honest formulation is:

> **When does asking the world beat thinking harder?**

Not:

> **We will prove that bigger AI cannot replace asking.**

The first is an open tournament between computation and information. The second begins with the answer—and that is how the previous six directions died.