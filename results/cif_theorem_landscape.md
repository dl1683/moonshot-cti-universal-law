# CIF Theorem Gate: Internet Research Landscape

Claude internet research (Jul 27, 2026) for the Compute-Interaction Frontier
5-day theorem gate. The 7 criteria from Codex R12 R2 are the standard.

## Existing Results That Threaten Novelty (Criterion 6)

### 1. Adaptive vs Non-Adaptive Query Complexity (Property Testing)
- Exponential separations exist: any q-query adaptive tester can be simulated
  non-adaptively with 2^q - 1 queries (at most exponential blowup).
- k-juntas: adaptive O_tilde(k) vs non-adaptive O_tilde(k^{3/2}) (polynomial gap)
- "An Adaptivity Hierarchy Theorem for Property Testing" (Blais & Yoshida, 2018):
  Hierarchy of properties with increasing adaptivity gaps.
- These are NOISELESS, unit-cost results. But the framework is mature.

### 2. Noisy 20 Questions
- "Resolution Limits for the Noisy Non-Adaptive 20 Questions Problem" (2019/2020)
- "Achievable Resolution Limits for the Noisy Adaptive 20 Questions Problem" (2021)
- Gap between adaptive and non-adaptive in noisy case is SECOND-ORDER (constant factor),
  NOT superconstant. Repetition overcomes noise.
- CIF criterion 3 demands superconstant gap. Noisy 20Q suggests this is hard to get.

### 3. Active Learning Separations
- Balcan & Long (COLT 2013): exponential separation between interactive and passive PAC learning
  for specific function classes (linear separators, decision lists).
- "Active Learning with Simple Questions" (COLT 2024): explicit query-language
  complexity vs interaction rounds tradeoff.
- CIF criterion 6 explicitly says "not obtainable from active learning by substitution."
  But the CIF setting (user answers questions about hidden state) IS active learning.

### 4. Adaptive Submodularity (Golovin & Krause, 2011)
- General framework: if objective is adaptive submodular, greedy achieves 1-1/e.
- Adaptivity gap bounded. Well-established theory.
- CIF's "adaptive advantage ratio" could be a special case of the adaptivity gap
  under adaptive submodularity.

### 5. Cognitive Friction / TCA (arXiv 2603.30031, Mar 2026)
- **DIRECT COMPETITOR.** Formalizes when tool-using agents should query vs act.
- Uses HJB optimal stopping theory with congestion-dependent costs.
- Empirical: 36 viability point improvement over greedy.
- Does NOT prove a separation theorem, but the FRAMEWORK is close to CIF.
- If CIF claims "new framework for when to ask vs compute," TCA preempts.

### 6. Sequential Testing with Heterogeneous LLMs (arXiv 2604.01086, Apr 2026)
- Asymptotically optimal strategies for querying heterogeneous LLMs.
- Proves: at most 2 LLMs optimal as error -> 0.
- Matching upper/lower bounds (1+o(1)).
- Addresses heterogeneous costs and accuracies.
- CIF criterion 4 (heterogeneous costs) is partially covered.

### 7. Unified Interactive Lower Bounds (arXiv 2410.05117, Oct 2024)
- "Assouad, Fano, and Le Cam with Interaction" — unified lower bound framework.
- Interactive Fano method + Fractional Covering Number.
- Characterizes learnability for any stochastic bandit problem.
- The TECHNIQUE for proving interactive lower bounds is mature.

### 8. Value of Information in Sequential Decision-Making
- Costly sequential information acquisition (arXiv 2401.00569, 2024)
- Viscosity solutions for optimal stopping with costly signals.
- Mature field (POMDPs, optimal stopping, Bayesian experimental design).

## Existing Empirical Evidence (CIF phenomenon exists)

### 9. "Reasoning While Asking" (arXiv 2601.22139, Jan 2026)
- PIR (Proactive Interactive Reasoning): interleave reasoning with clarification
- +32.7% accuracy (math), +22.9% pass rate (code), +41.36 BLEU (editing)
- ~50% reduction in reasoning computation
- Directly shows: asking beats thinking harder

### 10. Uncertainty-Aware Clarification (arXiv 2606.03135, Jun 2026)
- Small model with clarification: 17.8% vs large model (DeepSeek-V3.1): 18.1%
- Only 1.3 turns average vs 3.9-5.1 for large model
- Near-parity with dramatically fewer resources

### 11. Small Model + Questions = Large Model (general trend)
- Multiple 2025-2026 papers show small specialized models matching larger ones
- Test-time compute scaling (ICLR 2025): 7B + 100x inference matches 34B
- The EMPIRICAL phenomenon is robust. No THEOREM explains it.

## Gap Analysis: What Does NOT Exist

1. **No formal comparison of adaptive USER interaction vs test-time COMPUTE scaling.**
   Existing work compares either adaptive vs non-adaptive queries OR compute scaling.
   Nobody compares "ask 2 questions" vs "run 100x more inference."

2. **No three-way frontier (model size x interaction x compute).** Only two-way
   frontiers exist (model x compute from scaling laws, adaptive x non-adaptive from query
   complexity). The three-way object is unstudied.

3. **No superconstant separation in the noisy, heterogeneous-cost, decision-making setting.**
   Noisy 20Q gives constant factors. Active learning gives exponential but in different setting.

4. **No formal model pricing user time, answer noise, and compute jointly.**
   Cognitive Friction prices tool use and deliberation but not user cognitive cost.

## Criterion-by-Criterion Assessment

| Criterion | Passable? | Threat |
|-----------|-----------|--------|
| 1. Same external-info budget | YES | Standard constraint |
| 2. Unrestricted internal computation | PROBLEM | With unlimited compute, advantage is purely informational -> tautological |
| 3. Superconstant gap | HARD | Noisy 20Q gives constant. Need special structure. |
| 4. Noisy responses, heterogeneous costs | YES | Novel combination, partially covered by 2604.01086 |
| 5. Independently motivated task class | POSSIBLE | Medical, legal, agricultural tasks are natural |
| 6. Not obtainable from known results | VERY HARD | Reduces to active learning or Bayes risk under most formulations |
| 7. Matching constructive policy | DEPENDS | If theorem exists, policy follows from proof |

## The Core Dilemma

With unrestricted computation (criterion 2):
- Adaptive advantage is PURELY information-theoretic
- "User has info not in X" -> asking helps (tautological)
- "User doesn't have info not in X" -> asking doesn't help (trivial)
- No superconstant gap exists because Bayes-optimal is Bayes-optimal

With bounded computation:
- Reduces to query complexity + computational complexity
- Known separations exist -> criterion 6 fails

## Honest Assessment

The 5-day theorem gate will probably FAIL at 7/10.

The empirical phenomenon (small model + questions beats large model) IS REAL.
But the THEOREM explaining it is either:
(a) tautological (information the user has and X doesn't), or
(b) reducible to known results (active learning, query complexity, Bayes risk), or
(c) a conjunction of known techniques (criterion 6 / antipattern 5).

The gap between "robust empirical phenomenon" and "novel theorem" is the problem.

## Possible Surviving Angle (Speculative)

The ONLY angle I can see surviving criterion 6 is if the interaction between
COMPUTATIONAL cost and INFORMATIONAL cost creates a phase transition that neither
alone exhibits. Specifically: if there exists a task family where the three-way
frontier (model size, interaction, compute) has a nontrivial topology — a region
where interaction is exponentially cheaper than compute, and this region is NOT
predicted by any two-way frontier.

This would require the interaction between capacity constraints and information
constraints to be multiplicative, not additive. I.e., the cost of extracting
information from X by computation is exponential in some parameter, while obtaining
the same information by asking is linear, AND this relationship doesn't follow from
either complexity theory or information theory alone.

This is a high bar. But it's the only angle that might survive.
