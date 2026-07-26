# Question Loop 07 — Evidence Ladder: From "Useful Loss" to "Portable Geometric Program"

**Date:** 2026-07-25
**Decision:** Reserve "portable geometric program" for the final rung of a 5-rung cumulative evidence ladder. The 6.7x Qwen3-4B-to-0.6B pilot can reach at most Rung 2 (useful auxiliary loss). The moonshot claim requires Rung 5.

## The ladder

Each rung is **cumulative** — higher rungs require all lower ones to hold. No rung can be skipped. Evidence quality increases monotonically. The claim at each rung is the strongest claim the evidence supports.

### Rung 1: Functional auxiliary loss

**Claim:** A_r matching produces a working distillation loss that trains to completion without divergence.

**Evidence required:**
- Student trains successfully with A_r loss component.
- Loss decreases and downstream accuracy improves over SFT baseline.
- No numerical instabilities or degenerate solutions.

**What this proves:** The loss function works. Nothing about transfer, geometry, or universality.

### Rung 2: Competitive auxiliary loss

**Claim:** A_r-based distillation beats standard KD and matches or exceeds the best existing geometry-aware methods under exact resource matching.

**Evidence required (from QL4 baseline manifest):**
- Full/skew-R_r arm beats logit KD, FDD, MTA, Procrustes/Gram, LoRi, TSD-KD, and RG-OPD at identical data, teacher calls, optimizer steps, peak memory, and FLOPs.
- >= 3 absolute accuracy points with positive bootstrap 95% LCB.
- >= 20% incremental teacher-gap closure over strongest baseline.
- Skew-containing arm contributes >= 1.5 points beyond path-Gram + symmetric strain.
- Controls fail: depth permutation, skew sign-flip, spectrum-matched random skew.
- Reproduces over seeds and a second reasoning family.

**What this proves:** The method is competitive KD engineering. Still not transfer.

**Maximum reach of the Qwen3-4B-to-0.6B pilot (6.7x compression).**

### Rung 3: Teacher-specific transfer

**Claim:** The gain is caused by teacher-specific computational structure, not generic regularization.

**Evidence required (from QL6 tests):**
- Teacher identity specificity: crossed-competence teachers produce crossed student improvements.
- Causal use: ablating the matched subspace degrades capability; output Jacobian routes through it.
- Timing separation: persistent effect beyond warm-start window.
- Matched-strength regularizers fail to replicate the gain.
- A-only witness cannot exploit the null space at chance.

**What this proves:** The teacher's geometry carries task-relevant information to the student. This is genuine transfer, not regularization.

### Rung 4: Cross-substrate portability

**Claim:** The same geometric specification transfers across architecturally incompatible substrates.

**Evidence required:**
- Transformer-to-SSM (e.g., Qwen3-4B teacher to Mamba student) with same specification.
- Transformer-to-RNN (e.g., to RWKV student) with same specification.
- Same A_r/R_r target used across all substrates (the specification is reusable).
- Each substrate matches or exceeds its own best KD method.
- >= 10x parameter compression (the stated moonshot target).

**What this proves:** The geometric program is substrate-independent. The specification is a portable representation of the teacher's computation.

### Rung 5: Portable geometric program (the moonshot)

**Claim:** A compact geometric specification extracted once from a teacher enables any cheap substrate to acquire the teacher's differential competence at minimal cost.

**Evidence required (cumulative over all previous rungs):**
- All of Rungs 1-4.
- Specification is compact: low effective rank of R, O(Bsd) sketches sufficient.
- One-time teacher trace extraction amortizable across students.
- No student inference overhead.
- Student training overhead small relative to saved compute.
- Robust to probe resampling and held-out evaluation.
- Works on at least two genuinely different tasks (not just math reasoning).
- Survives a fresh adversarial full-repo review (Invariant #2).

**What this proves:** Intelligence = Geometry, operationalized. The teacher's computational structure is a transferable, substrate-independent geometric object. This is the democratization result: expensive teacher, cheap student, portable program.

## Current position on the ladder

**Rung 0** — no empirical evidence yet. The question loop has established theoretical groundwork (QL1-5) and mapped the baseline frontier (QL4), counterexample threats (QL5), and transfer/regularization distinction (QL6). The pilot experiment has not run.

## The 6.7x ceiling

The Qwen3-4B-to-0.6B pilot is only 6.7x compression. Even at Rung 2 success, the evidence does not support the moonshot narrative:

- 6.7x is not "expensive teacher, cheap student" — both models run on the same hardware class.
- Same-family transfer (Qwen3 to Qwen3) doesn't test substrate portability.
- GSM8K alone is insufficient (QL5).

The pilot licenses the next experiment (larger compression, cross-substrate, harder tasks). It cannot itself satisfy the mission.

## The cost of each rung

| Rung | Estimated GPU-hours (RTX 5090) | Risk of failure |
|------|-------------------------------|-----------------|
| 1 | 4-8h (single training run) | Low (loss should work) |
| 2 | 40-80h (full baseline sweep, 8+ arms, 3 seeds) | Medium (must beat strong baselines) |
| 3 | 20-40h (crossed teachers, ablations, timing) | High (regularization is the default) |
| 4 | 60-120h (cross-substrate, new student architectures) | Very high (gauge/clock issues) |
| 5 | Cumulative + validation | Very high (everything must hold) |

Total estimated path to Rung 5: ~150-300 GPU-hours if everything works. Much more with failures and pivots.

## Kill conditions at each rung

- **Rung 1 kill:** Loss diverges or student doesn't improve over SFT. -> Abandon A_r matching entirely.
- **Rung 2 kill:** Cannot beat the baseline Pareto set under resource matching. -> A_r is a weak loss; try different objectives.
- **Rung 3 kill:** Wrong-teacher and random targets match correct-teacher gains. -> A_r is regularization. Useful but not the moonshot. Publish as KD method and redirect.
- **Rung 4 kill:** Cross-substrate transfer fails (gauge/clock breaks on SSM/RNN). -> Within-family transfer only. Scale-limited story, not universal.
- **Rung 5 kill:** Specification is not compact, not reusable, or requires per-student tuning. -> Good distillation method, not portable program. Publish and redirect.

## NARRATIVE ATTACK

"You have a ladder with five rungs and zero data points. The theoretical work is thorough — but theory without experiment is philosophy, not science. The ladder itself is not novel; it's a restatement of the standard burden of proof for any transfer claim. Show me Rung 1 data before discussing Rung 5."

This attack is fair. The next action is running the pilot to reach Rung 1, with Rung 2 success criteria precommitted.

## MISSION TEST

The ladder ensures honest claiming. Each rung earns exactly the narrative it supports. The mission (cheap, ubiquitous intelligence) is served only at Rung 5. Everything below is useful science or engineering, honestly labeled. The ladder prevents premature celebration and late-stage deflation.

## Next action

1. Precommit the full Rung 1-2 protocol (baseline manifest, resource accounting, success criteria) in a JSON preregistration.
2. Run the pilot.
3. Score results against precommitted criteria.
4. If Rung 2 passes, design Rung 3 experiments (QL6 tests).
5. If Rung 2 fails, diagnose and decide: iterate or redirect.
