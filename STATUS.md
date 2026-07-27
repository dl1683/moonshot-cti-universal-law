# CTI Universal Law — Program Status

## Honest Assessment (Jul 25, 2026 — Codex Steering Dialogue)

**Nobel trajectory: 2/10.** The prior 8/10 was structurally inflated.

CTI is a sophisticated, theoretically motivated assay of labeled nearest-neighbor geometry.
It has repeatedly mistaken breadth of internal validation for depth of external consequence.

### Critical findings from steering dialogue

1. **Predictor and target are almost the same object.** q is 1-NN accuracy; kappa is a summary of the same geometry. A strong relationship is less surprising than predicting independent capability.
2. **The canonical R^2=0.955 uses per-dataset intercepts.** LODO is r=0.125. The law calibrates after being told which task, but cannot predict on a new task.
3. **LOAO CV=2.3% is overlapping jackknife.** Not 12 independent measurements of a constant. Small variation is expected.
4. **rho=1.0 downstream is scale confounding.** All 5 tasks track Pythia scaling. GPT-2 reversal destroys family-independent interpretation.
5. **CONFIRMED: whitening bug in equicorrelation.** `src/cti_cross_modal_rho.py:89` multiplies by sqrt(Lambda) instead of dividing. The "tightest invariant" (rho~0.462, CV=1.0%) may be a geometric null artifact.
6. **The scoring rubric rewards prestige, not manifesto.** Zero economic reduction demonstrated.
7. **AMCL built a vault with nothing inside.** 108 research docs, 61 scripts, zero empirical results.

---

## Current Program

### Phase 0 — Kill Gates (1-2 weeks)

**Gate A: Equicorrelation Audit — KILLED (WL5)**
- Whitening bug confirmed: sqrt(Lambda) instead of 1/sqrt(Lambda)
- Historical rho~0.462 within matched-null 99% CI
- Claims retracted from both papers

**Gate B: Benchmark Bridge — STRATEGICALLY TERMINATED (R6 Steering)**
- Bridge v1 completed only pythia-160m (2 pass / 1 fail / 1 timeout / 1 error out of 5 tasks)
- Leaderboard tasks (BBH, IFEval, MATH Hard, MuSR, MMLU Pro) too hard for 160M-1B models
- Codex R6 steering: skip Bridge v2 entirely. q_knn and kappa are nearly the same object; bridge pass/fail cannot produce new science. Cross-family contradiction (GPT-2 reversal) already exists.
- Public-benchmark prediction claim permanently retired. 20 GPU-hours reallocated to GAT reserve.

**Gate C: AMCL — PERMANENTLY KILLED (Steering R3)**
- 504 files deleted. Zero empirical results ever existed.
- 20 GPU-hours reallocated to Geometry Admission Test reserve.

### Phase 1 — Geometry Admission Test (PERMANENTLY CLOSED — Jul 26, 2026)

**Thesis (KILLED):** A teacher's differential competence can be extracted once into a compact,
causally grounded behavioral artifact that installs into multiple incompatible students
without teacher access or pair-specific tuning.

**GAT Result: STRUCTURAL_SCREEN_FAIL (WL25)**
- Stage A: PASS (3/3 students, teacher 99.99% target-family)
- Stage B: FAIL — both candidates (raw R, observable R) showed all deltas negative
  - Raw: mean delta = -0.0065, sign_test_p = 1.0
  - Observable: mean delta = -0.014, sign_test_p = 1.0
  - Withheld acc 10-13% (chance = 8.3%)
- Geometry installs (aux loss converges) but does NOT transfer capability
- Stage C-I: PERMANENTLY CANCELLED
- Conditional GRU: PERMANENTLY CANCELLED
- Evidence preserved in `results/geometry_admission/stage_b/decision.json`

**Codex R11 diagnosis (5 failure mechanisms):**
1. R describes sample flow, not the transition law
2. Same anchors optimized repeatedly (memorization)
3. No compositional factorization constraint
4. Geometry lives in sidecar subspace (QL5 counterexample predicted this)
5. Normalization removes functional information

**Kill conclusion:** Observational geometric resemblance is not functional equivalence.
The strong form "Intelligence = Geometry" is dead.

### Phase 2 — Causal Skill Organs (CSO) — DONOR GATE FAIL (Jul 27, 2026)

**Donor capacity training: ALL GATES FAIL.** 50K steps (4.1h), precommit hash verified.
- Train: 33.9% (gate: >=99.5%) — FAIL by 65.6pt
- Length extrapolation: 0.4% (gate: >=99.0%) — FAIL by 98.6pt
- Excluded bigrams: 24.5% (gate: >=99.0%) — FAIL by 74.5pt
- Withheld trigrams: 5.9% (gate: >=99.0%) — FAIL by 93.1pt

**Failure mode:** Model learned per-register prediction (~76% per register, ~34% exact 4-register match) but never learned the compositional state-transition function. Length extrapolation near 0% throughout — no generalizable algorithm discovered. Cosine LR schedule drove learning rate to ~7.5e-10 by step 45K, eliminating any possibility of late grokking.

**QL11 strategic assessment (2-round Codex dialogue):** CSO caps at 3/10 even with perfect admission. Six systematic direction-selection antipatterns diagnosed. ACQ-1 (Actionable Causal Quotient) proposed as next direction (6-7/10 ceiling).

**CSO disposition:** Donor never achieved capacity. Remaining pipeline (host training, intervention generation, organ extraction, installation) never reached. Direction closed at first gate.

**Original thesis:**
> Capabilities are causal mechanisms with interfaces — not geometric shapes and not monolithic checkpoints.

**Headline:** "From giant models to transplantable causal skills. Train an expensive model
once, extract the mechanism it learned, and let cheap local AIs reuse it."

**What CSO IS NOT (learned from GAT kill):**
- Not representation matching (geometry failed)
- Not program transmission (trivial for random automata)
- Not Kolmogorov compression (not about shortest description)
- Not standard KD (which matches outputs, not mechanisms)

**What CSO IS:**
- Extract a learned causal state-transition mechanism from a donor model
- Validate it with interchange interventions (counterfactual fidelity)
- Freeze it as a standalone executable organ
- Transplant identical bytes into cross-architecture hosts
- Eventually compose independently extracted organs without joint retraining

**Prior art gap (Claude internet search Jul 26):**
| Property | NOT (2601.13580) | Circuit Dist. | CT-SFT | DAS | CSO (ours) |
|----------|-----------------|--------------|--------|-----|------------|
| Auto extraction | Partial | Yes | Partial | Yes | YES |
| Causal validation | No | No | No | Yes | YES |
| Standalone execution | Yes | No | No | No | YES |
| Cross-architecture | No | No | No | N/A | YES |
| Zero-shot composition | No | No | No | N/A | YES |

**Admission test: Causal Register Transducer**
- 4 registers in Z_16 (65,536 states), 8 invertible non-commuting operations
- Donor: 18.9M recurrent-state Transformer; Hosts: 0.9M Transformer (5.6% compute) + 1.3M GRU
- Organ: max 32-dim state, 14K params (59.7 KiB), frozen before host installation
- Withheld: lengths 13-32, excluded bigrams, held-out initial states, counterfactuals
- Budget: 40 GPU-hours, 4 calendar weeks
- Admission: >=95% exact withheld acc, >=90% counterfactual fidelity, >=15pt over observational baseline, identical bytes in both hosts

**Kill criteria:**
1. Observational bottleneck within 3pt of organ -> ordinary compression, not causal transfer
2. Output KD matches organ within 3pt -> same
3. Counterfactual fidelity <90% -> no causal content
4. Organ ablation doesn't selectively remove behavior -> sidecar
5. Wrong-donor organ works similarly -> not teacher-specific
6. Same bytes fail in either host -> not cross-architecture
7. Ground-truth latent states needed -> not automatic extraction
8. >40 GPU-hours without >70% counterfactual fidelity -> infeasible

**Pivot ladder:**
1. Admission fails: reassess causal extraction feasibility  **<-- WE ARE HERE. Donor never achieved capacity.**
2. Admission passes: earn the right to test world-model organ (novel goals)
3. World-model passes: earn the right to test zero-shot composition
4. Composition passes: moonshot narrative earned

### Phase 3 — Direction Selection (R12 Steering Dialogue)

**ACQ-1: KILLED by Codex R12 R1 (Jul 27).** Block-MDP literature preempts core claim.
Revised ceiling: 3-4/10 (down from 6-7/10). Six antipatterns apply.

**R12 2-round stress test killed ALL proposed alternatives:**
- CCL-1 (Capability Conservation Law): RETRACTED — rate-distortion + teaching dimension with new nouns
- PCI-1 (Proof-Carrying Pocket Intelligence): DEMOTED to 4/10 — test-time compute scaling occupies the frontier
- VIL-1 (Village Intelligence Phase Transition): ~5/10 max — beyond-union exists weakly (EMNLP 2025 DDS), novelty preempted
- Interaction necessity theorem (as stated): 3/10 — almost tautological from Bayes risk

**Sole surviving direction concept: Compute-Interaction Frontier (CIF)**
- **Question:** "When does asking the world beat thinking harder?"
- **Ceiling:** Conditional 7/10 (not yet earned)
- **Key object:** Adaptive advantage ratio gamma(eps) = min-cost-nonadaptive / min-cost-adaptive
- **Surprising result target:** "Two adaptive questions let a 3B offline model beat a 30B model with 100x test-time compute, at 25x lower total cost including user time, across three natural task families."
- **Cost accounting:** inference joules, latency, user seconds, answer bits, user error probability, connectivity, memory, question complexity
- **5-day theorem gate (7 criteria):** (1) same external-info budget, (2) unrestricted internal computation, (3) superconstant gap, (4) noisy user responses with heterogeneous costs, (5) independently motivated task class, (6) not obtainable from Fano/active-learning/etc. by substitution, (7) matching constructive policy

**Deep pattern (R12 failure synthesis): The Compactness-Consequence Trap.**
All 6 prior directions searched for a small, universal internal object after intelligence already exists. When easy to identify, tautological. When genuinely causal, difficult to learn and not cheaply portable. Next direction must start from direct scale-defying consequence and remain mechanism-agnostic.

**Full dialogue:** `results/codex_steering_r12.md` (R1), `results/codex_steering_r12_r2.md` (R2)

---

## Loop State

### Work Loop (supervisor check-in at iteration 10)
| WL | Task | Status | Verdict |
|----|------|--------|---------|
| 1 | Canonical whitening repair | DONE | Legacy parity confirmed, corrected mode works |
| 2 | Matched-null engine | DONE | KILL: historical 0.455-0.467 within null 99% CI |
| 3 | Corrected NLP rho audit | DONE | VOID: no cached embeddings, kill from WL2 stands |
| 4 | Cross-modal audit | SKIPPED | Same VOID: no cached embeddings |
| 5 | Claim adjudication | DONE | KILLED: retracted from both papers, success criteria, experiment log |
| 6 | Benchmark freeze + smoke | DONE | Pipeline works (BBH passed), GPQA gated |
| 7 | Benchmark panel execution | TERMINATED | 5/60 done (pythia-160m only). Task mismatch: leaderboard tasks too hard |
| 8 | Bridge adjudication | TERMINATED | Codex R6: skip Bridge v2, retire public benchmark claim |
| 9 | AMCL demolition | DONE | 504 files deleted, Gate C permanently killed |
| 10 | GAT implementation | DONE | 11/11 modules complete, preflight fixes applied |
| 11 | GAT 9-bug Codex audit fix | DONE | All 9 WL10 bugs fixed, CPU smoke passes |
| 12 | Truth surface cleanup | DONE | Equicorrelation retracted from README, MANIFEST, CHECKLIST, BRIEF, GUIDE |
| 13 | R6 Stage A protocol repairs | DONE | Pre-ridge rank, resume->restart, float32 round-trip, timing instrumentation, GRU deferral |
| 14 | Gate B closure + CLAUDE.md honest score | DONE | Bridge terminated, Nobel 8/10->2/10 in CLAUDE.md+MEMORY.md |
| 15 | Codex R7 preflight audit + T0 gate fix | DONE | T0 rank gate relaxed (19 < 48, now ≥8), float32 U_basis round-trip added |
| 16 | Stage A-T execution (train16) | VOID | Spec bug: trained 1-16, gated on 17-32. Teacher quarantined as diagnostic |
| 17 | R7 spec fix + Stage A-T relaunch | DONE | max_length=32, steps=7000->15000, anchors 8-32. Teacher warm restart from step 7000. |
| 18 | CM-CKS implementation (R8 design) | DONE | 5 files: automaton, statistics, installer, stage_b, stage_c. All import clean |
| 19 | Codex CM-CKS code review bug fixes | DONE | 8/12 bugs fixed: edge_index crash, replay noise gate, teacher capacity, finiteness, n=8, aggregate floors, cal hashes, seed averaging. 4 deferred (wrapped seqs, empty-list, hash chain, exception cleanup). |
| 20 | Extended training relaunch | DONE | MAX_STEPS 7000->15000. Teacher completed 15000 steps. |
| 21 | Thermal fixes + training efficiency | DONE | Codex review: thermal throttle (90C/86C), checkpoint every 1000 steps, stress_long every 2500, cooldown 60s between runs, checkpoint object freed after resume. |
| 22 | Stage A-T results + R9 gate redesign | DONE | Teacher in_range=1.0, extrap=0.618 (diagnostic), target_family=99.99%. R9 steering: replace 99% random-long gate with 95-99.5% target-family gate (p,r in [0,7]). Teacher now PASSES all R9 gates. Student 1 training. |
| 23 | Stage A capacity gate adjudication | DONE | All 3 students pass R9: S1=95.7%/98.8%, S2=96.5%/99.3%, S3=97.1%/99.4% (in_range/target_family). 3/3 pass, stage_a_pass=true. Anchor range (8,32)->(8,20) per R9. |
| 24 | Codex pre-extraction review + fixes + extraction | DONE | 12 findings (0 CRITICAL), fixed 4 (float32, competence gate, provenance, gate key). Extraction: 32 banks, all numerical PASS, repeat bit-identical. Competence: 2048/2048 + 8192/8192. Stage A COMPLETE. |
| 25 | Stage B structural screen | DONE | STRUCTURAL_SCREEN_FAIL. 12 runs completed. Both candidates: all deltas negative (correct WORSE than Haar). Raw mean=-0.0065, obs mean=-0.014. Withheld acc 10-13% (chance=8.3%). Geometry installs but does not transfer. Clean kill. |
| 26 | R11 steering: post-kill pivot | DONE | 3 rounds converged. Kill "Intelligence = Geometry" strong form. New direction: Causal Skill Organs (CSO). Register transducer admission test. Prior art gap confirmed. |
| 27 | CSO simulator + model smoke | DONE | Register transducer verified (8 ops, 65536 states, 23/28 non-commuting, group >201K). Models: donor 18.9M, host T 0.9M (5.6% compute), host G 1.3M, organ 54.7 KiB. Codex NO-GO -> 13 bugs fixed (5 CRITICAL, 4 HIGH, 4 MEDIUM). Precommit frozen. All verification PASS. |
| 28 | CSO precommit hardening (v3-v12 review cycle) | DONE | 12-iteration Codex review cycle. v3-v11 NO-GO: NaN/tolerance, semantic gaps, thread-sensitive sockets, derived-set bypass, boolean type confusion, runtime env, training config binding, list recursion. All fixed. v12: **GO**. Integrity hash: 745f10e7... |
| 29 | Donor capacity training (50K steps) | DONE | ALL GATES FAIL. 50K steps, 4.1h. Train 33.9%, extrap 0.4%, excluded 24.5%, withheld 5.9%. Model learned per-register tracking (~76%/reg) but never composed across registers. No grokking. Cosine LR drove to ~7.5e-10 by step 45K. |
| 30 | CSO closure + R12 direction stress test | DONE | Results committed, STATUS updated. ACQ-1 killed (R12 R1), CCL-1/PCI-1/VIL-1 killed (R12 R2). CIF survives. 5-day theorem gate next. |
| 31 | CIF 5-day theorem gate | PENDING | State theorem satisfying all 7 R12 R2 criteria. Day 0 = today. Kill if gate fails by day 5. |

### Question Loop (supervisor check-in at iteration 8)
| QL | Question | Status | Finding |
|----|----------|--------|---------|
| 1 | Shared-anchor null | DONE | Kill scalar rho: E[cos]->1/2 |
| 2 | Geometry identifiability | DONE | Terminal geometry cannot prescribe architecture |
| 3 | Minimal dynamic object | DONE | skew(A_r) is minimal novel signal |
| 4 | Strongest distillation baseline | DONE | Pareto set: FDD, MTA, TSD-KD, RG-OPD, LoRi, Procrustes/Gram |
| 5 | A_r counterexamples | DONE | 3 families: cosmetic sidecar, different mechanism, teacher-shaped regularizer |
| 6 | Regularization vs transfer | DONE | 4 tests required: teacher identity, causal use, timing, matched regularizers |
| 7 | Evidence ladder to "portable program" | DONE | 5-rung ladder; Qwen pilot can reach Rung 2 max |
| 8 | Generative directions for DG-0 | DONE | Lead with observable connection codec; pursue differential skill patches, closed-loop transfer, and gauge-invariant loop structure |
| 9 | Memorization threat to GAT | DONE | Artifact 290,000x over-budgeted for 86.5 bits. Paired-key signed selectivity is decisive experiment. QL6 tests only partially implemented |
| 10 | Calibration-matched counterfactual | DONE | Wrong-key control fatally unmatched on calibration edges. CM-CKS design required. |
| 11 | CSO ceiling + Nobel path assessment | DONE | CSO caps at 3/10 even if perfect. Six direction-selection antipatterns identified. ACQ-1 proposed (6-7/10 ceiling). Codex QL11 R2: "What did the large model discover that the researchers did not already know? Nothing scientifically new." |

### Steering Dialogue (5-round supervisor check-in)
| Round | Topic | Status | Outcome |
|-------|-------|--------|---------|
| R1 | Pivot from DG-0 to Open Capability File | DONE | Kill DG-0, pivot to capability file thesis |
| R2 | Corrected plan: budget, scope, honest scores | DONE | 2-candidate GAT over 7-codec tournament, 3/10 today |
| R3 | Concrete Stage A specification | DONE | Full automaton, architecture, training, extraction spec |
| R4 | Stage B/C transfer design + implementation | DONE | Complete spec: 18-run screen + 144-run confirmation |
| R5 | Final consistency audit | DONE | Convergence confirmed, blockers fixed |
| R6 | Post-WL10 steering: bridge, GAT blockers, candidates | DONE | Kill bridge v2, fix 4 Stage A blockers, keep both candidates, defer GRU |
| R7 | Training length spec bug | DONE | Fix applied: max_length=32, MAX_STEPS=7000, anchors (8,32), config hash fixed, old run quarantined |
| R8 | CM-CKS design | DONE | 2 rounds converged. 0 new modules, 5 files modified. 8 sealed pairs, 7/8 threshold |
| R9 | Gate redesign: target-family capacity | DONE | 3 rounds converged. Replace 99% random-long gate with target-family (p,r in [0,7]). Teacher: 99.99% target_family (PASS). Anchors 8-20. |
| R10 | Stage B redesign | DONE | 3 rounds converged. Haar control (not deranged teacher), fresh init, target-only coefficient, effect-size screen. 5 binding amendments: PCG64DXSM, Helmert basis, frozen coefficients, diagnostic-only probe, VOID verdict. |
| R11 | Post-kill pivot: Causal Skill Organs | DONE | 3 rounds converged. Kill "Intelligence = Geometry" strong form. New direction: Causal Skill Organs (CSO). Register transducer admission test. Prior art gap confirmed (5-way conjunction novel). |
| R12 | Post-CSO failure: direction stress test | DONE | 2 rounds. ACQ-1 killed (3-4/10). CCL-1 retracted, PCI-1 demoted (4/10), VIL-1 weakened (5/10). Surviving: Compute-Interaction Frontier (conditional 7/10). 5-day theorem gate. |

### Dead Code Cleanup Log
| Direction | Files Deleted | Date | Evidence Preserved |
|-----------|--------------|------|-------------------|
| GAT (Phase 1) | 11 files: `src/cti_geometry_admission_*.py` | Jul 27 | `results/geometry_admission/stage_b/decision.json` |
| CSO (Phase 2) | 3 files: `src/cti_causal_organ_*.py`, `src/cti_causal_register_transducer.py` | Jul 27 | `results/causal_organ/donor_capacity_result.json` |
