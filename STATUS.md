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

**Gate B: Benchmark Bridge — RUNNING**
- lm-eval executing: 12 models x 5 tasks (excl GPQA)
- 2/60 complete (pythia-160m BBH pass, instruction_following fail)
- Survival requires rho>=0.60, LOFO>=0.35, partial after log-params>=0.30

**Gate C: AMCL — PERMANENTLY KILLED (Steering R3)**
- 504 files deleted. Zero empirical results ever existed.
- 20 GPU-hours reallocated to Geometry Admission Test reserve.

### Phase 1 — Open Capability File (primary moonshot, pivoted from DG-0)

**Thesis:** A teacher's differential competence can be extracted once into a compact,
causally grounded behavioral artifact that installs into multiple incompatible students
without teacher access or pair-specific tuning.

**Headline:** "Download a skill, not a giant AI."

**Geometry Admission Test (concrete spec in research/OPEN_CAPABILITY_FILE_GEOMETRY_ADMISSION_STAGE_A_2026_07_25.md):**
- 12-state, 4-symbol permutation automaton (115 bits key entropy)
- Teacher: 12L/384d/19.5M params Transformer
- Student: 6L/160d/1.9M Transformer (10.15x compression)
- Conditional: 6L/224d/1.8M GRU (10.55x compression)
- 2 candidates: raw R trace vs observable connection (VJP-balanced)
- 5 controls: no auxiliary, static G, wrong-key, Haar-matched, generic smoothness
- Staged: preflight (A) -> screen (B) -> sealed 8-key confirmation (C) -> conditional GRU

**Budget:**
| GPU-hours | Work |
|---:|---|
| 20 | Finish and adjudicate Gate B |
| 30 | Transformer Geometry Admission Test |
| 25 | Conditional GRU confirmation |
| 25 | Extra seeds, causal ablation, accounting, reserve |

**Honest scores (Steering R2):**
- Today: Turing 3/10
- After 100h best case (synthetic automaton passes): 4.5/10
- 8/10 requires: language competence + cross-substrate + composition + economics + replication

### Pivot Ladder (updated)

1. Bridge fails: close CTI universal-law project or reposition as training coordinate
2. GAT fails: kill "Intelligence = Geometry" strong form, consider Universal Compute Governor
3. GAT Transformer passes: earn the right to test cross-substrate (GRU)
4. GAT cross-substrate passes: earn the right to test language transfer

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
| 7 | Benchmark panel execution | RUNNING | 12 models x 5 tasks (excl GPQA), lm-eval in progress, 3/60 done |
| 8 | Bridge adjudication | BLOCKED | Depends on WL7 |
| 9 | AMCL demolition | DONE | 504 files deleted, Gate C permanently killed |
| 10 | GAT implementation | DONE | 11/11 modules complete, preflight fixes applied |

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

### Steering Dialogue (5-round supervisor check-in)
| Round | Topic | Status | Outcome |
|-------|-------|--------|---------|
| R1 | Pivot from DG-0 to Open Capability File | DONE | Kill DG-0, pivot to capability file thesis |
| R2 | Corrected plan: budget, scope, honest scores | DONE | 2-candidate GAT over 7-codec tournament, 3/10 today |
| R3 | Concrete Stage A specification | DONE | Full automaton, architecture, training, extraction spec |
| R4 | Stage B/C transfer design + implementation | DONE | Complete spec: 18-run screen + 144-run confirmation |
| R5 | Final consistency audit | DONE | Convergence confirmed, blockers fixed |

### GAT Implementation Status
| Module | File | Status |
|--------|------|--------|
| Automaton + data gen | `cti_geometry_admission_automaton.py` | DONE |
| Model architectures | `cti_geometry_admission_models.py` | DONE |
| Capacity trainer | `cti_geometry_admission_trainer.py` | DONE |
| Trace extraction | `cti_geometry_admission_extraction.py` | DONE |
| Geometry + losses | `cti_geometry_admission_geometry.py` | DONE |
| Installer (student) | `cti_geometry_admission_installer.py` | DONE |
| Statistics | `cti_geometry_admission_statistics.py` | DONE |
| Stage A orchestrator | `cti_geometry_admission_stage_a.py` | DONE |
| Stage B orchestrator | `cti_geometry_admission_stage_b.py` | DONE |
| Stage C orchestrator | `cti_geometry_admission_stage_c.py` | DONE |
| Verification | `cti_geometry_admission_verify.py` | DONE |
