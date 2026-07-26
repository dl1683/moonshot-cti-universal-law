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

**Gate A: Corrected Equicorrelation Audit**
- Fix whitening: Sigma_W^(-1/2) with Ledoit-Wolf shrinkage
- Run 6 matched null controls
- Survival requires all 4 modality groups outside 99% null interval after Holm correction

**Gate B: Benchmark Bridge**
- Run preregistered 6-task Open LLM Leaderboard panel via lm-eval
- Survival requires rho>=0.60, LOFO>=0.35, partial after log-params>=0.30, beats baselines by 0.10+

**Gate C: AMCL Demolition**
- Paired GSM8K (1000 problems), stop vs revise, Qwen3-0.6B + OLMo-2-1B
- CTI must beat confidence/entropy/margin by 1+ point with positive 95% LCB in both transfer directions
- If fails: kill AMCL permanently

### Phase 1 — Dynamic Geometry Compiler (primary moonshot)

**Thesis:** Intelligence is not weights; it is a compact geometric program of representation transformations realizable on different substrates.

**Headline:** "Copy how an AI thinks, not its weights."

**DG-0 Minimal Experiment:**
- Teacher: Qwen3-4B -> Student: Qwen3-0.6B (6.7x compression)
- Task: GSM8K (3000 train, 500 dev, official test frozen)
- 6 arms: SFT / SFT+KD / +static Gram / +dynamic update-to-state / +permuted control / +compute-matched
- Direction alive if D beats max(C,F) by 3+ points, 20%+ gap closed, reproduces on second benchmark

### Pivot Ladder

1. Bridge fails, equicorrelation survives: reposition CTI as training coordinate
2. Bridge AND equicorrelation fail: close CTI universal-law project
3. Dynamic geometry fails to beat KD: kill "Intelligence = Geometry" strong form
4. Dynamic geometry wins at 10x compression across architectures: manifesto has evidence

---

## Loop State

- Work loop iteration: 0
- Question loop iteration: 0
- Next supervisor check-in: WL after iteration 10, QL after iteration 7
