## R7 ruling

Stop the current teacher run and classify it as an invalid-protocol diagnostic. Do not extract artifacts from it.

Adopt:

```text
TRAIN_LENGTH_RANGE = 1..32
MAX_STEPS = 7000
WARMUP_STEPS = 250
EVAL_INTERVAL = 250
ANCHOR_LENGTH_RANGE = 8..32
INSTALLER_MAX_STEPS = 5000
CALIBRATION_LENGTHS = 1, 2, 4, 8, 16
```

### 1. Use 32, not 64

`MAX_TRAIN_LENGTH=32` is correct.

The operational evaluations only extend through length 32. Training through 64 would:

- Turn the diagnostic set into training data.
- Approximately double sequence compute again.
- Reduce the fraction of training examples at lengths 1–16 from 50% to 25%.
- Still provide no genuine extrapolation test beyond position 64, because the model supports only 65 input positions.

The failure mechanism is explicit in the model: it adds a separately learned embedding for every absolute position and classifies from the final position ([models.py:100](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_models.py:100>), [models.py:126](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_models.py:126>)). Training through operation length 16 updates positions 0–16; the first extrapolation example places its answer-bearing token at untrained position 17.

### 2. Increase the ceiling to 7,000 steps

Do not keep 5,000.

The exposure calculation is:

```text
Current passing exposure:
2750 × 512 / 16 = 88,000 examples per length

Proposed at 5000:
5000 × 512 / 32 = 80,000 examples per length

Proposed at 7000:
7000 × 512 / 32 = 112,000 examples per length
```

Seven thousand therefore provides 27% more per-length exposure than the point where the present teacher first crossed 99.5%, while accommodating harder 17–32-step compositions.

Keep the optimizer, LR, 250-step warmup, and evaluation cadence unchanged. Require the final two evaluations—steps 6,750 and 7,000—to pass. If the teacher misses after 7,000, adjudicate it as an optimization/capacity failure; do not extend post hoc.

Because batches are effectively padded to the maximum sampled length, Transformer step time will be roughly 1.9–2.0× the current run. Combined with 7,000 versus 5,000 steps, expect approximately 2.7–2.8× the original per-run compute. Based on the present timing, the teacher should take roughly 70–90 minutes, still comfortably inside the Stage A reserve.

### 3. Installer: do not add labeled length-32 examples

There is no `max_length=16` at installer line 329; that line collates the fixed calibration examples ([installer.py:329](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:329>)).

The installer’s 64 labeled examples deliberately stop at length 16 ([automaton.py:277](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:277>)). Keep that restriction. Adding labeled length-32 examples would change the restricted-supervision intervention.

But another positional-support hole exists:

- Current anchors cover lengths 8–24.
- Withheld installer sequences reach length 31 because \(p,r\in[0,15]\).
- Fresh Transformer installers therefore receive no gradient at positions 25–31.

Change the shared anchor range from 8–24 to **8–32**, using a new anchor protocol ID. All correct, wrong-key, Haar, static-G, and smoothness arms will then receive identical positional support. Keep installer training at 5,000 steps and remeasure its timing projection.

### 4. Trace and protocol implications

The current anchors are 8–24, not 1–16 ([spec:268](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/OPEN_CAPABILITY_FILE_GEOMETRY_ADMISSION_STAGE_A_2026_07_25.md:268>)). Thus part of the intended trace—lengths 17–24—is currently extracted from chance-level behavior. Training through 32 should improve trace validity substantially.

Before restarting:

- Assign a new protocol ID and clean result namespace. The trainer appends to existing logs ([trainer.py:184](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:184>)).
- Put training and anchor length ranges into every config and configuration hash. They are currently absent ([trainer.py:102](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:102>)).
- Hash the actual permutation bytes. The current NumPy-key path hashes the placeholder `{"array":"omitted"}`, not the key.
- Restart every teacher and capacity student from step zero. No old checkpoint or artifact remains admissible.
- Propagate the 1–32 capacity distribution to Stage B/C teachers and the three Transformer capacity students.
- Recompute all anchor manifests, artifacts, numerical audits, and timing projections.

### 5. Rename the gate honestly

Once training covers 1–32, `dev_extrapolation` is no longer extrapolation. Rename it conceptually to:

```text
dev_core:          lengths 1–16
dev_covered_long:  lengths 17–32
stress_positional_ood: lengths 33–64
```

Keep the 99.5%/99.0% thresholds. They are appropriate capacity gates.

Do not use the Transformer’s 33–64 score as evidence about algorithmic learning: positions 33–64 remain untrained, so that diagnostic conflates composition with positional OOD.

For a separate algorithmic-learning claim, preregister:

- Exact train/eval de-duplication and accuracy by individual length, with a ≥98.5% worst-length floor.
- A frozen prefix/suffix composition-consistency suite, e.g. 20,000 triples with ≥99% consistency.
- True 33–64 extrapolation using a position-generalizing architecture such as relative/RoPE attention, alongside the GRU.

That should be a separate experiment, not a Stage A launch blocker. Stage A’s job is to establish competent teachers and students over the complete operational support required by extraction and installation.

