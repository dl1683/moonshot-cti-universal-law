## BLOCK

Stage A‑T cannot pass as written.

- **GPU execution:** CUDA train-step, extraction, and serialization smoke tests passed. Orchestration correctly selects teacher + three Transformer students, excluding GRUs.
- **Fatal gate:** T0 is final-token `token_embedding + position_embedding`. With four operation tokens and 17 positions, centered rank is at most \(3+16=19\), but every raw transition requires rank ≥48. The GPU smoke measured exactly **19** for transition 0. [models](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_models.py:126), [gate](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_extraction.py:329)
- **Wc/Wo:** ranks are correctly measured before centered ridge. [extraction](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_extraction.py:275)
- **Resume:** checkpoints are deleted and never loaded, but a completed `summary.json` causes reuse rather than restart. Strict “every invocation starts at step 0” is therefore not fully enforced. [trainer](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:154)
- **Float32:** partial. Round-trip equality checks cover `R` and `R_obs`, but not serialized `Omega` or `U_basis`; NumPy centering also upcasts matrix computation to float64. [serialization](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_extraction.py:405)
- **Timings:** partially instrumented—bank totals and forward/VJP/perturbation/eigendecomposition phases exist, but raw-forward throughput and artifact-byte metrics are absent.
- **Estimated Stage A‑T cost:** roughly **0.5–1.0 GPU-hours** on an exclusive RTX 5090 Laptop GPU; likely lower on a desktop 5090. At audit time the live GPU was already 100% occupied, so direct timing calibration was contaminated.

Primary launch fix: revise the T0 representation or exclude transition 0 from the ≥48 raw-rank gate.

