The code does not show a classical per-step memory leak. Loss graphs are released after backward, evaluations use `no_grad`, DataLoader queues cannot accumulate with `num_workers=0`, and extraction retains only one bank at a time.

The shutdowns look more like thermal or power instability than OOM. Windows recorded unexpected shutdowns at 4:10 and 4:42 AM on July 26, with no corresponding Python/CUDA crash, bugcheck, or display-driver reset in the queried logs. Temperature was not recorded, so thermal shutdown is plausible—not proven.

## Ranked findings

1. High — no thermal guard or duty-cycle control

Files: [trainer.py:212](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:212>), [stage_a.py:64](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_a.py:64>), [stage_a.py:112](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_a.py:112>)

Problem: Training runs continuously through 15,000 steps, four large evaluations every 250 steps, then immediately starts the next model. Extraction similarly processes 32 banks twice without cooldown. Nothing monitors temperature, power, clock throttling, or fan failure.

Fix: Add configurable temperature-aware throttling using NVML, with a stop/sleep threshold and hysteresis—for example pause above 82–85°C and resume below 75–78°C. Also add cooldowns between runs and banks. If sensor integration is undesirable, add a configurable periodic duty-cycle sleep and cap GPU power/clock externally. Smaller batches alone may reduce peaks but do not guarantee lower average thermal load.

2. High — evaluation adds substantial unnecessary thermal load

Files: [trainer.py:240](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:240>), [automaton.py:192](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:192>)

Problem: Every 250 steps evaluates 60,048 examples. Across 60 evaluations, that is 3,602,880 predictions per run. The 20,000-example `stress_long` set is explicitly “not gated,” yet runs every time; its length-64 attention is the most expensive evaluation slice.

Fix: Evaluate gated in-range/extrapolation sets every 250 steps, but run `stress_long` only at the end or every 1,000–2,500 steps. Sort/bucket evaluation examples by length to reduce padding. Use `torch.inference_mode()` rather than `no_grad()` for slightly lower inference overhead.

3. High — 14 GB of teacher checkpoint writes per run

Files: [trainer.py:240](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:240>), [trainer.py:272](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:272>), [installer.py:412](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:412>)

Problem: The live teacher checkpoint is 234,284,481 bytes. Saving it 60 times writes approximately 14.1 GB, even though the same path is overwritten. Because the repository is inside OneDrive, each rewrite may also trigger sync/scanning activity. Saving directly to the final path is non-atomic, so a shutdown during serialization can corrupt the only resume checkpoint.

Student Stage A writes are about 1.39 GB per run; each 5,000-step installer run writes about 463 MB.

Fix:

- Save full model+optimizer checkpoints every 1,000–2,000 steps, not every evaluation.
- Write checkpoints to a non-synced local scratch directory.
- Save to a temporary file, flush, then atomically replace the previous checkpoint.
- Keep lightweight evaluation logs at 250-step resolution.
- Optionally save model-only snapshots frequently and optimizer state less often.

4. High — checkpoints are written but students cannot resume

Files: [stage_a.py:68](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_a.py:68>), [trainer.py:182](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:182>), [installer.py:353](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:353>)

Problem: Only the teacher receives `allow_resume=True`. Transformer student checkpoints are deleted on restart. Installer checkpoints are also always deleted and never loaded. Thus most checkpoint I/O provides no crash recovery, and every shutdown repeats the entire current student run and its thermal load.

Fix: Choose one coherent policy:

- Support deterministic resume for all runs, including model, optimizer, scaler, step, configuration identity, and dataset-stream position; or
- If frozen-contract restarts are mandatory, stop writing full optimizer checkpoints and write only final/model diagnostic snapshots.

5. Medium — manual attention and padding waste considerable compute

Files: [models.py:51](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_models.py:51>), [automaton.py:221](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:221>), [automaton.py:233](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:233>)

Problem: Attention explicitly materializes the full `B × heads × T × T` score and probability tensors. Training lengths are uniform from 1–32, but a 512-example batch almost always pads to 33 tokens. Average real input length is only 17.5, so roughly 47% of token/FFN work is padding; padded query positions still pass through every block.

Fix: Replace manual attention with `torch.nn.functional.scaled_dot_product_attention`, allowing PyTorch to select fused/Flash kernels. Generate length-homogeneous or bucketed batches. Evaluation can be sorted by length without changing training semantics.

6. Medium — installer performs its auxiliary model forward entirely in FP32

Files: [installer.py:176](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:176>), [installer.py:388](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:388>)

Problem: `autocast(enabled=False)` surrounds all of `compute_auxiliary_loss`, including the student forward. The code therefore performs a second full forward in FP32 every auxiliary step. Only the differentiable geometry calculations require FP32; `compute_student_R_sequence` already casts selected hidden vectors to float.

Fix: Run the auxiliary model forward under BF16 autocast, then execute the Gram/eigendecomposition geometry operations in an autocast-disabled FP32 block.

7. Medium — teacher targets are reconstructed and transferred every installer step

Files: [installer.py:187](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:187>), [installer.py:199](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:199>), [installer.py:212](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:212>)

Problem: Each step creates six or more new `torch.tensor(..., device="cuda")` objects from NumPy arrays. They are eventually freed, so this is not a leak, but it creates thousands of allocations and CPU-to-GPU transfers per run.

Fix: Convert artifacts to tensors once before training. All 32 raw banks occupy only about 3 MB, so caching the complete target set on a 24 GB GPU is inexpensive. Alternatively cache the currently selected bank.

8. Medium — resumed teacher retains the loaded checkpoint object

File: [trainer.py:184](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:184>)

Problem: `torch.load(..., map_location=device)` creates the full checkpoint on GPU, and `ckpt` remains referenced for the rest of `train_one_run`. Model weights are copied into the model, so at least the checkpoint’s model tensors can remain as redundant GPU storage. This is bounded—not a growing leak—but avoidable.

Fix: Load on CPU, restore model/optimizer/scaler, then explicitly `del ckpt`. Verify optimizer state has moved to the parameter device. Run `gc.collect()` and, if needed at this boundary, `torch.cuda.empty_cache()`.

9. Medium — resume restarts the infinite data stream from sample zero

Files: [trainer.py:197](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:197>), [automaton.py:213](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:213>)

Problem: `AutomatonTrainDataset.__iter__` reconstructs the RNG from the original seed. A teacher resumed at step N therefore uses the step-zero training stream again. The checkpoint also does not contain or validate its configuration hash.

Fix: Make generation counter-based from a global sample index, or checkpoint and restore the dataset RNG/sample offset. Store `config_hash` inside the checkpoint and reject incompatible resumes.

10. Low — files and model state are not exception-safe

Files: [trainer.py:203](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:203>), [installer.py:361](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:361>), [extraction.py:190](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_extraction.py:190>)

Problem: Log files are manually closed only after successful completion. An exception can leave them open while a traceback retains the frame. Extraction also mutates every parameter’s `requires_grad` state without a `finally` block.

Fix: Use `with`/`ExitStack` for logs and `try/finally` for parameter flags, model mode, and explicit model/optimizer teardown. A hard system power-off closes handles at the OS level, so this is not the likely shutdown cause.

11. Low — repeat extraction doubles heat and disk work

File: [stage_a.py:147](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_a.py:147>)

Problem: After extracting and serializing all 32 banks, Stage A immediately repeats the entire extraction. This is intentional determinism verification, not a leak, but doubles extraction compute and writes a duplicate artifact tree.

Fix: If the protocol permits, repeat a deterministic sample of banks or run the repeat pass in a separate cooled process. Otherwise insert a cooldown between passes.

## Confirmed non-leaks

- `loss.item()` is used correctly. Loss tensors are not appended to histories.
- `evaluate()` is correctly decorated with `torch.no_grad()`.
- `eval_history` reaches only 60 small dictionaries; it is negligible.
- `step_times` contains 15,000 Python floats, also negligible.
- The generated evaluation corpus occupied about 45.5 MB of CPU memory in a live measurement. It is fixed-size.
- `AutomatonTrainDataset` is intentionally infinite, so `next(loader_iter)` does not exhaust it. With `num_workers=0`, there is no worker or prefetch-queue accumulation.
- Extraction does not retain traces for all 32 banks. Only hashes, timings, and small numerical summaries accumulate.
- CUDA allocator reservation between sequential runs is not itself a leak; freed blocks remain cached for reuse. Explicit teardown can make boundaries clearer but `empty_cache()` should not be treated as a leak fix.
- `evaluate()` returning the model to training mode is correct for its current call site, although it should preserve and restore `model.training` for safe reuse.
- BF16 does not require gradient scaling; the current `GradScaler` is unnecessary but not materially destabilizing.

The most valuable stabilization sequence is: add a thermal throttle, stop evaluating `stress_long` every 250 steps, reduce/move checkpoint writes out of OneDrive, and make student checkpoints genuinely resumable. No source files were changed; all six requested modules compile successfully.

