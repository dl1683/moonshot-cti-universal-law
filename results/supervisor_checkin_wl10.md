Do not launch GAT Stage A in its current state. The design is coherent, but the implementation is not execution-ready or evidence-safe. I score it **4/10**.

I audited the current HEAD, `4df07e8`, including the two “repair” commits after `b411bbd`. All modules compile and parameter counts are correct, but direct CPU invariants exposed hard blockers.

## A) Work while the GPU is blocked

Priority order:

1. **Repair GAT and build a real CPU test harness.** This is the dominant task. Without it, Stage A will consume hours training seven models and then crash during extraction.
2. **Implement provenance and information-boundary enforcement.** Freeze run identities, artifact hashes, coefficients, data hashes, code hash, and restart behavior before generating sealed keys.
3. **Run a non-scientific end-to-end micro-smoke on CPU.** It must traverse automaton generation → model → extraction → auxiliary loss → statistics → verifier. Imports are insufficient.
4. **Clean the repository’s truth surface.**
   - Current `README.md` contains no AMCL text; stale AMCL instructions are in [CLAUDE.md:141](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/CLAUDE.md:141>).
   - The more serious README error is that killed equicorrelation remains marked `PASS` at [README.md:54](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/README.md:54>).
   - [CTI_MANIFEST.md:18](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/CTI_MANIFEST.md:18>) still canonizes retracted equicorrelation/alpha-rho artifacts and references two nonexistent anonymous-paper files.
   - `CTI_FIRST_PRINCIPLES_GUIDE`, `CLAIM_EVIDENCE_CHECKLIST`, and `CODEX_NOBEL_REVIEW_BRIEF` still present killed rho claims as valid.
   - Apply the seven R5 errata to the Stage A specification, including its false “file was not created” footer.
5. **Paper work: truth revision only.** Do not polish a CTI success narrative before the bridge verdict. The LaTeX already contains the equicorrelation retraction; synchronize front-door documents now, then revise abstract/conclusion after bridge adjudication.
6. **No further theoretical derivation.** The bottleneck is executable validity and evidence integrity, not another design round.

## B) Priority after the GPU frees

**Adjudicate the bridge first, then launch repaired Stage A.**

By definition, the bridge will already have completed when the GPU becomes free. Its adjudication is CPU work and should take far less time than Stage A. There is no rational reason to begin another GPU job before freezing the bridge verdict.

Exact order:

1. Verify all 60 bridge cells and freeze the raw manifest.
2. Run the preregistered total-panel, LOFO, and partial-correlation adjudication.
3. Record PASS/FAIL without reinterpretation.
4. Confirm the repaired GAT CPU harness and provenance gate pass.
5. Launch Stage A immediately afterward.

If the bridge fails `rho < 0.60`, **GAT remains the highest-priority experiment**. The failure kills or sharply demotes CTI’s external-capability narrative; it does not falsify the Open Capability File hypothesis. Strategically, GAT becomes more important because it is the only live moonshot—but it must be explicitly separated from CTI and cannot inherit CTI evidence or branding.

## C) GAT implementation assessment

The architecture is thoughtful; the evidence machinery is not trustworthy yet.

Critical findings:

- **Extraction cannot currently start.** Anchors deliberately have no labels, but [collate_fn](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:170>) unconditionally reads `b["label"]`. Every extraction and auxiliary-loss path sends unlabeled anchors through it. CPU validation produced `KeyError: 'label'`.
- **The observable VJP is broken.** [extraction.py:213](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_extraction.py:213>) computes logits first, then creates final-token views and asks for gradients with respect to those new views. They are not graph ancestors. The same pattern produced PyTorch’s “Tensor appears to not have been used in the graph” error.
- **Withheld labels enter the student training process.** [train_installer_run](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:286>) receives labeled withheld examples and evaluates them every 250 steps at [installer.py:390](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:390>). This directly contradicts the declared forbidden-channel boundary, even without early stopping.
- **Coefficient freezing is not implemented.** Stage B recalibrates coefficients separately per key at [stage_b.py:228](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:228>). Stage C recalibrates them per sealed key, seed, and arm at [stage_c.py:380](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:380>). The protocol requires one coefficient per arm frozen on the Stage A reference condition before Stage B outcomes.
- **The claimed identity-checked resume is ineffective.** The trainer overwrites `config.sha256` before reading the “existing” value, so it compares the new hash with itself. The key hash also omits every NumPy key. Mid-run resume is prohibited anyway, and the resumed data loader restarts its stream from batch zero.
- **Stage C is not precommitted.** It writes a minimal config and then generates keys and immediately trains. There is no committed code hash, artifact hash, coefficient table, data hash, control hash, analysis hash, or pause for publication before execution.
- **Numerical protocol mismatches remain.** The balanced ridge is divided by 64 twice at [extraction.py:266](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_extraction.py:266>); empirical ranks are measured after adding the ridge, making rank gates effectively vacuous; NumPy extraction silently promotes the declared float32 path to float64; Haar generation uses `default_rng` plus an SVD basis rather than frozen Helmert + `PCG64DXSM`.
- **Teacher and launch gates are incomplete.** Stage B ignores teacher capacity and numerical gates. Stage C omits teacher extrapolation and consecutive-evaluation checks. Stage A’s launch gate omits the ≤30 GPU-hour projection entirely.
- **The verifier cannot certify the claim.** Forbidden-channel verification only scans filenames/JSON keys; `no_forbidden_info` is hardcoded `True`. “Anchor coverage” checks only 2048/32/64 shape, not per-key edge counts. Bootstrap-LCB agreement is calculated but omitted from the verifier’s pass expression.

What is good: the automaton construction, calibration/withheld split, model dimensions, parameter counts, six-transition clock, arm matrix, Stage B selection equations, and Stage C statistics are mostly faithful. That earns the implementation its 4/10 rather than a lower score.

But the decisive answer is: **no GAT result produced by the current pipeline should be accepted as evidence.**

## D) Narrative test

The honest gossip-magazine sentence is:

> “Researchers found that AI geometry mostly predicts the label geometry it was built from, and have now built—but not yet run—a test of whether that geometry can be bottled into a skill for a model ten times smaller.”

- **“That is obvious” test:** The existing CTI half fails; predictor and target are nearly the same object. The capability-file question itself is non-obvious, but remains only a question.
- **“So what?” test:** Fails today. There is no demonstrated transferred capability, compute saving, language competence, or economic consequence.
- **Mission test:** Fails. A random person would say, “Interesting—show me that it works,” not “that changes everything.”

Even a Stage A PASS will not change this narrative: Stage A proves only capacity and extractability. The first potentially consequential result is a controlled Stage C PASS, followed by cross-substrate GRU confirmation.

