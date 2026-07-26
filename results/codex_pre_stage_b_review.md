**Verdict: NO-GO.** Stage B should not be launched. It will currently stop before training, and manually bypassing that would run a different experiment from the 12-run paired screen you described.

1. **CRITICAL — Three incompatible Stage B contracts exist.**

   - Your launch contract: two same-key teachers with seeds 102/103; three students; raw/observable × same-key/wrong-key = 12 runs.
   - Current orchestrator: two transposed-key teachers plus one same-key replay teacher, all teacher seed 101; eight crossover installations plus four replay installations.
   - R4: two independent development keys and 18 arms/runs, with seed 401 ([R4 spec](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/STEERING_DIALOGUE_R4_ROUND4_STAGE_BC_2026_07_26.md:509>)).

   The current total happens to equal 12, but its factorial structure is entirely different ([stage_b.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:284>)).

   **Required fix:** designate the prompt’s 12-run design as canonical, mark R4’s Stage B section superseded, and rewrite the orchestrator/run manifest accordingly.

2. **CRITICAL — Stage B immediately exits because a required Stage A file is absent.**

   Stage B requires `results/geometry_admission/stage_a/development_key.json` ([stage_b.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:145>)). The file does not exist, although the Stage A specification declares it mandatory ([Stage A spec](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/OPEN_CAPABILITY_FILE_GEOMETRY_ADMISSION_STAGE_A_2026_07_25.md:522>)).

   **Required fix:** write the canonical development key artifact atomically, then verify its canonical JSON hash against `anchor_manifest.json`. Alternatively, load the hard-coded key but still perform the same hash check.

3. **CRITICAL — The capacity gate is stale and rejects the completed R9 teacher.**

   Stage B still requires `final_extrapolation >= 0.990` ([stage_b.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:162>)). The qualified teacher has 0.6184 on that now-diagnostic endpoint and 0.99989 on the replacement `target_family` gate. Thus, after repairing the missing key, Stage B would print “below capacity gates” and exit. Partner-teacher qualification repeats the same obsolete check.

   **Required fix:** consume `capacity_summary.json` and the R9 conditions: teacher `target_family`, in-range, 48 edges, and two-evaluation stability. Apply the same R9 gate to every Stage B teacher.

4. **CRITICAL — The three qualified Stage A students are not connected to Stage B.**

   The orchestrator uses only one student seed, 401 ([stage_b.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:291>)). The installer reseeds and constructs a fresh Transformer for every run; it has no initial-checkpoint input ([installer.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:338>)). None of the `transformer_s1/s2/s3/model_final.pt` checkpoints is read.

   **Required fix:** if the three R9-qualified students are the paired units, add a hash-bound initial checkpoint to the run identity and loop over those exact three checkpoints. Same-key and wrong-key runs for a student/candidate must begin from byte-identical state.

5. **CRITICAL — The wrong-key control and accuracy endpoint are not implemented.**

   Current A/B runs change the artifact, calibration data, target key, and evaluation key together ([stage_b.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:311>)). That is not a wrong-key control where only the installed trace changes.

   Moreover, Stage B passes `withheld_examples=[]`. The installer therefore records `final_withheld_acc=0.0` for every unsealed run ([installer.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:481>)). Stage B instead compares 48 direct-edge logits; it never compares same-key versus wrong-key accuracy.

   **Required fix:** freeze one target calibration/evaluation set per student pair, hold initialization, labels, bank order, optimizer, coefficient, and evaluation examples fixed, and vary only the artifact. Populate and hash the withheld evaluation set.

6. **HIGH — Stage A’s provenance-bound trace bytes are bypassed.**

   Stage B re-extracts the base teacher in memory instead of loading and verifying the completed Stage A raw/observable artifacts. Partner traces are also kept only in memory, without artifact files, hashes, manifests, or numerical audits ([stage_b.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_b.py:236>)). Teacher training, extraction, and installation also occur in one process, contrary to the specified boundary ([R4 process contract](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/STEERING_DIALOGUE_R4_ROUND4_STAGE_BC_2026_07_26.md:883>)).

   **Required fix:** serialize each trace, bind checkpoint/config/key/anchor/code hashes, validate all Stage A manifests and numerical gates, and launch installation from artifact files in a teacher-free phase.

7. **CRITICAL — There is no valid paired CM-CKS accuracy test.**

   Stage B never calls a paired accuracy statistic. `stage_b_selection()` implements the obsolete 18-run deterministic screen. `cm_exact_sign_test()` instead treats a thresholded conjunction called “pair success” as a Bernoulli event with null probability 0.5 ([statistics.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_statistics.py:373>)). That null is not justified for a conjunction of crossover, effect-size, and stability gates.

   Its generic threshold is also inconsistent with its p-value: with three successes from three students it returns `p=0.125` but `pass=True`.

   **Required fix:** define paired differences
   \[
   \Delta_s=A_{s,\mathrm{same}}-A_{s,\mathrm{wrong}}
   \]
   and preregister a one-sided paired null such as \(H_0:\operatorname{median}(\Delta_s)\le0\), or the corresponding within-pair exchangeability null. With only three independent students, an exact one-sided sign/randomization test cannot reach 0.05—the smallest p-value is 0.125. Therefore Stage B must either be explicitly an effect-size screen without a significance claim or use more independent student seeds. Do not treat 4,000 deterministic examples as 4,000 scientific replicates.

8. **HIGH — Statistical verification fails open.**

   `cm_cks_verdict()` defaults missing protocol checks to `True` ([statistics.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_statistics.py:399>)). I confirmed that eight synthetic successes plus an empty `protocol_checks={}` produce a PASS.

   **Required fix:** require every protocol field explicitly and default missing checks to failure/VOID. Recompute run completeness, artifact identity, calibration hashes, and finite endpoints from raw records.

9. **HIGH — Installer restart and numerical contracts remain unsafe.**

   Completed runs are skipped using only directory name and `status=="complete"` ([installer.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:327>)); changed artifacts or configuration can silently reuse stale results. The differentiable geometry also clamps every negative eigenvalue to zero instead of failing below \(-10^{-6}\) ([geometry.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_geometry.py:42>)), and coefficient calibration silently returns 1.0 when auxiliary gradients are absent ([installer.py](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:149>)).

   **Required fix:** exact run identities incorporating checkpoint/artifact/data/config/code hashes; atomic summaries; clean restart logs; explicit nonfinite/eigendecomposition checks; reject materially negative eigenvalues and zero/nonfinite calibration gradients.

The reusable core is viable: all three modules compile, the actual Stage A raw and observable artifacts load successfully, and both auxiliary losses produced finite losses and finite gradients in a CPU smoke test using `transformer_s1`. The blockers are orchestration, experimental identity, endpoint construction, and statistics—not the basic tensor compatibility.

No files were changed.

