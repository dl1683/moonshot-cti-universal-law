The decision: stop the current student sweep, record the present protocol as a valid Stage A capacity failure, and redesign the admission test around the actual transfer distribution. Do not spend another cycle chasing 99%, and do not lower the current gate to 60–70%.

### The central finding

The reported “extrapolation” result is not extrapolation under the current code:

- Training samples lengths 1–32 uniformly ([trainer](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_trainer.py:237>)).
- `dev_extrapolation` is lengths 17–32 ([automaton](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:192>)).
- Thus 61.8% is performance on the long half of the training distribution.
- The real OOD split, 33–64, uses learned absolute positions that were never trained ([models](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_models.py:100>)). Its chance-level score is therefore not a clean test of algorithmic extrapolation.

The teacher has learned:

- The entire one-step transition table: 48/48.
- Reliable composition through length 16: 100%.
- A partially working, strongly horizon-dependent composition strategy through length 32: 61.8%.
- It has not learned a uniformly reliable finite-state execution algorithm.

A 38.2% error rate is far too large to call this a competent source model. It is also measured precisely on 20,000 examples, so this is not evaluation noise.

### Why more cyclic training is the wrong bet

The extension already supplied an approximate restart experiment:

- Step 7,000: 50.4%, LR \(3.0\times10^{-5}\).
- Step 7,250: 43.6%, LR jumped to \(1.76\times10^{-4}\).
- Step 9,000: finally recovered to 50.4%.
- Step 15,000: 61.8%.

The restart produced a net 11.4-point gain over 8,000 additional steps, while the final 2,000 steps gained only about 2.1 points. Reaching 99% would require another 37.2 points—more than three entire second-cycle gains despite clear deceleration.

This was not a perfectly clean SGDR cycle, so it does not prove cyclic schedules can never work. It does show that 30,000+ steps and repeated restarts are a speculative, poorly bounded experiment. That cost would later multiply across every development and sealed-key teacher.

### Recommended R9 protocol

1. Gracefully stop the current Transformer student after a checkpoint. Preserve its logs as diagnostic evidence. Do not launch seeds 202 and 203.

2. Close the current protocol as:

   > R7 Stage A FAIL: Transformer long-horizon composition capacity/optimization failure under output-only supervision.

   Do not retroactively relabel it a pass.

3. Replace the generic long-sequence gate with a task-matched capacity gate. Stage B actually tests sequences of the form \(q^p x q^r\), not arbitrary 32-step mixed programs ([design](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/research/STEERING_DIALOGUE_R4_ROUND4_STAGE_BC_2026_07_26.md:98>)).

4. For the new pre-sealed protocol:

   - Keep the 12 states, four permutations, key entropy, and 10× student compression.
   - Use \(p,r\in[0,7]\), giving withheld sequences of length 1–15.
   - Use all 64 \((p,r)\) combinations per withheld edge: 2,304 evaluations. Keys—not examples—are the scientific replicates anyway.
   - Restrict extraction anchors to lengths 8–16.
   - Train the capacity models on lengths 1–16.
   - Rename the splits honestly: `covered_short`, `covered_target`, and optional `length_ood_diagnostic`.

5. Pre-register these capacity gates before new results:

   - Teacher: ≥97.5% on the exact target distribution.
   - Transformer student: at least two of three full-supervision seeds ≥95%.
   - No relevant length bucket below 90%.
   - 48/48 direct edges.
   - Thresholds held at two consecutive evaluations.

   A teacher error ceiling of 2.5 points is small relative to the planned 20-point transfer effect and 10-point lower-confidence threshold. Sixty to seventy percent is not sufficient; its error is larger than the scientific effect being tested.

6. Pilot only three development runs:

   - One teacher.
   - One Transformer student.
   - One GRU student as an architecture diagnostic.

   If the GRU passes and Transformer fails, the problem is Transformer sequential inductive bias—not the 12-state automaton. If both fail, revise supervision or horizon. Only after the teacher and first Transformer pass should the remaining two student seeds run.

### Option verdicts

| Option | Verdict |
|---|---|
| A: 30,000+ steps | Reject as the main path. Poor expected value and multiplies badly downstream. |
| B: SGDR/T_mult=2 | At most a bounded diagnostic, not the rescue plan. |
| C: Larger student | Reject. The teacher already fails, and enlargement destroys the frozen 10× claim. |
| D: Accept 60–70% | Reject. This would convert a failed capacity preflight into an uninterpretable transfer experiment. |
| E: Easier task | Partly correct, but reduce the irrelevant sequence horizon—not the 12-state key space. |
| F: Task-aligned admission redesign | Recommended. It tests capability transfer without making long-horizon Transformer execution a hidden prerequisite. |

The scientific question is whether the artifact carries withheld transition information beyond controls. Thirty-two-step arbitrary composition is a separate question. The current protocol accidentally made that nuisance capability the gatekeeper for the moonshot.

