## Decision

Round 1 was partly a rebrand. The genuine pivot occurs only if we change the unit of success from “a geometric loss improves a student” to:

> **One immutable source-derived artifact installs the same differential competence into multiple incompatible students without teacher access or pair-specific tuning.**

Observable connection remains the only live scientific candidate. Geometry is on probation; we should not launch a seven-codec fishing expedition.

## 1. What actually changes day to day

The old DG-0 workflow was:

1. Choose \(R/\Omega\).
2. Build a loss around it.
3. Train Qwen students.
4. Ask whether accuracy improves.

The Open Capability File workflow must be:

1. Freeze an artifact contract before choosing the encoder.
2. Extract identical bytes once from the teacher.
3. Install those exact bytes into separately initialized students.
4. Test whether the artifact carries post-committed teacher-specific information.
5. Reject any result requiring student-specific extraction, layer-map tuning, or teacher re-querying.

A valid capability file must satisfy:

- immutable source-derived payload;
- identical payload across students;
- frozen architecture-agnostic installation rule;
- teacher absent during installation and deployment;
- differential competence, not generic stabilization;
- explicit byte, extraction, training, and inference costs.

So the answer is:

- **If we merely rename \(\Omega^{\mathrm{obs}}\) a file, this is a rebrand.**
- **If we make the artifact contract primary and force geometry to meet it, this is a genuine pivot.**

If a Jacobian later wins only as a pair-specific matching loss, it is ordinary distillation. If a frozen response-kernel artifact installs unchanged across substrates and composes, it can still support the paradigm—but the “intelligence is geometry” thesis is dead.

For now, do **not** implement Jacobian and perturbation codecs. Test the live hypothesis. If observable geometry fails its admission test, kill this incarnation instead of searching indefinitely for a winning loss.

## 2. The 20-hour tournament was unrealistic

Your arithmetic is correct:

\[
7\text{ codecs}\times6\text{ controls}\times3\text{ seeds}\times8\text{ keys}
=1008\text{ student runs}.
\]

Even for tiny models:

| Mean training time | Student GPU-hours |
|---:|---:|
| 2 minutes | 33.6 |
| 3 minutes | 50.4 |
| 5 minutes | 84.0 |
| 8 minutes | 134.4 |

That excludes teacher training, extraction, probes, evaluation, failed runs, and calibration. Implementation would add approximately 4–10 human days. **A proper seven-codec tournament is more plausibly 70–160 GPU-hours and 2–3 weeks.**

There is also a hardware correction: the live machine is an RTX 5090 **Laptop** GPU with 24 GB, not the 32 GB desktop 5090 assumed in [QL6](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\research\QUESTION_LOOP_06_TRANSFER_VS_REGULARIZATION_2026_07_25.md>) and [QL7](<C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\moonshot-cti-universal-law\research\QUESTION_LOOP_07_EVIDENCE_LADDER_2026_07_25.md>). The automaton remains easy to fit, but language estimates should be revised upward.

## 3. Launch this instead: Geometry Admission Test

Build only two candidate encoders:

1. Raw \(R/\Omega\).
2. Observable-\(R\), using the cheapest predeclared output-gradient/intervention balancing estimator.

Use shared controls:

- no auxiliary channel;
- generic smoothness;
- static \(G\);
- wrong-key trace;
- Haar/spectrum-matched trace.

Do not include FDD, Jacobian sketches, perturbation kernels, LoRA SkillPacks, composition, or relay yet.

### Staged run matrix

**Stage A — Capacity and timing preflight**

- One teacher key.
- Fully supervised 2M Transformer and 2M GRU.
- Require both to reach at least 99% transition accuracy.
- Time ten representative student runs.

The 2M size is intentional: QL6’s 20M→5M design is only **4× compression**, not 10×. A real 10× synthetic claim requires 20M→2M or a larger teacher.

**Stage B — Candidate screen**

- Two development keys.
- One seed.
- Raw and observable candidates plus controls.
- Approximately 14–18 runs.

Select raw or observable using transfer on withheld transitions—not training loss.

**Stage C — Sealed confirmation**

- Eight newly committed keys.
- Three seeds.
- Winning candidate plus five controls.
- \(8\times3\times6=144\) student runs.

For 144 runs:

| Mean run time | Student GPU-hours |
|---:|---:|
| 3 minutes | 7.2 |
| 5 minutes | 12.0 |
| 8 minutes | 19.2 |
| 10 minutes | 24.0 |

Allow another 4–6 hours for teachers, extraction, probes, and evaluation. Therefore:

> **Realistic Geometry Admission Test: 16–30 GPU-hours and 4–7 wall-clock days.**

Set a **30-hour cap**, not 20. The first timed preflight determines whether the sealed confirmation fits.

No per-arm hyperparameter sweep is permitted. Match median auxiliary-gradient norm using one disjoint development key and one frozen coefficient rule. Otherwise the experiment expands combinatorially again.

## 4. Cross-substrate scope

Twenty hours cannot establish language-model cross-substrate portability.

It may establish one narrow synthetic fact:

> The exact same frozen automaton artifact transfers the same withheld transition table into a 2M Transformer and a 2M GRU, both 10× smaller than the 20M teacher.

After the Transformer confirmation passes, duplicating the six-arm sealed matrix for the GRU adds another 144 runs:

- approximately 12–19 GPU-hours at 5–8 minutes per run;
- approximately 3–5 hours for architecture validation and interventions.

So incremental synthetic cross-substrate confirmation is realistically **15–25 GPU-hours**. Do not add an SSM in this budget. GRU is enough to test whether the installer survives a genuinely recurrent realization.

This proves neither language portability nor useful economic intelligence. It tests whether “same artifact, incompatible realization” is even coherent.

## Corrected 100-GPU-hour allocation

| Budget | Work |
|---:|---|
| 20 | Finish and adjudicate Gate B |
| 20 | Execute Gate C demolition |
| 30 | Geometry Admission Test on 2M Transformer |
| 20 | Conditional replication on 2M GRU using identical artifact bytes |
| 10 | Fresh-key reruns, causal ablation, accounting, and failure reserve |

The GRU phase runs only if the Transformer passes every controlled-channel gate. If it fails, those 20 hours remain unspent pending adjudication; they do not become a codec search budget.

Composition, relay, natural-language transfer, and real economics are explicitly outside these 100 hours.

## 5. Honest Turing scores

### Today, unconditional

- Raw DG-0: **2/10**
- Open Capability File thesis: **3/10**

The file thesis has a stronger target and story, but presently it has zero empirical evidence and competes with existing modular capability-transfer systems.

### After 100 hours, if only the single-substrate automaton passes

**3.5/10.**

That would establish a real scientific fact: the trace carries arbitrary, post-committed teacher-specific information that structure-matched regularization does not possess. It would justify a natural-language experiment.

It would not establish a capability file, democratization, or language transfer.

### After 100 hours, if everything realistically testable passes

Assume:

- correct-key specificity;
- withheld-transition recovery;
- causal use;
- 10× compression;
- identical bytes working in Transformer and GRU;
- fresh-key replication.

Score: **4.5/10.**

That is an unusually clean controlled proof of cross-realization information transfer. It is still a toy-domain result.

The previous **8/10** becomes appropriate only after:

- real differential language competence;
- the same artifact across natural model substrates;
- composition/edit/relay;
- favorable end-to-end economics;
- independent replication.

Those are several hundred to thousands of GPU-hours away.

## 6. Is the automaton interesting?

**Yes—but only as an admission test.**

A positive result would prove something language experiments struggle to isolate: an arbitrary secret transition program crossed the teacher–student channel through the proposed artifact. Generic regularization cannot know a post-committed key.

A negative result is equally valuable: it kills observable geometry before expensive language training.

But the automaton proves nothing directly about:

- natural-language semantics;
- reasoning transfer;
- pretrained-model compatibility;
- practical cost;
- capability-file markets;
- democratized intelligence.

It becomes independently publishable only if paired with a substantive result such as an information-rate law, a sharp separation theorem, or cross-substrate composition. Otherwise it is a rigorous internal gate.

## Launch order

1. Finish Gate B without interference.
2. Implement the two-candidate Geometry Admission Test.
3. Run the 2M capacity/timing preflight.
4. Freeze the winner and coefficient rule.
5. Run eight-key Transformer confirmation.
6. Only after a full pass, run the identical artifact on the GRU.
7. Do not begin Qwen DG-0, composition, relay, or additional codecs.

That is the honest next move: **test whether geometry can carry one tiny secret program across two incompatible machines. If it cannot, stop. If it can, earn the right to ask whether the same phenomenon exists in language.**

<oai-mem-citation>
<citation_entries>
MEMORY.md:1322-1324|note=[prior AMCL work established validation boundaries rather than empirical evidence]
MEMORY.md:1296-1298|note=[historical CTI rollout provenance]
</citation_entries>
<rollout_ids>
019ea8d0-f23d-7820-8fd2-f905499ec65b
</rollout_ids>
</oai-mem-citation>