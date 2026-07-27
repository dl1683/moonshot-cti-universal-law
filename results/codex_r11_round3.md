Converged. The next program is **Causal Skill Organs (CSO)**, and its minimal admission test should be a structured register transducer—not a stack machine.

The register task removes the three largest feasibility risks: unbounded memory, unconstrained Markov-state discovery, and excessive intervention cost. It is an admission test only; success licenses the world-model flagship but does not itself earn the moonshot claim.

## 1. Minimal admission test: Causal Register Transducer

### Task

Maintain four registers:

\[
r_t=(r_0,r_1,r_2,r_3)\in\mathbb Z_{16}^4.
\]

The state space has \(16^4=65{,}536\) states. Each instruction applies one of eight invertible, noncommuting rules:

```text
U0: r0 <- r0 + r1 mod 16
U1: r1 <- r1 + r2 mod 16
U2: r2 <- r2 + r3 mod 16
U3: r3 <- r3 + r0 mod 16
U4: swap(r0, r2)
U5: swap(r1, r3)
U6: rotate registers left
U7: negate r0 and r2 mod 16
```

Input:

```text
initial register values + instruction sequence
```

Output:

```text
four final register values
```

Use four independent 16-way output heads. Exact-state chance accuracy is \(1/65{,}536\).

This sits in the intended middle:

- Unlike random permutations, it has a compact reusable algorithm.
- Unlike a stack machine, memory is fixed and tiny.
- Composition is noncommutative and order-sensitive.
- The full lookup table is 1 MiB, while the causal rule is local and compact.
- Longer execution requires repeated correct application, not recognition of a fixed answer.

### Withheld dimensions

Training:

- Sequence lengths 1–12.
- Sixteen of the 64 ordered instruction bigrams excluded.
- Initial states selected by a frozen 75% hash partition.
- Online examples; no intermediate register supervision.

Evaluation:

- Lengths 13–32.
- Excluded bigrams and precommitted unseen trigrams.
- The 25% held-out initial-state partition.
- Counterfactual suffixes following state interchange interventions.
- Exact register-state accuracy primary; per-register accuracy diagnostic.

This tests three things simultaneously:

1. unseen compositions;
2. length extrapolation;
3. interventionally correct state evolution.

### Models

Use a constrained admission scaffold:

- Donor: 19.5M recurrent-state Transformer using the existing Transformer block family.
- Hosts: approximately 1.9M Transformer and 1.85M GRU.

The donor processes one instruction at a time through a designated but unlabeled latent state slot. The state slot is an architectural causal boundary; the true register values are never provided there.

This deliberately avoids asking Round 1 to discover a Markov boundary inside an unrestricted Transformer. The extraction problem becomes:

> Find a compact causal abstraction inside a known recurrent boundary.

That is sufficient for an admission test. General extraction from unconstrained pretrained models remains a later gate.

### Organ extraction

Freeze the donor. Fit a maximum 32-dimensional organ state and a transition core capped at 32,000 quantized parameters:

\[
z_{t+1}=F_\phi(z_t,U_t).
\]

Training evidence:

- ordinary donor state transitions;
- approximately 12,000 interchange-intervention tuples;
- paired prefixes with donor states swapped;
- shared frozen suffixes after the swap;
- donor behavioral response to the intervention.

Reserve at least 3,000 intervention tuples for final-only evaluation.

Forbidden:

- ground-truth register states as organ supervision;
- simulator transition labels;
- withheld answers;
- rule source code;
- per-host organ retraining.

The crucial control is the identical organ architecture trained only on observational donor trajectories. If ordinary transition fitting works equally well, causal intervention data added nothing.

### Installation

The organ bytes are frozen once.

Both hosts receive:

- the same organ;
- the same generic read/write socket contract;
- no new donor queries;
- no task-specific organ tuning.

The organ remains active at inference. The total host-plus-organ system—not the naked host—is the economic unit.

### Admission thresholds

The teacher must first reach:

- ≥99.5% exact accuracy on lengths 1–12;
- ≥99.0% on lengths 13–32;
- ≥99.0% on excluded compositions.

Each fully supervised host must independently reach ≥99.0% on the same evaluation. That establishes capacity.

CSO admission then requires all of:

- ≥95% exact accuracy on the full withheld intersection.
- ≥90% counterfactual state-swap fidelity.
- At least 15 points over the best observational bottleneck, output-KD, or trajectory-distillation control.
- Identical organ bytes succeed in both Transformer and GRU hosts.
- Wrong-donor organ produces the corresponding wrong transition behavior rather than generic degradation.
- Organ ablation removes at least 80% of the acquired advantage.
- Artifact ≤64 KiB—at most 6.25% of the explicit transition table.
- Total host-plus-organ inference compute ≤10% of the donor.
- Total admission budget ≤40 GPU-hours and four calendar weeks.

The 64 KiB limit is an anti-table gate, not a claim of Kolmogorov optimality.

### Expected budget

Precommit after a throughput smoke, but the working allocation should be:

| Work | Cap |
|---|---:|
| Simulator, tests, teacher smoke | 2 GPU-hours |
| Donor capacity run and repeats | 8 GPU-hours |
| Intervention generation | 10 GPU-hours |
| Organ extraction | 8 GPU-hours |
| Two hosts plus controls | 10 GPU-hours |
| Failure reserve | 2 GPU-hours |
| **Total** | **40 GPU-hours** |

If the intervention generator exceeds its ten-hour cap, reduce tuple count only through a predeclared power/timing review—not after seeing transfer results.

## 2. First three Monday steps

### Step 1 — Canonically close GAT

Create one closure commit that:

- updates `STATUS.md`;
- marks GAT Stage C-I permanently cancelled;
- marks conceptual Stage D-U permanently cancelled;
- cancels conditional GRU geometry confirmation;
- removes every active “proceed to Stage C” instruction;
- makes `cti_geometry_admission_stage_c.py` fail immediately with a permanent-closure message or moves it to a clearly historical surface;
- preserves Stage A provenance and all Stage B evidence.

Do not delete or rewrite:

- `results/geometry_admission/stage_b/precommit.json`;
- `decision.json`;
- the twelve run summaries;
- checkpoint hashes or protocol records.

### Step 2 — Lock the admission protocol

Create:

```text
research/CAUSAL_SKILL_ORGAN_ADMISSION_V1.md
```

Freeze before implementation:

- eight register operations;
- data hashes and withholding rules;
- donor and host configurations;
- explicit recurrent-state boundary;
- intervention construction;
- organ size and parameterization;
- observational and KD controls;
- success, FAIL, and VOID rules;
- 40-hour budget ceiling;
- code and artifact manifest requirements.

No task simplification after teacher results are visible.

### Step 3 — Implement only the simulator and capacity smoke

First implementation unit:

```text
src/cti_causal_register_transducer.py
src/cti_causal_organ_models.py
```

Monday’s executable gate:

- exhaustively verify all eight one-step operations over all 65,536 states;
- verify inverse and composition properties;
- verify frozen data partitions and hashes;
- run a 500-step donor smoke;
- measure throughput and memory;
- confirm training loss falls materially and no forbidden intermediate state reaches the model.

Stop after the smoke and review the receipt before building intervention extraction. Do not let simulator work silently expand into the whole program.

## 3. Updated manifesto

Retire the strong empirical axiom:

> Intelligence = Geometry.

Retain the anti-scale mission, but use a falsifiable working hypothesis:

> **Capabilities are causal mechanisms with interfaces—not geometric shapes and not monolithic checkpoints.**

Public framing:

> **From giant models to transplantable causal skills.** Train an expensive model once, extract the mechanism it learned, and let cheap local AIs reuse it as an open, inspectable organ.

The economic promise is:

- one extraction;
- many incompatible recipients;
- active local execution;
- no donor at deployment;
- no full-model redistribution;
- eventual composition without joint retraining.

The flagship headline remains reserved for a world-model organ solving novel goals. The register transducer is only the admission gate.

## 4. Kill criteria

### Protocol VOID

VOID—not scientific failure—if:

- forbidden register-state labels enter extraction;
- data partitions or hashes drift;
- host capacity is not established;
- donor interventions are incorrectly applied;
- organ bytes differ between hosts;
- withheld results influence extraction or selection.

### Admission-task failure

The register test itself fails, without yet killing CSO, if the donor cannot pass its full-supervision gates within eight GPU-hours. One predeclared fallback is allowed: reduce modulus from 16 to 8 while keeping four registers and all withheld composition tests. No other task redesign.

If that fallback also fails, stop the admission program and reassess feasibility.

### Scientific kill of causal extraction

Kill the distinctive causal-mechanism claim if any occurs:

1. The observational bottleneck is within three points of the interventionally trained organ at matched size and compute.
2. Output KD or trajectory distillation matches the organ within three points.
3. Counterfactual fidelity is below 90%, even if ordinary accuracy is high.
4. Organ ablation does not selectively remove the transferred behavior.
5. A wrong-donor or random organ works similarly.
6. The same bytes fail in either host.
7. Task-specific socket training is required.
8. Ground-truth latent-state labels are necessary.
9. Admission exceeds 40 GPU-hours or four weeks without crossing 70% counterfactual fidelity.

Any of 1–8 means the organ is ordinary compression, regularization, or interface engineering—not causal transplantation.

### Moonshot kill

Even after admission, do not retain the moonshot narrative unless the later world-model experiment demonstrates:

- novel-goal behavior without teacher actions for those goals;
- advantage over a directly trained world model at matched environment experience;
- reuse across at least three hosts;
- and eventually zero-shot composition without a joint teacher or joint labels.

Failure there can leave CSO as useful modular compression, but it kills the paradigm-level claim.

## 5. Permanent GAT closure

Confirmed:

- **GAT Stage C-I: permanently closed.**
- **GAT Stage D-U: permanently closed.**
- **Conditional GRU geometry confirmation: permanently closed.**
- **No extra seeds, coefficient rescues, skew-only variants, or observable-\(R\) revisions.**
- **Stage B remains the canonical clean kill.**

There is no live Stage D implementation; Stage D-U exists only as an optional concept in the earlier R8 design. It should be marked cancelled wherever referenced.

Stage A remains as provenance showing teacher/student capacity and artifact extractability. Stage B remains immutable evidence showing that installable geometric resemblance did not transfer capability.

Formal program name:

> **Causal Skill Organs: Interventional Mechanism Transfer Across Architectures**

This avoids collision with *Neural Organ Transplantation*. The organ metaphor should acknowledge that prior paper; the novelty claim is the causal, cross-architecture, standalone, and ultimately compositional conjunction.

