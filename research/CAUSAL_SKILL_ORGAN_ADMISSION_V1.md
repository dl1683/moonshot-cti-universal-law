# Causal Skill Organ Admission Test V1

**Protocol ID:** CSO_ADMISSION_V1
**Status:** LOCKED (Jul 26, 2026) -- do not modify post-lock
**Designed by:** Codex R11 steering dialogue (3 rounds)
**Prior art gap confirmed:** Claude internet search Jul 26, 2026

---

## 1. Thesis Under Test

> Capabilities are causal mechanisms with interfaces -- not geometric shapes and not
> monolithic checkpoints. A learned causal mechanism can be extracted as an executable
> organ, transplanted as frozen identical bytes into architecturally incompatible hosts.

This admission test establishes whether causal mechanism extraction and cross-architecture
transplantation is feasible on a structured synthetic task. It does NOT itself earn the
moonshot claim -- that requires the later world-model organ experiment (novel goals,
zero-shot composition).

---

## 2. Task: Causal Register Transducer

### State space

Four registers: r = (r0, r1, r2, r3) in Z_16^4.
Total state space: 16^4 = 65,536 states.

### Operations

Eight invertible, noncommuting rules:

| Op | Name | Rule |
|----|------|------|
| U0 | add_01 | r0 <- r0 + r1 mod 16 |
| U1 | add_12 | r1 <- r1 + r2 mod 16 |
| U2 | add_23 | r2 <- r2 + r3 mod 16 |
| U3 | add_30 | r3 <- r3 + r0 mod 16 |
| U4 | swap_02 | swap(r0, r2) |
| U5 | swap_13 | swap(r1, r3) |
| U6 | rotate_L | (r0,r1,r2,r3) <- (r1,r2,r3,r0) |
| U7 | neg_02 | r0 <- -r0 mod 16; r2 <- -r2 mod 16 |

Properties to verify exhaustively before any training:
- All 8 operations are invertible (bijections on Z_16^4)
- Non-commutativity: U_i(U_j(r)) != U_j(U_i(r)) for at least some (i,j,r) pairs
- The 8 operations generate a group of order >> 65,536

### Input/output format

Input: [r0_init, r1_init, r2_init, r3_init, op_1, op_2, ..., op_L]
Output: [r0_final, r1_final, r2_final, r3_final]

Four independent 16-way classification heads (one per register).
Per-register chance accuracy: 1/16 = 6.25%.
Exact-state chance accuracy: 1/65,536 = 0.0015%.

### Data partitions (frozen at protocol lock)

**Training:**
- Sequence lengths: 1-12
- Initial states: 75% partition (hash-based, frozen seed = 42)
- Instruction bigrams: 48 of 64 ordered bigrams included (16 excluded)
- Online sampling; no intermediate register supervision

**Evaluation (withheld):**
- Lengths 13-32 (length extrapolation)
- The 16 excluded bigrams (compositional generalization)
- Precommitted unseen trigrams (deeper composition)
- The 25% held-out initial-state partition
- Counterfactual suffixes following state interchange interventions

**Partition hashes:** Computed at implementation time, frozen in precommit.json before
any training begins. No changes to partitions after teacher results are visible.

---

## 3. Models

### Donor

19.5M-parameter recurrent-state Transformer.

Architecture: existing Transformer block family from GAT, but with a designated
(unlabeled) latent state slot that recurs across instruction steps. The state slot
is an architectural causal boundary -- the true register values are NEVER provided
there. The donor processes one instruction at a time through this recurrent state.

This deliberately constrains the extraction problem:
> Find a compact causal abstraction inside a known recurrent boundary.

General extraction from unconstrained pretrained models remains a later gate.

### Hosts

- Host T: ~1.9M Transformer (same block family, fewer layers/dims)
- Host G: ~1.85M GRU

Both hosts must independently pass fully-supervised capacity gates before organ
installation. This establishes that the architecture CAN solve the task.

---

## 4. Organ Specification

### Dimensions

- State: maximum 32-dimensional (z_t in R^32)
- Transition core: maximum 32,000 quantized parameters
- Total artifact size: <= 64 KiB (anti-table gate: at most 6.25% of the 1 MiB lookup table)

### Functional form

z_{t+1} = F_phi(z_t, U_t)
m_t = G_phi(z_t)

where:
- z_t is the organ state
- U_t is the current instruction
- F_phi is the transition function
- G_phi is the readout/message function

### Extraction procedure

Freeze the donor. Fit the organ from:
- Ordinary donor state transitions (observational)
- ~12,000 interchange-intervention tuples (causal)
  - Paired prefixes with donor states swapped
  - Shared frozen suffixes after the swap
  - Donor behavioral response to the intervention
- Reserve >= 3,000 intervention tuples for final-only evaluation

### Forbidden information

- Ground-truth register states as organ supervision
- Simulator transition labels
- Withheld answers
- Rule source code
- Per-host organ retraining

### Installation

The organ bytes are frozen ONCE, before either host receives them.

Both hosts receive:
- The same organ (identical bytes)
- The same generic read/write socket contract
- No new donor queries
- No task-specific organ tuning

The organ remains ACTIVE at inference. The total host-plus-organ system is the
economic unit.

---

## 5. Controls and Baselines

The organ must beat ALL of the following at equal teacher interactions, bytes,
and student compute:

1. Output/logit KD (teacher outputs only)
2. On-policy imitation (teacher actions)
3. Hidden-state / trajectory distillation
4. Raw and observable geometry (the GAT approach -- expected to fail)
5. Observational recurrent bottleneck trained WITHOUT interventions (critical control)
6. A world model trained directly from environment trajectories
7. LoRA or parameter-delta transfer where architecture permits
8. Retrieval of cached teacher trajectories
9. Randomly initialized, norm-matched organ (structural null)
10. Organ from wrong donor or wrong dynamics (specificity control)

The most dangerous comparator is #5 (observational bottleneck). If an ordinary
learned transition model from the same trajectories (without interventional extraction)
matches the transplanted organ, the causal claim dies.

---

## 6. Success Criteria (Admission)

### Teacher capacity gate

The donor must first reach:
- >= 99.5% exact accuracy on lengths 1-12
- >= 99.0% on lengths 13-32
- >= 99.0% on excluded compositions

### Host capacity gate

Each fully supervised host must independently reach:
- >= 99.0% exact accuracy on the same evaluation splits

This establishes capacity independently of transplantation.

### CSO admission (ALL required)

- >= 95% exact accuracy on the full withheld intersection
- >= 90% counterfactual state-swap fidelity
- At least 15 points over the best observational bottleneck/KD/trajectory control
- Identical organ bytes succeed in both Transformer and GRU hosts
- Wrong-donor organ produces corresponding wrong transition behavior (not generic degradation)
- Organ ablation removes at least 80% of the acquired advantage
- Artifact <= 64 KiB
- Total host-plus-organ inference compute <= 10% of the donor
- Total admission budget <= 40 GPU-hours and 4 calendar weeks

---

## 7. Verdict Rules

### PASS

All admission criteria met. Licenses the world-model flagship experiment.

### FAIL

Any scientific kill criterion triggered (Section 8). The causal extraction thesis
is dead for this task class.

### VOID

Protocol violated (Section 8). Re-run required with corrected protocol.

---

## 8. Kill Criteria

### Protocol VOID (not scientific failure)

VOID if:
- Forbidden register-state labels enter extraction
- Data partitions or hashes drift
- Host capacity is not established independently
- Donor interventions are incorrectly applied
- Organ bytes differ between hosts
- Withheld results influence extraction or selection

### Admission-task failure (does not kill CSO)

The register test itself fails, without killing CSO, if the donor cannot pass
its capacity gates within 8 GPU-hours. One predeclared fallback allowed:
reduce modulus from 16 to 8 while keeping 4 registers and all withheld
composition tests. No other task redesign. If the fallback also fails, stop
and reassess feasibility.

### Scientific kill of causal extraction

Kill the distinctive causal-mechanism claim if ANY occurs:

1. Observational bottleneck within 3 points of organ at matched size/compute
2. Output KD or trajectory distillation matches organ within 3 points
3. Counterfactual fidelity below 90%, even if ordinary accuracy is high
4. Organ ablation does not selectively remove the transferred behavior
5. Wrong-donor or random organ works similarly
6. Same bytes fail in either host
7. Task-specific socket training is required
8. Ground-truth latent-state labels are necessary
9. Admission exceeds 40 GPU-hours or 4 weeks without >70% counterfactual fidelity

Any of 1-8 means the organ is ordinary compression, regularization, or interface
engineering -- not causal transplantation.

### Moonshot kill (post-admission)

Even after admission passes, the moonshot narrative requires the later world-model
experiment to demonstrate:
- Novel-goal behavior without teacher actions for those goals
- Advantage over a directly trained world model at matched experience
- Reuse across at least 3 hosts
- Zero-shot composition without joint teacher or joint labels

Failure there leaves CSO as useful modular compression but kills the paradigm claim.

---

## 9. Budget

| Work | GPU-hours cap |
|------|---:|
| Simulator, tests, teacher smoke | 2 |
| Donor capacity run and repeats | 8 |
| Intervention generation | 10 |
| Organ extraction | 8 |
| Two hosts plus controls | 10 |
| Failure reserve | 2 |
| **Total** | **40** |

If the intervention generator exceeds its 10-hour cap, reduce tuple count only
through a predeclared power/timing review -- not after seeing transfer results.

---

## 10. Implementation Manifest

### Required files (cti_* naming convention)

| File | Purpose |
|------|---------|
| cti_causal_register_transducer.py | Simulator, data generation, partitions |
| cti_causal_organ_models.py | Donor, host, organ architectures |
| cti_causal_organ_trainer.py | Capacity training (donor + hosts) |
| cti_causal_organ_extraction.py | Intervention generation + organ fitting |
| cti_causal_organ_installer.py | Organ installation into hosts |
| cti_causal_organ_evaluation.py | Withheld + counterfactual evaluation |
| cti_causal_organ_controls.py | All 10 baselines/controls |
| cti_causal_organ_adjudicate.py | Verdict computation (PASS/FAIL/VOID) |

### Required result artifacts

| Artifact | Location |
|----------|----------|
| Precommit (frozen before training) | results/causal_organ/precommit.json |
| Donor capacity results | results/causal_organ/donor_capacity.json |
| Host capacity results | results/causal_organ/host_capacity.json |
| Intervention tuples (local only) | results/causal_organ/interventions/ |
| Organ checkpoint (local only) | results/causal_organ/organ.pt |
| Evaluation results | results/causal_organ/evaluation.json |
| Control results | results/causal_organ/controls.json |
| Verdict | results/causal_organ/verdict.json |

---

## 11. Prior Art Acknowledgment

This direction was informed by:
- GAT Stage B clean kill (this repo, Jul 26 2026)
- Neural Organ Transplantation (NOT), arxiv 2601.13580 (same-architecture only)
- Circuit Distillation, arxiv 2509.25002 (same architecture family)
- CT-SFT, arxiv 2601.08146 (same architecture, cross-lingual)
- Causal Abstraction / DAS / Boundless DAS (Geiger et al. 2021-2024)
- Compositional World Models (ICLR 2026 workshop)

The claimed novelty is the conjunction of: automatic extraction, causal/interventional
validation, standalone inference-time execution, cross-architecture transplantation,
and eventual zero-shot composition. Individual ingredients are not novel.
