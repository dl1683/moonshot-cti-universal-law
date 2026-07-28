# Affordable Intelligence Science

This repository began as **CTI Universal Law**. That work is retained as a
legacy research record, but it is no longer the active thesis or the
authoritative description of this project.

## Current program

The active program is the **Scale-Inversion Atlas**:

> Given a workload, hardware, volume, quality requirement, and safety floor,
> what is the cheapest complete AI system that meets them?

The goal is prospective model and system selection under explicit constraints,
not another correlation between model scale and benchmark accuracy. A useful
result must include the full system, quality and safety floors, adaptation and
verification cost, energy, latency, and failure modes.

## Current status

**Atlas R2.1 completed Gate A from W-D2 only. No prospective selector claim has
been earned.**

The canonical live record is [STATUS.md](STATUS.md). As of 2026-07-28:

- P0 preflight was only a partial pass.
- W-D2 (MKQA) selected six anchors plus three exploratory systems. Scaling was
  monotonic within all three local families; no local scale inversion appeared.
- R2.1 W-D3 (PolicyBench) is finalized as diagnostic-only with **0% Gate A
  weight**. Its all-zero collapse and 11% Qwen3-14B parse failure motivated a
  prospective repair, not a retroactive rescore.
- The narrow R2.2 amendment is frozen, but no R2.2 execution is authorized
  until its panels, implementation, and fail-closed verifier pass.

The binding protocol is
[the Atlas R2 design gate](results/codex_scale_inversion_atlas_design_gate_r2.md)
plus [R2.1](precommit/atlas_r2_protocol_r2_1.md) and the narrow
[R2.2 amendment](precommit/atlas_r2_protocol_r2_2.md). The R2.1 W-D3
adjudication is recorded in [Codex steering](results/codex_steering_wd3_r2.md).

## What happened to CTI Universal Law

The original project studied a relationship between normalized 1-NN accuracy
and nearest-class representation separation. It produced broad internal
validation and a paper artifact, but later adversarial review sharply reduced
the claim:

- predictor and target are closely related measurements of the same labeled
  geometry;
- the strongest reported fit used per-dataset intercepts and did not transfer
  well to unseen datasets;
- the leave-one-architecture-out stability statistic was an overlapping
  jackknife, not independent replication;
- downstream ranking evidence was confounded by scale; and
- the equicorrelation result contained a whitening bug and matched a geometric
  null after correction.

The paper and surviving artifacts remain in `paper/` and the status history for
auditability. They must not be described as a current universal law, a validated
economic reduction, or the portfolio flagship. The later Geometry Admission
Test, Causal Skill Organs, and several follow-on directions also failed their
gates; their negative evidence remains summarized in [STATUS.md](STATUS.md).

## Claim boundary

This repository currently supports:

- a carefully specified prospective evaluation program;
- completed raw screens and metric-collapse findings;
- preserved negative results from the CTI, GAT, CSO, and direction-selection
  arcs; and
- infrastructure for cost, energy, thermal, workload, and system accounting.

It does **not** currently support:

- a universal law of learned representation quality;
- a validated Scale-Inversion selector;
- a claim that smaller systems beat larger systems in general;
- a PolicyBench capability ordering from the current W-D3 metric; or
- a demonstrated end-to-end cost reduction for a real deployment.

## Repository map

```text
STATUS.md          Canonical program state, kill records, and current results
precommit/         Frozen Atlas protocol and verification contracts
scripts/           Canonical Atlas runner, scorer, and precommit verifier
src/               Atlas workloads, inference, API, and energy instrumentation
results/           Active task records and Codex steering/audit trail
paper/             Legacy CTI paper and retained reference artifacts
experiments/       Compact experiment ledger
```

## Reproduce or inspect the active Atlas

```powershell
# Validate the currently available precommit contracts
python scripts/verify_atlas_r2_precommit.py

# Check a run without executing it
python scripts/run_atlas_r2.py --phase P1 --workload W-D2 --dry-run

# Inspect the current ledger
python scripts/score_atlas_r2.py --summary
```

Do not run confirmation phases until the selector and task-seal files required
by the verifier exist and the relevant gate explicitly authorizes them.

## Hardware

Local experiments target one NVIDIA RTX 5090 laptop GPU with 24 GB VRAM.
External API systems are part of the Atlas only when their costs, versions,
quality, safety, and latency are included in the same accounting contract.
