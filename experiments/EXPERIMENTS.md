# Scale-Inversion Atlas -- Experiments Log

> Historical CTI Universal Law experiments (Sessions 1-100) removed Jul 28 2026.
> CTI paper retained in paper/ for reference only.

All experiments listed in reverse chronological order.

---

## Atlas R2.1 P1 -- W-D3 PolicyBench (Jul 2026) [DIAGNOSTIC ONLY]

- Config: `configs/atlas_r2_systems.yaml`, Protocol: `precommit/atlas_r2_protocol_r2_1.md`
- Records: `results/cti_atlas_r2_task_records/cti_atlas_r2_r2.1_P1_W-D3_qwen3_*.json`
- Codex steering: `results/codex_steering_wd3_emerging.md`, `results/codex_steering_wd3_r2.md`

100 households, 20 fields each. Qwen3 family only (0.6B, 4B, 14B). ALL COMPLETE.
Remaining 6 systems (gemma3 x3, falcon_h1 x3): NOT RUN per Codex R2 hard stop.
Label: `diagnostic_r2.1_deviation_384_120` (protocol violations: 384 vs 128 tokens, 120 vs 30s).

Results: qwen3_14b 88.0% pass / 0.7565 mean / 11% parse fail / valid-parse mean 0.8500;
qwen3_0.6b 94.0% pass / 0.7654 mean; qwen3_4b 55.0% pass / 0.5096 mean.

**What we learned:** All-zero baseline dominates (100% pass, 84.85% mean) because ~85% of
reference fields are zero. Metric is collapsed. The 14B valid-parse mean (0.8500) actually
EXCEEDS all-zero (0.8485) -- the aggregate gap is entirely from 11% parse failures caused by
token truncation at 384 tokens. This is a reliability bottleneck, not a capability gap.
Compact schema would likely eliminate most failures.
Gate A weight: 0%. R2.2 amendment required with compact schema and rescue/harm metrics.

## Atlas R2.1 P1 -- W-D2 MKQA (Jul 2026) [COMPLETE]

- Config: `configs/atlas_r2_systems.yaml`, Protocol: `precommit/atlas_r2_protocol_r2_1.md`
- Records: `results/cti_atlas_r2_task_records/cti_atlas_r2_r2.1_P1_W-D2_*.json`
- Ledger: `experiments/ledger.jsonl` (id: atlas_r2.1_p1_wd2)

9 local systems (3 families x 3 sizes) on 320 MKQA episodes (40 queries x 8 languages).
Best: gemma3_12b (F1=0.254). Worst: falcon_h1_0.5b (F1=0.040). Total: 1.15 GPU-hours.

**What we learned:** Monotonic scaling within all 3 families (bigger=better). No scale
inversion found in local systems. API ladder needed to establish Goliath ceiling for
cost ratio computation. Cross-family ordering differs from within-family (gemma3_12b >
qwen3_14b despite fewer params).
