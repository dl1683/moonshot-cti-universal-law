1. Current run — agree

Let the teacher finish, then terminate as soon as `transformer_s1` starts. The completed checkpoint is useful diagnostic evidence:

- Same architecture, seed, optimizer, and 5,000-step budget.
- Demonstrates perfect covered-core fitting alongside chance positional OOD.
- Provides timing and representation comparisons against the repaired teacher.

Quarantine the entire old teacher directory under something like:

```text
results/geometry_admission/diagnostics/r7_train16_teacher_seed101_step5000/
```

Preserve `config.json`, log, summary, checkpoint, and `model_final.pt`. Mark it `invalid_for_stage_a_gate=true`. Do not extract capability artifacts from it.

This quarantine is operationally necessary: the trainer skips any directory containing a completed summary. Also archive any partial `transformer_s1` directory created before termination, because logs are opened in append mode.

2. Evaluation names — agree

Keep these code identifiers:

```text
dev_in_range
dev_extrapolation
stress_long
```

A multi-file rename adds risk without changing behavior. Instead add explicit semantic metadata:

```json
{
  "split_semantics": {
    "dev_in_range": "covered core gate; lengths 1-16",
    "dev_extrapolation": "legacy name; covered long gate; lengths 17-32",
    "stress_long": "positional-OOD diagnostic; lengths 33-64; not gated"
  }
}
```

Update the spec and comments to prohibit describing `dev_extrapolation` as extrapolation after the training range becomes 1–32.

3. Anchor range — agree, with a protocol-version bump

Anchors are not used during capacity optimization. They are shared between teacher trace extraction and installer auxiliary training. Consequently, changing them does not affect the current teacher run or capacity batch memory.

Make this single-source change in [cti_geometry_admission_automaton.py:192](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_automaton.py:192>):

```python
ANCHOR_COUNT = 2048
ANCHOR_LENGTH_RANGE = (8, 32)
ANCHOR_PROTOCOL_ID = "OCF_GAT_ANCHORS_R7_V2"


def generate_anchors(
    n_anchors: int = ANCHOR_COUNT,
    length_range: tuple[int, int] = ANCHOR_LENGTH_RANGE,
    protocol_id: str = ANCHOR_PROTOCOL_ID,
) -> list[dict]:
```

Do not retain `OCF_GAT_ANCHORS_V1`: changing the generated distribution under the same protocol identifier would break provenance.

No change is needed in `partition_anchors_into_banks`:

```text
2048 anchors / 32 banks = 64 anchors per bank
```

Bank size, \(R\)-matrix dimensions, \(U\)-basis dimensions, and artifact byte dimensions remain unchanged. Only sequence width changes:

```text
Maximum input tokens: 25 → 33
Average input tokens: 17 → 21
```

The attention-score tensor grows about 74% at maximum length, but linear activation memory grows only 32%. With extraction batches of 64, this should be well below capacity-training memory. Still record peak memory during the first observable/VJP bank before completing all 32 banks.

Also add to `anchor_manifest.json`:

```json
{
  "protocol_id": "OCF_GAT_ANCHORS_R7_V2",
  "length_range": [8, 32],
  "n_anchors": 2048,
  "n_banks": 32,
  "bank_size": 64
}
```

All anchor hashes, bank membership, controls, extraction artifacts, and edge-coverage results must then be regenerated. The longer range increases expected edge traversals by 25%, so coverage should improve rather than deteriorate.