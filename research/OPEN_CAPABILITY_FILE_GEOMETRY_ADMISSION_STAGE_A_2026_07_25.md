## Gate C decision

**Accepted: kill Gate C permanently.**

This is not avoiding a falsifier; it is refusing to rebuild a low-prior fallback after its unvalidated infrastructure was deliberately removed. I verified that zero AMCL-named files remain.

Gate B is now CTI’s final live empirical gate. A Gate B success would justify a CTI diagnostic result—not automatically resurrect AMCL. Any future compute-governor proposal must enter as a fresh direction with its own evidence and budget.

Accepted allocation:

| GPU-hours | Work |
|---:|---|
| 20 | Finish and adjudicate Gate B |
| 30 | Transformer Geometry Admission Test |
| 25 | Conditional GRU confirmation |
| 25 | Extra seeds, causal ablation, accounting, and failure reserve |

## Stage A specification

Stage A is strictly a capacity, timing, and extraction preflight. Every model receives full supervision. It does not test transfer.

### 1. Automaton

States:

\[
S=\{0,\ldots,11\}
\]

Alphabet:

\[
\mathcal A=\{a,b,c,d\}
\]

A key is four independently sampled state permutations:

\[
K=(\pi_a,\pi_b,\pi_c,\pi_d),\qquad \pi_x\in S_{12}.
\]

For initial state \(s_0\) and operation sequence \(x_1,\ldots,x_L\):

\[
s_t=\pi_{x_t}(s_{t-1}),\qquad y=s_L.
\]

The model receives \(s_0\) and the operations, then predicts the final state as a 12-way classification problem. Chance accuracy is \(1/12=8.33\%\).

The key entropy is approximately:

\[
4\log_2(12!)=115.34\text{ bits}.
\]

Duplicate symbol permutations remain legal; rejecting them would alter the declared key distribution.

### Stage A development key

Use this fixed development key:

```json
{
  "a": [1, 5, 4, 6, 3, 7, 8, 0, 9, 2, 10, 11],
  "b": [8, 10, 11, 5, 6, 1, 9, 3, 4, 0, 7, 2],
  "c": [11, 5, 1, 0, 6, 10, 2, 7, 9, 4, 8, 3],
  "d": [1, 7, 2, 4, 10, 5, 9, 0, 3, 8, 6, 11]
}
```

It was generated from:

```text
SHA256("GAT_STAGE_A_DEV_KEY_V1")
= 44fdf7ee93401f92dd8f42c66e989b16bba0e9087b39d7080d8d93fb55ea3845
```

This key must never be reused among the sealed eight confirmation keys.

### Sealed keys later

After implementation, controls, and configuration are frozen:

1. Generate 32 random bytes per key using the operating-system CSPRNG.
2. Publish `SHA256(key_seed)` before training.
3. Derive four domain-separated seeds.
4. Sample each permutation with NumPy `PCG64DXSM`.
5. Store the resulting permutations in canonical JSON so library changes cannot alter them.

## 2. Data format

Vocabulary:

```text
0..11 = STATE_0 .. STATE_11
12    = operation a
13    = operation b
14    = operation c
15    = operation d
16    = PAD
```

Example:

```text
input_ids     = [STATE_s0, OP_x1, ..., OP_xL]
attention_mask = [1, 1, ..., 1]
label          = final state sL, integer 0..11
```

There is no BOS, EOS, rationale, teacher answer, or intermediate-state supervision. Right-pad only within a batch. Classify from the final non-padding operation token.

### Training stream

For every example:

- \(s_0\sim\mathrm{Uniform}(0,\ldots,11)\)
- \(L\sim\mathrm{Uniform}(1,\ldots,16)\)
- each operation sampled uniformly and independently
- label computed exactly by applying the key

Generate training examples online. Input RNG seeds must be independent of model initialization and identical across architectures.

### Frozen evaluation sets

Generate and hash before training:

| Split | Examples | Operation lengths |
|---|---:|---:|
| `dev_in_range` | 20,000 | 1–16 |
| `dev_extrapolation` | 20,000 | 17–32 |
| `stress_long` | 20,000 | 33–64 |
| `direct_edges` | 48 | Every state × one operation |

`stress_long` is diagnostic, not a Stage A gate.

## 3. Architectures

Implement the Transformers locally using bias-free attention/MLP projections, pre-RMSNorm, SwiGLU, learned absolute positions, causal attention, and zero dropout.

### Teacher Transformer

| Field | Value |
|---|---:|
| Layers | 12 |
| Width | 384 |
| Heads | 6 |
| Head dimension | 64 |
| SwiGLU hidden width | 896 |
| Maximum positions | 65 |
| Parameters | 19,509,900 |

Parameter formula per block:

\[
4d^2+3d\,d_{\mathrm{ff}}+2d=1,622,784.
\]

The classifier is `Linear(384, 12, bias=True)`.

### Transformer student

| Field | Value |
|---|---:|
| Layers | 6 |
| Width | 160 |
| Heads | 5 |
| Head dimension | 32 |
| SwiGLU hidden width | 448 |
| Maximum positions | 65 |
| Parameters | 1,921,772 |
| Compression | 10.15× |

The classifier is `Linear(160, 12, bias=True)`.

### GRU student

Use six separately instantiated one-layer `torch.nn.GRU` modules so every depth checkpoint is accessible.

| Field | Value |
|---|---:|
| Token embedding | 128 |
| Input projection | 128→224 |
| GRU layers | 6 |
| Hidden width | 224 |
| Layer input width | 224 |
| Inter-layer normalization | RMSNorm |
| Dropout | 0 |
| Parameters | 1,849,740 |
| Compression | 10.55× |

Flow:

```text
token embedding
→ Linear(128, 224)
→ GRU layer 1
→ RMSNorm
...
→ GRU layer 6
→ RMSNorm
→ final-token classifier
```

The classifier is `Linear(224, 12, bias=True)`.

Do not enlarge a failed student within this protocol. That would invalidate the 10× precommit.

## 4. Capacity training

Runs:

```text
teacher seed:             101
Transformer student:      201, 202, 203
GRU student:              301, 302, 303
```

Loss:

\[
\mathcal L_{\mathrm{task}}
=-\frac1B\sum_i\log p_\theta(y_i\mid s_{0,i},x_{1:L_i}).
\]

No trace loss or intermediate supervision.

Optimizer:

```text
AdamW
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.01

teacher lr = 3e-4
Transformer student lr = 5e-4
GRU student lr = 1e-3

batch size = 512
warmup = 250 steps
schedule = cosine decay to 10% of peak lr
maximum steps = 5000
gradient clipping = 1.0
precision = bf16 compute, fp32 optimizer states
evaluation every 250 steps
```

### Capacity gates

Teacher:

- ≥99.5% in-range
- ≥99.0% extrapolation
- 48/48 direct edges

Each student architecture:

- at least two of three seeds reach ≥99.0% in-range;
- those same seeds reach ≥99.0% extrapolation;
- those same seeds score 48/48 direct edges;
- no seed falls below 98.5% in-range.

Thresholds must hold at two consecutive evaluations.

If the Transformer fails, Stage B is blocked. If only the GRU fails, Transformer screening may proceed but cross-substrate confirmation is blocked.

## 5. Anchor selection

Use random, key-independent anchors—not adversarial or disagreement-selected anchors.

```text
anchor_seed = SHA256("OCF_GAT_ANCHORS_V1")
anchor_count = 2048
initial state = uniform
length = uniform from 8..24
operations = iid uniform
```

Generate and hash anchors before sealed keys. The same ordered anchors must be used for every key and architecture.

Partition them into 32 banks of 64 by sorting on:

```text
SHA256(protocol_id || canonical_input_tokens)
```

For every key, audit traversal counts for all 48 state-symbol edges. Do not resample keys or anchors based on coverage. A minimum count below 400 is a protocol failure requiring adjudication.

## 6. Depth clock

Use six computation transitions everywhere.

Teacher checkpoints:

```text
T0 = token + position embedding
T1 = after layer 2
T2 = after layer 4
T3 = after layer 6
T4 = after layer 8
T5 = after layer 10
T6 = after layer 12
```

Transformer student:

```text
S0 = token + position embedding
Sj = after student layer j
```

GRU student:

```text
S0 = 224-dimensional projected token sequence
Sj = after GRU layer j
```

No interpolation, learned clock, or outcome-tuned layer mapping is allowed.

## 7. Raw trace extraction

For a 64-example anchor bank, take the final-token state at checkpoint \(j\):

\[
H_j\in\mathbb R^{64\times d}.
\]

Center and globally normalize:

\[
C=I-\frac1{64}\mathbf1\mathbf1^\top,
\qquad
X_j=\frac{\sqrt{64}\,CH_j}{\|CH_j\|_F+10^{-12}}.
\]

Then:

\[
G_j=X_jX_j^\top,
\]

\[
A_j=(X_{j+1}-X_j)X_j^\top.
\]

Use ridge:

\[
\lambda_j=10^{-3}\frac{\operatorname{tr}(G_j)}{63}.
\]

With \(G_j=V\Lambda V^\top\):

\[
G_{j,\lambda}^{-1/2}
=C\,V(\Lambda+\lambda_jI)^{-1/2}V^\top C.
\]

Normalized generator:

\[
\mathcal R_j
=G_{j,\lambda}^{-1/2}A_jG_{j,\lambda}^{-1/2}.
\]

Skew component:

\[
\Omega_j=\frac12(\mathcal R_j-\mathcal R_j^\top).
\]

The raw candidate stores full \(\mathcal R_j\). Store \(\Omega_j\) as a declared diagnostic and later ablation—not as another Stage A candidate.

Use float32 for matrix formation, eigendecomposition, and reference serialization.

## 8. Observable connection

This is an empirical anchor-space balancing proxy, not yet a classical balanced-truncation theorem.

### Observability matrix

For anchor \(i\), let \(c_i\) be the teacher-predicted class and \(c_i'\) the runner-up. Define:

\[
m_i=z_{i,c_i}-z_{i,c_i'}.
\]

At checkpoint \(j\):

\[
J_j[i,:]=\frac{\partial m_i}{\partial H_j[i,:]}.
\]

Compute this with one exact VJP:

```python
torch.autograd.grad(margins.sum(), checkpoint_states)
```

Because batch examples do not interact, the returned rows are per-example margin gradients.

Define:

\[
S_j=J_jX_j^\top,
\]

\[
W_{o,j}
=\frac{S_j^\top S_j}
{\operatorname{tr}(S_j^\top S_j)+10^{-12}}.
\]

No stochastic JVP probes are needed in Stage A. Exact VJPs are cheaper and less noisy for 12 outputs.

### Controllability matrix

Create four key-independent perturbations per anchor. For perturbation \(k\):

1. Hash `(anchor_hash, k)`.
2. Select one operation position.
3. Replace it with one of the other three operations.
4. Never change the initial state or length.

For perturbed hidden states \(X_j^{(k)}\):

\[
D_j^{(k)}=(X_j^{(k)}-X_j)X_j^\top.
\]

Then:

\[
W_{c,j}
=
\frac{\sum_{k=1}^{4}(D_j^{(k)})^\top D_j^{(k)}}
{\operatorname{tr}\left(
\sum_{k=1}^{4}(D_j^{(k)})^\top D_j^{(k)}
\right)+10^{-12}}.
\]

Perturbation choices are frozen before examining outputs.

### Balanced basis

Add a centered numerical ridge of \(10^{-6}/64\) to \(W_c\) and \(W_o\). Form:

\[
M_j=W_{c,j}^{1/2}W_{o,j}W_{c,j}^{1/2}.
\]

Symmetrize numerically, take the top eight eigenvectors \(V_{j,8}\), and compute:

\[
U_{j,8}
=\operatorname{qr}\left(W_{c,j}^{1/2}V_{j,8}\right).
\]

Fix every column’s sign by requiring its largest-magnitude element to be positive.

The observable artifact is:

\[
\mathcal R_j^{\mathrm{obs}}
=U_{j,8}^\top\mathcal R_jU_{j,8}.
\]

Store, for every bank and transition:

- ordered anchor hashes;
- \(U_{j,8}\);
- \(\mathcal R_j^{\mathrm{obs}}\);
- ridge and depth-clock metadata.

Do not store teacher weights, logits, labels, or hidden states.

A future student installer compares:

\[
U_{j,8}^\top\mathcal R_j^S U_{j,8}
\quad\text{against}\quad
\mathcal R_j^{\mathrm{obs},T}.
\]

Those exact bytes must be used for both student architectures.

## 9. Numerical gates

Extraction passes only if:

- repeat extraction differs by at most \(10^{-6}\);
- all values are finite;
- centered \(G_j\) has numerical rank ≥48;
- ridged condition number ≤\(10^6\);
- \(W_c\) and \(W_o\) each have numerical rank ≥8;
- \(\|U^\top U-I\|_F\le10^{-5}\);
- serialization round-trip is float32-exact;
- artifact SHA-256 is stable across two extractions.

Record:

- examples and tokens per second;
- median and p95 step time;
- steps and wall time to capacity;
- peak GPU memory;
- teacher forward-extraction time;
- VJP time;
- perturbation-forward time;
- eigendecomposition time;
- raw and observable artifact bytes.

Stage B has 144 confirmation runs. If measured full-run time is \(t\) minutes:

\[
\text{student GPU-hours}=\frac{144t}{60}.
\]

Stage B launches only if the measured projection—including extraction, evaluation, and a 25% retry reserve—is ≤30 GPU-hours.

## 10. Stage A outputs

```text
results/geometry_admission/stage_a/config.json
results/geometry_admission/stage_a/config.sha256
results/geometry_admission/stage_a/development_key.json
results/geometry_admission/stage_a/anchor_manifest.json
results/geometry_admission/stage_a/capacity_runs.jsonl
results/geometry_admission/stage_a/capacity_summary.json
results/geometry_admission/stage_a/raw_trace_manifest.json
results/geometry_admission/stage_a/observable_trace_manifest.json
results/geometry_admission/stage_a/numerical_audit.json
results/geometry_admission/stage_a/timing_budget.json
```

## Launch gate

Proceed to the candidate screen only if:

1. Teacher passes.
2. Transformer passes its three-seed capacity gate.
3. Both extraction methods pass every numerical gate.
4. Projected sealed confirmation cost is ≤30 GPU-hours.

Authorize GRU transfer confirmation only if:

1. GRU passes its capacity gate.
2. No learned or outcome-tuned adapter is needed.
3. Transformer transfer later passes the full controlled-channel gate.

I attempted to save this as `research/OPEN_CAPABILITY_FILE_GEOMETRY_ADMISSION_STAGE_A_2026_07_25.md`, but the workspace rejected new-file writes despite its write ACL. No file was created or modified. The specification above is complete enough to implement directly; the first implementation task is the automaton generator plus the seven full-supervision capacity runs.

<oai-mem-citation>
<citation_entries>
MEMORY.md:1322-1324|note=[prior AMCL work had validation infrastructure but no empirical execution]
MEMORY.md:1290-1292|note=[historical CTI AMCL scope was rechecked against the live repository]
</citation_entries>
<rollout_ids>
019ea8d0-f23d-7820-8fd2-f905499ec65b
</rollout_ids>
</oai-mem-citation>