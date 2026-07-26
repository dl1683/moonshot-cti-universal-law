Agree—with one amendment.

Use \(p,r\in[0,7]\) for both the capacity gate and the actual Stage B/C primary evaluation. Do not use \(p,r\in[0,10]\). Also, do not retain anchors through length 32 unquestioningly; move the extraction support into the teacher’s competent region.

### Why \(p,r\in[0,7]\) is correct

The transfer question is whether the artifact provides information about the single withheld edge \(x\). The surrounding \(q^p\) and \(q^r\) operations provide context and prevent the evaluation from collapsing into a direct-edge probe.

With \(p,r\in[0,7]\):

- Every withheld edge is tested in all 64 distinct \((p,r)\) contexts.
- Sequences span lengths 1–15.
- The teacher is demonstrably perfect.
- Success still requires recovering 36 transitions never revealed by the 64 calibration labels.
- Wrong-key, Haar, static-geometry, smoothness, and no-auxiliary controls remain decisive.

Long-horizon composition is not the scientific treatment. Making it a prerequisite confounds transition transfer with Transformer execution depth.

A capacity gate being easy under full supervision is desirable. The hard result is learning withheld edges with restricted labels and beating every control—not passing the capacity preflight.

### Why not \(p,r\in[0,10]\)

That range is scientifically awkward:

- Maximum length 21 lands precisely on the observed phase boundary.
- The triangular length distribution means the easy shorter sequences could hide failures at lengths 20–21 in the aggregate.
- Transfer accuracy would partly measure whether a student crossed an unrelated composition cliff.
- Selecting 21 because it is “near the boundary” adds difficulty without improving identification of withheld-edge transfer.

There is no prize for placing the primary endpoint on the edge of model failure.

### Required anchor amendment

I would not keep anchors at 8–32 after this result.

The observable artifact uses the teacher’s predicted top-vs-runner margin. Above the capacity boundary, many VJPs therefore describe erroneous decisions. That may still contain key information, but it mixes the desired signal with a known failure regime.

Use:

- Training lengths: 1–32, unchanged.
- Primary transfer family: \(p,r\in[0,7]\), lengths 1–15.
- Extraction anchors: provisionally 8–20.
- Long mixed 21–32: diagnostic only.
- Stress 33–64: positional-OOD diagnostic only.

Before extraction, evaluate the exact 2,048 anchors and their perturbations. Require:

- ≥95% accuracy on anchors.
- ≥95% accuracy on perturbations.
- All existing rank, conditioning, observability, and reproducibility gates.

If 8–20 fails this exact audit, use 8–16. Do not select individual correctly answered anchors; the range must remain key-independent.

This preserves substantially deeper trajectories than the primary transfer endpoint while keeping the artifact grounded in competent computation.

### Final thresholds

Teacher:

- ≥99.5% on lengths 1–16.
- ≥99.5% on the \(p,r\in[0,7]\) target family.
- ≥95% on the frozen anchor and perturbation sets.
- 48/48 direct edges.
- All extraction numerical gates.

Students under full supervision:

- At least two of three seeds ≥95% on lengths 1–16.
- The same seeds ≥95% on the \(p,r\in[0,7]\) target family.
- 48/48 direct edges.
- No seed below 90% on either gated distribution.

### Honest protocol handling

Record the 91.09% result as:

> Long-target diagnostic failure for \(p,r\in[0,15]\), with a sharp composition boundary above length 20.

Then create a new R9 precommit before Stage B outcomes or sealed keys. Do not retroactively describe the old distribution as passed.

So the final decision is:

> Adopt \(p,r\in[0,7]\) for capacity and primary transfer evaluation; retain length-32 training; use competent-but-deeper anchors, initially 8–20; keep all longer-horizon results diagnostic.

That isolates the withheld-edge question while preserving the strongest defensible version of the rich-representation hypothesis.

