## Verdict

The memorization threat does not kill the narrow claim that teacher-specific information crossed the channel. It does threaten the much stronger claim that GAT demonstrates a compact, reusable “capability file.”

Capacity is not identifiability: without a key-dependent artifact, the student has no information with which to choose the three withheld random permutations. But the artifact is so vastly larger than the withheld task entropy that a Stage C pass could amount to an elaborate encoding of an 87-bit lookup table—not transferred computation.

### 1. How many bits must the student memorize?

The automaton contains exactly \(12\times4=48\) one-step edges.

The stated \(12^4=20{,}736\) is the number of possible four-output rows for one state if the four operators were arbitrary functions. A complete arbitrary table would have

\[
(12^4)^{12}=12^{48}
\]

possibilities, or

\[
48\log_2 12=172.08\text{ bits}.
\]

But GAT restricts every operator to a permutation, so the exact key entropy is

\[
H(K)=4\log_2(12!)=115.34\text{ bits}.
\]

The calibration set is even more revealing than “64 examples” suggests. It includes all 12 length-one examples for one operator, so it reveals that entire permutation exactly. The remaining uncertainty is therefore

\[
H(K\mid C)=3\log_2(12!)=86.51\text{ bits}.
\]

A simple fixed-width representation of the 36 withheld entries needs \(36\times4=144\) bits; optimal permutation-aware coding needs about 86.5 bits.

The student has 1,921,772 parameters. At fp32 storage that is 61,496,704 nominal bits, or 7.33 MiB—about 711,000 model bits per remaining task bit. Its representational capacity is not in doubt.

But it cannot “find the table anyway.” For any withheld operator \(x\), conditioned on calibration data and any key-independent regularizer,

\[
P(\hat\pi_x(s)=\pi_x(s))=\frac1{12}.
\]

The withheld examples are deliberately \(q^p\,x\,q^r\): because \(q\) is already known, their labels are bijective transforms of a single unknown withheld edge. A teacher-independent optimizer prior remains at chance in expectation, regardless of model size or training steps.

Also, the installer does not use batch size 512. It repeatedly trains on the same full batch of 64 calibration examples and one 64-anchor bank for 5,000 steps—320,000 repeated supervised exposures plus 320,000 anchor exposures ([installer](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:329>)).

### 2. Artifact information capacity

There are 32 banks, 6 depth transitions, and 64 anchors per bank.

| Artifact | Float32 values used by installer | Binary size | Nominal bits / 86.5 withheld bits |
|---|---:|---:|---:|
| Raw \(R\) | \(32\cdot6\cdot64^2=786{,}432\) | 3 MiB | 290,913× |
| Observable \(R_{\rm obs}\) alone | \(32\cdot6\cdot8^2=12{,}288\) | 48 KiB | 4,546× |
| Required \(U\) bases | \(32\cdot6\cdot64\cdot8=98{,}304\) | 384 KiB | — |
| Complete observable artifact | 110,592 | 432 KiB | 40,910× |

The serialized raw Stage A files additionally store \(\Omega\), doubling the raw matrix payload to 1,572,864 floats or 6 MiB, although the installer discards \(\Omega\) ([serialization](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_extraction.py:370>)). JSON files will be larger still.

These are upper bounds on information, not measured entropy—the matrices are highly correlated. Nevertheless, the channel has overwhelming capacity to encode an 86.5-bit table.

Thus a positive result has three possible interpretations:

1. All structured targets help similarly: generic regularization.
2. Only the correct key works: teacher-specific information transfer, possibly as a bloated table codec.
3. Correct key works under a severe bitrate choke and generalizes compositionally: evidence toward a portable program.

Current GAT separates 1 from 2 imperfectly. It does not separate 2 from 3.

### 3. The decisive experiment

Use paired keys differing in only one withheld permutation transposition.

For each of at least 16 base keys:

- Construct \(K_A\) and \(K_B\) identically except that two outputs of one withheld operator are swapped.
- Keep calibration examples, anchors, student initialization, optimizer, and training schedule identical.
- Train with \(R_A\), \(R_B\), no auxiliary, generic smoothness/Jacobian regularization, and a spectrum-, norm-, depth-autocorrelation-, and gradient-trajectory-matched scrambled target.
- Cross correct/partner targets with early-only, late-only, and full-training application.
- Evaluate the two changed edges separately from the 34 unchanged withheld edges.
- Lesion the predeclared geometry-bearing student subspace at evaluation and compare with an equal-norm orthogonal lesion.
- Blind all withheld labels and probe scores until every run is complete.

The decisive signature is signed selectivity: changing only \(R_A\rightarrow R_B\) must flip the two altered transitions toward the corresponding teacher while leaving unchanged transitions stable. The effect must survive late injection and be selectively destroyed by the geometry-bearing lesion.

A generic regularizer cannot know which two arbitrary outputs were swapped. Negative-transfer incompatibility also cannot explain a precise, bidirectional flip confined to those edges.

Add a frozen learned codec trained on development automata and sweep the available artifact budget—say 32, 64, 96, 128, 256, and 1,024 bits. If the effect requires hundreds of kilobytes to transmit 86.5 bits, it is not compelling compression.

#### Are the four QL6 tests implemented?

Only partially.

| QL6 discriminator | Live GAT implementation |
|---|---|
| Teacher identity | Partial. Correct-key versus wrong-key artifacts across eight keys is present, but there is no minimally paired key intervention or signed edge-specific identity statistic. |
| Causal use | No. The centroid probe is correlational; no matched-subspace lesion exists. |
| Timing | No. Auxiliary loss is applied throughout all 5,000 steps. |
| Matched regularizers | Partial. No-auxiliary, smoothness, static \(G\), wrong-key, and Haar arms exist, with initial gradient-ratio coefficient matching. There is no Jacobian/update-norm suite or gradient-trajectory matching. |

Two further problems matter:

- Haar conjugation is invertible, and its \(Q\) is deterministically recoverable from public code. It preserves all Shannon information about the key; it tests coordinate alignment, not an information-free null. For the observable arm, Haar rotates \(R_{\rm obs}\) while retaining the correct teacher-derived \(U\) basis ([control construction](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_stage_c.py:196>)).
- Direct withheld probes are evaluated every 250 steps and exposed in logs, although their labels do not enter gradients and the final checkpoint is fixed ([probe evaluation](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-cti-universal-law/src/cti_geometry_admission_installer.py:386>)). A sealed confirmation should reveal them only after completion.

Stage A itself explicitly does not test transfer, and only Stage A artifacts currently exist. Stage B/C are implemented but unexecuted.

### 4. If it is regularization

GAT remains useful, but the narrative changes completely:

> Trace-shaped geometric losses are optimization priors that can improve learning from extremely repetitive supervision.

That could still be publishable engineering if it beats simpler orthogonality, Jacobian, update-norm, and smoothness regularizers under equal compute. It would not support “download a skill,” a capability file, or teacher knowledge transfer.

If correct-key specificity appears but only through this enormous channel, the honest narrative is narrower:

> Teacher dynamics provide a high-bandwidth supervision codec for a finite transition table.

That is genuine weak information transfer, but not yet a portable computation.

### 5. Mission attack

A Stage C pass would not yet make intelligence cheap or democratic.

The remaining key is theoretically about 11 bytes. The raw artifact is 3 MiB, the observable artifact 432 KiB, and installation takes 5,000 gradient steps in a 1.9M-parameter network. Directly shipping the transition table would be thousands of times smaller, exact, instantaneous, and architecture-independent.

“10× teacher compression” is therefore the wrong denominator. On this task, the proper competitors are:

- an 11–18 byte key/table;
- direct labels for 36 edges;
- a tiny symbolic interpreter;
- a direct decoder from the artifact.

GAT serves the mission as a ruthless admission gate: it can show that a proposed channel contains teacher-specific information before language-scale spending. Celebrating the synthetic pass itself as democratized intelligence would be process drift.

### 6. Three hypotheses after memorization

1. **Rate–distortion hypothesis:** withheld accuracy is governed by recoverable key information \(I(K;Z\mid C)\), not \(R\)-loss. Train and freeze a codec across many development automata, then measure recovered transitions per transmitted bit on sealed keys. Compare directly with compressed transition-table bits.

2. **Rule-versus-table phase transition:** geometry transfers compact algorithms only when task structure permits extrapolation. Scale to 128–256 states and structured generators, cap the artifact below random-table entropy, and withhold entire states, operators, and composition depths. Success on unseen structure would distinguish algorithm transfer from table storage.

3. **Amortized installer hypothesis:** a real capability file should work through one frozen installer across unseen keys and substrates, without 5,000 per-key optimization steps. Train the installer only on development automata, freeze it, and test one-shot or few-step installation into Transformer and GRU students, including trace splicing. Failure means the “file” is merely bespoke training supervision.

The most honest bottom line: memorization does not refute that information can cross \(R\). It does refute treating that fact alone as evidence that intelligence has been compressed into geometry.

