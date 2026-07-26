# Question Loop 04 — Geometry-Aware Distillation Baseline Frontier

**Date:** 2026-07-25
**Decision:** DG-0 does not yet have a paradigm-level empirical target. Its current `dynamic > max(static, control) by 3 points` gate can establish a useful auxiliary loss, but it cannot distinguish the proposed update-to-state object from the strongest existing dynamic, relational, or on-policy distillation methods.

## Finding: the frontier is fragmented

There is no single July 2026 baseline that simultaneously has all four properties DG-0 wants: hidden-state geometry, cross-layer dynamics, math reasoning, and large cross-scale compression. The honest strongest-baseline answer is therefore a **Pareto set**, not one paper.

| Baseline | What is transferred | Model/task evidence | Compression evidence | Why DG-0 must compare |
|---|---|---|---|---|
| [FDD, ACL 2025](https://aclanthology.org/2025.acl-long.1125/) | The layerwise trajectory of LM-head predictions plus cosine alignment of adjacent-layer derivatives | Instruction following; no GSM8K | GPT-2 1.5B→0.1B, **15×**: 22.16 vs teacher 23.09 average; LLaMA-2 13B→TinyLLaMA 1B, **13×**: 27.95 vs 28.26 | This is the closest published claim that *first-order depth dynamics* add information beyond endpoint/logit matching. Calling \(A_r\) “dynamic” is not enough. |
| [MTA, ACL 2026](https://arxiv.org/abs/2605.01374) | Hidden-state cosine alignment plus dependency-derived word/phrase relational geometry across selected layers; composable with FDD | Instruction following; no math | GPT-2 1.5B→120M, **12.5×**: FDD 19.48, FDD+MTA 20.50; Qwen1.5 1.8B→0.5B, **3.6×**: 23.39→24.73 over DistiLLM2; OPT 6.7B→1.3B, **5.15×** | This is the strongest recent relational hidden-path comparator found. It also exposes a shared-tokenization/parsing dependence that DG-0 could genuinely avoid. |
| [Bhattarai et al., 2025](https://arxiv.org/abs/2509.25253) | Procrustes-aligned features and feature Gram matrices | BERT plus OPT instruction following; no math | OPT-13B→1.3B, **10×**: Gram 11.07 and Procrustes 11.11 vs MiniLLM 10.83 average | This is the clean static geometry baseline. Its gains are real but small, and its feature-only failure is a critical negative result. |
| [LoRi, 2026](https://arxiv.org/abs/2606.05315) | Low-rank teacher rationale trajectories; first and second moments in a shared subspace; answer-boundary anchor | GSM8K, GSM8K-Hard, SVAMP | **1× parameter compression**: teacher and student use the same base size; reasoning is compressed to five latent steps, giving 5.1–6.9× inference speedup | This is the closest geometry-aware *math-reasoning* result. On Qwen2.5-0.5B it reaches 50.0 GSM8K vs KAVA 46.9, but the answer anchor is worth 10.2 points. DG-0 must separate global geometry from anchor/logit transfer. |
| [TSD-KD, ICLR 2026](https://arxiv.org/abs/2603.13260) | Token-selective on-policy forward/reverse KL; no hidden geometry | GSM8K, MATH, GSM-Plus, MBPP, IFEval, MMLU-STEM | Qwen3-8B→1.7B, **4.7×**: GSM8K 68.7 vs GKD 67.2 and undistilled 66.3; MATH 28.0 vs 22.8 and 21.0 | This is the strongest accepted directly comparable Qwen3 cross-scale reasoning baseline found. Geometry must beat strong output-space selection, not weak offline KD. |
| [RG-OPD, July 2026](https://arxiv.org/abs/2607.04037) | Verifier-gated on-policy reverse KL; no hidden geometry | Six reasoning/instruction tasks with Qwen2.5 | Qwen2.5-14B→1.5B, **9.3×**; reports +2.9 average over reverse-KL KD, though its reproduced TSD result differs sharply from the original | This is the newest strong on-policy control found. It prevents attributing gains from better sample selection or rejection to geometry. |

Two nearby results sharpen the interpretation. [Yu et al.](https://arxiv.org/abs/2502.04499) show that reverse and out-of-order layer matching can rival forward matching, including a Qwen3-8B teacher compressed to a 10-layer student. This is direct evidence against “ordered layers transfer an ordered program.” [Guigon et al.](https://arxiv.org/abs/2605.11513) find that hidden-layer distillation on Gemma3 does not consistently beat logit KD under compute matching; its most stable benefit looks like a warm start. Any DG-0 gain must survive both attacks.

“SOTA” is protocol-dependent. TSD-KD is the cleanest published Qwen3 8B→1.7B cross-scale comparison on standard reasoning tasks. RG-OPD is newer and stronger on its Qwen2.5 protocol. For competition math, [TRD](https://arxiv.org/abs/2606.08432) tests output-trajectory correction on Qwen3-8B→4B, but that is only 2× compression and does not align hidden geometry. FDD/MTA dominate the relevant dynamic/relational design space but do not test math. LoRi tests math but does not compress parameters. No paper found occupies the complete DG-0 quadrant.

## What would be genuinely new?

The proposed \(A_r=(H_{r+1}-H_r)H_r^\top\), or its whitened \(R_r\), is not renamed FDD if it wins for the right reason:

- FDD aligns **vocabulary-space prediction derivatives**. It does not identify directed transport among examples in the hidden sample geometry.
- MTA and Procrustes/Gram losses align **within-layer relations**. They do not observe the cross-layer orientation carried by the skew part of \(R_r\).
- LoRi aligns **aggregate low-rank moments** over a rationale trajectory. It does not retain the sample-pair directed update-to-state relation.

That distinction is mathematical, not yet empirical. A result that merely shows full \(A_r\) beating one terminal Gram arm on GSM8K would reproduce the generic lesson “richer hidden supervision helps.” It would not establish the proposed object.

## Required DG-0 empirical result

For the Qwen3-4B→0.6B pilot, a **clear exceedance** should mean all of the following:

1. At identical data, teacher calls, optimizer steps, peak memory, and reported training FLOPs, the full/skew-\(R_r\) arm beats the best of:
   - logit KD and a strong on-policy KD arm;
   - FDD trajectory + derivative;
   - multilayer Gram/Procrustes or MTA-style relational alignment;
   - LoRi-style low-rank first/second-moment alignment adapted to the same traces;
   - static path, strain-only, permuted, and compute-matched controls.
2. The advantage is at least **3 absolute accuracy points with a positive family-bootstrap 95% lower confidence bound** on the frozen in-domain test and on a fresh semantic/complexity-shift test.
3. Relative to the strongest baseline \(B\), incremental teacher-gap closure
   \[
   \frac{q_{\mathrm{DG}}-q_B}{q_T-q_B}
   \]
   is at least **20%**. Report raw accuracy as primary; this ratio is a gate, not a substitute metric.
4. A skew-containing arm contributes at least **1.5 points beyond path-Gram + symmetric strain**, while depth permutation, skew sign-flip, spectrum-matched random skew, and wrong-teacher skew fail. Otherwise the result is generic trajectory regularization.
5. The result reproduces over seeds and a second reasoning family. GSM8K alone is a pipeline smoke test, not a claim surface.

The 4B→0.6B pair is only **6.7×**, below the stated 10× moonshot. Success there can license the next experiment; it cannot satisfy the mission. The paradigm claim requires a later ≥10× teacher/student pair and a genuinely different student substrate, with the same precommitted controls.

## NARRATIVE ATTACK

“DG-0 packages FDD’s depth derivative, MTA’s relational loss, and LoRi’s trajectory covariance into a more elaborate matrix. Its only win comes from more constraints and more teacher signal. Strong on-policy KD gets the same capability without assuming aligned internal programs, and reverse layer matching already showed that the program story is false.”

This attack wins unless the skew-specific, teacher-specific signal beats the full Pareto set under exact resource matching.

## MISSION TEST

Would a small lab obtain capability it could not obtain with logits, generated rationales, verifier filtering, or existing feature alignment at the same teacher-query and compute budget? A 3-point win over a weak static arm does not democratize anything. A reproducible ≥10× result that reduces teacher-query or student-training cost while surviving architecture change would.

## Next-gate specification

Freeze a **baseline-complete DG-0 manifest** before training: exact FDD, relational/Procrustes, LoRi-statistic, strong output-KD, full-\(R\), skew-only, strain-only, and four structured controls; resource-accounting equations; fresh-test generator seeds; and the confidence/teacher-gap rules above. Do not execute the expensive sweep until each comparator can be implemented from the same cached teacher traces or its extra teacher-query cost is explicitly charged.
