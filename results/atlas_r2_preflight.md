# Atlas R2 Preflight Gate

**Date:** 2026-07-27
**Status:** PARTIAL PASS (W-C1 source at risk)
**Binding protocol:** `results/codex_scale_inversion_atlas_design_gate_r2.md`

## 1. Workload Source Verification

### W-D1 / W-C2: SWE-bench-Live MultiLang
- **Source:** [microsoft/SWE-bench-Live](https://github.com/microsoft/SWE-bench-Live)
- **HuggingFace:** [SWE-bench-Live/SWE-bench-Live](https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live)
- **Task count:** 743 tasks (as of 2026-05-16), 6 languages (C/C++, C#, Java, TS/JS, Go, Rust), 381 repos
- **License:** TBD (verify from repo)
- **Containerized:** Yes, Linux container environments
- **Monthly updates:** Yes (W-C2 temporal confirmation depends on this)
- **Status:** AVAILABLE

### W-D2: Apple MKQA
- **Source:** [apple/ml-mkqa](https://github.com/apple/ml-mkqa)
- **HuggingFace:** [apple/mkqa](https://huggingface.co/datasets/apple/mkqa)
- **Task count:** 10,000 QA pairs x 26 languages = 260,000 total
- **License:** CC BY-SA 3.0
- **Format:** JSONL (mkqa.jsonl.gz)
- **Status:** AVAILABLE

### W-D3: PolicyBench US
- **Source:** [PolicyEngine/policybench](https://github.com/PolicyEngine/policybench)
- **Website:** [policybench.org](https://policybench.org/)
- **Task count:** 100 households, 18 scored outputs, 1,984 model-output targets
- **License:** Unlicense (public domain)
- **Install:** `pip install policybench`
- **Reference outputs:** Generated via `policybench reference-outputs -n 100 --seed 42`
- **Status:** AVAILABLE

### W-C1: RealClawBench
- **Paper:** [arxiv 2606.03889](https://arxiv.org/abs/2606.03889) (Jun 5, 2026)
- **Task count:** 281 executable tasks (protocol uses 72 hash-selected)
- **License:** CC BY 4.0
- **Containerized:** Yes (reconstructed execution environments, deterministic verifiers)
- **Public download:** **NOT CONFIRMED.** Anonymous review link only (anonymous.4open.science/r/real-claw-bench-582B). HuggingFace release issue (openclaw/openclaw#91465) closed without visible resolution. No public GitHub or HF dataset found.
- **Status:** AT RISK

**W-C1 risk assessment:** The R2 protocol explicitly handles this case: "If W-C1 fails reproducibility before outcomes are viewed, W-C2 remains the prospective confirmation. The experiment is not killed. However, without W-C1 it has no untouched cross-family confirmation and the scientific score is capped below 7/10." W-C2 (future SWE-bench-Live batch) preserves the experiment at reduced scientific strength.

**Action required:** Monitor RealClawBench release status. Check anonymous link accessibility. Contact openclaw maintainers if needed. If W-C1 unavailable at P1 start, proceed with W-C2 only and cap ceiling at 6/10.

## 2. Local Model Checkpoint Verification

### Qwen3 Family
| Size | HuggingFace ID | GGUF Available | License |
|------|---------------|----------------|---------|
| 0.6B | Qwen/Qwen3-0.6B, Qwen/Qwen3-0.6B-GGUF | Yes (official + unsloth + bartowski) | Apache 2.0 |
| 4B | Qwen/Qwen3-4B, Qwen/Qwen3-4B-GGUF | Yes | Apache 2.0 |
| 14B | Qwen/Qwen3-14B, Qwen/Qwen3-14B-GGUF | Yes (official + unsloth + bartowski) | Apache 2.0 |

### Gemma 3 Family
| Size | HuggingFace ID | GGUF Available | License |
|------|---------------|----------------|---------|
| 1B | google/gemma-3-1b-it | TBD | Gemma license |
| 4B | google/gemma-3-4b-it | TBD | Gemma license |
| 12B | google/gemma-3-12b-it | TBD | Gemma license |

Note: Gemma 4 exists (Apr 2026) but R2 freezes on Gemma 3 per design gate.

### Falcon-H1 Family
| Size | HuggingFace ID | GGUF Available | License |
|------|---------------|----------------|---------|
| 0.5B | tiiuae/Falcon-H1-0.5B-Instruct | TBD | Apache 2.0 |
| 3B | tiiuae/Falcon-H1-3B-Instruct | Yes (tiiuae/Falcon-H1-3B-Instruct-GGUF) | Apache 2.0 |
| 7B | tiiuae/Falcon-H1-7B-Instruct | TBD | Apache 2.0 |

**All 9 checkpoints: AVAILABLE on HuggingFace.**

## 3. API Provider Verification

| Role | System | Endpoint Status | Pricing Verified |
|------|--------|----------------|-----------------|
| Frontier | GPT-5.6 Sol | Verified (developers.openai.com/api/docs/models) | $5/M in, $30/M out |
| Value | GPT-5.6 Luna | Verified | $1/M in, $6/M out |
| Frontier | Claude Fable 5 | Verified (platform.claude.com) | $10/M in, $50/M out |
| Value | Claude Sonnet 5 | Verified | $3/M in, $15/M out |
| Frontier | Gemini 3.1 Pro Preview | Verified (ai.google.dev) | $2/M in, $12/M out (mutable preview ID) |
| Value | Gemini 3.1 Flash-Lite | Verified | $0.25/M in, $1.50/M out |

**API budget estimate:** ~$485 worst-case token bill. Binding gate: $1,200.
**Gemini drift risk:** R2 protocol requires 7-day completion window + 5% drift audit for mutable preview IDs.

## 4. Preflight Gate Decision

| Gate | Status | Notes |
|------|--------|-------|
| W-D1 source | PASS | 743 tasks, containerized, monthly updates |
| W-D2 source | PASS | 260K QA pairs, CC BY-SA 3.0 |
| W-D3 source | PASS | pip installable, public domain |
| W-C1 source | AT RISK | No confirmed public download |
| W-C2 source | PASS (contingent) | Depends on post-freeze SWE-bench-Live release |
| 9 local checkpoints | PASS | All on HuggingFace |
| 6 API systems | PASS (pricing verified) | Keys not yet tested |
| Energy measurement | NOT STARTED | NVML 10Hz sampling needs calibration |
| Harness implementation | NOT STARTED | Runner, scorer, verifier not yet built |

**Overall: PARTIAL PASS.** Proceed with P0 implementation (harness, energy calibration, model download smoke tests). W-C1 risk is documented and handled by protocol. Full P0 gate requires harness smoke on all 9 checkpoints before authorizing 360-hour budget.
