# Pre-Registration Addendum: Expanded Model Suite

## Date: 2026-03-03
## Status: LOCKED before first forward pass
## Relation: Extends CGF_GENERATION_PREREGISTRATION.md (original 8-model plan)

---

## Rationale for Expansion

The original pre-registration (2026-03-02) specified 8 models: 5 Pythia + GPT-2 +
Qwen3-0.6B + Llama-3.2-1B. Before any data was collected, we expand to ~25 models
to test a much stronger claim: the generation law is ARCHITECTURE-INDEPENDENT.

Key additions:
- Mamba (pure SSM) shares V=50280 with Pythia — same-V cancellation test
- Falcon-H1 (hybrid Transformer+SSM) tests hybrid architectures
- Granite-4.0 (IBM hybrid) and LFM2.5 (Liquid AI) test novel architectures
- Multiple model sizes within Mamba/Falcon families enable family-internal LOAO

---

## Expanded Model Suite

### Tier 1: Fixed-V Group (V ~ 50280, n=11)

All use GPT-NeoX tokenizer (V=50280) or near-identical (GPT-2, V=50257).
Beta * log(V-1) is constant — absorbed into intercept. Pure alpha test.

| # | Model | Params | V | Architecture | HF ID |
|---|-------|--------|---|--------------|-------|
| 1 | Pythia-160M | 160M | 50280 | Transformer | EleutherAI/pythia-160m |
| 2 | Pythia-410M | 410M | 50280 | Transformer | EleutherAI/pythia-410m |
| 3 | Pythia-1B | 1B | 50280 | Transformer | EleutherAI/pythia-1b |
| 4 | Pythia-1.4B | 1.4B | 50280 | Transformer | EleutherAI/pythia-1.4b |
| 5 | Pythia-2.8B | 2.8B | 50280 | Transformer | EleutherAI/pythia-2.8b |
| 6 | Mamba-130M | 130M | 50280 | SSM | state-spaces/mamba-130m |
| 7 | Mamba-370M | 370M | 50280 | SSM | state-spaces/mamba-370m |
| 8 | Mamba-790M | 790M | 50280 | SSM | state-spaces/mamba-790m |
| 9 | Mamba-1.4B | 1.4B | 50280 | SSM | state-spaces/mamba-1.4b |
| 10 | Mamba-2.8B | 2.8B | 50280 | SSM | state-spaces/mamba-2.8b |
| 11 | GPT-2 | 124M | 50257 | Transformer | openai-community/gpt2 |

### Tier 2: Cross-Architecture Group (diverse V, n=10+)

| # | Model | Params | V | Architecture | HF ID |
|---|-------|--------|---|--------------|-------|
| 12 | Qwen3-0.6B | 600M | 151936 | Transformer | Qwen/Qwen3-0.6B |
| 13 | Qwen3-1.7B | 1.7B | 151936 | Transformer | Qwen/Qwen3-1.7B |
| 14 | Qwen3-4B | 4B | 151936 | Transformer | Qwen/Qwen3-4B |
| 15 | Llama-3.2-3B | 3B | 128256 | Transformer | meta-llama/Llama-3.2-3B |
| 16 | Falcon-H1-0.5B | 500M | ~130048 | Hybrid | tiiuae/Falcon-H1-0.5B-Base |
| 17 | Falcon-H1-1.5B | 1.5B | ~130048 | Hybrid | tiiuae/Falcon-H1-1.5B-Base |
| 18 | Granite-4.0-Tiny | ~1B | 32000 | Hybrid | ibm-granite/granite-4.0-tiny |
| 19 | LFM2.5-1.2B | 1.2B | 65536 | Liquid | LiquidAI/LFM2.5-1.2B-Base |
| 20 | SmolLM2-360M | 360M | TBD | Transformer | HuggingFaceTB/SmolLM2-360M |
| 21 | Mistral-7B | 7B | 32000 | Transformer | mistralai/Mistral-7B-v0.3 |

### Tier 3: Stretch (if compute allows, n=4)

| # | Model | Params | V | Architecture |
|---|-------|--------|---|--------------|
| 22 | Qwen3-8B | 8B | 151936 | Transformer |
| 23 | Falcon-H1-3B | 3B | ~130048 | Hybrid |
| 24 | Codestral-Mamba-7B | 7B | 32768 | SSM |
| 25 | Gemma-3-4B | 4B | ~256000 | Transformer |

---

## Additional Pre-Registered Hypotheses

All original hypotheses (H_gen1 through H_gen9) remain unchanged and are
tested on the expanded suite. The following hypotheses are NEW:

### H_gen10: Architecture-independence within the fixed-V group
**Prediction**: For the n=11 fixed-V group (Pythia + Mamba + GPT-2), a
single linear regression log(PPL) = -alpha*kappa + C fits ALL models
regardless of architecture. Formally: adding an architecture indicator
variable (Transformer vs SSM) does NOT significantly improve the fit
(F-test p > 0.05 for the architecture interaction term).
**Success criterion**: F-test p > 0.05 for architecture * kappa interaction.
**Rationale**: The Architecture-Independence Lemma (Section 3.6 of CGF
framework) predicts that alpha depends only on the Gumbel-race competition,
not on how h(x) is computed. Pythia and Mamba share training data (Pile)
and tokenizer (V=50280), isolating architecture as the only variable.

### H_gen11: SSM models exhibit Neural Collapse
**Prediction**: Mamba models have R^2_NC > 0.2 (weaker threshold than
Transformer R^2_NC > 0.3, since SSMs lack attention for explicit token
comparison that may strengthen NC alignment).
**Success criterion**: R^2_NC > 0.2 for at least 3/5 Mamba models.
**Rationale**: NC has been studied primarily in Transformers. SSMs process
tokens through a recurrent state without explicit pairwise comparison.
If NC still emerges in SSMs, it validates NC as a consequence of the
training objective (minimize CE over V-way classification), not of the
attention mechanism specifically.

### H_gen12: Fixed-V group achieves stronger correlation than full suite
**Prediction**: Pearson r for the n=11 fixed-V group is stronger (more
negative) than for the full n=21+ suite, because the fixed-V group
eliminates vocabulary size as a confound.
**Success criterion**: |r_fixedV| > |r_full| (direction only, no threshold).
**Rationale**: Cross-tokenizer comparison introduces the beta*log(V-1) term
as a confound. Unless beta is correctly estimated, it adds noise.

### H_gen13: Mamba LOAO within-family test
**Prediction**: LOAO within the 5 Mamba models: fit alpha on 4, predict
the 5th. Mean residual < 0.15 nats AND kappa-based residual < mean-baseline
residual in at least 4/5 folds.
**Success criterion**: Same as H_gen4 but for Mamba family.
**Rationale**: If alpha_gen is a family constant for BOTH Pythia and Mamba,
and the two alpha values agree, it's extremely strong evidence for a
universal alpha_gen.

---

## Updated Decision Matrix

| Tier 1 (n=11) r | Architecture test | Interpretation |
|------------------|-------------------|----------------|
| r < -0.80 | H_gen10 PASS | Generation law is universal across architectures |
| r < -0.80 | H_gen10 FAIL | Law works but alpha differs by architecture |
| r > -0.80 | H_gen10 PASS | Law is weak but architecture-independent |
| r > -0.80 | H_gen10 FAIL | Law needs rethinking |

---

## This Document is LOCKED

No modifications after the first forward pass is computed.
All results will be reported regardless of outcome.
