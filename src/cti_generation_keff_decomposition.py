#!/usr/bin/env python -u
"""
K_eff Decomposition: CE = margin_deficit + log(K_eff)

Decomposes cross-entropy loss into:
  1. margin_deficit = z_max - z_y  (geometric quality)
  2. log(K_eff) = LSE(z) - z_max  (effective competition)

This decomposition is EXACT (not approximate). It separates the loss into:
- A term that kappa should predict (margin)
- A term that captures context quality (K_eff)

For each model, runs forward pass on WikiText-103 validation and collects
per-position logit statistics.

Theory: Section 3.20.2 of CGF_THEORETICAL_FRAMEWORK.md
"""
import json, time, gc, sys
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_FILE = RESULTS / "cti_generation_keff.json"
PPL_FILE = RESULTS / "cti_generation_ppl.json"
FREQ_KAPPA_FILE = RESULTS / "cti_generation_freq_kappa.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models to test: only those where we have WikiText-103 PPL AND can run inference
# Skip 7B+ (too large for full inference on 24GB GPU with this simple approach)
MODELS = [
    ("pythia-160m",  "EleutherAI/pythia-160m"),
    ("pythia-410m",  "EleutherAI/pythia-410m"),
    ("pythia-1b",    "EleutherAI/pythia-1b"),
    ("pythia-1.4b",  "EleutherAI/pythia-1.4b"),
    ("pythia-2.8b",  "EleutherAI/pythia-2.8b"),
    ("gpt2",         "openai-community/gpt2"),
    ("qwen3-0.6b",   "Qwen/Qwen3-0.6B"),
    ("qwen3-1.7b",   "Qwen/Qwen3-1.7B"),
    ("qwen3-4b",     "Qwen/Qwen3-4B"),
    ("qwen2-0.5b",   "Qwen/Qwen2-0.5B"),
    ("falcon-h1-0.5b", "tiiuae/Falcon-H1-0.5B-Base"),
    ("falcon-h1-1.5b", "tiiuae/Falcon-H1-1.5B-Base"),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M"),
    ("granite-micro", "ibm-granite/granite-4.0-micro"),
]


def compute_keff_stats(model_key, hf_id, max_tokens=20000):
    """Run forward pass and compute K_eff decomposition."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"    Loading model {hf_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    print(f"    Model loaded on {DEVICE}", flush=True)

    print(f"    Loading WikiText-103 validation...", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
    tokens = tokens[:max_tokens]
    n_tokens = len(tokens)
    print(f"    {n_tokens} tokens", flush=True)

    # Process in chunks (context window)
    ctx_len = min(1024, getattr(model.config, 'max_position_embeddings', 2048))
    stride = ctx_len // 2  # 50% overlap

    all_margin_deficits = []
    all_log_keff = []
    all_ce = []
    all_correct = []

    with torch.no_grad():
        for start in range(0, n_tokens - 1, stride):
            end = min(start + ctx_len, n_tokens)
            input_ids = tokens[start:end].unsqueeze(0).to(DEVICE)

            outputs = model(input_ids)
            logits = outputs.logits[0]  # (seq_len, V)

            # For each position (except last), predict next token
            # Only use the non-overlapping portion (except first chunk)
            if start == 0:
                pred_start = 0
            else:
                pred_start = stride // 2  # skip overlap region

            for pos in range(pred_start, logits.shape[0] - 1):
                z = logits[pos].float()  # (V,)
                y = input_ids[0, pos + 1].item()

                z_max = z.max().item()
                z_y = z[y].item()
                lse = torch.logsumexp(z, dim=0).item()

                margin_deficit = z_max - z_y  # >= 0
                log_keff = lse - z_max  # >= 0
                ce = lse - z_y  # = margin_deficit + log_keff

                all_margin_deficits.append(margin_deficit)
                all_log_keff.append(log_keff)
                all_ce.append(ce)
                all_correct.append(1.0 if z_y == z_max else 0.0)

            if end >= n_tokens:
                break

    margin_arr = np.array(all_margin_deficits)
    log_keff_arr = np.array(all_log_keff)
    ce_arr = np.array(all_ce)
    correct_arr = np.array(all_correct)

    # Compute statistics
    result = {
        "model": model_key,
        "hf_id": hf_id,
        "n_positions": len(ce_arr),
        "mean_ce": float(np.mean(ce_arr)),
        "mean_margin_deficit": float(np.mean(margin_arr)),
        "mean_log_keff": float(np.mean(log_keff_arr)),
        "median_margin_deficit": float(np.median(margin_arr)),
        "median_log_keff": float(np.median(log_keff_arr)),
        "std_margin_deficit": float(np.std(margin_arr)),
        "std_log_keff": float(np.std(log_keff_arr)),
        "ppl_from_ce": float(np.exp(np.mean(ce_arr))),
        "mean_keff": float(np.exp(np.mean(log_keff_arr))),
        "median_keff": float(np.exp(np.median(log_keff_arr))),
        "accuracy_top1": float(np.mean(correct_arr)),
        # Percentiles of K_eff
        "keff_q10": float(np.exp(np.percentile(log_keff_arr, 10))),
        "keff_q25": float(np.exp(np.percentile(log_keff_arr, 25))),
        "keff_q50": float(np.exp(np.percentile(log_keff_arr, 50))),
        "keff_q75": float(np.exp(np.percentile(log_keff_arr, 75))),
        "keff_q90": float(np.exp(np.percentile(log_keff_arr, 90))),
        # Fraction of CE from margin vs K_eff
        "fraction_margin": float(np.mean(margin_arr) / np.mean(ce_arr)),
        "fraction_keff": float(np.mean(log_keff_arr) / np.mean(ce_arr)),
    }

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def analyze_results(results):
    """Analyze K_eff decomposition."""
    print("\n" + "=" * 70)
    print("  K_eff DECOMPOSITION ANALYSIS")
    print("=" * 70)

    ok = {k: v for k, v in results.items() if "error" not in v}

    # Summary table
    print(f"\n{'Model':20s} {'CE':>7s} {'Margin':>7s} {'logKeff':>7s} {'%Margin':>8s} {'K_eff':>7s} {'Top1%':>7s}")
    print("-" * 75)
    for k in sorted(ok.keys(), key=lambda x: ok[x].get("mean_ce", 99)):
        v = ok[k]
        print(f"{k:20s} {v['mean_ce']:7.3f} {v['mean_margin_deficit']:7.3f} "
              f"{v['mean_log_keff']:7.3f} {v['fraction_margin']:7.1%} "
              f"{v['mean_keff']:7.1f} {v['accuracy_top1']:7.1%}")

    # Load kappa data for correlation
    if FREQ_KAPPA_FILE.exists():
        with open(FREQ_KAPPA_FILE) as f:
            kappa_data = json.load(f)

        both = [k for k in ok if k in kappa_data and "error" not in kappa_data.get(k, {})]
        if len(both) >= 5:
            print(f"\n--- Kappa vs decomposition components (n={len(both)}) ---")
            for kappa_metric in ["kappa_bar", "kappa_freq_p", "kappa_top1000"]:
                kappas = [kappa_data[k].get(kappa_metric, float('nan')) for k in both]
                valid = [not np.isnan(v) for v in kappas]
                valid_keys = [both[i] for i in range(len(both)) if valid[i]]
                kv = [kappas[i] for i in range(len(kappas)) if valid[i]]

                for ce_comp in ["mean_ce", "mean_margin_deficit", "mean_log_keff"]:
                    cv = [ok[k][ce_comp] for k in valid_keys]
                    r, p = pearsonr(kv, cv)
                    print(f"  r({kappa_metric:15s}, {ce_comp:22s}) = {r:+.3f} (p={p:.4f})")

    return ok


def main():
    print("=" * 70)
    print("  K_eff DECOMPOSITION: CE = margin + log(K_eff)")
    print("  Theory: CGF Section 3.20.2")
    print("=" * 70)

    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    to_run = [(k, hf) for k, hf in MODELS if k not in results]
    if not to_run:
        print("\nAll models computed. Running analysis.")
    else:
        print(f"\n{len(to_run)} models to compute, {len(results)} cached.\n")

        for key, hf_id in to_run:
            print(f"\n--- {key} ---")
            t0 = time.time()
            try:
                result = compute_keff_stats(key, hf_id)
                result["time_s"] = time.time() - t0
                results[key] = result

                print(f"    CE={result['mean_ce']:.3f}, "
                      f"margin={result['mean_margin_deficit']:.3f} ({result['fraction_margin']:.1%}), "
                      f"log(K_eff)={result['mean_log_keff']:.3f} ({result['fraction_keff']:.1%})")
                print(f"    K_eff={result['mean_keff']:.1f}, top1_acc={result['accuracy_top1']:.1%}")

                with open(OUT_FILE, "w") as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                results[key] = {"model": key, "error": str(e)[:300]}
                with open(OUT_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"    ERROR: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    analysis = analyze_results(results)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
