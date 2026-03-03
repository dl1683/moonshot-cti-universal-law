#!/usr/bin/env python -u
"""
CGF Generation Law: Local Equicorrelation Test
===============================================
Tests Theorem 3.8: alpha_gen is governed by LOCAL rho among top-K
competing tokens, not GLOBAL rho across all V tokens.

Prediction (from alpha_gen = 2.077):
  rho_local ~ 0.70

Test procedure:
1. For each model, run forward pass on ~5K tokens from WikiText-103
2. At each position, find the top-K tokens by probability
3. Compute mean pairwise cosine among these K rows of W_U
4. Average across all positions -> rho_local
5. Compare to predicted value of 0.70

Also computes:
- rho_local as function of K (K=5, 10, 20, 50, 100)
- rho_local std across positions (stability test)
- Comparison with rho_global (should be much higher than global)
"""

import json
import time
import gc
import numpy as np
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models for local rho test (those that can do forward passes)
LOCAL_RHO_MODELS = [
    ("pythia-160m", "EleutherAI/pythia-160m", "Pythia-160M"),
    ("pythia-410m", "EleutherAI/pythia-410m", "Pythia-410M"),
    ("pythia-1b", "EleutherAI/pythia-1b", "Pythia-1B"),
    ("pythia-1.4b", "EleutherAI/pythia-1.4b", "Pythia-1.4B"),
    ("pythia-2.8b", "EleutherAI/pythia-2.8b", "Pythia-2.8B"),
    ("gpt2", "openai-community/gpt2", "GPT-2"),
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    ("qwen3-1.7b", "Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", "SmolLM2-360M"),
]

K_VALUES = [5, 10, 20, 50, 100]


def compute_local_rho(model, tokenizer, model_key, n_tokens=5000):
    """Compute local equicorrelation among top-K tokens at each position."""
    from datasets import load_dataset

    print(f"  Computing local rho for {model_key}...", flush=True)

    # Load WikiText-103 validation
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [x["text"] for x in wiki if x["text"].strip()]

    # Get W_U (normalized rows)
    W_U = None
    for attr_path in [("lm_head", "weight"), ("embed_out", "weight")]:
        obj = model
        try:
            for a in attr_path:
                obj = getattr(obj, a)
            W_U = obj.detach().float()
            break
        except AttributeError:
            continue
    if W_U is None:
        try:
            W_U = model.model.embed_tokens.weight.detach().float()
        except Exception:
            raise ValueError("Cannot find W_U")

    V, d = W_U.shape
    print(f"    W_U shape: {V} x {d}", flush=True)

    # Normalize W_U rows
    norms = torch.norm(W_U, dim=1, keepdim=True).clamp(min=1e-10)
    W_U_normed = (W_U / norms).to(DEVICE)

    # Compute global rho for comparison
    n_sample = min(V, 5000)
    idx_sample = np.random.choice(V, n_sample, replace=False)
    W_sub = W_U_normed[idx_sample]
    cos_mat = W_sub @ W_sub.T
    mask = ~torch.eye(n_sample, dtype=torch.bool, device=DEVICE)
    rho_global = float(cos_mat[mask].mean().cpu())

    # Collect local rho measurements
    # For each K in K_VALUES, collect per-position rho_local
    local_rho_by_K = {K: [] for K in K_VALUES}
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            if total_tokens >= n_tokens:
                break
            if not text.strip():
                continue

            enc = tokenizer(text, truncation=True, max_length=512,
                            return_tensors="pt").to(DEVICE)
            if enc["input_ids"].shape[1] < 2:
                continue

            outputs = model(**enc)
            logits = outputs.logits[0, :-1, :]  # (seq_len-1, V)
            n_pos = logits.shape[0]

            for K in K_VALUES:
                if K > V:
                    continue
                # Get top-K token indices at each position
                _, top_k_idx = torch.topk(logits, K, dim=1)  # (n_pos, K)

                # Batch compute local cosines
                for pos in range(n_pos):
                    top_tokens = top_k_idx[pos]  # (K,)
                    W_top = W_U_normed[top_tokens]  # (K, d)
                    local_cos = W_top @ W_top.T  # (K, K)
                    local_mask = ~torch.eye(K, dtype=torch.bool, device=DEVICE)
                    rho_pos = float(local_cos[local_mask].mean().cpu())
                    local_rho_by_K[K].append(rho_pos)

            total_tokens += n_pos

    print(f"    Collected {total_tokens} positions", flush=True)

    # Compute summary statistics
    result = {
        "model": model_key,
        "V": int(V),
        "d_model": int(d),
        "n_positions": total_tokens,
        "rho_global": rho_global,
    }

    predicted_rho = 0.705  # from alpha_gen = 2.077
    print(f"    rho_global = {rho_global:.4f}")
    print(f"    Predicted rho_local = {predicted_rho:.4f}")
    print(f"    {'K':<6s} {'rho_local':<12s} {'std':<12s} {'ratio_vs_global':<18s}")
    print(f"    {'-'*48}")

    for K in K_VALUES:
        if K > V or not local_rho_by_K[K]:
            continue
        vals = np.array(local_rho_by_K[K])
        mean_rho = float(vals.mean())
        std_rho = float(vals.std())
        ratio = mean_rho / rho_global if abs(rho_global) > 0.001 else float("inf")
        print(f"    K={K:<4d} {mean_rho:<12.4f} {std_rho:<12.4f} {ratio:<18.2f}")

        result[f"rho_local_K{K}"] = mean_rho
        result[f"rho_local_K{K}_std"] = std_rho
        result[f"rho_local_K{K}_n"] = len(vals)

    return result


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 72)
    print("  CGF GENERATION LAW: LOCAL EQUICORRELATION TEST")
    print("  Theorem 3.8: alpha_gen governed by rho_local ~ 0.70")
    print("=" * 72)

    cache_path = RESULTS_DIR / "cti_generation_local_rho.json"
    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} cached results", flush=True)
    else:
        results = {}

    for model_key, hf_id, name in LOCAL_RHO_MODELS:
        if model_key in results and "rho_local_K10" in results[model_key]:
            print(f"  Skipping {name} (cached: rho_local_K10="
                  f"{results[model_key]['rho_local_K10']:.4f})")
            continue

        print(f"\n--- {name} ({hf_id}) ---", flush=True)
        t0 = time.time()

        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=torch.float16, trust_remote_code=True
            ).to(DEVICE)
            model.eval()

            result = compute_local_rho(model, tokenizer, model_key)
            result["name"] = name
            result["hf_id"] = hf_id
            result["time_s"] = time.time() - t0
            results[model_key] = result

            del model
            gc.collect()
            torch.cuda.empty_cache()

            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[model_key] = {"name": name, "error": str(e)}
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)
            gc.collect()
            torch.cuda.empty_cache()

    # Analysis
    print(f"\n{'=' * 72}")
    print("  LOCAL EQUICORRELATION ANALYSIS")
    print("=" * 72)

    alpha_gen = 2.077
    rho_pred = 1 - (4 / np.pi) / alpha_gen ** 2
    print(f"\n  Prediction: rho_local ~ {rho_pred:.4f} (from alpha_gen = {alpha_gen})")

    print(f"\n  {'Model':<20s} {'rho_global':<12s}", end="")
    for K in K_VALUES:
        print(f"  K={K:<4d}", end="")
    print()
    print(f"  {'-' * 20} {'-' * 12}" + "  ------" * len(K_VALUES))

    rho_k10_vals = []
    for key in results:
        r = results[key]
        if "rho_local_K10" not in r:
            continue
        name = r.get("name", key)
        print(f"  {name:<20s} {r['rho_global']:<12.4f}", end="")
        for K in K_VALUES:
            k_key = f"rho_local_K{K}"
            if k_key in r:
                print(f"  {r[k_key]:<6.4f}", end="")
                if K == 10:
                    rho_k10_vals.append(r[k_key])
            else:
                print(f"  {'N/A':<6s}", end="")
        print()

    if len(rho_k10_vals) >= 3:
        mean_local = np.mean(rho_k10_vals)
        std_local = np.std(rho_k10_vals)
        print(f"\n  Mean rho_local (K=10): {mean_local:.4f} +/- {std_local:.4f}")
        print(f"  Predicted:             {rho_pred:.4f}")
        print(f"  Deviation:             {abs(mean_local - rho_pred):.4f}")
        print(f"  Hypothesis: rho_local in [0.50, 0.90]?  "
              f"{'PASS' if 0.50 <= mean_local <= 0.90 else 'FAIL'}")

    print(f"\n  Saved: {cache_path}")


if __name__ == "__main__":
    main()
