#!/usr/bin/env python -u
"""
Frequency-Weighted Kappa for the Generation Law.

The key insight: PPL is a frequency-weighted metric (expectation under p(v)),
but kappa_bar averages UNIFORMLY over all V tokens. Under Zipf's law, the top
~2,500 tokens account for ~75% of CE loss (Limisiewicz et al. NeurIPS 2025).
kappa_bar is dominated by ~95% of tokens that contribute only ~25% of PPL.

This script computes:
  1. Per-token kappa_v (NN distance for each token v in W_U)
  2. Token frequency distribution from WikiText-103 validation
  3. Three frequency-weighted kappa variants:
     - kappa_freq_p:    sum_v p(v) * kappa_v          (PPL weighting)
     - kappa_freq_logp: sum_v softmax(log p(v)) * kappa_v  (log-freq weighting)
     - kappa_freq_sqrt: sum_v sqrt(p(v)) * kappa_v / Z (sqrt weighting, NC theory)
  4. Effective K from logit statistics (K_eff = exp(LSE(z) - z_max))
  5. Top-K kappa: kappa averaged over only the K most frequent tokens

Tests against both Pile PPL (fixed-V, n=10) and WikiText-103 PPL (cross-V, n~17).

Theory: Section 3.20 of CGF_THEORETICAL_FRAMEWORK.md
"""
import json, time, gc, sys, os
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
KAPPA_FILE = RESULTS / "cti_generation_kappa.json"
PPL_FILE = RESULTS / "cti_generation_ppl.json"
SPECTRAL_FILE = RESULTS / "cti_generation_spectral.json"
OUT_FILE = RESULTS / "cti_generation_freq_kappa.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pile PPL from Mamba paper Table 3
PILE_PPL = {
    "pythia-160m": 29.64, "pythia-410m": 9.95, "pythia-1b": 7.82,
    "pythia-1.4b": 7.51, "pythia-2.8b": 6.73,
    "mamba-130m": 10.56, "mamba-370m": 8.28, "mamba-790m": 7.33,
    "mamba-1.4b": 6.80, "mamba-2.8b": 6.22,
}

# Model definitions: key -> (display_name, hf_id, tokenizer_group)
# Tokenizer group determines which frequency distribution to use
MODELS = {
    # Fixed-V group: GPT-NeoX tokenizer (V=50304, used by Pythia + Mamba)
    "pythia-160m":  ("Pythia-160M",  "EleutherAI/pythia-160m",    "neox"),
    "pythia-410m":  ("Pythia-410M",  "EleutherAI/pythia-410m",    "neox"),
    "pythia-1b":    ("Pythia-1B",    "EleutherAI/pythia-1b",      "neox"),
    "pythia-1.4b":  ("Pythia-1.4B",  "EleutherAI/pythia-1.4b",   "neox"),
    "pythia-2.8b":  ("Pythia-2.8B",  "EleutherAI/pythia-2.8b",   "neox"),
    "mamba-130m":   ("Mamba-130M",   "state-spaces/mamba-130m",   "neox"),
    "mamba-370m":   ("Mamba-370M",   "state-spaces/mamba-370m",   "neox"),
    "mamba-790m":   ("Mamba-790M",   "state-spaces/mamba-790m",   "neox"),
    "mamba-1.4b":   ("Mamba-1.4B",   "state-spaces/mamba-1.4b",  "neox"),
    "mamba-2.8b":   ("Mamba-2.8B",   "state-spaces/mamba-2.8b",  "neox"),
    "gpt2":         ("GPT-2",        "openai-community/gpt2",     "gpt2"),
    # Cross-V group
    "qwen3-0.6b":   ("Qwen3-0.6B",  "Qwen/Qwen3-0.6B",          "qwen3"),
    "qwen3-1.7b":   ("Qwen3-1.7B",  "Qwen/Qwen3-1.7B",          "qwen3"),
    "qwen3-4b":     ("Qwen3-4B",    "Qwen/Qwen3-4B",            "qwen3"),
    "qwen2-0.5b":   ("Qwen2-0.5B",  "Qwen/Qwen2-0.5B",          "qwen2"),
    "falcon-h1-0.5b": ("Falcon-H1-0.5B", "tiiuae/Falcon-H1-0.5B-Base", "falcon-h1"),
    "falcon-h1-1.5b": ("Falcon-H1-1.5B", "tiiuae/Falcon-H1-1.5B-Base", "falcon-h1"),
    "falcon-h1-3b":   ("Falcon-H1-3B",   "tiiuae/Falcon-H1-3B-Base",   "falcon-h1"),
    "smollm2-360m":   ("SmolLM2-360M",   "HuggingFaceTB/SmolLM2-360M", "smollm2"),
    "granite-micro":  ("Granite-Micro",  "ibm-granite/granite-4.0-micro", "granite"),
    "granite-tiny":   ("Granite-Tiny",   "ibm-granite/granite-4.0-tiny-preview",  "granite"),
    "llama-3.2-3b":   ("Llama-3.2-3B",  "meta-llama/Llama-3.2-3B",       "llama3"),
    "gemma-3-4b":     ("Gemma-3-4B",    "google/gemma-3-4b-pt",          "gemma3"),
    "phi-4":          ("Phi-4",          "microsoft/phi-4",               "phi4"),
    "lfm2.5-1.2b":   ("LFM2.5-1.2B",   "LiquidAI/LFM2.5-1.2B-Base",   "lfm"),
    "mistral-7b":     ("Mistral-7B",    "mistralai/Mistral-7B-v0.3",   "mistral"),
}


def get_token_frequencies(tokenizer_name, tokenizer_hf_id, n_tokens=500000):
    """Get token frequency distribution from WikiText-103 validation."""
    cache_path = RESULTS / f"token_freq_cache_{tokenizer_name}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"    Loaded cached frequencies for {tokenizer_name} ({cached['n_tokens']} tokens)")
        return np.array(cached["freq_array"], dtype=np.float64)

    print(f"    Computing token frequencies for {tokenizer_name}...")
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_hf_id, trust_remote_code=True)
    ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:n_tokens]

    # Count frequencies
    V = tokenizer.vocab_size
    if hasattr(tokenizer, 'model') and hasattr(tokenizer.model, 'get_vocab'):
        V = max(V, len(tokenizer.get_vocab()))
    # Some tokenizers pad to next multiple
    freq_counts = Counter(tokens)
    total = len(tokens)

    # Create frequency array for full vocab
    max_token = max(max(freq_counts.keys()), V - 1)
    freq_array = np.zeros(max_token + 1, dtype=np.float64)
    for tok_id, count in freq_counts.items():
        freq_array[tok_id] = count / total

    # Cache
    cache_data = {
        "tokenizer": tokenizer_name,
        "hf_id": tokenizer_hf_id,
        "n_tokens": total,
        "vocab_size_reported": V,
        "vocab_size_actual": max_token + 1,
        "n_unique_tokens": len(freq_counts),
        "coverage_top100": float(sum(sorted(freq_counts.values(), reverse=True)[:100]) / total),
        "coverage_top1000": float(sum(sorted(freq_counts.values(), reverse=True)[:1000]) / total),
        "coverage_top2500": float(sum(sorted(freq_counts.values(), reverse=True)[:2500]) / total),
        "freq_array": freq_array.tolist(),
    }
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)
    print(f"    {len(freq_counts)} unique tokens from {total} total")
    print(f"    Coverage: top-100={cache_data['coverage_top100']:.1%}, "
          f"top-1K={cache_data['coverage_top1000']:.1%}, "
          f"top-2.5K={cache_data['coverage_top2500']:.1%}")

    del tokenizer, ds
    gc.collect()
    return freq_array


def extract_wu(hf_id):
    """Extract W_U from model checkpoint (reused from spectral metrics)."""
    from huggingface_hub import hf_hub_download
    sd = None

    try:
        from safetensors.torch import load_file
        path = hf_hub_download(hf_id, "model.safetensors")
        sd = load_file(path)
    except Exception:
        pass

    if sd is None:
        try:
            path = hf_hub_download(hf_id, "pytorch_model.bin")
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            pass

    if sd is None:
        try:
            from safetensors.torch import load_file
            idx_path = hf_hub_download(hf_id, "model.safetensors.index.json")
            with open(idx_path) as f:
                idx = json.load(f)
            weight_map = idx.get("weight_map", {})
            for key in ["lm_head.weight", "output.weight", "embed_out.weight",
                        "model.embed_tokens.weight", "backbone.embedding.weight",
                        "transformer.wte.weight", "wte.weight"]:
                if key in weight_map:
                    shard = weight_map[key]
                    shard_path = hf_hub_download(hf_id, shard)
                    sd_partial = load_file(shard_path)
                    if key in sd_partial:
                        return sd_partial[key].float().numpy()
        except Exception:
            pass

    if sd is None:
        raise ValueError(f"Cannot load checkpoint for {hf_id}")

    for key in ["lm_head.weight", "output.weight", "embed_out.weight"]:
        if key in sd:
            return sd[key].float().numpy()
    for key in ["backbone.embedding.weight", "model.embed_tokens.weight",
                "transformer.wte.weight", "wte.weight"]:
        if key in sd:
            return sd[key].float().numpy()
    raise ValueError(f"W_U not found. Keys: {list(sd.keys())[:20]}")


def compute_per_token_kappa(W_U):
    """Compute kappa_v (NN distance) for each token v. Returns array of shape (V,)."""
    V, d = W_U.shape

    # Normalize rows
    norms = np.linalg.norm(W_U, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    W_norm = W_U / norms

    W_t = torch.tensor(W_norm, device=DEVICE, dtype=torch.float32)

    batch_size = 512
    min_dists = []
    for i in range(0, V, batch_size):
        batch = W_t[i:i+batch_size]
        sims = batch @ W_t.T
        for j in range(batch.shape[0]):
            sims[j, i + j] = -float('inf')
        max_cos = sims.max(dim=1).values
        min_dist = torch.sqrt(2 * (1 - max_cos.clamp(max=1.0)))
        min_dists.append(min_dist.cpu().numpy())

    min_dists = np.concatenate(min_dists)

    del W_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return min_dists, norms.squeeze()


def compute_freq_weighted_kappa(kappa_v, freq, row_norms):
    """Compute frequency-weighted kappa variants.

    Args:
        kappa_v: per-token NN distances, shape (V_wu,)
        freq: token frequency array, shape (V_freq,)
        row_norms: W_U row norms, shape (V_wu,)

    Returns: dict of metrics
    """
    V_wu = len(kappa_v)
    V_freq = len(freq)
    V = min(V_wu, V_freq)  # Use the smaller of the two

    kv = kappa_v[:V]
    p = freq[:V]
    rn = row_norms[:V]

    # Ensure frequency sums to 1 over the tokens we consider
    p_sum = p.sum()
    if p_sum > 0:
        p_norm = p / p_sum
    else:
        p_norm = np.ones(V) / V

    # 1. Uniform kappa (baseline)
    kappa_bar = float(np.mean(kv))

    # 2. Frequency-weighted kappa (PPL weighting)
    kappa_freq_p = float(np.sum(p_norm * kv))

    # 3. Log-frequency weighting: softmax(log(p + eps))
    log_p = np.log(p + 1e-30)
    log_p_shifted = log_p - log_p.max()
    w_logp = np.exp(log_p_shifted)
    w_logp = w_logp / w_logp.sum()
    kappa_freq_logp = float(np.sum(w_logp * kv))

    # 4. Sqrt-frequency weighting (NC theory: ||w_k|| ~ sqrt(n_k))
    w_sqrt = np.sqrt(p + 1e-30)
    w_sqrt = w_sqrt / w_sqrt.sum()
    kappa_freq_sqrt = float(np.sum(w_sqrt * kv))

    # 5. Top-K kappa: average over only the K most frequent tokens
    top_k_results = {}
    for K in [100, 500, 1000, 2500, 5000]:
        if K <= V:
            # Get top-K most frequent token indices
            top_indices = np.argsort(p)[-K:]
            kappa_topK = float(np.mean(kv[top_indices]))
            kappa_topK_weighted = float(np.sum(p_norm[top_indices] * kv[top_indices]) /
                                        p_norm[top_indices].sum())
            top_k_results[f"kappa_top{K}"] = kappa_topK
            top_k_results[f"kappa_top{K}_weighted"] = kappa_topK_weighted

    # 6. Frequency-kappa correlation: do frequent tokens have different kappa?
    # Only among tokens that appear at least once
    observed = p > 0
    n_observed = int(observed.sum())
    if n_observed > 10:
        log_freq = np.log(p[observed] + 1e-30)
        kv_obs = kv[observed]
        r_freq_kappa, p_freq_kappa = pearsonr(log_freq, kv_obs)
        rho_freq_kappa, _ = spearmanr(log_freq, kv_obs)
    else:
        r_freq_kappa = float('nan')
        p_freq_kappa = float('nan')
        rho_freq_kappa = float('nan')

    # 7. Norm-frequency correlation
    if n_observed > 10:
        log_freq_obs = np.log(p[observed] + 1e-30)
        norms_obs = rn[observed]
        r_freq_norm, _ = pearsonr(log_freq_obs, norms_obs)
    else:
        r_freq_norm = float('nan')

    # 8. Effective coverage: what fraction of frequency mass is in top 2500?
    sorted_p = np.sort(p)[::-1]
    coverage_2500 = float(sorted_p[:2500].sum() / (p_sum + 1e-30)) if p_sum > 0 else 0

    # 9. Dynamic range comparison
    # kappa among top-1000 vs bottom tokens
    if V >= 2000:
        top1k_idx = np.argsort(p)[-1000:]
        bot_idx = np.argsort(p)[:max(1000, V // 2)]
        bot_idx = bot_idx[p[bot_idx] > 0]  # only tokens that appear
        kappa_top1k = float(np.mean(kv[top1k_idx]))
        kappa_bot = float(np.mean(kv[bot_idx])) if len(bot_idx) > 0 else float('nan')
        dynamic_range_ratio = kappa_top1k / kappa_bot if kappa_bot > 0 else float('nan')
    else:
        kappa_top1k = float('nan')
        kappa_bot = float('nan')
        dynamic_range_ratio = float('nan')

    return {
        "kappa_bar": kappa_bar,
        "kappa_freq_p": kappa_freq_p,
        "kappa_freq_logp": kappa_freq_logp,
        "kappa_freq_sqrt": kappa_freq_sqrt,
        **top_k_results,
        "n_tokens_observed": n_observed,
        "coverage_top2500": coverage_2500,
        "r_logfreq_kappa": float(r_freq_kappa),
        "p_logfreq_kappa": float(p_freq_kappa),
        "rho_logfreq_kappa": float(rho_freq_kappa),
        "r_logfreq_norm": float(r_freq_norm),
        "kappa_frequent_tokens": kappa_top1k,
        "kappa_rare_tokens": kappa_bot,
        "dynamic_range_ratio": dynamic_range_ratio,
        "V_used": V,
    }


def analyze_results(results):
    """Analyze frequency-weighted kappa vs PPL."""
    print("\n" + "=" * 70)
    print("  ANALYSIS: FREQUENCY-WEIGHTED KAPPA vs PPL")
    print("=" * 70)

    analysis = {}

    # Group 1: Fixed-V Pile PPL (n=10, most informative)
    pile_keys = [k for k in results if k in PILE_PPL and "error" not in results[k]]
    if len(pile_keys) >= 5:
        print(f"\n--- Fixed-V Pile PPL (n={len(pile_keys)}) ---")
        log_ppls = [np.log(PILE_PPL[k]) for k in pile_keys]

        metrics_to_test = ["kappa_bar", "kappa_freq_p", "kappa_freq_logp",
                           "kappa_freq_sqrt", "kappa_top100", "kappa_top500",
                           "kappa_top1000", "kappa_top2500"]

        pile_results = {}
        for metric in metrics_to_test:
            vals = []
            valid_ppls = []
            for k in pile_keys:
                v = results[k].get(metric)
                if v is not None and not np.isnan(v):
                    vals.append(v)
                    valid_ppls.append(np.log(PILE_PPL[k]))

            if len(vals) >= 5:
                r, p = pearsonr(vals, valid_ppls)
                rho, p_rho = spearmanr(vals, valid_ppls)
                pile_results[metric] = {
                    "pearson_r": float(r), "pearson_p": float(p),
                    "spearman_rho": float(rho), "spearman_p": float(p_rho),
                    "n": len(vals),
                }
                marker = " ***" if abs(rho) > 0.8 else (" **" if abs(rho) > 0.6 else "")
                print(f"  {metric:25s}: r={r:+.3f} (p={p:.3f}), rho={rho:+.3f} (p={p_rho:.3f}){marker}")

        # Without Pythia-160M (the leverage test)
        pile_keys_no160 = [k for k in pile_keys if k != "pythia-160m"]
        if len(pile_keys_no160) >= 5:
            print(f"\n--- Without Pythia-160M (n={len(pile_keys_no160)}) ---")
            pile_no160 = {}
            for metric in metrics_to_test:
                vals = []
                valid_ppls = []
                for k in pile_keys_no160:
                    v = results[k].get(metric)
                    if v is not None and not np.isnan(v):
                        vals.append(v)
                        valid_ppls.append(np.log(PILE_PPL[k]))

                if len(vals) >= 5:
                    r, p = pearsonr(vals, valid_ppls)
                    rho, p_rho = spearmanr(vals, valid_ppls)
                    pile_no160[metric] = {
                        "pearson_r": float(r), "pearson_p": float(p),
                        "spearman_rho": float(rho), "spearman_p": float(p_rho),
                        "n": len(vals),
                    }
                    marker = " ***" if abs(rho) > 0.8 else (" **" if abs(rho) > 0.6 else "")
                    print(f"  {metric:25s}: r={r:+.3f} (p={p:.3f}), rho={rho:+.3f} (p={p_rho:.3f}){marker}")

            pile_results["without_160m"] = pile_no160

        analysis["fixed_v_pile"] = pile_results

    # Group 2: WikiText-103 PPL (cross-V)
    if PPL_FILE.exists():
        with open(PPL_FILE) as f:
            wt_ppl = json.load(f)
        wt_keys = [k for k in results if k in wt_ppl and "error" not in results[k]
                    and "ppl" in wt_ppl[k] and wt_ppl[k]["ppl"] < 100]
        if len(wt_keys) >= 5:
            print(f"\n--- WikiText-103 PPL (n={len(wt_keys)}) ---")
            wt_results = {}
            metrics_to_test = ["kappa_bar", "kappa_freq_p", "kappa_freq_logp",
                               "kappa_freq_sqrt", "kappa_top1000", "kappa_top2500"]
            for metric in metrics_to_test:
                vals = []
                valid_ppls = []
                for k in wt_keys:
                    v = results[k].get(metric)
                    if v is not None and not np.isnan(v):
                        vals.append(v)
                        valid_ppls.append(np.log(wt_ppl[k]["ppl"]))

                if len(vals) >= 5:
                    r, p = pearsonr(vals, valid_ppls)
                    rho, p_rho = spearmanr(vals, valid_ppls)
                    wt_results[metric] = {
                        "pearson_r": float(r), "pearson_p": float(p),
                        "spearman_rho": float(rho), "spearman_p": float(p_rho),
                        "n": len(vals),
                    }
                    marker = " ***" if abs(rho) > 0.8 else (" **" if abs(rho) > 0.6 else "")
                    print(f"  {metric:25s}: r={r:+.3f} (p={p:.3f}), rho={rho:+.3f} (p={p_rho:.3f}){marker}")

            analysis["cross_v_wikitext"] = wt_results

    # Frequency-geometry diagnostics
    print(f"\n--- Frequency-Geometry Diagnostics ---")
    for k in sorted(results.keys()):
        if "error" in results[k]:
            continue
        r_fk = results[k].get("r_logfreq_kappa", float('nan'))
        r_fn = results[k].get("r_logfreq_norm", float('nan'))
        dr = results[k].get("dynamic_range_ratio", float('nan'))
        print(f"  {k:20s}: r(logfreq,kappa)={r_fk:+.3f}, "
              f"r(logfreq,norm)={r_fn:+.3f}, "
              f"kappa_top/kappa_bot={dr:.3f}")

    analysis["diagnostics"] = {
        k: {
            "r_logfreq_kappa": results[k].get("r_logfreq_kappa"),
            "r_logfreq_norm": results[k].get("r_logfreq_norm"),
            "dynamic_range_ratio": results[k].get("dynamic_range_ratio"),
            "kappa_bar": results[k].get("kappa_bar"),
            "kappa_freq_p": results[k].get("kappa_freq_p"),
        }
        for k in sorted(results.keys()) if "error" not in results[k]
    }

    return analysis


def main():
    print("=" * 70)
    print("  FREQUENCY-WEIGHTED KAPPA FOR THE GENERATION LAW")
    print("  Theory: CGF Section 3.20 (freq-weighted kappa + effective K)")
    print("=" * 70)

    # Load existing results
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    # Determine which tokenizer to use for each model
    tokenizer_map = {
        "neox": "EleutherAI/pythia-160m",
        "gpt2": "openai-community/gpt2",
        "qwen3": "Qwen/Qwen3-0.6B",
        "qwen2": "Qwen/Qwen2-0.5B",
        "falcon-h1": "tiiuae/Falcon-H1-0.5B-Base",
        "smollm2": "HuggingFaceTB/SmolLM2-360M",
        "mistral": "mistralai/Mistral-7B-v0.3",
        "granite": "ibm-granite/granite-4.0-micro",
    }

    to_run = [(k, v) for k, v in MODELS.items() if k not in results]
    if not to_run:
        print("\nAll models already computed. Running analysis only.")
    else:
        print(f"\n{len(to_run)} models to compute, {len(results)} already cached.\n")

        # Pre-compute token frequencies for each tokenizer group
        freq_cache = {}

        for key, (name, hf_id, tok_group) in to_run:
            print(f"\n--- {key} ({name}) ---")
            t0 = time.time()

            try:
                # Get token frequencies
                if tok_group not in freq_cache:
                    tok_hf_id = tokenizer_map.get(tok_group, hf_id)
                    freq_cache[tok_group] = get_token_frequencies(tok_group, tok_hf_id)

                freq = freq_cache[tok_group]

                # Extract W_U and compute per-token kappa
                print(f"    Extracting W_U from {hf_id}...")
                W_U = extract_wu(hf_id)
                print(f"    W_U shape: {W_U.shape}")

                print(f"    Computing per-token kappa (V={W_U.shape[0]})...")
                kappa_v, row_norms = compute_per_token_kappa(W_U)

                # Compute frequency-weighted metrics
                print(f"    Computing frequency-weighted metrics...")
                metrics = compute_freq_weighted_kappa(kappa_v, freq, row_norms)
                metrics["model"] = name
                metrics["hf_id"] = hf_id
                metrics["tokenizer_group"] = tok_group
                metrics["time_s"] = time.time() - t0

                results[key] = metrics
                print(f"    kappa_bar={metrics['kappa_bar']:.4f}, "
                      f"kappa_freq_p={metrics['kappa_freq_p']:.4f}, "
                      f"kappa_top1000={metrics.get('kappa_top1000', 'N/A')}")
                print(f"    r(logfreq,kappa)={metrics['r_logfreq_kappa']:+.3f}, "
                      f"dynamic_range={metrics['dynamic_range_ratio']:.3f}")

                # Save after each model
                with open(OUT_FILE, "w") as f:
                    json.dump(results, f, indent=2)

                del W_U, kappa_v, row_norms
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                results[key] = {"model": name, "error": str(e)[:200]}
                with open(OUT_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"    ERROR: {e}")

    # Run analysis
    analysis = analyze_results(results)

    # Save final results with analysis
    final = {k: v for k, v in results.items()}
    final["_analysis"] = analysis
    with open(OUT_FILE, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nResults saved to {OUT_FILE}")
    print("\nDone!")


if __name__ == "__main__":
    main()
