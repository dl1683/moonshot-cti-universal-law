#!/usr/bin/env python -u
"""
Compute alternative spectral/distributional W_U metrics for the generation law.

Tests whether these metrics predict PPL better than kappa_bar, which saturates
for well-trained models (d >> log(V)).

Metrics computed:
  1. Stable Rank: ||W||_F^2 / ||W||_2^2
  2. PL_Alpha_Hill: power law exponent of eigenvalue spectrum (WeightWatcher)
  3. Participation Ratio: (sum sigma^2)^2 / sum(sigma^4)
  4. Singular Entropy: KL(p_sigma || Uniform)
  5. Kappa percentiles: q01, q05, q10, q25, q50 of NN distance distribution
  6. Kappa tail heaviness: skewness, kurtosis of NN distances
  7. Mean cosine similarity (anisotropy) — already computed, included for comparison
"""
import json, time, gc, sys
import numpy as np
import torch
from pathlib import Path
from scipy.linalg import svdvals
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
KAPPA_FILE = RESULTS / "cti_generation_kappa.json"
PPL_FILE = RESULTS / "cti_generation_ppl.json"
OUT_FILE = RESULTS / "cti_generation_spectral.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pile PPL from Mamba paper Table 3
PILE_PPL = {
    "pythia-160m": 29.64, "pythia-410m": 9.95, "pythia-1b": 7.82,
    "pythia-1.4b": 7.51, "pythia-2.8b": 6.73,
    "mamba-130m": 10.56, "mamba-370m": 8.28, "mamba-790m": 7.33,
    "mamba-1.4b": 6.80, "mamba-2.8b": 6.22,
}

# Models to compute metrics for — all models with kappa data
MODELS = [
    ("pythia-160m", "EleutherAI/pythia-160m"),
    ("pythia-410m", "EleutherAI/pythia-410m"),
    ("pythia-1b", "EleutherAI/pythia-1b"),
    ("pythia-1.4b", "EleutherAI/pythia-1.4b"),
    ("pythia-2.8b", "EleutherAI/pythia-2.8b"),
    ("mamba-130m", "state-spaces/mamba-130m"),
    ("mamba-370m", "state-spaces/mamba-370m"),
    ("mamba-790m", "state-spaces/mamba-790m"),
    ("mamba-1.4b", "state-spaces/mamba-1.4b"),
    ("mamba-2.8b", "state-spaces/mamba-2.8b"),
    ("gpt2", "openai-community/gpt2"),
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    ("qwen3-1.7b", "Qwen/Qwen3-1.7B"),
    ("qwen3-4b", "Qwen/Qwen3-4B"),
    ("falcon-h1-0.5b", "tiiuae/Falcon-H1-0.5B-Base"),
    ("falcon-h1-1.5b", "tiiuae/Falcon-H1-1.5B-Base"),
    ("falcon-h1-3b", "tiiuae/Falcon-H1-3B-Base"),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M"),
    ("mistral-7b", "mistralai/Mistral-7B-v0.3"),
    ("qwen2-0.5b", "Qwen/Qwen2-0.5B"),
    ("phi-4", "microsoft/phi-4"),
    ("granite-micro", "ibm-granite/granite-4.0-micro"),
    ("lfm2.5-1.2b", "LiquidAI/LFM2.5-1.2B-Base"),
    ("mamba2-130m", "state-spaces/mamba2-130m"),
    ("mamba2-370m", "state-spaces/mamba2-370m"),
    ("mamba2-780m", "state-spaces/mamba2-780m"),
    ("mamba2-1.3b", "state-spaces/mamba2-1.3b"),
    ("mamba2-2.7b", "state-spaces/mamba2-2.7b"),
]


def extract_wu(hf_id):
    """Extract W_U from model checkpoint."""
    # Try safetensors first
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
                        "model.embed_tokens.weight", "backbone.embedding.weight"]:
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
                "transformer.wte.weight"]:
        if key in sd:
            return sd[key].float().numpy()
    raise ValueError(f"W_U not found. Keys: {list(sd.keys())[:20]}")


def compute_spectral_metrics(W_U):
    """Compute spectral metrics from W_U (V x d matrix)."""
    V, d = W_U.shape
    r = min(V, d)

    # SVD (subsample rows if V > 10K for speed — use all columns)
    if V > 10000:
        idx = np.random.RandomState(42).choice(V, 10000, replace=False)
        W_sub = W_U[idx]
    else:
        W_sub = W_U

    svs = svdvals(W_sub)
    svs = svs[svs > 1e-10]

    # 1. Stable Rank
    fro_sq = np.sum(svs**2)
    spec_sq = svs[0]**2
    stable_rank = float(fro_sq / spec_sq)

    # 2. Effective Rank (Shannon entropy of normalized SVs)
    p_sv = svs / svs.sum()
    eff_rank = float(np.exp(-np.sum(p_sv * np.log(p_sv + 1e-30))))

    # 3. Participation Ratio
    participation_ratio = float(fro_sq**2 / np.sum(svs**4))

    # 4. Singular Entropy (KL from uniform)
    p_sv_sq = svs**2 / fro_sq
    uniform = np.ones_like(p_sv_sq) / len(p_sv_sq)
    sing_entropy = float(np.sum(p_sv_sq * np.log(p_sv_sq / uniform + 1e-30)))

    # 5. PL_Alpha_Hill (power law exponent via Hill estimator)
    eigs = svs**2
    eigs_sorted = np.sort(eigs)[::-1]
    # Use Fix-finger method: k = max(1, min(n/2, sqrt(n)))
    n_eigs = len(eigs_sorted)
    k = max(1, min(n_eigs // 2, int(np.sqrt(n_eigs))))
    # Hill estimator: alpha = 1 + k / sum(log(x_i / x_k)) for i=1..k
    x_k = eigs_sorted[k-1]
    if x_k > 0:
        log_ratios = np.log(eigs_sorted[:k] / x_k + 1e-30)
        pl_alpha = float(1 + k / max(np.sum(log_ratios), 1e-10))
    else:
        pl_alpha = float('nan')

    # 6. Condition number
    cond_num = float(svs[0] / svs[-1]) if svs[-1] > 1e-10 else float('inf')

    # 7. Spectral decay rate (linear fit to log-log SV plot)
    log_rank = np.log(np.arange(1, len(svs) + 1))
    log_svs = np.log(svs)
    decay_slope, _ = np.polyfit(log_rank, log_svs, 1)

    return {
        "stable_rank": stable_rank,
        "effective_rank": eff_rank,
        "participation_ratio": participation_ratio,
        "singular_entropy": sing_entropy,
        "pl_alpha_hill": pl_alpha,
        "condition_number": cond_num,
        "spectral_decay_slope": float(decay_slope),
        "V": V,
        "d_model": d,
        "n_singular_values": len(svs),
    }


def compute_nn_distribution(W_U):
    """Compute full nearest-neighbor distance distribution."""
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

    # Percentiles
    percentiles = {
        "q01": float(np.percentile(min_dists, 1)),
        "q05": float(np.percentile(min_dists, 5)),
        "q10": float(np.percentile(min_dists, 10)),
        "q25": float(np.percentile(min_dists, 25)),
        "q50": float(np.percentile(min_dists, 50)),
        "q75": float(np.percentile(min_dists, 75)),
        "q90": float(np.percentile(min_dists, 90)),
    }

    # Distribution statistics
    stats = {
        "kappa_bar": float(np.mean(min_dists)),
        "kappa_std": float(np.std(min_dists)),
        "kappa_skew": float(((min_dists - np.mean(min_dists))**3).mean() / np.std(min_dists)**3),
        "kappa_kurtosis": float(((min_dists - np.mean(min_dists))**4).mean() / np.std(min_dists)**4 - 3),
        "kappa_iqr": float(percentiles["q75"] - percentiles["q25"]),
    }

    # Norms of rows (for frequency-norm analysis)
    row_norms = np.linalg.norm(W_U, axis=1)
    stats["norm_mean"] = float(np.mean(row_norms))
    stats["norm_std"] = float(np.std(row_norms))
    stats["norm_cv"] = float(np.std(row_norms) / np.mean(row_norms))

    return {**percentiles, **stats}


def main():
    # Load existing kappa data
    with open(KAPPA_FILE) as f:
        kappa_data = json.load(f)

    # Load existing results if any
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    to_run = [(k, hf) for k, hf in MODELS if k not in results]
    if not to_run:
        print("All models already computed.")
    else:
        print(f"\n{'='*70}")
        print(f"  SPECTRAL METRICS: {len(to_run)} models to compute")
        print(f"  Already have: {len(results)}")
        print(f"{'='*70}\n")

        for key, hf_id in to_run:
            print(f"\n--- {key} ({hf_id}) ---")
            t0 = time.time()
            try:
                W_U = extract_wu(hf_id)
                print(f"    W_U shape: {W_U.shape}")

                spectral = compute_spectral_metrics(W_U)
                nn_dist = compute_nn_distribution(W_U)
                elapsed = time.time() - t0

                results[key] = {
                    "hf_id": hf_id,
                    **spectral,
                    **nn_dist,
                    "time_s": elapsed,
                }
                print(f"    stable_rank={spectral['stable_rank']:.1f}, "
                      f"pl_alpha={spectral['pl_alpha_hill']:.3f}, "
                      f"q05={nn_dist['q05']:.4f}, "
                      f"time={elapsed:.1f}s")

                with open(OUT_FILE, 'w') as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                elapsed = time.time() - t0
                results[key] = {"hf_id": hf_id, "error": str(e)}
                print(f"    ERROR: {e} ({elapsed:.1f}s)")
                with open(OUT_FILE, 'w') as f:
                    json.dump(results, f, indent=2)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ===== ANALYSIS =====
    print(f"\n{'='*70}")
    print(f"  ANALYSIS: Which metrics predict Pile PPL? (n=10, fixed-V)")
    print(f"{'='*70}")

    # Fixed-V Pile analysis
    pile_keys = [k for k in PILE_PPL if k in results and 'error' not in results[k]]
    if len(pile_keys) < 5:
        print("  Too few models for analysis.")
        return

    log_ppls = np.array([np.log(PILE_PPL[k]) for k in pile_keys])

    # Collect all metrics
    metric_names = [
        "kappa_bar", "q01", "q05", "q10", "q25", "q50",
        "kappa_skew", "kappa_kurtosis", "kappa_iqr",
        "stable_rank", "effective_rank", "participation_ratio",
        "singular_entropy", "pl_alpha_hill", "spectral_decay_slope",
        "norm_cv",
    ]

    print(f"\n  {'Metric':<22} {'r':>7} {'p':>7} {'rho':>7} {'rho_p':>7} {'range':>10}")
    print(f"  {'-'*65}")

    analysis_results = {}
    for name in metric_names:
        vals = []
        valid = True
        for k in pile_keys:
            v = results[k].get(name, None)
            if v is None or v != v:  # nan check
                valid = False
                break
            vals.append(v)
        if not valid or len(vals) < 5:
            continue

        vals = np.array(vals)
        r, p = pearsonr(vals, log_ppls)
        rho, rho_p = spearmanr(vals, log_ppls)
        val_range = f"[{vals.min():.3f},{vals.max():.3f}]"
        print(f"  {name:<22} {r:>+7.4f} {p:>7.4f} {rho:>+7.4f} {rho_p:>7.4f} {val_range}")

        analysis_results[name] = {
            "pearson_r": float(r), "pearson_p": float(p),
            "spearman_rho": float(rho), "spearman_p": float(rho_p),
            "range": [float(vals.min()), float(vals.max())],
            "dynamic_ratio": float(vals.max() / vals.min()) if vals.min() > 0 else float('inf'),
        }

    # Same analysis without Pythia-160M
    print(f"\n  --- Without Pythia-160M (n={len(pile_keys)-1}) ---")
    pile_keys_9 = [k for k in pile_keys if k != "pythia-160m"]
    log_ppls_9 = np.array([np.log(PILE_PPL[k]) for k in pile_keys_9])

    print(f"\n  {'Metric':<22} {'r':>7} {'p':>7} {'rho':>7} {'rho_p':>7}")
    print(f"  {'-'*55}")
    for name in metric_names:
        vals = []
        valid = True
        for k in pile_keys_9:
            v = results[k].get(name, None)
            if v is None or v != v:
                valid = False
                break
            vals.append(v)
        if not valid:
            continue

        vals = np.array(vals)
        r, p = pearsonr(vals, log_ppls_9)
        rho, rho_p = spearmanr(vals, log_ppls_9)
        print(f"  {name:<22} {r:>+7.4f} {p:>7.4f} {rho:>+7.4f} {rho_p:>7.4f}")

        analysis_results[f"{name}_no160m"] = {
            "pearson_r": float(r), "pearson_p": float(p),
            "spearman_rho": float(rho), "spearman_p": float(rho_p),
        }

    # Per-model table
    print(f"\n  {'Model':<16} {'kappa':>6} {'q05':>6} {'SR':>6} {'PL_a':>6} {'logPPL':>7}")
    print(f"  {'-'*55}")
    for k in sorted(pile_keys, key=lambda k: PILE_PPL[k]):
        r = results[k]
        print(f"  {k:<16} {r.get('kappa_bar',0):>6.3f} {r.get('q05',0):>6.3f} "
              f"{r.get('stable_rank',0):>6.1f} {r.get('pl_alpha_hill',0):>6.3f} "
              f"{np.log(PILE_PPL[k]):>7.4f}")

    # Save analysis
    results["_analysis"] = analysis_results
    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
