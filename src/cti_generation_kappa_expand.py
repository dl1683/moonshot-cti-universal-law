#!/usr/bin/env python -u
"""
Expand kappa extraction to new models not in the original generation law.
Appends results to results/cti_generation_kappa.json.
"""
import json, time, gc, sys
import numpy as np
import torch
from pathlib import Path
from scipy.linalg import svdvals
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
KAPPA_FILE = RESULTS / "cti_generation_kappa.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# New models to add (key, hf_id, name, params_M, arch)
NEW_MODELS = [
    # Mamba2 series (fixed-V ~ 50288)
    ("mamba2-130m",  "state-spaces/mamba2-130m",  "Mamba2-130M",  130,  "ssm"),
    ("mamba2-370m",  "state-spaces/mamba2-370m",  "Mamba2-370M",  370,  "ssm"),
    ("mamba2-780m",  "state-spaces/mamba2-780m",  "Mamba2-780M",  780,  "ssm"),
    ("mamba2-1.3b",  "state-spaces/mamba2-1.3b",  "Mamba2-1.3B",  1300, "ssm"),
    ("mamba2-2.7b",  "state-spaces/mamba2-2.7b",  "Mamba2-2.7B",  2700, "ssm"),
    # New transformers
    ("qwen2-0.5b",   "Qwen/Qwen2-0.5B",          "Qwen2-0.5B",   500,  "transformer"),
    ("phi-4",        "microsoft/phi-4",            "Phi-4",        3800, "transformer"),
    ("gemma-3-4b",   "google/gemma-3-4b",          "Gemma-3-4B",   4000, "transformer"),
    # Hybrids
    ("falcon-h1-3b", "tiiuae/Falcon-H1-3B-Base",   "Falcon-H1-3B", 3000, "hybrid"),
    # Granite
    ("granite-micro","ibm-granite/granite-4.0-micro","Granite-4.0-Micro",350,"hybrid"),
    ("granite-tiny", "ibm-granite/granite-4.0-tiny", "Granite-4.0-Tiny",1000,"hybrid"),
    # Liquid
    ("lfm2.5-1.2b", "LiquidAI/LFM2.5-1.2B-Base",  "LFM2.5-1.2B", 1200, "liquid"),
]


def extract_wu_checkpoint(hf_id):
    """Extract W_U directly from checkpoint state_dict."""
    from huggingface_hub import hf_hub_download

    # Try safetensors first
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
        # Try sharded safetensors — find the index
        try:
            from safetensors.torch import load_file
            idx_path = hf_hub_download(hf_id, "model.safetensors.index.json")
            with open(idx_path) as f:
                idx = json.load(f)
            # Find which shard has lm_head.weight
            weight_map = idx.get("weight_map", {})
            for key in ["lm_head.weight", "output.weight", "embed_out.weight",
                        "model.embed_tokens.weight", "backbone.embedding.weight"]:
                if key in weight_map:
                    shard = weight_map[key]
                    shard_path = hf_hub_download(hf_id, shard)
                    sd_partial = load_file(shard_path)
                    if key in sd_partial:
                        return sd_partial[key].float().numpy(), key
        except Exception:
            pass

    if sd is None:
        raise ValueError(f"Cannot load checkpoint for {hf_id}")

    # Search for W_U
    for key in ["lm_head.weight", "output.weight", "embed_out.weight"]:
        if key in sd:
            return sd[key].float().numpy(), key

    # Fallback: tied embeddings
    for key in ["backbone.embedding.weight", "model.embed_tokens.weight",
                "transformer.wte.weight", "backbone.embeddings.word_embeddings.weight"]:
        if key in sd:
            return sd[key].float().numpy(), key

    raise ValueError(f"W_U not found. Keys: {list(sd.keys())[:20]}")


def extract_wu_automodel(hf_id):
    """Extract W_U via AutoModelForCausalLM."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cpu"
    )
    for attr in ["lm_head", "output", "embed_out"]:
        if hasattr(model, attr):
            head = getattr(model, attr)
            if hasattr(head, "weight"):
                W = head.weight.detach().float().numpy()
                del model; gc.collect()
                return W, attr + ".weight"
    # Tied embeddings
    for path in ["model.embed_tokens", "transformer.wte"]:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "weight"):
                W = obj.weight.detach().float().numpy()
                del model; gc.collect()
                return W, path + ".weight (tied)"
        except AttributeError:
            continue
    del model; gc.collect()
    raise ValueError("W_U not found via AutoModel")


def compute_kappa(W_U):
    """Compute kappa_bar from W_U (V x d) matrix.

    kappa = mean over v of min_{j!=v} ||w_v - w_j|| (unit-normalized rows).
    Also computes: kappa_std, effective_rank, mean_cossim, condition_number.
    """
    V, d = W_U.shape
    print(f"    Computing kappa: V={V}, d={d}...", flush=True)

    # Normalize rows to unit norm
    norms = np.linalg.norm(W_U, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    W_norm = W_U / norms

    # Move to GPU for batch distance computation
    W_t = torch.tensor(W_norm, device=DEVICE, dtype=torch.float32)

    # Compute min nearest-neighbor distance in batches
    batch_size = 512
    min_dists = []
    for i in range(0, V, batch_size):
        batch = W_t[i:i+batch_size]  # (B, d)
        # Cosine similarity (already unit norm)
        sims = batch @ W_t.T  # (B, V)
        # Set self-similarity to -inf
        for j in range(batch.shape[0]):
            sims[j, i + j] = -float('inf')
        # Distance = sqrt(2(1-cos)) for unit vectors
        max_cos = sims.max(dim=1).values  # nearest neighbor (max cos = min dist)
        min_dist = torch.sqrt(2 * (1 - max_cos.clamp(max=1.0)))
        min_dists.append(min_dist.cpu().numpy())

    min_dists = np.concatenate(min_dists)
    kappa_bar = float(np.mean(min_dists))
    kappa_std = float(np.std(min_dists))

    # Effective rank
    try:
        svs = svdvals(W_norm[:min(V, 5000), :])  # Subsample for speed
        svs_sq = svs ** 2
        p = svs_sq / svs_sq.sum()
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
    except Exception:
        eff_rank = float('nan')

    # Mean cosine similarity (subsample for V > 10K)
    n_sample = min(V, 5000)
    idx = np.random.RandomState(42).choice(V, n_sample, replace=False)
    W_sub = W_t[idx]
    cos_mat = W_sub @ W_sub.T
    mask = ~torch.eye(n_sample, dtype=bool, device=DEVICE)
    mean_cossim = float(cos_mat[mask].mean().cpu())

    # Condition number (of subsampled matrix)
    try:
        cond = float(svs[0] / svs[-1]) if len(svs) > 1 and svs[-1] > 1e-10 else float('inf')
    except Exception:
        cond = float('nan')

    # Random baseline kappa
    W_rand = np.random.randn(V, d).astype(np.float32)
    W_rand /= np.linalg.norm(W_rand, axis=1, keepdims=True)
    W_r = torch.tensor(W_rand, device=DEVICE, dtype=torch.float32)
    rand_dists = []
    for i in range(0, V, batch_size):
        batch = W_r[i:i+batch_size]
        sims = batch @ W_r.T
        for j in range(batch.shape[0]):
            sims[j, i + j] = -float('inf')
        max_cos = sims.max(dim=1).values
        min_dist = torch.sqrt(2 * (1 - max_cos.clamp(max=1.0)))
        rand_dists.append(min_dist.cpu().numpy())
    rand_dists = np.concatenate(rand_dists)
    kappa_random = float(np.mean(rand_dists))

    del W_t, W_r
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "kappa_bar": kappa_bar,
        "kappa_std": kappa_std,
        "kappa_cv": kappa_std / max(kappa_bar, 1e-10),
        "effective_rank": eff_rank,
        "mean_cossim": mean_cossim,
        "condition_number": cond,
        "V": V,
        "d_model": d,
        "kappa_random_mean": kappa_random,
        "kappa_random_std": float(np.std(rand_dists)),
    }


def main():
    # Load existing results
    if KAPPA_FILE.exists():
        with open(KAPPA_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    # Filter to models not yet computed
    to_run = [(k, hf, name, pm, arch) for k, hf, name, pm, arch in NEW_MODELS
              if k not in results]

    if not to_run:
        print("All models already computed. Nothing to do.")
        return

    print(f"\n{'='*70}")
    print(f"  KAPPA EXTRACTION: {len(to_run)} new models")
    print(f"  Already have: {len(results)} models")
    print(f"{'='*70}\n")

    for key, hf_id, name, params_m, arch in to_run:
        print(f"\n--- {name} ({hf_id}) ---")
        t0 = time.time()
        try:
            # Try checkpoint approach first (faster, works for Mamba)
            try:
                W_U, src = extract_wu_checkpoint(hf_id)
                print(f"    Loaded from checkpoint ({src})")
            except Exception as e1:
                print(f"    Checkpoint failed ({e1}), trying AutoModel...")
                W_U, src = extract_wu_automodel(hf_id)
                print(f"    Loaded via AutoModel ({src})")

            metrics = compute_kappa(W_U)
            elapsed = time.time() - t0

            results[key] = {
                "model": name,
                "hf_id": hf_id,
                "params_M": params_m,
                "vocab_size": metrics["V"],
                "arch": arch,
                "tier": 1 if metrics["V"] in range(50200, 50400) else 2,
                **metrics,
                "time_s": elapsed,
            }
            print(f"    kappa_bar={metrics['kappa_bar']:.4f}, V={metrics['V']}, "
                  f"d={metrics['d_model']}, time={elapsed:.1f}s")

            # Save after each model (in case of crash)
            with open(KAPPA_FILE, 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            elapsed = time.time() - t0
            results[key] = {"model": name, "error": str(e)}
            print(f"    ERROR: {e} ({elapsed:.1f}s)")
            with open(KAPPA_FILE, 'w') as f:
                json.dump(results, f, indent=2)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    n_ok = sum(1 for v in results.values() if 'kappa_bar' in v)
    n_err = sum(1 for v in results.values() if 'error' in v)
    print(f"  Total: {len(results)} models ({n_ok} OK, {n_err} errors)")

    # Print kappa table
    print(f"\n  {'Model':<25} {'kappa':>8} {'V':>8} {'d':>6} {'Arch':>12}")
    print(f"  {'-'*65}")
    for k, v in sorted(results.items(), key=lambda x: x[1].get('kappa_bar', 0)):
        if 'kappa_bar' in v:
            print(f"  {v['model']:<25} {v['kappa_bar']:>8.4f} {v['V']:>8} "
                  f"{v['d_model']:>6} {v.get('arch','?'):>12}")


if __name__ == "__main__":
    main()
