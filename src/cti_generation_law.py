#!/usr/bin/env python -u
"""
CGF GENERATION LAW: kappa from W_U predicts perplexity (Mar 3 2026)
====================================================================
Codex Design Gate: Session 85 (gpt-5.3-codex-spark, xhigh reasoning)

GENERATION LAW:
    log(PPL) = beta * log(V-1) - alpha_gen * kappa_bar + C_model

PHASES (config-driven, each saves to cache):
    nc_gate         - Step 0: measure Neural Collapse degree (6 models)
    kappa_extract   - Extract W_U, compute kappa_bar for all models
    ppl_eval        - Compute perplexity on WikiText-103 validation
    hypothesis_test - Test all pre-registered hypotheses
    proxy_b         - Whitened kappa (requires forward passes)
    all             - Run everything sequentially

PRE-REGISTRATION:
    research/CGF_GENERATION_PREREGISTRATION.md (original, 8 models)
    research/CGF_GENERATION_PREREGISTRATION_ADDENDUM.md (expanded, 25 models)

OUTPUTS:
    results/cti_generation_nc_gate.json
    results/cti_generation_kappa.json
    results/cti_generation_ppl.json
    results/cti_generation_law.json  (final hypothesis tests)
"""

import json
import time
import sys
import gc
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import svdvals
import torch

# ============================================================
# CONFIG
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Phase control: set via command line or modify here
PHASE = sys.argv[1] if len(sys.argv) > 1 else "nc_gate"

print(f"="*70)
print(f"  CGF GENERATION LAW EXPERIMENT")
print(f"  Phase: {PHASE}")
print(f"  Device: {DEVICE}")
print(f"="*70, flush=True)

# ============================================================
# MODEL REGISTRY
# ============================================================
# Each entry: (hf_id, short_name, params_M, vocab_size, arch_type, tier)
# vocab_size=0 means "detect from model config"
# arch_type: "transformer", "ssm", "hybrid", "liquid"
MODELS = {
    # Tier 1: Fixed-V group (V~50280)
    "pythia-160m":  ("EleutherAI/pythia-160m",  "Pythia-160M",  160,  50280, "transformer", 1),
    "pythia-410m":  ("EleutherAI/pythia-410m",  "Pythia-410M",  410,  50280, "transformer", 1),
    "pythia-1b":    ("EleutherAI/pythia-1b",    "Pythia-1B",    1000, 50280, "transformer", 1),
    "pythia-1.4b":  ("EleutherAI/pythia-1.4b",  "Pythia-1.4B",  1400, 50280, "transformer", 1),
    "pythia-2.8b":  ("EleutherAI/pythia-2.8b",  "Pythia-2.8B",  2800, 50280, "transformer", 1),
    "mamba-130m":   ("state-spaces/mamba-130m",  "Mamba-130M",   130,  50280, "ssm", 1),
    "mamba-370m":   ("state-spaces/mamba-370m",  "Mamba-370M",   370,  50280, "ssm", 1),
    "mamba-790m":   ("state-spaces/mamba-790m",  "Mamba-790M",   790,  50280, "ssm", 1),
    "mamba-1.4b":   ("state-spaces/mamba-1.4b",  "Mamba-1.4B",  1400, 50280, "ssm", 1),
    "mamba-2.8b":   ("state-spaces/mamba-2.8b",  "Mamba-2.8B",  2800, 50280, "ssm", 1),
    "gpt2":         ("openai-community/gpt2",    "GPT-2",        124,  50257, "transformer", 1),
    # Tier 2: Cross-architecture
    "qwen3-0.6b":   ("Qwen/Qwen3-0.6B",         "Qwen3-0.6B",  600,  151936, "transformer", 2),
    "qwen3-1.7b":   ("Qwen/Qwen3-1.7B",         "Qwen3-1.7B",  1700, 151936, "transformer", 2),
    "qwen3-4b":     ("Qwen/Qwen3-4B",           "Qwen3-4B",    4000, 151936, "transformer", 2),
    "llama-3.2-3b": ("meta-llama/Llama-3.2-3B",  "Llama-3.2-3B",3000, 128256, "transformer", 2),
    "falcon-h1-0.5b":("tiiuae/Falcon-H1-0.5B-Base","Falcon-H1-0.5B",500,0,"hybrid",2),
    "falcon-h1-1.5b":("tiiuae/Falcon-H1-1.5B-Base","Falcon-H1-1.5B",1500,0,"hybrid",2),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M","SmolLM2-360M",360, 0, "transformer", 2),
    "mistral-7b":   ("mistralai/Mistral-7B-v0.3","Mistral-7B",   7000, 32000, "transformer", 2),
}

# NC gate models (diverse subset)
NC_GATE_MODELS = [
    "pythia-410m", "pythia-2.8b",
    "mamba-790m", "mamba-2.8b",
    "qwen3-0.6b",
]

# Fixed-V subset for clean test
FIXED_V_MODELS = [k for k, v in MODELS.items() if v[5] == 1]

# ============================================================
# UTILITY: Load model and tokenizer
# ============================================================
def load_model_and_tokenizer(model_key, for_generation=False):
    """Load a model and tokenizer from HuggingFace.

    Args:
        model_key: Key in MODELS dict
        for_generation: If True, load as CausalLM. If False, load base model.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    hf_id = MODELS[model_key][0]
    arch = MODELS[model_key][4]
    print(f"  Loading {hf_id}...", flush=True)

    # Determine dtype based on model size
    params_m = MODELS[model_key][2]
    if params_m <= 3000:
        dtype = torch.float16
    else:
        dtype = torch.float16  # quantize later if needed

    # Special tokenizer handling for Mamba (uses GPT-NeoX tokenizer)
    if arch == "ssm" and "mamba" in hf_id.lower() and "codestral" not in hf_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b", trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Special handling for Mamba models
    if arch == "ssm" and "mamba" in hf_id.lower():
        try:
            from transformers import MambaForCausalLM
            model = MambaForCausalLM.from_pretrained(
                hf_id, dtype=dtype, trust_remote_code=True
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, dtype=dtype, trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=dtype, trust_remote_code=True
        )

    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer


def get_unembedding_matrix(model, model_key):
    """Extract the unembedding matrix W_U from a model.

    Returns W_U as a numpy array of shape (V, d_model) in float32.
    """
    # Try common attribute names for the LM head
    lm_head = None
    for attr in ["lm_head", "output", "embed_out"]:
        if hasattr(model, attr):
            head = getattr(model, attr)
            if hasattr(head, "weight"):
                lm_head = head.weight
                break

    if lm_head is None:
        # For models with tied embeddings, try the embedding layer
        for attr_path in [
            "model.embed_tokens",
            "transformer.wte",
            "backbone.embeddings",
            "backbone.embedding",
            "gpt_neox.embed_in",
        ]:
            obj = model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, "weight"):
                    lm_head = obj.weight
                    break
            except AttributeError:
                continue

    if lm_head is None:
        raise ValueError(f"Cannot find unembedding matrix for {model_key}")

    W_U = lm_head.detach().float().cpu().numpy()
    print(f"    W_U shape: {W_U.shape}", flush=True)
    return W_U


def get_unembedding_from_checkpoint(model_key):
    """Extract W_U directly from checkpoint files (for models that fail to load).

    Used for Mamba models where the transformers MambaForCausalLM has shape bugs.
    """
    from huggingface_hub import hf_hub_download

    hf_id = MODELS[model_key][0]
    print(f"  Loading W_U directly from checkpoint: {hf_id}...", flush=True)

    # Try safetensors first, then pytorch_model.bin
    try:
        from safetensors.torch import load_file
        path = hf_hub_download(hf_id, "model.safetensors")
        sd = load_file(path)
    except Exception:
        try:
            path = hf_hub_download(hf_id, "pytorch_model.bin")
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise ValueError(f"Cannot download checkpoint for {hf_id}: {e}")

    # Find lm_head or embedding weight
    for key in ["lm_head.weight", "output.weight", "embed_out.weight"]:
        if key in sd:
            W_U = sd[key].float().numpy()
            print(f"    W_U shape: {W_U.shape} (from {key})", flush=True)
            return W_U

    # Fallback: embedding weight (tied)
    for key in ["backbone.embedding.weight", "model.embed_tokens.weight",
                "transformer.wte.weight"]:
        if key in sd:
            W_U = sd[key].float().numpy()
            print(f"    W_U shape: {W_U.shape} (from {key}, tied)", flush=True)
            return W_U

    raise ValueError(f"Cannot find W_U in state dict. Keys: {list(sd.keys())[:10]}")


def free_model(model):
    """Free GPU memory after using a model."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# PHASE 1: NC GATE
# ============================================================
def measure_nc_degree(model, tokenizer, model_key, n_tokens=10000):
    """Measure Neural Collapse degree: R^2_NC.

    R^2_NC = 1 - mean(||eps||^2) / mean(||h(x)||^2)
    where eps = h(x) - gamma * w_y, gamma = h(x) @ w_y / ||w_y||^2
    """
    from datasets import load_dataset

    print(f"  Measuring NC degree for {model_key}...", flush=True)

    # Load WikiText-103 validation
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [x["text"] for x in wiki if x["text"].strip()]

    # Get W_U
    W_U = get_unembedding_matrix(model, model_key)
    W_U_tensor = torch.tensor(W_U, device=DEVICE, dtype=torch.float32)

    all_h_norms_sq = []
    all_eps_norms_sq = []
    total_tokens = 0

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

            # Get hidden states from the model
            outputs = model(**enc, output_hidden_states=True)

            # Final hidden state (before LM head)
            h = outputs.hidden_states[-1].float()  # (1, seq_len, d_model)
            h = h[0, :-1, :]  # Remove last position (no target)

            # Target tokens
            targets = enc["input_ids"][0, 1:]  # Shifted by 1

            # Get W_U rows for targets
            w_y = W_U_tensor[targets]  # (seq_len-1, d_model)

            # Compute gamma = h @ w_y / ||w_y||^2 (per-token scalar)
            w_y_norm_sq = (w_y * w_y).sum(dim=1, keepdim=True)  # (n, 1)
            gamma = (h * w_y).sum(dim=1, keepdim=True) / (w_y_norm_sq + 1e-10)

            # Residual
            eps = h - gamma * w_y

            # Norms
            h_norm_sq = (h * h).sum(dim=1)  # (n,)
            eps_norm_sq = (eps * eps).sum(dim=1)

            all_h_norms_sq.append(h_norm_sq.cpu().numpy())
            all_eps_norms_sq.append(eps_norm_sq.cpu().numpy())
            total_tokens += len(targets)

    if total_tokens == 0:
        return {"r2_nc": 0.0, "n_tokens": 0}

    h_norms = np.concatenate(all_h_norms_sq)
    eps_norms = np.concatenate(all_eps_norms_sq)

    r2_nc = 1.0 - eps_norms.mean() / (h_norms.mean() + 1e-10)

    # Dimensionally-corrected NC metric:
    # R^2_NC = cos^2(h, w_y) in expectation. For random vectors: E[cos^2] = 1/d.
    # NC alignment ratio = R^2_NC * d measures how many times stronger than random.
    d_model = W_U.shape[1]
    nc_alignment_ratio = r2_nc * d_model
    random_baseline = 1.0 / d_model

    print(f"    R^2_NC = {r2_nc:.4f}, d={d_model}, "
          f"alignment_ratio = {nc_alignment_ratio:.1f}x random "
          f"(random baseline = {random_baseline:.6f}), "
          f"n={total_tokens} tokens", flush=True)

    return {
        "r2_nc": float(r2_nc),
        "d_model": int(d_model),
        "nc_alignment_ratio": float(nc_alignment_ratio),
        "random_baseline": float(random_baseline),
        "n_tokens": int(total_tokens),
    }


def run_nc_gate():
    """Phase 1: NC gate check on diverse model subset."""
    print(f"\n{'='*70}")
    print(f"  PHASE 1: NC GATE CHECK")
    print(f"{'='*70}\n", flush=True)

    cache_path = RESULTS_DIR / "cti_generation_nc_gate.json"

    results = {}
    for model_key in NC_GATE_MODELS:
        print(f"\n--- {MODELS[model_key][1]} ---", flush=True)
        t0 = time.time()

        try:
            model, tokenizer = load_model_and_tokenizer(model_key, for_generation=True)
            nc_result = measure_nc_degree(model, tokenizer, model_key)
            nc_result["model"] = MODELS[model_key][1]
            nc_result["hf_id"] = MODELS[model_key][0]
            nc_result["arch"] = MODELS[model_key][4]
            nc_result["time_s"] = time.time() - t0
            results[model_key] = nc_result
            free_model(model)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results[model_key] = {
                "model": MODELS[model_key][1],
                "error": str(e),
                "r2_nc": None
            }

    # Evaluate gate criterion
    # CORRECTED: R^2_NC = cos^2(h, w_y) ~ 1/d for random vectors.
    # NC alignment ratio = R^2_NC * d measures alignment vs random.
    # Threshold: alignment_ratio > 10 (at least 10x random baseline).
    valid = [r for r in results.values() if r.get("r2_nc") is not None]
    passing = [r for r in valid if r.get("nc_alignment_ratio", 0) > 10]

    gate_result = {
        "n_tested": len(valid),
        "n_passing": len(passing),
        "pass_criterion": "NC alignment ratio (R^2_NC * d) > 10 for at least 3 models",
        "gate_pass": len(passing) >= 3,
        "note": "R^2_NC = cos^2(h, w_y). Random baseline = 1/d. "
                "Threshold 0.3 was dimensionally incorrect; corrected to ratio > 10.",
        "models": results
    }

    print(f"\n{'='*70}")
    print(f"  NC GATE RESULT: {'PASS' if gate_result['gate_pass'] else 'FAIL'}")
    print(f"  {len(passing)}/{len(valid)} models with alignment_ratio > 10")
    for k, r in results.items():
        r2 = r.get("r2_nc")
        ratio = r.get("nc_alignment_ratio")
        if ratio is not None:
            status = "PASS" if ratio > 10 else "FAIL"
            print(f"    {r.get('model', k):20s}  R^2_NC={r2:.4f}  "
                  f"ratio={ratio:.1f}x  [{status}]")
        elif r2 is not None:
            print(f"    {r.get('model', k):20s}  R^2_NC={r2:.4f}  [NO RATIO]")
        else:
            print(f"    {r.get('model', k):20s}  R^2_NC=N/A  [ERROR]")
    print(f"{'='*70}\n", flush=True)

    with open(cache_path, "w") as f:
        json.dump(gate_result, f, indent=2)
    print(f"Saved: {cache_path}", flush=True)

    return gate_result


# ============================================================
# PHASE 2: KAPPA EXTRACTION
# ============================================================
def compute_kappa_stats(W_U, batch_size=1000):
    """Compute kappa_bar and geometric baselines from W_U.

    For each token v, kappa_v = min_{j!=v} ||w_v - w_j|| (raw Euclidean).
    We normalize rows to unit norm first (Proxy A).

    Also computes: effective_rank, mean_cossim, condition_number.
    """
    V, d = W_U.shape
    print(f"    Computing kappa for V={V}, d={d}...", flush=True)

    # Normalize rows to unit norm
    norms = np.linalg.norm(W_U, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    W_norm = W_U / norms

    # Compute min nearest-neighbor distance using batched GPU computation
    W_tensor = torch.tensor(W_norm, device=DEVICE, dtype=torch.float32)

    min_dists = []
    for i in range(0, V, batch_size):
        end = min(i + batch_size, V)
        batch = W_tensor[i:end]  # (B, d)

        # Compute distances to ALL tokens
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a@b^T
        # Since unit norm: = 2 - 2*a@b^T
        sims = batch @ W_tensor.T  # (B, V)
        dists_sq = 2.0 - 2.0 * sims  # (B, V)
        dists_sq = torch.clamp(dists_sq, min=0.0)

        # Mask self-distances
        idx = torch.arange(i, end, device=DEVICE)
        dists_sq[torch.arange(end - i, device=DEVICE), idx] = float("inf")

        # Min distance for each token in batch
        min_d_sq, _ = dists_sq.min(dim=1)
        min_dists.append(torch.sqrt(min_d_sq).cpu().numpy())

    min_dists = np.concatenate(min_dists)
    kappa_bar = float(min_dists.mean())
    kappa_std = float(min_dists.std())
    kappa_cv = kappa_std / (kappa_bar + 1e-10)

    # Effective rank: exp(entropy of normalized singular values)
    print(f"    Computing SVD for effective rank...", flush=True)
    try:
        sv = svdvals(W_norm[:min(V, 10000), :])  # Subsample for speed
        sv_norm = sv / sv.sum()
        sv_norm = sv_norm[sv_norm > 1e-10]
        eff_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm))))
    except Exception:
        eff_rank = float("nan")

    # Mean cosine similarity (subsample for large V)
    n_sample = min(V, 5000)
    idx = np.random.choice(V, n_sample, replace=False)
    W_sub = W_tensor[idx]
    cossim_matrix = W_sub @ W_sub.T
    # Exclude diagonal
    mask = ~torch.eye(n_sample, dtype=torch.bool, device=DEVICE)
    mean_cossim = float(cossim_matrix[mask].mean().cpu())

    # Condition number (subsample)
    try:
        cond_num = float(sv[0] / sv[min(len(sv)-1, d-1)])
    except Exception:
        cond_num = float("nan")

    del W_tensor
    torch.cuda.empty_cache()

    return {
        "kappa_bar": kappa_bar,
        "kappa_std": kappa_std,
        "kappa_cv": kappa_cv,
        "effective_rank": eff_rank,
        "mean_cossim": mean_cossim,
        "condition_number": cond_num,
        "V": int(V),
        "d_model": int(d),
    }


def compute_random_kappa(V, d, n_repeats=3):
    """Compute kappa for random W_U (null check H_gen3)."""
    kappas = []
    for rep in range(n_repeats):
        W_rand = np.random.randn(V, d).astype(np.float32) / np.sqrt(d)
        norms = np.linalg.norm(W_rand, axis=1, keepdims=True)
        W_rand = W_rand / np.maximum(norms, 1e-10)

        W_t = torch.tensor(W_rand, device=DEVICE)
        # Subsample for speed with large V
        n_query = min(V, 5000)
        query_idx = np.random.choice(V, n_query, replace=False)
        query = W_t[query_idx]

        sims = query @ W_t.T
        dists_sq = 2.0 - 2.0 * sims
        dists_sq = torch.clamp(dists_sq, min=0.0)
        # Mask self
        for qi, vi in enumerate(query_idx):
            dists_sq[qi, vi] = float("inf")
        min_d_sq, _ = dists_sq.min(dim=1)
        kappas.append(float(torch.sqrt(min_d_sq).mean().cpu()))

        del W_t, query, sims, dists_sq
        torch.cuda.empty_cache()

    return {
        "kappa_random_mean": float(np.mean(kappas)),
        "kappa_random_std": float(np.std(kappas)),
    }


def run_kappa_extraction(model_keys=None):
    """Phase 2: Extract W_U and compute kappa for all models."""
    print(f"\n{'='*70}")
    print(f"  PHASE 2: KAPPA EXTRACTION (Proxy A)")
    print(f"{'='*70}\n", flush=True)

    cache_path = RESULTS_DIR / "cti_generation_kappa.json"

    # Load existing cache
    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} cached results", flush=True)
    else:
        results = {}

    if model_keys is None:
        model_keys = list(MODELS.keys())

    for model_key in model_keys:
        if model_key in results and "kappa_bar" in results[model_key]:
            print(f"  Skipping {model_key} (cached)", flush=True)
            continue

        info = MODELS[model_key]
        print(f"\n--- {info[1]} ({info[0]}) ---", flush=True)
        t0 = time.time()

        try:
            # Try normal model loading first, fall back to direct checkpoint
            W_U = None
            try:
                model, tokenizer = load_model_and_tokenizer(model_key, for_generation=True)
                W_U = get_unembedding_matrix(model, model_key)
                free_model(model)
            except Exception as load_err:
                print(f"    Normal load failed: {load_err}", flush=True)
                print(f"    Trying direct checkpoint extraction...", flush=True)
                W_U = get_unembedding_from_checkpoint(model_key)

            # Actual vocab size from W_U
            actual_V = W_U.shape[0]

            # Compute kappa and baselines
            stats = compute_kappa_stats(W_U)

            # Random null check
            print(f"    Computing random kappa null...", flush=True)
            rand_stats = compute_random_kappa(actual_V, W_U.shape[1])

            result = {
                "model": info[1],
                "hf_id": info[0],
                "params_M": info[2],
                "vocab_size": actual_V,
                "arch": info[4],
                "tier": info[5],
                **stats,
                **rand_stats,
                "time_s": time.time() - t0,
            }
            results[model_key] = result

            # Save after each model (resume-safe)
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

            print(f"    kappa_bar={stats['kappa_bar']:.6f}, "
                  f"kappa_random={rand_stats['kappa_random_mean']:.6f}, "
                  f"eff_rank={stats['effective_rank']:.1f}, "
                  f"time={time.time()-t0:.1f}s", flush=True)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[model_key] = {
                "model": info[1],
                "error": str(e),
            }
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n  Kappa extraction complete: {len(results)} models", flush=True)
    return results


# ============================================================
# PHASE 3: PPL EVALUATION
# ============================================================
def compute_perplexity_wikitext(model, tokenizer, max_tokens=100000, seq_len=512):
    """Compute perplexity on WikiText-103 validation split.

    Uses sliding window with stride = seq_len (non-overlapping).
    Returns token-level perplexity.
    """
    from datasets import load_dataset

    print(f"    Loading WikiText-103 validation...", flush=True)
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

    # Concatenate all text
    full_text = "\n\n".join([x["text"] for x in wiki if x["text"].strip()])

    # Tokenize the full text
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"][0]

    # Limit tokens
    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]

    total_tokens = len(input_ids)
    print(f"    Total tokens: {total_tokens}", flush=True)

    # Compute perplexity with sliding window
    total_loss = 0.0
    total_counted = 0

    model.eval()
    with torch.no_grad():
        for begin in range(0, total_tokens - 1, seq_len):
            end = min(begin + seq_len, total_tokens)
            chunk = input_ids[begin:end].unsqueeze(0).to(DEVICE)

            outputs = model(chunk, labels=chunk)
            loss = outputs.loss

            n_tok = chunk.shape[1] - 1  # loss is averaged over seq_len-1
            total_loss += loss.item() * n_tok
            total_counted += n_tok

    if total_counted == 0:
        return float("inf")

    avg_ce = total_loss / total_counted
    ppl = float(np.exp(avg_ce))
    print(f"    PPL = {ppl:.2f} (CE = {avg_ce:.4f}, n_tokens = {total_counted})",
          flush=True)
    return ppl


def run_ppl_evaluation(model_keys=None):
    """Phase 3: Compute PPL on WikiText-103 validation."""
    print(f"\n{'='*70}")
    print(f"  PHASE 3: PERPLEXITY EVALUATION (WikiText-103)")
    print(f"{'='*70}\n", flush=True)

    cache_path = RESULTS_DIR / "cti_generation_ppl.json"

    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} cached results", flush=True)
    else:
        results = {}

    if model_keys is None:
        model_keys = list(MODELS.keys())

    for model_key in model_keys:
        if model_key in results and "ppl" in results[model_key]:
            print(f"  Skipping {model_key} (cached: PPL={results[model_key]['ppl']:.2f})",
                  flush=True)
            continue

        info = MODELS[model_key]
        print(f"\n--- {info[1]} ({info[0]}) ---", flush=True)
        t0 = time.time()

        try:
            model, tokenizer = load_model_and_tokenizer(model_key, for_generation=True)
            ppl = compute_perplexity_wikitext(model, tokenizer)

            results[model_key] = {
                "model": info[1],
                "hf_id": info[0],
                "ppl": ppl,
                "log_ppl": float(np.log(ppl)),
                "time_s": time.time() - t0,
            }

            free_model(model)

            # Save after each model
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[model_key] = {"model": info[1], "error": str(e)}
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n  PPL evaluation complete: {len(results)} models", flush=True)
    return results


# ============================================================
# PHASE 4: HYPOTHESIS TESTING
# ============================================================
def run_hypothesis_tests():
    """Phase 4: Test all pre-registered hypotheses."""
    print(f"\n{'='*70}")
    print(f"  PHASE 4: HYPOTHESIS TESTING")
    print(f"{'='*70}\n", flush=True)

    # Load cached data
    kappa_path = RESULTS_DIR / "cti_generation_kappa.json"
    ppl_path = RESULTS_DIR / "cti_generation_ppl.json"

    if not kappa_path.exists() or not ppl_path.exists():
        print("  ERROR: Must run kappa_extract and ppl_eval first!", flush=True)
        return None

    with open(kappa_path) as f:
        kappa_data = json.load(f)
    with open(ppl_path) as f:
        ppl_data = json.load(f)

    # Merge data — only include models with both kappa and PPL
    merged = {}
    for key in kappa_data:
        if key in ppl_data and "kappa_bar" in kappa_data[key] and "ppl" in ppl_data[key]:
            merged[key] = {**kappa_data[key], **ppl_data[key]}

    if len(merged) < 3:
        print(f"  ERROR: Only {len(merged)} models with both kappa and PPL!", flush=True)
        return None

    print(f"  Models with both kappa and PPL: {len(merged)}", flush=True)

    # Extract arrays
    keys = sorted(merged.keys())
    kappa = np.array([merged[k]["kappa_bar"] for k in keys])
    log_ppl = np.array([merged[k]["log_ppl"] for k in keys])
    params = np.array([merged[k]["params_M"] for k in keys])
    vocab = np.array([merged[k]["vocab_size"] for k in keys])
    arch = np.array([merged[k]["arch"] for k in keys])
    names = [merged[k]["model"] for k in keys]

    # Also get baselines
    eff_rank = np.array([merged[k].get("effective_rank", np.nan) for k in keys])
    mean_cossim = np.array([merged[k].get("mean_cossim", np.nan) for k in keys])
    cond_num = np.array([merged[k].get("condition_number", np.nan) for k in keys])
    kappa_random = np.array([merged[k].get("kappa_random_mean", np.nan) for k in keys])

    results = {"n_models": len(keys), "models": names, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---- H_gen1: kappa correlates with log(PPL) ----
    r_all, p_all = pearsonr(kappa, log_ppl)
    results["H_gen1"] = {
        "description": "Pearson r(kappa_bar, log(PPL)) across all models",
        "r": float(r_all),
        "p": float(p_all),
        "n": len(keys),
        "threshold": "r < -0.80",
        "pass": bool(r_all < -0.80),
    }
    print(f"\n  H_gen1: r={r_all:.4f}, p={p_all:.4f} -> {'PASS' if r_all < -0.80 else 'FAIL'}")

    # ---- H_gen2: alpha_gen in [0.5, 3.5] ----
    # Fit: log_ppl = -alpha * kappa + C (or with log(V-1))
    from numpy.polynomial import polynomial as P
    slope, intercept = np.polyfit(kappa, log_ppl, 1)
    alpha_gen = -slope
    results["H_gen2"] = {
        "description": "alpha_gen magnitude in [0.5, 3.5]",
        "alpha_gen": float(alpha_gen),
        "slope": float(slope),
        "intercept": float(intercept),
        "threshold": "[0.5, 3.5]",
        "pass": bool(0.5 <= abs(alpha_gen) <= 3.5),
    }
    print(f"  H_gen2: alpha_gen={alpha_gen:.4f} -> {'PASS' if 0.5 <= abs(alpha_gen) <= 3.5 else 'FAIL'}")

    # ---- H_gen3: Random W_U null check ----
    valid_random = ~np.isnan(kappa_random)
    if valid_random.sum() >= 3:
        r_rand, p_rand = pearsonr(kappa_random[valid_random], log_ppl[valid_random])
        results["H_gen3"] = {
            "description": "Random W_U |rho| < 0.30",
            "r_random": float(r_rand),
            "p_random": float(p_rand),
            "threshold": "|r| < 0.30",
            "pass": bool(abs(r_rand) < 0.30),
        }
        print(f"  H_gen3: r_random={r_rand:.4f} -> {'PASS' if abs(r_rand) < 0.30 else 'FAIL'}")

    # ---- H_gen8: Partial correlation controlling for model size ----
    log_params = np.log(params)
    # Partial correlation: r(kappa, log_ppl | log_params)
    # Regress both on log_params, correlate residuals
    kappa_resid = kappa - np.polyval(np.polyfit(log_params, kappa, 1), log_params)
    ppl_resid = log_ppl - np.polyval(np.polyfit(log_params, log_ppl, 1), log_params)
    r_partial, p_partial = pearsonr(kappa_resid, ppl_resid)
    results["H_gen8"] = {
        "description": "Partial r(kappa, log(PPL) | log(N_params)) > 0.50",
        "r_partial": float(r_partial),
        "p_partial": float(p_partial),
        "threshold": "|r_partial| > 0.50",
        "pass": bool(abs(r_partial) > 0.50),
    }
    print(f"  H_gen8: r_partial={r_partial:.4f} -> {'PASS' if abs(r_partial) > 0.50 else 'FAIL'}")

    # ---- H_gen9: kappa outperforms simpler baselines ----
    baseline_rs = {}
    for name, vals in [("effective_rank", eff_rank), ("mean_cossim", mean_cossim),
                       ("condition_number", cond_num)]:
        valid = ~np.isnan(vals)
        if valid.sum() >= 3:
            r_b, _ = pearsonr(vals[valid], log_ppl[valid])
            baseline_rs[name] = float(r_b)

    results["H_gen9"] = {
        "description": "kappa outperforms simpler geometric baselines",
        "r_kappa": float(r_all),
        "baseline_rs": baseline_rs,
        "pass": bool(abs(r_all) > max(abs(v) for v in baseline_rs.values())) if baseline_rs else None,
    }
    max_baseline = max(abs(v) for v in baseline_rs.values()) if baseline_rs else 0
    print(f"  H_gen9: |r_kappa|={abs(r_all):.4f} vs max_baseline={max_baseline:.4f} -> "
          f"{'PASS' if abs(r_all) > max_baseline else 'FAIL'}")

    # ---- Fixed-V analysis (Tier 1) ----
    fixed_v_mask = np.array([k in FIXED_V_MODELS for k in keys])
    if fixed_v_mask.sum() >= 3:
        kappa_fv = kappa[fixed_v_mask]
        log_ppl_fv = log_ppl[fixed_v_mask]
        arch_fv = arch[fixed_v_mask]
        names_fv = [n for n, m in zip(names, fixed_v_mask) if m]

        r_fv, p_fv = pearsonr(kappa_fv, log_ppl_fv)
        slope_fv, intercept_fv = np.polyfit(kappa_fv, log_ppl_fv, 1)

        results["fixed_v_analysis"] = {
            "n": int(fixed_v_mask.sum()),
            "r": float(r_fv),
            "p": float(p_fv),
            "alpha_gen": float(-slope_fv),
            "intercept": float(intercept_fv),
            "models": names_fv,
        }
        print(f"\n  Fixed-V group (n={fixed_v_mask.sum()}): r={r_fv:.4f}, "
              f"p={p_fv:.6f}, alpha_gen={-slope_fv:.4f}")

        # H_gen10: Architecture independence (F-test for interaction)
        is_transformer = np.array([a == "transformer" for a in arch_fv], dtype=float)
        # Fit with and without architecture indicator
        X_base = np.column_stack([kappa_fv, np.ones(len(kappa_fv))])
        X_full = np.column_stack([kappa_fv, is_transformer, kappa_fv * is_transformer,
                                  np.ones(len(kappa_fv))])

        from numpy.linalg import lstsq
        beta_base, res_base, _, _ = lstsq(X_base, log_ppl_fv, rcond=None)
        beta_full, res_full, _, _ = lstsq(X_full, log_ppl_fv, rcond=None)

        rss_base = float(np.sum((log_ppl_fv - X_base @ beta_base)**2))
        rss_full = float(np.sum((log_ppl_fv - X_full @ beta_full)**2))

        df_extra = 2  # architecture indicator + interaction
        df_resid = len(kappa_fv) - 4
        if df_resid > 0 and rss_full > 0:
            f_stat = ((rss_base - rss_full) / df_extra) / (rss_full / df_resid)
            from scipy.stats import f as f_dist
            p_arch = float(1 - f_dist.cdf(f_stat, df_extra, df_resid))
        else:
            f_stat = 0.0
            p_arch = 1.0

        results["H_gen10"] = {
            "description": "Architecture independence (F-test p > 0.05)",
            "f_stat": float(f_stat),
            "p_arch": p_arch,
            "pass": bool(p_arch > 0.05),
        }
        print(f"  H_gen10: F={f_stat:.4f}, p_arch={p_arch:.4f} -> "
              f"{'PASS' if p_arch > 0.05 else 'FAIL'}")

    # ---- H_gen4: LOAO within Pythia ----
    pythia_mask = np.array(["pythia" in k for k in keys])
    if pythia_mask.sum() >= 4:
        kp = kappa[pythia_mask]
        lp = log_ppl[pythia_mask]
        n_p = len(kp)
        residuals = []
        baseline_residuals = []
        for i in range(n_p):
            train_k = np.delete(kp, i)
            train_l = np.delete(lp, i)
            s, c = np.polyfit(train_k, train_l, 1)
            pred = s * kp[i] + c
            residuals.append(abs(lp[i] - pred))
            baseline_residuals.append(abs(lp[i] - train_l.mean()))

        mean_resid = float(np.mean(residuals))
        mean_baseline = float(np.mean(baseline_residuals))
        beats_baseline = sum(r < b for r, b in zip(residuals, baseline_residuals))

        results["H_gen4_pythia"] = {
            "description": "LOAO within Pythia: residual < 0.15 nats",
            "mean_residual_nats": mean_resid,
            "mean_baseline_residual": mean_baseline,
            "beats_baseline": int(beats_baseline),
            "n_folds": n_p,
            "pass_residual": bool(mean_resid < 0.15),
            "pass_baseline": bool(beats_baseline >= n_p - 1),
            "pass": bool(mean_resid < 0.15 and beats_baseline >= n_p - 1),
        }
        print(f"  H_gen4 (Pythia LOAO): mean_resid={mean_resid:.4f} nats, "
              f"beats_baseline={beats_baseline}/{n_p} -> "
              f"{'PASS' if mean_resid < 0.15 and beats_baseline >= n_p - 1 else 'FAIL'}")

    # ---- H_gen13: LOAO within Mamba ----
    mamba_mask = np.array(["mamba" in k and "codestral" not in k for k in keys])
    if mamba_mask.sum() >= 4:
        km = kappa[mamba_mask]
        lm = log_ppl[mamba_mask]
        n_m = len(km)
        residuals_m = []
        baseline_m = []
        for i in range(n_m):
            train_k = np.delete(km, i)
            train_l = np.delete(lm, i)
            s, c = np.polyfit(train_k, train_l, 1)
            pred = s * km[i] + c
            residuals_m.append(abs(lm[i] - pred))
            baseline_m.append(abs(lm[i] - train_l.mean()))

        mean_resid_m = float(np.mean(residuals_m))
        mean_base_m = float(np.mean(baseline_m))
        beats_m = sum(r < b for r, b in zip(residuals_m, baseline_m))

        results["H_gen13_mamba"] = {
            "description": "LOAO within Mamba: residual < 0.15 nats",
            "mean_residual_nats": mean_resid_m,
            "mean_baseline_residual": mean_base_m,
            "beats_baseline": int(beats_m),
            "n_folds": n_m,
            "pass": bool(mean_resid_m < 0.15 and beats_m >= n_m - 1),
        }
        print(f"  H_gen13 (Mamba LOAO): mean_resid={mean_resid_m:.4f} nats, "
              f"beats_baseline={beats_m}/{n_m}")

    # ---- H_gen7: Residual vs log(V-1) ----
    unique_vs = np.unique(vocab)
    if len(unique_vs) >= 3:
        residuals_v = log_ppl - (slope * kappa + intercept)
        log_v = np.log(vocab - 1)
        r_v, p_v = pearsonr(log_v, residuals_v)
        results["H_gen7"] = {
            "description": "Residual correlates with log(V-1)",
            "r_residual_logV": float(r_v),
            "p": float(p_v),
            "direction_correct": bool(r_v > 0),
            "n_unique_V": int(len(unique_vs)),
        }
        print(f"  H_gen7: r(residual, log(V-1))={r_v:.4f}, direction={'correct' if r_v > 0 else 'wrong'}")

    # ---- Print summary table ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Model':<20s} {'Arch':<12s} {'V':>8s} {'kappa':>10s} "
          f"{'PPL':>8s} {'log(PPL)':>10s}")
    print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")
    for k in keys:
        m = merged[k]
        print(f"  {m['model']:<20s} {m['arch']:<12s} {m['vocab_size']:>8d} "
              f"{m['kappa_bar']:>10.6f} {m['ppl']:>8.2f} {m['log_ppl']:>10.4f}")
    print()

    # Save
    output_path = RESULTS_DIR / "cti_generation_law.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {output_path}", flush=True)

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    t_start = time.time()

    if PHASE == "nc_gate":
        result = run_nc_gate()
        if not result.get("gate_pass"):
            print("\n  NC GATE FAILED. Aborting generation experiment.")
            print("  The NC assumption is too weak for the generation law.")
            sys.exit(1)

    elif PHASE == "kappa_extract":
        run_kappa_extraction()

    elif PHASE == "ppl_eval":
        run_ppl_evaluation()

    elif PHASE == "hypothesis_test":
        run_hypothesis_tests()

    elif PHASE == "tier1":
        # Run only Tier 1 (fixed-V) models through all phases
        tier1_keys = FIXED_V_MODELS
        print(f"  Running Tier 1 only: {len(tier1_keys)} models\n")
        run_kappa_extraction(tier1_keys)
        run_ppl_evaluation(tier1_keys)
        run_hypothesis_tests()

    elif PHASE == "all":
        result = run_nc_gate()
        if not result.get("gate_pass"):
            print("\n  NC GATE FAILED. Aborting.")
            sys.exit(1)
        run_kappa_extraction()
        run_ppl_evaluation()
        run_hypothesis_tests()

    else:
        print(f"  Unknown phase: {PHASE}")
        print(f"  Valid phases: nc_gate, kappa_extract, ppl_eval, hypothesis_test, tier1, all")
        sys.exit(1)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
