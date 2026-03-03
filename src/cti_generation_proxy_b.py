#!/usr/bin/env python -u
"""
CGF Generation Law: Proxy B (Whitened Kappa)
=============================================
Computes kappa from whitened W_U: W_whitened = Sigma_W^{-1/2} @ W_U^T

This extends Proxy A (raw kappa) by accounting for hidden-state noise.
Proxy B is the theoretically correct measurement:
  kappa_y = min_{j!=y} ||Sigma_W^{-1/2} (w_y - w_j)|| / sqrt(d_eff)

Also measures rho_whitened: mean off-diagonal cosine in whitened space.

Tests H_gen5: |r(kappa_whitened, logCE)| - |r(kappa_raw, logCE)| > 0.05
"""

import json
import time
import gc
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr
from scipy.linalg import svdvals

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# All models matching the 25-model generation law suite
PROXY_B_MODELS = [
    # --- Pythia (Transformer, V=50280) ---
    ("pythia-160m",    "EleutherAI/pythia-160m",           "Pythia-160M"),
    ("pythia-410m",    "EleutherAI/pythia-410m",           "Pythia-410M"),
    ("pythia-1b",      "EleutherAI/pythia-1b",             "Pythia-1B"),
    ("pythia-1.4b",    "EleutherAI/pythia-1.4b",           "Pythia-1.4B"),
    ("pythia-2.8b",    "EleutherAI/pythia-2.8b",           "Pythia-2.8B"),
    # --- Mamba (SSM, V=50280) ---
    ("mamba-130m",     "state-spaces/mamba-130m-hf",       "Mamba-130M"),
    ("mamba-370m",     "state-spaces/mamba-370m-hf",       "Mamba-370M"),
    ("mamba-790m",     "state-spaces/mamba-790m-hf",       "Mamba-790M"),
    ("mamba-1.4b",     "state-spaces/mamba-1.4b-hf",       "Mamba-1.4B"),
    ("mamba-2.8b",     "state-spaces/mamba-2.8b-hf",       "Mamba-2.8B"),
    # --- GPT-2 (Transformer, V=50257) ---
    ("gpt2",           "openai-community/gpt2",            "GPT-2"),
    # --- Qwen3 (Transformer, V=151936) ---
    ("qwen3-0.6b",     "Qwen/Qwen3-0.6B",                 "Qwen3-0.6B"),
    ("qwen3-1.7b",     "Qwen/Qwen3-1.7B",                 "Qwen3-1.7B"),
    ("qwen3-4b",       "Qwen/Qwen3-4B",                   "Qwen3-4B"),
    # --- Qwen2 (Transformer, V=151936) ---
    ("qwen2-0.5b",     "Qwen/Qwen2-0.5B",                 "Qwen2-0.5B"),
    # --- Falcon-H1 (Hybrid, V=131072) ---
    ("falcon-h1-0.5b", "tiiuae/Falcon-H1-0.5B-Base",      "Falcon-H1-0.5B"),
    ("falcon-h1-1.5b", "tiiuae/Falcon-H1-1.5B-Base",      "Falcon-H1-1.5B"),
    ("falcon-h1-3b",   "tiiuae/Falcon-H1-3B-Base",        "Falcon-H1-3B"),
    # --- SmolLM2 (Transformer, V=49152) ---
    ("smollm2-360m",   "HuggingFaceTB/SmolLM2-360M",      "SmolLM2-360M"),
    # --- Granite (Hybrid, V=49152) ---
    ("granite-micro",  "ibm-granite/granite-4.0-micro",    "Granite-Micro"),
    # --- Phi-4 (Transformer, V=100352) ---
    ("phi-4",          "microsoft/phi-4",                   "Phi-4"),
    # --- Liquid AI (Novel, V=65536) ---
    ("lfm2.5-1.2b",   "LiquidAI/LFM2.5-1.2B-Base",       "LFM2.5-1.2B"),
    # --- Mistral (Transformer, V=32768) ---
    ("mistral-7b",     "mistralai/Mistral-7B-v0.3",        "Mistral-7B"),
]


def compute_proxy_b(model, tokenizer, model_key, hf_id, n_tokens=10000):
    """Compute Proxy B: whitened kappa + rho_whitened.

    Steps:
    1. Run ~10K tokens, collect hidden states h(x) and targets y
    2. Get W_U rows w_y, compute residuals eps = h - gamma*w_y
    3. Build Sigma_W = cov(eps) (d x d)
    4. Whiten W_U: W_w = W_U @ Sigma_W^{-1/2}
    5. Compute kappa from W_w
    6. Compute rho from mean off-diagonal cosine of W_w
    """
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Computing Proxy B for {model_key}...", flush=True)

    # Load WikiText-103 validation
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [x["text"] for x in wiki if x["text"].strip()]

    # Get W_U from model
    W_U = None
    for attr_path in [
        ("lm_head", "weight"),
        ("embed_out", "weight"),
    ]:
        obj = model
        try:
            for a in attr_path:
                obj = getattr(obj, a)
            W_U = obj.detach().float().cpu().numpy()
            break
        except AttributeError:
            continue

    if W_U is None:
        # Try model.model.embed_tokens for tied weights
        try:
            W_U = model.model.embed_tokens.weight.detach().float().cpu().numpy()
        except Exception:
            raise ValueError("Cannot find W_U")

    V, d = W_U.shape
    print(f"    W_U shape: {V} x {d}", flush=True)

    W_U_tensor = torch.tensor(W_U, device=DEVICE, dtype=torch.float32)

    # Collect residuals
    all_eps = []
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

            outputs = model(**enc, output_hidden_states=True)
            h = outputs.hidden_states[-1].float()  # (1, seq_len, d)
            h = h[0, :-1, :]  # (seq_len-1, d)

            targets = enc["input_ids"][0, 1:]
            w_y = W_U_tensor[targets]  # (n, d)

            # Project h onto w_y: gamma = h @ w_y / ||w_y||^2
            w_y_norm_sq = (w_y * w_y).sum(dim=1, keepdim=True) + 1e-10
            gamma = (h * w_y).sum(dim=1, keepdim=True) / w_y_norm_sq

            eps = h - gamma * w_y  # (n, d)
            all_eps.append(eps.cpu().numpy())
            total_tokens += len(targets)

    print(f"    Collected {total_tokens} tokens", flush=True)

    if total_tokens < 100:
        raise ValueError(f"Too few tokens collected: {total_tokens}")

    eps_all = np.concatenate(all_eps, axis=0)  # (N, d)

    # Compute Sigma_W = covariance of residuals
    print(f"    Computing Sigma_W ({d}x{d})...", flush=True)
    eps_centered = eps_all - eps_all.mean(axis=0, keepdims=True)
    # Use only a subsample if too many tokens (for numerical stability)
    n_use = min(len(eps_centered), 10000)
    eps_sub = eps_centered[:n_use]
    Sigma_W = (eps_sub.T @ eps_sub) / (n_use - 1)  # (d, d)

    # Compute Sigma_W^{-1/2} via eigendecomposition
    print(f"    Computing Sigma_W^{{-1/2}} via eigendecomposition...", flush=True)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_W)

    # Regularize: clamp small eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    d_eff = float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2))

    # Sigma_W^{-1/2} = V @ diag(1/sqrt(lambda)) @ V^T
    inv_sqrt_eig = 1.0 / np.sqrt(eigenvalues)
    Sigma_W_inv_sqrt = eigenvectors @ np.diag(inv_sqrt_eig) @ eigenvectors.T

    # Whiten W_U: each row w_v -> Sigma_W^{-1/2} @ w_v
    print(f"    Whitening W_U...", flush=True)
    W_whitened = W_U @ Sigma_W_inv_sqrt.T  # (V, d)

    # Normalize whitened rows to unit norm
    norms_w = np.linalg.norm(W_whitened, axis=1, keepdims=True)
    norms_w = np.maximum(norms_w, 1e-10)
    W_w_normed = W_whitened / norms_w

    # Compute kappa_whitened (min-NN distance in whitened space)
    print(f"    Computing kappa_whitened (V={V})...", flush=True)
    W_t = torch.tensor(W_w_normed, device=DEVICE, dtype=torch.float32)

    batch_size = 1000
    min_dists = []
    for i in range(0, V, batch_size):
        end = min(i + batch_size, V)
        batch = W_t[i:end]
        sims = batch @ W_t.T
        dists_sq = 2.0 - 2.0 * sims
        dists_sq = torch.clamp(dists_sq, min=0.0)
        idx = torch.arange(i, end, device=DEVICE)
        dists_sq[torch.arange(end - i, device=DEVICE), idx] = float("inf")
        min_d_sq, _ = dists_sq.min(dim=1)
        min_dists.append(torch.sqrt(min_d_sq).cpu().numpy())

    min_dists = np.concatenate(min_dists)
    kappa_whitened = float(min_dists.mean())

    # Compute rho_whitened: mean off-diagonal cosine in whitened space
    print(f"    Computing rho_whitened (subsample)...", flush=True)
    n_sample = min(V, 5000)
    idx_sample = np.random.choice(V, n_sample, replace=False)
    W_sub = W_t[idx_sample]
    cos_matrix = W_sub @ W_sub.T  # (n, n)
    mask = ~torch.eye(n_sample, dtype=torch.bool, device=DEVICE)
    rho_whitened = float(cos_matrix[mask].mean().cpu())

    del W_t, W_U_tensor
    torch.cuda.empty_cache()

    result = {
        "model": model_key,
        "kappa_whitened": kappa_whitened,
        "kappa_whitened_std": float(min_dists.std()),
        "rho_whitened": rho_whitened,
        "d_eff": d_eff,
        "V": int(V),
        "d_model": int(d),
        "n_tokens": total_tokens,
        "sigma_W_trace": float(np.trace(Sigma_W)),
        "sigma_W_top_eigenvalue": float(eigenvalues[-1]),
    }

    print(f"    kappa_whitened = {kappa_whitened:.6f}")
    print(f"    rho_whitened   = {rho_whitened:.6f}")
    print(f"    d_eff          = {d_eff:.1f}")
    print(f"    sigma_W_trace  = {np.trace(Sigma_W):.4f}")

    return result


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 72)
    print("  CGF GENERATION LAW: PROXY B (WHITENED KAPPA)")
    print("=" * 72)

    cache_path = RESULTS_DIR / "cti_generation_proxy_b.json"
    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} cached results", flush=True)
    else:
        results = {}

    for model_key, hf_id, name in PROXY_B_MODELS:
        if model_key in results and "kappa_whitened" in results[model_key]:
            print(f"  Skipping {name} (cached: kappa_w={results[model_key]['kappa_whitened']:.4f})")
            continue

        print(f"\n--- {name} ({hf_id}) ---", flush=True)
        t0 = time.time()

        try:
            # Load model (bf16 first, fall back to fp16)
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id, torch_dtype=torch.bfloat16, trust_remote_code=True
                ).to(DEVICE)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id, torch_dtype=torch.float16, trust_remote_code=True
                ).to(DEVICE)
            model.eval()

            result = compute_proxy_b(model, tokenizer, model_key, hf_id)
            result["name"] = name
            result["hf_id"] = hf_id
            result["time_s"] = time.time() - t0

            results[model_key] = result

            # Cleanup
            del model
            gc.collect()
            torch.cuda.empty_cache()

            # Save after each model
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
    print("  PROXY B ANALYSIS")
    print("=" * 72)

    # Load Proxy A kappa and CE data for comparison
    with open(RESULTS_DIR / "cti_generation_freq_kappa.json") as f:
        kappa_a = json.load(f)
    with open(RESULTS_DIR / "cti_generation_keff.json") as f:
        keff_data = json.load(f)

    # Compare Proxy A vs Proxy B
    print(f"\n  {'Model':<20s} {'kt1K_A':>10s} {'kappa_B':>10s} {'rho_w':>10s} {'d_eff':>8s}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")

    keys_with_both = []
    for key in results:
        if "kappa_whitened" not in results[key]:
            continue
        ka_entry = kappa_a.get(key, {})
        if "error" in ka_entry or "kappa_top1000" not in ka_entry:
            continue
        if key not in keff_data or "error" in keff_data.get(key, {}):
            continue
        ka = ka_entry["kappa_top1000"]
        kb = results[key]["kappa_whitened"]
        rho_w = results[key]["rho_whitened"]
        d_eff = results[key]["d_eff"]
        name = results[key].get("name", key)
        print(f"  {name:<20s} {ka:>10.4f} {kb:>10.4f} {rho_w:>10.4f} {d_eff:>8.1f}")
        keys_with_both.append(key)

    # Correlations with log(CE)
    if len(keys_with_both) >= 3:
        kappa_a_vals = np.array([kappa_a[k]["kappa_top1000"] for k in keys_with_both])
        kappa_b_vals = np.array([results[k]["kappa_whitened"] for k in keys_with_both])
        rho_w_vals = np.array([results[k]["rho_whitened"] for k in keys_with_both])
        log_ces = np.array([np.log(keff_data[k]["mean_ce"]) for k in keys_with_both])

        # Exclude LFM outlier
        no_lfm = np.array(["lfm" not in k for k in keys_with_both])
        labels = ["All models", "Without LFM"]
        masks = [np.ones(len(keys_with_both), dtype=bool), no_lfm]

        for label, mask in zip(labels, masks):
            if mask.sum() < 3:
                continue
            r_a, p_a = pearsonr(kappa_a_vals[mask], log_ces[mask])
            r_b, p_b = pearsonr(kappa_b_vals[mask], log_ces[mask])
            print(f"\n  --- {label} (n={mask.sum()}) ---")
            print(f"    r(kappa_A, logCE) = {r_a:.4f} (p={p_a:.4f})")
            print(f"    r(kappa_B, logCE) = {r_b:.4f} (p={p_b:.4f})")
            delta = abs(r_b) - abs(r_a)
            print(f"    Delta |r| = {delta:+.4f}")
            status = "PASS" if delta > 0.05 else "FAIL"
            print(f"    H_gen5: {status} (threshold: +0.05)")

        # Fixed-V group (V ~ 50280: Pythia + Mamba + GPT-2)
        fixed_v = np.array([k.startswith("pythia") or k.startswith("mamba")
                            or k == "gpt2" for k in keys_with_both])
        if fixed_v.sum() >= 3:
            r_a_fv, _ = pearsonr(kappa_a_vals[fixed_v], log_ces[fixed_v])
            r_b_fv, _ = pearsonr(kappa_b_vals[fixed_v], log_ces[fixed_v])
            print(f"\n  --- Fixed-V (n={fixed_v.sum()}) ---")
            print(f"    r(kappa_A, logCE) = {r_a_fv:.4f}")
            print(f"    r(kappa_B, logCE) = {r_b_fv:.4f}")
            print(f"    Delta |r| = {abs(r_b_fv) - abs(r_a_fv):+.4f}")

        # rho_whitened analysis
        mean_rho = float(np.mean(rho_w_vals))
        print(f"\n  rho_whitened mean = {mean_rho:.4f}")
        print(f"  Predicted from alpha_gen=0.95: rho = {1 - (4/np.pi)/0.95**2:.4f}")

    print(f"\n  Saved: {cache_path}")


if __name__ == "__main__":
    main()
