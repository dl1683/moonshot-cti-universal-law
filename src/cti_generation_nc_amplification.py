#!/usr/bin/env python -u
"""
CGF Generation Law: NC Amplification Test
==========================================
Tests the Amplification Theorem (Section 3.8.1):
  alpha_gen = alpha_race * lambda_NC

where:
  alpha_race = sqrt(4/pi) / sqrt(1 - rho_local) ~ 1.25 (from measured rho_local)
  lambda_NC ~ 1.66 (predicted from alpha_gen = 2.08)

Direct measurements per model:
1. gamma_NC: mean projection coefficient E[h @ w_y / ||w_y||^2]
2. cos_NC: mean cosine alignment E[cos(h, w_y)]
3. logit_margin: mean(z_y - max_{j!=y} z_j)
4. effective_kappa: logit_margin / (sigma_noise * sqrt(d))
5. amplification: effective_kappa / kappa_bar

If the theorem holds:
- gamma_NC should increase with kappa_bar (better models = better alignment)
- amplification should be approximately constant ~ lambda_NC ~ 1.66
- Or: amplification increases slightly with kappa_bar (explaining the super-linear slope)
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

# Models that support forward passes (exclude Mamba — HF loading fails)
NC_AMP_MODELS = [
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


def measure_nc_amplification(model, tokenizer, model_key, n_tokens=5000):
    """Measure NC alignment and logit margin for a model."""
    from datasets import load_dataset

    print(f"  Measuring NC amplification for {model_key}...", flush=True)

    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [x["text"] for x in wiki if x["text"].strip()]

    # Get W_U
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
    W_U_device = W_U.to(DEVICE)
    print(f"    W_U: {V} x {d}", flush=True)

    # Collect per-position measurements
    gammas = []         # projection coefficient
    cos_angles = []     # cosine alignment
    margins = []        # z_y - max_{j!=y} z_j
    h_norms = []        # ||h(x)||
    wy_norms = []       # ||w_y||
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
            h = outputs.hidden_states[-1].float()[0, :-1, :]  # (seq-1, d)
            targets = enc["input_ids"][0, 1:]  # (seq-1,)
            logits = outputs.logits.float()[0, :-1, :]  # (seq-1, V)
            n_pos = h.shape[0]

            # Get w_y for each target
            w_y = W_U_device[targets]  # (n, d)

            # gamma = h @ w_y / ||w_y||^2
            wy_norm_sq = (w_y * w_y).sum(dim=1) + 1e-10  # (n,)
            h_dot_wy = (h * w_y).sum(dim=1)  # (n,)
            gamma = h_dot_wy / wy_norm_sq  # (n,)

            # cos(h, w_y)
            h_norm = torch.norm(h, dim=1) + 1e-10  # (n,)
            wy_norm = torch.sqrt(wy_norm_sq)  # (n,)
            cos_hw = h_dot_wy / (h_norm * wy_norm)  # (n,)

            # logit margin: z_y - max_{j!=y} z_j
            z_y = logits[torch.arange(n_pos, device=DEVICE), targets]  # (n,)
            # Mask out the target token
            logits_masked = logits.clone()
            logits_masked[torch.arange(n_pos, device=DEVICE), targets] = -float("inf")
            z_max_other = logits_masked.max(dim=1).values  # (n,)
            margin = z_y - z_max_other  # (n,)

            gammas.append(gamma.cpu().numpy())
            cos_angles.append(cos_hw.cpu().numpy())
            margins.append(margin.cpu().numpy())
            h_norms.append(h_norm.cpu().numpy())
            wy_norms.append(wy_norm.cpu().numpy())
            total_tokens += n_pos

    print(f"    Collected {total_tokens} positions", flush=True)

    gammas = np.concatenate(gammas)
    cos_angles = np.concatenate(cos_angles)
    margins = np.concatenate(margins)
    h_norms_arr = np.concatenate(h_norms)
    wy_norms_arr = np.concatenate(wy_norms)

    # Compute summary statistics
    result = {
        "model": model_key,
        "V": int(V),
        "d_model": int(d),
        "n_tokens": total_tokens,
        # NC alignment
        "gamma_NC_mean": float(np.mean(gammas)),
        "gamma_NC_median": float(np.median(gammas)),
        "gamma_NC_std": float(np.std(gammas)),
        "cos_NC_mean": float(np.mean(cos_angles)),
        "cos_NC_median": float(np.median(cos_angles)),
        "cos_NC_std": float(np.std(cos_angles)),
        # Logit margin
        "margin_mean": float(np.mean(margins)),
        "margin_median": float(np.median(margins)),
        "margin_std": float(np.std(margins)),
        "margin_positive_frac": float((margins > 0).mean()),
        # Norms
        "h_norm_mean": float(np.mean(h_norms_arr)),
        "wy_norm_mean": float(np.mean(wy_norms_arr)),
    }

    print(f"    gamma_NC = {result['gamma_NC_mean']:.4f} +/- {result['gamma_NC_std']:.4f}")
    print(f"    cos(h,w_y) = {result['cos_NC_mean']:.4f}")
    print(f"    margin = {result['margin_mean']:.4f} +/- {result['margin_std']:.4f}")
    print(f"    frac correct = {result['margin_positive_frac']:.4f}")

    return result


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy.stats import pearsonr

    print("=" * 72)
    print("  CGF GENERATION LAW: NC AMPLIFICATION TEST")
    print("  Theorem 3.8.1: alpha_gen = alpha_race * lambda_NC")
    print("=" * 72)

    cache_path = RESULTS_DIR / "cti_generation_nc_amplification.json"
    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} cached results", flush=True)
    else:
        results = {}

    # Load kappa data
    with open(RESULTS_DIR / "cti_generation_kappa.json") as f:
        kappa_data = json.load(f)

    for model_key, hf_id, name in NC_AMP_MODELS:
        if model_key in results and "gamma_NC_mean" in results[model_key]:
            print(f"  Skipping {name} (cached: gamma={results[model_key]['gamma_NC_mean']:.4f})")
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

            result = measure_nc_amplification(model, tokenizer, model_key)
            result["name"] = name
            result["hf_id"] = hf_id
            result["time_s"] = time.time() - t0

            # Add kappa data
            if model_key in kappa_data:
                result["kappa_bar"] = kappa_data[model_key]["kappa_bar"]

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

    # ==== ANALYSIS ====
    print(f"\n{'=' * 72}")
    print("  NC AMPLIFICATION ANALYSIS")
    print("=" * 72)

    # Collect valid models
    valid_keys = [k for k in results if "gamma_NC_mean" in results[k] and "kappa_bar" in results[k]]

    if len(valid_keys) < 3:
        print("  Too few models for analysis")
        return

    print(f"\n  {'Model':<16s} {'kappa':>8s} {'gamma':>8s} {'cos_NC':>8s} "
          f"{'margin':>8s} {'frac_ok':>8s}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    kappas = []
    gammas = []
    cosines = []
    margins_mean = []
    margins_std = []

    for k in valid_keys:
        r = results[k]
        name = r.get("name", k)
        kap = r["kappa_bar"]
        gam = r["gamma_NC_mean"]
        cos_nc = r["cos_NC_mean"]
        mar = r["margin_mean"]
        frac = r["margin_positive_frac"]

        print(f"  {name:<16s} {kap:>8.4f} {gam:>8.4f} {cos_nc:>8.4f} "
              f"{mar:>8.3f} {frac:>8.4f}")

        kappas.append(kap)
        gammas.append(gam)
        cosines.append(cos_nc)
        margins_mean.append(mar)
        margins_std.append(r["margin_std"])

    kappas = np.array(kappas)
    gammas = np.array(gammas)
    cosines = np.array(cosines)
    margins_mean = np.array(margins_mean)
    margins_std = np.array(margins_std)

    n = len(kappas)

    # Test 1: Does gamma_NC increase with kappa_bar?
    print(f"\n  === Test 1: gamma_NC vs kappa_bar ===")
    if n >= 3:
        r_gk, p_gk = pearsonr(kappas, gammas)
        print(f"  r(gamma, kappa) = {r_gk:.4f} (p = {p_gk:.4f})")
        print(f"  VERDICT: {'PASS' if r_gk > 0.5 else 'FAIL'} "
              f"(expect positive correlation)")

    # Test 2: Does margin increase with kappa_bar?
    print(f"\n  === Test 2: margin vs kappa_bar ===")
    if n >= 3:
        r_mk, p_mk = pearsonr(kappas, margins_mean)
        print(f"  r(margin, kappa) = {r_mk:.4f} (p = {p_mk:.4f})")

    # Test 3: Compute effective kappa and amplification
    print(f"\n  === Test 3: Amplification factor ===")
    # effective kappa = margin / sigma_margin (signal-to-noise in logit space)
    kappa_eff = margins_mean / (margins_std + 1e-10)
    amplification = kappa_eff / (kappas + 1e-10)

    print(f"  {'Model':<16s} {'kappa_bar':>10s} {'kappa_eff':>10s} "
          f"{'lambda_NC':>10s}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
    for i, k in enumerate(valid_keys):
        name = results[k].get("name", k)
        print(f"  {name:<16s} {kappas[i]:>10.4f} {kappa_eff[i]:>10.4f} "
              f"{amplification[i]:>10.4f}")

    mean_lambda = float(np.mean(amplification))
    std_lambda = float(np.std(amplification))
    print(f"\n  Mean lambda_NC = {mean_lambda:.4f} +/- {std_lambda:.4f}")
    print(f"  Predicted lambda_NC = 1.66 (from alpha_gen/alpha_race = 2.08/1.25)")
    print(f"  Deviation = {abs(mean_lambda - 1.66):.4f}")

    # Test 4: Product kappa_bar * gamma vs log(PPL)
    print(f"\n  === Test 4: kappa * gamma as composite predictor ===")
    kappa_gamma = kappas * gammas
    # Load PPL data
    try:
        with open(RESULTS_DIR / "cti_generation_ppl_pile.json") as f:
            pile_ppl = json.load(f)
        with open(RESULTS_DIR / "cti_generation_ppl.json") as f:
            wiki_ppl = json.load(f)

        log_ppls = []
        for k in valid_keys:
            if k in pile_ppl and "log_ppl" in pile_ppl[k]:
                log_ppls.append(pile_ppl[k]["log_ppl"])
            elif k in wiki_ppl and "ppl" in wiki_ppl[k]:
                log_ppls.append(np.log(wiki_ppl[k]["ppl"]))
            else:
                log_ppls.append(np.nan)
        log_ppls = np.array(log_ppls)
        valid_ppl = ~np.isnan(log_ppls)

        if valid_ppl.sum() >= 3:
            r_kp, p_kp = pearsonr(kappas[valid_ppl], log_ppls[valid_ppl])
            r_kgp, p_kgp = pearsonr(kappa_gamma[valid_ppl], log_ppls[valid_ppl])
            r_gp, p_gp = pearsonr(gammas[valid_ppl], log_ppls[valid_ppl])
            r_mp, p_mp = pearsonr(margins_mean[valid_ppl], log_ppls[valid_ppl])

            print(f"  r(kappa, log(PPL))          = {r_kp:.4f} (p = {p_kp:.4f})")
            print(f"  r(kappa*gamma, log(PPL))    = {r_kgp:.4f} (p = {p_kgp:.4f})")
            print(f"  r(gamma, log(PPL))          = {r_gp:.4f} (p = {p_gp:.4f})")
            print(f"  r(margin, log(PPL))         = {r_mp:.4f} (p = {p_mp:.4f})")
            print(f"\n  Does kappa*gamma improve over kappa alone? "
                  f"{'YES' if abs(r_kgp) > abs(r_kp) + 0.01 else 'NO'}")

    except Exception as e:
        print(f"  Could not load PPL data: {e}")

    # Test 5: Is lambda_NC approximately constant?
    print(f"\n  === Test 5: Lambda_NC constancy ===")
    if n >= 3:
        r_la, p_la = pearsonr(kappas, amplification)
        cv_lambda = float(std_lambda / abs(mean_lambda)) if abs(mean_lambda) > 0.01 else float("inf")
        print(f"  CV(lambda_NC) = {cv_lambda:.4f}")
        print(f"  r(lambda_NC, kappa) = {r_la:.4f} (p = {p_la:.4f})")
        print(f"  If CV < 0.30 and |r| < 0.50: lambda_NC is approximately constant")
        print(f"  VERDICT: {'CONSTANT' if cv_lambda < 0.30 and abs(r_la) < 0.50 else 'VARIES'}")

    # Save analysis
    analysis = {
        "n_models": n,
        "model_keys": valid_keys,
        "mean_lambda_NC": mean_lambda,
        "std_lambda_NC": std_lambda,
        "cv_lambda_NC": float(std_lambda / abs(mean_lambda)) if abs(mean_lambda) > 0.01 else None,
        "predicted_lambda_NC": 1.66,
        "alpha_race": 1.25,
        "alpha_gen": 2.077,
    }
    results["_analysis"] = analysis

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {cache_path}")


if __name__ == "__main__":
    main()
