#!/usr/bin/env python -u
"""
Centroid-Overlap Dispersion: 2-Parameter Alpha Model
=====================================================
Extends the alpha(rho) formula by capturing the SPREAD of off-diagonal
whitened cosine similarities, not just their mean.

Theory: alpha = sqrt(4/pi) / sqrt(1 - rho_eff)
where rho_eff = rho_mean + c * sigma_rho  (dispersion correction)

The hypothesis is that architectures with higher variance in pairwise
centroid cosines (more heterogeneous geometry) have different effective
rho, explaining per-model alpha variation that mean rho alone cannot.

Reuses infrastructure from cti_alpha_rho_multidataset.py.
"""

import json
import time
import gc
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, skew as scipy_skew
from scipy.optimize import minimize_scalar, minimize
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_PCA = 256
BATCH_SIZE = 32

A_RENORM = float(np.sqrt(4.0 / np.pi))  # 1.12838

print(f"Device: {DEVICE}", flush=True)

# Load LOAO alpha
with open(RESULTS_DIR / "cti_kappa_loao_per_dataset.json") as f:
    loao_data = json.load(f)
LOAO_ALPHA = {}
for model_name, entry in loao_data["loao_results"].items():
    short = model_name.split("/")[-1]
    LOAO_ALPHA[short] = float(entry["alpha"])

# Models (same as multidataset)
MODELS = {
    "pythia-160m":       "EleutherAI/pythia-160m",
    "pythia-410m":       "EleutherAI/pythia-410m",
    "pythia-1b":         "EleutherAI/pythia-1b",
    "gpt-neo-125m":      "EleutherAI/gpt-neo-125m",
    "Qwen2.5-0.5B":      "Qwen/Qwen2.5-0.5B",
    "OLMo-1B-hf":        "allenai/OLMo-1B-hf",
    "TinyLlama-1.1B-intermediate-step-1431k-3T":
                         "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "Qwen3-0.6B":        "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B":        "Qwen/Qwen3-1.7B",
    "Mistral-7B-v0.3":   "mistralai/Mistral-7B-v0.3",
    "rwkv-4-169m-pile":  "RWKV/rwkv-4-169m-pile",
}

MODEL_LAYERS = {
    "pythia-160m":       [3, 6, 9, 12],
    "pythia-410m":       [6, 12, 18, 24],
    "pythia-1b":         [4, 8, 12, 16],
    "gpt-neo-125m":      [3, 6, 9, 12],
    "Qwen2.5-0.5B":      [7, 14, 21, 24],
    "OLMo-1B-hf":        [4, 8, 12, 16],
    "TinyLlama-1.1B-intermediate-step-1431k-3T": [5, 11, 16, 22],
    "Qwen3-0.6B":        [7, 14, 21, 28],
    "Qwen3-1.7B":        [7, 14, 21, 28],
    "Mistral-7B-v0.3":   [8, 16, 24, 32],
    "rwkv-4-169m-pile":  [3, 6, 9, 12],
}

MODEL_BATCH_SIZE = {"Mistral-7B-v0.3": 4}

DATASETS = {
    "agnews": {
        "hf_name": "ag_news", "split": "test",
        "text_field": "text", "label_field": "label", "K": 4, "N": 2000,
    },
    "dbpedia": {
        "hf_name": "fancyzhx/dbpedia_14", "split": "test",
        "text_field": "content", "label_field": "label", "K": 14, "N": 2000,
    },
    "banking77": {
        "hf_name": "PolyAI/banking77", "split": "test",
        "text_field": "text", "label_field": "label", "K": 77, "N": 2000,
    },
}


def extract_embeddings(model, tokenizer, texts, layers, batch_size=32):
    """Extract hidden-state embeddings at specified layers."""
    model.eval()
    layer_embs = {l: [] for l in layers}

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(DEVICE)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states

            mask = enc["attention_mask"].unsqueeze(-1).float()
            for l in layers:
                if l < len(hs):
                    h = hs[l] * mask
                    pooled = h.sum(1) / mask.sum(1).clamp(min=1)
                    layer_embs[l].append(pooled.cpu().numpy())

    return {l: np.concatenate(layer_embs[l], axis=0) for l in layers}


def compute_rho_with_dispersion(embeddings, labels, classes):
    """Compute rho AND its dispersion (variance, skew, kurtosis).

    Returns dict with rho_mean, rho_std, rho_var_off, rho_skew_off, rho_kurt_off,
    and all_off_diag (flattened array of ALL off-diagonal cosines).
    """
    K_local = len(classes)
    N, d = embeddings.shape

    centroids = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = embeddings[mask].mean(0).astype(np.float64)
    if len(centroids) < K_local:
        return None

    centroid_array = np.array([centroids[c] for c in classes])

    # SVD for whitening
    Xc_list = []
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            Xc_list.append((embeddings[mask] - centroids[c]).astype(np.float64))
    Z = np.concatenate(Xc_list, axis=0)
    N_total = len(Z)

    n_comp = min(N_PCA, d, N_total - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(Z)
    V = svd.components_.T
    Lambda = (svd.singular_values_ ** 2) / N_total
    sqrt_Lambda = np.sqrt(Lambda + 1e-12)

    # Compute per-class off-diagonal cosines
    rho_per_class = []
    var_per_class = []
    skew_per_class = []
    all_off_diag = []

    for i_c, c in enumerate(classes):
        other_idx = [i for i in range(K_local) if i != i_c]
        deltas = centroid_array[other_idx] - centroids[c]
        proj = deltas @ V
        whitened = proj * sqrt_Lambda[None, :]
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        w_norm = whitened / norms
        cos_mat = w_norm @ w_norm.T
        n_off = K_local - 1
        off_vals = cos_mat[~np.eye(n_off, dtype=bool)]

        rho_per_class.append(float(off_vals.mean()))
        var_per_class.append(float(np.var(off_vals, ddof=1)) if len(off_vals) > 1 else 0.0)
        if len(off_vals) > 2:
            skew_per_class.append(float(scipy_skew(off_vals)))
        else:
            skew_per_class.append(0.0)
        all_off_diag.extend(off_vals.tolist())

    all_off = np.array(all_off_diag)

    return {
        "rho_mean": float(np.mean(rho_per_class)),
        "rho_std": float(np.std(rho_per_class)),
        "rho_var_per_class_mean": float(np.mean(var_per_class)),
        "rho_skew_per_class_mean": float(np.mean(skew_per_class)),
        # Global off-diagonal statistics
        "off_diag_mean": float(all_off.mean()),
        "off_diag_var": float(np.var(all_off, ddof=1)),
        "off_diag_std": float(np.std(all_off, ddof=1)),
        "off_diag_skew": float(scipy_skew(all_off)),
        "off_diag_q25": float(np.percentile(all_off, 25)),
        "off_diag_q75": float(np.percentile(all_off, 75)),
        "off_diag_min": float(all_off.min()),
        "off_diag_max": float(all_off.max()),
        "n_off_diag": len(all_off),
    }


def main():
    print("=" * 72)
    print("  CENTROID-OVERLAP DISPERSION: 2-PARAMETER ALPHA MODEL")
    print("=" * 72)

    cache_path = RESULTS_DIR / "cti_centroid_dispersion.json"
    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} cached results", flush=True)
    else:
        results = {}

    # Load datasets
    rng = np.random.default_rng(SEED)
    dataset_cache = {}
    for ds_name, ds_cfg in DATASETS.items():
        print(f"\nLoading {ds_name} (K={ds_cfg['K']}, N={ds_cfg['N']})...", flush=True)
        ds = load_dataset(ds_cfg["hf_name"], split=ds_cfg["split"])
        texts_all = [str(x[ds_cfg["text_field"]]) for x in ds]
        labels_all = [int(x[ds_cfg["label_field"]]) for x in ds]
        idx = rng.choice(len(texts_all), size=min(ds_cfg["N"], len(texts_all)),
                         replace=False)
        texts = [texts_all[i] for i in idx]
        labels = np.array([labels_all[i] for i in idx])
        classes = sorted(list(set(labels)))
        dataset_cache[ds_name] = (texts, labels, classes)
        print(f"  {ds_name}: {len(texts)} texts, K={len(classes)}", flush=True)

    # Process each model
    for model_key, hf_id in MODELS.items():
        cache_key = model_key
        if cache_key in results and "per_dataset" in results[cache_key]:
            n_ds = len(results[cache_key]["per_dataset"])
            if n_ds >= 3:
                print(f"\n  Skipping {model_key} (cached, {n_ds} datasets)")
                continue

        print(f"\n{'='*60}")
        print(f"  {model_key} ({hf_id})")
        print(f"{'='*60}", flush=True)
        t0 = time.time()

        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            bs_model = MODEL_BATCH_SIZE.get(model_key, BATCH_SIZE)
            model = AutoModel.from_pretrained(
                hf_id, torch_dtype=torch.float16, trust_remote_code=True
            ).to(DEVICE)
            model.eval()

            layers = MODEL_LAYERS.get(model_key, [4, 8, 12, 16])
            best_layer = layers[-1]  # Use deepest layer

            per_dataset = {}
            for ds_name, (texts, labels, classes) in dataset_cache.items():
                print(f"\n  --- {ds_name} (K={len(classes)}) ---", flush=True)
                emb_dict = extract_embeddings(model, tokenizer, texts,
                                              [best_layer], batch_size=bs_model)
                emb = emb_dict[best_layer]
                print(f"    Embeddings: {emb.shape}", flush=True)

                disp = compute_rho_with_dispersion(emb, labels, classes)
                if disp is None:
                    print(f"    FAILED (insufficient classes)")
                    continue

                per_dataset[ds_name] = disp
                print(f"    rho_mean = {disp['rho_mean']:.4f}")
                print(f"    off_diag_var = {disp['off_diag_var']:.4f}")
                print(f"    off_diag_skew = {disp['off_diag_skew']:.4f}")
                print(f"    off_diag_std = {disp['off_diag_std']:.4f}")

            # Pool across datasets
            if len(per_dataset) >= 2:
                pooled_rho = np.mean([v["rho_mean"] for v in per_dataset.values()])
                pooled_var = np.mean([v["off_diag_var"] for v in per_dataset.values()])
                pooled_std = np.mean([v["off_diag_std"] for v in per_dataset.values()])
                pooled_skew = np.mean([v["off_diag_skew"] for v in per_dataset.values()])
            else:
                ds0 = list(per_dataset.values())[0]
                pooled_rho = ds0["rho_mean"]
                pooled_var = ds0["off_diag_var"]
                pooled_std = ds0["off_diag_std"]
                pooled_skew = ds0["off_diag_skew"]

            alpha_loao = LOAO_ALPHA.get(model_key, None)

            results[cache_key] = {
                "model": model_key,
                "hf_id": hf_id,
                "layer": best_layer,
                "per_dataset": per_dataset,
                "rho_pooled": float(pooled_rho),
                "off_diag_var_pooled": float(pooled_var),
                "off_diag_std_pooled": float(pooled_std),
                "off_diag_skew_pooled": float(pooled_skew),
                "alpha_loao": float(alpha_loao) if alpha_loao else None,
                "alpha_pred_0param": float(A_RENORM / np.sqrt(1 - pooled_rho)),
                "time_s": time.time() - t0,
            }

            del model
            gc.collect()
            torch.cuda.empty_cache()

            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[cache_key] = {"model": model_key, "error": str(e)}
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)
            gc.collect()
            torch.cuda.empty_cache()

    # ================================================================
    # ANALYSIS: 2-PARAMETER ALPHA MODEL
    # ================================================================
    print(f"\n{'='*72}")
    print("  CENTROID-OVERLAP DISPERSION ANALYSIS")
    print("=" * 72)

    valid_keys = [k for k in results
                  if "rho_pooled" in results[k]
                  and results[k].get("alpha_loao") is not None]

    if len(valid_keys) < 3:
        print("  Too few models for analysis")
        return

    rhos = np.array([results[k]["rho_pooled"] for k in valid_keys])
    vars_od = np.array([results[k]["off_diag_var_pooled"] for k in valid_keys])
    stds_od = np.array([results[k]["off_diag_std_pooled"] for k in valid_keys])
    skews_od = np.array([results[k]["off_diag_skew_pooled"] for k in valid_keys])
    alpha_loao = np.array([results[k]["alpha_loao"] for k in valid_keys])
    alpha_0p = np.array([results[k]["alpha_pred_0param"] for k in valid_keys])
    n = len(valid_keys)

    print(f"\n  {'Model':<40s} {'rho':>6s} {'var':>8s} {'std':>8s} "
          f"{'skew':>8s} {'a_loao':>7s} {'a_pred':>7s}")
    print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
    for k in valid_keys:
        r = results[k]
        print(f"  {k:<40s} {r['rho_pooled']:>6.4f} {r['off_diag_var_pooled']:>8.4f} "
              f"{r['off_diag_std_pooled']:>8.4f} {r['off_diag_skew_pooled']:>8.4f} "
              f"{r['alpha_loao']:>7.4f} {r['alpha_pred_0param']:>7.4f}")

    # Model 0: alpha = A / sqrt(1 - rho)  (current, 0 free params)
    mae_0 = float(np.mean(np.abs(alpha_0p - alpha_loao)))
    r_0, p_0 = pearsonr(alpha_0p, alpha_loao)
    rho_sp_0, p_sp_0 = spearmanr(alpha_0p, alpha_loao)
    print(f"\n  === Model 0 (0 free params): alpha = A/sqrt(1-rho) ===")
    print(f"  MAE = {mae_0:.4f}, r = {r_0:.4f} (p={p_0:.4f}), rho_sp = {rho_sp_0:.4f}")

    # Model 1: alpha = A / sqrt(1 - (rho + c*sigma))  (1 free param: c)
    def mae_model1(c):
        rho_eff = rhos + c * stds_od
        rho_eff = np.clip(rho_eff, -0.99, 0.99)
        alpha_pred = A_RENORM / np.sqrt(1 - rho_eff)
        return float(np.mean(np.abs(alpha_pred - alpha_loao)))

    res1 = minimize_scalar(mae_model1, bounds=(-5.0, 5.0), method='bounded')
    c_opt = res1.x
    rho_eff_1 = np.clip(rhos + c_opt * stds_od, -0.99, 0.99)
    alpha_1p = A_RENORM / np.sqrt(1 - rho_eff_1)
    mae_1 = float(np.mean(np.abs(alpha_1p - alpha_loao)))
    r_1, p_1 = pearsonr(alpha_1p, alpha_loao)
    rho_sp_1, p_sp_1 = spearmanr(alpha_1p, alpha_loao)
    print(f"\n  === Model 1 (1 free param c): alpha = A/sqrt(1 - (rho + c*sigma)) ===")
    print(f"  c_opt = {c_opt:.4f}")
    print(f"  MAE = {mae_1:.4f}, r = {r_1:.4f} (p={p_1:.4f}), rho_sp = {rho_sp_1:.4f}")
    print(f"  Improvement over Model 0: MAE {mae_0-mae_1:+.4f}, r {r_1-r_0:+.4f}")

    # Model 2: alpha = A / sqrt(1 - (rho + c1*sigma + c2*skew))  (2 free params)
    def mae_model2(params):
        c1, c2 = params
        rho_eff = rhos + c1 * stds_od + c2 * skews_od
        rho_eff = np.clip(rho_eff, -0.99, 0.99)
        alpha_pred = A_RENORM / np.sqrt(1 - rho_eff)
        return float(np.mean(np.abs(alpha_pred - alpha_loao)))

    res2 = minimize(mae_model2, x0=[0.0, 0.0], method='Nelder-Mead')
    c1_opt, c2_opt = res2.x
    rho_eff_2 = np.clip(rhos + c1_opt * stds_od + c2_opt * skews_od, -0.99, 0.99)
    alpha_2p = A_RENORM / np.sqrt(1 - rho_eff_2)
    mae_2 = float(np.mean(np.abs(alpha_2p - alpha_loao)))
    r_2, p_2 = pearsonr(alpha_2p, alpha_loao)
    rho_sp_2, p_sp_2 = spearmanr(alpha_2p, alpha_loao)
    print(f"\n  === Model 2 (2 free params): alpha = A/sqrt(1 - (rho + c1*sigma + c2*skew)) ===")
    print(f"  c1_opt = {c1_opt:.4f}, c2_opt = {c2_opt:.4f}")
    print(f"  MAE = {mae_2:.4f}, r = {r_2:.4f} (p={p_2:.4f}), rho_sp = {rho_sp_2:.4f}")
    print(f"  Improvement over Model 0: MAE {mae_0-mae_2:+.4f}, r {r_2-r_0:+.4f}")

    # Model 3: Direct linear regression alpha = a + b*rho + c*sigma
    X = np.column_stack([np.ones(n), rhos, stds_od])
    beta3 = np.linalg.lstsq(X, alpha_loao, rcond=None)[0]
    alpha_3p = X @ beta3
    mae_3 = float(np.mean(np.abs(alpha_3p - alpha_loao)))
    r_3, p_3 = pearsonr(alpha_3p, alpha_loao)
    SS_res = np.sum((alpha_loao - alpha_3p)**2)
    SS_tot = np.sum((alpha_loao - alpha_loao.mean())**2)
    R2_3 = 1 - SS_res / SS_tot
    adj_R2_3 = 1 - (1 - R2_3) * (n - 1) / (n - 3)
    print(f"\n  === Model 3 (linear): alpha = a + b*rho + c*sigma ===")
    print(f"  a={beta3[0]:.4f}, b={beta3[1]:.4f}, c={beta3[2]:.4f}")
    print(f"  R2 = {R2_3:.4f}, adj_R2 = {adj_R2_3:.4f}")
    print(f"  MAE = {mae_3:.4f}, r = {r_3:.4f}")

    # Correlations between dispersion metrics and alpha
    print(f"\n  === Correlations with alpha_loao ===")
    for label, vals in [("rho_mean", rhos), ("off_diag_std", stds_od),
                         ("off_diag_var", vars_od), ("off_diag_skew", skews_od)]:
        r_val, p_val = pearsonr(vals, alpha_loao)
        print(f"  r({label}, alpha) = {r_val:.4f} (p = {p_val:.4f})")

    # Save analysis
    analysis = {
        "n_models": n,
        "model_keys": valid_keys,
        "model0_mae": mae_0, "model0_r": float(r_0),
        "model1_mae": mae_1, "model1_r": float(r_1), "model1_c": float(c_opt),
        "model2_mae": mae_2, "model2_r": float(r_2),
        "model2_c1": float(c1_opt), "model2_c2": float(c2_opt),
        "model3_R2": float(R2_3), "model3_adj_R2": float(adj_R2_3),
        "model3_coeffs": beta3.tolist(),
    }
    results["_analysis"] = analysis

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {cache_path}")


if __name__ == "__main__":
    main()
