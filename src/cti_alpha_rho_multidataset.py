#!/usr/bin/env python -u
"""
MULTI-DATASET ALPHA-RHO VALIDATION WITH BOOTSTRAP (Mar 1 2026)
================================================================
Codex Design Gate: Session 84 (gpt-5.3-codex-spark, xhigh reasoning)

HYPOTHESIS: alpha = sqrt(4/pi) / sqrt(1 - rho)
  Same as Session 83, but now validated across 3 datasets with bootstrap
  uncertainty quantification and Spearman disattenuation correction.

DATASETS:
  AG News       K=4    (small K, high samples/class)
  DBpedia       K=14   (medium K, canonical baseline)
  Banking77     K=77   (large K, ~26 samples/class)

BOOTSTRAP:
  200 stratified 80% subsamples per (model, dataset) pair.
  rho estimated per replicate -> pooled across datasets per model.

DISATTENUATION (Spearman correction for attenuation, 1904):
  r_disattenuated = r_observed / sqrt(rel_alpha_pred * rel_alpha_loao)
  rel_alpha_pred estimated from bootstrap variance decomposition.
  rel_alpha_loao = 1.0 (LOAO from 444 pts is ground truth).

PRE-REGISTERED SUCCESS CRITERIA (set BEFORE running):
  H1: MAE(alpha_pred_pooled, alpha_loao) < 0.15
  H2: raw Pearson r > 0.70  (likely FAIL, known from Session 83)
  H3: disattenuated Spearman r > 0.70  (KEY new test)

OUTPUTS:
  results/cti_alpha_rho_multidataset.json
"""

import json
import time
import sys
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_PATH = RESULTS_DIR / "cti_alpha_rho_multidataset_cache.json"

print(f"Device: {DEVICE}", flush=True)

# Theoretical constant (zero free parameters)
A_RENORM = float(np.sqrt(4.0 / np.pi))   # 1.12838

# ============================================================
# PRE-REGISTERED CRITERIA (DO NOT MODIFY AFTER COMMIT)
# ============================================================
PASS_RAW_R_MIN             = 0.70
PASS_RAW_MAE_MAX           = 0.15
PASS_DISATTEN_SPEARMAN_MIN = 0.70
RHO_BOOT                   = 200       # bootstrap replicates per (model, dataset)
RHO_BOOT_FRAC              = 0.80      # fraction of data per bootstrap
N_BOOT_CORR                = 2000      # BCa bootstrap for disattenuated r
SEED                       = 42

# ============================================================
# CANONICAL LOAO ALPHA VALUES
# ============================================================
LOAO_JSON = RESULTS_DIR / "cti_kappa_loao_per_dataset.json"
with open(LOAO_JSON) as f:
    loao_data = json.load(f)

LOAO_ALPHA = {}
for model_name, entry in loao_data["loao_results"].items():
    short = model_name.split("/")[-1]
    LOAO_ALPHA[short] = float(entry["alpha"])

print(f"\nCanonical LOAO alphas ({len(LOAO_ALPHA)} models):", flush=True)
for k, v in sorted(LOAO_ALPHA.items()):
    print(f"  {k}: {v:.4f}", flush=True)

# ============================================================
# MODEL REGISTRY (same as cti_alpha_rho_derivation.py)
# ============================================================
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

TRUST_REMOTE_CODE = set()
SKIP_MODELS = {"Falcon-H1-0.5B-Base"}

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

BATCH_SIZE = 32
MODEL_BATCH_SIZE = {
    "Mistral-7B-v0.3": 4,
}

# ============================================================
# DATASETS
# ============================================================
DATASETS = {
    "agnews": {
        "hf_name": "ag_news",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "K": 4,
        "N": 2000,
    },
    "dbpedia": {
        "hf_name": "fancyzhx/dbpedia_14",
        "split": "test",
        "text_field": "content",
        "label_field": "label",
        "K": 14,
        "N": 2000,
    },
    "banking77": {
        "hf_name": "PolyAI/banking77",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "K": 77,
        "N": 2000,
    },
}

# ============================================================
# LOAD ALL DATASETS
# ============================================================
rng_global = np.random.default_rng(SEED)
dataset_cache = {}

for ds_name, ds_cfg in DATASETS.items():
    print(f"\nLoading {ds_name} (K={ds_cfg['K']}, N={ds_cfg['N']})...", flush=True)
    ds = load_dataset(ds_cfg["hf_name"], split=ds_cfg["split"])
    texts_all  = [str(x[ds_cfg["text_field"]]) for x in ds]
    labels_all = [int(x[ds_cfg["label_field"]]) for x in ds]

    idx = rng_global.choice(len(texts_all), size=min(ds_cfg["N"], len(texts_all)),
                            replace=False)
    texts  = [texts_all[i] for i in idx]
    labels = np.array([labels_all[i] for i in idx])
    classes = sorted(set(labels))
    print(f"  Got {len(texts)} samples, {len(classes)} classes "
          f"(expected K={ds_cfg['K']})", flush=True)

    dataset_cache[ds_name] = {
        "texts": texts,
        "labels": labels,
        "classes": classes,
        "K": len(classes),
    }


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================
N_PCA = 256


@torch.no_grad()
def load_model_and_tokenizer(model_id, model_short):
    """Load model once. Returns (model, tokenizer)."""
    trust = model_short in TRUST_REMOTE_CODE
    print(f"  Loading {model_short}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    model = AutoModel.from_pretrained(
        model_id,
        output_hidden_states=True,
        dtype=torch.float16,
        trust_remote_code=trust,
    ).to(DEVICE).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def embed_texts(model, tokenizer, model_short, text_list, layers):
    """Embed texts using already-loaded model. Returns layer -> ndarray."""
    bs = MODEL_BATCH_SIZE.get(model_short, BATCH_SIZE)
    layer_embs = {l: [] for l in layers}
    for i in range(0, len(text_list), bs):
        batch = text_list[i:i+bs]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to(DEVICE)
        out = model(**enc)
        hidden = out.hidden_states
        valid_layers = [min(l, len(hidden)-1) for l in layers]
        attn_mask = enc["attention_mask"].float()
        seq_len = attn_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        for l, vl in zip(layers, valid_layers):
            h = hidden[vl].float()
            mask3 = attn_mask.unsqueeze(-1)
            pooled = (h * mask3).sum(dim=1) / seq_len.squeeze(-1)
            layer_embs[l].append(pooled.cpu().numpy())
    return {l: np.concatenate(layer_embs[l], axis=0) for l in layers}


# ============================================================
# RHO COMPUTATION (canonical Sigma_W^{1/2} whitening)
# ============================================================
def compute_rho(embeddings, labels, classes):
    """Compute mean Sigma_W-whitened cosine similarity of centroid diffs."""
    from sklearn.decomposition import TruncatedSVD

    K_local = len(classes)
    N, d = embeddings.shape

    centroids = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = embeddings[mask].mean(0).astype(np.float64)
    if len(centroids) < K_local:
        return float("nan"), float("nan")

    centroid_array = np.array([centroids[c] for c in classes])

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

    rho_per_class = []
    for i_c, c in enumerate(classes):
        other_idx = [i for i in range(K_local) if i != i_c]
        deltas = centroid_array[other_idx] - centroids[c]
        proj     = deltas @ V
        whitened = proj * sqrt_Lambda[None, :]
        norms    = np.linalg.norm(whitened, axis=1, keepdims=True)
        norms    = np.maximum(norms, 1e-12)
        w_norm   = whitened / norms
        cos_mat  = w_norm @ w_norm.T
        n_off = K_local - 1
        off_vals = cos_mat[~np.eye(n_off, dtype=bool)]
        rho_per_class.append(float(off_vals.mean()))

    return float(np.mean(rho_per_class)), float(np.std(rho_per_class))


def compute_kappa_nearest(embeddings, labels, classes):
    d = embeddings.shape[1]
    mu = {c: embeddings[labels == c].mean(0) for c in classes}
    within_var = sum(np.sum((embeddings[labels == c] - mu[c])**2) for c in classes)
    sigma_W = float(np.sqrt(within_var / (len(embeddings) * d)))
    kappas = []
    for c in classes:
        dists = [np.linalg.norm(mu[c] - mu[j]) for j in classes if j != c]
        kappas.append(min(dists) / (sigma_W * np.sqrt(d)))
    return float(np.mean(kappas))


# ============================================================
# FAST BOOTSTRAP RHO (fit SVD once, recompute centroids per replicate)
# ============================================================
def precompute_svd(embeddings, labels, classes):
    """Fit TruncatedSVD on full data. Returns V, sqrt_Lambda for fast rho."""
    from sklearn.decomposition import TruncatedSVD

    N, d = embeddings.shape
    centroids = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = embeddings[mask].mean(0).astype(np.float64)

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
    return V, sqrt_Lambda


def compute_rho_fast(embeddings, labels, classes, V, sqrt_Lambda):
    """Compute rho using pre-computed SVD. ~100x faster than compute_rho."""
    K_local = len(classes)

    centroids = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = embeddings[mask].mean(0).astype(np.float64)
    if len(centroids) < K_local:
        return float("nan")

    centroid_array = np.array([centroids[c] for c in classes])

    rho_per_class = []
    for i_c, c in enumerate(classes):
        other_idx = [i for i in range(K_local) if i != i_c]
        deltas = centroid_array[other_idx] - centroids[c]
        proj     = deltas @ V
        whitened = proj * sqrt_Lambda[None, :]
        norms    = np.linalg.norm(whitened, axis=1, keepdims=True)
        norms    = np.maximum(norms, 1e-12)
        w_norm   = whitened / norms
        cos_mat  = w_norm @ w_norm.T
        n_off = K_local - 1
        off_vals = cos_mat[~np.eye(n_off, dtype=bool)]
        rho_per_class.append(float(off_vals.mean()))

    return float(np.mean(rho_per_class))


def bootstrap_rho(embeddings, labels, classes, n_boot=200, frac=0.80, seed=42):
    """Fast stratified bootstrap: fit SVD once, recompute centroids per rep."""
    rng = np.random.default_rng(seed)
    K_local = len(classes)

    # Fit SVD once on full data
    V, sqrt_Lambda = precompute_svd(embeddings, labels, classes)

    # Pre-compute class indices
    class_idx = {c: np.where(labels == c)[0] for c in classes}

    rho_boot = []
    n_invalid = 0

    for b in range(n_boot):
        boot_idx = []
        valid = True
        for c in classes:
            cidx = class_idx[c]
            n_take = max(2, int(len(cidx) * frac))
            if len(cidx) < 2:
                valid = False
                break
            chosen = rng.choice(cidx, size=n_take, replace=False)
            boot_idx.append(chosen)
        if not valid:
            n_invalid += 1
            continue

        boot_idx = np.concatenate(boot_idx)
        emb_boot = embeddings[boot_idx]
        lab_boot = labels[boot_idx]
        rho_val = compute_rho_fast(emb_boot, lab_boot, classes, V, sqrt_Lambda)
        if not np.isnan(rho_val):
            rho_boot.append(rho_val)
        else:
            n_invalid += 1

    return np.array(rho_boot), n_invalid


# ============================================================
# DISATTENUATION + BCa CI
# ============================================================
def compute_reliability(alpha_pred_boots, alpha_pred_point, n_models):
    """
    Reliability = 1 - var_within / var_total
    var_within = mean of per-model bootstrap variance of alpha_pred
    var_total = variance of alpha_pred point estimates across models
    """
    # alpha_pred_boots: dict model -> array of bootstrap alpha_pred values
    # alpha_pred_point: dict model -> single alpha_pred
    models = sorted(alpha_pred_boots.keys())
    if len(models) < 2:
        return 1.0

    var_within_vals = []
    for m in models:
        boots = alpha_pred_boots[m]
        if len(boots) > 1:
            var_within_vals.append(np.var(boots, ddof=1))
    var_within = np.mean(var_within_vals) if var_within_vals else 0.0

    point_vals = np.array([alpha_pred_point[m] for m in models])
    var_total = np.var(point_vals, ddof=1)

    if var_total < 1e-15:
        return 0.0
    rel = max(0.0, 1.0 - var_within / var_total)
    return float(rel)


def bca_ci(data_x, data_y, stat_func, n_boot=2000, alpha=0.05, seed=42):
    """BCa bootstrap CI for a statistic computed from paired (x, y) data."""
    rng = np.random.default_rng(seed)
    n = len(data_x)
    theta_hat = stat_func(data_x, data_y)

    # Bootstrap distribution
    boot_thetas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_thetas[b] = stat_func(data_x[idx], data_y[idx])

    # Bias correction
    z0 = float(np.quantile(np.random.default_rng(0).standard_normal(1), 0.5))
    prop_below = np.mean(boot_thetas < theta_hat)
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
    from scipy.stats import norm
    z0 = float(norm.ppf(prop_below))

    # Acceleration (jackknife)
    jack_thetas = np.empty(n)
    for i in range(n):
        idx_jack = np.concatenate([np.arange(i), np.arange(i+1, n)])
        jack_thetas[i] = stat_func(data_x[idx_jack], data_y[idx_jack])
    jack_mean = jack_thetas.mean()
    num = np.sum((jack_mean - jack_thetas)**3)
    den = 6.0 * (np.sum((jack_mean - jack_thetas)**2))**1.5
    a_hat = float(num / den) if abs(den) > 1e-15 else 0.0

    # Adjusted percentiles
    z_alpha = norm.ppf(alpha / 2)
    z_1alpha = norm.ppf(1 - alpha / 2)

    p_lo = norm.cdf(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha)))
    p_hi = norm.cdf(z0 + (z0 + z_1alpha) / (1 - a_hat * (z0 + z_1alpha)))

    p_lo = np.clip(p_lo, 0.5 / n_boot, 1 - 0.5 / n_boot)
    p_hi = np.clip(p_hi, 0.5 / n_boot, 1 - 0.5 / n_boot)

    ci_lo = float(np.percentile(boot_thetas, 100 * p_lo))
    ci_hi = float(np.percentile(boot_thetas, 100 * p_hi))
    return theta_hat, ci_lo, ci_hi


# ============================================================
# LOAD CACHE (resume support)
# ============================================================
model_dataset_results = {}
if CACHE_PATH.exists():
    print(f"\nLoading cache from {CACHE_PATH}...", flush=True)
    with open(CACHE_PATH) as f:
        model_dataset_results = json.load(f)
    n_cached = sum(len(v) for v in model_dataset_results.values())
    print(f"  Loaded {n_cached} cached (model, dataset) pairs.", flush=True)


def save_cache():
    with open(CACHE_PATH, "w") as f:
        json.dump(model_dataset_results, f, indent=2)


# ============================================================
# MAIN LOOP: embed each model, compute bootstrap rho for 3 datasets
# ============================================================
t_start = time.time()

for model_short, model_id in MODELS.items():
    if model_short in SKIP_MODELS:
        print(f"\n[SKIP] {model_short} (in SKIP_MODELS)", flush=True)
        continue
    if model_short not in LOAO_ALPHA:
        print(f"\n[SKIP] {model_short} (no LOAO alpha)", flush=True)
        continue

    # Check which datasets still need computation
    cached_ds = model_dataset_results.get(model_short, {})
    needed_ds = [ds for ds in DATASETS if ds not in cached_ds]

    if not needed_ds:
        print(f"\n[CACHED] {model_short}: all 3 datasets cached.", flush=True)
        continue

    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_short}  (need: {needed_ds})", flush=True)

    layers = MODEL_LAYERS.get(model_short, [3, 6, 9, 12])
    t0 = time.time()

    try:
        # Load model ONCE, embed all datasets, then free
        model_obj, tokenizer = load_model_and_tokenizer(model_id, model_short)

        ds_embeddings = {}
        for ds_name in needed_ds:
            ds_info = dataset_cache[ds_name]
            print(f"  Embedding {ds_name} (K={ds_info['K']}, "
                  f"N={len(ds_info['texts'])})...", flush=True)
            layer_embs = embed_texts(model_obj, tokenizer, model_short,
                                     ds_info["texts"], layers)
            best_layer = max(
                layers,
                key=lambda l: compute_kappa_nearest(
                    layer_embs[l], ds_info["labels"], ds_info["classes"]),
            )
            ds_embeddings[ds_name] = {
                "emb": layer_embs[best_layer],
                "best_layer": best_layer,
            }
            kappa = compute_kappa_nearest(
                layer_embs[best_layer], ds_info["labels"], ds_info["classes"])
            print(f"    best_layer={best_layer}, kappa={kappa:.4f}", flush=True)
            del layer_embs

        # Free GPU
        del model_obj, tokenizer
        torch.cuda.empty_cache()
        print(f"  Model freed. Running bootstrap (CPU)...", flush=True)

        # Bootstrap rho for each dataset (CPU-only, fast SVD)
        if model_short not in model_dataset_results:
            model_dataset_results[model_short] = {}

        for ds_name in needed_ds:
            ds_info = dataset_cache[ds_name]
            emb = ds_embeddings[ds_name]["emb"]

            # Point estimate
            rho_point, rho_std = compute_rho(
                emb, ds_info["labels"], ds_info["classes"])
            print(f"  {ds_name}: rho_point={rho_point:.4f} +/- {rho_std:.4f}",
                  flush=True)

            # Fast bootstrap (SVD fitted once, centroids recomputed per rep)
            print(f"  {ds_name}: Running {RHO_BOOT} bootstrap resamples...",
                  flush=True)
            t_boot = time.time()
            rho_boots, n_invalid = bootstrap_rho(
                emb, ds_info["labels"], ds_info["classes"],
                n_boot=RHO_BOOT, frac=RHO_BOOT_FRAC,
                seed=SEED + hash(model_short + ds_name) % 10000,
            )
            rho_boot_mean = float(np.mean(rho_boots)) if len(rho_boots) > 0 else float("nan")
            rho_boot_std  = float(np.std(rho_boots))  if len(rho_boots) > 0 else float("nan")
            print(f"    boot: mean={rho_boot_mean:.4f}, std={rho_boot_std:.4f}, "
                  f"n_ok={len(rho_boots)}/{RHO_BOOT}, n_invalid={n_invalid}, "
                  f"took {time.time()-t_boot:.1f}s", flush=True)

            alpha_pred_boots = A_RENORM / np.sqrt(np.maximum(1.0 - rho_boots, 1e-6))

            model_dataset_results[model_short][ds_name] = {
                "best_layer": ds_embeddings[ds_name]["best_layer"],
                "K": ds_info["K"],
                "rho_point": rho_point,
                "rho_std": rho_std,
                "rho_boot_mean": rho_boot_mean,
                "rho_boot_std": rho_boot_std,
                "n_boot_ok": len(rho_boots),
                "n_invalid": n_invalid,
                "alpha_pred_boots": alpha_pred_boots.tolist(),
            }

        del ds_embeddings
        elapsed = time.time() - t0
        print(f"\n  {model_short} done in {elapsed:.0f}s", flush=True)

        # Save cache after each model
        save_cache()

    except Exception as e:
        print(f"  ERROR on {model_short}: {e}", flush=True)
        import traceback; traceback.print_exc()
        if model_short not in model_dataset_results:
            model_dataset_results[model_short] = {}
        model_dataset_results[model_short]["error"] = str(e)
        save_cache()

# ============================================================
# AGGREGATE: Pool rho across datasets, compute disattenuated r
# ============================================================
print(f"\n{'='*60}", flush=True)
print("AGGREGATION", flush=True)

# Collect per-model pooled results
model_results = {}
alpha_pred_boot_traces = {}  # for reliability estimation

for model_short in MODELS:
    if model_short in SKIP_MODELS or model_short not in LOAO_ALPHA:
        continue
    if model_short not in model_dataset_results:
        continue
    mdr = model_dataset_results[model_short]
    if "error" in mdr:
        continue

    # Pool rho across datasets (simple mean)
    rho_vals = []
    rho_boot_all = []  # bootstrap traces across datasets
    for ds_name in DATASETS:
        if ds_name in mdr and "rho_boot_mean" in mdr[ds_name]:
            rho_vals.append(mdr[ds_name]["rho_boot_mean"])
            if "alpha_pred_boots" in mdr[ds_name]:
                rho_boot_all.append(np.array(mdr[ds_name]["alpha_pred_boots"]))

    if not rho_vals:
        continue

    rho_pooled = float(np.mean(rho_vals))
    rho_pooled_std = float(np.std(rho_vals)) if len(rho_vals) > 1 else 0.0
    alpha_pred = float(A_RENORM / np.sqrt(max(1.0 - rho_pooled, 1e-6)))
    alpha_loao = LOAO_ALPHA[model_short]
    error = alpha_pred - alpha_loao
    abs_error = abs(error)

    # Bootstrap alpha_pred traces (average across datasets per replicate)
    if rho_boot_all:
        min_len = min(len(x) for x in rho_boot_all)
        # Trim to same length, then average alpha_pred across datasets
        trimmed = [x[:min_len] for x in rho_boot_all]
        alpha_pred_boot_mean = np.mean(trimmed, axis=0)
        alpha_pred_boot_traces[model_short] = alpha_pred_boot_mean

    model_results[model_short] = {
        "rho_pooled": rho_pooled,
        "rho_pooled_std": rho_pooled_std,
        "rho_per_dataset": {ds: mdr[ds]["rho_boot_mean"]
                            for ds in DATASETS if ds in mdr
                            and "rho_boot_mean" in mdr[ds]},
        "alpha_pred": alpha_pred,
        "alpha_loao": alpha_loao,
        "error": error,
        "abs_error": abs_error,
        "n_datasets": len(rho_vals),
    }

    print(f"  {model_short}: rho_pooled={rho_pooled:.4f}, "
          f"alpha_pred={alpha_pred:.4f}, alpha_loao={alpha_loao:.4f}, "
          f"err={error:+.4f}", flush=True)

# ============================================================
# COMPUTE STATISTICS
# ============================================================
valid_models = sorted(model_results.keys())
N_models = len(valid_models)
print(f"\nN_models = {N_models}", flush=True)

alpha_pred_arr = np.array([model_results[m]["alpha_pred"] for m in valid_models])
alpha_loao_arr = np.array([model_results[m]["alpha_loao"] for m in valid_models])

# Raw statistics
if N_models >= 3:
    raw_pearson_r, raw_pearson_p = pearsonr(alpha_pred_arr, alpha_loao_arr)
    raw_spearman_r, raw_spearman_p = spearmanr(alpha_pred_arr, alpha_loao_arr)
else:
    raw_pearson_r = raw_pearson_p = float("nan")
    raw_spearman_r = raw_spearman_p = float("nan")

raw_mae = float(np.mean(np.abs(alpha_pred_arr - alpha_loao_arr)))
raw_rmse = float(np.sqrt(np.mean((alpha_pred_arr - alpha_loao_arr)**2)))
mean_error = float(np.mean(alpha_pred_arr - alpha_loao_arr))
mean_rel_error = float(np.mean(np.abs(alpha_pred_arr - alpha_loao_arr) / alpha_loao_arr))

print(f"\n--- RAW STATISTICS ---", flush=True)
print(f"  Pearson r  = {raw_pearson_r:.4f} (p={raw_pearson_p:.4f})", flush=True)
print(f"  Spearman r = {raw_spearman_r:.4f} (p={raw_spearman_p:.4f})", flush=True)
print(f"  MAE        = {raw_mae:.4f}", flush=True)
print(f"  RMSE       = {raw_rmse:.4f}", flush=True)
print(f"  Mean error = {mean_error:+.4f}", flush=True)
print(f"  Mean relative error = {mean_rel_error:.3f} ({mean_rel_error*100:.1f}%)", flush=True)

# Reliability of alpha_pred
alpha_pred_point = {m: model_results[m]["alpha_pred"] for m in valid_models}
rel_alpha_pred = compute_reliability(alpha_pred_boot_traces, alpha_pred_point,
                                     N_models)
rel_alpha_loao = 1.0  # LOAO from 444 pts is ground truth

print(f"\n--- RELIABILITY ---", flush=True)
print(f"  rel(alpha_pred) = {rel_alpha_pred:.4f}", flush=True)
print(f"  rel(alpha_loao) = {rel_alpha_loao:.4f} (assumed)", flush=True)

# Disattenuated Spearman r
if N_models >= 3 and rel_alpha_pred > 0.01:
    disatten_denom = np.sqrt(rel_alpha_pred * rel_alpha_loao)
    disatten_spearman = float(raw_spearman_r / disatten_denom)
    disatten_spearman = float(np.clip(disatten_spearman, -1.0, 1.0))

    # BCa CI for disattenuated Spearman r
    def disatten_stat(x, y):
        r_sp, _ = spearmanr(x, y)
        return float(np.clip(r_sp / disatten_denom, -1.0, 1.0))

    _, ci_lo, ci_hi = bca_ci(
        alpha_pred_arr, alpha_loao_arr,
        disatten_stat, n_boot=N_BOOT_CORR, seed=SEED,
    )
else:
    disatten_spearman = float("nan")
    ci_lo = ci_hi = float("nan")

print(f"\n--- DISATTENUATED SPEARMAN ---", flush=True)
print(f"  r_disatten = {disatten_spearman:.4f}", flush=True)
print(f"  BCa 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)

# ============================================================
# PASS/FAIL
# ============================================================
pass_mae       = raw_mae <= PASS_RAW_MAE_MAX
pass_raw_r     = raw_pearson_r >= PASS_RAW_R_MIN
pass_disatten  = disatten_spearman >= PASS_DISATTEN_SPEARMAN_MIN

print(f"\n{'='*60}", flush=True)
print(f"PRE-REGISTERED CRITERIA", flush=True)
print(f"  H1: MAE < {PASS_RAW_MAE_MAX}:          "
      f"{raw_mae:.4f} -> {'PASS' if pass_mae else 'FAIL'}", flush=True)
print(f"  H2: raw Pearson r > {PASS_RAW_R_MIN}:    "
      f"{raw_pearson_r:.4f} -> {'PASS' if pass_raw_r else 'FAIL'}", flush=True)
print(f"  H3: disatten Spearman > {PASS_DISATTEN_SPEARMAN_MIN}: "
      f"{disatten_spearman:.4f} -> {'PASS' if pass_disatten else 'FAIL'}", flush=True)

# ============================================================
# SAVE RESULTS
# ============================================================
output = {
    "experiment": "cti_alpha_rho_multidataset",
    "date": "2026-03-01",
    "description": (
        "Multi-dataset alpha(rho) validation with bootstrap. "
        "Tests alpha = sqrt(4/pi)/sqrt(1-rho) across 3 datasets "
        "(AG News K=4, DBpedia K=14, Banking77 K=77) with 200 "
        "stratified bootstrap resamples per (model, dataset) pair. "
        "Includes Spearman disattenuation correction for measurement noise."
    ),
    "theory": {
        "formula": "alpha = sqrt(4/pi) / sqrt(1 - rho)",
        "A_renorm": A_RENORM,
        "datasets": list(DATASETS.keys()),
        "K_values": {ds: cfg["K"] for ds, cfg in DATASETS.items()},
    },
    "pre_registered_criteria": {
        "H1_mae_max": PASS_RAW_MAE_MAX,
        "H2_raw_pearson_r_min": PASS_RAW_R_MIN,
        "H3_disatten_spearman_min": PASS_DISATTEN_SPEARMAN_MIN,
    },
    "config": {
        "n_boot_rho": RHO_BOOT,
        "boot_frac": RHO_BOOT_FRAC,
        "n_boot_corr": N_BOOT_CORR,
        "seed": SEED,
        "n_pca": N_PCA,
        "n_samples_per_dataset": 2000,
    },
    "aggregate": {
        "N_models": N_models,
        "raw_pearson_r": float(raw_pearson_r),
        "raw_pearson_p": float(raw_pearson_p),
        "raw_spearman_r": float(raw_spearman_r),
        "raw_spearman_p": float(raw_spearman_p),
        "raw_mae": raw_mae,
        "raw_rmse": raw_rmse,
        "mean_error": mean_error,
        "mean_rel_error": mean_rel_error,
        "reliability_alpha_pred": rel_alpha_pred,
        "reliability_alpha_loao": rel_alpha_loao,
        "disattenuated_spearman": float(disatten_spearman),
        "disattenuated_spearman_bca_ci_95": [float(ci_lo), float(ci_hi)],
        "pass_H1_mae": bool(pass_mae),
        "pass_H2_raw_r": bool(pass_raw_r),
        "pass_H3_disatten": bool(pass_disatten),
        "alpha_pred_mean": float(alpha_pred_arr.mean()),
        "alpha_loao_mean": float(alpha_loao_arr.mean()),
    },
    "per_model": model_results,
    "elapsed_s": time.time() - t_start,
}

out_path = RESULTS_DIR / "cti_alpha_rho_multidataset.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {out_path}", flush=True)
print(f"Total elapsed: {time.time()-t_start:.0f}s", flush=True)
print("Done.", flush=True)
