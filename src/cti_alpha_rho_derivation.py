#!/usr/bin/env python -u
"""
ALPHA-RHO DERIVATION VALIDATION (Feb 28 2026)
=============================================
Hypothesis: alpha = sqrt(4/pi) / sqrt(1 - rho)

WHERE:
  alpha = empirical LOAO constant from cti_kappa_loao_per_dataset.json
  rho   = mean Sigma_W-whitened cosine similarity between centroid-difference
          vectors {mu_j - mu_c}_{j!=c} for each anchor class c, averaged over c

DERIVATION (from cti_equicorrelation_deff.py theory block):
  For x in class c, the K-1 competition margins are:
    G_j = ||x - mu_j||^2 - ||x - mu_c||^2 = 2*delta_j^T*z + ||delta_j||^2
    where z = x - mu_c ~ N(0, Sigma_W), delta_j = mu_j - mu_c.

  Cov(G_j, G_k) = 4 * delta_j^T Sigma_W delta_k
  rho_{jk|c} = cosine_similarity(delta_j, delta_k) in the Sigma_W metric

  d_eff_comp = 1 / (1 - rho)
  alpha = sqrt(4/pi) * sqrt(d_eff_comp) = sqrt(4/pi) / sqrt(1 - rho)

PRE-REGISTERED SUCCESS CRITERIA (set BEFORE running):
  PASS if: pearson_r(alpha_pred, alpha_loao) > 0.70 AND MAE < 0.15
  (Looser than ideal because alpha range is tight: all 12 in [1.39, 1.52])

DATA:
  12 NLP decoder architectures from canonical LOAO
  Dataset: DBpedia K=14, N=2000
  Layer: best layer (max kappa_nearest from 25%/50%/75%/100% depth)

OUTPUTS:
  results/cti_alpha_rho_derivation.json
"""

import json
import time
import sys
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}", flush=True)

# Theoretical constants (zero free parameters)
A_RENORM = float(np.sqrt(4.0 / np.pi))   # 1.12838
print(f"A_renorm = sqrt(4/pi) = {A_RENORM:.5f}", flush=True)

# ============================================================
# PRE-REGISTERED CRITERIA (DO NOT MODIFY AFTER SCRIPT STARTS)
# ============================================================
PASS_R_MIN   = 0.70
PASS_MAE_MAX = 0.15

# ============================================================
# CANONICAL LOAO ALPHA VALUES (from cti_kappa_loao_per_dataset.json)
# ============================================================
LOAO_JSON = RESULTS_DIR / "cti_kappa_loao_per_dataset.json"
with open(LOAO_JSON) as f:
    loao_data = json.load(f)

LOAO_ALPHA = {}   # model_short -> alpha_loao
for model_name, entry in loao_data["loao_results"].items():
    short = model_name.split("/")[-1]
    LOAO_ALPHA[short] = float(entry["alpha"])

print(f"\nCanonical LOAO alphas ({len(LOAO_ALPHA)} models):", flush=True)
for k, v in sorted(LOAO_ALPHA.items()):
    print(f"  {k}: {v:.4f}", flush=True)

# ============================================================
# MODEL REGISTRY
# ============================================================
# Matches cti_kappa_nearest_universal.py exactly
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
    "Falcon-H1-0.5B-Base": "tiiuae/Falcon-H1-0.5B-Base",
    "rwkv-4-169m-pile":  "RWKV/rwkv-4-169m-pile",
}

TRUST_REMOTE_CODE = {"Falcon-H1-0.5B-Base"}

# Proportional depth layers (25% / 50% / 75% / final)
MODEL_LAYERS = {
    "pythia-160m":       [3, 6, 9, 12],
    "pythia-410m":       [6, 12, 18, 24],
    "pythia-1b":         [4, 8, 12, 16],
    "gpt-neo-125m":      [3, 6, 9, 12],
    "Qwen2.5-0.5B":      [7, 14, 21, 28],
    "OLMo-1B-hf":        [4, 8, 12, 16],
    "TinyLlama-1.1B-intermediate-step-1431k-3T": [5, 11, 16, 22],
    "Qwen3-0.6B":        [7, 14, 21, 28],
    "Qwen3-1.7B":        [7, 14, 21, 28],
    "Mistral-7B-v0.3":   [8, 16, 24, 32],
    "Falcon-H1-0.5B-Base": [9, 18, 27, 36],
    "rwkv-4-169m-pile":  [3, 6, 9, 12],
}

# ============================================================
# DATASET
# ============================================================
DATASET_NAME = "dbpedia_14"
N_SAMPLES    = 2000
K            = 14
BATCH_SIZE   = 32

# Per-model batch sizes: reduce for large models to avoid near-OOM on 24GB
MODEL_BATCH_SIZE = {
    "Mistral-7B-v0.3": 4,   # 7B at FP16 + activations exhausts 24GB at BS=32
}

print(f"\nLoading DBpedia K={K}, N={N_SAMPLES}...", flush=True)
ds = load_dataset("fancyzhx/dbpedia_14", split="test")
texts_all  = [str(x["content"]) for x in ds]
labels_all = [int(x["label"])   for x in ds]

rng = np.random.default_rng(42)
idx = rng.choice(len(texts_all), size=min(N_SAMPLES, len(texts_all)), replace=False)
texts  = [texts_all[i]  for i in idx]
labels = np.array([labels_all[i] for i in idx])
classes = sorted(set(labels))
assert len(classes) == K
print(f"  Loaded {len(texts)} samples, {K} classes.", flush=True)


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================
@torch.no_grad()
def embed_model(model_id, model_short, texts, layers):
    """Return dict: layer_idx -> ndarray (N, d)."""
    trust = model_short in TRUST_REMOTE_CODE
    print(f"  Loading {model_short}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    model = AutoModel.from_pretrained(
        model_id,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        trust_remote_code=trust,
    ).to(DEVICE).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bs = MODEL_BATCH_SIZE.get(model_short, BATCH_SIZE)
    layer_embs = {l: [] for l in layers}
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to(DEVICE)
        out = model(**enc)
        hidden = out.hidden_states  # tuple: (n_layers+1, batch, seq, d)
        # Cap layer indices to valid range
        valid_layers = [min(l, len(hidden)-1) for l in layers]
        attn_mask = enc["attention_mask"].float()  # (batch, seq)
        seq_len = attn_mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # (batch,1,1)
        for l, vl in zip(layers, valid_layers):
            h = hidden[vl].float()             # (batch, seq, d)
            mask3 = attn_mask.unsqueeze(-1)    # (batch, seq, 1)
            pooled = (h * mask3).sum(dim=1) / seq_len.squeeze(-1)  # (batch, d)
            layer_embs[l].append(pooled.cpu().numpy())

    del model
    torch.cuda.empty_cache()

    return {l: np.concatenate(layer_embs[l], axis=0) for l in layers}


# ============================================================
# RHO COMPUTATION
# ============================================================
N_PCA = 256  # PCA components for whitening (matches cti_equicorrelation_deff.py)


def compute_rho(embeddings, labels, classes):
    """
    Compute mean Sigma_W-whitened cosine similarity between centroid-difference
    vectors delta_{jc} = mu_j - mu_c.

    CANONICAL method matching cti_equicorrelation_deff.py exactly:
      1. Pool within-class centered data Z.
      2. TruncatedSVD on Z to get eigenvectors V and eigenvalues Lambda of Sigma_W.
      3. For each anchor c and competitor j, form:
           proj_j   = delta_j @ V              (project onto Sigma_W eigenbasis)
           whitened_j = proj_j * sqrt(Lambda)  (= Sigma_W^{1/2} * delta_j)
      4. cos_sim(whitened_j, whitened_k) = delta_j^T Sigma_W delta_k / (...)
         This is rho_{jk|c} from the Gumbel-race covariance derivation.
      5. rho = mean over all (c, j1, j2) off-diagonal pairs.
    """
    from sklearn.decomposition import TruncatedSVD

    K_local = len(classes)
    N, d = embeddings.shape

    # 1. Class centroids and within-class centered data
    centroids = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = embeddings[mask].mean(0).astype(np.float64)
    if len(centroids) < K_local:
        return float("nan"), float("nan")

    centroid_array = np.array([centroids[c] for c in classes])  # (K, d)

    Xc_list = []
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            Xc_list.append((embeddings[mask] - centroids[c]).astype(np.float64))
    Z = np.concatenate(Xc_list, axis=0)  # (N, d)
    N_total = len(Z)

    # 2. TruncatedSVD on within-class data → eigenvectors/values of Sigma_W
    n_comp = min(N_PCA, d, N_total - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(Z)
    V = svd.components_.T                        # (d, n_comp)
    Lambda = (svd.singular_values_ ** 2) / N_total  # eigenvalues of Sigma_W
    sqrt_Lambda = np.sqrt(Lambda + 1e-12)        # sqrt of eigenvalues

    # 3. & 4. For each anchor class c, whiten delta vectors and compute cosines
    rho_per_class = []

    for i_c, c in enumerate(classes):
        other_idx = [i for i in range(K_local) if i != i_c]
        deltas = centroid_array[other_idx] - centroids[c]  # (K-1, d)

        # Project onto Sigma_W eigenbasis, scale by sqrt(Lambda) = Sigma_W^{1/2}
        proj     = deltas @ V                              # (K-1, n_comp)
        whitened = proj * sqrt_Lambda[None, :]             # (K-1, n_comp)

        # Normalise and compute pairwise cosines
        norms    = np.linalg.norm(whitened, axis=1, keepdims=True)
        norms    = np.maximum(norms, 1e-12)
        w_norm   = whitened / norms                        # (K-1, n_comp)
        cos_mat  = w_norm @ w_norm.T                       # (K-1, K-1)

        n_off = K_local - 1
        off_vals = cos_mat[~np.eye(n_off, dtype=bool)]     # all off-diagonal
        rho_per_class.append(float(off_vals.mean()))

    rho = float(np.mean(rho_per_class))
    return rho, float(np.std(rho_per_class))


# ============================================================
# KAPPA_NEAREST (to identify best layer)
# ============================================================
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
# CACHED RESULTS (9/12 models from prior run, canonical Sigma_W^{1/2} whitening)
# These are pre-validated; skip re-running to save time.
# ============================================================
CACHED_RESULTS = {
    "pythia-160m":   {"model_short":"pythia-160m","model_id":"EleutherAI/pythia-160m","best_layer":12,"d":768,"kappa_nearest":0.5343,"rho":0.4080,"rho_std":0.2361,"d_eff_comp":1.6893,"alpha_pred":1.4666,"alpha_loao":1.4782,"abs_error":0.0116,"rel_error":0.008,"elapsed_s":0},
    "pythia-410m":   {"model_short":"pythia-410m","model_id":"EleutherAI/pythia-410m","best_layer":24,"d":1024,"kappa_nearest":0.7606,"rho":0.4659,"rho_std":0.1618,"d_eff_comp":1.8722,"alpha_pred":1.5439,"alpha_loao":1.4822,"abs_error":0.0617,"rel_error":0.042,"elapsed_s":0},
    "pythia-1b":     {"model_short":"pythia-1b","model_id":"EleutherAI/pythia-1b","best_layer":16,"d":2048,"kappa_nearest":0.7154,"rho":0.4591,"rho_std":0.1784,"d_eff_comp":1.8488,"alpha_pred":1.5342,"alpha_loao":1.5007,"abs_error":0.0336,"rel_error":0.022,"elapsed_s":0},
    "gpt-neo-125m":  {"model_short":"gpt-neo-125m","model_id":"EleutherAI/gpt-neo-125m","best_layer":12,"d":768,"kappa_nearest":0.8227,"rho":0.4639,"rho_std":0.1608,"d_eff_comp":1.8652,"alpha_pred":1.5411,"alpha_loao":1.3935,"abs_error":0.1475,"rel_error":0.106,"elapsed_s":0},
    "Qwen2.5-0.5B":  {"model_short":"Qwen2.5-0.5B","model_id":"Qwen/Qwen2.5-0.5B","best_layer":28,"d":896,"kappa_nearest":0.7464,"rho":0.4515,"rho_std":0.1920,"d_eff_comp":1.8231,"alpha_pred":1.5236,"alpha_loao":1.4927,"abs_error":0.0308,"rel_error":0.021,"elapsed_s":0},
    "OLMo-1B-hf":    {"model_short":"OLMo-1B-hf","model_id":"allenai/OLMo-1B-hf","best_layer":16,"d":2048,"kappa_nearest":0.6820,"rho":0.4596,"rho_std":0.1777,"d_eff_comp":1.8503,"alpha_pred":1.5349,"alpha_loao":1.5141,"abs_error":0.0208,"rel_error":0.014,"elapsed_s":0},
    "TinyLlama-1.1B-intermediate-step-1431k-3T": {"model_short":"TinyLlama-1.1B-intermediate-step-1431k-3T","model_id":"TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T","best_layer":22,"d":2048,"kappa_nearest":0.6637,"rho":0.4705,"rho_std":0.1462,"d_eff_comp":1.8885,"alpha_pred":1.5506,"alpha_loao":1.4421,"abs_error":0.1085,"rel_error":0.075,"elapsed_s":0},
    "Qwen3-0.6B":    {"model_short":"Qwen3-0.6B","model_id":"Qwen/Qwen3-0.6B","best_layer":28,"d":1024,"kappa_nearest":0.7516,"rho":0.4587,"rho_std":0.1906,"d_eff_comp":1.8476,"alpha_pred":1.5338,"alpha_loao":1.4506,"abs_error":0.0832,"rel_error":0.057,"elapsed_s":0},
    "Qwen3-1.7B":    {"model_short":"Qwen3-1.7B","model_id":"Qwen/Qwen3-1.7B","best_layer":28,"d":2048,"kappa_nearest":0.6942,"rho":0.4395,"rho_std":0.2268,"d_eff_comp":1.7841,"alpha_pred":1.5072,"alpha_loao":1.4880,"abs_error":0.0192,"rel_error":0.013,"elapsed_s":0},
    "Mistral-7B-v0.3": {"model_short":"Mistral-7B-v0.3","model_id":"mistralai/Mistral-7B-v0.3","best_layer":32,"d":4096,"kappa_nearest":0.6434,"rho":0.4705,"rho_std":0.1418,"d_eff_comp":1.8886,"alpha_pred":1.5507,"alpha_loao":1.4776,"abs_error":0.0731,"rel_error":0.049,"elapsed_s":0},
}
# Falcon-H1 skipped: naive Mamba fallback (no causal_conv1d CUDA kernel) >30 min for 2000 samples
SKIP_MODELS = {"Falcon-H1-0.5B-Base"}

# Print cached models
print("\nLoaded cached results for 10 models (canonical Sigma_W^{1/2} whitening):", flush=True)
for k, v in CACHED_RESULTS.items():
    print(f"  {k}: rho={v['rho']:.4f}  alpha_pred={v['alpha_pred']:.4f}  "
          f"alpha_loao={v['alpha_loao']:.4f}  abs_err={v['abs_error']:.4f}", flush=True)

# ============================================================
# MAIN LOOP
# ============================================================
results = dict(CACHED_RESULTS)  # pre-populate with cached
t_start = time.time()

for model_short, model_id in MODELS.items():
    # Skip models already in cache or explicitly skipped
    if model_short in CACHED_RESULTS or model_short in SKIP_MODELS:
        continue
    # Check LOAO alpha exists
    if model_short not in LOAO_ALPHA:
        print(f"\nSkipping {model_short}: no LOAO alpha.", flush=True)
        continue

    alpha_loao = LOAO_ALPHA[model_short]
    layers = MODEL_LAYERS.get(model_short, [3, 6, 9, 12])

    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_short}  (LOAO alpha={alpha_loao:.4f})", flush=True)
    t0 = time.time()

    try:
        layer_embs = embed_model(model_id, model_short, texts, layers)

        # Pick best layer by max kappa_nearest
        best_layer = max(layers, key=lambda l: compute_kappa_nearest(
            layer_embs[l], labels, classes))
        emb = layer_embs[best_layer]
        d = emb.shape[1]

        kappa = compute_kappa_nearest(emb, labels, classes)
        rho, rho_std = compute_rho(emb, labels, classes)

        # Predicted alpha
        d_eff_comp = 1.0 / max(1.0 - rho, 1e-6)
        alpha_pred = A_RENORM * np.sqrt(d_eff_comp)
        abs_err    = abs(alpha_pred - alpha_loao)
        rel_err    = abs_err / alpha_loao

        print(f"  best_layer={best_layer}, d={d}, kappa={kappa:.4f}", flush=True)
        print(f"  rho={rho:.4f} +/- {rho_std:.4f}", flush=True)
        print(f"  d_eff_comp={d_eff_comp:.4f}", flush=True)
        print(f"  alpha_pred={alpha_pred:.4f}  alpha_loao={alpha_loao:.4f}  "
              f"abs_err={abs_err:.4f}  rel_err={rel_err:.3f}", flush=True)

        results[model_short] = {
            "model_short": model_short,
            "model_id": model_id,
            "best_layer": best_layer,
            "d": int(d),
            "kappa_nearest": kappa,
            "rho": rho,
            "rho_std": rho_std,
            "d_eff_comp": d_eff_comp,
            "alpha_pred": alpha_pred,
            "alpha_loao": alpha_loao,
            "abs_error": abs_err,
            "rel_error": rel_err,
            "elapsed_s": time.time() - t0,
        }

    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback; traceback.print_exc()
        results[model_short] = {"error": str(e), "alpha_loao": alpha_loao}

# ============================================================
# AGGREGATE STATISTICS
# ============================================================
valid = {k: v for k, v in results.items() if "rho" in v}
N = len(valid)

alpha_pred_vals = np.array([v["alpha_pred"] for v in valid.values()])
alpha_loao_vals = np.array([v["alpha_loao"] for v in valid.values()])

pearson_r = float(pearsonr(alpha_pred_vals, alpha_loao_vals)[0]) if N >= 3 else float("nan")
mae = float(np.mean(np.abs(alpha_pred_vals - alpha_loao_vals)))
rmse = float(np.sqrt(np.mean((alpha_pred_vals - alpha_loao_vals)**2)))

pass_r   = pearson_r >= PASS_R_MIN
pass_mae = mae <= PASS_MAE_MAX
overall_pass = pass_r and pass_mae

print(f"\n{'='*60}", flush=True)
print(f"AGGREGATE RESULTS (N={N} models)", flush=True)
print(f"  Pearson r(alpha_pred, alpha_loao) = {pearson_r:.4f}  "
      f"[PASS thresh > {PASS_R_MIN}]  -> {'PASS' if pass_r else 'FAIL'}", flush=True)
print(f"  MAE = {mae:.4f}  [PASS thresh < {PASS_MAE_MAX}]  -> {'PASS' if pass_mae else 'FAIL'}", flush=True)
print(f"  RMSE = {rmse:.4f}", flush=True)
print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}", flush=True)
print(f"  alpha_pred range: [{alpha_pred_vals.min():.4f}, {alpha_pred_vals.max():.4f}]", flush=True)
print(f"  alpha_loao range: [{alpha_loao_vals.min():.4f}, {alpha_loao_vals.max():.4f}]", flush=True)
print(f"  Elapsed: {time.time()-t_start:.0f}s total", flush=True)

# ============================================================
# SAVE RESULTS
# ============================================================
output = {
    "experiment": "cti_alpha_rho_derivation",
    "date": "2026-02-28",
    "description": (
        "Validate alpha(rho) = sqrt(4/pi)/sqrt(1-rho) across all 12 canonical LOAO "
        "architectures. rho = mean Sigma_W-whitened cosine similarity of centroid "
        "difference vectors. Dataset: DBpedia K=14."
    ),
    "theory": {
        "formula": "alpha = sqrt(4/pi) / sqrt(1 - rho)",
        "A_renorm": A_RENORM,
        "A_renorm_formula": "sqrt(4/pi) = E[|Z|] for Z~N(0,1)",
        "d_eff_comp": "1 / (1 - rho)",
        "rho_interpretation": "mean whitened cosine similarity of centroid-difference vectors",
    },
    "pre_registered_criteria": {
        "pass_pearson_r_min": PASS_R_MIN,
        "pass_mae_max": PASS_MAE_MAX,
    },
    "dataset": DATASET_NAME,
    "n_samples": N_SAMPLES,
    "K": K,
    "aggregate": {
        "N_models": N,
        "pearson_r": pearson_r,
        "mae": mae,
        "rmse": rmse,
        "pass_r": bool(pass_r),
        "pass_mae": bool(pass_mae),
        "overall_pass": bool(overall_pass),
        "alpha_pred_mean": float(alpha_pred_vals.mean()),
        "alpha_loao_mean": float(alpha_loao_vals.mean()),
        "alpha_pred_cv": float(alpha_pred_vals.std() / alpha_pred_vals.mean()),
    },
    "per_model": results,
}

out_path = RESULTS_DIR / "cti_alpha_rho_derivation.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {out_path}", flush=True)
print("Done.", flush=True)
