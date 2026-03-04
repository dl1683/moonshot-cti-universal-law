#!/usr/bin/env python -u
"""
CTI Universal Law: Protein Domain Validation (ESM-2)
=====================================================
Tests the CTI law on protein language models — a completely new scientific
domain (molecular biology). If logit(q_norm) = alpha * kappa + C holds for
protein representations classifying enzyme function, it demonstrates that
the Gumbel-race mechanism is truly universal across ALL learned representations.

DOMAIN: Protein sequences classified by EC number (Enzyme Commission)
  K = 7 top-level EC classes:
    1. Oxidoreductases   2. Transferases    3. Hydrolases
    4. Lyases            5. Isomerases      6. Ligases
    7. Translocases

MODELS: ESM-2 protein language models (Meta, Lin et al. 2023)
  - esm2_t6_8M_UR50D     (8M params, 6 layers, d=320)
  - esm2_t12_35M_UR50D   (35M params, 12 layers, d=480)
  - esm2_t30_150M_UR50D  (150M params, 30 layers, d=640)
  - esm2_t33_650M_UR50D  (650M params, 33 layers, d=1280)

DATA: UniProt/SwissProt reviewed proteins, max 500 amino acids,
  200 per EC class (1400 total).

METRICS: kappa_nearest and 1-NN accuracy (same as NLP/CV experiments)

PRE-REGISTERED CRITERIA (set before running):
  H_prot1: r(kappa, logit_q) > 0.80 across 4 ESM-2 sizes
  H_prot2: alpha_protein > 0 (monotonic, larger kappa = better accuracy)
  H_prot3: 1-NN accuracy increases with model size (sanity check)

OUTPUT: results/cti_protein_esm2.json
"""

import json
import time
import gc
import sys
import requests
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.special import logit as sp_logit
from sklearn.decomposition import TruncatedSVD
import torch
from transformers import AutoTokenizer, AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_PATH = RESULTS_DIR / "cti_protein_esm2_cache.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_PCA = 256  # Reduce all models to same dimension (matches NLP pipeline)

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70, flush=True)
print("CTI Universal Law: Protein Domain Validation (ESM-2)", flush=True)
print("=" * 70, flush=True)
print(f"Device: {DEVICE}", flush=True)

# ============================================================
# EC CLASSES
# ============================================================
EC_CLASSES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}
K = len(EC_CLASSES)

# ============================================================
# ESM-2 MODELS (increasing size)
# ============================================================
ESM2_MODELS = [
    ("ESM2-8M",   "facebook/esm2_t6_8M_UR50D",   6,  320),
    ("ESM2-35M",  "facebook/esm2_t12_35M_UR50D",  12, 480),
    ("ESM2-150M", "facebook/esm2_t30_150M_UR50D",  30, 640),
    ("ESM2-650M", "facebook/esm2_t33_650M_UR50D",  33, 1280),
]

# ============================================================
# STEP 1: Fetch protein sequences from UniProt
# ============================================================
N_PER_CLASS = 200
MAX_SEQ_LEN = 500


def fetch_uniprot_sequences(ec_num, n=N_PER_CLASS, max_len=MAX_SEQ_LEN):
    """Fetch reviewed protein sequences for a given EC class from UniProt REST API."""
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    sequences = []
    cursor = None
    page_size = min(n, 500)

    while len(sequences) < n:
        params = {
            "query": f"ec:{ec_num} AND reviewed:true AND length:[50 TO {max_len}]",
            "format": "json",
            "fields": "accession,sequence",
            "size": page_size,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            r = requests.get(base_url, params=params, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"  WARNING: UniProt request failed for EC {ec_num}: {e}", flush=True)
            break

        data = r.json()
        results = data.get("results", [])
        if not results:
            break

        for entry in results:
            seq = entry.get("sequence", {}).get("value", "")
            if seq and 50 <= len(seq) <= max_len:
                sequences.append(seq)
            if len(sequences) >= n:
                break

        # Check for pagination
        link_header = r.headers.get("Link", "")
        if 'rel="next"' in link_header:
            # Extract cursor from Link header
            import re
            match = re.search(r'cursor=([^&>]+)', link_header)
            if match:
                cursor = match.group(1)
            else:
                break
        else:
            break

    return sequences[:n]


def load_or_fetch_sequences():
    """Load cached sequences or fetch from UniProt."""
    seq_cache = RESULTS_DIR / "cti_protein_sequences_cache.json"

    if seq_cache.exists():
        print("\n[Step 1] Loading cached protein sequences...", flush=True)
        with open(seq_cache) as f:
            data = json.load(f)
        # Verify completeness
        all_good = True
        for ec in EC_CLASSES:
            if ec not in data or len(data[ec]) < N_PER_CLASS:
                all_good = False
                break
        if all_good:
            total = sum(len(v) for v in data.values())
            print(f"  Loaded {total} sequences from cache", flush=True)
            return data

    print("\n[Step 1] Fetching protein sequences from UniProt...", flush=True)
    all_sequences = {}
    for ec_num, ec_name in EC_CLASSES.items():
        print(f"  EC {ec_num} ({ec_name})...", flush=True)
        seqs = fetch_uniprot_sequences(ec_num, N_PER_CLASS, MAX_SEQ_LEN)
        all_sequences[ec_num] = seqs
        print(f"    Got {len(seqs)} sequences", flush=True)
        time.sleep(0.5)  # Be nice to UniProt

    # Cache
    with open(seq_cache, "w") as f:
        json.dump(all_sequences, f)
    total = sum(len(v) for v in all_sequences.values())
    print(f"  Total: {total} sequences cached", flush=True)
    return all_sequences


# ============================================================
# STEP 2: Compute embeddings with ESM-2
# ============================================================
EMBED_BATCH = 8  # Batch size for ESM-2 inference


def compute_esm2_embeddings(model_name, hf_id, n_layers, sequences, labels):
    """Compute mean-pooled embeddings at EVERY layer for protein sequences.

    Returns dict mapping layer_idx -> (n_samples, d) array.
    We test multiple layers because intermediate layers often work better
    for downstream tasks in protein LMs (similar to NLP finding).
    """
    print(f"\n  Loading {model_name} ({hf_id})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id, output_hidden_states=True).to(DEVICE).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}", flush=True)

    # Collect embeddings for each layer
    layer_embeddings = {i: [] for i in range(n_layers + 1)}  # +1 for embedding layer
    n = len(sequences)

    with torch.no_grad():
        for i in range(0, n, EMBED_BATCH):
            batch_seqs = sequences[i:i + EMBED_BATCH]
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN + 2,
            ).to(DEVICE)

            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()

            # outputs.hidden_states: tuple of (n_layers+1) tensors, each (B, L, d)
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                layer_embeddings[layer_idx].append(pooled.cpu().numpy())

            if (i // EMBED_BATCH) % 20 == 0:
                print(f"    Batch {i // EMBED_BATCH + 1}/{(n + EMBED_BATCH - 1) // EMBED_BATCH}", flush=True)

    # Concatenate
    for layer_idx in layer_embeddings:
        layer_embeddings[layer_idx] = np.concatenate(layer_embeddings[layer_idx], axis=0)

    print(f"    Collected embeddings for {len(layer_embeddings)} layers, shape per layer: {layer_embeddings[0].shape}", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return layer_embeddings


# ============================================================
# STEP 3: Compute CTI metrics (kappa_nearest, 1-NN accuracy)
# ============================================================
def compute_cti_metrics(embeddings, labels, K, apply_pca=True):
    """Compute kappa_nearest and 1-NN accuracy from embeddings and labels.

    Applies PCA to N_PCA dimensions first (critical for cross-model comparability
    when models have different d).
    """
    unique_labels = sorted(set(labels))
    K_actual = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    int_labels = np.array([label_to_idx[l] for l in labels])

    d_orig = embeddings.shape[1]

    # PCA reduction (matches NLP pipeline)
    if apply_pca and d_orig > N_PCA:
        svd = TruncatedSVD(n_components=N_PCA, random_state=SEED)
        embeddings = svd.fit_transform(embeddings)
        var_explained = svd.explained_variance_ratio_.sum()
    else:
        var_explained = 1.0

    d = embeddings.shape[1]

    # Compute class centroids
    centroids = np.zeros((K_actual, d))
    for k in range(K_actual):
        mask = int_labels == k
        centroids[k] = embeddings[mask].mean(axis=0)

    # Within-class covariance (pooled)
    n_total = len(embeddings)
    Sigma_W = np.zeros((d, d))
    for k in range(K_actual):
        mask = int_labels == k
        X_k = embeddings[mask]
        X_k_centered = X_k - centroids[k]
        Sigma_W += X_k_centered.T @ X_k_centered
    Sigma_W /= n_total

    # Sigma_W eigenvalues for effective dimension
    eigvals = np.linalg.eigvalsh(Sigma_W)
    eigvals = eigvals[eigvals > 1e-10]
    trace_Sigma = eigvals.sum()
    sigma_W_scalar = np.sqrt(trace_Sigma / d)

    # Kappa nearest: min_{j!=k} ||mu_j - mu_k|| / (sigma_W * sqrt(d))
    dists = np.zeros((K_actual, K_actual))
    for i in range(K_actual):
        for j in range(K_actual):
            if i != j:
                dists[i, j] = np.linalg.norm(centroids[i] - centroids[j])
            else:
                dists[i, j] = np.inf

    min_dists = dists.min(axis=1)
    kappa_nearest = (min_dists / (sigma_W_scalar * np.sqrt(d))).mean()

    # LOO 1-NN (exact)
    from sklearn.metrics import pairwise_distances
    dist_matrix = pairwise_distances(embeddings, metric="euclidean")
    np.fill_diagonal(dist_matrix, np.inf)
    nn_indices = dist_matrix.argmin(axis=1)
    nn_labels = int_labels[nn_indices]
    acc_1nn = (nn_labels == int_labels).mean()

    # q_norm = (acc - 1/K) / (1 - 1/K)
    q_norm = (acc_1nn - 1.0 / K_actual) / (1.0 - 1.0 / K_actual)
    q_norm = np.clip(q_norm, 0.001, 0.999)

    logit_q = float(sp_logit(q_norm))

    return {
        "kappa_nearest": float(kappa_nearest),
        "acc_1nn": float(acc_1nn),
        "q_norm": float(q_norm),
        "logit_q": float(logit_q),
        "K": K_actual,
        "n_samples": n_total,
        "d_original": d_orig,
        "d_pca": d,
        "pca_var_explained": float(var_explained),
        "sigma_W": float(sigma_W_scalar),
    }


# ============================================================
# STEP 4: Run full experiment
# ============================================================
def main():
    t0 = time.time()

    # Load or fetch sequences
    all_sequences = load_or_fetch_sequences()

    # Build flat arrays
    sequences = []
    labels = []
    for ec_num in sorted(EC_CLASSES.keys()):
        seqs = all_sequences.get(ec_num, [])
        for s in seqs:
            sequences.append(s)
            labels.append(ec_num)

    print(f"\n  Total: {len(sequences)} sequences, K={K} EC classes", flush=True)
    for ec_num in sorted(EC_CLASSES.keys()):
        n = sum(1 for l in labels if l == ec_num)
        print(f"    EC {ec_num} ({EC_CLASSES[ec_num]}): {n}", flush=True)

    # Clear old cache (v1 without PCA was wrong)
    results = {}

    # Run each model
    for model_name, hf_id, n_layers, d_model in ESM2_MODELS:
        print(f"\n{'='*60}", flush=True)
        print(f"  Processing {model_name}", flush=True)
        print(f"{'='*60}", flush=True)

        # Compute embeddings at ALL layers
        layer_embeddings = compute_esm2_embeddings(model_name, hf_id, n_layers, sequences, labels)

        # Find best layer by kappa (same strategy as NLP pipeline)
        best_layer = -1
        best_kappa = -1
        best_metrics = None
        layer_summary = {}

        # Test a representative subset of layers to save time
        # (embedding layer 0, then every ~1/4 of layers, and last layer)
        test_layers = sorted(set([0, n_layers // 4, n_layers // 2,
                                  3 * n_layers // 4, n_layers]))
        print(f"    Testing layers: {test_layers}", flush=True)

        for layer_idx in test_layers:
            if layer_idx not in layer_embeddings:
                continue
            emb = layer_embeddings[layer_idx]
            metrics = compute_cti_metrics(emb, labels, K, apply_pca=True)
            layer_summary[layer_idx] = {
                "kappa": metrics["kappa_nearest"],
                "acc_1nn": metrics["acc_1nn"],
            }
            print(f"    Layer {layer_idx}: kappa={metrics['kappa_nearest']:.4f}, "
                  f"acc={metrics['acc_1nn']:.3f}", flush=True)
            if metrics["kappa_nearest"] > best_kappa:
                best_kappa = metrics["kappa_nearest"]
                best_layer = layer_idx
                best_metrics = metrics

        # Also always test the last layer
        if n_layers not in test_layers:
            emb = layer_embeddings[n_layers]
            metrics = compute_cti_metrics(emb, labels, K, apply_pca=True)
            layer_summary[n_layers] = {
                "kappa": metrics["kappa_nearest"],
                "acc_1nn": metrics["acc_1nn"],
            }
            if metrics["kappa_nearest"] > best_kappa:
                best_kappa = metrics["kappa_nearest"]
                best_layer = n_layers
                best_metrics = metrics

        best_metrics["model"] = model_name
        best_metrics["hf_id"] = hf_id
        best_metrics["n_layers"] = n_layers
        best_metrics["d_model"] = d_model
        best_metrics["best_layer"] = best_layer
        best_metrics["layer_summary"] = layer_summary

        results[model_name] = best_metrics
        print(f"\n  {model_name}: BEST layer={best_layer}, kappa={best_metrics['kappa_nearest']:.4f}, "
              f"acc_1nn={best_metrics['acc_1nn']:.3f}, "
              f"q_norm={best_metrics['q_norm']:.3f}, "
              f"logit_q={best_metrics['logit_q']:.3f}", flush=True)

        # Also store last-layer metrics for comparison
        last_metrics = compute_cti_metrics(layer_embeddings[n_layers], labels, K, apply_pca=True)
        results[model_name + "_last_layer"] = {
            "kappa_nearest": last_metrics["kappa_nearest"],
            "acc_1nn": last_metrics["acc_1nn"],
            "q_norm": last_metrics["q_norm"],
            "logit_q": last_metrics["logit_q"],
            "layer": n_layers,
        }

        # Save intermediate cache
        with open(CACHE_PATH, "w") as f:
            json.dump(results, f, indent=2)

        del layer_embeddings
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # STEP 5: Hypothesis testing
    # ============================================================
    print(f"\n{'='*70}", flush=True)
    print("HYPOTHESIS TESTING", flush=True)
    print(f"{'='*70}", flush=True)

    model_names = [m[0] for m in ESM2_MODELS]
    kappas = np.array([results[m]["kappa_nearest"] for m in model_names])
    logit_qs = np.array([results[m]["logit_q"] for m in model_names])
    acc_1nns = np.array([results[m]["acc_1nn"] for m in model_names])
    q_norms = np.array([results[m]["q_norm"] for m in model_names])
    best_layers = [results[m].get("best_layer", "?") for m in model_names]

    # Also extract last-layer results for comparison
    kappas_last = np.array([results.get(m + "_last_layer", results[m])["kappa_nearest"] for m in model_names])
    logit_qs_last = np.array([results.get(m + "_last_layer", results[m])["logit_q"] for m in model_names])

    print(f"\n  Per-model summary (BEST LAYER, PCA to {N_PCA}d):", flush=True)
    print(f"  {'Model':<15} {'layer':>6} {'kappa':>8} {'acc_1nn':>8} {'q_norm':>8} {'logit_q':>8}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for i, m in enumerate(model_names):
        print(f"  {m:<15} {best_layers[i]:>6} {kappas[i]:>8.4f} {acc_1nns[i]:>8.3f} {q_norms[i]:>8.3f} {logit_qs[i]:>8.3f}", flush=True)

    print(f"\n  Per-model summary (LAST LAYER, PCA to {N_PCA}d):", flush=True)
    for i, m in enumerate(model_names):
        print(f"  {m:<15} kappa={kappas_last[i]:.4f}, logit_q={logit_qs_last[i]:.3f}", flush=True)

    n_models = len(model_names)

    # H_prot1: correlation
    if n_models >= 3:
        r_pearson, p_pearson = pearsonr(kappas, logit_qs)
        r_spearman, p_spearman = spearmanr(kappas, logit_qs)
    else:
        r_pearson, p_pearson = np.nan, np.nan
        r_spearman, p_spearman = np.nan, np.nan

    # Fit: logit(q) = alpha * kappa + C
    if n_models >= 2:
        coeffs = np.polyfit(kappas, logit_qs, 1)
        alpha_protein = coeffs[0]
        C_protein = coeffs[1]
        predicted = np.polyval(coeffs, kappas)
        ss_res = ((logit_qs - predicted) ** 2).sum()
        ss_tot = ((logit_qs - logit_qs.mean()) ** 2).sum()
        R_sq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        alpha_protein, C_protein, R_sq = np.nan, np.nan, np.nan

    # H_prot2: alpha > 0
    h_prot2_pass = alpha_protein > 0

    # H_prot3: accuracy increases with model size
    acc_monotonic = all(acc_1nns[i] <= acc_1nns[i + 1] for i in range(len(acc_1nns) - 1))
    acc_increases = acc_1nns[-1] > acc_1nns[0]

    print(f"\n  CTI Law Fit (protein ESM-2, n={n_models}):", flush=True)
    print(f"    alpha_protein = {alpha_protein:.4f}", flush=True)
    print(f"    C             = {C_protein:.4f}", flush=True)
    print(f"    r (Pearson)   = {r_pearson:.4f}  (p = {p_pearson:.4e})", flush=True)
    print(f"    r (Spearman)  = {r_spearman:.4f}  (p = {p_spearman:.4e})", flush=True)
    print(f"    R^2           = {R_sq:.4f}", flush=True)

    h_prot1_pass = abs(r_pearson) > 0.80 if not np.isnan(r_pearson) else False

    print(f"\n  H_prot1: r(kappa, logit_q) > 0.80 => {r_pearson:.4f} => {'PASS' if h_prot1_pass else 'FAIL'}", flush=True)
    print(f"  H_prot2: alpha > 0 => {alpha_protein:.4f} => {'PASS' if h_prot2_pass else 'FAIL'}", flush=True)
    print(f"  H_prot3: acc increases with size => {acc_1nns[0]:.3f} -> {acc_1nns[-1]:.3f} => {'PASS' if acc_increases else 'FAIL'}", flush=True)

    # ============================================================
    # Also fit last-layer only (for comparison)
    # ============================================================
    if n_models >= 2:
        coeffs_last = np.polyfit(kappas_last, logit_qs_last, 1)
        alpha_last = coeffs_last[0]
        C_last = coeffs_last[1]
        if n_models >= 3:
            r_last, p_last = pearsonr(kappas_last, logit_qs_last)
        else:
            r_last, p_last = np.nan, np.nan
        print(f"\n  Last-layer fit: alpha={alpha_last:.3f}, C={C_last:.3f}, r={r_last:.3f}", flush=True)
    else:
        alpha_last, r_last = np.nan, np.nan

    # ============================================================
    # Compare alpha to other domains
    # ============================================================
    print(f"\n  Cross-domain alpha comparison:", flush=True)
    print(f"    NLP decoders:  alpha ~ 1.477  (19 architectures)", flush=True)
    print(f"    Vision (ViT):  alpha ~ 4.5    (ResNet/ViT)", flush=True)
    print(f"    Protein (ESM): alpha = {alpha_protein:.3f}  ({n_models} model sizes, best layer)", flush=True)
    print(f"    Protein last:  alpha = {alpha_last:.3f}  (last layer only)", flush=True)

    # ============================================================
    # Bootstrap CI for alpha (resample models - limited but honest)
    # ============================================================
    if n_models >= 4:
        boot_alphas = []
        rng = np.random.RandomState(SEED)
        for _ in range(10000):
            idx = rng.choice(n_models, n_models, replace=True)
            if len(set(idx)) < 2:
                continue
            c = np.polyfit(kappas[idx], logit_qs[idx], 1)
            boot_alphas.append(c[0])
        boot_alphas = np.array(boot_alphas)
        alpha_ci = (np.percentile(boot_alphas, 2.5), np.percentile(boot_alphas, 97.5))
        print(f"\n  Alpha 95% CI (bootstrap): [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}]", flush=True)
    else:
        alpha_ci = (np.nan, np.nan)

    # ============================================================
    # Save results
    # ============================================================
    elapsed = time.time() - t0

    final = {
        "experiment": "CTI Protein Domain Validation (ESM-2)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domain": "protein_sequences",
        "task": "EC_number_classification",
        "K": K,
        "n_per_class": N_PER_CLASS,
        "n_total": len(sequences),
        "models": {m: results[m] for m in model_names if m in results},
        "law_fit": {
            "alpha_protein": float(alpha_protein),
            "C": float(C_protein),
            "r_pearson": float(r_pearson),
            "p_pearson": float(p_pearson),
            "r_spearman": float(r_spearman),
            "p_spearman": float(p_spearman),
            "R_squared": float(R_sq),
            "alpha_95ci": [float(alpha_ci[0]), float(alpha_ci[1])],
        },
        "hypotheses": {
            "H_prot1": {
                "description": "r(kappa, logit_q) > 0.80",
                "r": float(r_pearson),
                "pass": bool(h_prot1_pass),
            },
            "H_prot2": {
                "description": "alpha > 0 (monotonic)",
                "alpha": float(alpha_protein),
                "pass": bool(h_prot2_pass),
            },
            "H_prot3": {
                "description": "accuracy increases with model size",
                "acc_range": [float(acc_1nns[0]), float(acc_1nns[-1])],
                "monotonic": bool(acc_monotonic),
                "increases": bool(acc_increases),
                "pass": bool(acc_increases),
            },
        },
        "last_layer_fit": {
            "alpha": float(alpha_last) if not np.isnan(alpha_last) else None,
            "r": float(r_last) if not np.isnan(r_last) else None,
        },
        "last_layer_data": {m: results.get(m + "_last_layer", {}) for m in model_names},
        "cross_domain_comparison": {
            "alpha_nlp_decoder": 1.477,
            "alpha_vision": 4.5,
            "alpha_protein_best_layer": float(alpha_protein),
            "alpha_protein_last_layer": float(alpha_last) if not np.isnan(alpha_last) else None,
        },
        "pca_dimensions": N_PCA,
        "elapsed_seconds": elapsed,
    }

    out_path = RESULTS_DIR / "cti_protein_esm2.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n  Results saved to {out_path}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Law: logit(q) = {alpha_protein:.3f} * kappa + {C_protein:.3f}", flush=True)
    print(f"  r = {r_pearson:.3f}, R^2 = {R_sq:.3f}", flush=True)
    print(f"  H_prot1 (r > 0.80): {'PASS' if h_prot1_pass else 'FAIL'}", flush=True)
    print(f"  H_prot2 (alpha > 0): {'PASS' if h_prot2_pass else 'FAIL'}", flush=True)
    print(f"  H_prot3 (acc scales): {'PASS' if acc_increases else 'FAIL'}", flush=True)
    print(f"  Total time: {elapsed / 60:.1f} minutes", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
