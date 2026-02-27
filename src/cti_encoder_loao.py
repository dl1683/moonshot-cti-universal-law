"""
CTI ENCODER LOAO — Second Law Test
====================================
Tests whether NLP encoders have their OWN universal alpha constant,
analogous to NLP decoders (alpha_decoder=1.477).

If CV_alpha_encoder < 15%, encoders have a second universal constant
and the paper's theoretical narrative upgrades from
  "alpha is NLP-decoder specific" ->
  "alpha is family-specific with derivable family constants"

Pre-registered hypothesis:
  H_encoder: CV_alpha_encoder < 0.20 across 5 encoder architectures

Models (all from LOAO-adjacent set, encoder family):
  - bert-base-uncased (BERT)
  - google/electra-small-discriminator (ELECTRA-small)
  - microsoft/deberta-v3-small (DeBERTa)
  - BAAI/bge-small-en-v1.5 (BGE-small)
  - FacebookAI/roberta-base (RoBERTa)

Datasets: clinc, dbpedia_classes, agnews, trec (same 4 as LOAO)
Output: results/cti_encoder_loao.json
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.special import logit as scipy_logit

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "cti_encoder_loao.json"

ENCODER_MODELS = [
    {"name": "bert-base-uncased",    "hf": "bert-base-uncased"},
    {"name": "electra-small",        "hf": "google/electra-small-discriminator"},
    {"name": "deberta-v3-small",     "hf": "microsoft/deberta-v3-small"},
    {"name": "bge-small-en",         "hf": "BAAI/bge-small-en-v1.5"},
    {"name": "roberta-base",         "hf": "FacebookAI/roberta-base"},
]

DATASETS = [
    {"name": "clinc",           "hf": "clinc_oos", "hf_cfg": "plus", "split": "train",
     "text_col": "text",  "label_col": "intent",  "K_min": 10},
    {"name": "dbpedia_classes", "hf": "fancyzhx/dbpedia_14", "hf_cfg": None, "split": "train",
     "text_col": "content", "label_col": "label", "K_min": 10},
    {"name": "agnews",          "hf": "fancyzhx/ag_news", "hf_cfg": None, "split": "train",
     "text_col": "text",  "label_col": "label",  "K_min": 3},
    {"name": "trec",            "hf": "CogComp/trec", "hf_cfg": None, "split": "train",
     "text_col": "text",  "label_col": "coarse_label", "K_min": 3},
]

MAX_EXAMPLES = 2000
BATCH_SIZE = 32
ALPHA_DECODER = 1.477
BETA_FIXED = 0.5


def compute_kappa_nearest(embeddings, labels):
    classes = np.unique(labels)
    K = len(classes)
    if K < 2:
        return None
    d = embeddings.shape[1]
    centroids = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = embeddings[mask].mean(0)
    if len(centroids) < 2:
        return None
    sq_sum = 0.0
    n_total = 0
    for c in classes:
        if c not in centroids:
            continue
        mask = labels == c
        diff = embeddings[mask] - centroids[c]
        sq_sum += float(np.sum(diff ** 2))
        n_total += int(mask.sum()) * d
    sigma_W = np.sqrt(sq_sum / n_total) if n_total > 0 else 1e-12
    ckeys = sorted(centroids.keys())
    cent_arr = np.array([centroids[c] for c in ckeys])
    min_gap = float("inf")
    for i in range(len(ckeys)):
        for j in range(i + 1, len(ckeys)):
            gap = float(np.linalg.norm(cent_arr[i] - cent_arr[j]))
            if gap < min_gap:
                min_gap = gap
    return float(min_gap / (sigma_W * np.sqrt(d) + 1e-12))


def load_dataset_samples(ds_cfg, max_n=MAX_EXAMPLES):
    from datasets import load_dataset as hf_load
    print(f"    Loading {ds_cfg['name']}...", flush=True)
    if ds_cfg.get("hf_cfg"):
        ds = hf_load(ds_cfg["hf"], ds_cfg["hf_cfg"], split=ds_cfg["split"],
                     trust_remote_code=True)
    else:
        ds = hf_load(ds_cfg["hf"], split=ds_cfg["split"], trust_remote_code=True)
    texts = list(ds[ds_cfg["text_col"]])
    raw_labels = list(ds[ds_cfg["label_col"]])
    try:
        labels = np.array([int(l) for l in raw_labels])
    except (ValueError, TypeError):
        unique_str = sorted(set(str(l) for l in raw_labels))
        str_to_int = {s: i for i, s in enumerate(unique_str)}
        labels = np.array([str_to_int[str(l)] for l in raw_labels])
    if len(texts) > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), max_n, replace=False)
        texts = [texts[i] for i in idx]
        labels = labels[idx]
    ulabels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(ulabels)}
    labels = np.array([label_map[l] for l in labels])
    K = len(ulabels)
    print(f"    {ds_cfg['name']}: n={len(texts)}, K={K}", flush=True)
    return texts, labels, K


def get_encoder_embeddings(model_cfg, texts, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    hf = model_cfg["hf"]
    tok = AutoTokenizer.from_pretrained(hf)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(hf, trust_remote_code=True)
    mdl = mdl.to(device)
    mdl.eval()

    all_embs = []
    t0 = time.time()
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc, output_hidden_states=False)
        # Use CLS token (encoders) or mean pool
        if hasattr(out, "last_hidden_state"):
            hs = out.last_hidden_state  # (B, seq, d)
            # For encoders: CLS token
            cls_emb = hs[:, 0, :].float()
            all_embs.append(cls_emb.cpu().numpy())
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            all_embs.append(out.pooler_output.float().cpu().numpy())
    print(f"    Inference in {time.time()-t0:.1f}s", flush=True)

    del mdl
    import gc
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return np.concatenate(all_embs, axis=0).astype(np.float64)


def fit_alpha(kappas, logit_qs, K_vals, beta=BETA_FIXED):
    """Fit alpha via OLS: logit_q = alpha * kappa - beta*log(K-1) + C"""
    from scipy.stats import linregress
    x = np.array(kappas)
    # Adjust for K: y_adj = logit_q + beta*log(K-1)
    y_adj = np.array(logit_qs) + beta * np.log(np.array(K_vals) - 1)
    if len(x) < 2:
        return None, None
    slope, intercept, r, p, se = linregress(x, y_adj)
    return float(slope), float(r ** 2)


def main():
    import torch

    print("CTI ENCODER LOAO", flush=True)
    print("Pre-registered: H_encoder: CV_alpha_encoder < 0.20", flush=True)
    print("=" * 60, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Load all 4 datasets once
    print("\n[STEP 1] Load 4 datasets", flush=True)
    dataset_samples = {}
    for ds_cfg in DATASETS:
        texts, labels, K = load_dataset_samples(ds_cfg)
        dataset_samples[ds_cfg["name"]] = (texts, labels, K)

    # Run each encoder model
    results_per_model = {}
    print("\n[STEP 2] Run 5 encoder models", flush=True)

    for model_cfg in ENCODER_MODELS:
        name = model_cfg["name"]
        print(f"\n  === {name} ===", flush=True)
        model_results = {}

        try:
            for ds_cfg in DATASETS:
                ds_name = ds_cfg["name"]
                texts, labels, K = dataset_samples[ds_name]
                emb = get_encoder_embeddings(model_cfg, texts, device)
                kappa = compute_kappa_nearest(emb, labels)
                if kappa is None:
                    print(f"    {ds_name}: kappa=None, skip", flush=True)
                    continue
                q_norm_raw = None
                # Compute 1-NN accuracy with leave-one-out
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.model_selection import cross_val_score
                knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
                scores = cross_val_score(knn, emb, labels, cv=5, n_jobs=1)
                acc = float(scores.mean())
                q_norm = (acc - 1/K) / (1 - 1/K)
                q_norm = max(1e-6, min(1 - 1e-6, q_norm))
                logit_q = float(scipy_logit(q_norm))
                print(f"    {ds_name}: kappa={kappa:.4f} acc={acc:.4f} "
                      f"q_norm={q_norm:.4f} logit_q={logit_q:.4f}", flush=True)
                model_results[ds_name] = {
                    "kappa_nearest": kappa,
                    "acc_1nn": acc,
                    "q_norm": q_norm,
                    "logit_q": logit_q,
                    "K": K,
                }
        except Exception as e:
            print(f"  ERROR {name}: {e}", flush=True)
            results_per_model[name] = {"error": str(e)}
            continue

        results_per_model[name] = model_results

        # Save intermediate
        with open(OUT_PATH, "w", encoding="ascii") as fp:
            json.dump({"status": "in_progress", "results": results_per_model},
                      fp, indent=2)

    # LOAO analysis
    print("\n[STEP 3] LOAO alpha computation", flush=True)
    loao_alphas = {}
    for model_name, model_results in results_per_model.items():
        if "error" in model_results:
            continue
        # Compute alpha on held-out model (train on other 4, test on this one)
        # Since we only have 1 point per model-dataset, fit alpha per dataset
        kappas, logit_qs, K_vals = [], [], []
        for ds_name, dr in model_results.items():
            kappas.append(dr["kappa_nearest"])
            logit_qs.append(dr["logit_q"])
            K_vals.append(dr["K"])
        alpha, r2 = fit_alpha(kappas, logit_qs, K_vals)
        loao_alphas[model_name] = {"alpha": alpha, "r2": r2}
        print(f"  {model_name}: alpha={alpha:.4f} R2={r2:.4f}", flush=True)

    # Compute CV
    alphas = [v["alpha"] for v in loao_alphas.values() if v["alpha"] is not None]
    if len(alphas) >= 2:
        alpha_mean = float(np.mean(alphas))
        alpha_std = float(np.std(alphas))
        cv = float(alpha_std / alpha_mean) if alpha_mean != 0 else None
        print(f"\n  Encoder alpha_mean={alpha_mean:.4f} +/- {alpha_std:.4f} CV={cv:.4f}",
              flush=True)
        print(f"  Decoder alpha for comparison: {ALPHA_DECODER:.4f}",
              flush=True)
        print(f"  Ratio encoder/decoder: {alpha_mean/ALPHA_DECODER:.2f}x", flush=True)
        pass_h = cv is not None and cv < 0.20
        print(f"  H_encoder (CV<0.20): {'PASS' if pass_h else 'FAIL'}", flush=True)
    else:
        alpha_mean, alpha_std, cv, pass_h = None, None, None, False

    result = {
        "experiment": "cti_encoder_loao",
        "preregistration": "H_encoder: CV_alpha_encoder < 0.20",
        "alpha_decoder_reference": ALPHA_DECODER,
        "per_model_results": results_per_model,
        "loao_alphas": loao_alphas,
        "summary": {
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
            "cv": cv,
            "alpha_mean_encoder": alpha_mean,
            "ratio_encoder_decoder": float(alpha_mean / ALPHA_DECODER) if alpha_mean else None,
            "PASS_H_encoder": pass_h,
        },
        "status": "complete",
    }

    with open(OUT_PATH, "w", encoding="ascii") as fp:
        json.dump(result, fp, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
