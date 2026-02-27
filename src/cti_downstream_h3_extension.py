"""
CTI DOWNSTREAM H3 EXTENSION
============================
Extends H3 cross-model ranking from n=5 to n=9 by adding 4 additional
models from the LOAO 12-architecture set to the banking77 test.

Pre-existing V3 results (5 models) are loaded from cti_downstream_protocol_v3.json.
4 new models are run at FINAL LAYER ONLY on banking77 (K=77).
Combined H3 Spearman rho is computed across n=9 models.

Pre-registered extension hypothesis:
  H3_extended: Spearman rho(kappa_nearest_final, MAP@10_final) > 0.50
               across 9 models on banking77 (K=77), p < 0.05 (two-sided)

New models added (from LOAO 12-arch set):
  - OLMo-1B (allenai/OLMo-1B-hf)
  - TinyLlama-1.1B (TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
  - Qwen3-0.6B (Qwen/Qwen3-0.6B)
  - Qwen3-1.7B (Qwen/Qwen3-1.7B)

Output: results/cti_downstream_h3_n9.json
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
V3_PATH = RESULTS_DIR / "cti_downstream_protocol_v3.json"
OUT_PATH = RESULTS_DIR / "cti_downstream_h3_n9.json"

BANKING77_CFG = {
    "name": "banking77",
    "hf": "PolyAI/banking77",
    "split": "train",
    "text_col": "text",
    "label_col": "label",
    "K_expected": 77,
}

NEW_MODELS = [
    {"name": "OLMo-1B",       "hf": "allenai/OLMo-1B-hf"},
    {"name": "TinyLlama-1.1B", "hf": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"},
    {"name": "Qwen3-0.6B",    "hf": "Qwen/Qwen3-0.6B"},
    {"name": "Qwen3-1.7B",    "hf": "Qwen/Qwen3-1.7B"},
]

MAX_EXAMPLES = 5000
BATCH_SIZE = 64
MAP_K = 10


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
        n_c = int(mask.sum())
        diff = embeddings[mask] - centroids[c]
        sq_sum += float(np.sum(diff ** 2))
        n_total += n_c * d
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


def compute_map_at_k(embeddings, labels, k=10):
    n = len(embeddings)
    if n < k + 1:
        return None, None
    emb_normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    ap_sum = 0.0
    ceiling_count = 0
    chunk = 256
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = emb_normed[start:end] @ emb_normed.T
        for li in range(end - start):
            sims[li, start + li] = -1e9
        top_k_idx = np.argsort(sims, axis=1)[:, -k:][:, ::-1]
        for li in range(end - start):
            qi = start + li
            retrieved_labels = labels[top_k_idx[li]]
            relevant = (retrieved_labels == labels[qi]).astype(float)
            n_rel = relevant.sum()
            if n_rel == 0:
                continue
            cumrel = np.cumsum(relevant)
            prec_at_j = cumrel / (np.arange(k) + 1.0)
            ap = float((prec_at_j * relevant).sum() / min(n_rel, k))
            ap_sum += ap
            if n_rel == k:
                ceiling_count += 1
    return float(ap_sum / n), float(ceiling_count / n)


def load_banking77(max_n=MAX_EXAMPLES):
    from datasets import load_dataset as hf_load
    print("  Loading dataset banking77...", flush=True)
    ds = hf_load(BANKING77_CFG["hf"], split=BANKING77_CFG["split"], trust_remote_code=True)
    texts = list(ds[BANKING77_CFG["text_col"]])
    raw_labels = list(ds[BANKING77_CFG["label_col"]])
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
    print(f"  banking77: n={len(texts)}, K={len(ulabels)}", flush=True)
    return texts, labels


def run_final_layer(model_cfg, texts, labels, device):
    """Run only final-layer inference for H3."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = model_cfg["name"]
    hf = model_cfg["hf"]
    print(f"\n  {name} x banking77 (final layer only)", flush=True)
    print(f"  Loading model {hf}...", flush=True)

    tok = AutoTokenizer.from_pretrained(hf)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(hf, trust_remote_code=True)
    mdl = mdl.to(device)
    mdl.eval()

    n_layers = mdl.config.num_hidden_layers
    final_idx = n_layers  # final layer hidden state index

    # Batch inference
    all_embs = []
    t0 = time.time()
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc, output_hidden_states=True)
        hs = out.hidden_states[final_idx].float()  # (B, seq, d)
        attn_mask = enc["attention_mask"].float().unsqueeze(-1)
        pooled = (hs * attn_mask).sum(1) / (attn_mask.sum(1) + 1e-12)
        all_embs.append(pooled.cpu().numpy())
    print(f"  Inference done in {time.time()-t0:.1f}s", flush=True)

    del mdl
    import gc
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    emb = np.concatenate(all_embs, axis=0).astype(np.float64)
    kappa = compute_kappa_nearest(emb, labels)
    map_k, ceil_frac = compute_map_at_k(emb, labels, k=MAP_K)
    print(f"    Final layer {final_idx}/{n_layers}: kappa={kappa:.4f}  MAP@10={map_k:.4f}",
          flush=True)

    return {
        "model": name,
        "kappa_nearest_final": kappa,
        "map_at_10_final": map_k,
        "ceiling_frac": ceil_frac,
        "layer_idx": final_idx,
        "n_layers_total": n_layers,
    }


def main():
    import torch

    print("CTI DOWNSTREAM H3 EXTENSION", flush=True)
    print("Target: H3 cross-model ranking n=9 (p<0.05 at rho=0.70)", flush=True)
    print("=" * 60, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # ── Load V3 existing H3 data (5 models) ──────────────────────────────────
    print("\n[STEP 1] Load V3 H3 data (5 models)", flush=True)
    with open(V3_PATH, encoding="ascii") as f:
        v3 = json.load(f)

    existing_models = v3["hypothesis_results"]["H3"]["models"]
    print(f"  Loaded {len(existing_models)} existing models from V3:", flush=True)
    for m in existing_models:
        print(f"    {m['model']}: kappa={m['kappa_nearest_final']:.4f}  "
              f"MAP@10={m['map_at_10_final']:.4f}", flush=True)

    # ── Load banking77 once ───────────────────────────────────────────────────
    print("\n[STEP 2] Load banking77", flush=True)
    texts, labels = load_banking77()

    # ── Run new models ────────────────────────────────────────────────────────
    print("\n[STEP 3] Run 4 new models (final layer only)", flush=True)
    new_results = []
    for model_cfg in NEW_MODELS:
        try:
            res = run_final_layer(model_cfg, texts, labels, device)
            new_results.append(res)
            # Save intermediate
            with open(OUT_PATH, "w", encoding="ascii") as fp:
                json.dump({"status": "in_progress",
                           "existing": existing_models,
                           "new": new_results}, fp, indent=2)
        except Exception as e:
            print(f"  ERROR {model_cfg['name']}: {e}", flush=True)
            new_results.append({"model": model_cfg["name"], "error": str(e)})

    # ── H3 combined analysis ──────────────────────────────────────────────────
    print("\n[STEP 4] H3 combined n=9 analysis", flush=True)
    all_h3 = []
    for m in existing_models:
        if "kappa_nearest_final" in m and "map_at_10_final" in m:
            all_h3.append(m)
    for m in new_results:
        if "kappa_nearest_final" in m and "map_at_10_final" in m:
            all_h3.append(m)

    n_models = len(all_h3)
    kappas = [m["kappa_nearest_final"] for m in all_h3]
    maps = [m["map_at_10_final"] for m in all_h3]
    rho, pval = spearmanr(kappas, maps)
    print(f"  n_models={n_models}", flush=True)
    print(f"  H3 Spearman rho(kappa, MAP@10) = {rho:.4f}  p={pval:.4f}", flush=True)
    print(f"  {'PASS (significant p<0.05)' if pval < 0.05 else 'INDICATIVE (p>=0.05)'}", flush=True)

    for m in all_h3:
        print(f"    {m['model']}: kappa={m['kappa_nearest_final']:.4f}  "
              f"MAP@10={m['map_at_10_final']:.4f}", flush=True)

    result = {
        "experiment": "cti_downstream_h3_extension",
        "description": "H3 cross-model ranking extended to n=9 models on banking77 K=77",
        "preregistration": "H3_extended: rho>0.50 AND p<0.05 (two-sided)",
        "H3_extended": {
            "n_models": n_models,
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "PASS_threshold": bool(float(rho) > 0.50),
            "PASS_significance": bool(float(pval) < 0.05),
            "models": all_h3,
        },
        "v3_models": existing_models,
        "new_models": new_results,
        "status": "complete",
    }

    with open(OUT_PATH, "w", encoding="ascii") as fp:
        json.dump(result, fp, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
