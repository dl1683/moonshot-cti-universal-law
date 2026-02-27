"""
CTI DOWNSTREAM DECISION PROTOCOL V2
=====================================

PRE-REGISTERED HYPOTHESES (committed before running):
  H1_existing: Within-model Spearman rho(kappa_nearest, q_1NN) > 0.50
               mean across existing 4-dataset NLP data (n=48 pairs, 4 layers each)
               Threshold: mean rho > 0.50, >=60% positive pairs
  H1_new: Same metric on NEW hard datasets (banking77 K=77, amazon_massive K=59)
               Threshold: mean rho > 0.50, >=60% positive pairs
  H2: Within-model Spearman rho(kappa_nearest, MAP@10) > 0.50
               on new hard datasets, averaged across (model, dataset) pairs
               Threshold: mean rho > 0.50
  H3: Cross-model Spearman rho(kappa_nearest_final, MAP@10_final) > 0.50
               across models using final-layer representations on banking77

Scientific framing (Codex design gate):
  kappa_nearest ranks LAYERS within a model by any geometry-sensitive metric.
  Hard datasets (K=77, K=59) ensure no ceiling effects.
  MAP@10 proves the result generalizes beyond 1-NN classification.
  Failure = honest variance diagnostics documenting ceiling/floor effects.

Output: results/cti_downstream_protocol_v2.json
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DATA_FILE = RESULTS_DIR / "cti_kappa_nearest_universal.json"
OUT_PATH = RESULTS_DIR / "cti_downstream_protocol_v2.json"

MODELS = [
    {"name": "pythia-160m", "hf": "EleutherAI/pythia-160m"},
    {"name": "pythia-410m", "hf": "EleutherAI/pythia-410m"},
    {"name": "Qwen2.5-0.5B", "hf": "Qwen/Qwen2.5-0.5B"},
]

DATASETS_CFG = [
    {
        "name": "banking77",
        "hf": "PolyAI/banking77",
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "K_expected": 77,
        "filter_lang": None,
    },
    {
        "name": "amazon_massive_en",
        "hf": "mteb/amazon_massive_intent",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "K_expected": 59,
        "filter_lang": "en",
    },
]

N_LAYER_SAMPLES = 6   # layer checkpoints per model
MAX_EXAMPLES = 5000   # subsample cap per dataset
BATCH_SIZE = 64
MAP_K = 10

# Pre-registered thresholds
THRESH_MEAN_RHO = 0.50
THRESH_FRAC_POS = 0.60


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_kappa_nearest(embeddings, labels):
    """
    kappa_nearest = min_{j!=k} ||mu_j - mu_k|| / (sigma_W * sqrt(d))
    Same formula as the paper.
    """
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

    # Within-class std (pooled)
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

    # Min centroid gap
    ckeys = sorted(centroids.keys())
    cent_arr = np.array([centroids[c] for c in ckeys])
    min_gap = float("inf")
    for i in range(len(ckeys)):
        for j in range(i + 1, len(ckeys)):
            gap = float(np.linalg.norm(cent_arr[i] - cent_arr[j]))
            if gap < min_gap:
                min_gap = gap

    kappa = min_gap / (sigma_W * np.sqrt(d) + 1e-12)
    return float(kappa)


def compute_q_1nn(embeddings, labels):
    """1-NN leave-one-out accuracy (normalized to [0,1])."""
    n = len(embeddings)
    K = len(np.unique(labels))
    if n < 2 or K < 2:
        return None

    # Cosine similarity (batch to stay memory-safe)
    emb_normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    correct = 0
    chunk = 512
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = emb_normed[start:end] @ emb_normed.T  # (chunk, n)
        # Exclude self
        for li in range(end - start):
            sims[li, start + li] = -1e9
        nn_idx = np.argmax(sims, axis=1)
        correct += int(np.sum(labels[nn_idx] == labels[start:end]))

    acc = correct / n
    q_norm = (acc - 1.0 / K) / (1.0 - 1.0 / K)
    return float(q_norm)


def compute_map_at_k(embeddings, labels, k=10):
    """
    MAP@k using cosine similarity leave-one-out retrieval.
    Returns (map_k, ceiling_fraction) where ceiling_fraction =
    fraction of queries where all top-k are same-class.
    """
    n = len(embeddings)
    if n < k + 1:
        return None, None

    emb_normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

    ap_sum = 0.0
    ceiling_count = 0
    chunk = 256
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = emb_normed[start:end] @ emb_normed.T  # (chunk, n)
        for li in range(end - start):
            sims[li, start + li] = -1e9  # exclude self
        # Top-k indices
        top_k_idx = np.argsort(sims, axis=1)[:, -k:][:, ::-1]  # (chunk, k) descending

        for li in range(end - start):
            qi = start + li
            retrieved_labels = labels[top_k_idx[li]]
            relevant = (retrieved_labels == labels[qi]).astype(float)
            n_rel = relevant.sum()
            if n_rel == 0:
                continue
            # AP = sum_j P@j * rel(j) / min(n_total_relevant, k)
            cumrel = np.cumsum(relevant)
            prec_at_j = cumrel / (np.arange(k) + 1.0)
            ap = float((prec_at_j * relevant).sum() / min(n_rel, k))
            ap_sum += ap
            if n_rel == k:
                ceiling_count += 1

    map_k = ap_sum / n
    ceiling_frac = ceiling_count / n
    return float(map_k), float(ceiling_frac)


def variance_diagnostics(values, name):
    """Log variance stats to detect ceiling/floor effects."""
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "cv": float(arr.std() / (arr.mean() + 1e-12)),
        "name": name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# H1_EXISTING: Use cached 4-layer data
# ─────────────────────────────────────────────────────────────────────────────

def h1_from_existing(data_file):
    """Re-run H1 on existing 48 (model, dataset) pairs with 4 layers each."""
    print("H1_existing: within-model rho from existing data...", flush=True)
    with open(data_file) as f:
        d = json.load(f)
    pts = d["all_points"]

    layer_map = defaultdict(list)
    for p in pts:
        key = (p["model"], p["dataset"])
        layer_map[key].append((p["layer"], p["kappa_nearest"], p["q"]))

    rhos = []
    details = []
    for (model, ds), entries in layer_map.items():
        if len(entries) < 3:
            continue
        entries.sort(key=lambda x: x[0])
        kappas = [e[1] for e in entries]
        qs = [e[2] for e in entries]
        if np.std(kappas) < 1e-10 or np.std(qs) < 1e-10:
            continue
        rho, pval = spearmanr(kappas, qs)
        rhos.append(float(rho))
        details.append({"model": model, "dataset": ds, "n_layers": len(entries),
                        "spearman_rho": float(rho), "p_value": float(pval)})

    mean_rho = float(np.mean(rhos)) if rhos else None
    frac_pos = float(np.mean([r > 0 for r in rhos])) if rhos else None
    h1_pass = (mean_rho is not None and mean_rho > THRESH_MEAN_RHO
               and frac_pos is not None and frac_pos >= THRESH_FRAC_POS)

    print(f"  n_pairs={len(rhos)}, mean_rho={mean_rho:.4f}, "
          f"frac_positive={frac_pos:.2f} -> {'PASS' if h1_pass else 'FAIL'}", flush=True)

    return {
        "n_pairs": len(rhos),
        "mean_spearman_rho": mean_rho,
        "frac_positive": frac_pos,
        "threshold_mean_rho": THRESH_MEAN_RHO,
        "threshold_frac_pos": THRESH_FRAC_POS,
        "PASS": h1_pass,
        "pair_details": details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_examples(cfg, max_n=MAX_EXAMPLES):
    """Load text+labels from a HuggingFace dataset, optionally filtering by lang."""
    from datasets import load_dataset as hf_load
    print(f"  Loading dataset {cfg['name']}...", flush=True)
    ds = hf_load(cfg["hf"], split=cfg["split"], trust_remote_code=True)

    if cfg.get("filter_lang"):
        lang = cfg["filter_lang"]
        ds = ds.filter(lambda x: x.get("lang", "") == lang)

    texts = ds[cfg["text_col"]]
    raw_labels = ds[cfg["label_col"]]

    # Normalize labels to int (handle string category labels gracefully)
    try:
        labels = np.array([int(l) for l in raw_labels])
    except (ValueError, TypeError):
        unique_str = sorted(set(str(l) for l in raw_labels))
        str_to_int = {s: i for i, s in enumerate(unique_str)}
        labels = np.array([str_to_int[str(l)] for l in raw_labels])
    texts = list(texts)

    # Subsample
    if len(texts) > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), max_n, replace=False)
        texts = [texts[i] for i in idx]
        labels = labels[idx]

    # Re-index labels 0..K-1
    ulabels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(ulabels)}
    labels = np.array([label_map[l] for l in labels])

    K = len(ulabels)
    print(f"  {cfg['name']}: n={len(texts)}, K={K}", flush=True)
    return texts, labels


def get_layer_embeddings(model, tokenizer, texts, layer_indices, device, batch_size=BATCH_SIZE):
    """
    Extract mean-pooled embeddings from specified hidden state indices.
    layer_indices: list of ints (1-indexed transformer block outputs).
    Returns dict {layer_idx: np.ndarray shape (n, d)}
    """
    import torch
    model.eval()
    all_embeddings = {li: [] for li in layer_indices}

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=128
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple len n_layers+1

        attn_mask = enc["attention_mask"].float()  # (B, seq)
        mask_exp = attn_mask.unsqueeze(-1)          # (B, seq, 1)

        for li in layer_indices:
            hs = hidden_states[li].float()          # (B, seq, d)
            pooled = (hs * mask_exp).sum(1) / (mask_exp.sum(1) + 1e-12)  # (B, d)
            all_embeddings[li].append(pooled.cpu().numpy())

    return {li: np.concatenate(all_embeddings[li], axis=0) for li in layer_indices}


# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_model_dataset(model_cfg, ds_cfg, device):
    """Full per-layer analysis for one (model, dataset) pair."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = model_cfg["name"]
    ds_name = ds_cfg["name"]
    print(f"\n  {model_name} x {ds_name}", flush=True)

    # Load dataset
    texts, labels = load_dataset_examples(ds_cfg)
    n, K = len(texts), int(labels.max()) + 1

    # Load model
    print(f"  Loading model {model_cfg['hf']}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_cfg["hf"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(model_cfg["hf"], trust_remote_code=True)
    mdl = mdl.to(device)

    n_layers = mdl.config.num_hidden_layers
    # Sample layer indices (1-indexed, up to n_layers)
    fracs = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
    layer_indices = sorted(set(max(1, round(n_layers * f)) for f in fracs))
    print(f"  n_layers={n_layers}, sampling at {layer_indices}", flush=True)

    # Extract embeddings
    t0 = time.time()
    emb_by_layer = get_layer_embeddings(mdl, tok, texts, layer_indices, device)
    print(f"  Inference done in {time.time()-t0:.1f}s", flush=True)

    # Free GPU memory
    del mdl
    import gc
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # Per-layer metrics
    layer_results = []
    for li in layer_indices:
        emb = emb_by_layer[li].astype(np.float64)
        kappa = compute_kappa_nearest(emb, labels)
        q = compute_q_1nn(emb, labels)
        map_k, ceil_frac = compute_map_at_k(emb, labels, k=MAP_K)

        layer_results.append({
            "layer_idx": li,
            "layer_frac": round(li / n_layers, 3),
            "kappa_nearest": kappa,
            "q_1nn": q,
            "map_at_10": map_k,
            "ceiling_frac": ceil_frac,
        })
        print(f"    Layer {li:2d}/{n_layers}: kappa={kappa:.4f}  "
              f"q_1nn={q:.4f}  MAP@10={map_k:.4f}", flush=True)

    return {
        "model": model_name,
        "dataset": ds_name,
        "n_examples": n,
        "K": K,
        "n_layers_total": n_layers,
        "layer_results": layer_results,
    }


def compute_within_model_spearman(pair_result):
    """Compute Spearman rho(kappa, q_1nn) and rho(kappa, map@10) for one pair."""
    lr = pair_result["layer_results"]
    valid = [r for r in lr if r["kappa_nearest"] is not None
             and r["q_1nn"] is not None and r["map_at_10"] is not None]
    if len(valid) < 3:
        return None, None, None

    kappas = [r["kappa_nearest"] for r in valid]
    qs = [r["q_1nn"] for r in valid]
    maps = [r["map_at_10"] for r in valid]

    rho_q, pval_q = spearmanr(kappas, qs)
    rho_map, pval_map = spearmanr(kappas, maps)

    return (float(rho_q), float(pval_q)), (float(rho_map), float(pval_map)), valid


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import torch

    print("CTI DOWNSTREAM DECISION PROTOCOL V2", flush=True)
    print("Pre-registered thresholds: mean Spearman rho > 0.50, >=60% positive", flush=True)
    print("=" * 60, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # ── H1_existing ──────────────────────────────────────────────────────────
    print("\n[STEP 1] H1 from existing 4-dataset NLP data", flush=True)
    h1_existing = h1_from_existing(DATA_FILE)

    # ── New inference on hard datasets ───────────────────────────────────────
    print("\n[STEP 2] New inference on banking77 + amazon_massive", flush=True)
    all_pair_results = []
    for model_cfg in MODELS:
        for ds_cfg in DATASETS_CFG:
            try:
                result = analyze_model_dataset(model_cfg, ds_cfg, device)
                all_pair_results.append(result)
                # Save intermediate
                with open(OUT_PATH, "w", encoding="ascii") as fp:
                    json.dump({"status": "in_progress",
                               "h1_existing": h1_existing,
                               "pairs": all_pair_results}, fp, indent=2)
            except Exception as e:
                print(f"  ERROR {model_cfg['name']} x {ds_cfg['name']}: {e}", flush=True)
                all_pair_results.append({
                    "model": model_cfg["name"],
                    "dataset": ds_cfg["name"],
                    "error": str(e),
                })

    # ── H1_new and H2: within-model Spearman ──────────────────────────────
    print("\n[STEP 3] Within-model Spearman analysis", flush=True)
    rho_q_list = []
    rho_map_list = []
    within_model_details = []

    for pr in all_pair_results:
        if "error" in pr:
            continue
        res_q, res_map, valid = compute_within_model_spearman(pr)
        if res_q is None:
            continue
        rho_q, pval_q = res_q
        rho_map, pval_map = res_map

        rho_q_list.append(rho_q)
        rho_map_list.append(rho_map)
        within_model_details.append({
            "model": pr["model"],
            "dataset": pr["dataset"],
            "n_layers_valid": len(valid),
            "spearman_rho_kappa_vs_q1nn": rho_q,
            "p_kappa_vs_q1nn": pval_q,
            "spearman_rho_kappa_vs_map10": rho_map,
            "p_kappa_vs_map10": pval_map,
        })
        print(f"  {pr['model']} x {pr['dataset']}: "
              f"rho(kappa,q)={rho_q:.3f}  rho(kappa,MAP@10)={rho_map:.3f}", flush=True)

    mean_rho_q = float(np.mean(rho_q_list)) if rho_q_list else None
    frac_pos_q = float(np.mean([r > 0 for r in rho_q_list])) if rho_q_list else None
    mean_rho_map = float(np.mean(rho_map_list)) if rho_map_list else None
    frac_pos_map = float(np.mean([r > 0 for r in rho_map_list])) if rho_map_list else None

    h1_new_pass = (mean_rho_q is not None
                   and mean_rho_q > THRESH_MEAN_RHO
                   and frac_pos_q >= THRESH_FRAC_POS)
    h2_pass = (mean_rho_map is not None
               and mean_rho_map > THRESH_MEAN_RHO)

    print(f"\n  H1_new: mean_rho(kappa,q_1nn)={mean_rho_q:.4f}, "
          f"frac_pos={frac_pos_q:.2f} -> {'PASS' if h1_new_pass else 'FAIL'}", flush=True)
    print(f"  H2:     mean_rho(kappa,MAP@10)={mean_rho_map:.4f}, "
          f"frac_pos={frac_pos_map:.2f} -> {'PASS' if h2_pass else 'FAIL'}", flush=True)

    # ── H3: cross-model on banking77 ─────────────────────────────────────
    print("\n[STEP 4] H3: cross-model ranking on banking77 (final layer)", flush=True)
    cross_model = []
    for pr in all_pair_results:
        if "error" in pr or pr.get("dataset") != "banking77":
            continue
        lr = pr["layer_results"]
        final = max(lr, key=lambda r: r["layer_idx"])
        if final["kappa_nearest"] is not None and final["map_at_10"] is not None:
            cross_model.append({
                "model": pr["model"],
                "kappa_nearest_final": final["kappa_nearest"],
                "map_at_10_final": final["map_at_10"],
                "q_1nn_final": final["q_1nn"],
                "layer_idx": final["layer_idx"],
            })

    h3_pass = False
    h3_detail = {}
    if len(cross_model) >= 3:
        kappas_cross = [x["kappa_nearest_final"] for x in cross_model]
        maps_cross = [x["map_at_10_final"] for x in cross_model]
        rho_h3, pval_h3 = spearmanr(kappas_cross, maps_cross)
        h3_pass = float(rho_h3) > THRESH_MEAN_RHO
        h3_detail = {
            "n_models": len(cross_model),
            "spearman_rho": float(rho_h3),
            "p_value": float(pval_h3),
            "models": cross_model,
            "PASS": h3_pass,
        }
        print(f"  H3: Spearman rho(kappa,MAP@10) across {len(cross_model)} models = "
              f"{rho_h3:.3f} (p={pval_h3:.3f}) -> {'PASS' if h3_pass else 'FAIL'}", flush=True)
    else:
        print(f"  H3: insufficient models ({len(cross_model)} < 3)", flush=True)
        h3_detail = {"n_models": len(cross_model), "PASS": False,
                     "note": "insufficient models"}

    # ── Variance diagnostics ─────────────────────────────────────────────
    print("\n[STEP 5] Variance diagnostics", flush=True)
    diag = {}
    for pr in all_pair_results:
        if "error" in pr:
            continue
        lr = pr["layer_results"]
        valid_lr = [r for r in lr if r["map_at_10"] is not None]
        if not valid_lr:
            continue
        maps = [r["map_at_10"] for r in valid_lr]
        ceil_fracs = [r["ceiling_frac"] for r in valid_lr if r["ceiling_frac"] is not None]
        key = f"{pr['model']}_{pr['dataset']}"
        diag[key] = {
            "map_variance": variance_diagnostics(maps, "map@10"),
            "mean_ceiling_frac": float(np.mean(ceil_fracs)) if ceil_fracs else None,
        }
        print(f"  {key}: MAP@10 range [{min(maps):.4f}, {max(maps):.4f}], "
              f"ceiling_frac={float(np.mean(ceil_fracs)) if ceil_fracs else 'N/A'}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("DOWNSTREAM DECISION PROTOCOL V2 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  H1_existing: {'PASS' if h1_existing['PASS'] else 'FAIL'} "
          f"(mean_rho={h1_existing['mean_spearman_rho']:.4f}, "
          f"frac_pos={h1_existing['frac_positive']:.2f})", flush=True)
    print(f"  H1_new:      {'PASS' if h1_new_pass else 'FAIL'} "
          f"(mean_rho={mean_rho_q:.4f})", flush=True)
    print(f"  H2:          {'PASS' if h2_pass else 'FAIL'} "
          f"(mean_rho={mean_rho_map:.4f})", flush=True)
    print(f"  H3:          {'PASS' if h3_pass else 'FAIL'}", flush=True)

    # ── Save output ───────────────────────────────────────────────────────
    out = {
        "experiment": "cti_downstream_protocol_v2",
        "preregistration": "H1_existing, H1_new, H2, H3 (committed before run)",
        "thresholds": {
            "mean_spearman_rho": THRESH_MEAN_RHO,
            "frac_positive": THRESH_FRAC_POS,
        },
        "hypothesis_results": {
            "H1_existing": h1_existing,
            "H1_new": {
                "n_pairs": len(rho_q_list),
                "mean_spearman_rho": mean_rho_q,
                "frac_positive": frac_pos_q,
                "PASS": h1_new_pass,
                "pair_details": within_model_details,
            },
            "H2": {
                "n_pairs": len(rho_map_list),
                "mean_spearman_rho_kappa_vs_map10": mean_rho_map,
                "frac_positive": frac_pos_map,
                "PASS": h2_pass,
            },
            "H3": h3_detail,
        },
        "variance_diagnostics": diag,
        "all_pair_results": all_pair_results,
        "status": "complete",
    }

    with open(OUT_PATH, "w", encoding="ascii") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
