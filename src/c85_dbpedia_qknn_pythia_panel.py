#!/usr/bin/env python -u
"""
C85 DBPedia q_knn Pythia Panel
==============================
Compute q_knn for the 4 Pythia models on DBPedia (replacing AG News).

For each model:
1. Load DBPedia train split, sample 100 random examples
2. Compute mean-pooled embeddings from last hidden state
3. Compute 1-NN LOO CV accuracy and q_knn

Models:
- EleutherAI/pythia-160m
- EleutherAI/pythia-410m
- EleutherAI/pythia-1b
- EleutherAI/pythia-2.8b

Output: results/c85_dbpedia_qknn_pythia_panel.json
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_DIR / "c85_dbpedia_qknn_pythia_panel.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-2.8b",
]

DATASET_HF = "fancyzhx/dbpedia_14"
SPLIT = "train"
TEXT_COL = "content"
LABEL_COL = "label"
N_SAMPLE = 100
RANDOM_SEED = 42
BATCH_SIZE = 8
K_FULL = 14  # DBPedia has 14 classes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_dbpedia_sample(n_sample: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Load DBPedia train split and sample n random examples."""
    ds = load_dataset(DATASET_HF, split=SPLIT, trust_remote_code=True)
    rng = np.random.default_rng(seed)
    total = len(ds)
    idx = rng.choice(total, size=min(n_sample, total), replace=False)
    texts = [str(ds[TEXT_COL][int(i)]) for i in idx]
    labels = np.array([int(ds[LABEL_COL][int(i)]) for i in idx])

    # Relabel to contiguous integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return texts, labels


@torch.no_grad()
def extract_embeddings(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Extract mean-pooled last-hidden-state embeddings."""
    model.to(DEVICE)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            outputs = model(**enc, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
            embeddings.append(pooled.cpu().float().numpy())
    return np.vstack(embeddings)


def compute_1nn_loo_cv(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute leave-one-out 1-NN classification accuracy."""
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    for train_idx, test_idx in loo.split(embeddings):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
        knn.fit(X_train, y_train)
        y_true.append(y_test[0])
        y_pred.append(knn.predict(X_test)[0])
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def compute_q_knn(acc: float, K: int) -> float:
    """Normalized 1-NN quality: (acc - 1/K) / (1 - 1/K)."""
    return (acc - 1.0 / K) / (1.0 - 1.0 / K)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 60, flush=True)
    print("C85 DBPedia q_knn — Pythia Panel", flush=True)
    print("=" * 60, flush=True)

    texts, labels = load_dbpedia_sample(N_SAMPLE, RANDOM_SEED)
    K_eff = int(len(np.unique(labels)))
    print(f"\nLoaded DBPedia train: n={len(texts)}, K_eff={K_eff} (full K={K_FULL})", flush=True)

    # For q_knn use the full dataset K if all classes present, else effective K
    K_for_q = K_FULL if K_eff == K_FULL else K_eff

    panel = []
    for model_id in MODELS:
        model_key = model_id.split("/")[-1]
        print(f"\n--- {model_key} ---", flush=True)
        t0 = time.time()

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            embs = extract_embeddings(model, tokenizer, texts)

            acc_1nn = compute_1nn_loo_cv(embs, labels)
            q_knn = compute_q_knn(acc_1nn, K_for_q)

            elapsed = time.time() - t0
            print(
                f"  acc_1nn={acc_1nn:.4f}  q_knn={q_knn:.4f}  ({elapsed:.1f}s)",
                flush=True,
            )

            panel.append({
                "model": model_key,
                "model_id": model_id,
                "dataset": "dbpedia_14",
                "n_samples": len(texts),
                "K_full": K_FULL,
                "K_eff": K_eff,
                "K_for_q": K_for_q,
                "acc_1nn": round(acc_1nn, 6),
                "q_knn": round(q_knn, 6),
                "elapsed_seconds": round(elapsed, 2),
            })

            del model
            torch.cuda.empty_cache()

        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            panel.append({
                "model": model_key,
                "model_id": model_id,
                "dataset": "dbpedia_14",
                "status": "error",
                "error": str(exc),
            })

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("Summary — DBPedia q_knn (Pythia panel)", flush=True)
    print("=" * 60, flush=True)
    for row in panel:
        if "q_knn" in row:
            print(f"  {row['model']:<18} q_knn = {row['q_knn']:.4f}", flush=True)

    out = {
        "experiment": "c85_dbpedia_qknn_pythia_panel",
        "dataset_hf": DATASET_HF,
        "split": SPLIT,
        "n_sample": N_SAMPLE,
        "random_seed": RANDOM_SEED,
        "device": str(DEVICE),
        "panel": panel,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
