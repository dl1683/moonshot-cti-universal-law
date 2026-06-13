"""C86: DBPedia-14 500-sample q_knn panel for 4 Pythia models.

Loads DBPedia train split, samples 500 random examples (seed=42), computes
mean-pooled embeddings from the last hidden state, and evaluates 1-NN CV
accuracy and corrected q_knn for each model.
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoModel, AutoTokenizer

# =============================================================================
# Configuration
# =============================================================================

MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-2.8b",
]

N_SAMPLES = 500
DATASET = "fancyzhx/dbpedia_14"
SPLIT = "train"
RANDOM_SEED = 42
BATCH_SIZE = 8
MAX_LENGTH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =============================================================================
# Helpers
# =============================================================================

def load_dbpedia_sample(n: int = 500) -> tuple[List[str], np.ndarray]:
    """Load n random samples from DBPedia-14 train split."""
    print(f"  Loading {DATASET} ({SPLIT})...")
    ds = load_dataset(DATASET, split=SPLIT, trust_remote_code=True)

    total = len(ds)
    indices = random.sample(range(total), min(n, total))
    texts = [ds[i]["content"] for i in indices]
    labels = np.array([ds[i]["label"] for i in indices])
    return texts, labels


def extract_mean_pooled_hidden_states(
    model_name: str, texts: List[str], batch_size: int = BATCH_SIZE
) -> Optional[np.ndarray]:
    """Load a model/tokenizer and return mean-pooled final-layer hidden states."""
    print(f"  Loading model & tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(DEVICE)
        model.to(DTYPE)
        model.eval()
    except Exception as e:
        print(f"  Failed to load model/tokenizer: {e}")
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(DEVICE)

            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]

            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embs.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


def compute_q_knn(embeddings: np.ndarray, labels: np.ndarray, k_eff: int) -> float:
    """Compute corrected 1-NN accuracy q."""
    if k_eff <= 1:
        return 0.0
    cv = StratifiedKFold(
        n_splits=min(5, len(np.unique(labels))), shuffle=True, random_state=42
    )
    knn = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(knn, embeddings, labels, cv=cv, scoring="accuracy")
    acc = float(np.mean(scores))
    q = (acc - 1.0 / k_eff) / (1.0 - 1.0 / k_eff)
    return float(q)


# =============================================================================
# Main
# =============================================================================

def main() -> Dict[str, Any]:
    print("=" * 70)
    print("C86: DBPEDIA-14 500-SAMPLE Q_KNN PANEL (PYTHIA)")
    print("=" * 70)
    print(f"Models: {MODELS}")
    print(f"Dataset: {DATASET} ({SPLIT}, n={N_SAMPLES})")
    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    print()

    # Load data once
    texts, labels = load_dbpedia_sample(N_SAMPLES)
    k_eff = len(np.unique(labels))
    print(f"  Effective classes (k_eff) = {k_eff}")
    print()

    records: List[Dict[str, Any]] = []

    for model_name in MODELS:
        print("-" * 70)
        print(f"MODEL: {model_name}")
        print("-" * 70)

        record: Dict[str, Any] = {
            "model": model_name,
            "q_knn": None,
            "accuracy": None,
            "status": "pending",
            "error": None,
        }

        # --- Embedding extraction ---
        embs = extract_mean_pooled_hidden_states(model_name, texts, batch_size=BATCH_SIZE)
        if embs is None:
            record["status"] = "embedding_failed"
            record["error"] = "Failed to extract embeddings"
            print(f"  Skipping {model_name} (embedding extraction failed).")
            records.append(record)
            print()
            continue

        # --- CTI metrics ---
        cv = StratifiedKFold(
            n_splits=min(5, len(np.unique(labels))), shuffle=True, random_state=42
        )
        knn = KNeighborsClassifier(n_neighbors=1)
        scores = cross_val_score(knn, embs, labels, cv=cv, scoring="accuracy")
        acc = float(np.mean(scores))
        q = compute_q_knn(embs, labels, k_eff)

        record["accuracy"] = acc
        record["q_knn"] = q
        record["status"] = "success"
        print(f"  1-NN CV accuracy = {acc:.4f}")
        print(f"  q_knn            = {q:.4f}")
        print()

        records.append(record)

        # Clean up GPU memory before next model
        del embs
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    output = {
        "experiment": "c86_dbpedia_500_qknn_pythia_panel",
        "config": {
            "models": MODELS,
            "dataset": DATASET,
            "split": SPLIT,
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "device": str(DEVICE),
            "dtype": str(DTYPE),
        },
        "records": records,
    }

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "c86_dbpedia_500_qknn_pythia_panel.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {out_path}")

    return output


if __name__ == "__main__":
    main()
