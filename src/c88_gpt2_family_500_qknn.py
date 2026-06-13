"""C88: GPT-2 Family 500-Sample q_knn Measurement.

Computes 1-NN CV accuracy and corrected q_knn on 500 AG News samples
for gpt2 and gpt2-medium using mean-pooled last-hidden-state embeddings.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import AutoModel, AutoTokenizer

# =============================================================================
# Configuration
# =============================================================================

MODELS = [
    "gpt2",
    "gpt2-medium",
]

N_AG_NEWS_SAMPLES = 500
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

def compute_1nn_cv_accuracy(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute 1-NN cross-validated accuracy using stratified 5-fold CV."""
    n_classes = len(np.unique(labels))
    n_splits = min(5, n_classes)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    knn = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(knn, embeddings, labels, cv=cv, scoring="accuracy")
    return float(np.mean(scores))


def compute_q_knn(embeddings: np.ndarray, labels: np.ndarray, k_eff: int) -> float:
    """Compute corrected 1-NN accuracy q."""
    if k_eff <= 1:
        return 0.0
    acc = compute_1nn_cv_accuracy(embeddings, labels)
    q = (acc - 1.0 / k_eff) / (1.0 - 1.0 / k_eff)
    return float(q)


def load_ag_news_sample(n: int = 500) -> tuple[List[str], np.ndarray]:
    """Load n random samples from fancyzhx/ag_news train split."""
    print(f"  Loading fancyzhx/ag_news (train split, sampling {n} texts)...")
    ds = load_dataset("fancyzhx/ag_news", split="train", trust_remote_code=True)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    texts = [ds[i]["text"] for i in indices]
    labels = np.array([ds[i]["label"] for i in indices])
    return texts, labels


def extract_mean_pooled_hidden_states(
    model_name: str, texts: List[str], batch_size: int = BATCH_SIZE
) -> Optional[np.ndarray]:
    """Load a model/tokenizer and return mean-pooled final-layer hidden states."""
    print(f"  Loading model & tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(DEVICE)
        model.to(DTYPE)
        model.eval()
    except Exception as e:
        print(f"  Failed to load model/tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Ensure pad token exists
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
            # Final-layer hidden states: shape (batch, seq_len, hidden)
            last_hidden = outputs.hidden_states[-1]

            # Mean pool using attention mask
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embs.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


# =============================================================================
# Main sweep
# =============================================================================

def main() -> Dict[str, Any]:
    print("=" * 70)
    print("C88: GPT-2 FAMILY 500-SAMPLE q_knn")
    print("=" * 70)
    print(f"Models: {MODELS}")
    print(f"Dataset: fancyzhx/ag_news (n={N_AG_NEWS_SAMPLES})")
    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    print()

    # Load AG News once
    ag_texts, ag_labels = load_ag_news_sample(N_AG_NEWS_SAMPLES)
    k_eff = len(np.unique(ag_labels))
    print(f"  Effective classes (k_eff) = {k_eff}")
    print()

    records: List[Dict[str, Any]] = []

    for model_name in MODELS:
        print("-" * 70)
        print(f"MODEL: {model_name}")
        print("-" * 70)

        record: Dict[str, Any] = {
            "model": model_name,
            "1nn_cv_accuracy": None,
            "q_knn": None,
            "status": "pending",
            "error": None,
        }

        # --- Embedding extraction ---
        embs = extract_mean_pooled_hidden_states(model_name, ag_texts, batch_size=BATCH_SIZE)
        if embs is None:
            record["status"] = "embedding_failed"
            print(f"  Skipping {model_name} (embedding extraction failed).")
            records.append(record)
            print()
            continue

        # --- 1-NN CV accuracy ---
        acc = compute_1nn_cv_accuracy(embs, ag_labels)
        record["1nn_cv_accuracy"] = acc
        print(f"  1-NN CV acc   = {acc:.4f}")

        # --- q_knn ---
        q = compute_q_knn(embs, ag_labels, k_eff)
        record["q_knn"] = q
        record["status"] = "success"
        print(f"  q_knn         = {q:.4f}")
        print()

        records.append(record)

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    output = {
        "experiment": "c88_gpt2_family_500_qknn",
        "config": {
            "models": MODELS,
            "n_ag_news_samples": N_AG_NEWS_SAMPLES,
            "random_seed": RANDOM_SEED,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "device": str(DEVICE),
            "dtype": str(DTYPE),
        },
        "records": records,
    }

    out_path = Path(__file__).resolve().parents[1] / "results" / "c88_gpt2_family_500_qknn.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {out_path}")

    return output


if __name__ == "__main__":
    main()
