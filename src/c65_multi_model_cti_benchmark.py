"""C65: Multi-Model CTI Benchmark Sweep.

Evaluates three small language models on arc_easy via lm_eval, computes
CTI kappa_nearest on AG News embeddings, and tests the correlation
between benchmark performance and geometric separability.
"""

import json
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import AutoModel, AutoTokenizer

# lm-eval harness
from lm_eval import simple_evaluate


# =============================================================================
# Configuration
# =============================================================================

MODELS = [
    "gpt2",
    "EleutherAI/pythia-160m",
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/pythia-410m",
]

TASK = "arc_easy"
NUM_FEWSHOT = 0
LM_EVAL_BATCH_SIZE = "auto"
EMB_BATCH_SIZE = 8
N_AG_NEWS_SAMPLES = 100
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =============================================================================
# CTI geometry helpers (adapted from c59)
# =============================================================================

def compute_kappa_nearest(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute nearest-class separation (kappa_nearest).

    For each class centroid, measure the distance from every sample to the
    nearest *other* class centroid.  Kappa is the mean of these distances.
    """
    classes = np.unique(labels)
    if len(classes) <= 1:
        return 0.0

    centroids = []
    for c in classes:
        centroids.append(embeddings[labels == c].mean(axis=0))
    centroids = np.array(centroids)

    min_distances = []
    for i, c in enumerate(classes):
        class_points = embeddings[labels == c]
        other_centroids = np.delete(centroids, i, axis=0)
        if other_centroids.shape[0] == 0:
            continue
        dists = np.linalg.norm(
            class_points[:, None, :] - other_centroids[None, :, :], axis=2
        )
        min_dist = dists.min(axis=1)
        min_distances.extend(min_dist.tolist())

    return float(np.mean(min_distances)) if min_distances else 0.0


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
# Benchmark helper
# =============================================================================

def run_lm_eval(model_name: str) -> Optional[float]:
    """Run arc_easy zero-shot evaluation and return the accuracy score."""
    print(f"  Running lm_eval on {TASK} (zero-shot, batch_size={LM_EVAL_BATCH_SIZE})...")

    try:
        results = simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name},dtype={str(DTYPE).split('.')[-1]}",
            tasks=[TASK],
            num_fewshot=NUM_FEWSHOT,
            batch_size=LM_EVAL_BATCH_SIZE,
            device=str(DEVICE),
        )
    except Exception as e:
        print(f"  lm_eval failed: {e}")
        traceback.print_exc()
        return None

    task_results = results.get("results", {}).get(TASK, {})
    # arc_easy primary metric is "acc"; fall back to "acc_norm" if needed
    score = task_results.get("acc")
    if score is None:
        score = task_results.get("acc_norm")
    if score is None:
        # last resort: grab the first scalar metric we see
        for k, v in task_results.items():
            if isinstance(v, (int, float)):
                score = float(v)
                print(f"  (used fallback metric '{k}' = {score:.4f})")
                break

    if score is None:
        print(f"  Could not extract score from results keys: {list(task_results.keys())}")
        return None

    print(f"  {TASK} score = {score:.4f}")
    return float(score)


# =============================================================================
# Embedding extraction
# =============================================================================

def load_ag_news_sample(n: int = 100) -> tuple[List[str], np.ndarray]:
    """Load n random samples from fancyzhx/ag_news train split."""
    print(f"  Loading fancyzhx/ag_news (train split, sampling {n} texts)...")
    ds = load_dataset("fancyzhx/ag_news", split="train")

    # ag_news has 'text' and 'label' fields
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    texts = [ds[i]["text"] for i in indices]
    labels = np.array([ds[i]["label"] for i in indices])
    return texts, labels


def extract_mean_pooled_hidden_states(
    model_name: str, texts: List[str], batch_size: int = 8
) -> Optional[np.ndarray]:
    """Load a model/tokenizer and return mean-pooled final-layer hidden states."""
    print(f"  Loading model & tokenizer for embedding extraction...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(DEVICE)
        model.to(DTYPE)
        model.eval()
    except Exception as e:
        print(f"  Failed to load model/tokenizer: {e}")
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
                max_length=512,
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
    print("C65: MULTI-MODEL CTI BENCHMARK SWEEP")
    print("=" * 70)
    print(f"Models: {MODELS}")
    print(f"Benchmark: {TASK} (zero-shot)")
    print(f"CTI dataset: fancyzhx/ag_news (n={N_AG_NEWS_SAMPLES})")
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
            "benchmark_score": None,
            "kappa_nearest": None,
            "q_knn": None,
            "status": "pending",
            "error": None,
        }

        # --- Benchmark ---
        score = run_lm_eval(model_name)
        if score is None:
            record["status"] = "benchmark_failed"
            print(f"  Skipping CTI for {model_name} (benchmark failed).")
            records.append(record)
            print()
            continue
        record["benchmark_score"] = score

        # --- Embedding extraction ---
        embs = extract_mean_pooled_hidden_states(model_name, ag_texts, batch_size=EMB_BATCH_SIZE)
        if embs is None:
            record["status"] = "embedding_failed"
            print(f"  Skipping CTI for {model_name} (embedding extraction failed).")
            records.append(record)
            print()
            continue

        # --- CTI metrics ---
        kappa = compute_kappa_nearest(embs, ag_labels)
        q = compute_q_knn(embs, ag_labels, k_eff)
        record["kappa_nearest"] = kappa
        record["q_knn"] = q
        record["status"] = "success"
        print(f"  kappa_nearest = {kappa:.4f}")
        print(f"  q_knn         = {q:.4f}")
        print()

        records.append(record)

    # -------------------------------------------------------------------------
    # Pearson correlation across successful models
    # -------------------------------------------------------------------------
    successful = [r for r in records if r["status"] == "success"]
    correlation: Dict[str, Any] = {
        "n_models": len(successful),
        "r": None,
        "p_value": None,
        "valid": False,
    }

    if len(successful) >= 2:
        scores = np.array([r["benchmark_score"] for r in successful])
        kappas = np.array([r["kappa_nearest"] for r in successful])

        if len(set(kappas)) > 1 and len(set(scores)) > 1:
            r_val, p_val = pearsonr(kappas, scores)
            correlation["r"] = float(r_val)
            correlation["p_value"] = float(p_val)
            correlation["valid"] = True
            print("=" * 70)
            print("PEARSON CORRELATION (kappa vs. benchmark score)")
            print("=" * 70)
            print(f"  n = {len(successful)}")
            print(f"  r = {r_val:.4f}")
            print(f"  p = {p_val:.4f}")
            print()
        else:
            print("=" * 70)
            print("PEARSON CORRELATION")
            print("=" * 70)
            print("  Insufficient variance for correlation (constant kappa or score).")
            print()
    else:
        print("=" * 70)
        print("PEARSON CORRELATION")
        print("=" * 70)
        print(f"  Only {len(successful)} model(s) succeeded; need >= 2 for correlation.")
        print()

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    output = {
        "experiment": "c65_multi_model_cti_benchmark",
        "config": {
            "models": MODELS,
            "task": TASK,
            "num_fewshot": NUM_FEWSHOT,
            "lm_eval_batch_size": LM_EVAL_BATCH_SIZE,
            "emb_batch_size": EMB_BATCH_SIZE,
            "n_ag_news_samples": N_AG_NEWS_SAMPLES,
            "device": str(DEVICE),
            "dtype": str(DTYPE),
        },
        "records": records,
        "correlation": correlation,
    }

    out_path = Path(__file__).resolve().parents[1] / "results" / "c67_4model_arc_easy_cti_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {out_path}")

    return output


if __name__ == "__main__":
    main()
