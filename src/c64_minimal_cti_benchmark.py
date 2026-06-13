"""
c64_minimal_cti_benchmark.py
============================
Minimal end-to-end script that:
1. Runs a lightweight lm-eval benchmark on a small causal LM.
2. Computes CTI kappa_nearest on a small classification sample (agnews, 100 samples).
3. Records both numbers and saves them to results/c64_minimal_cti_benchmark.json.

The script is self-contained, catches lm-eval failures, and falls back to
recording the error rather than crashing.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "c64_minimal_cti_benchmark.json"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "gpt2"               # small, fast, widely cached
TASK = "arc_easy"               # lightweight multiple-choice QA
BATCH_SIZE = "auto"             # lm-eval batch size
NUM_FEWSHOT = 0                 # zero-shot for speed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CTI_DATASET = "fancyzhx/ag_news"
CTI_SPLIT = "train"
CTI_TEXT_COL = "text"
CTI_LABEL_COL = "label"
CTI_MAX_SAMPLES = 100           # small dataset as requested
CTI_BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# CTI geometry helpers (kappa_nearest)
# ---------------------------------------------------------------------------

def compute_kappa_nearest(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    """Canonical CTI kappa_nearest = min centroid gap / (sigma_W * sqrt(d))."""
    classes = np.unique(labels)
    K = len(classes)
    if K < 2:
        return None
    d = embeddings.shape[1]

    centroids: dict[int, np.ndarray] = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = embeddings[mask].mean(axis=0)
    if len(centroids) < 2:
        return None

    # pooled within-class standard deviation
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

    # nearest-centroid gap
    ckeys = sorted(centroids.keys())
    cent_arr = np.stack([centroids[c] for c in ckeys])
    min_gap = float("inf")
    for i in range(len(ckeys)):
        for j in range(i + 1, len(ckeys)):
            gap = float(np.linalg.norm(cent_arr[i] - cent_arr[j]))
            if gap < min_gap:
                min_gap = gap

    return float(min_gap / (sigma_W * np.sqrt(d) + 1e-12))


def extract_embeddings_decoder(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int = CTI_BATCH_SIZE,
) -> np.ndarray:
    """Extract mean-pooled final-layer hidden states from a decoder model."""
    model.to(DEVICE)
    model.eval()
    embeddings: list[np.ndarray] = []
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
            # final layer hidden states: (batch, seq_len, hidden_dim)
            last_hidden = outputs.hidden_states[-1]
            # mean pool over real tokens (exclude padding)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# lm-eval benchmark runner
# ---------------------------------------------------------------------------

def run_lm_eval_benchmark(model_id: str, task: str, batch_size: str | int, num_fewshot: int) -> dict[str, Any]:
    """Run a single lm-eval task via the simple_evaluate API."""
    # Try both common import paths for robustness across lm-eval versions
    try:
        from lm_eval import simple_evaluate  # type: ignore
    except ImportError:
        from lm_eval.evaluator import simple_evaluate  # type: ignore

    model_args = f"pretrained={model_id},dtype=float16" if DEVICE == "cuda" else f"pretrained={model_id}"

    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=[task],
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        device=DEVICE,
    )
    return results


def extract_score(results: dict[str, Any], task: str) -> float | None:
    """Extract a single scalar accuracy-like score from lm-eval results."""
    # lm-eval result structure: results[task] -> dict of metrics
    task_results = results.get("results", {}).get(task, {})
    if not task_results:
        return None

    # Prefer standard accuracy keys in order of likelihood for arc_easy
    for key in ("acc,none", "acc_norm,none", "acc", "acc_norm", "exact_match,none", "exact_match"):
        if key in task_results:
            val = task_results[key]
            if isinstance(val, (int, float)):
                return float(val)
    # Fallback: return first numeric metric
    for k, v in task_results.items():
        if isinstance(v, (int, float)):
            return float(v)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "model": MODEL_ID,
        "task": TASK,
        "device": DEVICE,
        "cti_dataset": CTI_DATASET,
        "cti_max_samples": CTI_MAX_SAMPLES,
    }

    # ------------------------------------------------------------------
    # 1. Benchmark
    # ------------------------------------------------------------------
    print("[1/3] Running lm-eval benchmark...", flush=True)
    try:
        bench_results = run_lm_eval_benchmark(MODEL_ID, TASK, BATCH_SIZE, NUM_FEWSHOT)
        bench_score = extract_score(bench_results, TASK)
        record["benchmark"] = {
            "status": "success",
            "score": bench_score,
            "raw_results": bench_results.get("results", {}),
        }
        print(f"    Benchmark score ({TASK}): {bench_score}", flush=True)
    except Exception as exc:
        record["benchmark"] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"    Benchmark failed: {exc}", flush=True)
        bench_score = None

    # ------------------------------------------------------------------
    # 2. CTI kappa_nearest on small agnews sample
    # ------------------------------------------------------------------
    print("[2/3] Computing CTI kappa_nearest...", flush=True)
    try:
        ds = load_dataset(CTI_DATASET, split=CTI_SPLIT, trust_remote_code=True)
        texts = [str(t) for t in ds[CTI_TEXT_COL]]
        labels = np.array([int(l) for l in ds[CTI_LABEL_COL]])

        if len(texts) > CTI_MAX_SAMPLES:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(texts), CTI_MAX_SAMPLES, replace=False)
            texts = [texts[i] for i in idx]
            labels = labels[idx]

        # relabel to contiguous integers
        ulabels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(ulabels)}
        labels = np.array([label_map[l] for l in labels])

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use base model (not the lm_head) for hidden-state embeddings
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        embs = extract_embeddings_decoder(model, tokenizer, texts)
        kappa = compute_kappa_nearest(embs, labels)

        record["cti"] = {
            "status": "success",
            "kappa_nearest": kappa,
            "n_samples": len(labels),
            "n_classes": int(len(np.unique(labels))),
            "embedding_dim": int(embs.shape[1]),
        }
        print(f"    kappa_nearest: {kappa}", flush=True)
    except Exception as exc:
        record["cti"] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"    CTI computation failed: {exc}", flush=True)
        kappa = None

    # ------------------------------------------------------------------
    # 3. Simple prediction test (single point, just report both)
    # ------------------------------------------------------------------
    print("[3/3] Evaluating kappa->score prediction...", flush=True)
    if bench_score is not None and kappa is not None:
        record["prediction_test"] = {
            "status": "single_point",
            "note": (
                "With only 1 model, a strict correlation test is impossible. "
                "Both quantities were successfully computed and can be used in a multi-model sweep."
            ),
            "bench_score": bench_score,
            "kappa_nearest": kappa,
        }
        print(f"    Both values recorded: score={bench_score}, kappa={kappa}", flush=True)
    else:
        record["prediction_test"] = {
            "status": "incomplete",
            "reason": "benchmark or kappa missing due to earlier error",
        }
        print("    Prediction test skipped because one value is missing.", flush=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    OUT_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
