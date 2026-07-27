"""Workload data loaders for Atlas R2.

Handles task selection (deterministic SHA256 hash), formatting prompts,
and scoring outputs for each workload type.
"""

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"


def _hash_rank(seed, task_id):
    """Deterministic rank via SHA256(seed || task_id)."""
    h = hashlib.sha256(f"{seed}{task_id}".encode("utf-8")).hexdigest()
    return h


def load_mkqa(n_queries=40, languages=None):
    """Load MKQA (W-D2) with deterministic task selection.

    Returns list of episodes: {query_id, lang, question, answers, aliases}.
    """
    if languages is None:
        languages = ["en", "es", "fr", "de", "ja", "zh_cn", "ar", "ko"]

    mkqa_path = DATA_DIR / "mkqa.jsonl"
    if not mkqa_path.exists():
        raise FileNotFoundError(
            f"MKQA data not found at {mkqa_path}. "
            "Download: https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"
        )

    examples = []
    with open(mkqa_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)

    ranked = sorted(
        examples,
        key=lambda ex: _hash_rank("atlas-r2-d2", str(ex["example_id"])),
    )
    selected = ranked[:n_queries]

    episodes = []
    for ex in selected:
        for lang in languages:
            query = ex.get("queries", {}).get(lang, "")
            answers_raw = ex.get("answers", {}).get(lang, [])

            if isinstance(answers_raw, list):
                answer_texts = []
                aliases = []
                for a in answers_raw:
                    if isinstance(a, dict):
                        t = a.get("text", "")
                        if t:
                            answer_texts.append(t)
                        for alias in a.get("aliases", []):
                            aliases.append(alias)
                    elif isinstance(a, str):
                        answer_texts.append(a)
            else:
                answer_texts = [str(answers_raw)]
                aliases = []

            if not query:
                continue

            episodes.append({
                "query_id": ex["example_id"],
                "lang": lang,
                "task_id": f"mkqa_{ex['example_id']}_{lang}",
                "question": query,
                "answers": answer_texts,
                "aliases": aliases,
            })

    return episodes


def format_mkqa_prompt(episode):
    """Format an MKQA episode as a generation prompt."""
    q = episode["question"]
    return (
        f"Answer the following question concisely. "
        f"Give only the answer, no explanation.\n\n"
        f"Question: {q}\n"
        f"Answer:"
    )


def _normalize(text):
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,;:!?]$", "", text)
    return text


def score_mkqa(prediction, episode):
    """Score a prediction against MKQA ground truth.

    Returns dict with exact_match, f1, and pass/fail status.
    """
    pred_norm = _normalize(prediction)

    candidates = episode["answers"] + episode["aliases"]
    candidates = [_normalize(c) for c in candidates if c]

    if not candidates:
        return {"exact_match": False, "f1": 0.0, "status": "no_answer"}

    exact = any(pred_norm == c for c in candidates)

    best_f1 = 0.0
    for cand in candidates:
        pred_tokens = set(pred_norm.split())
        cand_tokens = set(cand.split())
        if not pred_tokens or not cand_tokens:
            continue
        common = pred_tokens & cand_tokens
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(cand_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    passed = exact or best_f1 >= 0.5

    return {
        "exact_match": exact,
        "f1": round(best_f1, 4),
        "status": "pass" if passed else "fail",
    }


if __name__ == "__main__":
    print("Loading MKQA (W-D2)...")
    episodes = load_mkqa(n_queries=5, languages=["en", "es"])
    print(f"Episodes: {len(episodes)}")

    for ep in episodes[:4]:
        print(f"\n  [{ep['lang']}] {ep['question'][:80]}")
        print(f"  Answers: {ep['answers'][:3]}")
        print(f"  Prompt: {format_mkqa_prompt(ep)[:100]}...")

        fake_pred = ep["answers"][0] if ep["answers"] else "unknown"
        score = score_mkqa(fake_pred, ep)
        print(f"  Score (gold answer): {score}")

        score_wrong = score_mkqa("definitely wrong answer", ep)
        print(f"  Score (wrong): {score_wrong}")
