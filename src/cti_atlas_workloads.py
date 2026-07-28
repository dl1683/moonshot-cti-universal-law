"""Workload data loaders and scientific scorers for Atlas R2.

Handles deterministic task selection (SHA256 hash), prompt formatting,
and scoring with official evaluation semantics.

Protocol: R2.1 (precommit/atlas_r2_protocol_r2_1.md)
  - MKQA: official Apple mixed_segmentation, Counter F1, language macro-average
  - PolicyBench: 100 household-level structured JSON generations
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

SCORER_VERSION = "r2.1.0"


def _hash_rank(seed, task_id):
    """Deterministic rank via SHA256(seed || task_id)."""
    h = hashlib.sha256(f"{seed}{task_id}".encode("utf-8")).hexdigest()
    return h


# ---------------------------------------------------------------------------
# MKQA (W-D2)
# ---------------------------------------------------------------------------

_MKQA_LANGUAGES = ["en", "es", "fr", "de", "ja", "zh_cn", "ar", "ko"]

ARTICLE_REGEX = {
    "en": re.compile(r"\b(a|an|the)\b"),
    "es": re.compile(r"\b(el|la|los|las|un|una|unos|unas)\b"),
    "fr": re.compile(r"\b(le|la|les|l'|un|une|des)\b"),
    "de": re.compile(
        r"\b(der|die|das|den|dem|des|ein|eine|einem|einen|eines|einer)\b"
    ),
    "ar": re.compile(r"\bال\b"),
}


def _is_cjk(char):
    """Check if character is CJK (Chinese, Japanese kanji, kana)."""
    cp = ord(char)
    if 0x4E00 <= cp <= 0x9FFF:
        return True
    if 0x3400 <= cp <= 0x4DBF:
        return True
    if 0x20000 <= cp <= 0x2A6DF:
        return True
    if 0x2A700 <= cp <= 0x2B73F:
        return True
    if 0x2B740 <= cp <= 0x2B81F:
        return True
    if 0x2B820 <= cp <= 0x2CEAF:
        return True
    if 0xF900 <= cp <= 0xFAFF:
        return True
    if 0x2F800 <= cp <= 0x2FA1F:
        return True
    if 0x3040 <= cp <= 0x309F:
        return True
    if 0x30A0 <= cp <= 0x30FF:
        return True
    if 0x31F0 <= cp <= 0x31FF:
        return True
    if 0xFF65 <= cp <= 0xFF9F:
        return True
    return False


def _mixed_segmentation(text):
    """Official MKQA mixed segmentation for ja/zh_cn.

    CJK characters become individual tokens; non-CJK runs split on whitespace.
    """
    tokens = []
    buf = []
    for ch in text:
        if _is_cjk(ch):
            if buf:
                tokens.extend("".join(buf).split())
                buf = []
            tokens.append(ch)
        else:
            buf.append(ch)
    if buf:
        tokens.extend("".join(buf).split())
    return tokens


def _normalize_mkqa(text, lang):
    """Official MKQA normalization: lowercase, remove articles, collapse whitespace."""
    text = text.lower().strip()
    regex = ARTICLE_REGEX.get(lang)
    if regex:
        text = regex.sub(" ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def _tokenize_mkqa(text, lang):
    """Tokenize per MKQA language rules."""
    if lang in ("ja", "zh_cn"):
        return _mixed_segmentation(text)
    return text.split()


def _compute_f1(pred_tokens, gold_tokens):
    """Token-level F1 with multiplicities via Counter."""
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _query_is_answerable(example, languages):
    """True if the query has at least one non-empty answer in any target language."""
    answers = example.get("answers", {})
    for lang in languages:
        lang_answers = answers.get(lang, [])
        if isinstance(lang_answers, list):
            for a in lang_answers:
                if isinstance(a, dict):
                    if a.get("text", ""):
                        return True
                elif isinstance(a, str) and a:
                    return True
        elif lang_answers:
            return True
    return False


def load_mkqa(n_queries=40, languages=None):
    """Load MKQA (W-D2) with R2.1 deterministic task selection.

    Pre-filters to answerable queries, then selects first n_queries by hash rank.
    Returns list of episodes: {query_id, lang, question, answers, aliases, task_id}.
    """
    if languages is None:
        languages = list(_MKQA_LANGUAGES)

    mkqa_path = DATA_DIR / "mkqa.jsonl"
    if not mkqa_path.exists():
        raise FileNotFoundError(
            f"MKQA data not found at {mkqa_path}. "
            "Download: https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"
        )

    examples = []
    with open(mkqa_path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    answerable = [ex for ex in examples
                  if _query_is_answerable(ex, languages)]

    ranked = sorted(
        answerable,
        key=lambda ex: _hash_rank("atlas-r2-d2", str(ex["example_id"])),
    )
    selected = ranked[:n_queries]

    episodes = []
    for ex in selected:
        for lang in languages:
            query = ex.get("queries", {}).get(lang, "")
            answers_raw = ex.get("answers", {}).get(lang, [])

            answer_texts = []
            aliases = []
            if isinstance(answers_raw, list):
                for a in answers_raw:
                    if isinstance(a, dict):
                        t = a.get("text", "")
                        if t:
                            answer_texts.append(t)
                        for alias in a.get("aliases", []):
                            aliases.append(alias)
                    elif isinstance(a, str):
                        answer_texts.append(a)
            elif answers_raw:
                answer_texts = [str(answers_raw)]

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


def score_mkqa(prediction, episode):
    """Score using official MKQA evaluation semantics (R2.1).

    Uses mixed_segmentation for ja/zh_cn, Counter F1, language-specific articles.
    Returns dict with exact_match, f1, status.
    """
    lang = episode["lang"]
    pred_norm = _normalize_mkqa(prediction, lang)

    candidates = episode["answers"] + episode.get("aliases", [])
    candidates = [c for c in candidates if c]

    if not candidates:
        return {"exact_match": False, "f1": 0.0, "status": "no_answer",
                "scorer_version": SCORER_VERSION}

    pred_tokens = _tokenize_mkqa(pred_norm, lang)

    best_em = False
    best_f1 = 0.0
    for cand in candidates:
        cand_norm = _normalize_mkqa(cand, lang)
        if pred_norm == cand_norm:
            best_em = True
        cand_tokens = _tokenize_mkqa(cand_norm, lang)
        if pred_tokens and cand_tokens:
            f1 = _compute_f1(pred_tokens, cand_tokens)
            best_f1 = max(best_f1, f1)

    passed = best_em or best_f1 >= 0.5

    return {
        "exact_match": best_em,
        "f1": round(best_f1, 4),
        "status": "pass" if passed else "fail",
        "scorer_version": SCORER_VERSION,
    }


def mkqa_language_macro_average(task_records):
    """Compute macro-averaged F1 and EM across languages from task records.

    Returns dict: {lang: {mean_f1, mean_em, n}, macro_f1, macro_em}.
    """
    by_lang = defaultdict(lambda: {"f1_sum": 0.0, "em_sum": 0, "n": 0})
    for rec in task_records:
        lang = rec["lang"]
        by_lang[lang]["f1_sum"] += rec["f1"]
        by_lang[lang]["em_sum"] += int(rec["exact_match"])
        by_lang[lang]["n"] += 1

    result = {}
    f1_values = []
    em_values = []
    for lang, stats in sorted(by_lang.items()):
        n = stats["n"]
        mean_f1 = stats["f1_sum"] / n if n else 0.0
        mean_em = stats["em_sum"] / n if n else 0.0
        result[lang] = {"mean_f1": round(mean_f1, 4),
                        "mean_em": round(mean_em, 4), "n": n}
        f1_values.append(mean_f1)
        em_values.append(mean_em)

    result["macro_f1"] = round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0
    result["macro_em"] = round(sum(em_values) / len(em_values), 4) if em_values else 0.0
    return result


# ---------------------------------------------------------------------------
# PolicyBench (W-D3) -- household-level structured generations
# ---------------------------------------------------------------------------

def load_policybench(n_households=100):
    """Load PolicyBench (W-D3) -- one episode per household.

    Each episode contains ALL reference fields for that household.
    Hash-ranks scenarios and selects first n_households.
    """
    import csv

    pb_dir = DATA_DIR / "policybench"
    scenarios_path = pb_dir / "scenarios.csv"
    refs_path = pb_dir / "reference_outputs.csv"

    if not scenarios_path.exists():
        raise FileNotFoundError(
            f"PolicyBench data not found at {pb_dir}. "
            "Run: policybench reference-outputs -n 100 --seed 42"
        )

    scenarios = {}
    with open(scenarios_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios[row["scenario_id"]] = row

    refs_by_scenario = defaultdict(list)
    with open(refs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs_by_scenario[row["scenario_id"]].append(row)

    binary_vars = set()
    for refs_list in refs_by_scenario.values():
        for r in refs_list:
            if "eligible" in r["variable"]:
                binary_vars.add(r["variable"])

    ranked_ids = sorted(
        [sid for sid in refs_by_scenario if sid in scenarios],
        key=lambda sid: _hash_rank("atlas-r2-d3", sid),
    )[:n_households]

    episodes = []
    for sid in ranked_ids:
        sc = scenarios[sid]
        refs = refs_by_scenario[sid]

        fields = {}
        for r in refs:
            var = r["variable"]
            fields[var] = {
                "ref_value": float(r["value"]),
                "is_binary": var in binary_vars,
            }

        episodes.append({
            "scenario_id": sid,
            "task_id": f"pb_{sid}",
            "fields": fields,
            "scenario_json": sc.get("scenario_json", ""),
            "state": sc.get("state", ""),
            "filing_status": sc.get("filing_status", ""),
        })

    return episodes


def format_policybench_prompt(episode):
    """Format a household episode as a single structured JSON generation prompt."""
    sc = episode["scenario_json"]
    fields = episode["fields"]

    field_specs = []
    for var in sorted(fields):
        info = fields[var]
        if info["is_binary"]:
            field_specs.append(f'  "{var}": 0 or 1')
        else:
            field_specs.append(f'  "{var}": <dollar amount as number>')

    fields_str = ",\n".join(field_specs)

    return (
        "Given the following US household for tax year 2026, "
        "compute ALL of the following tax variables.\n\n"
        f"Household: {sc}\n\n"
        "Return ONLY a JSON object with these fields:\n"
        "{\n" + fields_str + "\n}\n\n"
        "Return ONLY the JSON object, nothing else."
    )


def score_policybench(prediction, episode):
    """Score a household-level PolicyBench prediction (R2.1).

    Parses JSON, scores each field, returns household-level metrics.
    """
    fields = episode["fields"]
    n_fields = len(fields)

    pred_obj = _try_parse_json(prediction)
    if pred_obj is None:
        return {
            "parse_valid": False,
            "fields_correct": 0,
            "fields_total": n_fields,
            "household_score": 0.0,
            "all_correct": False,
            "status": "parse_fail",
            "field_results": {},
            "scorer_version": SCORER_VERSION,
        }

    field_results = {}
    correct = 0
    for var, info in fields.items():
        ref = info["ref_value"]
        is_binary = info["is_binary"]

        raw_val = pred_obj.get(var)
        if raw_val is None:
            field_results[var] = {"status": "missing", "correct": False}
            continue

        try:
            pred_val = float(
                str(raw_val).replace("$", "").replace(",", "").strip()
            )
        except (ValueError, TypeError):
            field_results[var] = {"status": "parse_fail", "correct": False}
            continue

        if is_binary:
            match = (pred_val > 0.5) == (ref > 0.5)
        elif ref == 0:
            match = abs(pred_val) < 50
        else:
            rel_err = abs(pred_val - ref) / abs(ref)
            abs_err = abs(pred_val - ref)
            match = rel_err < 0.05 or abs_err < 50

        field_results[var] = {
            "status": "pass" if match else "fail",
            "correct": match,
            "predicted": pred_val,
            "reference": ref,
        }
        if match:
            correct += 1

    household_score = correct / n_fields if n_fields else 0.0
    all_correct = correct == n_fields

    return {
        "parse_valid": True,
        "fields_correct": correct,
        "fields_total": n_fields,
        "household_score": round(household_score, 4),
        "all_correct": all_correct,
        "status": "pass" if household_score >= 0.5 else "fail",
        "field_results": field_results,
        "scorer_version": SCORER_VERSION,
    }


def _try_parse_json(text):
    """Attempt to parse JSON from model output, with fallback extraction."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# R2.2 PolicyBench (indexed array format, fail-closed)
# ---------------------------------------------------------------------------

R2_2_SCORER_VERSION = "r2.4.0"


def load_r2_2_panel(panel_path):
    """Load a sealed R2.2 panel (prevalence or challenge) from JSON."""
    data = json.loads(Path(panel_path).read_bytes().decode("utf-8"))
    return data


def format_r2_2_prompt(household):
    """R2.2 indexed-array prompt per protocol Section 1.1."""
    fields = household["fields"]
    sc_json = household["scenario_json"]
    n = len(fields)

    index_lines = []
    for i, fname in enumerate(fields):
        index_lines.append(f"{i}={fname}")
    index_block = "\n".join(index_lines)

    prompt = (
        "Given the following US household for tax year 2026, "
        "compute ALL of the following tax and benefit variables.\n\n"
        f"Household: {sc_json}\n\n"
        f"Required output fields (indexed 0 to {n-1}):\n"
        f"{index_block}\n\n"
        f"Return exactly one minified JSON array and nothing else.\n"
        f"The array must contain exactly {n} entries in the indexed order above.\n"
        "Eligibility entries must be integer 0 or 1.\n"
        "Dollar entries must be signed integer dollars, "
        "rounded to the nearest dollar.\n"
        "Do not emit keys, prose, Markdown, null, NaN, Infinity, "
        "or an explanation."
    )
    return prompt


def _strip_think_block(text):
    """Strip model-family think blocks (e.g. <think>...</think>)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_r2_2_output(raw_output, expected_length, fields):
    """Fail-closed R2.2 parser per protocol Section 1.1.

    Returns (parsed_array_or_None, schema_valid, error_code).
    """
    text = _strip_think_block(raw_output)

    if not text:
        return None, False, "EMPTY"

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None, False, "NOT_JSON"

    if not isinstance(parsed, list):
        return None, False, "NOT_ARRAY"

    if len(parsed) != expected_length:
        return None, False, "WRONG_LENGTH"

    result = []
    for i, val in enumerate(parsed):
        if val is None or isinstance(val, bool):
            return None, False, "NON_INTEGER"
        if isinstance(val, float):
            if val != val:  # NaN
                return None, False, "NONFINITE"
            if val == float("inf") or val == float("-inf"):
                return None, False, "NONFINITE"
            return None, False, "NON_INTEGER"
        if isinstance(val, str):
            return None, False, "NON_INTEGER"
        if not isinstance(val, int):
            return None, False, "NON_INTEGER"

        fname = fields[i]
        if "eligible" in fname:
            if val not in (0, 1):
                return None, False, "BAD_ELIGIBILITY"

        result.append(val)

    return result, True, None


def score_r2_2_field(fname, predicted, reference):
    """Score a single R2.2 field. Returns (correct, details)."""
    is_binary = "eligible" in fname

    if is_binary:
        correct = (predicted == int(reference))
        return correct, {
            "type": "eligibility",
            "predicted": predicted,
            "reference": int(reference),
            "correct": correct,
        }

    ref_val = float(reference)
    pred_val = float(predicted)

    if ref_val == 0.0:
        correct = abs(pred_val) < 50.0
    else:
        rel_err = abs(pred_val - ref_val) / abs(ref_val)
        abs_err = abs(pred_val - ref_val)
        correct = rel_err < 0.05 or abs_err < 50.0

    zero_ref_correct = (ref_val == 0.0) or (abs(ref_val) < 50.0)

    nme = min(abs(pred_val - ref_val) / max(abs(ref_val), 50.0), 10.0)

    return correct, {
        "type": "amount",
        "predicted": predicted,
        "reference": ref_val,
        "correct": correct,
        "zero_baseline_correct": zero_ref_correct,
        "normalized_magnitude_error": round(nme, 6),
    }


def score_r2_2_household(parsed_array, household):
    """Score a complete R2.2 household. Returns detailed result dict."""
    fields = household["fields"]
    gold = household["gold_array"]
    n = len(fields)

    if parsed_array is None:
        field_results = []
        for i, fname in enumerate(fields):
            is_binary = "eligible" in fname
            ref_val = float(gold[i])
            zero_correct = (ref_val == 0.0) if is_binary else (
                ref_val == 0.0 or abs(ref_val) < 50.0
            )
            field_results.append({
                "field": fname,
                "type": "eligibility" if is_binary else "amount",
                "predicted": None,
                "reference": gold[i],
                "correct": False,
                "zero_baseline_correct": zero_correct,
                "rescue": False,
                "harm": zero_correct,
                "normalized_magnitude_error": (
                    10 if not is_binary else None),
            })
        return {
            "schema_valid": False,
            "fields": field_results,
            "agreement": 0.0,
            "n_correct": 0,
            "n_fields": n,
        }

    field_results = []
    n_correct = 0

    for i, fname in enumerate(fields):
        ref_val = gold[i]
        pred_val = parsed_array[i]

        correct, details = score_r2_2_field(fname, pred_val, ref_val)
        if correct:
            n_correct += 1

        is_binary = "eligible" in fname
        if is_binary:
            zero_correct = (float(ref_val) == 0.0)
        else:
            zero_correct = (float(ref_val) == 0.0) or (abs(float(ref_val)) < 50.0)

        rescue = (not zero_correct) and correct
        harm = zero_correct and (not correct)

        field_results.append({
            "field": fname,
            "type": details["type"],
            "predicted": pred_val,
            "reference": ref_val,
            "correct": correct,
            "zero_baseline_correct": zero_correct,
            "rescue": rescue,
            "harm": harm,
            "normalized_magnitude_error": details.get(
                "normalized_magnitude_error"),
        })

    agreement = n_correct / n if n > 0 else 0.0

    return {
        "schema_valid": True,
        "fields": field_results,
        "agreement": round(agreement, 6),
        "n_correct": n_correct,
        "n_fields": n,
    }


def gold_answer_hash(episode, workload="W-D2"):
    """SHA256 hash of ground-truth answers for provenance tracking."""
    if workload == "W-D2":
        parts = sorted(episode.get("answers", []) + episode.get("aliases", []))
    else:
        parts = sorted(
            f"{k}={v['ref_value']}" for k, v in episode.get("fields", {}).items()
        )
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


if __name__ == "__main__":
    print("=== MKQA (W-D2) R2.1 ===")
    episodes = load_mkqa(n_queries=5, languages=["en", "ja", "ko"])
    print(f"Episodes: {len(episodes)} (5 queries x 3 languages)")

    for ep in episodes[:6]:
        q = ep["question"][:60].encode("ascii", errors="replace").decode()
        print(f"\n  [{ep['lang']}] {q}")
        print(f"  Answers: {ep['answers'][:2]}")

        if ep["answers"]:
            score = score_mkqa(ep["answers"][0], ep)
            print(f"  Gold score: {score}")
        score_wrong = score_mkqa("definitely wrong answer", ep)
        print(f"  Wrong score: {score_wrong}")

    print("\n=== PolicyBench (W-D3) R2.1 ===")
    try:
        pb_eps = load_policybench(n_households=3)
        print(f"Households: {len(pb_eps)}")
        for ep in pb_eps[:2]:
            print(f"\n  Scenario: {ep['scenario_id']}")
            print(f"  Fields: {len(ep['fields'])}")
            print(f"  Prompt (first 200): "
                  f"{format_policybench_prompt(ep)[:200]}...")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
