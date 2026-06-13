#!/usr/bin/env python3
"""C69: Comprehensive downstream benchmark analysis across CTI experiments."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, ttest_ind


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

FILES = {
    "c65_arc_easy_3": RESULTS_DIR / "c65_multi_model_cti_benchmark.json",
    "c66_hellaswag_3": RESULTS_DIR / "c66_hellaswag_cti_benchmark.json",
    "c67_arc_easy_4": RESULTS_DIR / "c67_4model_arc_easy_cti_benchmark.json",
    "c68_arc_easy_5": RESULTS_DIR / "c68_5model_arc_easy_cti_benchmark.json",
    "c69_arc_easy_pythia": RESULTS_DIR / "c69_pythia_2.8b_single.json",
}

MODEL_INFO = {
    "gpt2": {"family": "gpt2", "size_m": 117, "hidden_dim": 768},
    "EleutherAI/pythia-160m": {"family": "pythia", "size_m": 160, "hidden_dim": 768},
    "EleutherAI/gpt-neo-125m": {"family": "gpt-neo", "size_m": 125, "hidden_dim": 768},
    "EleutherAI/pythia-410m": {"family": "pythia", "size_m": 410, "hidden_dim": 1024},
    "EleutherAI/pythia-1b": {"family": "pythia", "size_m": 1000, "hidden_dim": 2048},
    "EleutherAI/pythia-2.8b": {"family": "pythia", "size_m": 2800, "hidden_dim": 2560},
}


# =============================================================================
# Helpers
# =============================================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        print(f"  WARNING: {path.name} not found, skipping.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_records(data: Any, source: str) -> List[Dict[str, Any]]:
    """Normalize various JSON layouts into a list of record dicts."""
    if data is None:
        return []
    records: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            records = data["records"]
        elif "model" in data:
            # Single-record dict (e.g., c69)
            records = [data]
        else:
            print(f"  WARNING: Unrecognized dict structure in {source}.")
            return []
    elif isinstance(data, list):
        records = data
    else:
        print(f"  WARNING: Unrecognized data type in {source}: {type(data)}")
        return []

    normalized: List[Dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        norm: Dict[str, Any] = {
            "model": r.get("model"),
            "status": r.get("status", "success"),
            "benchmark_score": r.get("benchmark_score") if "benchmark_score" in r else r.get("score"),
            "kappa_nearest": r.get("kappa_nearest") if "kappa_nearest" in r else r.get("kappa"),
            "q_knn": r.get("q_knn"),
        }
        normalized.append(norm)
    return normalized


def safe_pearsonr(x: List[float], y: List[float]) -> Dict[str, Any]:
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if len(x_arr) < 2 or len(y_arr) < 2 or len(np.unique(x_arr)) <= 1 or len(np.unique(y_arr)) <= 1:
        return {"r": None, "p_value": None, "n": len(x_arr), "valid": False}
    r, p = pearsonr(x_arr, y_arr)
    return {"r": float(r), "p_value": float(p), "n": len(x_arr), "valid": True}


def safe_ttest(x: List[float], y: List[float]) -> Dict[str, Any]:
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if len(x_arr) < 2 or len(y_arr) < 2:
        return {"t_statistic": None, "p_value": None, "n1": len(x_arr), "n2": len(y_arr), "valid": False}
    t, p = ttest_ind(x_arr, y_arr, equal_var=False)
    return {"t_statistic": float(t), "p_value": float(p), "n1": len(x_arr), "n2": len(y_arr), "valid": True}


# =============================================================================
# Main
# =============================================================================

def main() -> Dict[str, Any]:
    print("=" * 70)
    print("C69: DOWNSTREAM BENCHMARK ANALYSIS")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    # Load raw data
    # -------------------------------------------------------------------------
    raw_data: Dict[str, List[Dict[str, Any]]] = {}
    for key, path in FILES.items():
        print(f"Loading {key} ...")
        data = load_json(path)
        raw_data[key] = extract_records(data, key)
    print()

    # -------------------------------------------------------------------------
    # Build unified table
    # -------------------------------------------------------------------------
    table: Dict[str, Dict[str, Any]] = {}
    for model_name, info in MODEL_INFO.items():
        table[model_name] = {
            "model": model_name,
            "family": info["family"],
            "size_m": info["size_m"],
            "hidden_dim": info["hidden_dim"],
            "arc_easy": None,
            "hellaswag": None,
            "kappa_nearest": None,
            "q_knn": None,
        }

    for key, records in raw_data.items():
        for r in records:
            model = r.get("model")
            if model not in table:
                continue
            if key == "c66_hellaswag_3":
                if r.get("benchmark_score") is not None:
                    table[model]["hellaswag"] = r["benchmark_score"]
            else:
                if r.get("benchmark_score") is not None:
                    table[model]["arc_easy"] = r["benchmark_score"]

            if r.get("kappa_nearest") is not None and table[model]["kappa_nearest"] is None:
                table[model]["kappa_nearest"] = r["kappa_nearest"]
            if r.get("q_knn") is not None and table[model]["q_knn"] is None:
                table[model]["q_knn"] = r["q_knn"]

    table_list = list(table.values())

    # -------------------------------------------------------------------------
    # Print unified table
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("UNIFIED MODEL TABLE")
    print("-" * 70)
    header = f"{'Model':<25} {'Family':<10} {'Size':<8} {'arc_easy':<12} {'hellaswag':<12} {'kappa':<12} {'q_knn':<12}"
    print(header)
    print("-" * 70)
    for row in table_list:
        arc_str = f"{row['arc_easy']:.4f}" if row["arc_easy"] is not None else "N/A"
        hell_str = f"{row['hellaswag']:.4f}" if row["hellaswag"] is not None else "N/A"
        kappa_str = f"{row['kappa_nearest']:.4f}" if row["kappa_nearest"] is not None else "N/A"
        q_str = f"{row['q_knn']:.4f}" if row["q_knn"] is not None else "N/A"
        size_str = f"{row['size_m']}M" if row["size_m"] < 1000 else f"{row['size_m']/1000:.1f}B"
        print(f"{row['model']:<25} {row['family']:<10} {size_str:<8} {arc_str:<12} {hell_str:<12} {kappa_str:<12} {q_str:<12}")
    print()

    # -------------------------------------------------------------------------
    # Compute correlations
    # -------------------------------------------------------------------------
    def get_xy(rows: List[Dict[str, Any]], x_key: str, y_key: str, filter_fn=None) -> Tuple[List[float], List[float]]:
        xs, ys = [], []
        for row in rows:
            if filter_fn and not filter_fn(row):
                continue
            x = row.get(x_key)
            y = row.get(y_key)
            if x is not None and y is not None:
                xs.append(float(x))
                ys.append(float(y))
        return xs, ys

    corr_results = {}

    xs, ys = get_xy(table_list, "kappa_nearest", "arc_easy")
    corr_results["all_6_kappa_vs_arc_easy"] = safe_pearsonr(xs, ys)

    xs, ys = get_xy(table_list, "kappa_nearest", "arc_easy", lambda r: r["family"] == "pythia")
    corr_results["pythia_only_kappa_vs_arc_easy"] = safe_pearsonr(xs, ys)

    xs, ys = get_xy(table_list, "q_knn", "arc_easy")
    corr_results["all_6_q_knn_vs_arc_easy"] = safe_pearsonr(xs, ys)

    xs, ys = get_xy(table_list, "q_knn", "kappa_nearest")
    corr_results["all_6_q_knn_vs_kappa"] = safe_pearsonr(xs, ys)

    print("-" * 70)
    print("CORRELATIONS")
    print("-" * 70)
    for label, res in corr_results.items():
        valid_str = "valid" if res["valid"] else "INVALID"
        if res["valid"]:
            print(f"{label:<40}  r = {res['r']:.4f}, p = {res['p_value']:.4f}, n = {res['n']} ({valid_str})")
        else:
            print(f"{label:<40}  n = {res['n']} ({valid_str})")
    print()

    # -------------------------------------------------------------------------
    # Phase transition hypothesis
    # -------------------------------------------------------------------------
    low_kappa_models = [r for r in table_list if r["kappa_nearest"] is not None and r["kappa_nearest"] < 40]
    high_kappa_models = [r for r in table_list if r["kappa_nearest"] is not None and r["kappa_nearest"] > 40]

    low_scores = [r["arc_easy"] for r in low_kappa_models if r["arc_easy"] is not None]
    high_scores = [r["arc_easy"] for r in high_kappa_models if r["arc_easy"] is not None]

    phase_transition = {
        "threshold": 40,
        "low_kappa": {
            "models": [r["model"] for r in low_kappa_models],
            "mean_arc_easy": float(np.mean(low_scores)) if low_scores else None,
            "std_arc_easy": float(np.std(low_scores, ddof=1)) if len(low_scores) > 1 else 0.0,
            "n": len(low_scores),
        },
        "high_kappa": {
            "models": [r["model"] for r in high_kappa_models],
            "mean_arc_easy": float(np.mean(high_scores)) if high_scores else None,
            "std_arc_easy": float(np.std(high_scores, ddof=1)) if len(high_scores) > 1 else 0.0,
            "n": len(high_scores),
        },
        "t_test": safe_ttest(low_scores, high_scores),
    }

    print("-" * 70)
    print("PHASE TRANSITION HYPOTHESIS (kappa threshold = 40)")
    print("-" * 70)
    print(f"Low kappa  (< 40): n = {phase_transition['low_kappa']['n']}, mean arc_easy = {phase_transition['low_kappa']['mean_arc_easy']:.4f} ± {phase_transition['low_kappa']['std_arc_easy']:.4f}")
    print(f"  Models: {', '.join(phase_transition['low_kappa']['models'])}")
    print(f"High kappa (> 40): n = {phase_transition['high_kappa']['n']}, mean arc_easy = {phase_transition['high_kappa']['mean_arc_easy']:.4f} ± {phase_transition['high_kappa']['std_arc_easy']:.4f}")
    print(f"  Models: {', '.join(phase_transition['high_kappa']['models'])}")
    if phase_transition["t_test"]["valid"]:
        print(f"t-test: t = {phase_transition['t_test']['t_statistic']:.4f}, p = {phase_transition['t_test']['p_value']:.4f}")
    else:
        print("t-test: insufficient data for test.")
    print()

    # -------------------------------------------------------------------------
    # Normalized kappa analysis
    # -------------------------------------------------------------------------
    norm_records = []
    for row in table_list:
        kappa = row.get("kappa_nearest")
        d = row.get("hidden_dim")
        size_m = row.get("size_m")
        arc = row.get("arc_easy")
        if kappa is not None and d is not None and size_m is not None and size_m > 0 and arc is not None:
            kappa_norm = kappa * math.sqrt(d) / math.log(size_m)
            norm_records.append({
                "model": row["model"],
                "kappa": kappa,
                "hidden_dim": d,
                "size_m": size_m,
                "kappa_norm": kappa_norm,
                "arc_easy": arc,
            })

    norm_corr = {"r": None, "p_value": None, "n": 0, "valid": False}
    if len(norm_records) >= 2:
        x = [r["kappa_norm"] for r in norm_records]
        y = [r["arc_easy"] for r in norm_records]
        norm_corr = safe_pearsonr(x, y)

    normalized_kappa = {
        "formula": "kappa_norm = kappa * sqrt(hidden_dim) / log(size_m)  [natural log]",
        "records": norm_records,
        "correlation": norm_corr,
    }

    print("-" * 70)
    print("NORMALIZED KAPPA ANALYSIS")
    print("-" * 70)
    print(f"Formula: {normalized_kappa['formula']}")
    print(f"{'Model':<25} {'kappa':<12} {'d':<8} {'size_m':<10} {'kappa_norm':<14} {'arc_easy':<12}")
    print("-" * 70)
    for r in norm_records:
        size_label = f"{r['size_m']}M" if r['size_m'] < 1000 else f"{r['size_m']/1000:.1f}B"
        print(f"{r['model']:<25} {r['kappa']:<12.4f} {r['hidden_dim']:<8} {size_label:<10} {r['kappa_norm']:<14.4f} {r['arc_easy']:<12.4f}")
    if norm_corr["valid"]:
        print(f"\nCorrelation (kappa_norm vs arc_easy): r = {norm_corr['r']:.4f}, p = {norm_corr['p_value']:.4f}, n = {norm_corr['n']}")
    else:
        print(f"\nCorrelation (kappa_norm vs arc_easy): insufficient data (n = {norm_corr['n']})")
    print()

    # -------------------------------------------------------------------------
    # Save comprehensive JSON
    # -------------------------------------------------------------------------
    output = {
        "experiment": "c69_downstream_benchmark_analysis",
        "unified_table": table_list,
        "correlations": corr_results,
        "phase_transition": phase_transition,
        "normalized_kappa": normalized_kappa,
    }

    out_path = RESULTS_DIR / "c69_downstream_benchmark_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved comprehensive results to {out_path}")
    print()
    print("=" * 70)
    print("C69 ANALYSIS COMPLETE")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
