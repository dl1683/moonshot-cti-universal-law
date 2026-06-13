"""
C79: HellaSwag downstream correlation analysis.
Builds a unified table of q_knn (from arc_easy) vs HellaSwag scores,
and computes Pearson correlations across model families.
"""

import json
import os
from scipy.stats import pearsonr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    # 1. Load result files
    c66 = load_json(os.path.join(RESULTS_DIR, "c66_hellaswag_cti_benchmark.json"))
    c77 = load_json(os.path.join(RESULTS_DIR, "c77_pythia_410m_hellaswag.json"))
    c78 = load_json(os.path.join(RESULTS_DIR, "c78_pythia_1b_hellaswag.json"))
    c79 = load_json(os.path.join(RESULTS_DIR, "c79_pythia_2.8b_hellaswag.json"))
    c68 = load_json(os.path.join(RESULTS_DIR, "c68_5model_arc_easy_cti_benchmark.json"))
    c69 = load_json(os.path.join(RESULTS_DIR, "c69_pythia_2.8b_single.json"))
    c76 = load_json(os.path.join(RESULTS_DIR, "c76_qwen_0.5b_single.json"))

    # Build HellaSwag lookup
    hellaswag_scores = {}
    for rec in c66["records"]:
        hellaswag_scores[rec["model"]] = rec["benchmark_score"]

    hellaswag_scores[c77["model"]] = c77["score"]
    hellaswag_scores[c78["model"]] = c78["score"]
    hellaswag_scores[c79["model"]] = c79["score"]

    # Build arc_easy + q_knn lookup
    arc_easy_scores = {}
    q_knn_values = {}
    for rec in c68["records"]:
        arc_easy_scores[rec["model"]] = rec["benchmark_score"]
        q_knn_values[rec["model"]] = rec["q_knn"]

    arc_easy_scores[c69["model"]] = c69["score"]
    q_knn_values[c69["model"]] = c69["q_knn"]

    arc_easy_scores[c76["model"]] = c76["score"]
    q_knn_values[c76["model"]] = c76["q_knn"]

    # 2. Build unified table
    models = [
        "gpt2",
        "EleutherAI/pythia-160m",
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-2.8b",
        "Qwen/Qwen2.5-0.5B",
    ]

    unified_table = []
    for model in models:
        row = {
            "model": model,
            "q_knn": q_knn_values.get(model),
            "arc_easy": arc_easy_scores.get(model),
            "hellaswag": hellaswag_scores.get(model),
        }
        unified_table.append(row)

    # 3. Compute Pearson correlations
    correlations = {}

    # All models with both q_knn and hellaswag (6 models; Qwen missing HellaSwag)
    valid_hellaswag = [r for r in unified_table if r["q_knn"] is not None and r["hellaswag"] is not None]
    q_knn_all_h = [r["q_knn"] for r in valid_hellaswag]
    hellaswag_all = [r["hellaswag"] for r in valid_hellaswag]
    r, p = pearsonr(q_knn_all_h, hellaswag_all)
    correlations["q_knn_vs_hellaswag_all_available"] = {
        "n": len(valid_hellaswag),
        "models": [r["model"] for r in valid_hellaswag],
        "r": float(r),
        "p_value": float(p),
        "valid": True,
        "note": "Qwen/Qwen2.5-0.5B HellaSwag not found; computed on 6 models",
    }

    # Pythia family only: q_knn vs HellaSwag
    pythia_models = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "EleutherAI/pythia-2.8b"]
    pythia_h = [r for r in unified_table if r["model"] in pythia_models and r["q_knn"] is not None and r["hellaswag"] is not None]
    q_knn_pythia_h = [r["q_knn"] for r in pythia_h]
    hellaswag_pythia = [r["hellaswag"] for r in pythia_h]
    r, p = pearsonr(q_knn_pythia_h, hellaswag_pythia)
    correlations["q_knn_vs_hellaswag_pythia_only"] = {
        "n": len(pythia_h),
        "models": [r["model"] for r in pythia_h],
        "r": float(r),
        "p_value": float(p),
        "valid": True,
    }

    # Pythia family only: q_knn vs arc_easy
    pythia_ae = [r for r in unified_table if r["model"] in pythia_models and r["q_knn"] is not None and r["arc_easy"] is not None]
    q_knn_pythia_ae = [r["q_knn"] for r in pythia_ae]
    arc_easy_pythia = [r["arc_easy"] for r in pythia_ae]
    r, p = pearsonr(q_knn_pythia_ae, arc_easy_pythia)
    correlations["q_knn_vs_arc_easy_pythia_only"] = {
        "n": len(pythia_ae),
        "models": [r["model"] for r in pythia_ae],
        "r": float(r),
        "p_value": float(p),
        "valid": True,
    }

    # All models with both q_knn and arc_easy (7 models)
    valid_arc_easy = [r for r in unified_table if r["q_knn"] is not None and r["arc_easy"] is not None]
    q_knn_all_ae = [r["q_knn"] for r in valid_arc_easy]
    arc_easy_all = [r["arc_easy"] for r in valid_arc_easy]
    r, p = pearsonr(q_knn_all_ae, arc_easy_all)
    correlations["q_knn_vs_arc_easy_all"] = {
        "n": len(valid_arc_easy),
        "models": [r["model"] for r in valid_arc_easy],
        "r": float(r),
        "p_value": float(p),
        "valid": True,
    }

    # 4. Print clear table
    print("=" * 80)
    print("C79: HellaSwag Downstream Analysis")
    print("=" * 80)
    print()
    print(f"{'Model':<30} {'q_knn':>10} {'arc_easy':>12} {'hellaswag':>12}")
    print("-" * 80)
    for row in unified_table:
        q_str = f"{row['q_knn']:.4f}" if row["q_knn"] is not None else "N/A"
        ae_str = f"{row['arc_easy']:.4f}" if row["arc_easy"] is not None else "N/A"
        h_str = f"{row['hellaswag']:.4f}" if row["hellaswag"] is not None else "N/A"
        print(f"{row['model']:<30} {q_str:>10} {ae_str:>12} {h_str:>12}")
    print("-" * 80)
    print()

    print("Pearson Correlations:")
    print("-" * 80)
    for label, corr in correlations.items():
        print(f"  {label}:")
        print(f"    n={corr['n']}, r={corr['r']:.6f}, p={corr['p_value']:.6f}")
        if "note" in corr:
            print(f"    NOTE: {corr['note']}")
    print()

    # 5. Save to JSON
    output = {
        "unified_table": unified_table,
        "correlations": correlations,
    }

    output_path = os.path.join(RESULTS_DIR, "c79_hellaswag_downstream_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
