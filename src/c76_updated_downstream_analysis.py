"""
c76_updated_downstream_analysis.py

Loads three result files, creates a unified 7-model table,
computes Pearson correlations, prints a summary, and saves to JSON.
"""

import json
import os
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# 1. Load result files
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

with open(os.path.join(RESULTS_DIR, "c68_5model_arc_easy_cti_benchmark.json"), "r") as f:
    c68 = json.load(f)

with open(os.path.join(RESULTS_DIR, "c69_pythia_2.8b_single.json"), "r") as f:
    c69 = json.load(f)

with open(os.path.join(RESULTS_DIR, "c76_qwen_0.5b_single.json"), "r") as f:
    c76 = json.load(f)

# ---------------------------------------------------------------------------
# 2. Build unified 7-model table
# ---------------------------------------------------------------------------
models = []

# 5 models from c68
for rec in c68["records"]:
    models.append({
        "model": rec["model"],
        "arc_easy": rec["benchmark_score"],
        "kappa_nearest": rec["kappa_nearest"],
        "q_knn": rec["q_knn"],
    })

# pythia-2.8b from c69
models.append({
    "model": c69["model"],
    "arc_easy": c69["score"],
    "kappa_nearest": c69["kappa"],
    "q_knn": c69["q_knn"],
})

# Qwen2.5-0.5B from c76
models.append({
    "model": c76["model"],
    "arc_easy": c76["score"],
    "kappa_nearest": c76["kappa"],
    "q_knn": c76["q_knn"],
})

# ---------------------------------------------------------------------------
# 3. Compute Pearson correlations
# ---------------------------------------------------------------------------

def compute_corr(x, y, label):
    if len(x) < 2:
        return {"label": label, "n": len(x), "r": None, "p_value": None, "note": "insufficient data"}
    r, p = pearsonr(x, y)
    return {"label": label, "n": len(x), "r": r, "p_value": p}

arc_easy_all = [m["arc_easy"] for m in models]
kappa_all = [m["kappa_nearest"] for m in models]
q_knn_all = [m["q_knn"] for m in models]

correlations = []

# (a) kappa vs arc_easy (all 7 models)
correlations.append(compute_corr(kappa_all, arc_easy_all, "kappa vs arc_easy (all 7)"))

# (b) q_knn vs arc_easy (all 7 models)
correlations.append(compute_corr(q_knn_all, arc_easy_all, "q_knn vs arc_easy (all 7)"))

# (c) q_knn vs arc_easy (excluding pythia-2.8b and Qwen due to anomalous kappa)
filtered = [m for m in models if "pythia-2.8b" not in m["model"] and "Qwen2.5-0.5B" not in m["model"]]
arc_easy_f = [m["arc_easy"] for m in filtered]
q_knn_f = [m["q_knn"] for m in filtered]
correlations.append(compute_corr(q_knn_f, arc_easy_f, "q_knn vs arc_easy (excl. pythia-2.8b & Qwen)"))

# (d) Pythia family only (160m, 410m, 1b, 2.8b): q_knn vs arc_easy
pythia_family = [m for m in models if "pythia" in m["model"]]
arc_easy_p = [m["arc_easy"] for m in pythia_family]
q_knn_p = [m["q_knn"] for m in pythia_family]
correlations.append(compute_corr(q_knn_p, arc_easy_p, "q_knn vs arc_easy (Pythia family only)"))

# ---------------------------------------------------------------------------
# 4. Print clear summary table
# ---------------------------------------------------------------------------
print("=" * 70)
print("Unified 7-Model Downstream Analysis")
print("=" * 70)
print(f"\n{'Model':<45} {'arc_easy':>10} {'kappa_nearest':>14} {'q_knn':>10}")
print("-" * 70)
for m in models:
    print(f"{m['model']:<45} {m['arc_easy']:>10.4f} {m['kappa_nearest']:>14.4f} {m['q_knn']:>10.4f}")

print("\n" + "=" * 70)
print("Pearson Correlations")
print("=" * 70)
for corr in correlations:
    if corr["r"] is None:
        print(f"\n{corr['label']}")
        print(f"  n={corr['n']}  -> {corr['note']}")
    else:
        print(f"\n{corr['label']}")
        print(f"  n={corr['n']}  r={corr['r']:.4f}  p={corr['p_value']:.4g}")

# ---------------------------------------------------------------------------
# 5. Save to JSON
# ---------------------------------------------------------------------------
output = {
    "models": models,
    "correlations": correlations,
}

out_path = os.path.join(RESULTS_DIR, "c76_updated_downstream_analysis.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved results to: {out_path}")
