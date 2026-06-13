"""
C84 3-task cross-validation analysis for Pythia family.
Computes Pearson, Spearman, and Kendall correlations between q_knn
and downstream task accuracies (arc_easy, hellaswag, winogrande).
"""

import json
import os
from scipy.stats import pearsonr, spearmanr, kendalltau

# ---------------------------------------------------------------------------
# 1. Unified Pythia-family table
# ---------------------------------------------------------------------------
DATA = {
    "pythia-160m":  {"q_knn": 0.613, "arc_easy": 0.437, "hellaswag": 0.284, "winogrande": 0.513},
    "pythia-410m":  {"q_knn": 0.667, "arc_easy": 0.520, "hellaswag": 0.338, "winogrande": 0.536},
    "pythia-1b":    {"q_knn": 0.720, "arc_easy": 0.570, "hellaswag": 0.378, "winogrande": 0.537},
    "pythia-2.8b":  {"q_knn": 0.773, "arc_easy": 0.642, "hellaswag": 0.453, "winogrande": 0.600},
}

MODELS = list(DATA.keys())
q_knn_vals = [DATA[m]["q_knn"] for m in MODELS]
arc_easy_vals = [DATA[m]["arc_easy"] for m in MODELS]
hellaswag_vals = [DATA[m]["hellaswag"] for m in MODELS]
winogrande_vals = [DATA[m]["winogrande"] for m in MODELS]
average_3task = [(a + h + w) / 3.0 for a, h, w in zip(arc_easy_vals, hellaswag_vals, winogrande_vals)]

# ---------------------------------------------------------------------------
# 2. Correlation helpers
# ---------------------------------------------------------------------------
def corr_summary(x, y, label):
    """Return dict with Pearson, Spearman, Kendall for x vs y."""
    pr, pr_p = pearsonr(x, y)
    sr, sr_p = spearmanr(x, y)
    kt, kt_p = kendalltau(x, y)
    return {
        "label": label,
        "n": len(x),
        "pearson_r": round(pr, 6),
        "pearson_pvalue": round(pr_p, 6),
        "spearman_rho": round(sr, 6),
        "spearman_pvalue": round(sr_p, 6),
        "kendall_tau": round(kt, 6),
        "kendall_pvalue": round(kt_p, 6),
    }

results = {
    "models": MODELS,
    "data": DATA,
    "correlations": [
        corr_summary(q_knn_vals, arc_easy_vals, "q_knn vs arc_easy"),
        corr_summary(q_knn_vals, hellaswag_vals, "q_knn vs hellaswag"),
        corr_summary(q_knn_vals, winogrande_vals, "q_knn vs winogrande"),
        corr_summary(q_knn_vals, average_3task, "q_knn vs avg(3 tasks)"),
    ]
}

# ---------------------------------------------------------------------------
# 3. Pretty print
# ---------------------------------------------------------------------------
print("=" * 70)
print("C84 3-Task Cross-Validation — Pythia Family")
print("=" * 70)

print("\n  Model        q_knn   arc_easy  hellaswag  winogrande")
print("  " + "-" * 52)
for m in MODELS:
    d = DATA[m]
    print(f"  {m:<12} {d['q_knn']:.3f}   {d['arc_easy']:.3f}     {d['hellaswag']:.3f}      {d['winogrande']:.3f}")

print("\n" + "-" * 70)
print("Correlation Metrics (n = 4)")
print("-" * 70)
print(f"{'Comparison':<25} {'Pearson r':>12} {'Spearman ρ':>12} {'Kendall τ':>12}")
print("  " + "-" * 62)
for c in results["correlations"]:
    print(f"  {c['label']:<23} {c['pearson_r']:>10.4f}   {c['spearman_rho']:>10.4f}   {c['kendall_tau']:>10.4f}")

print("\n" + "-" * 70)
print("P-values")
print("-" * 70)
print(f"{'Comparison':<25} {'Pearson p':>12} {'Spearman p':>12} {'Kendall p':>12}")
print("  " + "-" * 62)
for c in results["correlations"]:
    print(f"  {c['label']:<23} {c['pearson_pvalue']:>10.4f}   {c['spearman_pvalue']:>10.4f}   {c['kendall_pvalue']:>10.4f}")

# ---------------------------------------------------------------------------
# 4. Save JSON
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "c84_3task_cross_validation.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[Saved] {out_path}")
