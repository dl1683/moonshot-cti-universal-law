"""
Compare AG News vs DBPedia q_knn correlations with downstream benchmarks.
"""
import json
import os
from scipy.stats import pearsonr, spearmanr

# Data
ag_news_qknn = [0.613, 0.667, 0.720, 0.773]
dbpedia_qknn = [0.806, 0.795, 0.828, 0.817]

tasks = {
    "arc_easy":    [0.437, 0.520, 0.570, 0.642],
    "hellaswag":   [0.284, 0.338, 0.378, 0.453],
    "winogrande":  [0.513, 0.536, 0.537, 0.600],
}

results = []

print("=" * 70)
print("AG News q_knn vs DBPedia q_knn — Downstream Correlation Comparison")
print("=" * 70)
print()
print(f"{'Task':<15} {'Metric':<10} {'AG News r':<12} {'DBPedia r':<12} {'Winner'}")
print("-" * 70)

for task_name, task_scores in tasks.items():
    # Pearson
    ag_pearson, ag_pearson_p = pearsonr(ag_news_qknn, task_scores)
    db_pearson, db_pearson_p = pearsonr(dbpedia_qknn, task_scores)

    # Spearman
    ag_spearman, ag_spearman_p = spearmanr(ag_news_qknn, task_scores)
    db_spearman, db_spearman_p = spearmanr(dbpedia_qknn, task_scores)

    results.append({
        "task": task_name,
        "ag_pearson_r": ag_pearson,
        "ag_pearson_p": ag_pearson_p,
        "db_pearson_r": db_pearson,
        "db_pearson_p": db_pearson_p,
        "ag_spearman_rho": ag_spearman,
        "ag_spearman_p": ag_spearman_p,
        "db_spearman_rho": db_spearman,
        "db_spearman_p": db_spearman_p,
    })

    for metric, ag_val, db_val in [
        ("Pearson", ag_pearson, db_pearson),
        ("Spearman", ag_spearman, db_spearman),
    ]:
        winner = "AG News" if ag_val > db_val else "DBPedia" if db_val > ag_val else "Tie"
        print(f"{task_name:<15} {metric:<10} {ag_val:>11.4f} {db_val:>11.4f}  {winner}")
    print("-" * 70)

# Overall summary
print()
print("Summary:")
ag_wins = sum(1 for r in results if r["ag_pearson_r"] > r["db_pearson_r"]) + \
          sum(1 for r in results if r["ag_spearman_rho"] > r["db_spearman_rho"])
db_wins = sum(1 for r in results if r["db_pearson_r"] > r["ag_pearson_r"]) + \
          sum(1 for r in results if r["db_spearman_rho"] > r["ag_spearman_rho"])
ties  = 6 - ag_wins - db_wins
print(f"  AG News wins: {ag_wins} / 6")
print(f"  DBPedia wins: {db_wins} / 6")
print(f"  Ties:         {ties} / 6")
print()

# Save to JSON
output_path = os.path.join(
    "C:\\Users\\devan\\OneDrive\\Desktop\\Projects\\AI Moonshots\\moonshot-cti-universal-law",
    "results",
    "c85_dataset_probe_comparison.json"
)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved detailed results to: {output_path}")
