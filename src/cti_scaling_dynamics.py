"""
CTI SCALING & TRAINING DYNAMICS ANALYSIS
==========================================
E1: Pythia scaling law — kappa_nearest ~ N_params^gamma
    Using H3 n=9 kappa values for Pythia models (160m, 410m, 1b, 1.4b)
    + optionally running pythia-70m and pythia-2.8b if missing

E2: Training dynamics — does kappa_nearest (or knn_l0 proxy) lead q_norm?
    Using cti_checkpoint_sweep_all.json (4 Pythia models x 12 steps x 4 datasets)
    Analysis: lag-correlation between knn_l0 at step T vs step T+1

Output: results/cti_scaling_dynamics.json
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, linregress

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
H3_PATH = RESULTS_DIR / "cti_downstream_h3_n9.json"
CHKPT_PATH = RESULTS_DIR / "cti_checkpoint_sweep_all.json"
OUT_PATH = RESULTS_DIR / "cti_scaling_dynamics.json"

# Pythia model sizes in parameters (from HuggingFace model cards)
PYTHIA_SIZES = {
    "pythia-70m":  70426624,
    "pythia-160m": 162322944,
    "pythia-410m": 405334016,
    "pythia-1b":   1011781632,
    "pythia-1.4b": 1414647808,
    "pythia-2.8b": 2775208960,
}


def fit_power_law(x, y):
    """Fit y = a * x^gamma via log-log OLS. Returns (a, gamma, r2)."""
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept, r, p, se = linregress(log_x, log_y)
    return float(np.exp(intercept)), float(slope), float(r ** 2), float(p)


def e1_scaling_law():
    """E1: Fit kappa_nearest ~ N_params^gamma for Pythia models."""
    print("\n[E1] Pythia scaling law: kappa_nearest vs N_params", flush=True)

    # Load H3 n=9 results for Pythia kappa values
    with open(H3_PATH) as f:
        h3 = json.load(f)

    all_models = h3["H3_extended"]["models"]
    pythia_data = []
    for m in all_models:
        name = m["model"]
        short = name.replace("pythia-", "").replace("m", "m")
        # Match to size table
        size_key = f"pythia-{name.split('pythia-')[-1]}" if "pythia" in name else None
        if size_key and size_key in PYTHIA_SIZES:
            pythia_data.append({
                "model": name,
                "N_params": PYTHIA_SIZES[size_key],
                "kappa_nearest": m["kappa_nearest_final"],
                "MAP_at_10": m["map_at_10_final"],
            })
            print(f"  {name}: N={PYTHIA_SIZES[size_key]:.2e} "
                  f"kappa={m['kappa_nearest_final']:.4f} "
                  f"MAP@10={m['map_at_10_final']:.4f}", flush=True)

    if len(pythia_data) < 3:
        print("  Not enough Pythia models for scaling law", flush=True)
        return {"error": "insufficient_data", "n": len(pythia_data)}

    N_arr = np.array([d["N_params"] for d in pythia_data])
    kappa_arr = np.array([d["kappa_nearest"] for d in pythia_data])
    map_arr = np.array([d["MAP_at_10"] for d in pythia_data])

    a_kappa, gamma_kappa, r2_kappa, p_kappa = fit_power_law(N_arr, kappa_arr)
    a_map, gamma_map, r2_map, p_map = fit_power_law(N_arr, map_arr)

    print(f"\n  kappa ~ N^gamma: a={a_kappa:.4f} gamma={gamma_kappa:.4f} "
          f"R2={r2_kappa:.4f} p={p_kappa:.4f}", flush=True)
    print(f"  MAP@10 ~ N^gamma: a={a_map:.4f} gamma={gamma_map:.4f} "
          f"R2={r2_map:.4f} p={p_map:.4f}", flush=True)

    # Spearman correlation kappa vs N_params
    rho_kappa, p_rho_kappa = spearmanr(N_arr, kappa_arr)
    print(f"  Spearman rho(kappa, N): {rho_kappa:.4f} p={p_rho_kappa:.4f}", flush=True)

    return {
        "n_models": len(pythia_data),
        "models": pythia_data,
        "kappa_scaling": {
            "a": a_kappa, "gamma": gamma_kappa, "R2": r2_kappa, "p": p_kappa,
            "spearman_rho": rho_kappa, "spearman_p": p_rho_kappa,
        },
        "map_scaling": {
            "a": a_map, "gamma": gamma_map, "R2": r2_map, "p": p_map,
        },
        "interpretation": (
            f"kappa_nearest scales as N^{gamma_kappa:.3f} across {len(pythia_data)} "
            f"Pythia model sizes (R2={r2_kappa:.3f}). "
            f"{'Monotonically increasing' if gamma_kappa > 0 else 'Decreasing'} "
            f"with model size."
        ),
    }


def e2_training_dynamics():
    """E2: Does knn_l0 lead q at step T? Lag-correlation analysis."""
    print("\n[E2] Training dynamics: knn_l0 lag-correlation", flush=True)

    with open(CHKPT_PATH) as f:
        chk = json.load(f)

    results = chk["results"]
    models = chk["models"]
    datasets_list = chk["datasets"]

    lag_results = {}

    for model_name in models:
        model_results = [r for r in results if r["model"] == model_name]
        model_results.sort(key=lambda x: x["step"])

        for ds_name in datasets_list:
            # Extract final-layer knn_l0 at each step
            series = []
            for r in model_results:
                if ds_name not in r["datasets"]:
                    continue
                ds_data = r["datasets"][ds_name]
                layers = ds_data["layers"]
                # Final layer
                final_layer = str(r["num_layers"])
                if final_layer not in layers:
                    final_layer = str(max(int(k) for k in layers.keys()))
                knn = layers[final_layer]["knn_l0"]
                series.append({"step": r["step"], "knn_l0": knn})

            if len(series) < 4:
                continue

            steps = [s["step"] for s in series]
            knn_vals = np.array([s["knn_l0"] for s in series])

            # Lag-1 autocorrelation: does knn(T) predict knn(T+1)?
            # Granger-like: compare r(knn(T), knn(T+1)) vs r(knn(T), knn(T+1)) given knn(T)
            # Simple: compute lag-0 and lag-1 correlations
            x_lag0 = knn_vals[:-1]  # T=0..N-2
            y_lag1 = knn_vals[1:]   # T=1..N-1

            if len(x_lag0) < 3:
                continue

            r_lag1, p_lag1 = pearsonr(x_lag0, y_lag1)

            # Also compute the difference series (acceleration of learning)
            diff = np.diff(knn_vals)
            steps_diff = np.array(steps[1:])

            # Fit power law to step: knn(T) ~ a * log(T+1) + b
            log_steps = np.log(np.array(steps[1:]) + 1)  # avoid log(0)
            if len(log_steps) >= 3:
                slope_log, intercept_log, r_log, p_log, _ = linregress(
                    log_steps, knn_vals[1:])
            else:
                r_log, p_log, slope_log = None, None, None

            key = f"{model_name}_{ds_name}"
            lag_results[key] = {
                "model": model_name,
                "dataset": ds_name,
                "n_steps": len(series),
                "knn_series": knn_vals.tolist(),
                "steps": steps,
                "lag1_r": float(r_lag1),
                "lag1_p": float(p_lag1),
                "log_fit_slope": float(slope_log) if slope_log is not None else None,
                "log_fit_r": float(r_log) if r_log is not None else None,
                "final_knn": float(knn_vals[-1]),
                "peak_knn": float(knn_vals.max()),
                "peak_step": int(steps[np.argmax(knn_vals)]),
            }
            print(f"  {model_name.split('/')[-1]} x {ds_name}: "
                  f"final={knn_vals[-1]:.4f} peak={knn_vals.max():.4f}@step{steps[np.argmax(knn_vals)]} "
                  f"lag1_r={r_lag1:.3f}", flush=True)

    # Summary: mean lag-1 autocorrelation
    lag1_rs = [v["lag1_r"] for v in lag_results.values() if "lag1_r" in v]
    print(f"\n  Mean lag-1 autocorrelation: {np.mean(lag1_rs):.4f} "
          f"+/- {np.std(lag1_rs):.4f}", flush=True)
    print(f"  Min lag-1 r: {min(lag1_rs):.4f}", flush=True)
    print(f"  Fraction > 0.80: {np.mean([r > 0.80 for r in lag1_rs]):.2f}", flush=True)

    # Check if knn peaks before final step (non-monotone = generalization gap)
    non_monotone = []
    for v in lag_results.values():
        series = v["knn_series"]
        if v["peak_knn"] > series[-1] + 0.01:  # peak significantly above final
            non_monotone.append({
                "model": v["model"],
                "dataset": v["dataset"],
                "peak_step": v["peak_step"],
                "peak_knn": v["peak_knn"],
                "final_knn": v["final_knn"],
                "drop": v["peak_knn"] - v["final_knn"],
            })
    print(f"\n  Non-monotone series (peak > final by >0.01): {len(non_monotone)}/{len(lag_results)}", flush=True)
    for nm in non_monotone:
        print(f"    {nm['model'].split('/')[-1]} x {nm['dataset']}: "
              f"peak={nm['peak_knn']:.4f}@{nm['peak_step']} "
              f"final={nm['final_knn']:.4f} drop={nm['drop']:.4f}", flush=True)

    return {
        "n_series": len(lag_results),
        "mean_lag1_r": float(np.mean(lag1_rs)),
        "std_lag1_r": float(np.std(lag1_rs)),
        "frac_lag1_above_0.80": float(np.mean([r > 0.80 for r in lag1_rs])),
        "non_monotone_series": non_monotone,
        "per_series": lag_results,
    }


def main():
    print("CTI SCALING & TRAINING DYNAMICS", flush=True)
    print("=" * 60, flush=True)

    e1_result = e1_scaling_law()
    e2_result = e2_training_dynamics()

    result = {
        "experiment": "cti_scaling_dynamics",
        "E1_pythia_scaling": e1_result,
        "E2_training_dynamics": e2_result,
        "status": "complete",
    }

    with open(OUT_PATH, "w", encoding="ascii") as fp:
        json.dump(result, fp, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
