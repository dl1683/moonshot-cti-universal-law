#!/usr/bin/env python -u
"""
Generation Law Analysis with Normalized Kappa (Cai-Jiang Theory).

Key insight: raw kappa conflates dimensional scaling with learned structure.
kappa_random -> sqrt(2) as d -> inf (concentration of measure).
Normalizing kappa_norm = kappa / kappa_random removes d and V dependence,
isolating the learned geometric structure.

Tests:
  1. Fixed-V Pile PPL (10 models: 5 Pythia + 5 Mamba-1) — published values
  2. Fixed-V WikiText-103 PPL (6 models: 5 Pythia + GPT-2) — computed
  3. Cross-V WikiText-103 PPL with raw vs normalized kappa (16 models)
  4. Partial correlations controlling for model size
  5. Architecture-independence test (Pythia vs Mamba on same regression line)
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
KAPPA_FILE = RESULTS / "cti_generation_kappa.json"
PPL_FILE = RESULTS / "cti_generation_ppl.json"
OUT_FILE = RESULTS / "cti_generation_normalized.json"

# Published Pile PPL from Mamba paper Table 3 (Gu & Dao 2023)
# All evaluated on Pile validation, 2048 context, 300B training tokens
PILE_PPL = {
    "pythia-160m": 29.64,
    "pythia-410m": 9.95,
    "pythia-1b":   7.82,
    "pythia-1.4b": 7.51,
    "pythia-2.8b": 6.73,
    "mamba-130m":  10.56,
    "mamba-370m":  8.28,
    "mamba-790m":  7.33,
    "mamba-1.4b":  6.80,
    "mamba-2.8b":  6.22,
}


def partial_corr(x, y, z):
    """Partial Pearson correlation of x and y controlling for z."""
    x, y, z = np.array(x), np.array(y), np.array(z)
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return pearsonr(rx, ry)


def loao_cv(kappa, log_ppl):
    """Leave-one-architecture-out: leave one model out, refit, predict."""
    n = len(kappa)
    errors = []
    for i in range(n):
        k_train = np.delete(kappa, i)
        p_train = np.delete(log_ppl, i)
        slope, intercept = np.polyfit(k_train, p_train, 1)
        pred = slope * kappa[i] + intercept
        errors.append(abs(pred - log_ppl[i]))
    return float(np.mean(errors)), errors


def analyze_group(name, keys, kappa_data, ppl_dict, use_pile=False):
    """Analyze a group of models."""
    kappas, log_ppls, params, d_models, names = [], [], [], [], []
    kappas_norm = []

    for k in keys:
        if k not in kappa_data or 'kappa_bar' not in kappa_data[k]:
            continue
        if use_pile:
            if k not in PILE_PPL:
                continue
            ppl = PILE_PPL[k]
        else:
            if k not in ppl_dict or 'ppl' not in ppl_dict[k]:
                continue
            ppl = ppl_dict[k]['ppl']

        kd = kappa_data[k]
        kappas.append(kd['kappa_bar'])
        kappas_norm.append(kd['kappa_bar'] / kd['kappa_random_mean'])
        log_ppls.append(np.log(ppl))
        params.append(kd.get('params_M', 0))
        d_models.append(kd['d_model'])
        names.append(kd.get('model', k))

    if len(kappas) < 3:
        return {"name": name, "n": len(kappas), "error": "too few models"}

    kappas = np.array(kappas)
    kappas_norm = np.array(kappas_norm)
    log_ppls = np.array(log_ppls)
    params = np.array(params)
    d_models = np.array(d_models)

    # Raw kappa analysis
    r_raw, p_raw = pearsonr(kappas, log_ppls)
    slope_raw, intercept_raw = np.polyfit(kappas, log_ppls, 1)

    # Normalized kappa analysis
    r_norm, p_norm = pearsonr(kappas_norm, log_ppls)
    slope_norm, intercept_norm = np.polyfit(kappas_norm, log_ppls, 1)

    # LOAO
    loao_mae_raw, _ = loao_cv(kappas, log_ppls)
    loao_mae_norm, _ = loao_cv(kappas_norm, log_ppls)

    result = {
        "name": name,
        "n": len(kappas),
        "models": list(names),
        "raw_kappa": {
            "r": float(r_raw),
            "p": float(p_raw),
            "slope": float(slope_raw),
            "intercept": float(intercept_raw),
            "alpha_gen": float(-slope_raw),
            "loao_mae": float(loao_mae_raw),
        },
        "normalized_kappa": {
            "r": float(r_norm),
            "p": float(p_norm),
            "slope": float(slope_norm),
            "intercept": float(intercept_norm),
            "alpha_gen": float(-slope_norm),
            "loao_mae": float(loao_mae_norm),
        },
        "per_model": [
            {
                "key": keys[i] if i < len(keys) else "?",
                "model": names[i],
                "kappa": float(kappas[i]),
                "kappa_norm": float(kappas_norm[i]),
                "log_ppl": float(log_ppls[i]),
                "params_M": int(params[i]),
                "d_model": int(d_models[i]),
            }
            for i in range(len(kappas))
        ],
    }

    # Partial correlations (if enough variation)
    if len(set(params)) > 2:
        log_params = np.log(params + 1)
        r_partial_params_raw, p_partial_params_raw = partial_corr(kappas, log_ppls, log_params)
        r_partial_params_norm, p_partial_params_norm = partial_corr(kappas_norm, log_ppls, log_params)
        result["partial_corr_controlling_params"] = {
            "raw": {"r": float(r_partial_params_raw), "p": float(p_partial_params_raw)},
            "norm": {"r": float(r_partial_params_norm), "p": float(p_partial_params_norm)},
        }

    if len(set(d_models)) > 2:
        log_d = np.log(d_models)
        r_partial_d_raw, p_partial_d_raw = partial_corr(kappas, log_ppls, log_d)
        r_partial_d_norm, p_partial_d_norm = partial_corr(kappas_norm, log_ppls, log_d)
        result["partial_corr_controlling_d_model"] = {
            "raw": {"r": float(r_partial_d_raw), "p": float(p_partial_d_raw)},
            "norm": {"r": float(r_partial_d_norm), "p": float(p_partial_d_norm)},
        }

    return result


def architecture_independence_test(keys, kappa_data, arch_groups):
    """Test whether different architectures lie on the same regression line.

    Uses F-test for coincidence of regression lines (ANCOVA).
    """
    all_kappa, all_log_ppl, all_arch = [], [], []

    for k in keys:
        if k not in kappa_data or 'kappa_bar' not in kappa_data[k]:
            continue
        if k not in PILE_PPL:
            continue
        kd = kappa_data[k]
        arch = kd.get('arch', 'unknown')
        all_kappa.append(kd['kappa_bar'])
        all_log_ppl.append(np.log(PILE_PPL[k]))
        all_arch.append(arch)

    all_kappa = np.array(all_kappa)
    all_log_ppl = np.array(all_log_ppl)
    all_arch = np.array(all_arch)

    # Full model (separate slopes and intercepts per architecture)
    unique_archs = sorted(set(all_arch))
    if len(unique_archs) < 2:
        return {"error": "need at least 2 architectures"}

    # SSR for pooled model (one line)
    slope_pooled, intercept_pooled = np.polyfit(all_kappa, all_log_ppl, 1)
    resid_pooled = all_log_ppl - (slope_pooled * all_kappa + intercept_pooled)
    ssr_pooled = float(np.sum(resid_pooled**2))

    # SSR for separate models (one line per architecture)
    ssr_separate = 0.0
    arch_results = {}
    for arch in unique_archs:
        mask = all_arch == arch
        k_arch = all_kappa[mask]
        p_arch = all_log_ppl[mask]
        if len(k_arch) < 2:
            continue
        s, i = np.polyfit(k_arch, p_arch, 1)
        resid = p_arch - (s * k_arch + i)
        ssr_separate += float(np.sum(resid**2))
        r_arch, p_arch_val = pearsonr(k_arch, p_arch) if len(k_arch) >= 3 else (float('nan'), float('nan'))
        arch_results[arch] = {
            "n": int(mask.sum()),
            "slope": float(s),
            "intercept": float(i),
            "r": float(r_arch),
            "p": float(p_arch_val),
            "alpha_gen": float(-s),
        }

    n = len(all_kappa)
    p_full = 2 * len(unique_archs)  # 2 params per arch
    p_reduced = 2  # 1 slope + 1 intercept
    df_num = p_full - p_reduced
    df_den = n - p_full

    if df_den > 0 and ssr_separate > 0:
        f_stat = ((ssr_pooled - ssr_separate) / df_num) / (ssr_separate / df_den)
        f_p = 1 - stats.f.cdf(f_stat, df_num, df_den)
    else:
        f_stat = float('nan')
        f_p = float('nan')

    return {
        "pooled": {
            "slope": float(slope_pooled),
            "intercept": float(intercept_pooled),
            "ssr": ssr_pooled,
            "alpha_gen": float(-slope_pooled),
        },
        "per_architecture": arch_results,
        "f_test": {
            "f_stat": float(f_stat),
            "p_value": float(f_p),
            "df_numerator": int(df_num),
            "df_denominator": int(df_den),
            "interpretation": "p > 0.05 means architectures share the same line (GOOD)"
        },
    }


def main():
    with open(KAPPA_FILE) as f:
        kappa_data = json.load(f)
    with open(PPL_FILE) as f:
        ppl_data = json.load(f)

    results = {}

    # ===== Analysis 1: Fixed-V, Pile PPL (10 models) =====
    print("\n" + "="*70)
    print("  ANALYSIS 1: Fixed-V, Pile PPL (Pythia + Mamba-1)")
    print("="*70)

    pile_keys = list(PILE_PPL.keys())
    a1 = analyze_group("fixed_v_pile", pile_keys, kappa_data, ppl_data, use_pile=True)
    results["fixed_v_pile"] = a1
    print(f"  n={a1['n']}")
    print(f"  Raw kappa:  r={a1['raw_kappa']['r']:.4f}, p={a1['raw_kappa']['p']:.6f}")
    print(f"              alpha_gen={a1['raw_kappa']['alpha_gen']:.4f}")
    print(f"  Norm kappa: r={a1['normalized_kappa']['r']:.4f}, p={a1['normalized_kappa']['p']:.6f}")
    print(f"              alpha_gen={a1['normalized_kappa']['alpha_gen']:.4f}")
    if 'partial_corr_controlling_params' in a1:
        pc = a1['partial_corr_controlling_params']
        print(f"  Partial r(kappa, PPL | params):")
        print(f"    Raw:  r={pc['raw']['r']:.4f}, p={pc['raw']['p']:.4f}")
        print(f"    Norm: r={pc['norm']['r']:.4f}, p={pc['norm']['p']:.4f}")

    # Per-model table
    print(f"\n  {'Model':<18} {'kappa':>8} {'k_norm':>8} {'log_PPL':>8} {'d':>6} {'params':>8}")
    print(f"  {'-'*60}")
    for m in sorted(a1['per_model'], key=lambda x: x['kappa']):
        print(f"  {m['model']:<18} {m['kappa']:>8.4f} {m['kappa_norm']:>8.4f} "
              f"{m['log_ppl']:>8.4f} {m['d_model']:>6} {m['params_M']:>8}")

    # ===== Analysis 2: Architecture independence (Pythia vs Mamba) =====
    print("\n" + "="*70)
    print("  ANALYSIS 2: Architecture Independence (Pythia vs Mamba)")
    print("="*70)

    arch_test = architecture_independence_test(pile_keys, kappa_data,
                                                {"transformer": "transformer", "ssm": "ssm"})
    results["architecture_independence"] = arch_test

    print(f"  Pooled: alpha_gen={arch_test['pooled']['alpha_gen']:.4f}")
    for arch, ar in arch_test['per_architecture'].items():
        print(f"  {arch}: alpha_gen={ar['alpha_gen']:.4f}, r={ar['r']:.4f}, n={ar['n']}")
    ft = arch_test['f_test']
    print(f"  F-test: F={ft['f_stat']:.4f}, p={ft['p_value']:.4f}")
    print(f"  --> {'SAME LINE (architecture-independent)' if ft['p_value'] > 0.05 else 'DIFFERENT LINES'}")

    # ===== Analysis 3: Fixed-V WikiText-103 (Pythia + GPT-2) =====
    print("\n" + "="*70)
    print("  ANALYSIS 3: Fixed-V, WikiText-103 (Pythia + GPT-2)")
    print("="*70)

    wt_fixed_keys = ["pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b",
                     "pythia-2.8b", "gpt2"]
    a3 = analyze_group("fixed_v_wikitext", wt_fixed_keys, kappa_data, ppl_data)
    results["fixed_v_wikitext"] = a3
    print(f"  n={a3['n']}")
    print(f"  Raw kappa:  r={a3['raw_kappa']['r']:.4f}, p={a3['raw_kappa']['p']:.6f}")
    print(f"  Norm kappa: r={a3['normalized_kappa']['r']:.4f}, p={a3['normalized_kappa']['p']:.6f}")

    # ===== Analysis 4: Cross-V WikiText-103 (all models with WT-103 PPL) =====
    print("\n" + "="*70)
    print("  ANALYSIS 4: Cross-V, WikiText-103 (all models)")
    print("="*70)

    # All models with both kappa and WikiText-103 PPL (exclude LFM2.5 anomalous)
    cross_v_keys = [k for k in kappa_data
                    if 'kappa_bar' in kappa_data[k]
                    and k in ppl_data and 'ppl' in ppl_data[k]
                    and k != 'lfm2.5-1.2b']  # exclude anomalous PPL
    a4 = analyze_group("cross_v_wikitext", cross_v_keys, kappa_data, ppl_data)
    results["cross_v_wikitext"] = a4
    print(f"  n={a4['n']}")
    print(f"  Raw kappa:  r={a4['raw_kappa']['r']:.4f}, p={a4['raw_kappa']['p']:.6f}")
    print(f"  Norm kappa: r={a4['normalized_kappa']['r']:.4f}, p={a4['normalized_kappa']['p']:.6f}")
    if 'partial_corr_controlling_params' in a4:
        pc = a4['partial_corr_controlling_params']
        print(f"  Partial r(kappa, PPL | params):")
        print(f"    Raw:  r={pc['raw']['r']:.4f}, p={pc['raw']['p']:.4f}")
        print(f"    Norm: r={pc['norm']['r']:.4f}, p={pc['norm']['p']:.4f}")
    if 'partial_corr_controlling_d_model' in a4:
        pc = a4['partial_corr_controlling_d_model']
        print(f"  Partial r(kappa, PPL | d_model):")
        print(f"    Raw:  r={pc['raw']['r']:.4f}, p={pc['raw']['p']:.4f}")
        print(f"    Norm: r={pc['norm']['r']:.4f}, p={pc['norm']['p']:.4f}")

    # Per-model table
    print(f"\n  {'Model':<20} {'kappa':>7} {'k_norm':>7} {'k_rand':>7} {'logPPL':>7} {'d':>5} {'V':>7}")
    print(f"  {'-'*65}")
    for m in sorted(a4['per_model'], key=lambda x: x['kappa_norm']):
        k_rand = m['kappa'] / m['kappa_norm'] if m['kappa_norm'] > 0 else 0
        v = kappa_data.get(cross_v_keys[a4['per_model'].index(m)], {}).get('V', 0)
        print(f"  {m['model']:<20} {m['kappa']:>7.4f} {m['kappa_norm']:>7.4f} "
              f"{k_rand:>7.4f} {m['log_ppl']:>7.4f} {m['d_model']:>5} {v:>7}")

    # ===== Analysis 5: Cross-V 2-parameter model with normalized kappa =====
    print("\n" + "="*70)
    print("  ANALYSIS 5: Cross-V 2-Parameter Model (kappa_norm + log(V))")
    print("="*70)

    kappas_n, log_ppls_5, log_vs = [], [], []
    names_5 = []
    for k in cross_v_keys:
        if 'kappa_bar' not in kappa_data[k]:
            continue
        if k not in ppl_data or 'ppl' not in ppl_data[k]:
            continue
        kd = kappa_data[k]
        kappas_n.append(kd['kappa_bar'] / kd['kappa_random_mean'])
        log_ppls_5.append(np.log(ppl_data[k]['ppl']))
        log_vs.append(np.log(kd['V'] - 1))
        names_5.append(kd.get('model', k))

    kappas_n = np.array(kappas_n)
    log_ppls_5 = np.array(log_ppls_5)
    log_vs = np.array(log_vs)

    if len(kappas_n) >= 5:
        # 2-param fit: log(PPL) = -alpha * kappa_norm + beta * log(V-1) + C
        X = np.column_stack([kappas_n, log_vs, np.ones(len(kappas_n))])
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, log_ppls_5, rcond=None)
        alpha_2p = -coeffs[0]
        beta_2p = coeffs[1]
        C_2p = coeffs[2]
        preds = X @ coeffs
        ss_res = np.sum((log_ppls_5 - preds)**2)
        ss_tot = np.sum((log_ppls_5 - np.mean(log_ppls_5))**2)
        R2_2p = 1 - ss_res / ss_tot

        results["cross_v_2param_normalized"] = {
            "n": len(kappas_n),
            "alpha_gen": float(alpha_2p),
            "beta": float(beta_2p),
            "C": float(C_2p),
            "R2": float(R2_2p),
            "residual_std": float(np.std(log_ppls_5 - preds)),
        }

        print(f"  n={len(kappas_n)}")
        print(f"  log(PPL) = -{alpha_2p:.4f} * kappa_norm + {beta_2p:.4f} * log(V-1) + {C_2p:.4f}")
        print(f"  R^2 = {R2_2p:.4f}")
        print(f"  alpha_gen = {alpha_2p:.4f}, beta = {beta_2p:.4f}")

    # ===== Analysis 6: Cai-Jiang theoretical prediction check =====
    print("\n" + "="*70)
    print("  ANALYSIS 6: Cai-Jiang Theory Validation")
    print("="*70)

    print(f"\n  Theoretical: kappa_random ~ sqrt(2) * (1 - A * sqrt(log(V)/d))")
    print(f"  sqrt(2) = {np.sqrt(2):.6f}")
    print(f"\n  {'Model':<18} {'d':>5} {'V':>7} {'k_rand':>7} {'theory':>7} {'err%':>6}")
    print(f"  {'-'*55}")

    cai_data = []
    for k, kd in sorted(kappa_data.items()):
        if 'kappa_random_mean' not in kd:
            continue
        d = kd['d_model']
        V = kd['V']
        k_rand = kd['kappa_random_mean']
        # Fit A from data
        cai_data.append((d, V, k_rand))

    # Fit A parameter
    if cai_data:
        ds, Vs, k_rands = zip(*cai_data)
        ds, Vs, k_rands = np.array(ds), np.array(Vs), np.array(k_rands)
        # kappa_random = sqrt(2) * (1 - A * sqrt(log(V)/d))
        # => A = (1 - kappa_random/sqrt(2)) / sqrt(log(V)/d)
        ratios = np.sqrt(np.log(Vs) / ds)
        deficits = 1 - k_rands / np.sqrt(2)
        A_values = deficits / ratios
        A_fit = float(np.mean(A_values))
        print(f"  Fitted A = {A_fit:.4f} (std={np.std(A_values):.4f})")

        for k, kd in sorted(kappa_data.items()):
            if 'kappa_random_mean' not in kd:
                continue
            d = kd['d_model']
            V = kd['V']
            k_rand = kd['kappa_random_mean']
            k_theory = np.sqrt(2) * (1 - A_fit * np.sqrt(np.log(V) / d))
            err_pct = 100 * (k_theory - k_rand) / k_rand
            name = kd.get('model', k)[:18]
            print(f"  {name:<18} {d:>5} {V:>7} {k_rand:>7.4f} {k_theory:>7.4f} {err_pct:>+6.2f}")

        results["cai_jiang_validation"] = {
            "A_fit": A_fit,
            "A_std": float(np.std(A_values)),
            "n_models": len(cai_data),
            "interpretation": "kappa_random = sqrt(2) * (1 - A * sqrt(log(V)/d)), A calibrated from data",
        }

    # ===== Summary =====
    print("\n" + "="*70)
    print("  SUMMARY: Does Normalized Kappa Improve Predictions?")
    print("="*70)

    for label in ["fixed_v_pile", "fixed_v_wikitext", "cross_v_wikitext"]:
        if label not in results or 'raw_kappa' not in results[label]:
            continue
        r = results[label]
        print(f"\n  {label} (n={r['n']}):")
        print(f"    Raw kappa:  r={r['raw_kappa']['r']:.4f} (p={r['raw_kappa']['p']:.4f})")
        print(f"    Norm kappa: r={r['normalized_kappa']['r']:.4f} (p={r['normalized_kappa']['p']:.4f})")
        delta_r = abs(r['normalized_kappa']['r']) - abs(r['raw_kappa']['r'])
        print(f"    Delta |r|: {delta_r:+.4f} {'IMPROVED' if delta_r > 0 else 'DEGRADED'}")

    # Architecture independence summary
    if 'architecture_independence' in results:
        ai = results['architecture_independence']
        ft = ai['f_test']
        print(f"\n  Architecture independence: F={ft['f_stat']:.3f}, p={ft['p_value']:.3f}")
        print(f"    {'PASS: Pythia and Mamba share the same kappa-PPL line' if ft['p_value'] > 0.05 else 'FAIL: Different lines'}")

    # Save
    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
