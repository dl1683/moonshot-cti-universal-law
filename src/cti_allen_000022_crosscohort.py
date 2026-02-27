"""
CTI Cross-Cohort Replication: DANDI:000022
==========================================
Pre-registered hypothesis H_cross_cohort:
  r(kappa_nearest, logit_q) > 0.50 for >= 80% of qualifying sessions
  in DANDI:000022 (Allen Brain Observatory FC Stimulus Set -- different
  experimental cohort from DANDI:000021).

If PASS: biological result generalises across cohorts, ruling out
         lab-batch artifacts. Elevates biological validation substantially.

Protocol frozen at commit time (identical to DANDI:000021 pipeline):
- Response window: 50-250 ms post-stimulus onset
- Stimulus: natural_scenes (K=118 images if available, else detect K)
- Min quality-filtered units: 50
- Min presentations: 1000
- PCA: min(100, n_units-1) components
- LOO 1-NN accuracy
- kappa_nearest: min centroid gap / (sigma_W_global * sqrt(d_pca))
- H1 pass: r(kappa_nearest, logit_q) > 0.50
- Primary criterion: frac_pass >= 0.80
"""
import numpy as np
import h5py
import remfile
import time
import json
import sys
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "cti_allen_000022_crosscohort.json"

RESPONSE_START = 0.05
RESPONSE_END = 0.25
MIN_UNITS = 50
MIN_PRES = 1000
MIN_ACTIVE_CLASSES = 10
N_PCA = 100
H1_THRESHOLD = 0.50
PASS_FRAC_THRESHOLD = 0.80


def logit_safe(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def process_session(url, session_key):
    t0 = time.time()
    try:
        rf = remfile.File(url)
        f = h5py.File(rf, "r")
    except Exception as e:
        return None, f"open_fail: {e}"

    try:
        intervals = f["intervals"]
        # Find natural scenes key (may differ across datasets)
        ns_key = None
        for k in intervals.keys():
            if "natural_scene" in k.lower():
                ns_key = k
                break
        if ns_key is None:
            keys_list = list(intervals.keys())[:10]
            f.close()
            return None, f"no_natural_scenes_key: {keys_list}"

        stim = intervals[ns_key]
        # Try common column names for frame index
        frame_col = None
        for col in ["frame_index", "frame", "stimulus_index"]:
            if col in stim:
                frame_col = col
                break
        if frame_col is None:
            stim_keys = list(stim.keys())[:10]
            f.close()
            return None, f"no_frame_col: {stim_keys}"

        frame_idx = stim[frame_col][:]
        start_times = stim["start_time"][:]
        valid_mask = frame_idx != -1
        valid_starts = start_times[valid_mask]
        valid_frames = frame_idx[valid_mask]
        n_pres = len(valid_starts)
        if n_pres < MIN_PRES:
            f.close()
            return None, f"too_few_pres: {n_pres}"
        K = len(np.unique(valid_frames))
        if K < 10:
            f.close()
            return None, f"too_few_classes: K={K}"
    except KeyError as e:
        f.close()
        return None, f"intervals_error: {e}"

    try:
        units = f["units"]
        quality = units["quality"][:].astype(str) if "quality" in units else None
        if quality is not None:
            good_mask = np.array([q.strip() == "good" for q in quality])
        else:
            good_mask = np.ones(len(units["id"][:]), dtype=bool)
        good_idx = np.where(good_mask)[0]
        n_units_good = len(good_idx)
        if n_units_good < MIN_UNITS:
            f.close()
            return None, f"too_few_units: {n_units_good}"

        all_spike_times = units["spike_times"][:]
        all_idx = units["spike_times_index"][:]
    except Exception as e:
        f.close()
        return None, f"units_error: {e}"

    f.close()

    # Build response matrix
    n_pca_actual = min(N_PCA, n_units_good - 1)
    prev_ends = np.concatenate([[0], all_idx[:-1]])
    R = np.zeros((n_pres, n_units_good), dtype=np.float32)
    stim_starts_arr = valid_starts + RESPONSE_START
    stim_stops_arr = valid_starts + RESPONSE_END

    for j, unit_j in enumerate(good_idx):
        spikes = np.sort(
            all_spike_times[int(prev_ends[unit_j]):int(all_idx[unit_j])]
        )
        R[:, j] = (
            np.searchsorted(spikes, stim_stops_arr)
            - np.searchsorted(spikes, stim_starts_arr)
        )

    pca = PCA(n_components=n_pca_actual)
    R_pca = pca.fit_transform(R)
    d = n_pca_actual

    # LOO 1-NN via full distance matrix
    dist_mat = np.sum(
        (R_pca[:, None, :] - R_pca[None, :, :]) ** 2, axis=-1
    )
    np.fill_diagonal(dist_mat, np.inf)
    nn_idx = np.argmin(dist_mat, axis=1)
    nn_labels = valid_frames[nn_idx]
    correct = (nn_labels == valid_frames).astype(float)

    classes = np.unique(valid_frames)
    q_arr = np.array([correct[valid_frames == c].mean() for c in classes])

    # kappa_nearest: min centroid gap / (sigma_W_global * sqrt(d))
    centroids = np.array([R_pca[valid_frames == c].mean(0) for c in classes])
    Sigma_W = np.zeros((d, d))
    for c in classes:
        Xi = R_pca[valid_frames == c] - centroids[classes == c][0]
        Sigma_W += Xi.T @ Xi
    Sigma_W /= n_pres
    sigma_W_global = np.sqrt(np.trace(Sigma_W) / d)

    kappa_arr = np.zeros(len(classes))
    for ci, c in enumerate(classes):
        diffs = centroids - centroids[ci]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        dists[ci] = np.inf
        delta_min = dists.min()
        kappa_arr[ci] = delta_min / (sigma_W_global * np.sqrt(d) + 1e-12)

    # Active classes only (above floor)
    floor = 1.5 / K
    active_mask = q_arr > floor
    q_a = q_arr[active_mask]
    kappa_a = kappa_arr[active_mask]
    n_active = int(active_mask.sum())
    if n_active < MIN_ACTIVE_CLASSES:
        return None, f"too_few_active: {n_active}"

    logit_q = logit_safe(q_a)
    r_kappa, p_kappa = pearsonr(kappa_a, logit_q)

    # Mean q regime classification
    mean_q = float(q_a.mean())
    regime = "normal"
    if mean_q < 0.15:
        regime = "noise_floor"
    elif mean_q > 0.75:
        regime = "ceiling"

    t1 = time.time()
    return {
        "session": session_key,
        "K": int(K),
        "n_pres": int(n_pres),
        "n_active": n_active,
        "n_units": n_units_good,
        "d_pca": d,
        "r_kappa": float(r_kappa),
        "p_kappa": float(p_kappa),
        "mean_q": mean_q,
        "H1": bool(r_kappa > H1_THRESHOLD),
        "regime": regime,
        "elapsed_s": round(t1 - t0, 1),
    }, None


def main():
    print("CTI Allen DANDI:000022 Cross-Cohort Replication", flush=True)
    print("Pre-registered: H_cross_cohort: frac_H1_pass >= 0.80", flush=True)
    print("=" * 60, flush=True)

    from dandi.dandiapi import DandiAPIClient
    client = DandiAPIClient()
    dandiset = client.get_dandiset("000022")
    assets = list(dandiset.get_assets())

    # Main session files (no probe sub-files)
    main_nwb = [
        a for a in assets
        if a.path.endswith(".nwb")
        and "_probe-" not in a.path
        and "_ecephys" not in a.path
        and "behavior" not in a.path.lower()
    ]
    print(f"Found {len(main_nwb)} main NWB files", flush=True)

    results = []
    errors = []
    intermediate = {}

    for idx, asset in enumerate(main_nwb):
        session_key = asset.path.replace("/", "_").replace(".nwb", "")
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
        print(f"\n[{idx+1}/{len(main_nwb)}] {session_key}", flush=True)

        res, err = process_session(url, session_key)
        if err:
            print(f"  SKIP: {err}", flush=True)
            errors.append({"session": session_key, "error": err})
        else:
            status = "PASS" if res["H1"] else "FAIL"
            print(
                f"  r_kappa={res['r_kappa']:.3f} mean_q={res['mean_q']:.3f} "
                f"K={res['K']} n_units={res['n_units']} "
                f"regime={res['regime']} H1={status} "
                f"({res['elapsed_s']:.0f}s)",
                flush=True
            )
            results.append(res)

        # Save intermediate
        intermediate = {
            "dandiset": "000022",
            "status": "in_progress",
            "results": results,
            "errors": errors,
            "n_done": idx + 1,
            "n_total": len(main_nwb),
        }
        with open(OUT_PATH, "w", encoding="ascii") as fp:
            json.dump(intermediate, fp, indent=2)

    # Compute summary
    n_valid = len(results)
    n_pass = sum(r["H1"] for r in results)
    frac_pass = n_pass / n_valid if n_valid > 0 else 0.0
    pass_h = frac_pass >= PASS_FRAC_THRESHOLD

    # Regime breakdown
    normal_results = [r for r in results if r["regime"] == "normal"]
    n_normal_pass = sum(r["H1"] for r in normal_results)
    frac_normal = n_normal_pass / len(normal_results) if normal_results else 0.0

    mean_r = float(np.mean([r["r_kappa"] for r in results])) if results else None
    cv_r = float(np.std([r["r_kappa"] for r in results]) / mean_r) if mean_r else None

    print("\n" + "=" * 60, flush=True)
    print(f"SUMMARY: {n_pass}/{n_valid} sessions H1 PASS ({100*frac_pass:.1f}%)", flush=True)
    print(f"Normal-regime only: {n_normal_pass}/{len(normal_results)} "
          f"({100*frac_normal:.1f}%)", flush=True)
    print(f"Mean r_kappa={mean_r:.3f}, CV={cv_r:.3f}" if mean_r else "No valid sessions", flush=True)
    print(f"H_cross_cohort (frac >= 0.80): {'PASS' if pass_h else 'FAIL'}", flush=True)

    final = {
        "experiment": "cti_allen_000022_crosscohort",
        "dandiset": "000022",
        "preregistration": "H_cross_cohort: frac_H1_pass(r>0.50) >= 0.80",
        "reference_dandiset_000021": "30/32 PASS, mean_r=0.736",
        "results": results,
        "errors": errors,
        "summary": {
            "n_valid": n_valid,
            "n_pass": n_pass,
            "frac_pass": frac_pass,
            "n_normal": len(normal_results),
            "n_normal_pass": n_normal_pass,
            "frac_normal_pass": frac_normal,
            "mean_r_kappa": mean_r,
            "cv_r_kappa": cv_r,
            "PASS_H_cross_cohort": pass_h,
        },
        "status": "complete",
    }

    with open(OUT_PATH, "w", encoding="ascii") as fp:
        json.dump(final, fp, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
