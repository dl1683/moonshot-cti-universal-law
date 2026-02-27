"""
CTI ALLEN EQUICORRELATION MULTI-AREA TEST
==========================================

PRE-REGISTERED HYPOTHESES (commit before running):
  H_equicorr1: rho in VISl and VISam is within 0.08 of VISp rho (near-simplex preserved)
  H_equicorr_alt: rho degrades monotonically with area hierarchy order (alternative)
  Either outcome is scientifically significant:
    - If H_equicorr1: near-simplex geometry is substrate-independent across cortical hierarchy
    - If H_equicorr_alt: CTI geometry tracks biological hierarchy

APPROACH:
  Use same 5 sessions as cti_allen_equicorrelation.py (highest r_kappa, all VISp passing).
  For each session: extract units per area, run equicorrelation (rho) pipeline per area.
  Compare rho_VISp vs rho_VISl vs rho_VISam etc. across sessions.

Output: results/cti_allen_equicorr_multiarea.json
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Same 5 sessions used in cti_allen_equicorrelation.py
TARGET_SESSIONS = [
    "sub-699733573_ses-715093703",
    "sub-718643564_ses-737581020",
    "sub-726170927_ses-746083955",
    "sub-734865729_ses-756029989",
    "sub-738651046_ses-760693773",
]

TARGET_AREAS = ["VISp", "VISl", "VISal", "VISam", "VISrl"]
MIN_UNITS_PER_AREA = 40  # slightly relaxed vs batch; equicorr just needs centroids
N_PCA = 100
RESPONSE_START = 0.05
RESPONSE_END = 0.25

# Hierarchy order (lower = closer to primary visual cortex = expected higher rho)
HIERARCHY_ORDER = ["VISp", "VISl", "VISal", "VISam", "VISrl"]


def compute_equicorr(R_pca, labels):
    """
    Compute competition equicorrelation rho.
    Same formula as cti_allen_equicorrelation.py.
    Returns (rho, rho_std, d_eff) or (None, None, None).
    """
    classes = np.unique(labels)
    K = len(classes)
    n_comp = R_pca.shape[1]

    centroids = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = R_pca[mask].mean(0)

    if len(centroids) < max(10, K // 2):
        return None, None, None

    # Within-class covariance
    Xc_list = []
    for c in classes:
        mask = labels == c
        if c in centroids and mask.sum() >= 2:
            Xc_list.append(R_pca[mask] - centroids[c])
    if not Xc_list:
        return None, None, None
    Xc = np.concatenate(Xc_list)
    N_total = len(Xc)

    Sigma_W = (Xc.T @ Xc) / N_total
    Sigma_W_reg = Sigma_W + 1e-6 * np.trace(Sigma_W) / n_comp * np.eye(n_comp)

    try:
        L = np.linalg.cholesky(Sigma_W_reg)
        use_chol = True
    except np.linalg.LinAlgError:
        use_chol = False
        U, s, _ = np.linalg.svd(Sigma_W_reg)
        sqrt_S = np.diag(np.sqrt(np.maximum(s, 1e-12)))

    classes_sorted = sorted(centroids.keys())
    centroid_arr = np.array([centroids[c] for c in classes_sorted])
    rho_per_class = []

    for ci, c in enumerate(classes_sorted):
        other = [i for i in range(len(classes_sorted)) if i != ci]
        deltas = centroid_arr[other] - centroids[c]

        if use_chol:
            wh = (L.T @ deltas.T).T
        else:
            wh = (sqrt_S @ U.T @ deltas.T).T

        norms = np.linalg.norm(wh, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        wh_n = wh / norms
        cos_mat = wh_n @ wh_n.T
        off = ~np.eye(len(other), dtype=bool)
        rho_per_class.append(float(cos_mat[off].mean()))

    rho = float(np.mean(rho_per_class))
    rho_std = float(np.std(rho_per_class))
    d_eff = 1.0 / (1.0 - rho) if rho < 1.0 else float("inf")
    return rho, rho_std, d_eff


def process_session_equicorr(url, session_key):
    """Load NWB and compute equicorrelation per area."""
    import h5py
    import remfile

    t0 = time.time()
    print(f"  Loading {session_key}...", flush=True)

    try:
        rf = remfile.File(url)
        f = h5py.File(rf, "r")
    except Exception as e:
        return None, f"open_fail: {e}"

    try:
        stim = f["intervals"]["natural_scenes_presentations"]
        frame_idx = stim["frame_index"][:] if "frame_index" in stim else stim["frame"][:]
        start_times = stim["start_time"][:]
        valid_mask = frame_idx != -1
        valid_starts = start_times[valid_mask]
        valid_frames = frame_idx[valid_mask]
        n_pres = len(valid_starts)
        K = len(np.unique(valid_frames))
        if n_pres < 1000 or K != 118:
            f.close()
            return None, f"bad_stim"
    except KeyError as e:
        f.close()
        return None, f"no_natural_scenes: {e}"

    units = f["units"]
    quality = units["quality"][:].astype(str) if "quality" in units else None
    good_mask = (np.array([q.strip() == "good" for q in quality])
                 if quality is not None
                 else np.ones(len(units["id"][:]), dtype=bool))

    area_field = None
    for fname in ["location", "ecephys_structure_acronym", "brain_area"]:
        if fname in units:
            area_field = fname
            break
    if area_field is None:
        f.close()
        return None, "no_area_field"

    area_labels = units[area_field][:].astype(str)
    all_spike_times = units["spike_times"][:]
    all_idx = units["spike_times_index"][:]
    f.close()

    good_idx = np.where(good_mask)[0]
    if len(good_idx) < MIN_UNITS_PER_AREA:
        return None, f"too_few_units: {len(good_idx)}"

    prev_ends = np.concatenate([[0], all_idx[:-1]])
    stim_starts_arr = valid_starts + RESPONSE_START
    stim_stops_arr = valid_starts + RESPONSE_END

    R_full = np.zeros((n_pres, len(good_idx)), dtype=np.float32)
    for j, unit_j in enumerate(good_idx):
        spikes = np.sort(all_spike_times[int(prev_ends[unit_j]):int(all_idx[unit_j])])
        R_full[:, j] = (np.searchsorted(spikes, stim_stops_arr)
                        - np.searchsorted(spikes, stim_starts_arr))

    area_rho = {}
    for area in TARGET_AREAS:
        area_mask = np.array([a.strip() == area for a in area_labels[good_idx]])
        n_area = int(area_mask.sum())
        if n_area < MIN_UNITS_PER_AREA:
            area_rho[area] = {"status": "insufficient_units", "n_units": n_area}
            continue

        R_area = R_full[:, area_mask]
        n_pca = min(N_PCA, n_area - 1)
        pca = PCA(n_components=n_pca)
        try:
            R_pca = pca.fit_transform(R_area).astype(np.float64)
        except Exception as e:
            area_rho[area] = {"status": f"pca_fail: {e}", "n_units": n_area}
            continue

        rho, rho_std, d_eff = compute_equicorr(R_pca, valid_frames)
        if rho is None:
            area_rho[area] = {"status": "equicorr_fail", "n_units": n_area}
            continue

        area_rho[area] = {
            "status": "ok",
            "rho": float(rho),
            "rho_std": float(rho_std),
            "d_eff": float(d_eff),
            "n_units": n_area,
        }
        print(f"    {area}: rho={rho:.4f} +/- {rho_std:.4f}, d_eff={d_eff:.3f}", flush=True)

    elapsed = time.time() - t0
    return {"session": session_key, "areas": area_rho, "elapsed_s": round(elapsed, 1)}, None


def main():
    from dandi.dandiapi import DandiAPIClient

    print("CTI ALLEN EQUICORRELATION MULTI-AREA TEST", flush=True)
    print("Reference (VISp, 5 sessions): rho=0.466, CV=1.6%", flush=True)
    print("LM decoders: rho=0.452, simplex: rho=0.500", flush=True)

    # Get DANDI URLs
    client = DandiAPIClient()
    dandiset = client.get_dandiset("000021")
    assets = list(dandiset.get_assets())
    main_nwb = [
        a for a in assets
        if a.path.endswith(".nwb")
        and "_probe-" not in a.path
        and "_ecephys" not in a.path
    ]
    session_url_map = {}
    for a in main_nwb:
        key = a.path.split("/")[-1].replace(".nwb", "")
        session_url_map[key] = a.get_content_url(follow_redirects=1, strip_query=True)
    print(f"Found {len(session_url_map)} DANDI sessions", flush=True)

    all_results = []
    out_path = RESULTS_DIR / "cti_allen_equicorr_multiarea.json"

    for session_key in TARGET_SESSIONS:
        url = session_url_map.get(session_key)
        if url is None:
            print(f"\nWARNING: {session_key} not found", flush=True)
            continue
        print(f"\n--- {session_key} ---", flush=True)
        result, err = process_session_equicorr(url, session_key)
        if err:
            print(f"  ERROR: {err}", flush=True)
        else:
            all_results.append(result)
        with open(out_path, "w", encoding="ascii") as fp:
            json.dump({"sessions": all_results, "status": "in_progress"}, fp, indent=2)
        sys.stdout.flush()

    # --- ANALYSIS ---
    print("\n" + "="*60, flush=True)
    print("EQUICORRELATION MULTI-AREA SUMMARY", flush=True)
    print("="*60, flush=True)

    per_area_rho = {a: [] for a in TARGET_AREAS}
    for r in all_results:
        for area in TARGET_AREAS:
            ar = r["areas"].get(area, {})
            if ar.get("status") == "ok":
                per_area_rho[area].append(ar["rho"])

    area_means = {}
    for area in TARGET_AREAS:
        vals = per_area_rho[area]
        if vals:
            print(f"  {area}: n={len(vals)}, "
                  f"mean_rho={np.mean(vals):.4f} +/- {np.std(vals):.4f}", flush=True)
            area_means[area] = float(np.mean(vals))
        else:
            print(f"  {area}: no valid sessions", flush=True)

    # H_equicorr1: VISl and VISam within 0.08 of VISp
    h_equicorr1 = False
    h_equicorr1_detail = {}
    if "VISp" in area_means and "VISl" in area_means and "VISam" in area_means:
        visp_rho = area_means["VISp"]
        visl_diff = abs(area_means["VISl"] - visp_rho)
        visam_diff = abs(area_means["VISam"] - visp_rho)
        h_equicorr1 = visl_diff < 0.08 and visam_diff < 0.08
        h_equicorr1_detail = {
            "VISp_rho": visp_rho,
            "VISl_rho": area_means.get("VISl"),
            "VISam_rho": area_means.get("VISam"),
            "VISl_diff": float(visl_diff),
            "VISam_diff": float(visam_diff),
            "threshold": 0.08,
            "PASS": bool(h_equicorr1),
        }

    # H_equicorr_alt: rho degrades monotonically with hierarchy
    hierarchy_areas = [a for a in HIERARCHY_ORDER if a in area_means]
    hierarchy_rhos = [area_means[a] for a in hierarchy_areas]
    h_equicorr_alt = False
    h_equicorr_alt_detail = {}
    if len(hierarchy_rhos) >= 3:
        predicted_ranks = list(range(len(hierarchy_areas)))  # lower idx = higher expected rho
        observed_rho_ranks = np.argsort(np.argsort(-np.array(hierarchy_rhos))).tolist()
        rho_spear, p_spear = spearmanr(predicted_ranks, observed_rho_ranks)
        h_equicorr_alt = bool(rho_spear > 0.70)
        h_equicorr_alt_detail = {
            "areas": hierarchy_areas,
            "mean_rhos": hierarchy_rhos,
            "spearman_rho": float(rho_spear),
            "p_value": float(p_spear),
            "PASS": h_equicorr_alt,
        }
        print(f"\n  Hierarchy Spearman rho={rho_spear:.3f}, p={p_spear:.3f} "
              f"({'PASS' if h_equicorr_alt else 'FAIL'})", flush=True)

    print(f"\n  H_equicorr1 (preservation): {'PASS' if h_equicorr1 else 'FAIL'}", flush=True)
    print(f"  H_equicorr_alt (degradation): {'PASS' if h_equicorr_alt else 'FAIL'}", flush=True)

    # Reference comparison (VISp, this run vs prior run)
    lm_ref = 0.452
    if "VISp" in area_means:
        visp = area_means["VISp"]
        print(f"\n  VISp rho this run: {visp:.4f} (prior run: 0.466, LM: {lm_ref}, simplex: 0.500)", flush=True)

    out = {
        "experiment": "cti_allen_equicorr_multiarea",
        "preregistration": "H_equicorr1, H_equicorr_alt (pre-registered before run)",
        "reference_visp_prior": {"mean_rho": 0.466, "cv_rho": 0.016, "n_sessions": 5},
        "reference_lm": {"mean_rho": 0.452, "simplex": 0.500},
        "per_area_summary": {
            a: {
                "n_sessions": len(per_area_rho[a]),
                "mean_rho": float(np.mean(per_area_rho[a])) if per_area_rho[a] else None,
                "std_rho": float(np.std(per_area_rho[a])) if per_area_rho[a] else None,
                "all_rho": [float(x) for x in per_area_rho[a]],
            }
            for a in TARGET_AREAS
        },
        "hypothesis_results": {
            "H_equicorr1": h_equicorr1_detail,
            "H_equicorr_alt": h_equicorr_alt_detail,
        },
        "sessions": all_results,
        "status": "complete",
    }

    with open(out_path, "w", encoding="ascii") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
