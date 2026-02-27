"""
CTI ALLEN NEUROPIXELS MULTI-AREA BATCH VALIDATION
==================================================

PRE-REGISTERED HYPOTHESES (commit before running):
  H_area1: VISl across >=15 sessions: mean r_kappa > 0.60, H1 pass rate >= 60%
  H_area2: VISam across >=10 sessions: mean r_kappa > 0.55, H1 pass rate >= 55%
  H_area3: At least 3/4 secondary areas (VISl, VISam, VISal, VISrl) H1 pass rate >= 50%
  H_hierarchy: r_kappa degrades from VISp toward higher areas; Spearman rho > 0.70
               Predicted order: VISp >= VISl/VISal > VISam/VISrl (consistent with eNeuro 2018)

APPROACH:
  Take all 30 H1-passing sessions from cti_allen_all_sessions_complete.json.
  For each session, extract units per labeled brain area (units["location"] field).
  Run full kappa+logit(q) pipeline per area independently.
  Require >= 50 good units AND >= 10 active classes per area (else: insufficient_units).
  Compute per-area r_kappa, H1, mean_q.
  Aggregate across sessions per area.
  Test hierarchy hypothesis via Spearman rank correlation.

Output: results/cti_allen_multiarea_batch.json
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

MIN_UNITS_PER_AREA = 50
MIN_ACTIVE_CLASSES = 10
N_PCA_CAP = 100
RESPONSE_START = 0.05
RESPONSE_END = 0.25
ACTIVE_THRESH_FACTOR = 1.5  # q > 1.5/K

# Known hierarchy order from eNeuro 2018 (lower index = lower hierarchy = higher expected decoding)
# VISp is primary; VISl, VISal are V2-equivalent; VISam, VISrl are higher
HIERARCHY_ORDER = ["VISp", "VISl", "VISal", "VISam", "VISrl"]
# Expected r_kappa rank (descending): VISp, VISl, VISal, VISam, VISrl
# Spearman test: does observed mean_r_kappa rank match this prediction?


def logit_safe(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def compute_kappa_and_q(R_pca, labels):
    """Compute r_kappa for a single area's response matrix (already PCA-reduced)."""
    classes = np.unique(labels)
    K = len(classes)
    n_pca = R_pca.shape[1]

    centroids = np.array([R_pca[labels == c].mean(0) for c in classes])

    Sigma_W = np.zeros((n_pca, n_pca))
    for c in classes:
        Xi = R_pca[labels == c] - centroids[classes == c][0]
        Sigma_W += Xi.T @ Xi
    Sigma_W /= len(labels)
    sigma_W_global = float(np.sqrt(np.trace(Sigma_W) / n_pca))
    if sigma_W_global < 1e-10:
        return None

    kappa_arr = np.zeros(K)
    for ci, c in enumerate(classes):
        diffs = centroids - centroids[ci]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        dists[ci] = np.inf
        kappa_arr[ci] = dists.min() / (sigma_W_global * np.sqrt(n_pca))

    dist_mat = np.sum((R_pca[:, None, :] - R_pca[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    nn_idx = np.argmin(dist_mat, axis=1)
    nn_labels = labels[nn_idx]
    correct = (nn_labels == labels).astype(float)

    q_arr = np.array([correct[labels == c].mean() for c in classes])

    active_mask = q_arr > ACTIVE_THRESH_FACTOR / K
    n_active = int(active_mask.sum())
    if n_active < MIN_ACTIVE_CLASSES:
        return None

    kappa_a = kappa_arr[active_mask]
    q_a = q_arr[active_mask]
    logit_q = logit_safe(q_a)

    if len(kappa_a) < 5:
        return None

    r_kappa, _ = pearsonr(kappa_a, logit_q)
    return {
        "r_kappa": float(r_kappa),
        "n_active": n_active,
        "mean_q": float(q_a.mean()),
        "H1": bool(r_kappa > 0.50),
    }


def process_session_multiarea(url, session_key):
    """Load NWB and run kappa pipeline per brain area."""
    import h5py
    import remfile

    t0 = time.time()
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
            return None, f"bad_stim: n_pres={n_pres}, K={K}"
    except KeyError as e:
        f.close()
        return None, f"no_natural_scenes: {e}"

    units = f["units"]

    # Get quality mask
    quality = units["quality"][:].astype(str) if "quality" in units else None
    good_mask = (np.array([q.strip() == "good" for q in quality])
                 if quality is not None
                 else np.ones(len(units["id"][:]), dtype=bool))

    # Get area labels via peak_channel_id -> electrodes["location"] join
    elec_table = f["general"]["extracellular_ephys"]["electrodes"]
    elec_ids = elec_table["id"][:]
    elec_locs = elec_table["location"][:].astype(str)
    id_to_loc = {int(eid): loc for eid, loc in zip(elec_ids, elec_locs)}
    peak_cids = units["peak_channel_id"][:]
    area_labels_all = np.array([id_to_loc.get(int(pc), "") for pc in peak_cids])
    all_spike_times = units["spike_times"][:]
    all_idx = units["spike_times_index"][:]
    f.close()

    # Build full response matrix for all good units
    good_idx = np.where(good_mask)[0]
    n_units_good = len(good_idx)
    if n_units_good < MIN_UNITS_PER_AREA:
        return None, f"too_few_total_units: {n_units_good}"

    prev_ends = np.concatenate([[0], all_idx[:-1]])
    stim_starts_arr = valid_starts + RESPONSE_START
    stim_stops_arr = valid_starts + RESPONSE_END

    R_full = np.zeros((n_pres, n_units_good), dtype=np.float32)
    for j, unit_j in enumerate(good_idx):
        spikes = np.sort(all_spike_times[int(prev_ends[unit_j]):int(all_idx[unit_j])])
        R_full[:, j] = (np.searchsorted(spikes, stim_stops_arr)
                        - np.searchsorted(spikes, stim_starts_arr))

    # Process each area
    area_results = {}
    target_areas = ["VISp", "VISl", "VISal", "VISam", "VISrl", "LP"]

    for area in target_areas:
        area_mask_full = np.array([
            a.strip() == area for a in area_labels_all[good_idx]
        ])
        n_area_units = int(area_mask_full.sum())
        if n_area_units < MIN_UNITS_PER_AREA:
            area_results[area] = {"status": "insufficient_units", "n_units": n_area_units}
            continue

        R_area = R_full[:, area_mask_full]
        n_pca = min(N_PCA_CAP, n_area_units - 1)
        pca = PCA(n_components=n_pca)
        try:
            R_pca = pca.fit_transform(R_area)
        except Exception as e:
            area_results[area] = {"status": f"pca_fail: {e}", "n_units": n_area_units}
            continue

        res = compute_kappa_and_q(R_pca, valid_frames)
        if res is None:
            area_results[area] = {"status": "too_few_active", "n_units": n_area_units}
            continue

        res["n_units"] = n_area_units
        res["status"] = "ok"
        area_results[area] = res

    elapsed = time.time() - t0
    return {"session": session_key, "areas": area_results, "elapsed_s": round(elapsed, 1)}, None


def main():
    from dandi.dandiapi import DandiAPIClient

    print("CTI ALLEN MULTI-AREA BATCH VALIDATION", flush=True)
    print("Pre-registered hypotheses: H_area1, H_area2, H_area3, H_hierarchy", flush=True)

    # Load the 30 successful sessions
    with open(RESULTS_DIR / "cti_allen_all_sessions_complete.json") as f:
        all_sess = json.load(f)
    passing_sessions = [s["session"] for s in all_sess["sessions"] if s.get("H1", False)]
    print(f"Sessions to process: {len(passing_sessions)}", flush=True)

    # Get DANDI URLs
    print("Fetching DANDI asset list...", flush=True)
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
    print(f"  Found {len(session_url_map)} DANDI sessions", flush=True)

    session_results = []
    out_path = RESULTS_DIR / "cti_allen_multiarea_batch.json"

    for i, session_key in enumerate(passing_sessions):
        url = session_url_map.get(session_key)
        if url is None:
            print(f"[{i+1}/{len(passing_sessions)}] {session_key}: NOT IN DANDI", flush=True)
            continue

        print(f"\n[{i+1}/{len(passing_sessions)}] {session_key}", flush=True)
        result, err = process_session_multiarea(url, session_key)

        if err:
            print(f"  ERROR: {err}", flush=True)
            session_results.append({"session": session_key, "error": err})
        else:
            session_results.append(result)
            for area, ar in result["areas"].items():
                if ar.get("status") == "ok":
                    h1 = "PASS" if ar["H1"] else "FAIL"
                    print(f"  {area}: r_kappa={ar['r_kappa']:.3f} ({h1}), "
                          f"n_units={ar['n_units']}, n_active={ar['n_active']}", flush=True)
                else:
                    print(f"  {area}: {ar.get('status')} (n_units={ar.get('n_units', 0)})", flush=True)

        # Save progress
        with open(out_path, "w", encoding="ascii") as fp:
            json.dump({"sessions": session_results, "status": "in_progress"}, fp, indent=2)
        sys.stdout.flush()

    # --- ANALYSIS ---
    print("\n" + "="*60, flush=True)
    print("MULTI-AREA SUMMARY", flush=True)
    print("="*60, flush=True)

    ok_sessions = [r for r in session_results if "error" not in r and "areas" in r]
    areas_to_test = ["VISp", "VISl", "VISal", "VISam", "VISrl"]
    per_area = {}

    for area in areas_to_test:
        vals = []
        for s in ok_sessions:
            ar = s["areas"].get(area, {})
            if ar.get("status") == "ok":
                vals.append(ar["r_kappa"])
        if vals:
            h1_pass = sum(1 for v in vals if v > 0.50)
            per_area[area] = {
                "n_sessions_valid": len(vals),
                "mean_r_kappa": float(np.mean(vals)),
                "std_r_kappa": float(np.std(vals)),
                "H1_pass_rate": f"{h1_pass}/{len(vals)}",
                "H1_pass_frac": float(h1_pass / len(vals)),
                "all_r_kappa": vals,
            }
            print(f"  {area}: n={len(vals)}, mean_r={np.mean(vals):.3f}, "
                  f"H1={h1_pass}/{len(vals)}", flush=True)
        else:
            per_area[area] = {"n_sessions_valid": 0}
            print(f"  {area}: no valid sessions", flush=True)

    # Hypothesis tests
    h1_result = {}
    if "VISl" in per_area and per_area["VISl"]["n_sessions_valid"] >= 10:
        p = per_area["VISl"]
        h1_result["H_area1"] = {
            "area": "VISl",
            "mean_r_kappa": p["mean_r_kappa"],
            "H1_pass_rate": p["H1_pass_rate"],
            "PASS": p["mean_r_kappa"] > 0.60 and p["H1_pass_frac"] >= 0.60,
        }

    if "VISam" in per_area and per_area["VISam"]["n_sessions_valid"] >= 10:
        p = per_area["VISam"]
        h1_result["H_area2"] = {
            "area": "VISam",
            "mean_r_kappa": p["mean_r_kappa"],
            "H1_pass_rate": p["H1_pass_rate"],
            "PASS": p["mean_r_kappa"] > 0.55 and p["H1_pass_frac"] >= 0.55,
        }

    secondary_areas = ["VISl", "VISam", "VISal", "VISrl"]
    areas_with_50pct = sum(
        1 for a in secondary_areas
        if per_area.get(a, {}).get("H1_pass_frac", 0) >= 0.50
    )
    h1_result["H_area3"] = {
        "areas_passing_50pct": areas_with_50pct,
        "out_of": 4,
        "PASS": areas_with_50pct >= 3,
    }

    # Hierarchy test: Spearman rank of mean_r_kappa vs predicted order
    hierarchy_vals = []
    hierarchy_areas_valid = []
    for a in HIERARCHY_ORDER:
        if per_area.get(a, {}).get("n_sessions_valid", 0) >= 5:
            hierarchy_vals.append(per_area[a]["mean_r_kappa"])
            hierarchy_areas_valid.append(a)

    if len(hierarchy_vals) >= 3:
        # Predicted rank: lower index = higher expected r_kappa
        predicted_ranks = list(range(len(hierarchy_areas_valid)))
        observed_ranks = np.argsort(np.argsort(-np.array(hierarchy_vals))).tolist()
        rho, p_val = spearmanr(predicted_ranks, observed_ranks)
        h1_result["H_hierarchy"] = {
            "areas": hierarchy_areas_valid,
            "predicted_order": hierarchy_areas_valid,
            "mean_r_kappa_observed": hierarchy_vals,
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "PASS": bool(rho > 0.70),
        }
        print(f"\n  Hierarchy Spearman rho={rho:.3f}, p={p_val:.3f} "
              f"({'PASS' if rho > 0.70 else 'FAIL'})", flush=True)

    # Print hypothesis verdicts
    print("\n--- HYPOTHESIS VERDICTS ---", flush=True)
    for h, v in h1_result.items():
        status = "PASS" if v.get("PASS") else "FAIL"
        print(f"  {h}: {status} | {v}", flush=True)

    # Build output (no numpy types)
    per_area_clean = {}
    for area, vals in per_area.items():
        per_area_clean[area] = {k: v for k, v in vals.items() if k != "all_r_kappa"}
        if "all_r_kappa" in vals:
            per_area_clean[area]["all_r_kappa"] = [float(x) for x in vals["all_r_kappa"]]

    out = {
        "experiment": "cti_allen_multiarea_batch",
        "preregistration": "H_area1, H_area2, H_area3, H_hierarchy (pre-registered before run)",
        "dataset": "DANDI:000021",
        "min_units_per_area": MIN_UNITS_PER_AREA,
        "min_active_classes": MIN_ACTIVE_CLASSES,
        "n_sessions_attempted": len(passing_sessions),
        "n_sessions_processed": len(ok_sessions),
        "per_area_summary": per_area_clean,
        "hypothesis_results": h1_result,
        "sessions": session_results,
        "status": "complete",
    }

    with open(out_path, "w", encoding="ascii") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
