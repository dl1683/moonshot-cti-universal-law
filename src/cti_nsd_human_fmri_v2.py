"""
CTI Universal Law: Human fMRI Validation V2 (Natural Scenes Dataset)
====================================================================
Improved version with proper fMRI preprocessing:
1. Z-score per voxel per session (standard fMRI decoding practice)
2. ncsnr-based voxel selection (noise ceiling SNR > threshold)
3. Per-class kappa and per-class 1-NN accuracy (same as Allen analysis)
4. Pooled regression across ROIs x classes

Dataset: Natural Scenes Dataset (NSD) - Allen, St-Yves et al. 2022
"""

import numpy as np
from scipy import stats, special
from sklearn.decomposition import PCA
import json
import os
import time
import warnings
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'nsd')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

SUBJECT = 1
N_SESSIONS = 10
N_PCA = 50  # Fewer PCA dims for noisy fMRI data
NCSNR_THRESHOLD = 0.3  # Minimum noise ceiling SNR
MIN_IMAGES_PER_CLASS = 30
MIN_VOXELS_PER_ROI = 50  # After ncsnr filtering

MERGED_ROIS = {
    'V1': [1, 2], 'V2': [3, 4], 'V3': [5, 6], 'hV4': [7],
    'VO': [8, 9], 'PHC': [10, 11], 'TO': [12, 13], 'LO': [14, 15],
    'V3AB': [16, 17], 'IPS01': [18, 19], 'IPS23': [20, 21],
}


def build_primary_labels():
    """Build cocoId -> primary supercategory mapping using largest annotation area."""
    import pandas as pd

    annotations_dir = os.path.join(DATA_DIR, 'annotations')
    with open(os.path.join(annotations_dir, 'instances_val2017.json')) as f:
        val_data = json.load(f)
    with open(os.path.join(annotations_dir, 'instances_train2017.json')) as f:
        train_data = json.load(f)

    cat_map = {}
    for cat in val_data['categories']:
        cat_map[cat['id']] = cat['supercategory']

    img_area = defaultdict(lambda: defaultdict(float))
    for ann in val_data['annotations'] + train_data['annotations']:
        img_area[ann['image_id']][cat_map[ann['category_id']]] += ann['area']

    nsd_info = pd.read_csv(os.path.join(DATA_DIR, 'nsd_stim_info_merged.csv'))
    nsd_to_coco = dict(zip(nsd_info['nsdId'], nsd_info['cocoId']))
    primary_labels = {}
    for nsd_id, coco_id in nsd_to_coco.items():
        if coco_id in img_area:
            areas = img_area[coco_id]
            primary_labels[nsd_id] = max(areas, key=areas.get)

    return primary_labels


def get_trial_to_nsdid_mapping(subject_idx=0):
    """Map (session_0idx, trial_0idx) -> nsdId."""
    import scipy.io
    expdesign = scipy.io.loadmat(os.path.join(DATA_DIR, 'nsd_expdesign.mat'))
    masterordering = expdesign['masterordering'].flatten()
    subjectim = expdesign['subjectim']
    mapping = {}
    for session in range(40):
        for trial in range(750):
            global_trial = session * 750 + trial
            if global_trial >= len(masterordering):
                break
            subj_img_id = masterordering[global_trial]
            nsd_id = int(subjectim[subject_idx, subj_img_id - 1])
            mapping[(session, trial)] = nsd_id
    return mapping


def download_session_betas(subject, session):
    """Download one session of betas from S3 using boto3."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    fname = f'betas_session{session:02d}.nii.gz'
    s3_key = (f'nsddata_betas/ppdata/subj{subject:02d}/func1pt8mm/'
              f'betas_fithrf_GLMdenoise_RR/{fname}')

    local_dir = os.path.join(DATA_DIR, 'betas_tmp')
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, fname)

    if os.path.exists(local_path):
        return local_path

    print(f"  Downloading {fname} (~490 MB)...", flush=True)
    t0 = time.time()
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3.download_file('natural-scenes-dataset', s3_key, local_path)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(local_path) / (1024**2)
    print(f"  Downloaded {size_mb:.1f} MB in {elapsed:.0f}s", flush=True)
    return local_path


def compute_cti_per_class(responses, labels, n_pca=50, roi_name=''):
    """
    Compute PER-CLASS kappa_nearest and per-class 1-NN accuracy.
    This matches the Allen Neuropixels analysis where each class is a data point.

    Returns list of dicts, one per class.
    """
    K = len(np.unique(labels))
    n_images, n_voxels = responses.shape

    n_pca_actual = min(n_pca, n_voxels - 1, n_images - K)
    if n_pca_actual < 5:
        print(f"  {roi_name}: too few PCA dims ({n_pca_actual}), skipping", flush=True)
        return None

    pca = PCA(n_components=n_pca_actual)
    R_pca = pca.fit_transform(responses.astype(np.float64))
    d = n_pca_actual

    class_ids = np.unique(labels)
    centroids = np.array([R_pca[labels == c].mean(axis=0) for c in class_ids])

    # Global within-class covariance
    all_within = []
    for c in class_ids:
        X_c = R_pca[labels == c]
        all_within.append(X_c - X_c.mean(axis=0))
    all_within = np.vstack(all_within)
    sigma_W_sq_per_dim = np.var(all_within, axis=0, ddof=1)
    tr_Sigma_W = sigma_W_sq_per_dim.sum()
    sigma_W_global = np.sqrt(tr_Sigma_W / d)

    # Centroid distances
    centroid_dists = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)
    np.fill_diagonal(centroid_dists, np.inf)
    min_dists = centroid_dists.min(axis=1)
    kappa_per_class = min_dists / (sigma_W_global * np.sqrt(d))

    # BALANCED 1-NN: subsample to equal class sizes, repeat N_BOOT times
    N_BOOT = 20
    min_n = min(np.sum(labels == c) for c in class_ids)
    print(f"  {roi_name}: balanced 1-NN (K={K}, n_per_class={min_n}, "
          f"{N_BOOT} bootstrap)...", flush=True)

    # Bootstrap balanced 1-NN
    per_class_accs_boot = np.zeros((N_BOOT, K))
    global_accs_boot = np.zeros(N_BOOT)
    rng = np.random.RandomState(42)

    for b in range(N_BOOT):
        # Subsample each class to min_n
        bal_indices = []
        for c in class_ids:
            idx_c = np.where(labels == c)[0]
            chosen = rng.choice(idx_c, size=min_n, replace=False)
            bal_indices.extend(chosen.tolist())
        bal_indices = np.array(bal_indices)
        R_bal = R_pca[bal_indices]
        y_bal = labels[bal_indices]

        # LOO 1-NN on balanced set
        dist_bal = np.linalg.norm(R_bal[:, None] - R_bal[None, :], axis=2)
        np.fill_diagonal(dist_bal, np.inf)
        nn_idx = np.argmin(dist_bal, axis=1)
        nn_lab = y_bal[nn_idx]

        global_accs_boot[b] = np.mean(nn_lab == y_bal)
        for i, c in enumerate(class_ids):
            mask_c = (y_bal == c)
            per_class_accs_boot[b, i] = np.mean((nn_lab == y_bal)[mask_c])

    # Average across bootstrap
    per_class_accs = per_class_accs_boot.mean(axis=0)
    global_q = float(global_accs_boot.mean())

    # Per-class results
    results = []
    for i, c in enumerate(class_ids):
        q_c = float(per_class_accs[i])
        q_norm_c = max((q_c - 1.0 / K) / (1.0 - 1.0 / K), 1e-6)
        q_norm_c = min(q_norm_c, 1.0 - 1e-6)
        logit_c = float(special.logit(q_norm_c))

        results.append({
            'class_id': int(c),
            'n_images_total': int(np.sum(labels == c)),
            'n_images_balanced': int(min_n),
            'kappa_nearest': float(kappa_per_class[i]),
            'q_1nn_balanced': float(q_c),
            'q_norm': float(q_norm_c),
            'logit_q_norm': float(logit_c),
        })

    # Equicorrelation
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_normed = centroids / (norms + 1e-10)
    cosines = centroids_normed @ centroids_normed.T
    off_diag = cosines[~np.eye(K, dtype=bool)]
    rho = float(np.mean(off_diag))

    # Global summary
    global_q_norm = max((global_q - 1.0/K) / (1.0 - 1.0/K), 1e-6)
    global_q_norm = min(global_q_norm, 1 - 1e-6)

    print(f"  {roi_name}: balanced q_1nn={global_q:.3f}, "
          f"kappa_range=[{min(kappa_per_class):.4f}, {max(kappa_per_class):.4f}], "
          f"rho={rho:.3f}, n_per_class={min_n}", flush=True)

    return {
        'roi': roi_name,
        'per_class': results,
        'global_q_1nn': float(global_q),
        'global_q_norm': float(global_q_norm),
        'global_logit': float(special.logit(global_q_norm)),
        'rho_equicorr': rho,
        'K': K,
        'n_images': n_images,
        'n_voxels': n_voxels,
        'n_pca': n_pca_actual,
        'sigma_W_global': float(sigma_W_global),
        'kappa_mean': float(np.mean(kappa_per_class)),
    }


def main():
    import nibabel as nib

    print("=" * 70, flush=True)
    print("CTI Universal Law: Human fMRI (NSD) V2 — Per-Class Analysis", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()

    # Check for cached ROI data (skip download if available)
    cache_path = os.path.join(DATA_DIR, 'roi_data_cache.npz')
    cache_labels_path = os.path.join(DATA_DIR, 'roi_labels_cache.json')

    # Step 1: Labels
    print("\n[Step 1] Building category labels...", flush=True)
    primary_labels = build_primary_labels()

    # Check cache
    if os.path.exists(cache_path) and os.path.exists(cache_labels_path):
        print("\n[CACHE HIT] Loading cached ROI data (skip download)...", flush=True)
        cache = np.load(cache_path)
        with open(cache_labels_path) as f:
            cache_meta = json.load(f)
        roi_data = {}
        for roi_name, meta in cache_meta.items():
            roi_data[roi_name] = {
                'responses': cache[f'{roi_name}_responses'],
                'labels': cache[f'{roi_name}_labels'],
                'cat_names': meta['cat_names'],
                'K': meta['K'],
            }
            n = len(roi_data[roi_name]['responses'])
            print(f"  {roi_name}: {n} images, K={meta['K']}", flush=True)

        # Skip to Step 6
        print("\n  Skipping Steps 2-5 (using cached data)", flush=True)

    else:
        # No cache — download and process

        # Step 2: Trial mapping
        print("[Step 2] Building trial mapping...", flush=True)
        trial_map = get_trial_to_nsdid_mapping(subject_idx=SUBJECT - 1)

        # Step 3: ROI masks + ncsnr filter
        print("[Step 3] Loading ROI masks with ncsnr filtering...", flush=True)
        roi_mask = nib.load(os.path.join(DATA_DIR, 'roi', 'Kastner2015_vol.nii.gz')).get_fdata().astype(int)
        ncsnr = nib.load(os.path.join(DATA_DIR, 'ncsnr_vol.nii.gz')).get_fdata()
        ncsnr = np.nan_to_num(ncsnr, nan=0.0)

        roi_voxel_indices = {}
        for roi_name, label_ids in MERGED_ROIS.items():
            mask = np.zeros_like(roi_mask, dtype=bool)
            for lid in label_ids:
                mask |= (roi_mask == lid)
            mask &= (ncsnr >= NCSNR_THRESHOLD)
            n_voxels = np.sum(mask)
            if n_voxels < MIN_VOXELS_PER_ROI:
                print(f"  Skipping {roi_name}: only {n_voxels} voxels after ncsnr filter", flush=True)
                continue
            roi_voxel_indices[roi_name] = np.where(mask)
            print(f"  {roi_name}: {n_voxels} voxels (ncsnr>{NCSNR_THRESHOLD})", flush=True)

        # Step 4: Download and extract with z-scoring
        print(f"\n[Step 4] Processing {N_SESSIONS} sessions with z-scoring...", flush=True)
        image_responses = {roi: defaultdict(list) for roi in roi_voxel_indices}

        for sess_idx in range(N_SESSIONS):
            session = sess_idx + 1
            print(f"\n--- Session {session}/{N_SESSIONS} ---", flush=True)

            beta_path = download_session_betas(SUBJECT, session)

            print(f"  Loading and z-scoring...", flush=True)
            t0 = time.time()
            img = nib.load(beta_path)
            data = img.get_fdata()
            n_trials = data.shape[3]

            for roi_name, voxel_idx in roi_voxel_indices.items():
                roi_betas = data[voxel_idx[0], voxel_idx[1], voxel_idx[2], :]
                voxel_mean = np.mean(roi_betas, axis=1, keepdims=True)
                voxel_std = np.std(roi_betas, axis=1, keepdims=True)
                voxel_std[voxel_std < 1e-10] = 1.0
                roi_betas_z = (roi_betas - voxel_mean) / voxel_std

                for trial_idx in range(n_trials):
                    key = (sess_idx, trial_idx)
                    if key not in trial_map:
                        continue
                    nsd_id = trial_map[key]
                    if nsd_id not in primary_labels:
                        continue
                    image_responses[roi_name][nsd_id].append(
                        roi_betas_z[:, trial_idx].astype(np.float32)
                    )

            elapsed = time.time() - t0
            print(f"  Processed in {elapsed:.1f}s", flush=True)
            os.remove(beta_path)
            n_imgs = len(image_responses[list(roi_voxel_indices.keys())[0]])
            print(f"  Cleaned up. Accumulated {n_imgs} unique images", flush=True)

        # Step 5: Average repetitions
        print("\n[Step 5] Averaging repetitions...", flush=True)
        roi_data = {}
        for roi_name in roi_voxel_indices:
            nsd_ids = sorted(image_responses[roi_name].keys())
            responses_list = []
            labels_list = []
            label_to_int = {}

            for nsd_id in nsd_ids:
                reps = image_responses[roi_name][nsd_id]
                if len(reps) == 0:
                    continue
                avg_resp = np.mean(reps, axis=0)
                cat = primary_labels[nsd_id]
                if cat not in label_to_int:
                    label_to_int[cat] = len(label_to_int)
                responses_list.append(avg_resp)
                labels_list.append(label_to_int[cat])

            responses = np.array(responses_list)
            labels = np.array(labels_list)

            cat_counts = Counter(labels)
            valid_cats = {c for c, n in cat_counts.items() if n >= MIN_IMAGES_PER_CLASS}
            mask = np.array([l in valid_cats for l in labels])
            responses = responses[mask]
            labels = labels[mask]

            unique_labels = np.unique(labels)
            label_remap = {old: new for new, old in enumerate(unique_labels)}
            labels = np.array([label_remap[l] for l in labels])

            int_to_label = {v: k for k, v in label_to_int.items()}
            cat_names = [int_to_label[unique_labels[i]] for i in range(len(unique_labels))]
            K = len(unique_labels)

            print(f"  {roi_name}: {len(responses)} images, K={K}", flush=True)
            roi_data[roi_name] = {
                'responses': responses,
                'labels': labels,
                'cat_names': cat_names,
                'K': K,
            }

        # Cache
        print("  Caching ROI data...", flush=True)
        cache_arrays = {}
        cache_meta = {}
        for roi_name, d in roi_data.items():
            cache_arrays[f'{roi_name}_responses'] = d['responses']
            cache_arrays[f'{roi_name}_labels'] = d['labels']
            cache_meta[roi_name] = {'cat_names': d['cat_names'], 'K': d['K']}
        np.savez_compressed(cache_path, **cache_arrays)
        with open(cache_labels_path, 'w') as f:
            json.dump(cache_meta, f)
        print(f"  Cached to {cache_path}", flush=True)

    # Step 6: Per-class CTI analysis
    print("\n[Step 6] Per-class CTI analysis...", flush=True)

    all_roi_results = []
    all_kappas = []
    all_logits = []
    all_roi_labels = []

    for roi_name, data in roi_data.items():
        print(f"\n  === {roi_name} ===", flush=True)
        result = compute_cti_per_class(
            data['responses'], data['labels'],
            n_pca=N_PCA, roi_name=roi_name
        )
        if result is None:
            continue

        result['cat_names'] = data['cat_names']
        all_roi_results.append(result)

        # Within-ROI per-class regression
        kappas = [r['kappa_nearest'] for r in result['per_class']]
        logits = [r['logit_q_norm'] for r in result['per_class']]

        if len(kappas) >= 4:
            slope, intercept, r, p, se = stats.linregress(kappas, logits)
            print(f"  {roi_name} within-ROI: r={r:.3f}, p={p:.3e}, "
                  f"alpha={slope:.2f}", flush=True)
            result['within_roi_fit'] = {
                'alpha': float(slope),
                'r': float(r),
                'p': float(p),
                'R_sq': float(r**2),
            }

        # Collect for pooled analysis
        for r_c in result['per_class']:
            all_kappas.append(r_c['kappa_nearest'])
            all_logits.append(r_c['logit_q_norm'])
            all_roi_labels.append(roi_name)

    # Step 7: Pooled analysis across ROIs x classes
    print("\n[Step 7] Pooled analysis...", flush=True)
    print(f"  Total points: {len(all_kappas)} (ROIs x classes)", flush=True)

    all_kappas = np.array(all_kappas)
    all_logits = np.array(all_logits)

    if len(all_kappas) >= 5:
        slope, intercept, r_val, p_val, se = stats.linregress(all_kappas, all_logits)
        rho_sp, p_sp = stats.spearmanr(all_kappas, all_logits)

        print(f"\n  POOLED CTI Law (n={len(all_kappas)} points):", flush=True)
        print(f"    alpha = {slope:.3f} +/- {se:.3f}", flush=True)
        print(f"    C     = {intercept:.3f}", flush=True)
        print(f"    r     = {r_val:.3f}", flush=True)
        print(f"    R^2   = {r_val**2:.3f}", flush=True)
        print(f"    p     = {p_val:.2e}", flush=True)
        print(f"    Spearman rho = {rho_sp:.3f}, p = {p_sp:.2e}", flush=True)

        pass_r = abs(r_val) > 0.50
        pass_dir = slope > 0

        pooled_fit = {
            'alpha': float(slope), 'alpha_se': float(se),
            'C': float(intercept), 'r': float(r_val),
            'R_sq': float(r_val**2), 'p_value': float(p_val),
            'spearman_rho': float(rho_sp), 'spearman_p': float(p_sp),
            'n_points': len(all_kappas),
            'pass_r': pass_r, 'pass_direction': pass_dir,
        }
    else:
        pooled_fit = {'error': 'too_few_points'}

    # Step 8: ROI-level analysis (mean kappa vs mean accuracy)
    print("\n[Step 8] ROI-level analysis...", flush=True)
    roi_kappas = [r['kappa_mean'] for r in all_roi_results]
    roi_logits = [r['global_logit'] for r in all_roi_results]
    roi_names = [r['roi'] for r in all_roi_results]
    roi_rhos = [r['rho_equicorr'] for r in all_roi_results]

    if len(roi_kappas) >= 3:
        slope_r, int_r, r_r, p_r, se_r = stats.linregress(roi_kappas, roi_logits)
        print(f"  ROI-level: r={r_r:.3f}, p={p_r:.3e}, alpha={slope_r:.2f}", flush=True)
        print(f"  Equicorr: mean={np.mean(roi_rhos):.3f}, CV={np.std(roi_rhos)/abs(np.mean(roi_rhos)+1e-10)*100:.1f}%", flush=True)
        roi_level_fit = {
            'alpha': float(slope_r), 'r': float(r_r),
            'p': float(p_r), 'R_sq': float(r_r**2),
            'rho_equicorr_mean': float(np.mean(roi_rhos)),
            'rho_equicorr_cv': float(np.std(roi_rhos)/abs(np.mean(roi_rhos)+1e-10)*100),
        }
    else:
        roi_level_fit = {'error': 'too_few_rois'}

    # Save results
    elapsed = time.time() - t_start
    output = {
        'experiment': 'CTI Human fMRI (NSD) V2 — Per-Class Analysis',
        'version': 'v2_zscore_ncsnr_perclass',
        'subject': SUBJECT,
        'n_sessions': N_SESSIONS,
        'n_pca': N_PCA,
        'ncsnr_threshold': NCSNR_THRESHOLD,
        'preprocessing': 'z-score per voxel per session, ncsnr filter',
        'per_roi': [{k: v for k, v in r.items() if k != 'per_class'}
                    for r in all_roi_results],
        'per_roi_per_class': all_roi_results,
        'pooled_fit': pooled_fit,
        'roi_level_fit': roi_level_fit,
        'elapsed_seconds': float(elapsed),
    }

    out_path = os.path.join(RESULTS_DIR, 'cti_nsd_human_fmri.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    if 'alpha' in pooled_fit:
        print(f"  Pooled (n={pooled_fit['n_points']}): "
              f"r={pooled_fit['r']:.3f}, R^2={pooled_fit['R_sq']:.3f}, "
              f"alpha={pooled_fit['alpha']:.3f}", flush=True)
        print(f"  PASS r>0.50: {pooled_fit['pass_r']}", flush=True)
        print(f"  PASS alpha>0: {pooled_fit['pass_direction']}", flush=True)
    if 'alpha' in roi_level_fit:
        print(f"  ROI-level: r={roi_level_fit['r']:.3f}, "
              f"alpha={roi_level_fit['alpha']:.3f}", flush=True)
    print(f"  Total time: {elapsed/60:.1f} minutes", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
