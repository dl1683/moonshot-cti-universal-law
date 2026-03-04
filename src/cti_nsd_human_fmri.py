"""
CTI Universal Law: Human fMRI Validation (Natural Scenes Dataset)
================================================================
Tests whether logit(q_norm) = alpha * kappa_nearest + C holds for
HUMAN visual cortex fMRI responses to natural images.

Dataset: Natural Scenes Dataset (NSD) - Allen, St-Yves et al. 2022
         8 subjects, 7T fMRI, Kastner2015 visual ROIs
         K=12 COCO supercategories (primary-area labeling)

This is Nobel criterion #9: second species biological data.
If CTI law holds in human brain with same functional form as
mouse brain (Allen Neuropixels), this is substrate-independent universality.

Pre-registration: research/CTI_NSD_PREREGISTRATION.md
"""

import numpy as np
from scipy import stats, special
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter, defaultdict
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'nsd')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
S3_BUCKET = 's3://natural-scenes-dataset'

# Subject and sessions to process
SUBJECT = 1  # 1-indexed
N_SESSIONS = 10  # Download first 10 sessions (4489 unique images)
N_PCA = 100  # PCA dimensions for CTI computation
MIN_IMAGES_PER_CLASS = 30  # Minimum images per category for inclusion

# Kastner2015 ROI labels (from NSD documentation)
KASTNER_ROIS = {
    1: 'V1v', 2: 'V1d', 3: 'V2v', 4: 'V2d', 5: 'V3v', 6: 'V3d',
    7: 'hV4', 8: 'VO1', 9: 'VO2', 10: 'PHC1', 11: 'PHC2',
    12: 'TO2', 13: 'TO1', 14: 'LO2', 15: 'LO1',
    16: 'V3B', 17: 'V3A', 18: 'IPS0', 19: 'IPS1',
    20: 'IPS2', 21: 'IPS3', 22: 'IPS4', 23: 'IPS5',
    24: 'SPL1', 25: 'FEF'
}

# Merged ROIs for CTI analysis (combine dorsal+ventral)
MERGED_ROIS = {
    'V1': [1, 2],
    'V2': [3, 4],
    'V3': [5, 6],
    'hV4': [7],
    'VO': [8, 9],
    'PHC': [10, 11],
    'TO': [12, 13],
    'LO': [14, 15],
    'V3AB': [16, 17],
    'IPS01': [18, 19],
    'IPS23': [20, 21],
}

# COCO category mapping (category_id -> supercategory)
COCO_CATEGORIES = {
    1: 'person', 2: 'vehicle', 3: 'vehicle', 4: 'vehicle', 5: 'vehicle',
    6: 'vehicle', 7: 'vehicle', 8: 'vehicle', 9: 'vehicle',
    10: 'outdoor', 11: 'outdoor', 13: 'outdoor', 14: 'outdoor', 15: 'outdoor',
    16: 'animal', 17: 'animal', 18: 'animal', 19: 'animal', 20: 'animal',
    21: 'animal', 22: 'animal', 23: 'animal', 24: 'animal', 25: 'animal',
    27: 'accessory', 28: 'accessory', 31: 'accessory', 32: 'accessory', 33: 'accessory',
    34: 'sports', 35: 'sports', 36: 'sports', 37: 'sports', 38: 'sports',
    39: 'sports', 40: 'sports', 41: 'sports', 42: 'sports', 43: 'sports',
    44: 'kitchen', 46: 'kitchen', 47: 'kitchen', 48: 'kitchen', 49: 'kitchen',
    50: 'kitchen', 51: 'kitchen',
    52: 'food', 53: 'food', 54: 'food', 55: 'food', 56: 'food',
    57: 'food', 58: 'food', 59: 'food', 60: 'food', 61: 'food',
    62: 'furniture', 63: 'furniture', 64: 'furniture', 65: 'furniture',
    67: 'furniture', 70: 'furniture',
    72: 'electronic', 73: 'electronic', 74: 'electronic', 75: 'electronic',
    76: 'electronic', 77: 'electronic',
    78: 'appliance', 79: 'appliance', 80: 'appliance', 81: 'appliance', 82: 'appliance',
    84: 'indoor', 85: 'indoor', 86: 'indoor', 87: 'indoor',
    88: 'indoor', 89: 'indoor', 90: 'indoor',
}


def build_primary_labels():
    """Build cocoId -> primary supercategory mapping using largest annotation area."""
    import pandas as pd

    print("Building primary category labels...")

    # Load COCO annotations
    annotations_dir = os.path.join(DATA_DIR, 'annotations')
    with open(os.path.join(annotations_dir, 'instances_val2017.json')) as f:
        val_data = json.load(f)
    with open(os.path.join(annotations_dir, 'instances_train2017.json')) as f:
        train_data = json.load(f)

    cat_map = {}
    for cat in val_data['categories']:
        cat_map[cat['id']] = cat['supercategory']

    # Compute total area per supercategory per image
    img_area = defaultdict(lambda: defaultdict(float))
    for ann in val_data['annotations'] + train_data['annotations']:
        img_area[ann['image_id']][cat_map[ann['category_id']]] += ann['area']

    # Load NSD stimulus info
    nsd_info = pd.read_csv(os.path.join(DATA_DIR, 'nsd_stim_info_merged.csv'))

    # Build nsdId -> primary supercategory
    nsd_to_coco = dict(zip(nsd_info['nsdId'], nsd_info['cocoId']))
    primary_labels = {}
    for nsd_id, coco_id in nsd_to_coco.items():
        if coco_id in img_area:
            areas = img_area[coco_id]
            primary = max(areas, key=areas.get)
            primary_labels[nsd_id] = primary

    print(f"  Built labels for {len(primary_labels)} NSD images")
    cat_counts = Counter(primary_labels.values())
    for cat, count in cat_counts.most_common():
        print(f"    {cat}: {count}")

    return primary_labels


def get_trial_to_nsdid_mapping(subject_idx=0):
    """
    Map (session, trial_within_session) -> nsdId for a subject.

    Returns: dict mapping (session_0idx, trial_0idx) -> nsdId
    """
    import scipy.io

    expdesign = scipy.io.loadmat(os.path.join(DATA_DIR, 'nsd_expdesign.mat'))
    masterordering = expdesign['masterordering'].flatten()  # (30000,) 1-indexed subject image ids
    subjectim = expdesign['subjectim']  # (8, 10000) subject image id -> nsdId

    trials_per_session = 750
    mapping = {}

    for session in range(40):
        for trial in range(trials_per_session):
            global_trial = session * trials_per_session + trial
            if global_trial >= len(masterordering):
                break
            subj_img_id = masterordering[global_trial]  # 1-indexed
            nsd_id = int(subjectim[subject_idx, subj_img_id - 1])  # 0-indexed
            mapping[(session, trial)] = nsd_id

    return mapping


def download_session_betas(subject, session, fmt='nii.gz'):
    """Download one session of betas from S3 using boto3. Returns local path."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    subj_str = f'subj{subject:02d}'
    sess_str = f'session{session:02d}'
    fname = f'betas_{sess_str}.{fmt}'
    s3_key = (f'nsddata_betas/ppdata/{subj_str}/func1pt8mm/'
              f'betas_fithrf_GLMdenoise_RR/{fname}')

    local_dir = os.path.join(DATA_DIR, 'betas_tmp')
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, fname)

    if os.path.exists(local_path):
        print(f"  Already downloaded: {fname}")
        return local_path

    print(f"  Downloading {fname} (~490 MB)...")
    t0 = time.time()
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3.download_file('natural-scenes-dataset', s3_key, local_path)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(local_path) / (1024**2)
    print(f"  Downloaded {size_mb:.1f} MB in {elapsed:.0f}s ({size_mb/elapsed:.1f} MB/s)")
    return local_path


def extract_roi_responses(beta_path, roi_mask, roi_labels):
    """
    Extract per-ROI response vectors from a session's beta file.

    Args:
        beta_path: path to nii.gz beta file
        roi_mask: 3D array of ROI labels (Kastner2015)
        roi_labels: dict mapping roi_id -> list of voxel indices

    Returns:
        dict mapping roi_name -> (n_trials, n_voxels) array
    """
    import nibabel as nib

    print(f"  Loading betas from {os.path.basename(beta_path)}...")
    t0 = time.time()
    img = nib.load(beta_path)
    data = img.get_fdata()  # (x, y, z, n_trials) float32
    print(f"  Loaded: shape={data.shape}, took {time.time()-t0:.1f}s")

    n_trials = data.shape[3]
    roi_responses = {}

    for roi_name, voxel_indices in roi_labels.items():
        # voxel_indices is a tuple of (x_indices, y_indices, z_indices)
        responses = data[voxel_indices[0], voxel_indices[1], voxel_indices[2], :]
        # responses shape: (n_voxels, n_trials) -> transpose to (n_trials, n_voxels)
        roi_responses[roi_name] = responses.T.astype(np.float32)

    return roi_responses, n_trials


def build_roi_voxel_indices(roi_mask, merged_rois):
    """Build voxel index arrays for each merged ROI."""
    roi_labels = {}
    for roi_name, label_ids in merged_rois.items():
        mask = np.zeros_like(roi_mask, dtype=bool)
        for lid in label_ids:
            mask |= (roi_mask == lid)
        n_voxels = np.sum(mask)
        if n_voxels < 10:
            print(f"  Skipping {roi_name}: only {n_voxels} voxels")
            continue
        indices = np.where(mask)
        roi_labels[roi_name] = indices
        print(f"  {roi_name}: {n_voxels} voxels")
    return roi_labels


def compute_cti_per_roi(responses, labels, n_pca=100, roi_name=''):
    """
    Compute CTI metrics for one ROI.
    Reuses the same computation as Allen Neuropixels validation.

    Args:
        responses: (n_images, n_voxels) array
        labels: (n_images,) integer class labels
        n_pca: number of PCA dimensions
        roi_name: for logging

    Returns:
        dict with kappa_nearest, q_1nn, q_norm, logit_q_norm, etc.
    """
    K = len(np.unique(labels))
    n_images, n_voxels = responses.shape

    # PCA reduction
    n_pca_actual = min(n_pca, n_voxels - 1, n_images - K)
    if n_pca_actual < 5:
        print(f"  {roi_name}: too few PCA dims ({n_pca_actual}), skipping")
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

    # kappa_nearest (global, not per-class)
    centroid_dists = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)
    np.fill_diagonal(centroid_dists, np.inf)
    min_dists = centroid_dists.min(axis=1)
    kappa_per_class = min_dists / (sigma_W_global * np.sqrt(d))
    kappa_nearest = float(np.mean(kappa_per_class))

    # 1-NN accuracy (LOO using sklearn for speed)
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    # LOO cross-validation
    correct = 0
    for i in range(n_images):
        X_train = np.delete(R_pca, i, axis=0)
        y_train = np.delete(labels, i)
        knn.fit(X_train, y_train)
        pred = knn.predict(R_pca[i:i+1])[0]
        if pred == labels[i]:
            correct += 1
    q_1nn = correct / n_images

    # Normalized accuracy and logit
    q_norm = max((q_1nn - 1.0 / K) / (1.0 - 1.0 / K), 1e-6)
    q_norm = min(q_norm, 1.0 - 1e-6)
    logit_q = float(special.logit(q_norm))

    # Effective dimensionality
    explained_var_ratio = pca.explained_variance_ratio_
    d_eff = float(1.0 / np.sum(explained_var_ratio**2))

    print(f"  {roi_name}: kappa={kappa_nearest:.4f}, q_1nn={q_1nn:.3f}, "
          f"q_norm={q_norm:.4f}, logit={logit_q:.3f}, d_eff={d_eff:.1f}, "
          f"n_voxels={n_voxels}, n_images={n_images}, K={K}")

    return {
        'roi': roi_name,
        'kappa_nearest': kappa_nearest,
        'q_1nn': q_1nn,
        'q_norm': q_norm,
        'logit_q_norm': logit_q,
        'd_eff': d_eff,
        'n_voxels': n_voxels,
        'n_images': n_images,
        'K': K,
        'n_pca': n_pca_actual,
        'sigma_W_global': float(sigma_W_global),
        'kappa_per_class_mean': float(np.mean(kappa_per_class)),
        'kappa_per_class_std': float(np.std(kappa_per_class)),
    }


def compute_cti_fast(responses, labels, n_pca=100, roi_name=''):
    """
    Fast CTI computation using vectorized 1-NN (no LOO loop).
    Uses leave-one-out via precomputed distance matrix.
    """
    K = len(np.unique(labels))
    n_images, n_voxels = responses.shape

    n_pca_actual = min(n_pca, n_voxels - 1, n_images - K)
    if n_pca_actual < 5:
        print(f"  {roi_name}: too few PCA dims ({n_pca_actual}), skipping")
        return None

    pca = PCA(n_components=n_pca_actual)
    R_pca = pca.fit_transform(responses.astype(np.float64))
    d = n_pca_actual

    class_ids = np.unique(labels)
    centroids = np.array([R_pca[labels == c].mean(axis=0) for c in class_ids])

    # Within-class covariance
    all_within = []
    for c in class_ids:
        X_c = R_pca[labels == c]
        all_within.append(X_c - X_c.mean(axis=0))
    all_within = np.vstack(all_within)
    sigma_W_sq_per_dim = np.var(all_within, axis=0, ddof=1)
    tr_Sigma_W = sigma_W_sq_per_dim.sum()
    sigma_W_global = np.sqrt(tr_Sigma_W / d)

    # kappa_nearest
    centroid_dists = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)
    np.fill_diagonal(centroid_dists, np.inf)
    min_dists = centroid_dists.min(axis=1)
    kappa_per_class = min_dists / (sigma_W_global * np.sqrt(d))
    kappa_nearest = float(np.mean(kappa_per_class))

    # Vectorized LOO 1-NN using precomputed distances
    print(f"  {roi_name}: computing distance matrix ({n_images}x{n_images})...")
    # Compute all pairwise distances
    # For memory: n_images ~ 3000-5000, d=100 -> distance matrix is ~100 MB (fine)
    dist_matrix = np.linalg.norm(R_pca[:, None] - R_pca[None, :], axis=2)
    np.fill_diagonal(dist_matrix, np.inf)  # exclude self

    # Find nearest neighbor for each point
    nn_indices = np.argmin(dist_matrix, axis=1)
    nn_labels = labels[nn_indices]
    correct = np.sum(nn_labels == labels)
    q_1nn = correct / n_images

    # Normalized accuracy and logit
    q_norm = max((q_1nn - 1.0 / K) / (1.0 - 1.0 / K), 1e-6)
    q_norm = min(q_norm, 1.0 - 1e-6)
    logit_q = float(special.logit(q_norm))

    # Effective dimensionality
    explained_var_ratio = pca.explained_variance_ratio_
    d_eff = float(1.0 / np.sum(explained_var_ratio**2))

    # Equicorrelation (mean pairwise cosine of centroids)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_normed = centroids / (norms + 1e-10)
    cosines = centroids_normed @ centroids_normed.T
    mask = ~np.eye(K, dtype=bool)
    rho = float(np.mean(cosines[mask]))

    print(f"  {roi_name}: kappa={kappa_nearest:.4f}, q_1nn={q_1nn:.3f}, "
          f"q_norm={q_norm:.4f}, logit={logit_q:.3f}, rho={rho:.3f}, "
          f"n_vox={n_voxels}, n_img={n_images}, K={K}")

    return {
        'roi': roi_name,
        'kappa_nearest': kappa_nearest,
        'q_1nn': q_1nn,
        'q_norm': q_norm,
        'logit_q_norm': logit_q,
        'd_eff': d_eff,
        'rho_equicorr': rho,
        'n_voxels': n_voxels,
        'n_images': n_images,
        'K': K,
        'n_pca': n_pca_actual,
        'sigma_W_global': float(sigma_W_global),
        'kappa_per_class_mean': float(np.mean(kappa_per_class)),
        'kappa_per_class_std': float(np.std(kappa_per_class)),
    }


def main():
    import nibabel as nib

    print("=" * 70)
    print("CTI Universal Law: Human fMRI (NSD) Validation")
    print("=" * 70)

    t_start = time.time()

    # --------------------------------------------------------
    # Step 1: Build category labels
    # --------------------------------------------------------
    print("\n[Step 1] Building primary category labels...")
    primary_labels = build_primary_labels()

    # --------------------------------------------------------
    # Step 2: Build trial-to-image mapping
    # --------------------------------------------------------
    print("\n[Step 2] Building trial-to-image mapping...")
    trial_map = get_trial_to_nsdid_mapping(subject_idx=SUBJECT - 1)

    # --------------------------------------------------------
    # Step 3: Load ROI masks
    # --------------------------------------------------------
    print("\n[Step 3] Loading ROI masks...")
    roi_path = os.path.join(DATA_DIR, 'roi', 'Kastner2015_vol.nii.gz')
    roi_img = nib.load(roi_path)
    roi_mask = roi_img.get_fdata().astype(int)
    print(f"  ROI mask shape: {roi_mask.shape}")

    roi_voxel_indices = build_roi_voxel_indices(roi_mask, MERGED_ROIS)
    print(f"  {len(roi_voxel_indices)} ROIs with sufficient voxels")

    # --------------------------------------------------------
    # Step 4: Download and extract session betas
    # --------------------------------------------------------
    print(f"\n[Step 4] Processing {N_SESSIONS} sessions...")

    # Accumulate per-image, per-ROI responses
    # image_responses[roi_name][nsd_id] = list of response vectors
    image_responses = {roi: defaultdict(list) for roi in roi_voxel_indices}

    for sess_idx in range(N_SESSIONS):
        session = sess_idx + 1  # 1-indexed
        print(f"\n--- Session {session}/{N_SESSIONS} ---")

        # Download
        beta_path = download_session_betas(SUBJECT, session, fmt='nii.gz')

        # Extract ROI responses
        roi_resp, n_trials = extract_roi_responses(
            beta_path, roi_mask, roi_voxel_indices
        )

        # Map trials to nsdIds and accumulate
        for trial_idx in range(n_trials):
            key = (sess_idx, trial_idx)
            if key not in trial_map:
                continue
            nsd_id = trial_map[key]
            if nsd_id not in primary_labels:
                continue

            for roi_name in roi_voxel_indices:
                resp_vec = roi_resp[roi_name][trial_idx]
                image_responses[roi_name][nsd_id].append(resp_vec)

        # Delete raw file to save disk
        os.remove(beta_path)
        print(f"  Cleaned up raw file. "
              f"Accumulated {sum(len(v) for v in image_responses['V1'].values())} V1 trials")

    # --------------------------------------------------------
    # Step 5: Average repetitions and build response matrices
    # --------------------------------------------------------
    print("\n[Step 5] Averaging repetitions per image...")

    # Build averaged response matrix per ROI
    roi_data = {}
    for roi_name in roi_voxel_indices:
        nsd_ids = sorted(image_responses[roi_name].keys())
        if not nsd_ids:
            continue

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

        # Filter: only keep categories with enough images
        cat_counts = Counter(labels)
        valid_cats = {c for c, n in cat_counts.items() if n >= MIN_IMAGES_PER_CLASS}
        mask = np.array([l in valid_cats for l in labels])
        responses = responses[mask]
        labels = labels[mask]

        # Re-index labels to be contiguous
        unique_labels = np.unique(labels)
        label_remap = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_remap[l] for l in labels])

        int_to_label = {v: k for k, v in label_to_int.items()}
        cat_names = [int_to_label[unique_labels[i]] for i in range(len(unique_labels))]

        K = len(unique_labels)
        n_images = len(responses)
        n_voxels = responses.shape[1]

        print(f"  {roi_name}: {n_images} images, {n_voxels} voxels, K={K} categories")
        for i, cat in enumerate(cat_names):
            print(f"    {cat}: {np.sum(labels == i)} images")

        roi_data[roi_name] = {
            'responses': responses,
            'labels': labels,
            'cat_names': cat_names,
            'K': K,
        }

    # --------------------------------------------------------
    # Step 6: Compute CTI metrics per ROI
    # --------------------------------------------------------
    print("\n[Step 6] Computing CTI metrics per ROI...")

    results_per_roi = []
    for roi_name, data in roi_data.items():
        print(f"\n  Processing {roi_name}...")
        result = compute_cti_fast(
            data['responses'], data['labels'],
            n_pca=N_PCA, roi_name=roi_name
        )
        if result is not None:
            result['cat_names'] = data['cat_names']
            results_per_roi.append(result)

    # --------------------------------------------------------
    # Step 7: Fit CTI law across ROIs
    # --------------------------------------------------------
    print("\n[Step 7] Fitting CTI law across ROIs...")

    kappas = [r['kappa_nearest'] for r in results_per_roi]
    logits = [r['logit_q_norm'] for r in results_per_roi]
    rois = [r['roi'] for r in results_per_roi]

    if len(kappas) >= 3:
        # Linear fit: logit(q_norm) = alpha * kappa + C
        slope, intercept, r_value, p_value, std_err = stats.linregress(kappas, logits)
        r_sq = r_value ** 2

        print(f"\n  CTI Law Fit (human fMRI, n={len(kappas)} ROIs):")
        print(f"    alpha = {slope:.3f} +/- {std_err:.3f}")
        print(f"    C     = {intercept:.3f}")
        print(f"    r     = {r_value:.3f}")
        print(f"    R^2   = {r_sq:.3f}")
        print(f"    p     = {p_value:.2e}")

        # Also compute Spearman
        rho_spearman, p_spearman = stats.spearmanr(kappas, logits)
        print(f"    Spearman rho = {rho_spearman:.3f}, p = {p_spearman:.2e}")

        # Pass/fail criteria (aligned with Allen Neuropixels)
        pass_r = abs(r_value) > 0.50
        pass_direction = slope > 0

        # Equicorrelation analysis
        rhos_equicorr = [r['rho_equicorr'] for r in results_per_roi]
        rho_mean = np.mean(rhos_equicorr)
        rho_cv = np.std(rhos_equicorr) / abs(rho_mean) * 100

        print(f"\n  Equicorrelation across ROIs:")
        print(f"    mean rho = {rho_mean:.3f}")
        print(f"    CV(rho)  = {rho_cv:.1f}%")
        for roi, rho_eq in zip(rois, rhos_equicorr):
            print(f"    {roi}: rho={rho_eq:.3f}")

        law_fit = {
            'alpha': float(slope),
            'alpha_se': float(std_err),
            'C': float(intercept),
            'r': float(r_value),
            'R_sq': float(r_sq),
            'p_value': float(p_value),
            'spearman_rho': float(rho_spearman),
            'spearman_p': float(p_spearman),
            'n_rois': len(kappas),
            'pass_r': pass_r,
            'pass_direction': pass_direction,
            'rho_equicorr_mean': float(rho_mean),
            'rho_equicorr_cv': float(rho_cv),
        }
    else:
        print("  WARNING: Too few ROIs for regression")
        law_fit = {'error': 'too_few_rois', 'n_rois': len(kappas)}

    # --------------------------------------------------------
    # Step 8: Save results
    # --------------------------------------------------------
    elapsed = time.time() - t_start

    output = {
        'experiment': 'CTI Human fMRI (NSD) Validation',
        'dataset': 'Natural Scenes Dataset (Allen, St-Yves et al. 2022)',
        'subject': SUBJECT,
        'n_sessions': N_SESSIONS,
        'n_pca': N_PCA,
        'min_images_per_class': MIN_IMAGES_PER_CLASS,
        'per_roi': results_per_roi,
        'law_fit': law_fit,
        'elapsed_seconds': float(elapsed),
    }

    out_path = os.path.join(RESULTS_DIR, 'cti_nsd_human_fmri.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if 'alpha' in law_fit:
        print(f"  CTI Law: logit(q) = {law_fit['alpha']:.3f} * kappa + {law_fit['C']:.3f}")
        print(f"  r = {law_fit['r']:.3f} (PASS: {law_fit['pass_r']})")
        print(f"  R^2 = {law_fit['R_sq']:.3f}")
        print(f"  Direction: alpha > 0 = {law_fit['pass_direction']}")
        print(f"  Equicorrelation: rho = {law_fit['rho_equicorr_mean']:.3f} (CV={law_fit['rho_equicorr_cv']:.1f}%)")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print("=" * 70)


if __name__ == '__main__':
    main()
