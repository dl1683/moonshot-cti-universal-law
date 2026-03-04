#!/usr/bin/env python -u
"""
CTI Cross-Modal Alpha(rho) Unification Test
=============================================
Tests whether a SINGLE formula predicts alpha across ALL modalities:

    alpha = sqrt(4/pi) / sqrt(1-rho)

where rho is the equicorrelation (mean whitened cosine similarity of
centroid-difference vectors).

If this holds, it means alpha is NOT a free parameter per modality --
it is DERIVED from the representation geometry alone.

MODALITIES TESTED:
  1. NLP (from existing alpha-rho results: rho~0.46, alpha~1.48)
  2. Audio (Speech Commands K=36, 2 representative models)
  3. Vision (CIFAR-10 K=10, ViT and ResNet models)
  4. Biology (from existing Allen results: rho~0.47, alpha~0.03-0.07)

OUTPUT: results/cti_cross_modal_rho.json
"""

import json
import time
import gc
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.special import logit as sp_logit
from sklearn.decomposition import TruncatedSVD
import torch
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_PCA = 256

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70, flush=True)
print("CTI Cross-Modal Alpha(rho) Unification Test", flush=True)
print("=" * 70, flush=True)
print(f"Device: {DEVICE}", flush=True)


# ============================================================
# SHARED: compute_rho and compute_cti_metrics
# ============================================================
def compute_rho(embeddings, labels, classes):
    """Compute mean Sigma_W-whitened cosine similarity of centroid diffs.
    Uses the SAME convention as cti_alpha_rho_multidataset.py for consistency."""
    K_local = len(classes)
    N, d = embeddings.shape

    centroids = {}
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            centroids[c] = embeddings[mask].mean(0).astype(np.float64)
    if len(centroids) < K_local:
        return float("nan"), float("nan")

    centroid_array = np.array([centroids[c] for c in classes])

    Xc_list = []
    for c in classes:
        mask = labels == c
        if mask.sum() >= 2:
            Xc_list.append((embeddings[mask] - centroids[c]).astype(np.float64))
    Z = np.concatenate(Xc_list, axis=0)
    N_total = len(Z)

    n_comp = min(N_PCA, d, N_total - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(Z)
    V = svd.components_.T
    Lambda = (svd.singular_values_ ** 2) / N_total
    sqrt_Lambda = np.sqrt(Lambda + 1e-12)

    rho_per_class = []
    for i_c, c in enumerate(classes):
        other_idx = [i for i in range(K_local) if i != i_c]
        deltas = centroid_array[other_idx] - centroids[c]
        proj = deltas @ V
        whitened = proj * sqrt_Lambda[None, :]
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        w_norm = whitened / norms
        cos_mat = w_norm @ w_norm.T
        n_off = K_local - 1
        off_vals = cos_mat[~np.eye(n_off, dtype=bool)]
        rho_per_class.append(float(off_vals.mean()))

    return float(np.mean(rho_per_class)), float(np.std(rho_per_class))


def compute_cti_metrics(embeddings, labels, K):
    """Compute kappa_nearest, 1-NN accuracy, logit_q, and rho."""
    d = embeddings.shape[1]
    classes = sorted(list(set(labels)))

    # Class centroids
    mu = {c: embeddings[labels == c].mean(0) for c in classes}

    # Within-class scatter
    within_var = sum(np.sum((embeddings[labels == c] - mu[c])**2) for c in classes)
    sigma_W = float(np.sqrt(within_var / (len(embeddings) * d)))

    # kappa_nearest per class
    kappas = []
    for c in classes:
        dists = [np.linalg.norm(mu[c] - mu[j]) for j in classes if j != c]
        kappas.append(min(dists) / (sigma_W * np.sqrt(d)))
    kappa_nearest = float(np.mean(kappas))

    # 1-NN accuracy (LOO using centroids)
    from sklearn.metrics import pairwise_distances as pdist
    D = pdist(embeddings, np.array([mu[c] for c in classes]))
    correct = 0
    for i in range(len(embeddings)):
        true_c = labels[i]
        pred_c = classes[np.argmin(D[i])]
        if pred_c == true_c:
            correct += 1
    acc = correct / len(embeddings)
    q_norm = (acc - 1.0/K) / (1.0 - 1.0/K)
    q_norm = max(min(q_norm, 0.9999), 0.0001)
    logit_q = float(sp_logit(q_norm))

    # Equicorrelation rho
    rho_mean, rho_std = compute_rho(embeddings, labels, classes)

    return {
        "kappa_nearest": kappa_nearest,
        "acc_1nn": acc,
        "q_norm": q_norm,
        "logit_q": logit_q,
        "rho_mean": rho_mean,
        "rho_std": rho_std,
        "sigma_W": sigma_W,
        "K": K,
    }


# ============================================================
# AUDIO: measure rho for 2 representative speech models
# ============================================================
def measure_audio_rho():
    """Measure rho for representative audio models on Speech Commands."""
    print("\n" + "=" * 60, flush=True)
    print("[AUDIO] Measuring rho on Speech Commands (K=36)", flush=True)
    print("=" * 60, flush=True)

    from transformers import AutoFeatureExtractor, AutoModel
    from datasets import load_dataset

    ds = load_dataset("google/speech_commands", "v0.02", split="test",
                      trust_remote_code=True)
    label_names = ds.features["label"].names
    K = len(label_names)
    print(f"  {len(ds)} samples, K={K}", flush=True)

    audio_models = [
        ("WavLM-Base+", "microsoft/wavlm-base-plus", 12, 768),
        ("HuBERT-Base",  "facebook/hubert-base-ls960", 12, 768),
    ]

    results = {}
    for model_name, hf_id, n_layers, hidden_size in audio_models:
        print(f"\n  Processing {model_name}...", flush=True)
        target_layer = round(2 * n_layers / 3)

        feature_extractor = AutoFeatureExtractor.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id).to(DEVICE).eval()

        all_embeddings = []
        n = len(ds)
        batch_size = 16

        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = ds[i:i + batch_size]
                audio_arrays = [a["array"] for a in batch["audio"]]
                sr = batch["audio"][0]["sampling_rate"]

                inputs = feature_extractor(
                    audio_arrays, sampling_rate=sr,
                    return_tensors="pt", padding=True
                ).to(DEVICE)

                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[target_layer]
                pooled = hidden.mean(dim=1)
                all_embeddings.append(pooled.cpu().numpy())

                if (i // batch_size) % 50 == 0:
                    print(f"    {i}/{n}...", flush=True)

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        labels = np.array(ds["label"])

        # PCA if needed
        if embeddings.shape[1] > N_PCA:
            svd = TruncatedSVD(n_components=N_PCA, random_state=SEED)
            embeddings = svd.fit_transform(embeddings)

        metrics = compute_cti_metrics(embeddings, labels, K)
        results[model_name] = metrics
        print(f"    kappa={metrics['kappa_nearest']:.4f}, acc={metrics['acc_1nn']:.3f}, "
              f"rho={metrics['rho_mean']:.4f} +/- {metrics['rho_std']:.4f}", flush=True)

        del model, feature_extractor, embeddings, all_embeddings
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ============================================================
# VISION: measure rho for ViT on CIFAR-10
# ============================================================
def measure_vision_rho():
    """Measure rho for vision models on CIFAR-10."""
    print("\n" + "=" * 60, flush=True)
    print("[VISION] Measuring rho on CIFAR-10 (K=10)", flush=True)
    print("=" * 60, flush=True)

    from transformers import ViTModel, ViTFeatureExtractor
    from torchvision import datasets, transforms

    # Load CIFAR-10 test set
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.CIFAR10(root=str(REPO_ROOT / "data" / "cifar10"),
                                train=False, download=True,
                                transform=transform)
    K = 10
    print(f"  {len(test_ds)} samples, K={K}", flush=True)

    vision_models = [
        ("ViT-Base-16-224", "google/vit-base-patch16-224", 12),
    ]

    results = {}
    for model_name, hf_id, n_layers in vision_models:
        print(f"\n  Processing {model_name}...", flush=True)
        target_layer = round(2 * n_layers / 3)

        model = ViTModel.from_pretrained(hf_id).to(DEVICE).eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}, target layer: {target_layer}/{n_layers}", flush=True)

        all_embeddings = []
        all_labels = []
        batch_size = 64
        loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )

        with torch.no_grad():
            for i_batch, (images, targets) in enumerate(loader):
                images = images.to(DEVICE)
                outputs = model(images, output_hidden_states=True)
                hidden = outputs.hidden_states[target_layer]
                # ViT: use CLS token or mean pool
                pooled = hidden[:, 0, :]  # CLS token
                all_embeddings.append(pooled.cpu().numpy())
                all_labels.append(targets.numpy())

                if i_batch % 25 == 0:
                    print(f"    batch {i_batch}/{len(loader)}...", flush=True)

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        labels = np.concatenate(all_labels, axis=0)

        # PCA if needed
        if embeddings.shape[1] > N_PCA:
            svd = TruncatedSVD(n_components=N_PCA, random_state=SEED)
            embeddings = svd.fit_transform(embeddings)

        metrics = compute_cti_metrics(embeddings, labels, K)
        results[model_name] = metrics
        print(f"    kappa={metrics['kappa_nearest']:.4f}, acc={metrics['acc_1nn']:.3f}, "
              f"rho={metrics['rho_mean']:.4f} +/- {metrics['rho_std']:.4f}", flush=True)

        del model, embeddings, all_embeddings
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ============================================================
# VISION: ResNet50 on CIFAR-100
# ============================================================
def measure_vision_resnet_rho():
    """Measure rho for ResNet50 on CIFAR-100."""
    print("\n" + "=" * 60, flush=True)
    print("[VISION] Measuring rho for ResNet50 on CIFAR-100 (K=100)", flush=True)
    print("=" * 60, flush=True)

    import torchvision.models as tv_models
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.CIFAR100(root=str(REPO_ROOT / "data" / "cifar100"),
                                 train=False, download=True,
                                 transform=transform)
    K = 100
    print(f"  {len(test_ds)} samples, K={K}", flush=True)

    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    model = model.to(DEVICE).eval()

    # Hook to capture layer3 features (penultimate block)
    features_list = []
    def hook_fn(module, input, output):
        features_list.append(output.mean(dim=[2, 3]).cpu().numpy())  # Global avg pool

    handle = model.layer3.register_forward_hook(hook_fn)

    all_labels = []
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=False
    )

    with torch.no_grad():
        for i_batch, (images, targets) in enumerate(loader):
            images = images.to(DEVICE)
            _ = model(images)
            all_labels.append(targets.numpy())
            if i_batch % 25 == 0:
                print(f"    batch {i_batch}/{len(loader)}...", flush=True)

    handle.remove()
    embeddings = np.concatenate(features_list, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0)
    print(f"  Embeddings shape: {embeddings.shape}", flush=True)

    # PCA
    if embeddings.shape[1] > N_PCA:
        svd = TruncatedSVD(n_components=N_PCA, random_state=SEED)
        embeddings = svd.fit_transform(embeddings)

    metrics = compute_cti_metrics(embeddings, labels, K)
    print(f"    kappa={metrics['kappa_nearest']:.4f}, acc={metrics['acc_1nn']:.3f}, "
          f"rho={metrics['rho_mean']:.4f} +/- {metrics['rho_std']:.4f}", flush=True)

    del model, embeddings
    gc.collect()
    torch.cuda.empty_cache()

    return {"ResNet50-CIFAR100": metrics}


# ============================================================
# MAIN: Combine all modalities
# ============================================================
def main():
    t0 = time.time()
    results = {}

    # --- 1. NLP (from existing results) ---
    print("\n[NLP] Using existing alpha-rho measurements...", flush=True)
    nlp_rho_file = RESULTS_DIR / "cti_alpha_rho_multidataset.json"
    with open(nlp_rho_file) as f:
        nlp_data = json.load(f)
    nlp_rhos = [v["rho_pooled"] for v in nlp_data["per_model"].values()]
    nlp_rho = float(np.mean(nlp_rhos))
    nlp_alpha = nlp_data["aggregate"]["alpha_loao_mean"]
    print(f"  NLP: rho={nlp_rho:.4f}, alpha={nlp_alpha:.4f}", flush=True)
    results["NLP_decoders"] = {
        "modality": "NLP",
        "rho_mean": nlp_rho,
        "alpha_measured": nlp_alpha,
        "K": "4-77 (multi-dataset)",
        "n_models": 11,
        "source": "cti_alpha_rho_multidataset.json",
    }

    # --- 2. Biology (from existing Allen results) ---
    print("\n[BIOLOGY] Using existing Allen equicorrelation measurements...", flush=True)
    bio_equi_file = RESULTS_DIR / "cti_allen_equicorrelation.json"
    if bio_equi_file.exists():
        with open(bio_equi_file) as f:
            bio_data = json.load(f)
        bio_rho = bio_data["summary"]["mean_rho"]
        print(f"  Biology (Mouse V1): rho={bio_rho:.4f}", flush=True)
        results["Mouse_V1"] = {
            "modality": "Biology",
            "rho_mean": bio_rho,
            "alpha_measured": None,  # Biological alpha uses different renormalization
            "K": 118,
            "source": "cti_allen_equicorrelation.json",
            "note": "Bio alpha not directly comparable (different renormalization)",
        }

    # --- 3. Audio ---
    audio_results = measure_audio_rho()
    for name, metrics in audio_results.items():
        results[f"Audio_{name}"] = {
            "modality": "Audio",
            **metrics,
            "source": "computed (Speech Commands K=36)",
        }

    # --- 4. Vision: ViT on CIFAR-10 ---
    vit_results = measure_vision_rho()
    for name, metrics in vit_results.items():
        results[f"Vision_{name}"] = {
            "modality": "Vision",
            **metrics,
            "source": "computed (CIFAR-10 K=10)",
        }

    # --- 5. Vision: ResNet50 on CIFAR-100 ---
    resnet_results = measure_vision_resnet_rho()
    for name, metrics in resnet_results.items():
        results[f"Vision_{name}"] = {
            "modality": "Vision",
            **metrics,
            "source": "computed (CIFAR-100 K=100)",
        }

    # ============================================================
    # ALPHA(RHO) PREDICTION
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("ALPHA(RHO) UNIVERSAL FORMULA TEST", flush=True)
    print("=" * 60, flush=True)
    print(f"Formula: alpha = sqrt(4/pi) / sqrt(1-rho)", flush=True)

    formula_test = []
    for name, data in results.items():
        rho = data.get("rho_mean")
        if rho is None or np.isnan(rho):
            continue

        alpha_predicted = np.sqrt(4.0 / np.pi) / np.sqrt(1.0 - rho)

        # For NLP and audio, we have measured alpha
        alpha_measured = data.get("alpha_measured")
        # For audio/vision within-modality, we need to get alpha from the original results
        modality = data.get("modality", "")

        formula_test.append({
            "name": name,
            "modality": modality,
            "rho": rho,
            "alpha_predicted": float(alpha_predicted),
            "alpha_measured": alpha_measured,
            "K": data.get("K"),
        })

        status = ""
        if alpha_measured is not None:
            error_pct = abs(alpha_predicted - alpha_measured) / alpha_measured * 100
            status = f"error={error_pct:.1f}%"
        print(f"  {name:30s}: rho={rho:.4f} -> alpha_pred={alpha_predicted:.3f} "
              f"(measured={alpha_measured}) {status}", flush=True)

    # Add known measured alphas for modalities where we have them
    # Audio alpha = 4.669 (from 7-model fit)
    audio_alpha_measured = 4.669
    # ViT-Base CIFAR-10 alpha = 0.592
    vit_alpha_measured = 0.592
    # ResNet50 CIFAR-100 alpha = 4.418
    resnet_alpha_measured = 4.418

    # Build the cross-modal comparison
    print("\n" + "-" * 60, flush=True)
    print("CROSS-MODAL ALPHA(RHO) COMPARISON", flush=True)
    print("-" * 60, flush=True)

    cross_modal = []
    # NLP
    if "NLP_decoders" in results:
        rho = results["NLP_decoders"]["rho_mean"]
        alpha_pred = np.sqrt(4.0 / np.pi) / np.sqrt(1.0 - rho)
        cross_modal.append({
            "modality": "NLP decoders",
            "rho": rho,
            "alpha_measured": nlp_alpha,
            "alpha_predicted": float(alpha_pred),
            "error_pct": abs(alpha_pred - nlp_alpha) / nlp_alpha * 100,
            "K": "4-77",
        })

    # Audio (use mean rho across measured models)
    audio_rhos = [v["rho_mean"] for k, v in results.items()
                  if k.startswith("Audio_") and "rho_mean" in v and not np.isnan(v["rho_mean"])]
    if audio_rhos:
        rho = np.mean(audio_rhos)
        alpha_pred = np.sqrt(4.0 / np.pi) / np.sqrt(1.0 - rho)
        cross_modal.append({
            "modality": "Audio (speech)",
            "rho": float(rho),
            "alpha_measured": audio_alpha_measured,
            "alpha_predicted": float(alpha_pred),
            "error_pct": abs(alpha_pred - audio_alpha_measured) / audio_alpha_measured * 100,
            "K": 36,
        })

    # ViT CIFAR-10
    vit_key = [k for k in results if k.startswith("Vision_ViT")]
    if vit_key:
        rho = results[vit_key[0]]["rho_mean"]
        alpha_pred = np.sqrt(4.0 / np.pi) / np.sqrt(1.0 - rho)
        cross_modal.append({
            "modality": "ViT (CIFAR-10)",
            "rho": float(rho),
            "alpha_measured": vit_alpha_measured,
            "alpha_predicted": float(alpha_pred),
            "error_pct": abs(alpha_pred - vit_alpha_measured) / vit_alpha_measured * 100,
            "K": 10,
        })

    # ResNet50 CIFAR-100
    resnet_key = [k for k in results if k.startswith("Vision_ResNet")]
    if resnet_key:
        rho = results[resnet_key[0]]["rho_mean"]
        alpha_pred = np.sqrt(4.0 / np.pi) / np.sqrt(1.0 - rho)
        cross_modal.append({
            "modality": "CNN (CIFAR-100)",
            "rho": float(rho),
            "alpha_measured": resnet_alpha_measured,
            "alpha_predicted": float(alpha_pred),
            "error_pct": abs(alpha_pred - resnet_alpha_measured) / resnet_alpha_measured * 100,
            "K": 100,
        })

    for cm in cross_modal:
        print(f"  {cm['modality']:20s}: rho={cm['rho']:.4f} -> "
              f"alpha_pred={cm['alpha_predicted']:.3f} vs measured={cm['alpha_measured']:.3f} "
              f"(error={cm['error_pct']:.1f}%)", flush=True)

    # Overall correlation
    if len(cross_modal) >= 3:
        rhos = [cm["rho"] for cm in cross_modal]
        alpha_m = [cm["alpha_measured"] for cm in cross_modal]
        alpha_p = [cm["alpha_predicted"] for cm in cross_modal]
        r_pred, p_pred = pearsonr(alpha_p, alpha_m)
        mean_error = np.mean([cm["error_pct"] for cm in cross_modal])
        print(f"\n  Prediction quality: r={r_pred:.3f}, p={p_pred:.4f}", flush=True)
        print(f"  Mean error: {mean_error:.1f}%", flush=True)

        # Also test: does rho alone predict alpha?
        r_rho_alpha, p_rho_alpha = pearsonr(rhos, alpha_m)
        print(f"  r(rho, alpha_measured): {r_rho_alpha:.3f}, p={p_rho_alpha:.4f}", flush=True)

    # ============================================================
    # SAVE
    # ============================================================
    elapsed = time.time() - t0
    output = {
        "experiment": "Cross-Modal Alpha(rho) Unification Test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "formula": "alpha = sqrt(4/pi) / sqrt(1-rho)",
        "per_model_results": results,
        "cross_modal_comparison": cross_modal,
        "elapsed_seconds": elapsed,
    }

    if len(cross_modal) >= 3:
        output["summary"] = {
            "n_modalities": len(cross_modal),
            "r_predicted_vs_measured": float(r_pred),
            "p_value": float(p_pred),
            "mean_error_pct": float(mean_error),
            "r_rho_vs_alpha": float(r_rho_alpha),
        }

    out_path = RESULTS_DIR / "cti_cross_modal_rho.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for cm in cross_modal:
        star = "***" if cm["error_pct"] < 15 else "**" if cm["error_pct"] < 30 else ""
        print(f"  {cm['modality']:20s}: error={cm['error_pct']:.1f}% {star}", flush=True)
    if len(cross_modal) >= 3:
        print(f"\n  Overall: r={r_pred:.3f}, mean error={mean_error:.1f}%", flush=True)
        if r_pred > 0.90 and mean_error < 25:
            print("  >>> FORMULA VALIDATED CROSS-MODALLY <<<", flush=True)
        elif r_pred > 0.70:
            print("  >>> PARTIAL SUPPORT (direction correct, moderate accuracy) <<<", flush=True)
        else:
            print("  >>> FORMULA DOES NOT GENERALIZE ACROSS MODALITIES <<<", flush=True)


if __name__ == "__main__":
    main()
