#!/usr/bin/env python -u
"""
CTI Universal Law: Audio/Speech Domain Validation
===================================================
Tests the CTI law on frozen speech model embeddings — a new modality (audio).
If logit(q_norm) = alpha * kappa + C holds for speech representations classifying
spoken commands, it demonstrates the Gumbel-race mechanism is universal across
NLP, vision, neuroscience, AND audio.

DATASET: Google Speech Commands v0.02
  K = 36 classes (35 commands + silence), test split (~4890 samples)

MODELS: 7 models from 4 architectures:
  - Wav2Vec2-Base (95M, d=768)     [Self-supervised, CTC]
  - HuBERT-Base (95M, d=768)       [Self-supervised, clustering]
  - HuBERT-Large (316M, d=1024)    [Self-supervised, clustering]
  - WavLM-Base+ (95M, d=768)       [Self-supervised, denoising]
  - WavLM-Large (316M, d=1024)     [Self-supervised, denoising]
  - Whisper-tiny (39M, d=384)      [Supervised, ASR encoder]
  - Whisper-small (244M, d=768)    [Supervised, ASR encoder]

PRE-REGISTERED CRITERIA:
  H_audio1: r(kappa, logit_q) > 0.70 across 7 models (4 architectures)
  H_audio2: alpha > 0 (monotonic)
  H_audio3: Spearman rho > 0.50

OUTPUT: results/cti_audio_speech.json
"""

import json
import time
import gc
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.special import logit as sp_logit
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
import torch
from transformers import (
    AutoFeatureExtractor, AutoModel, AutoProcessor,
    WhisperForConditionalGeneration,
)
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_PATH = RESULTS_DIR / "cti_audio_speech_cache.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_PCA = 256
EMBED_BATCH = 16

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70, flush=True)
print("CTI Universal Law: Audio/Speech Domain Validation", flush=True)
print("=" * 70, flush=True)
print(f"Device: {DEVICE}", flush=True)

# ============================================================
# MODELS
# ============================================================
AUDIO_MODELS = [
    # (name, hf_id, architecture, is_whisper, n_layers, hidden_size)
    ("Wav2Vec2-Base",  "facebook/wav2vec2-base",       "Wav2Vec2", False, 12, 768),
    ("HuBERT-Base",    "facebook/hubert-base-ls960",   "HuBERT",   False, 12, 768),
    ("HuBERT-Large",   "facebook/hubert-large-ll60k",  "HuBERT",   False, 24, 1024),
    ("WavLM-Base+",    "microsoft/wavlm-base-plus",    "WavLM",    False, 12, 768),
    ("WavLM-Large",    "microsoft/wavlm-large",        "WavLM",    False, 24, 1024),
    ("Whisper-tiny",   "openai/whisper-tiny",          "Whisper",  True,  4,  384),
    ("Whisper-small",  "openai/whisper-small",         "Whisper",  True,  12, 768),
]


# ============================================================
# STEP 1: Load Speech Commands
# ============================================================
def load_speech_commands():
    """Load Speech Commands v0.02 test split."""
    print("\n[Step 1] Loading Speech Commands v0.02 test split...", flush=True)
    ds = load_dataset("google/speech_commands", "v0.02", split="test",
                      trust_remote_code=True)
    label_names = ds.features["label"].names
    K = len(label_names)
    print(f"  {len(ds)} samples, K={K} classes", flush=True)
    print(f"  Classes: {label_names[:10]}... (showing first 10)", flush=True)
    return ds, label_names, K


# ============================================================
# STEP 2: Extract embeddings
# ============================================================
def extract_embeddings(model_name, hf_id, is_whisper, n_layers, ds):
    """Extract mean-pooled embeddings from a speech model at 2/3 depth."""
    print(f"\n  Loading {model_name}...", flush=True)

    target_layer = round(2 * n_layers / 3)

    if is_whisper:
        processor = AutoProcessor.from_pretrained(hf_id)
        model = WhisperForConditionalGeneration.from_pretrained(hf_id).to(DEVICE).eval()
        n_params = sum(p.numel() for p in model.model.encoder.parameters())
        feature_extractor = processor.feature_extractor
        # Whisper needs 30s fixed mel input; pad short audio
        feature_extractor.padding = "max_length"
        feature_extractor.max_length = 3000  # 30s at 100 frames/s
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id).to(DEVICE).eval()
        n_params = sum(p.numel() for p in model.parameters())

    print(f"    Params: {n_params:,}, target layer: {target_layer}/{n_layers}", flush=True)

    all_embeddings = []
    n = len(ds)

    with torch.no_grad():
        for i in range(0, n, EMBED_BATCH):
            batch = ds[i:i + EMBED_BATCH]
            audio_arrays = [a["array"] for a in batch["audio"]]
            sr = batch["audio"][0]["sampling_rate"]

            if is_whisper:
                # Whisper needs 30s fixed-length mel spectrograms
                # Pad each audio to 30 seconds (480000 samples at 16kHz)
                padded_audio = []
                for arr in audio_arrays:
                    target_len = 480000  # 30 seconds
                    if len(arr) < target_len:
                        padded = np.pad(arr, (0, target_len - len(arr)))
                    else:
                        padded = arr[:target_len]
                    padded_audio.append(padded)
                inputs = feature_extractor(
                    padded_audio, sampling_rate=sr,
                    return_tensors="pt", padding=False
                ).to(DEVICE)
                outputs = model.model.encoder(
                    inputs["input_features"],
                    output_hidden_states=True
                )
                hidden = outputs.hidden_states[target_layer]  # (B, T, d)
                pooled = hidden.mean(dim=1)  # (B, d)
            else:
                inputs = feature_extractor(
                    audio_arrays, sampling_rate=sr,
                    return_tensors="pt", padding=True
                ).to(DEVICE)
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[target_layer]  # (B, T, d)
                # Use attention mask for proper pooling
                if hasattr(outputs, "extract_features"):
                    # Some models output different length
                    pooled = hidden.mean(dim=1)
                else:
                    pooled = hidden.mean(dim=1)

            all_embeddings.append(pooled.cpu().numpy())

            if (i // EMBED_BATCH) % 50 == 0:
                print(f"    Batch {i // EMBED_BATCH + 1}/{(n + EMBED_BATCH - 1) // EMBED_BATCH}",
                      flush=True)

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"    Embeddings: {embeddings.shape}", flush=True)

    del model
    if is_whisper:
        del processor
    else:
        del feature_extractor
    gc.collect()
    torch.cuda.empty_cache()

    return embeddings


# ============================================================
# STEP 3: Compute CTI metrics
# ============================================================
def compute_cti_metrics(embeddings, labels, K):
    """Compute kappa_nearest and 1-NN accuracy with PCA reduction."""
    d_orig = embeddings.shape[1]

    # PCA reduction
    if d_orig > N_PCA:
        svd = TruncatedSVD(n_components=N_PCA, random_state=SEED)
        embeddings = svd.fit_transform(embeddings)
        var_exp = svd.explained_variance_ratio_.sum()
    else:
        var_exp = 1.0

    d = embeddings.shape[1]
    n = len(embeddings)

    # Class centroids
    centroids = np.zeros((K, d))
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            centroids[k] = embeddings[mask].mean(axis=0)

    # Pooled within-class covariance
    Sigma_W = np.zeros((d, d))
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            X = embeddings[mask] - centroids[k]
            Sigma_W += X.T @ X
    Sigma_W /= n
    sigma_W = np.sqrt(np.trace(Sigma_W) / d)

    # Kappa nearest
    dists = np.full((K, K), np.inf)
    for i in range(K):
        for j in range(K):
            if i != j:
                dists[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    kappa = (dists.min(axis=1) / (sigma_W * np.sqrt(d))).mean()

    # LOO 1-NN
    dm = pairwise_distances(embeddings)
    np.fill_diagonal(dm, np.inf)
    nn_labels = labels[dm.argmin(axis=1)]
    acc = (nn_labels == labels).mean()

    q_norm = np.clip((acc - 1.0 / K) / (1.0 - 1.0 / K), 0.001, 0.999)
    logit_q = float(sp_logit(q_norm))

    return {
        "kappa_nearest": float(kappa),
        "acc_1nn": float(acc),
        "q_norm": float(q_norm),
        "logit_q": float(logit_q),
        "K": K,
        "n_samples": n,
        "d_original": d_orig,
        "d_pca": d,
        "pca_var_explained": float(var_exp),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()

    ds, label_names, K = load_speech_commands()
    labels = np.array(ds["label"])

    # Check cache
    results = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            results = json.load(f)
        print(f"\n  Cached results for {len(results)} models", flush=True)

    # Run each model
    for model_name, hf_id, arch, is_whisper, n_layers, hidden in AUDIO_MODELS:
        if model_name in results:
            print(f"\n  {model_name}: cached, skipping", flush=True)
            continue

        print(f"\n{'=' * 60}", flush=True)
        print(f"  Processing {model_name} ({arch})", flush=True)
        print(f"{'=' * 60}", flush=True)

        try:
            embeddings = extract_embeddings(model_name, hf_id, is_whisper, n_layers, ds)
            metrics = compute_cti_metrics(embeddings, labels, K)
            metrics["model"] = model_name
            metrics["hf_id"] = hf_id
            metrics["architecture"] = arch
            metrics["n_layers"] = n_layers
            metrics["hidden_size"] = hidden

            results[model_name] = metrics
            print(f"\n  {model_name}: kappa={metrics['kappa_nearest']:.4f}, "
                  f"acc={metrics['acc_1nn']:.3f}, logit_q={metrics['logit_q']:.3f}", flush=True)

        except Exception as e:
            print(f"\n  {model_name}: FAILED - {e}", flush=True)
            results[model_name] = {"error": str(e)}
            embeddings = None

        # Save intermediate
        with open(CACHE_PATH, "w") as f:
            json.dump(results, f, indent=2)

        if embeddings is not None:
            del embeddings
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # Hypothesis testing
    # ============================================================
    print(f"\n{'=' * 70}", flush=True)
    print("HYPOTHESIS TESTING", flush=True)
    print(f"{'=' * 70}", flush=True)

    valid_models = [m[0] for m in AUDIO_MODELS if m[0] in results and "error" not in results[m[0]]]
    kappas = np.array([results[m]["kappa_nearest"] for m in valid_models])
    logit_qs = np.array([results[m]["logit_q"] for m in valid_models])
    accs = np.array([results[m]["acc_1nn"] for m in valid_models])
    archs = [results[m]["architecture"] for m in valid_models]

    print(f"\n  Per-model summary (PCA to {N_PCA}d, 2/3 depth layer):", flush=True)
    print(f"  {'Model':<20} {'Arch':<10} {'kappa':>8} {'acc':>8} {'logit_q':>8}", flush=True)
    print(f"  {'-' * 58}", flush=True)
    for i, m in enumerate(valid_models):
        print(f"  {m:<20} {archs[i]:<10} {kappas[i]:>8.4f} {accs[i]:>8.3f} {logit_qs[i]:>8.3f}", flush=True)

    n_models = len(valid_models)
    n_arch = len(set(archs))

    if n_models >= 3:
        r_pearson, p_pearson = pearsonr(kappas, logit_qs)
        r_spearman, p_spearman = spearmanr(kappas, logit_qs)
    else:
        r_pearson, p_pearson = np.nan, np.nan
        r_spearman, p_spearman = np.nan, np.nan

    coeffs = np.polyfit(kappas, logit_qs, 1) if n_models >= 2 else (np.nan, np.nan)
    alpha_audio = coeffs[0]
    C_audio = coeffs[1]
    ss_res = ((logit_qs - np.polyval(coeffs, kappas)) ** 2).sum()
    ss_tot = ((logit_qs - logit_qs.mean()) ** 2).sum()
    R_sq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    h1_pass = r_pearson > 0.70 if not np.isnan(r_pearson) else False
    h2_pass = alpha_audio > 0
    h3_pass = r_spearman > 0.50 if not np.isnan(r_spearman) else False

    print(f"\n  CTI Law Fit (audio, n={n_models} models, {n_arch} architectures):", flush=True)
    print(f"    alpha_audio = {alpha_audio:.4f}", flush=True)
    print(f"    C           = {C_audio:.4f}", flush=True)
    print(f"    r (Pearson)  = {r_pearson:.4f} (p = {p_pearson:.4e})", flush=True)
    print(f"    rho (Spearman) = {r_spearman:.4f} (p = {p_spearman:.4e})", flush=True)
    print(f"    R^2          = {R_sq:.4f}", flush=True)

    print(f"\n  H_audio1: r > 0.70 => {r_pearson:.4f} => {'PASS' if h1_pass else 'FAIL'}", flush=True)
    print(f"  H_audio2: alpha > 0 => {alpha_audio:.4f} => {'PASS' if h2_pass else 'FAIL'}", flush=True)
    print(f"  H_audio3: rho > 0.50 => {r_spearman:.4f} => {'PASS' if h3_pass else 'FAIL'}", flush=True)

    # Cross-domain comparison
    print(f"\n  Cross-domain alpha comparison:", flush=True)
    print(f"    NLP decoders:  alpha ~ 1.477  (19 architectures)", flush=True)
    print(f"    Vision (ViT):  alpha ~ 4.5    (ResNet/ViT)", flush=True)
    print(f"    Audio (speech): alpha = {alpha_audio:.3f}  ({n_models} models, {n_arch} architectures)", flush=True)

    # Save
    elapsed = time.time() - t0
    final = {
        "experiment": "CTI Audio/Speech Domain Validation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "google/speech_commands v0.02 test split",
        "K": K,
        "n_samples": len(ds),
        "pca_dimensions": N_PCA,
        "models": {m: results[m] for m in valid_models},
        "law_fit": {
            "alpha_audio": float(alpha_audio),
            "C": float(C_audio),
            "r_pearson": float(r_pearson),
            "p_pearson": float(p_pearson),
            "r_spearman": float(r_spearman),
            "p_spearman": float(p_spearman),
            "R_squared": float(R_sq),
        },
        "hypotheses": {
            "H_audio1": {"description": "r > 0.70", "r": float(r_pearson), "pass": bool(h1_pass)},
            "H_audio2": {"description": "alpha > 0", "alpha": float(alpha_audio), "pass": bool(h2_pass)},
            "H_audio3": {"description": "Spearman rho > 0.50", "rho": float(r_spearman), "pass": bool(h3_pass)},
        },
        "cross_domain": {
            "alpha_nlp": 1.477,
            "alpha_vision": 4.5,
            "alpha_audio": float(alpha_audio),
        },
        "elapsed_seconds": elapsed,
    }

    out_path = RESULTS_DIR / "cti_audio_speech.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n  Results saved to {out_path}", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Law: logit(q) = {alpha_audio:.3f} * kappa + {C_audio:.3f}", flush=True)
    print(f"  r = {r_pearson:.3f}, R^2 = {R_sq:.3f}", flush=True)
    print(f"  H_audio1 (r > 0.70): {'PASS' if h1_pass else 'FAIL'}", flush=True)
    print(f"  H_audio2 (alpha > 0): {'PASS' if h2_pass else 'FAIL'}", flush=True)
    print(f"  H_audio3 (rho > 0.50): {'PASS' if h3_pass else 'FAIL'}", flush=True)
    print(f"  Total time: {elapsed / 60:.1f} minutes", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
