#!/usr/bin/env python
"""Canonical CTI shared-anchor equicorrelation estimator.

The historical estimator called ``Sigma_W^(1/2)`` whitening. That transform
amplifies high-variance directions. This module preserves it as explicit
``legacy`` mode and adds ``raw`` (identity) and ``corrected``
(Ledoit-Wolf-regularized inverse square root) modes.

Ledoit-Wolf gives ``S_LW=(1-a)S+a*mu*I``. Up to a scalar irrelevant to cosine,
this is ``S+lambda*I`` with ``lambda=a*mu/(1-a)``.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import sklearn
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import TruncatedSVD

Mode = Literal["raw", "legacy", "corrected"]
MODES: tuple[Mode, ...] = ("raw", "legacy", "corrected")
N_COMPONENTS = 256
EIGEN_FLOOR_REL = 1e-6
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OLD_RESULT = ROOT / "results" / "cti_cross_modal_rho.json"
OLD_SOURCE = ROOT / "src" / "cti_cross_modal_rho.py"
VERIFY_OUTPUT = ROOT / "results" / "cti_equicorr_verification.json"


@dataclass(frozen=True)
class Transform:
    mode: str
    input_dim: int
    basis: np.ndarray | None
    scales: np.ndarray | None
    metadata: dict[str, Any]

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected (*,{self.input_dim}), got {x.shape}")
        if self.basis is None:
            return x.copy()
        assert self.scales is not None
        return (x @ self.basis) * self.scales[None, :]


def finite(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite result: {value}")
    return value


def validate(embeddings: np.ndarray, labels: np.ndarray):
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Need 2D embeddings and one 1D label per row")
    if not x.size or not np.all(np.isfinite(x)):
        raise ValueError("Embeddings are empty or non-finite")
    classes = np.unique(y)
    if len(classes) < 3:
        raise ValueError("Shared-anchor rho requires at least three classes")
    return x, y, classes


def centroids(embeddings: np.ndarray, labels: np.ndarray,
              classes: Sequence[Any] | None = None, min_count: int = 2):
    x, y, observed = validate(embeddings, labels)
    ordered = observed if classes is None else np.asarray(classes)
    means, counts = [], []
    for class_id in ordered:
        mask = y == class_id
        count = int(mask.sum())
        if count < min_count:
            raise ValueError(f"Class {class_id!r} has {count} rows, need {min_count}")
        means.append(x[mask].mean(axis=0))
        counts.append(count)
    return np.vstack(means), np.asarray(counts, dtype=np.int64)


def residuals(embeddings: np.ndarray, labels: np.ndarray,
              classes: Sequence[Any] | None = None):
    x, y, observed = validate(embeddings, labels)
    ordered = observed if classes is None else np.asarray(classes)
    means, counts = centroids(x, y, ordered)
    pooled = [x[y == c] - means[i] for i, c in enumerate(ordered)]
    return np.concatenate(pooled), means, counts


def fit_transform(embeddings: np.ndarray, labels: np.ndarray, mode: Mode,
                  n_components: int = N_COMPONENTS) -> Transform:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    x, y, classes = validate(embeddings, labels)
    d = int(x.shape[1])
    if mode == "raw":
        return Transform("raw", d, None, None, {"formula": "W_raw = I", "rank": d})
    z, _, _ = residuals(x, y, classes)
    n = len(z)
    if mode == "legacy":
        rank = min(int(n_components), d, n - 1)
        svd = TruncatedSVD(n_components=rank, random_state=SEED).fit(z)
        eigenvalues = svd.singular_values_.astype(np.float64) ** 2 / n
        return Transform("legacy", d, svd.components_.T.astype(np.float64),
                         np.sqrt(eigenvalues + 1e-12), {
            "formula": "W_legacy = Sigma_W^(1/2)", "rank": rank,
            "n_components_cap": int(n_components), "covariance_divisor": n,
            "eigenvalue_min": finite(eigenvalues.min()),
            "eigenvalue_max": finite(eigenvalues.max()),
        })
    lw = LedoitWolf(assume_centered=True, store_precision=False).fit(z)
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(lw.covariance_))
    if eigenvalues[-1] <= 0:
        raise ValueError("Ledoit-Wolf covariance is not positive definite")
    eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).tiny)
    alpha = float(lw.shrinkage_)
    mu = float(np.trace((z.T @ z) / n) / d)
    equivalent_lambda = alpha * mu / (1.0 - alpha) if alpha < 1.0 else None
    return Transform("corrected", d, eigenvectors, 1.0 / np.sqrt(eigenvalues), {
        "formula": "W_corrected = Sigma_LW^(-1/2)",
        "ledoit_wolf_shrinkage": alpha, "ledoit_wolf_mu": mu,
        "equivalent_ridge_lambda": equivalent_lambda, "rank": d,
        "eigenvalue_min": finite(eigenvalues.min()),
        "eigenvalue_max": finite(eigenvalues.max()),
    })


def fit_pseudoinverse(embeddings: np.ndarray, labels: np.ndarray,
                      floor_rel: float = EIGEN_FLOOR_REL) -> Transform:
    if not 0.0 < floor_rel < 1.0:
        raise ValueError("floor_rel must be in (0,1)")
    x, y, classes = validate(embeddings, labels)
    z, _, _ = residuals(x, y, classes)
    covariance = z.T @ z / len(z)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    lambda_max = float(eigenvalues[-1])
    if lambda_max <= 0:
        raise ValueError("Empirical covariance has no positive eigenvalue")
    floor = floor_rel * lambda_max
    keep = eigenvalues >= floor
    return Transform("corrected_pseudoinverse_sensitivity", x.shape[1],
                     eigenvectors[:, keep], 1.0 / np.sqrt(eigenvalues[keep]), {
        "formula": "W_pinv = Sigma_W^(-1/2) for lambda_i >= floor",
        "eigenvalue_floor_relative": float(floor_rel),
        "eigenvalue_floor_absolute": float(floor), "lambda_max": lambda_max,
        "rank_retained": int(keep.sum()), "rank_dropped": int((~keep).sum()),
    })


def equicorr_from_centroids(means: np.ndarray, transform: Transform,
                            per_class: bool = True) -> dict[str, Any]:
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != transform.input_dim or len(means) < 3:
        raise ValueError("Invalid centroid matrix")
    k = len(means)
    anchor_values = []
    for anchor in range(k):
        delta = means[np.arange(k) != anchor] - means[anchor]
        transformed = transform.apply(delta)
        norms = np.linalg.norm(transformed, axis=1)
        if np.any(norms <= 1e-15):
            raise ValueError("Coincident centroids make cosine undefined")
        unit = transformed / norms[:, None]
        n = k - 1
        # Ordered off-diagonal mean, algebraically identical to the old matrix mask.
        value = (np.dot(unit.sum(0), unit.sum(0)) - n) / (n * (n - 1))
        anchor_values.append(float(value))
    output = {
        "rho_mean": finite(np.mean(anchor_values)),
        "rho_std_across_anchors": finite(np.std(anchor_values)),
        "K": int(k), "mode": transform.mode, "transform": transform.metadata,
    }
    if per_class:
        output["rho_per_class"] = [finite(v) for v in anchor_values]
    return output


def estimate_equicorr(embeddings: np.ndarray, labels: np.ndarray, mode: Mode):
    x, y, classes = validate(embeddings, labels)
    transform = fit_transform(x, y, mode)
    means, counts = centroids(x, y, classes)
    output = equicorr_from_centroids(means, transform)
    output.update({"class_counts": counts.tolist(), "estimation": "same_sample"})
    return output


def stratified_halves(labels: np.ndarray, seed: int = SEED):
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    first, second = [], []
    for class_id in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == class_id))
        if len(indices) < 4:
            raise ValueError(f"Class {class_id!r} needs at least four rows")
        cut = len(indices) // 2
        first.append(indices[:cut])
        second.append(indices[cut:])
    return np.sort(np.concatenate(first)), np.sort(np.concatenate(second))


def _cross_direction(x, y, fit_idx, eval_idx, mode: Mode,
                     pseudoinverse: bool = False, floor_rel: float = EIGEN_FLOOR_REL):
    transform = (fit_pseudoinverse(x[fit_idx], y[fit_idx], floor_rel)
                 if pseudoinverse else fit_transform(x[fit_idx], y[fit_idx], mode))
    means, counts = centroids(x[eval_idx], y[eval_idx], np.unique(y))
    output = equicorr_from_centroids(means, transform, per_class=False)
    output.update({"fit_n": int(len(fit_idx)), "eval_n": int(len(eval_idx)),
                   "eval_class_counts": counts.tolist()})
    return output


def crossfit_equicorr(embeddings: np.ndarray, labels: np.ndarray, mode: Mode,
                      seed: int = SEED):
    """Fit covariance on one stratified half and evaluate centroids on the other.

    Both directions are evaluated and averaged. Training-half centroids are used
    to residualize the covariance; held-out centroids define the evaluated
    shared-anchor geometry.
    """
    x, y, classes = validate(embeddings, labels)
    first, second = stratified_halves(y, seed)
    forward = _cross_direction(x, y, first, second, mode)
    reverse = _cross_direction(x, y, second, first, mode)
    values = [forward["rho_mean"], reverse["rho_mean"]]
    return {"mode": mode, "estimation": "two_way_stratified_crossfit",
            "seed": int(seed), "K": int(len(classes)),
            "split_disjoint": len(np.intersect1d(first, second)) == 0,
            "split_complete": np.array_equal(np.sort(np.r_[first, second]),
                                              np.arange(len(x))),
            "directions": {"first_to_second": forward, "second_to_first": reverse},
            "rho_mean": finite(np.mean(values)),
            "rho_half_range": finite(abs(values[0] - values[1]) / 2)}

def pseudoinverse_sensitivity(embeddings: np.ndarray, labels: np.ndarray,
                              seed: int = SEED, floor_rel: float = EIGEN_FLOOR_REL):
    x, y, _ = validate(embeddings, labels)
    first, second = stratified_halves(y, seed)
    forward = _cross_direction(x, y, first, second, "corrected", True, floor_rel)
    reverse = _cross_direction(x, y, second, first, "corrected", True, floor_rel)
    pinv_rho = float(np.mean([forward["rho_mean"], reverse["rho_mean"]]))
    corrected = crossfit_equicorr(x, y, "corrected", seed)
    return {"eigenvalue_floor_relative": float(floor_rel),
            "pseudoinverse_rho_mean": finite(pinv_rho),
            "ledoit_wolf_corrected_rho_mean": corrected["rho_mean"],
            "absolute_delta": finite(abs(pinv_rho - corrected["rho_mean"])),
            "directions": {"first_to_second": forward, "second_to_first": reverse}}


def covariance_diagnostic(z: np.ndarray, transform: Transform):
    transformed = transform.apply(np.asarray(z, dtype=np.float64))
    covariance = transformed.T @ transformed / len(transformed)
    identity = np.eye(covariance.shape[0])
    delta = covariance - identity
    off = delta.copy()
    np.fill_diagonal(off, 0.0)
    return {"output_dim": int(len(covariance)),
            "relative_frobenius_error": finite(np.linalg.norm(delta) / np.linalg.norm(identity)),
            "max_abs_diagonal_error": finite(np.max(np.abs(np.diag(covariance) - 1))),
            "max_abs_off_diagonal": finite(np.max(np.abs(off))),
            "condition_number": finite(np.linalg.cond(covariance))}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(src_dir: Path):
    patterns = {"rho": re.compile(r"\brho\b", re.I),
                "equicorr": re.compile("equicorr", re.I),
                "whitened": re.compile("whitened", re.I),
                "sqrt_Lambda": re.compile("sqrt_Lambda", re.I)}
    legacy = {"cti_allen_equicorr_multiarea.py", "cti_allen_equicorrelation.py",
              "cti_alpha_rho_derivation.py", "cti_alpha_rho_multidataset.py",
              "cti_centroid_dispersion.py", "cti_cross_modal_rho.py",
              "cti_equicorr_K_sweep.py", "cti_equicorrelation_deff.py"}
    different = {"cti_generation_local_rho.py", "cti_generation_proxy_b.py",
                 "cti_nsd_human_fmri_v2.py", "cti_synthetic_gumbel_validation.py",
                 "cti_whitening_intervention.py"}
    implementations = []
    broad_match_count = 0
    consumer_count = 0
    for path in sorted(src_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        counts = {key: len(pattern.findall(text)) for key, pattern in patterns.items()}
        if not any(counts.values()):
            continue
        broad_match_count += 1
        category = ("canonical_shared_anchor_implementation"
                    if path.name == "cti_equicorr_canonical.py" else
                    "legacy_shared_anchor_implementation" if path.name in legacy else
                    "different_statistic_or_transform_implementation" if path.name in different else
                    "consumer_theory_or_unrelated_rho")
        if category == "consumer_theory_or_unrelated_rho":
            consumer_count += 1
            continue
        implementations.append({"path": path.relative_to(ROOT).as_posix(),
                                "category": category, "match_counts": counts})
    return {"grep_terms": list(patterns),
            "broad_grep_file_count": broad_match_count,
            "consumer_theory_or_unrelated_rho_file_count": consumer_count,
            "implementation_files": implementations,
            "legacy_shared_anchor_files": sorted(f["path"] for f in implementations
                if f["category"] == "legacy_shared_anchor_implementation"),
            "scope_note": "Token cosines, raw centroid cosines, synthetic generators, and generic rho consumers are not estimator-equivalent."}


def anisotropic_fixture(seed: int, k: int, n_per: int, d: int,
                        eig_min: float, eig_max: float):
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
    noise_map = rotation @ np.diag(np.sqrt(np.geomspace(eig_min, eig_max, d)))
    means = rng.normal(scale=0.7, size=(k, d))
    rows, labels = [], []
    for class_id in range(k):
        rows.append(means[class_id] + rng.normal(size=(n_per, d)) @ noise_map.T)
        labels.extend([class_id] * n_per)
    return np.vstack(rows), np.asarray(labels)


def historical_compute_rho():
    spec = importlib.util.spec_from_file_location("_cti_cross_modal_old", OLD_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {OLD_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module.compute_rho


def run_verification():
    old_hash_before = sha256(OLD_RESULT)
    old_data = json.loads(OLD_RESULT.read_text(encoding="utf-8"))
    parity_x, parity_y = anisotropic_fixture(20260725, 7, 48, 24, 0.05, 12.0)
    old_rho, old_std = historical_compute_rho()(parity_x, parity_y, np.unique(parity_y))
    canonical = estimate_equicorr(parity_x, parity_y, "legacy")
    rho_error = abs(canonical["rho_mean"] - old_rho)
    std_error = abs(canonical["rho_std_across_anchors"] - old_std)
    tolerance = 1e-10
    legacy_pass = rho_error <= tolerance and std_error <= tolerance

    white_x, white_y = anisotropic_fixture(20260726, 6, 2000, 16, 0.1, 10.0)
    z, _, _ = residuals(white_x, white_y)
    corrected_transform = fit_transform(white_x, white_y, "corrected")
    diagnostic = covariance_diagnostic(z, corrected_transform)
    thresholds = {"relative_frobenius_error_max": 0.08,
                  "max_abs_diagonal_error_max": 0.08,
                  "max_abs_off_diagonal_max": 0.05}
    whitening_pass = (diagnostic["relative_frobenius_error"] <= 0.08 and
                      diagnostic["max_abs_diagonal_error"] <= 0.08 and
                      diagnostic["max_abs_off_diagonal"] <= 0.05)

    crossfit = {mode: crossfit_equicorr(parity_x, parity_y, mode, 20260727)
                for mode in MODES}
    crossfit_pass = all(result["split_disjoint"] and result["split_complete"] and
                        np.isfinite(result["rho_mean"]) and
                        all(min(direction["eval_class_counts"]) >= 2
                            for direction in result["directions"].values())
                        for result in crossfit.values())
    pinv = pseudoinverse_sensitivity(parity_x, parity_y, 20260727, EIGEN_FLOOR_REL)
    pinv_pass = (np.isfinite(pinv["pseudoinverse_rho_mean"]) and
                 np.isfinite(pinv["absolute_delta"]) and
                 all(d["transform"]["rank_retained"] >= 1
                     for d in pinv["directions"].values()))
    old_hash_after = sha256(OLD_RESULT)
    immutable_pass = old_hash_before == old_hash_after
    checks = {"historical_artifact_immutable": immutable_pass,
              "legacy_live_source_parity": legacy_pass,
              "corrected_anisotropic_whitening": whitening_pass,
              "crossfit_integrity": crossfit_pass,
              "pseudoinverse_sensitivity_finite": pinv_pass}
    verdict = "CONFIRM" if all(checks.values()) else "KILL"
    return {
        "experiment": "CTI canonical equicorrelation Iteration 1 verification",
        "precommit": {
            "confirm": "All five executable checks pass: immutable history, <=1e-10 legacy parity, corrected whitening thresholds, disjoint finite two-way cross-fit, and finite 1e-6-floor pseudoinverse audit.",
            "kill": "Any executable check fails.",
            "void": "Direct replay of historical model rho values is void because the preserved JSON contains summaries but no source embeddings. Live implementation parity is tested instead; no empirical rerun is claimed."},
        "inventory": inventory(ROOT / "src"),
        "historical_artifact": {
            "path": OLD_RESULT.relative_to(ROOT).as_posix(),
            "sha256_before": old_hash_before, "sha256_after": old_hash_after,
            "unchanged": immutable_pass,
            "preserved_rho_values": old_data["key_finding"]["rho_values"],
            "preserved_rho_range": old_data["key_finding"]["rho_range"],
            "direct_empirical_replay": "VOID_NO_SOURCE_EMBEDDINGS"},
        "legacy_parity_self_test": {
            "fixture": {"seed": 20260725, "K": 7, "N": len(parity_x), "d": 24,
                        "anisotropic_covariance_eigenvalue_range": [0.05, 12.0]},
            "historical_live_function_rho": finite(old_rho),
            "canonical_legacy_rho": canonical["rho_mean"],
            "absolute_rho_error": finite(rho_error),
            "historical_live_function_anchor_std": finite(old_std),
            "canonical_legacy_anchor_std": canonical["rho_std_across_anchors"],
            "absolute_anchor_std_error": finite(std_error),
            "tolerance": tolerance, "pass": legacy_pass},
        "corrected_whitening_self_test": {
            "fixture": {"seed": 20260726, "K": 6, "N": len(white_x), "d": 16,
                        "anisotropic_covariance_eigenvalue_range": [0.1, 10.0]},
            "transform": corrected_transform.metadata,
            "post_transform_covariance": diagnostic,
            "thresholds": thresholds, "pass": whitening_pass},
        "crossfit_self_test": {"fixture_seed": 20260725, "split_seed": 20260727,
                               "modes": crossfit, "pass": crossfit_pass},
        "pseudoinverse_sensitivity": {**pinv, "pass": pinv_pass},
        "checks": checks, "verdict": verdict,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__, "model_execution": False}}


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)

HISTORICAL_BAND = (0.455, 0.467)
NULL_OUTPUT = ROOT / "results" / "cti_equicorr_null_distributions.json"
NULL_DIMS = (4, 16, 64, 256)
NULL_K = (4, 10, 14, 20, 28, 36, 118)


def shared_anchor_statistic(class_means: np.ndarray) -> float:
    """Fast shared-anchor statistic for a K by d centroid matrix."""
    means = np.asarray(class_means, dtype=np.float64)
    if means.ndim != 2 or len(means) < 3:
        raise ValueError("Need a K by d matrix with K >= 3")
    gram = means @ means.T
    diagonal = np.diag(gram)
    distance_sq = diagonal[:, None] + diagonal[None, :] - 2.0 * gram
    distance_sq = np.maximum(distance_sq, 0.0)
    np.fill_diagonal(distance_sq, np.inf)
    weights = 1.0 / np.sqrt(distance_sq)
    direction_sums = weights @ means - weights.sum(axis=1)[:, None] * means
    n = len(means) - 1
    per_anchor = ((direction_sums * direction_sums).sum(axis=1) - n) / (n * (n - 1))
    return finite(per_anchor.mean())


def iid_gaussian_centroids(rng: np.random.Generator, k: int, n: int, d: int):
    """IID Gaussian sample centroids matched on K, total N, and dimension d."""
    if n < k:
        raise ValueError("N must be at least K")
    counts = np.full(k, n // k, dtype=np.int64)
    counts[: n % k] += 1
    means = rng.normal(size=(k, d)) / np.sqrt(counts[:, None])
    return means, counts


def spectrum_matched_gaussian_centroids(rng: np.random.Generator,
                                         spectrum: np.ndarray,
                                         counts: np.ndarray):
    """Null class means with covariance diag(spectrum)/class_count."""
    eigenvalues = np.maximum(np.asarray(spectrum, dtype=np.float64), 0.0)
    counts = np.asarray(counts, dtype=np.float64)
    return (rng.normal(size=(len(counts), len(eigenvalues))) *
            np.sqrt(eigenvalues[None, :] / counts[:, None]))


def distribution_summary(values: np.ndarray, band=HISTORICAL_BAND):
    values = np.asarray(values, dtype=np.float64)
    quantile_levels = [0.005, 0.025, 0.25, 0.5, 0.75, 0.975, 0.995]
    quantiles = np.quantile(values, quantile_levels)
    low, high = band
    q005, q995 = quantiles[0], quantiles[-1]
    relation = ("below_central_99pct" if high < q005 else
                "above_central_99pct" if low > q995 else
                "overlaps_central_99pct")
    return {"n_trials": int(len(values)), "mean": finite(values.mean()),
            "std": finite(values.std(ddof=1)),
            "quantiles": {str(level): finite(value)
                          for level, value in zip(quantile_levels, quantiles)},
            "historical_band": [float(low), float(high)],
            "cdf_at_band_low": finite(np.mean(values <= low)),
            "cdf_at_band_high": finite(np.mean(values <= high)),
            "probability_inside_band": finite(np.mean((values >= low) & (values <= high))),
            "band_location": relation,
            "band_overlaps_central_99pct": relation == "overlaps_central_99pct",
            "mean_inside_band": bool(low <= values.mean() <= high)}


def monte_carlo_centroid_null(k: int, n: int, d: int, trials: int, seed: int,
                              spectrum: np.ndarray | None = None,
                              counts: np.ndarray | None = None,
                              scales: np.ndarray | None = None):
    rng = np.random.default_rng(seed)
    values = np.empty(trials, dtype=np.float64)
    for trial in range(trials):
        if spectrum is None:
            means, generated_counts = iid_gaussian_centroids(rng, k, n, d)
        else:
            if counts is None:
                raise ValueError("Spectrum-matched null requires class counts")
            means = spectrum_matched_gaussian_centroids(rng, spectrum, counts)
            generated_counts = np.asarray(counts)
        if scales is not None:
            means = means * np.asarray(scales)[None, :]
        values[trial] = shared_anchor_statistic(means)
    output = distribution_summary(values)
    output.update({"K": int(k), "N": int(np.sum(generated_counts)), "d": int(d),
                   "seed": int(seed)})
    return output


def analytic_shared_anchor_null():
    return {
        "setup": "mu_c, mu_j, mu_k iid N(0, sigma^2 I_d); Delta_j=mu_j-mu_c",
        "derivation": [
            "E[Delta_j dot Delta_k] = E[||mu_c||^2] = d sigma^2",
            "E[||Delta_j||^2] = E[||Delta_k||^2] = 2 d sigma^2",
            "population coordinate correlation = d sigma^2 / (2 d sigma^2) = 1/2",
            "cos(Delta_j,Delta_k) converges almost surely to 1/2 as d tends to infinity"],
        "asymptotic_expected_cosine": 0.5,
        "finite_dimension_correction": "E[cos] is not exactly 1/2 at finite d because expectation does not commute with random norm normalization.",
        "requested_exact_equality_status": "FALSIFIED_AT_FINITE_d; CONFIRMED_AS_ASYMPTOTIC_NULL",
    }


def real_digits_null(permutations: int, spectrum_trials: int, seed: int):
    """Real non-model embedding control using sklearn's handwritten digits."""
    from sklearn.datasets import load_digits

    dataset = load_digits()
    x = np.asarray(dataset.data, dtype=np.float64)
    y = np.asarray(dataset.target)
    classes = np.unique(y)
    counts = np.asarray([(y == c).sum() for c in classes], dtype=np.int64)
    centered = x - x.mean(axis=0)
    empirical_covariance = centered.T @ centered / len(centered)
    empirical_eigenvalues = np.maximum(np.linalg.eigvalsh(empirical_covariance), 0.0)
    lw = LedoitWolf(assume_centered=True, store_precision=False).fit(centered)
    lw_eigenvalues = np.maximum(np.linalg.eigvalsh(lw.covariance_),
                                np.finfo(np.float64).tiny)
    scale_by_mode = {"raw": np.ones(x.shape[1]),
                     "legacy": np.sqrt(empirical_eigenvalues + 1e-12),
                     "corrected": 1.0 / np.sqrt(lw_eigenvalues)}
    spectrum_matched = {}
    for offset, mode in enumerate(MODES):
        spectrum_matched[mode] = monte_carlo_centroid_null(
            len(classes), len(x), x.shape[1], spectrum_trials, seed + offset,
            empirical_eigenvalues, counts, scale_by_mode[mode])
        effective = empirical_eigenvalues * scale_by_mode[mode] ** 2
        spectrum_matched[mode]["effective_dimension_after_transform"] = finite(
            effective.sum() ** 2 / np.sum(effective ** 2))

    rng = np.random.default_rng(seed + 100)
    permuted = {mode: np.empty(permutations) for mode in MODES}
    for index in range(permutations):
        permuted_labels = rng.permutation(y)
        for mode in MODES:
            permuted[mode][index] = estimate_equicorr(x, permuted_labels, mode)["rho_mean"]
    permutation_summaries = {mode: distribution_summary(values)
                             for mode, values in permuted.items()}
    return {
        "dataset": "sklearn.datasets.load_digits handwritten-image pixels",
        "real_embedding_control": True, "model_execution": False,
        "K": int(len(classes)), "N": int(len(x)), "d": int(x.shape[1]),
        "class_counts": counts.tolist(),
        "empirical_covariance_effective_dimension": finite(
            empirical_eigenvalues.sum() ** 2 / np.sum(empirical_eigenvalues ** 2)),
        "ledoit_wolf_shrinkage": float(lw.shrinkage_),
        "gaussian_centroids_matched_to_empirical_spectrum_and_counts": spectrum_matched,
        "random_label_permutations_on_real_embeddings": {
            "n_permutations": int(permutations), "seed": int(seed + 100),
            "modes": permutation_summaries}}


def run_null_engine(trials: int = 512, permutations: int = 128):
    if trials < 200 or permutations < 50:
        raise ValueError("Need at least 200 MC trials and 50 permutations")
    old_hash_before = sha256(OLD_RESULT)
    verify_hash_before = sha256(VERIFY_OUTPUT)
    grid = []
    for d in NULL_DIMS:
        for k in NULL_K:
            seed = 2026072500 + d * 1000 + k
            grid.append(monte_carlo_centroid_null(k, 32 * k, d, trials, seed))
    digits = real_digits_null(permutations, max(trials, 512), 2026072600)

    expected_cells = {(d, k) for d in NULL_DIMS for k in NULL_K}
    actual_cells = {(row["d"], row["K"]) for row in grid}
    high_d = [row for row in grid if row["d"] == 256]
    grid_pass = actual_cells == expected_cells and len(grid) == len(expected_cells)
    asymptotic_pass = all(abs(row["mean"] - 0.5) <= 0.01 for row in high_d)
    digits_pass = all(np.isfinite(summary["mean"])
                      for summary in digits["random_label_permutations_on_real_embeddings"]["modes"].values())
    old_hash_after = sha256(OLD_RESULT)
    verify_hash_after = sha256(VERIFY_OUTPUT)
    immutable_pass = (old_hash_before == old_hash_after and
                      verify_hash_before == verify_hash_after)
    checks = {"all_28_dimension_K_cells_present": grid_pass,
              "d256_means_within_0.01_of_asymptotic_half": asymptotic_pass,
              "real_spectrum_counts_and_random_labels_finite": digits_pass,
              "prior_artifacts_immutable": immutable_pass}
    engine_verdict = "CONFIRM" if all(checks.values()) else "KILL"

    grid_overlap = [{"d": row["d"], "K": row["K"], "mean": row["mean"],
                     "q005": row["quantiles"]["0.005"],
                     "q995": row["quantiles"]["0.995"]}
                    for row in grid if row["band_overlaps_central_99pct"]]
    grid_mean_inside = [{"d": row["d"], "K": row["K"], "mean": row["mean"]}
                        for row in grid if row["mean_inside_band"]]
    digit_overlap = []
    for generator_name, container in [
        ("spectrum_matched", digits["gaussian_centroids_matched_to_empirical_spectrum_and_counts"]),
        ("random_labels", digits["random_label_permutations_on_real_embeddings"]["modes"])]:
        for mode, summary in container.items():
            if summary["band_overlaps_central_99pct"]:
                digit_overlap.append({"generator": generator_name, "mode": mode,
                                      "mean": summary["mean"],
                                      "q005": summary["quantiles"]["0.005"],
                                      "q995": summary["quantiles"]["0.995"]})
    null_compatible = bool(grid_overlap or digit_overlap)
    if engine_verdict == "KILL":
        verdict = "KILL_NULL_ENGINE"
    elif null_compatible:
        verdict = "KILL_UNCALIBRATED_UNIVERSAL_0.46_CLAIM"
    else:
        verdict = "VOID_PENDING_HISTORICAL_EMBEDDINGS"
    return {
        "experiment": "CTI equicorrelation Iteration 2 matched-null distributions",
        "precommit": {
            "confirm_engine": "All 28 grid cells exist; d=256 means are within 0.01 of 0.5; real spectrum/count and random-label controls are finite; prior artifacts are immutable.",
            "kill_engine": "Any executable engine check fails.",
            "kill_claim": "Historical 0.455-0.467 overlaps any central 99% matched-null interval.",
            "void_claim": "No overlap is found without the unavailable historical source embeddings."},
        "methods": {
            "iid_gaussian_centroids_matched_K_N_d": "Balanced class means of N iid isotropic Gaussian rows; N=32K. Scale invariance means balanced N does not shift rho.",
            "empirical_spectrum_and_class_counts": "Gaussian sample means use the 64-dimensional empirical covariance eigenvalues and exact class counts of real handwritten-digit embeddings.",
            "random_labels": "Exact label-count-preserving permutations on the real digit embeddings; transform refit for every permutation.",
            "shared_anchor_monte_carlo": "Each replicate averages ordered off-diagonal competitor-direction cosines across every anchor."},
        "analytic_shared_anchor_null": analytic_shared_anchor_null(),
        "iid_gaussian_shared_anchor_grid": {"dimensions": list(NULL_DIMS),
                                             "class_counts_K": list(NULL_K),
                                             "trials_per_cell": int(trials),
                                             "cells": grid},
        "real_embedding_matched_nulls": digits,
        "historical_band_location": {
            "band": list(HISTORICAL_BAND),
            "grid_cells_overlapping_central_99pct": grid_overlap,
            "grid_cells_with_mean_inside_band": grid_mean_inside,
            "real_embedding_nulls_overlapping_central_99pct": digit_overlap,
            "null_compatible": null_compatible,
            "interpretation": "Overlap makes 0.455-0.467 non-diagnostic without a preregistered matched-null comparison; it does not estimate corrected rho on the unavailable historical embeddings."},
        "immutability": {"historical_result_sha256_before": old_hash_before,
                         "historical_result_sha256_after": old_hash_after,
                         "verification_sha256_before": verify_hash_before,
                         "verification_sha256_after": verify_hash_after},
        "checks": checks, "engine_verdict": engine_verdict, "verdict": verdict,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__, "model_execution": False}}

def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verification")
    verify.add_argument("--output", type=Path, default=VERIFY_OUTPUT)
    null = commands.add_parser("null")
    null.add_argument("--output", type=Path, default=NULL_OUTPUT)
    null.add_argument("--trials", type=int, default=512)
    null.add_argument("--permutations", type=int, default=128)
    return root


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "verification":
        payload = run_verification()
        write_json(args.output, payload)
        print(json.dumps({"output": str(args.output), "verdict": payload["verdict"],
                          "checks": payload["checks"]}, indent=2))
        return 0 if payload["verdict"] == "CONFIRM" else 1
    if args.command == "null":
        payload = run_null_engine(args.trials, args.permutations)
        write_json(args.output, payload)
        print(json.dumps({"output": str(args.output), "verdict": payload["verdict"],
                          "engine_verdict": payload["engine_verdict"],
                          "checks": payload["checks"]}, indent=2))
        return 0 if payload["engine_verdict"] == "CONFIRM" else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())