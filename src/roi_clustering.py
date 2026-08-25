"""Memory-conscious global and ROI hyperspectral clustering.

The normal workflow fits one model to the full image, then computes separate
spectral statistics for every user-defined ROI from that shared label map.
An individual ROI can still be re-clustered when the global result is poor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .roi_utils import box_region


@dataclass
class ROIClusterResult:
    """Clustering output for one ROI.

    ``label_map`` covers only ``bounds`` and uses -1 outside a lasso ROI.
    Cluster IDs are zero based inside this object and one based in exported
    tables/images.
    """

    name: str
    region: dict
    bounds: list[int]
    label_map: np.ndarray
    wavelengths: Optional[list[float]]
    counts: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    std: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    method: str
    pca_components: int
    explained_variance: float
    fit_pixels: int
    value_units: str = "raw DN"
    cluster_ids: Optional[np.ndarray] = None
    source_scope: str = "roi_recluster"

    @property
    def n_pixels(self) -> int:
        return int(self.counts.sum())

    @property
    def n_clusters(self) -> int:
        return int(len(self.counts))

    @property
    def display_cluster_ids(self) -> np.ndarray:
        """One-based IDs shown/exported for the clusters in this result."""
        if self.cluster_ids is None:
            return np.arange(1, self.n_clusters + 1, dtype=np.int16)
        return np.asarray(self.cluster_ids, dtype=np.int16)


def region_local_mask(region: dict, height: int, width: int) -> tuple[np.ndarray, list[int]]:
    """Return a boolean mask local to the clamped ROI bounding box."""
    r0, r1, c0, c1 = box_region(
        region.get("roi", [0, height, 0, width]), height, width
    )["roi"]
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"Empty ROI bounds: {[r0, r1, c0, c1]}")

    local = np.ones((r1 - r0, c1 - c0), dtype=bool)
    if region.get("type", "box") in {"lasso", "polygon"}:
        from matplotlib.path import Path as MplPath

        xs = np.asarray(region.get("x", []), dtype=float)
        ys = np.asarray(region.get("y", []), dtype=float)
        if len(xs) < 3 or len(xs) != len(ys):
            raise ValueError("Lasso/polygon ROI needs at least three x/y coordinates")
        yy, xx = np.mgrid[r0:r1, c0:c1]
        points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
        local = MplPath(np.column_stack((xs, ys))).contains_points(points)
        local = local.reshape((r1 - r0, c1 - c0))

    if not local.any():
        raise ValueError("ROI contains no pixel centers")
    return local, [r0, r1, c0, c1]


def _chunks(values: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def cluster_roi(
    data: np.ndarray,
    region: dict,
    *,
    name: str = "ROI 1",
    wavelengths: Optional[list[float]] = None,
    method: str = "kmeans",
    n_clusters: int = 4,
    pca_components: int = 10,
    fit_sample_limit: int = 100_000,
    median_sample_limit: int = 30_000,
    batch_size: int = 50_000,
    random_state: int = 42,
    calibration: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> ROIClusterResult:
    """Cluster only pixels inside ``region`` and return per-cluster spectra.

    The model is fitted on at most ``fit_sample_limit`` spatially even pixels.
    Every selected pixel is then predicted in batches and contributes to exact
    means/standard deviations.  Median and quartiles use a deterministic,
    evenly spaced sample capped by ``median_sample_limit`` per cluster.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected H x W x bands data, got {data.shape}")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    method = method.lower()
    if method not in {"kmeans", "gmm"}:
        raise ValueError("method must be 'kmeans' or 'gmm'")

    height, width, bands = data.shape
    if calibration is not None:
        cal_a = np.asarray(calibration[0], dtype=np.float64)
        cal_b = np.asarray(calibration[1], dtype=np.float64)
        if cal_a.shape != (bands,) or cal_b.shape != (bands,):
            raise ValueError(
                f"Calibration bands ({len(cal_a)}, {len(cal_b)}) do not match data ({bands})"
            )
    else:
        cal_a = cal_b = None

    def calibrated(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if cal_a is None:
            return values
        return (values * cal_a + cal_b).astype(np.float32)

    local_mask, bounds = region_local_mask(region, height, width)
    r0, r1, c0, c1 = bounds
    crop_flat = data[r0:r1, c0:c1, :].reshape(-1, bands)
    selected = np.flatnonzero(local_mask.ravel())
    if len(selected) < n_clusters:
        raise ValueError(
            f"ROI has only {len(selected):,} pixels for {n_clusters} clusters"
        )

    # Even spatial sampling is deterministic and avoids a large random-index
    # allocation.  Oversample before finite-value filtering.
    target = min(len(selected), max(n_clusters * 20, int(fit_sample_limit)))
    sample_at = np.linspace(0, len(selected) - 1, target, dtype=np.int64)
    sample = calibrated(crop_flat[selected[sample_at]])
    sample = sample[np.all(np.isfinite(sample), axis=1)]
    if len(sample) < n_clusters:
        raise ValueError("Too few finite spectra remain inside the ROI")

    from sklearn.decomposition import PCA

    n_pca = max(1, min(int(pca_components), bands, len(sample) - 1))
    pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=random_state)
    sample_reduced = pca.fit_transform(sample)

    if method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(8192, max(2048, len(sample))),
            n_init=5,
            random_state=random_state,
        )
    else:
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            max_iter=150,
            random_state=random_state,
        )
    model.fit(sample_reduced)

    label_flat = np.full(crop_flat.shape[0], -1, dtype=np.int16)
    sums = np.zeros((n_clusters, bands), dtype=np.float64)
    sumsq = np.zeros((n_clusters, bands), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)

    for positions in _chunks(selected, max(1, int(batch_size))):
        px = calibrated(crop_flat[positions])
        finite = np.all(np.isfinite(px), axis=1)
        if not finite.any():
            continue
        valid_px = px[finite]
        labels = model.predict(pca.transform(valid_px)).astype(np.int16)
        label_flat[positions[finite]] = labels
        for cluster_id in range(n_clusters):
            chosen = valid_px[labels == cluster_id]
            if not len(chosen):
                continue
            counts[cluster_id] += len(chosen)
            sums[cluster_id] += chosen.sum(axis=0, dtype=np.float64)
            sumsq[cluster_id] += np.square(chosen, dtype=np.float64).sum(axis=0)

    if np.any(counts == 0):
        empty = (np.flatnonzero(counts == 0) + 1).tolist()
        raise ValueError(f"Empty cluster(s) after prediction: {empty}")

    mean = sums / counts[:, None]
    variance = np.maximum(sumsq / counts[:, None] - mean**2, 0.0)
    std = np.sqrt(variance)

    # Stable representative quantiles.  Sampling happens after all labels are
    # known so each cluster receives its own cap.
    median = np.empty_like(mean)
    q25 = np.empty_like(mean)
    q75 = np.empty_like(mean)
    for cluster_id in range(n_clusters):
        pos = np.flatnonzero(label_flat == cluster_id)
        take = min(len(pos), max(1, int(median_sample_limit)))
        at = np.linspace(0, len(pos) - 1, take, dtype=np.int64)
        values = calibrated(crop_flat[pos[at]])
        q25[cluster_id], median[cluster_id], q75[cluster_id] = np.percentile(
            values, [25, 50, 75], axis=0
        )

    # Cluster numbers otherwise change arbitrarily between runs.  Ordering by
    # average signal produces a deterministic, easily explained convention.
    order = np.argsort(np.nanmean(mean, axis=1))[::-1]
    remap = np.empty(n_clusters, dtype=np.int16)
    remap[order] = np.arange(n_clusters, dtype=np.int16)
    valid = label_flat >= 0
    label_flat[valid] = remap[label_flat[valid]]

    return ROIClusterResult(
        name=name,
        region=dict(region),
        bounds=bounds,
        label_map=label_flat.reshape(local_mask.shape),
        wavelengths=(list(wavelengths) if wavelengths is not None else None),
        counts=counts[order],
        mean=mean[order],
        median=median[order],
        std=std[order],
        q25=q25[order],
        q75=q75[order],
        method=method,
        pca_components=n_pca,
        explained_variance=float(np.sum(pca.explained_variance_ratio_)),
        fit_pixels=int(len(sample)),
        value_units="reflectance" if calibration is not None else "raw DN",
        cluster_ids=np.arange(1, n_clusters + 1, dtype=np.int16),
        source_scope="roi_recluster",
    )


def summarize_region_from_global(
    data: np.ndarray,
    region: dict,
    global_result: ROIClusterResult,
    *,
    name: str = "ROI 1",
    wavelengths: Optional[list[float]] = None,
    median_sample_limit: int = 30_000,
    calibration: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> ROIClusterResult:
    """Extract per-cluster spectra in ``region`` from a shared global label map.

    No clustering model is fitted here. Cluster IDs therefore remain directly
    comparable across all ROIs. Clusters absent from a region are omitted.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected H x W x bands data, got {data.shape}")
    height, width, bands = data.shape
    if global_result.label_map.shape != (height, width):
        raise ValueError(
            "Global label map shape does not match the loaded hyperspectral image"
        )

    if calibration is not None:
        cal_a = np.asarray(calibration[0], dtype=np.float64)
        cal_b = np.asarray(calibration[1], dtype=np.float64)
        if cal_a.shape != (bands,) or cal_b.shape != (bands,):
            raise ValueError("Calibration bands do not match the image bands")
    else:
        cal_a = cal_b = None

    def calibrated(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if cal_a is None:
            return values
        return (values * cal_a + cal_b).astype(np.float32)

    local_mask, bounds = region_local_mask(region, height, width)
    r0, r1, c0, c1 = bounds
    crop_flat = data[r0:r1, c0:c1, :].reshape(-1, bands)
    global_flat = global_result.label_map[r0:r1, c0:c1].reshape(-1)
    selected = np.flatnonzero(local_mask.ravel() & (global_flat >= 0))
    if not len(selected):
        raise ValueError(f"ROI '{name}' contains no globally clustered pixels")

    present = np.unique(global_flat[selected]).astype(np.int16)
    label_flat = np.full(global_flat.shape, -1, dtype=np.int16)
    counts = np.zeros(len(present), dtype=np.int64)
    mean = np.empty((len(present), bands), dtype=np.float64)
    median = np.empty_like(mean)
    std = np.empty_like(mean)
    q25 = np.empty_like(mean)
    q75 = np.empty_like(mean)

    for local_id, global_id in enumerate(present):
        positions = selected[global_flat[selected] == global_id]
        label_flat[positions] = local_id
        values = calibrated(crop_flat[positions])
        finite = np.all(np.isfinite(values), axis=1)
        values = values[finite]
        if not len(values):
            raise ValueError(f"ROI '{name}' cluster {int(global_id) + 1} has no finite spectra")
        counts[local_id] = len(values)
        mean[local_id] = np.mean(values, axis=0, dtype=np.float64)
        std[local_id] = np.std(values, axis=0, dtype=np.float64)
        take = min(len(values), max(1, int(median_sample_limit)))
        at = np.linspace(0, len(values) - 1, take, dtype=np.int64)
        q25[local_id], median[local_id], q75[local_id] = np.percentile(
            values[at], [25, 50, 75], axis=0
        )

    return ROIClusterResult(
        name=name,
        region=dict(region),
        bounds=bounds,
        label_map=label_flat.reshape(local_mask.shape),
        wavelengths=(list(wavelengths) if wavelengths is not None else None),
        counts=counts,
        mean=mean,
        median=median,
        std=std,
        q25=q25,
        q75=q75,
        method=global_result.method,
        pca_components=global_result.pca_components,
        explained_variance=global_result.explained_variance,
        fit_pixels=global_result.fit_pixels,
        value_units=global_result.value_units,
        cluster_ids=present + 1,
        source_scope="global",
    )


def summarize_region_from_class_map(
    data: np.ndarray,
    region: dict,
    class_map: np.ndarray,
    *,
    name: str = "ROI 1",
    wavelengths: Optional[list[float]] = None,
    method: str = "unknown",
    source_scope: str = "global",
    median_sample_limit: int = 30_000,
    value_units: str = "processed",
) -> ROIClusterResult:
    """Summarize an existing classifier map inside one ROI.

    Unlike ``summarize_region_from_global``, this accepts the class IDs emitted
    by the application's shared ``HyperspectralClassifier`` unchanged. This is
    what allows Hybrid/SAM/supervised/deep methods to share the exact same
    classification implementation as the main analysis screen.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected H x W x bands data, got {data.shape}")
    height, width, bands = data.shape
    if class_map.shape != (height, width):
        raise ValueError("Class map shape does not match hyperspectral data")

    local_mask, bounds = region_local_mask(region, height, width)
    r0, r1, c0, c1 = bounds
    crop_flat = np.asarray(data[r0:r1, c0:c1, :], dtype=np.float32).reshape(-1, bands)
    classes_flat = np.asarray(class_map[r0:r1, c0:c1]).reshape(-1)
    selected = np.flatnonzero(local_mask.ravel() & (classes_flat >= 0))
    if not len(selected):
        raise ValueError(f"ROI '{name}' contains no classified pixels")

    present = np.unique(classes_flat[selected]).astype(np.int32)
    label_flat = np.full(classes_flat.shape, -1, dtype=np.int16)
    counts = np.zeros(len(present), dtype=np.int64)
    mean = np.empty((len(present), bands), dtype=np.float64)
    median = np.empty_like(mean)
    std = np.empty_like(mean)
    q25 = np.empty_like(mean)
    q75 = np.empty_like(mean)

    for local_id, class_id in enumerate(present):
        positions = selected[classes_flat[selected] == class_id]
        values = crop_flat[positions]
        finite = np.all(np.isfinite(values), axis=1)
        finite_positions = positions[finite]
        values = values[finite]
        if not len(values):
            continue
        label_flat[finite_positions] = local_id
        counts[local_id] = len(values)
        mean[local_id] = np.mean(values, axis=0, dtype=np.float64)
        std[local_id] = np.std(values, axis=0, dtype=np.float64)
        take = min(len(values), max(1, int(median_sample_limit)))
        at = np.linspace(0, len(values) - 1, take, dtype=np.int64)
        q25[local_id], median[local_id], q75[local_id] = np.percentile(
            values[at], [25, 50, 75], axis=0
        )

    nonempty = counts > 0
    if not nonempty.any():
        raise ValueError(f"ROI '{name}' contains no finite spectra")

    return ROIClusterResult(
        name=name,
        region=dict(region),
        bounds=bounds,
        label_map=label_flat.reshape(local_mask.shape),
        wavelengths=(list(wavelengths) if wavelengths is not None else None),
        counts=counts[nonempty],
        mean=mean[nonempty],
        median=median[nonempty],
        std=std[nonempty],
        q25=q25[nonempty],
        q75=q75[nonempty],
        method=method,
        pca_components=0,
        explained_variance=float("nan"),
        fit_pixels=int(np.sum(nonempty)),
        value_units=value_units,
        cluster_ids=present[nonempty],
        source_scope=source_scope,
    )


def result_spectra_frame(result: ROIClusterResult) -> pd.DataFrame:
    """Return long-form per-cluster spectral statistics for export."""
    bands = result.mean.shape[1]
    x = (
        np.asarray(result.wavelengths, dtype=float)
        if result.wavelengths is not None and len(result.wavelengths) == bands
        else np.arange(bands, dtype=int)
    )
    axis_name = "wavelength_nm" if result.wavelengths is not None and len(result.wavelengths) == bands else "band_index"
    rows = []
    total = max(1, result.n_pixels)
    for cluster_id in range(result.n_clusters):
        exported_id = int(result.display_cluster_ids[cluster_id])
        rows.append(
            pd.DataFrame(
                {
                    "roi_name": result.name,
                    "cluster_id": exported_id,
                    "clustering_scope": result.source_scope,
                    "pixel_count": int(result.counts[cluster_id]),
                    "fraction": float(result.counts[cluster_id] / total),
                    axis_name: x,
                    "mean": result.mean[cluster_id],
                    "median": result.median[cluster_id],
                    "std": result.std[cluster_id],
                    "q25": result.q25[cluster_id],
                    "q75": result.q75[cluster_id],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize_result_labels_on_data(
    data: np.ndarray,
    result: ROIClusterResult,
    *,
    wavelengths: Optional[list[float]] = None,
    value_units: str = "raw DN",
    median_sample_limit: int = 30_000,
) -> ROIClusterResult:
    """Recompute spectra on another cube using an existing ROI label map.

    This is used to export raw DN statistics alongside corrected reflectance
    without rerunning or changing the accepted clustering result.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected H x W x bands data, got {data.shape}")
    r0, r1, c0, c1 = result.bounds
    crop = np.asarray(data[r0:r1, c0:c1, :], dtype=np.float32)
    if crop.shape[:2] != result.label_map.shape:
        raise ValueError("Result label map does not match the raw-data ROI bounds")
    bands = crop.shape[2]
    flat = crop.reshape(-1, bands)
    labels = np.asarray(result.label_map).reshape(-1)
    count = result.n_clusters
    counts = np.zeros(count, dtype=np.int64)
    mean = np.empty((count, bands), dtype=np.float64)
    median = np.empty_like(mean)
    std = np.empty_like(mean)
    q25 = np.empty_like(mean)
    q75 = np.empty_like(mean)
    for local_id in range(count):
        values = flat[labels == local_id]
        values = values[np.all(np.isfinite(values), axis=1)]
        if not len(values):
            raise ValueError(
                f"ROI '{result.name}' cluster {local_id + 1} has no finite raw spectra"
            )
        counts[local_id] = len(values)
        mean[local_id] = np.mean(values, axis=0, dtype=np.float64)
        std[local_id] = np.std(values, axis=0, dtype=np.float64)
        take = min(len(values), max(1, int(median_sample_limit)))
        at = np.linspace(0, len(values) - 1, take, dtype=np.int64)
        q25[local_id], median[local_id], q75[local_id] = np.percentile(
            values[at], [25, 50, 75], axis=0
        )
    return ROIClusterResult(
        name=result.name,
        region=dict(result.region),
        bounds=list(result.bounds),
        label_map=np.asarray(result.label_map).copy(),
        wavelengths=list(wavelengths) if wavelengths is not None else None,
        counts=counts,
        mean=mean,
        median=median,
        std=std,
        q25=q25,
        q75=q75,
        method=result.method,
        pca_components=result.pca_components,
        explained_variance=result.explained_variance,
        fit_pixels=result.fit_pixels,
        value_units=value_units,
        cluster_ids=np.asarray(result.display_cluster_ids).copy(),
        source_scope=result.source_scope,
    )


def save_spectra_csv(result: ROIClusterResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result_spectra_frame(result).to_csv(out, index=False)
    return out
