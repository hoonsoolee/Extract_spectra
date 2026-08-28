"""Streamlit app: cluster a field globally and export spectra by region.

Run locally or from an Open OnDemand/remote Streamlit session:

    python -m streamlit run app_roi_clustering.py

For ENVI data, enter the .hdr path.  Large files are memory-mapped and should
normally be opened with spatial downsampling while ROIs are being designed.
"""

from __future__ import annotations

import html
import importlib
import json
import re
import copy
import sys
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import src.data_loader as _data_loader_module
import src.radiometry as _radiometry_module
import src.roi_clustering as _roi_clustering_module

# A Streamlit process may already have imported an older loader before BIL
# support was installed. Reloading here makes the web page self-healing and
# avoids requiring a PowerShell/server restart.
_data_loader_module = importlib.reload(_data_loader_module)
HyperspectralLoader = _data_loader_module.HyperspectralLoader
_radiometry_module = importlib.reload(_radiometry_module)
_roi_clustering_module = importlib.reload(_roi_clustering_module)
resolve_calibration = _radiometry_module.resolve_calibration
discover_calibration_candidates = _radiometry_module.discover_calibration_candidates
from src.classifier import HyperspectralClassifier, _CLASS_PALETTE
from src.preprocessor import Preprocessor
from src.timing import (
    array_work_units as _array_work_units,
    estimate_seconds as _estimate_seconds,
    format_duration as _format_duration,
    format_estimate as _format_estimate,
)
ROIClusterResult = _roi_clustering_module.ROIClusterResult
region_local_mask = _roi_clustering_module.region_local_mask
result_spectra_frame = _roi_clustering_module.result_spectra_frame
summarize_result_labels_on_data = _roi_clustering_module.summarize_result_labels_on_data
summarize_region_from_class_map = _roi_clustering_module.summarize_region_from_class_map
from src.roi_utils import box_region, display_rgb, polygon_region, selection_to_region
from src.calibration_provenance import add_calibration_provenance
from src.local_open import open_local_path as _open_local_path
from streamlit_image_coordinates import streamlit_image_coordinates
from src.path_picker import (
    choose_directory as _choose_directory,
    choose_file as _choose_file,
    native_dialogs_available as _native_dialogs_available,
)


def _rc_browse_directory(target_key: str, title: str) -> None:
    try:
        selected = _choose_directory(title, st.session_state.get(target_key, ""))
        if selected:
            st.session_state[target_key] = selected
            if target_key == "rc_local_folder":
                _clear_source_scan()
            st.session_state["rc_path_notice"] = f"선택됨: {selected}"
    except Exception as exc:
        st.session_state["rc_path_error"] = str(exc)


def _rc_browse_file(target_key: str, title: str) -> None:
    try:
        selected = _choose_file(
            title,
            st.session_state.get(target_key, ""),
            filetypes=(("보정 파일", "*.npz"), ("모든 파일", "*.*")),
        )
        if selected:
            st.session_state[target_key] = selected
            st.session_state["rc_path_notice"] = f"선택됨: {selected}"
    except Exception as exc:
        st.session_state["rc_path_error"] = str(exc)


st.set_page_config(
    page_title="구역별 초분광 클러스터링",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>[data-testid='stSidebarNav']{display:none}</style>",
    unsafe_allow_html=True,
)
with st.sidebar:
    st.markdown("### 🧭 분석 화면")
    if Path(__file__).name == "app_roi_clustering.py":
        st.caption("🗺️ ROI 구역 분석·재클러스터링 · 단독 실행")
        st.caption("전체 필드 화면은 `streamlit run app.py`로 실행합니다.")
    else:
        st.page_link("app.py", label="🌿 전체 필드 자동 분석")
        st.page_link(
            "pages/2_구역별_클러스터링.py",
            label="🗺️ ROI 구역 분석·재클러스터링",
        )
    st.divider()


# Use the exact palette from the existing classifier/report pipeline so class
# colours mean the same thing in both screens (e.g. sunlit/shadow leaf/soil).
PALETTE = np.asarray(_CLASS_PALETTE, dtype=np.uint8)
ROI_PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["select2d", "lasso2d"],
}

METHODS = {
    "hybrid": "🌿 Hybrid  (NDVI + 밝기 + K-means)",
    "kmeans": "📊 K-Means  (비지도)",
    "sam": "📐 SAM  (스펙트럼 각도 매핑)",
    "supervised": "🎯 Random Forest  (지도학습)",
    "autoencoder": "🤖 Autoencoder  (딥러닝 비지도)",
    "cnn": "🧠 1D-CNN  (딥러닝 지도학습)",
    "hdbscan": "🔵 HDBSCAN  (밀도 기반 클러스터링)",
    "gmm": "📈 GMM  (가우시안 혼합 모델)",
    "nmf": "🧩 NMF  (스펙트럼 언믹싱)",
}


def _shared_config(
    method: str,
    n_classes: int,
    normalize_mode: str,
    ndvi_threshold: float,
    brightness_threshold: float,
    angle_threshold: float,
    ae_epochs: int,
    cnn_epochs: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
) -> dict:
    """Build the same preprocessing/classification config as the main tab."""
    count = n_classes or 6
    base, remainder = max(1, count // 3), count % 3
    return {
        "preprocessing": {
            "normalize": normalize_mode != "none",
            "normalize_mode": normalize_mode,
            "remove_bad_bands": True,
            "bad_band_ranges": [[1340, 1460], [1790, 1960]],
            "smooth_spectra": False,
            "spatial_downsample": 1,
        },
        "classification": {
            "method": method,
            "input_space": "auto",
            "classes": [],
            "kmeans": {"n_clusters": count, "pca_components": 15, "n_init": 10,
                       "max_iter": 300, "random_state": 42},
            "hybrid": {"ndvi_threshold": ndvi_threshold,
                       "brightness_threshold": brightness_threshold,
                       "kmeans_refinement": True,
                       "n_clusters_sunlit": base + (1 if remainder >= 1 else 0),
                       "n_clusters_shadow": base + (1 if remainder >= 2 else 0),
                       "n_clusters_soil": base, "pca_components": 10},
            "sam": {"angle_threshold": angle_threshold,
                    "n_endmembers": count, "endmember_pca": 15},
            "autoencoder": {"latent_dim": 16, "n_clusters": count,
                            "epochs": ae_epochs, "batch_size": 1024,
                            "learning_rate": 0.001, "max_pixels": 100_000},
            "cnn": {"epochs": cnn_epochs, "batch_size": 512,
                    "learning_rate": 0.001, "test_split": 0.2, "patience": 15},
            "hdbscan": {"min_cluster_size": hdbscan_min_cluster_size,
                        "min_samples": hdbscan_min_samples, "pca_components": 15},
            "gmm": {"n_components": count, "covariance_type": "full",
                    "max_iter": 100, "pca_components": 15, "random_state": 42},
            "nmf": {"n_components": count, "max_iter": 500, "random_state": 42},
        },
    }


def _prepare_shared_data(
    data: np.ndarray,
    wavelengths: list[float] | None,
    config: dict,
    calibration_path: str,
    source_path: str = "",
) -> tuple[np.ndarray, list[float] | None, str, dict | None]:
    """Apply the same preprocessing used by the main analysis pipeline."""
    shared_config = copy.deepcopy(config)
    shared_config.setdefault("preprocessing", {})["calibration_file"] = (
        calibration_path.strip() or None
    )
    shared_config["preprocessing"]["auto_discover_calibration"] = True
    shared_config["preprocessing"]["calibration_search_roots"] = ["./output"]
    preprocessor = Preprocessor(shared_config)
    processed, processed_wl = preprocessor.process(
        np.asarray(data, dtype=np.float32),
        wavelengths,
        skip_downsample=True,
        source_path=source_path or None,
    )
    calibration_info = preprocessor.last_calibration_info
    calibrated = calibration_info is not None
    mode = preprocessor.last_effective_normalize_mode
    units = (
        "reflectance" if calibrated and mode == "none"
        else "raw DN" if not calibrated and mode == "none"
        else "normalized (global)" if mode == "global"
        else "normalized (per-band)"
    )
    return processed, processed_wl, units, calibration_info


def _prepare_clustering_data(
    raw_data: np.ndarray,
    raw_wavelengths: list[float] | None,
    config: dict,
    processed: np.ndarray,
    processed_wavelengths: list[float] | None,
    calibration_info: dict | None,
    source_path: str = "",
) -> tuple[np.ndarray, list[float] | None, str]:
    """Use raw spectral structure by default; keep Hybrid thresholds physical."""
    method = str(config.get("classification", {}).get("method", "kmeans")).lower()
    if method == "hybrid" and calibration_info is not None:
        return processed, processed_wavelengths, "reflectance"
    cluster_config = copy.deepcopy(config)
    preprocessing = cluster_config.setdefault("preprocessing", {})
    preprocessing["calibration_file"] = None
    preprocessing["auto_discover_calibration"] = False
    preprocessing["normalize"] = True
    preprocessing["normalize_mode"] = "global"
    preprocessing["spatial_downsample"] = 1
    cluster_preprocessor = Preprocessor(cluster_config)
    cluster_data, cluster_wavelengths = cluster_preprocessor.process(
        np.asarray(raw_data, dtype=np.float32),
        raw_wavelengths,
        skip_downsample=True,
        source_path=source_path or None,
    )
    return cluster_data, cluster_wavelengths, "raw DN (global scale)"


def _state_defaults() -> None:
    defaults = {
        "rc_data": None,
        "rc_wl": None,
        "rc_rgb": None,
        "rc_meta": None,
        "rc_file": "",
        "rc_scan_files": [],
        "rc_source_kind": "로컬 폴더",
        "rc_downsample": 4,
        "rc_draft": None,
        "rc_polygon_points": [],
        "rc_polygon_last_click": None,
        "rc_regions": [],
        "rc_results": [],
        "rc_global_result": None,
        "rc_global_settings": {},
        "rc_roi_baselines": {},
        "rc_roi_baseline_metrics": {},
        "rc_roi_trials": {},
        "rc_roi_accepted": {},
        "rc_last_report": "",
        "rc_timing_history": [],
        "rc_last_timing": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_source_scan() -> None:
    st.session_state["rc_scan_files"] = []
    st.session_state.pop("rc_selected_file", None)


def _compatible_calibration_for_source(
    source_path: str,
    wavelengths: list[float] | None,
) -> tuple[str, str]:
    """Return the first conservatively named, band-compatible calibration."""
    if not source_path or source_path.startswith("github:"):
        return "", "원격 파일은 보정파일을 직접 지정해 주세요."
    candidates = discover_calibration_candidates(
        source_path,
        search_roots=["./output"],
    )
    rejected = []
    for candidate in candidates:
        try:
            resolve_calibration(
                str(candidate),
                target_source=source_path,
                wavelengths=wavelengths,
            )
            return str(candidate), f"자동 연결됨: {candidate.name}"
        except Exception as exc:
            rejected.append(f"{candidate.name}: {exc}")
    if rejected:
        return "", "이름이 맞는 보정 후보가 있었지만 밴드/센서가 맞지 않아 제외했습니다."
    return "", "이 영상 이름과 일치하는 보정파일을 찾지 못했습니다."


def _region_id() -> str:
    return uuid.uuid4().hex[:12]


def _ensure_region_ids() -> None:
    """Backfill stable IDs for sessions created before ROI history existed."""
    for item in st.session_state.get("rc_regions", []):
        item.setdefault("id", _region_id())


def _clear_roi_widget_state() -> None:
    """Reset ROI-local widgets when the source or global baseline changes."""
    prefixes = (
        "rc_roi_cfg_",
        "rc_roi_trial_select_",
        "rc_global_overlay_",
        "rc_current_",
        "rc_baseline_overlay_",
        "rc_trial_overlay_",
    )
    for key in list(st.session_state):
        if key.startswith(prefixes):
            del st.session_state[key]


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", value.strip())
    return cleaned.strip("._") or "roi"


def _add_region_overlay(fig: go.Figure, item: dict, color: str) -> None:
    region = item["region"]
    if region.get("type") in {"lasso", "polygon"}:
        xs, ys = region.get("x", []), region.get("y", [])
        if len(xs) >= 3:
            path = "M " + " L ".join(f"{x},{y}" for x, y in zip(xs, ys)) + " Z"
            fig.add_shape(
                type="path", path=path, line=dict(color=color, width=3),
                fillcolor="rgba(0,0,0,0)",
            )
    else:
        r0, r1, c0, c1 = region["roi"]
        fig.add_shape(
            type="rect", x0=c0, x1=c1, y0=r0, y1=r1,
            line=dict(color=color, width=3), fillcolor="rgba(0,0,0,0)",
        )
    r0, _, c0, _ = region["roi"]
    fig.add_annotation(
        x=c0, y=r0, text=item["name"], showarrow=False,
        bgcolor=color, font=dict(color="white", size=12), xanchor="left",
    )


def _class_boundaries(
    class_map: np.ndarray,
    selected_ids: list[int] | None = None,
) -> np.ndarray:
    """Return boundaries touching the selected cluster labels."""
    labels = np.asarray(class_map)
    boundary = np.zeros(labels.shape, dtype=bool)
    if selected_ids is None:
        selected_mask = labels >= 0
    else:
        selected_mask = np.isin(labels, np.asarray(selected_ids, dtype=np.int64))
    if not np.any(selected_mask):
        return boundary

    vertical = (labels[1:, :] != labels[:-1, :]) & (
        selected_mask[1:, :] | selected_mask[:-1, :]
    )
    horizontal = (labels[:, 1:] != labels[:, :-1]) & (
        selected_mask[:, 1:] | selected_mask[:, :-1]
    )
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def _cluster_overlay(
    rgb: np.ndarray,
    result: ROIClusterResult,
    opacity: float = 0.60,
    show_boundaries: bool = True,
    selected_display_ids: list[int] | None = None,
) -> np.ndarray:
    r0, r1, c0, c1 = result.bounds
    base = rgb[r0:r1, c0:c1].astype(np.float32)
    alpha = float(np.clip(opacity, 0.0, 1.0))
    overlay = base.copy()
    selected = (
        {int(value) for value in result.display_cluster_ids}
        if selected_display_ids is None
        else {int(value) for value in selected_display_ids}
    )
    selected_local_ids: list[int] = []
    for cluster_id in range(result.n_clusters):
        shown_id = int(result.display_cluster_ids[cluster_id])
        if shown_id not in selected:
            continue
        selected_local_ids.append(cluster_id)
        mask = result.label_map == cluster_id
        color = PALETTE[shown_id % len(PALETTE)].astype(np.float32)
        overlay[mask] = np.clip(base[mask] * (1.0 - alpha) + color * alpha, 0, 255)
    overlay[result.label_map < 0] = (base[result.label_map < 0] * 0.18).astype(np.uint8)
    if show_boundaries:
        overlay[_class_boundaries(result.label_map, selected_local_ids)] = np.array(
            [255, 255, 255], dtype=np.float32
        )
    return overlay.astype(np.uint8)


def _solid_class_map(class_map: np.ndarray) -> np.ndarray:
    """Render every class as a solid colour using the shared palette."""
    rendered = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for class_id in np.unique(class_map):
        if int(class_id) < 0:
            continue
        rendered[class_map == class_id] = PALETTE[int(class_id) % len(PALETTE)]
    return rendered


def _global_overlay(
    rgb: np.ndarray,
    class_map: np.ndarray,
    opacity: float = 0.60,
    show_boundaries: bool = True,
    selected_class_ids: list[int] | None = None,
) -> np.ndarray:
    """Blend selected classes with RGB while retaining unselected context."""
    alpha = float(np.clip(opacity, 0.0, 1.0))
    result = rgb.astype(np.float32).copy()
    selected = (
        [int(value) for value in np.unique(class_map) if int(value) >= 0]
        if selected_class_ids is None
        else [int(value) for value in selected_class_ids]
    )
    for class_id in selected:
        mask = class_map == class_id
        colour = PALETTE[class_id % len(PALETTE)].astype(np.float32)
        result[mask] = result[mask] * (1.0 - alpha) + colour * alpha
    if show_boundaries:
        result[_class_boundaries(class_map, selected)] = np.array(
            [255, 255, 255], dtype=np.float32
        )
    return np.clip(result, 0, 255).astype(np.uint8)


def _highlight_class(
    rgb: np.ndarray,
    class_map: np.ndarray,
    class_id: int,
    colour: np.ndarray | None = None,
) -> np.ndarray:
    """Highlight one class on the dark grayscale background used by the old report."""
    gray = np.mean(rgb.astype(np.float32), axis=2, keepdims=True)
    result = np.repeat(gray * 0.30, 3, axis=2)
    mask = class_map == class_id
    chosen = PALETTE[class_id % len(PALETTE)] if colour is None else np.asarray(colour)
    result[mask] = chosen
    return np.clip(result, 0, 255).astype(np.uint8)


def _roi_single_cluster_image(
    rgb: np.ndarray, result: ROIClusterResult, local_cluster_id: int
) -> np.ndarray:
    r0, r1, c0, c1 = result.bounds
    base = rgb[r0:r1, c0:c1]
    shown_id = int(result.display_cluster_ids[local_cluster_id])
    return _highlight_class(
        base,
        result.label_map,
        local_cluster_id,
        PALETTE[shown_id % len(PALETTE)],
    )


def _class_name(class_id: int, class_info: list[dict]) -> str:
    match = next((item for item in class_info if int(item.get("id", -999)) == class_id), None)
    return str(match.get("name")) if match else f"Cluster {class_id}"


def _quality_metrics(data: np.ndarray, result: ROIClusterResult) -> dict:
    """Compute compact, label-order-independent diagnostics for one ROI run."""
    r0, r1, c0, c1 = result.bounds
    crop = np.asarray(data[r0:r1, c0:c1, :], dtype=np.float32)
    labels = result.label_map.reshape(-1)
    pixels = crop.reshape(-1, crop.shape[2])
    valid = (labels >= 0) & np.all(np.isfinite(pixels), axis=1)
    labels, pixels = labels[valid], pixels[valid]
    unique = np.unique(labels)
    metrics = {
        "silhouette": None,
        "davies_bouldin": None,
        "mean_centroid_sam_deg": None,
        "min_centroid_sam_deg": None,
        "changed_pixel_fraction": None,
    }
    if len(unique) < 2 or len(pixels) <= len(unique):
        return metrics

    take = min(len(pixels), 5000)
    at = np.linspace(0, len(pixels) - 1, take, dtype=np.int64)
    sample, sample_labels = pixels[at], labels[at]
    from sklearn.decomposition import PCA
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(sample)
    n_pca = min(12, scaled.shape[1], len(scaled) - 1)
    features = PCA(n_components=n_pca, random_state=42).fit_transform(scaled)
    if len(np.unique(sample_labels)) >= 2:
        metrics["silhouette"] = float(silhouette_score(features, sample_labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(features, sample_labels))

    means = np.asarray(result.mean, dtype=np.float64)
    norms = np.linalg.norm(means, axis=1)
    angles = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            denom = max(norms[i] * norms[j], 1e-12)
            angle = np.degrees(np.arccos(np.clip(np.dot(means[i], means[j]) / denom, -1, 1)))
            angles.append(float(angle))
    if angles:
        metrics["mean_centroid_sam_deg"] = float(np.mean(angles))
        metrics["min_centroid_sam_deg"] = float(np.min(angles))
    return metrics


def _cluster_matches(
    baseline: ROIClusterResult, candidate: ROIClusterResult
) -> list[tuple[int, int, float]]:
    """Match clusters by minimum spectral angle, independent of numeric IDs."""
    a = np.asarray(baseline.mean, dtype=np.float64)
    b = np.asarray(candidate.mean, dtype=np.float64)
    denom = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    cosine = np.clip((a @ b.T) / np.maximum(denom, 1e-12), -1, 1)
    cost = np.degrees(np.arccos(cosine))
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
    except Exception:
        rows, cols = [], []
        available = set(range(len(b)))
        for row in range(len(a)):
            if not available:
                break
            col = min(available, key=lambda item: cost[row, item])
            rows.append(row)
            cols.append(col)
            available.remove(col)
    return [(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols)]


def _comparison_figure(
    baseline: ROIClusterResult, candidate: ROIClusterResult
) -> go.Figure:
    """Overlay automatically matched baseline/candidate spectra."""
    bands = baseline.mean.shape[1]
    has_wl = baseline.wavelengths is not None and len(baseline.wavelengths) == bands
    x = baseline.wavelengths if has_wl else list(range(bands))
    fig = go.Figure()
    for base_id, candidate_id, angle in _cluster_matches(baseline, candidate):
        shown_base = int(baseline.display_cluster_ids[base_id])
        shown_candidate = int(candidate.display_cluster_ids[candidate_id])
        colour = PALETTE[shown_base % len(PALETTE)]
        css = f"rgb({colour[0]},{colour[1]},{colour[2]})"
        fig.add_trace(go.Scatter(
            x=x, y=baseline.mean[base_id], mode="lines",
            name=f"기준 C{shown_base}", line=dict(color=css, dash="dash", width=1.8),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=candidate.mean[candidate_id], mode="lines",
            name=f"시험 C{shown_candidate} ↔ C{shown_base} ({angle:.1f}°)",
            line=dict(color=css, width=2.5),
        ))
    fig.update_layout(
        height=430, margin=dict(l=45, r=15, t=25, b=45),
        xaxis_title="Wavelength (nm)" if has_wl else "Band index",
        yaxis_title=baseline.value_units,
        legend=dict(orientation="h", y=1.2),
    )
    return fig


def _changed_pixel_fraction(
    baseline: ROIClusterResult, candidate: ROIClusterResult
) -> float | None:
    if baseline.label_map.shape != candidate.label_map.shape:
        return None
    remapped = np.full(candidate.label_map.shape, -999, dtype=np.int32)
    for base_id, candidate_id, _ in _cluster_matches(baseline, candidate):
        remapped[candidate.label_map == candidate_id] = base_id
    valid = (baseline.label_map >= 0) & (candidate.label_map >= 0)
    if not valid.any():
        return None
    return float(np.mean(remapped[valid] != baseline.label_map[valid]))


def _metric_frame(base: dict, trial: dict) -> pd.DataFrame:
    labels = {
        "silhouette": "Silhouette (높을수록 좋음)",
        "davies_bouldin": "Davies–Bouldin (낮을수록 좋음)",
        "mean_centroid_sam_deg": "평균 중심 SAM° (높을수록 분리)",
        "min_centroid_sam_deg": "최소 중심 SAM° (높을수록 분리)",
        "changed_pixel_fraction": "변경 픽셀 비율",
    }
    rows = []
    for key, label in labels.items():
        b, t = base.get(key), trial.get(key)
        rows.append({
            "지표": label,
            "전역 기준": "—" if b is None else f"{b:.4f}",
            "ROI 시험": "—" if t is None else f"{t:.4f}",
        })
    return pd.DataFrame(rows)


def _run_roi_trial(
    data: np.ndarray,
    wavelengths: list[float] | None,
    item: dict,
    settings: dict,
    global_settings: dict,
) -> tuple[ROIClusterResult, dict]:
    """Run one isolated ROI trial while keeping global preprocessing fixed."""
    cfg = _shared_config(
        settings["method"], settings["n_classes"],
        global_settings["normalize_mode"],
        settings["ndvi_threshold"], settings["brightness_threshold"],
        settings["angle_threshold"], settings["ae_epochs"],
        global_settings.get("cnn_epochs", 100),
        settings["hdbscan_min_cluster_size"], settings["hdbscan_min_samples"],
    )
    processed, processed_wl, value_units, calibration_info = _prepare_shared_data(
        data, wavelengths, cfg, global_settings.get("calibration_path", ""),
        global_settings.get("source_file", ""),
    )
    cluster_data, cluster_wl, _ = _prepare_clustering_data(
        data,
        wavelengths,
        cfg,
        processed,
        processed_wl,
        calibration_info,
        global_settings.get("source_file", ""),
    )
    local_mask, bounds = region_local_mask(
        item["region"], processed.shape[0], processed.shape[1]
    )
    r0, r1, c0, c1 = bounds
    selected_pixels = cluster_data[r0:r1, c0:c1, :][local_mask]
    compact = selected_pixels.reshape(-1, 1, cluster_data.shape[2])
    local_classifier = HyperspectralClassifier(cfg)
    local_map_compact, _ = local_classifier.classify(compact, cluster_wl, None)
    class_map = np.full(processed.shape[:2], -1, dtype=np.int32)
    local_target = class_map[r0:r1, c0:c1]
    local_target[local_mask] = local_map_compact.reshape(-1)
    result = summarize_region_from_class_map(
        processed, item["region"], class_map,
        name=item["name"], wavelengths=processed_wl,
        method=settings["method"], source_scope="roi_recluster",
        value_units=value_units,
    )
    return result, _quality_metrics(cluster_data, result)


def _render_result(
    result: ROIClusterResult,
    rgb: np.ndarray,
    key_prefix: str,
) -> None:
    """Render one accepted/baseline ROI result consistently."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROI 픽셀", f"{result.n_pixels:,}")
    c2.metric("클러스터", result.n_clusters)
    c3.metric("분류 방법", result.method.upper())
    c4.metric(
        "분석 범위",
        "전역 모델" if result.source_scope == "global" else "ROI 재분석 채택",
    )
    display_ids = [int(value) for value in result.display_cluster_ids]
    preview_col, control_col = st.columns([4.2, 1.35], vertical_alignment="top")
    with control_col:
        selected_ids = st.multiselect(
            "표시할 클러스터",
            options=display_ids,
            default=display_ids,
            format_func=lambda value: f"Cluster {value}",
            key=f"{key_prefix}_overlay_clusters",
            help="여러 개를 선택하거나 하나만 남겨 원하는 클러스터를 확인합니다.",
        )
        overlay_opacity = st.slider(
            "색상 투명도",
            min_value=0.0, max_value=1.0, value=0.60, step=0.05,
            key=f"{key_prefix}_overlay_opacity",
            help="낮추면 원본 RGB가, 높이면 클러스터 색상이 더 선명합니다.",
        )
        show_boundaries = st.checkbox(
            "흰색 경계선", value=True, key=f"{key_prefix}_overlay_boundaries"
        )
        st.caption("선택하지 않은 클러스터는 원본 RGB로 남습니다.")
    with preview_col:
        st.image(
            _cluster_overlay(
                rgb, result, overlay_opacity, show_boundaries, selected_ids
            ),
            caption="RGB + 선택한 ROI 클러스터",
            width="stretch",
        )
    table_col, spec_col = st.columns([1, 2.1], vertical_alignment="top")
    with table_col:
        st.dataframe(
            pd.DataFrame({
                "cluster": result.display_cluster_ids,
                "pixels": result.counts,
                "fraction_%": np.round(100 * result.counts / result.n_pixels, 2),
            }),
            hide_index=True, width="stretch",
        )
    with spec_col:
        st.plotly_chart(_spectra_figure(result), width="stretch", key=f"{key_prefix}_spec")
    with st.expander("클러스터별 단독 이미지", expanded=False):
        cluster_cols = st.columns(min(3, max(1, result.n_clusters)))
        for local_id in range(result.n_clusters):
            shown_id = int(result.display_cluster_ids[local_id])
            count = int(result.counts[local_id])
            pct = 100.0 * count / max(1, result.n_pixels)
            cluster_cols[local_id % len(cluster_cols)].image(
                _roi_single_cluster_image(rgb, result, local_id),
                caption=f"Cluster {shown_id} · {count:,} px ({pct:.1f}%)",
                width="stretch",
            )


def _open_output_folder(folder: Path) -> None:
    """Open a local output directory in the operating-system file browser."""
    resolved = folder.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"결과 폴더가 없습니다: {resolved}")
    _open_local_path(resolved)


def _spectra_figure(result: ROIClusterResult) -> go.Figure:
    bands = result.mean.shape[1]
    has_wl = result.wavelengths is not None and len(result.wavelengths) == bands
    x = result.wavelengths if has_wl else list(range(bands))
    fig = go.Figure()
    for cluster_id in range(result.n_clusters):
        shown_id = int(result.display_cluster_ids[cluster_id])
        color = PALETTE[shown_id % len(PALETTE)]
        css = f"rgb({color[0]},{color[1]},{color[2]})"
        label = (
            f"Cluster {shown_id} "
            f"({100 * result.counts[cluster_id] / result.n_pixels:.1f}%)"
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=result.mean[cluster_id], mode="lines", name=label,
                line=dict(color=css, width=2.2),
                customdata=result.median[cluster_id],
                hovertemplate="x=%{x}<br>mean=%{y:.5g}<br>median=%{customdata:.5g}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        height=390,
        margin=dict(l=45, r=15, t=25, b=45),
        xaxis_title="Wavelength (nm)" if has_wl else "Band index",
        yaxis_title=result.value_units,
        legend=dict(orientation="h", y=1.15),
    )
    return fig


def _save_report(
    source_file: str,
    rgb: np.ndarray,
    regions: list[dict],
    results: list[ROIClusterResult],
    output_root: str,
    config: dict,
    global_result: dict | None = None,
    raw_data: np.ndarray | None = None,
    raw_wavelengths: list[float] | None = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root).expanduser().resolve() / f"{_slug(Path(source_file).stem)}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    all_frames = []
    raw_frames = []
    summary_rows = []
    sections = []
    global_section = ""
    global_settings = config.get("global_analysis", {})
    calibration_info = global_settings.get("selected_calibration")
    effective_normalization = str(
        global_settings.get(
            "effective_normalization",
            global_settings.get("normalize_mode", ""),
        )
    )

    if global_result and isinstance(global_result.get("class_map"), np.ndarray):
        global_map = np.asarray(global_result["class_map"])
        class_info = list(global_result.get("class_info") or [])
        Image.fromarray(_solid_class_map(global_map)).save(run_dir / "global_class_map.png")
        Image.fromarray(_global_overlay(rgb, global_map, 0.60, True)).save(
            run_dir / "global_overlay.png"
        )
        Image.fromarray(_global_overlay(rgb, global_map, 0.12, True)).save(
            run_dir / "global_boundaries.png"
        )
        global_cards = []
        for class_id in sorted(int(v) for v in np.unique(global_map) if int(v) >= 0):
            name = _class_name(class_id, class_info)
            filename = f"global_cluster_{class_id}.png"
            Image.fromarray(_highlight_class(rgb, global_map, class_id)).save(run_dir / filename)
            count = int(np.sum(global_map == class_id))
            pct = 100.0 * count / max(1, global_map.size)
            global_cards.append(
                f'<figure><img src="{filename}"><figcaption>'
                f'{html.escape(name)} · ID {class_id} · {count:,} px ({pct:.1f}%)'
                f'</figcaption></figure>'
            )
        semantic_html = ""
        hybrid_base = global_result.get("hybrid_base_map")
        if isinstance(hybrid_base, np.ndarray):
            semantic_names = {
                0: "배경/깊은 그림자", 1: "밝은 잎", 2: "그림자 잎", 3: "토양"
            }
            semantic_cards = []
            for semantic_id in range(4):
                filename = f"hybrid_semantic_{semantic_id}.png"
                Image.fromarray(_highlight_class(rgb, hybrid_base, semantic_id)).save(
                    run_dir / filename
                )
                count = int(np.sum(hybrid_base == semantic_id))
                pct = 100.0 * count / max(1, hybrid_base.size)
                semantic_cards.append(
                    f'<figure><img src="{filename}"><figcaption>'
                    f'{semantic_names[semantic_id]} · ID {semantic_id} · '
                    f'{count:,} px ({pct:.1f}%)</figcaption></figure>'
                )
            semantic_html = (
                '<h3>Hybrid 1차 의미 분리</h3>'
                f'<div class="class-grid">{"".join(semantic_cards)}</div>'
            )
        global_section = f"""
        <section><h2>전체 이미지 클러스터 검수</h2>
        <div class="grid">
          <figure><img src="global_overlay.png"><figcaption>RGB + 전체 클러스터</figcaption></figure>
          <figure><img src="global_boundaries.png"><figcaption>RGB + 클러스터 경계선</figcaption></figure>
          <figure><img src="global_class_map.png"><figcaption>클러스터 컬러 맵</figcaption></figure>
        </div>{semantic_html}<h3>클러스터별 단독 이미지</h3>
        <div class="class-grid">{''.join(global_cards)}</div></section>
        """

    for result_index, result in enumerate(results, 1):
        roi_slug = f"{result_index:02d}_{_slug(result.name)}"
        roi_dir = run_dir / roi_slug
        roi_dir.mkdir(parents=True, exist_ok=True)

        spectra = add_calibration_provenance(
            result_spectra_frame(result),
            source_file=source_file,
            value_units=result.value_units,
            normalization_mode=effective_normalization,
            calibration_info=calibration_info,
            calibration_applied=(
                result.value_units == "reflectance" and calibration_info is not None
            ),
            coefficients_a=(calibration_info or {}).get("a"),
            coefficients_b=(calibration_info or {}).get("b"),
        )
        spectra.to_csv(roi_dir / "cluster_spectra.csv", index=False)
        processed_suffix = (
            "reflectance" if result.value_units == "reflectance" else "processed"
        )
        spectra.to_csv(
            roi_dir / f"cluster_spectra_{processed_suffix}.csv", index=False
        )
        all_frames.append(spectra)
        if raw_data is not None:
            raw_result = summarize_result_labels_on_data(
                raw_data, result, wavelengths=raw_wavelengths, value_units="raw DN"
            )
            raw_spectra = add_calibration_provenance(
                result_spectra_frame(raw_result),
                source_file=source_file,
                value_units="raw DN",
                normalization_mode="none",
                calibration_info=calibration_info,
                calibration_applied=False,
            )
            raw_spectra.to_csv(roi_dir / "cluster_spectra_raw_dn.csv", index=False)
            raw_frames.append(raw_spectra)
        np.savez_compressed(
            roi_dir / "cluster_map.npz",
            label_map=result.label_map,
            bounds=np.asarray(result.bounds, dtype=np.int64),
        )

        overlay = _cluster_overlay(rgb, result, 0.60, True)
        Image.fromarray(overlay).save(roi_dir / "cluster_map.png")
        Image.fromarray(_cluster_overlay(rgb, result, 0.12, True)).save(
            roi_dir / "cluster_boundaries.png"
        )

        bands = result.mean.shape[1]
        has_wl = result.wavelengths is not None and len(result.wavelengths) == bands
        x = result.wavelengths if has_wl else np.arange(bands)
        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=130)
        for cluster_id in range(result.n_clusters):
            shown_id = int(result.display_cluster_ids[cluster_id])
            color = PALETTE[shown_id % len(PALETTE)] / 255.0
            ax.plot(x, result.mean[cluster_id], color=color, linewidth=1.7,
                    label=f"Cluster {shown_id}")
        ax.set_xlabel("Wavelength (nm)" if has_wl else "Band index")
        ax.set_ylabel(result.value_units)
        ax.grid(alpha=0.2)
        ax.legend(ncol=min(4, result.n_clusters), fontsize=8)
        fig.tight_layout()
        fig.savefig(roi_dir / "cluster_spectra.png", bbox_inches="tight")
        plt.close(fig)

        cluster_cards = []
        for local_id in range(result.n_clusters):
            shown_id = int(result.display_cluster_ids[local_id])
            filename = f"cluster_{shown_id}.png"
            Image.fromarray(_roi_single_cluster_image(rgb, result, local_id)).save(
                roi_dir / filename
            )
            count = int(result.counts[local_id])
            pct = 100.0 * count / max(1, result.n_pixels)
            cluster_cards.append(
                f'<figure><img src="{html.escape(roi_slug)}/{filename}"><figcaption>'
                f'Cluster {shown_id} · {count:,} px ({pct:.1f}%)'
                f'</figcaption></figure>'
            )

        for local_id, count in enumerate(result.counts):
            summary_rows.append(
                {
                    "roi_name": result.name,
                    "cluster_id": int(result.display_cluster_ids[local_id]),
                    "pixel_count": int(count),
                    "fraction": float(count / result.n_pixels),
                    "method": result.method,
                    "clustering_scope": result.source_scope,
                    "value_units": result.value_units,
                }
            )

        sections.append(
            f"""
            <section><h2>{html.escape(result.name)}</h2>
            <p>{result.n_pixels:,} pixels · {result.n_clusters} clusters ·
            method {html.escape(result.method)} ·
            {html.escape(result.value_units)} · {html.escape(result.source_scope)}</p>
            <p><a href="{html.escape(roi_slug)}/cluster_spectra_{processed_suffix}.csv">보정/처리 스펙트럼 CSV</a>
            · <a href="{html.escape(roi_slug)}/cluster_spectra_raw_dn.csv">보정 전 raw DN CSV</a></p>
            <div class="grid">
              <figure><img src="{html.escape(roi_slug)}/cluster_map.png"><figcaption>ROI cluster map</figcaption></figure>
              <figure><img src="{html.escape(roi_slug)}/cluster_boundaries.png"><figcaption>RGB + cluster boundaries</figcaption></figure>
              <figure><img src="{html.escape(roi_slug)}/cluster_spectra.png"><figcaption>Cluster mean spectra</figcaption></figure>
            </div><h3>ROI 클러스터별 단독 이미지</h3>
            <div class="class-grid">{''.join(cluster_cards)}</div></section>
            """
        )

    pd.concat(all_frames, ignore_index=True).to_csv(run_dir / "all_roi_cluster_spectra.csv", index=False)
    if raw_frames:
        pd.concat(raw_frames, ignore_index=True).to_csv(
            run_dir / "all_roi_cluster_spectra_raw_dn.csv", index=False
        )
    pd.DataFrame(summary_rows).to_csv(run_dir / "cluster_summary.csv", index=False)

    manifest = {
        "source_file": str(Path(source_file).resolve()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config,
        "regions": regions,
    }
    (run_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
    <title>구역별 초분광 클러스터링</title><style>
    body{{font-family:Arial,sans-serif;max-width:1800px;margin:28px auto;padding:0 24px;color:#24322b}}
    h1{{color:#176b43}} section{{border-top:1px solid #ccd8d1;margin-top:30px;padding-top:12px}}
    .grid,.class-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}}
    img{{width:100%;max-width:100%;height:auto;border:1px solid #ddd;cursor:zoom-in}}
    figcaption{{color:#5d6d64;font-size:13px}}
    .image-modal{{display:none;position:fixed;z-index:9999;inset:0;padding:24px;background:rgba(0,0,0,.88);
      align-items:center;justify-content:center;flex-direction:column}}
    .image-modal img{{width:auto;max-width:96vw;max-height:88vh;object-fit:contain;background:#fff}}
    .image-modal button{{position:fixed;top:12px;right:22px;border:0;background:transparent;color:#fff;
      font-size:42px;cursor:pointer}}
    .image-modal-caption{{color:#fff;margin-top:10px;font-size:14px}}
    @media(max-width:1050px){{.grid,.class-grid{{grid-template-columns:1fr}}body{{padding:0 12px}}}}
    </style></head><body><h1>구역별 초분광 클러스터링 리포트</h1>
    <p><b>Source:</b> {html.escape(str(source_file))}<br><b>Created:</b> {html.escape(manifest['created_at'])}</p>
    {global_section}{''.join(sections)}
    <div id="image-modal" class="image-modal" role="dialog" aria-modal="true" aria-label="이미지 확대">
      <button type="button" aria-label="닫기">&times;</button><img id="modal-image" alt="">
      <div id="modal-caption" class="image-modal-caption"></div>
    </div>
    <script>(function(){{
      const modal=document.getElementById('image-modal');
      const modalImage=document.getElementById('modal-image');
      const caption=document.getElementById('modal-caption');
      function closeImage(){{modal.style.display='none';document.body.style.overflow='';}}
      document.querySelectorAll('figure img').forEach(function(img){{
        img.title='클릭하여 크게 보기';
        img.addEventListener('click',function(){{modalImage.src=img.src;modalImage.alt=img.alt||'';
          caption.textContent=img.alt||'';modal.style.display='flex';document.body.style.overflow='hidden';}});
      }});
      modal.addEventListener('click',function(event){{
        if(event.target===modal||event.target.tagName==='BUTTON')closeImage();
      }});
      document.addEventListener('keydown',function(event){{if(event.key==='Escape')closeImage();}});
    }})();</script></body></html>"""
    (run_dir / "report.html").write_text(report, encoding="utf-8")
    return run_dir / "report.html"


_state_defaults()
_ensure_region_ids()

with st.sidebar:
    st.markdown("## 🗺️ 구역별 분석 설정")
    if st.session_state.get("rc_path_error"):
        st.error(st.session_state.pop("rc_path_error"))
    if st.session_state.get("rc_path_notice"):
        st.success(st.session_state.pop("rc_path_notice"))
    st.markdown("### 📂 데이터 소스")
    data_src = st.radio(
        "소스",
        ["로컬 폴더", "GitHub 저장소"],
        horizontal=True,
        label_visibility="collapsed",
        key="rc_source_kind",
        on_change=_clear_source_scan,
    )

    local_folder = github_repo = github_folder = github_token = ""
    if data_src == "로컬 폴더":
        _rc_lf1, _rc_lf2 = st.columns([4, 1])
        with _rc_lf1:
            local_folder = st.text_input(
                "폴더 경로", value="./data", placeholder="C:/data/field_images",
                key="rc_local_folder",
            )
        with _rc_lf2:
            st.write("")
            st.button(
                "🪟 선택",
                width="stretch",
                key="rc_browse_local_folder",
                on_click=_rc_browse_directory,
                args=("rc_local_folder", "초분광 데이터 폴더 선택"),
                disabled=not _native_dialogs_available(),
            )
    else:
        github_repo = st.text_input("저장소 (owner/repo)", placeholder="username/repo", key="rc_gh_repo")
        github_folder = st.text_input("서브폴더", value="", placeholder="data/2024", key="rc_gh_folder")
        github_token = st.text_input("GitHub 토큰 (비공개용)", type="password", key="rc_gh_token")

    if st.button("📂 폴더 스캔", width="stretch", key="rc_scan_btn"):
        try:
            scan_loader = HyperspectralLoader({"spatial_downsample": 1})
            if data_src == "로컬 폴더":
                st.session_state["rc_scan_files"] = [
                    str(p) for p in scan_loader.list_local_files(local_folder)
                ]
            else:
                st.session_state["rc_scan_files"] = scan_loader.list_github_files(
                    github_repo, github_folder, github_token or None
                )
            if not st.session_state["rc_scan_files"]:
                st.warning("지원 형식 파일을 찾지 못했습니다.")
        except Exception as exc:
            st.session_state["rc_scan_files"] = []
            st.error(f"폴더 스캔 실패: {exc}")

    selected_file = None
    if st.session_state["rc_scan_files"]:
        selected_file = st.selectbox(
            "처리할 파일",
            st.session_state["rc_scan_files"],
            format_func=lambda p: Path(p).name,
            key="rc_selected_file",
        )
        st.caption(f"📄 {Path(selected_file).name}")
    else:
        st.caption("📂 스캔하여 파일을 선택하세요.")

    st.divider()
    st.markdown("### ⚡ 대용량 파일")
    downsample = st.select_slider(
        "공간 다운샘플링",
        options=[1, 2, 4, 8],
        value=int(st.session_state.get("rc_downsample", 4)),
        help="구역 설계와 1차 분석은 4를 권장합니다. 좌표는 로드된 해상도 기준입니다.",
    )
    if st.button("📂 선택 파일 로드", type="primary", width="stretch", disabled=not selected_file):
        try:
            with st.spinner("선택한 초분광 파일을 읽는 중입니다..."):
                loader = HyperspectralLoader({"spatial_downsample": int(downsample)})
                if data_src == "로컬 폴더":
                    data, meta = loader.load_local(selected_file)
                    source_ref = selected_file
                else:
                    data, meta = loader.load_github(
                        github_repo, selected_file, github_token or None
                    )
                    source_ref = f"github:{github_repo}/{selected_file}"
                wavelengths = meta.get("wavelengths")
                auto_calibration, calibration_status = _compatible_calibration_for_source(
                    source_ref, wavelengths
                )
                _clear_roi_widget_state()
                st.session_state.update(
                    rc_data=data,
                    rc_wl=wavelengths,
                    rc_rgb=display_rgb(data, wavelengths),
                    rc_meta=meta,
                    rc_file=source_ref,
                    rc_downsample=int(downsample),
                    rc_draft=None,
                    rc_polygon_points=[],
                    rc_polygon_last_click=None,
                    rc_regions=[],
                    rc_results=[],
                    rc_global_result=None,
                    rc_global_settings={},
                    rc_roi_baselines={},
                    rc_roi_baseline_metrics={},
                    rc_roi_trials={},
                    rc_roi_accepted={},
                    rc_last_report="",
                    rc_last_timing=None,
                    rc_calibration_path=auto_calibration,
                    rc_calibration_status=calibration_status,
                    rc_normalize_mode=("none" if auto_calibration else "global"),
                )
            st.success(f"로드 완료: {data.shape}")
        except Exception:
            st.error("파일을 불러오지 못했습니다.")
            st.code(traceback.format_exc(), language="python")

    st.divider()
    method = st.selectbox(
        "분류 방법 — 기존 분석과 동일",
        options=list(METHODS),
        format_func=lambda x: METHODS[x],
        index=list(METHODS).index("kmeans"),
    )
    if method == "hybrid":
        st.caption(
            "클러스터링 입력: 보정파일이 있으면 반사율, 없으면 전역 배율 DN."
        )
    else:
        st.caption(
            "클러스터링은 원본 DN 구조로 하고, 같은 마스크에서 보정 반사율 "
            "스펙트럼을 추출합니다."
        )
    if method in {"supervised", "hdbscan"}:
        n_classes = 0
        st.caption("클래스 수는 라벨 또는 알고리즘이 자동으로 결정합니다.")
    else:
        n_classes = st.slider("클러스터(클래스) 수", 2, 20, 6)

    ndvi_threshold, brightness_threshold = 0.15, 0.08
    if method == "hybrid":
        ndvi_threshold = st.slider("NDVI 임계값 (식생 기준)", 0.0, 1.0, 0.15, 0.01)
        brightness_threshold = st.slider("밝기 임계값 (그림자 기준)", 0.0, 0.5, 0.08, 0.01)
    angle_threshold = 0.10
    if method == "sam":
        angle_threshold = st.slider("각도 임계값 (radians, 0=제한없음)", 0.0, 0.5, 0.10, 0.01)
    ae_epochs, cnn_epochs = 60, 100
    if method == "autoencoder":
        ae_epochs = st.slider("학습 epochs", 10, 200, 60, 10)
    if method == "cnn":
        cnn_epochs = st.slider("학습 epochs", 10, 200, 100, 10)
    hdbscan_min_cluster_size, hdbscan_min_samples = 50, 5
    if method == "hdbscan":
        hdbscan_min_cluster_size = st.slider("min_cluster_size", 10, 500, 50, 10)
        hdbscan_min_samples = st.slider("min_samples", 1, 50, 5, 1)

    labels_csv = ""
    if method in {"supervised", "cnn", "sam"}:
        labels_csv = st.text_input(
            "라벨 CSV" if method in {"supervised", "cnn"} else "라벨 CSV (선택 — SAM 지도 모드)",
            placeholder="labels.csv",
        )

    normalize_mode = st.selectbox(
        "정규화 방식",
        options=["global", "per_band", "none"],
        format_func=lambda x: {
            "global": "전역 배율 (스펙트럼 형태 보존)",
            "per_band": "밴드별 스트레치 (대비 강조)",
            "none": "정규화 안 함 (DN/보정 반사도 유지)",
        }[x],
        help=(
            "전역 배율은 모든 밴드를 같은 값으로 나눠 스펙트럼 형태를 보존합니다. "
            "밴드별 스트레치는 영상 대비용이며 스펙트럼 형태가 달라집니다. "
            "보정된 실제 반사도 값을 그대로 저장하려면 보정계수를 지정하고 "
            "'정규화 안 함'을 선택하세요."
        ),
        key="rc_normalize_mode",
    )
    _rc_cf1, _rc_cf2 = st.columns([4, 1])
    with _rc_cf1:
        calibration_path = st.text_input(
            "반사도 보정 (.npz 또는 프로파일 폴더, 선택)",
            placeholder="./calibration_profiles",
            help=(
                "White/Dark 프로파일 폴더를 지정하면 영상 촬영시각과 가장 가까운 White를 "
                "자동 선택합니다. 단일 .npz와 기존 empirical-line 계수도 지원합니다."
            ),
            key="rc_calibration_path",
        )
    with _rc_cf2:
        st.write("")
        st.button(
            "🪟 파일",
            width="stretch",
            key="rc_browse_calibration_file",
            on_click=_rc_browse_file,
            args=("rc_calibration_path", "반사도 보정 .npz 선택"),
            disabled=not _native_dialogs_available(),
            help="프로파일 폴더는 경로를 직접 입력하고, .npz는 이 버튼으로 선택합니다.",
        )
    if calibration_path and normalize_mode != "none":
        st.warning(
            "보정파일이 적용되면 반사율 값을 보존하기 위해 실제 분석에서는 "
            "추가 정규화를 자동으로 끕니다."
        )
    _rc_calibration_status = st.session_state.get("rc_calibration_status", "")
    if calibration_path:
        st.success(
            "✅ 새 분석에 반사율 보정이 적용됩니다. "
            + (_rc_calibration_status or Path(calibration_path).name)
        )
    elif _rc_calibration_status:
        st.caption(f"ℹ️ {_rc_calibration_status}")
    _rc_of1, _rc_of2 = st.columns([4, 1])
    with _rc_of1:
        output_root = st.text_input(
            "결과 폴더", value="./roi_cluster_output",
            help=(
                "상대 경로는 이 프로그램 폴더를 기준으로 합니다. 예: "
                "./roi_cluster_output. 저장 후 화면 아래의 '결과 폴더 열기'로 "
                "바로 열 수 있습니다."
            ),
            key="rc_output_root",
        )
    with _rc_of2:
        st.write("")
        st.button(
            "🪟 선택",
            width="stretch",
            key="rc_browse_output_folder",
            on_click=_rc_browse_directory,
            args=("rc_output_root", "구역별 분석 결과 폴더 선택"),
            disabled=not _native_dialogs_available(),
        )
    with st.expander("ℹ️ 정규화·보정·결과 폴더 사용법", expanded=False):
        st.markdown(
            "- **반사도 자료가 필요할 때:** White/Dark 프로파일 폴더 지정 + "
            "`정규화 안 함`\n"
            "- **클러스터링 안정성이 우선일 때:** 보정 프로파일 지정 + "
            "`전역 배율`\n"
            "- **밴드별 스트레치:** 시각적 대비용이며 과학용 스펙트럼 저장에는 "
            "권장하지 않음\n"
            "- **결과 폴더:** HTML, CSV, 클러스터 맵과 분석 설정이 함께 저장됨"
        )


st.title("전역 클러스터링 · 구역별 스펙트럼")
st.caption(
    "전체 이미지는 한 번만 클러스터링하고, 직접 지정한 각 구역에서는 "
    "동일한 클러스터 기준으로 평균·중간값 스펙트럼을 따로 추출합니다. "
    "결과가 좋지 않은 구역만 선택하여 별도로 재클러스터링할 수도 있습니다."
)

data = st.session_state.get("rc_data")
rgb = st.session_state.get("rc_rgb")
wavelengths = st.session_state.get("rc_wl")

if data is None or rgb is None:
    st.info("왼쪽에서 데이터 폴더를 스캔하고 파일을 선택한 후 **선택 파일 로드**를 누르세요.")
    st.stop()

height, width, bands = data.shape
meta = st.session_state.get("rc_meta") or {}
st.success(
    f"{meta.get('filename', Path(st.session_state['rc_file']).name)} · "
    f"{height:,} × {width:,} px · {bands} bands · downsample ×{st.session_state['rc_downsample']}"
)

tab_regions, tab_run, tab_results, tab_export = st.tabs(
    [
        "1️⃣ 구역 나누기", "2️⃣ 전체 클러스터링",
        "3️⃣ 결과 및 재분석", "4️⃣ 보정 BIL 내보내기",
    ]
)

with tab_regions:
    st.markdown("### 여러 ROI 만들기")
    _rc_mouse1, _rc_mouse2 = st.columns([3, 1])
    with _rc_mouse1:
        rc_mouse_mode = st.radio(
            "마우스 조작 모드",
            (
                "⬚ Box ROI",
                "✏️ Lasso ROI",
                "🔺 Polygon 클릭 ROI",
                "🔍 확대",
            ),
            horizontal=True,
            key="rc_mouse_mode",
            help="확대한 뒤 ROI 선택으로 전환해 구역을 그리세요.",
        )
    with _rc_mouse2:
        st.write("")
        if st.button("↩️ 확대 초기화", key="rc_zoom_reset", width="stretch"):
            st.session_state["rc_zoom_revision"] = (
                int(st.session_state.get("rc_zoom_revision", 0)) + 1
            )
            st.rerun()
    st.caption(
        "① 필요하면 확대 → ② Box/Lasso/Polygon 선택 → ③ 영역을 만든 뒤 이름 입력. "
        "Polygon은 잎 둘레의 꼭짓점을 차례로 클릭하고 완료합니다."
    )
    left, right = st.columns([3, 1.35])
    with left:
        if rc_mouse_mode == "🔺 Polygon 클릭 ROI":
            from PIL import Image, ImageDraw

            polygon_points = list(st.session_state.get("rc_polygon_points", []))
            pc1, pc2, pc3 = st.columns(3)
            if pc1.button(
                "↶ 마지막 점 취소",
                key="rc_polygon_undo",
                width="stretch",
                disabled=not polygon_points,
            ):
                st.session_state["rc_polygon_points"] = polygon_points[:-1]
                st.rerun()
            if pc2.button(
                "🗑️ 모두 지우기",
                key="rc_polygon_clear",
                width="stretch",
                disabled=not polygon_points,
            ):
                st.session_state["rc_polygon_points"] = []
                st.rerun()
            if pc3.button(
                "✅ Polygon 완료",
                key="rc_polygon_finish",
                type="primary",
                width="stretch",
                disabled=len(polygon_points) < 3,
            ):
                st.session_state["rc_draft"] = polygon_region(
                    [point[0] for point in polygon_points],
                    [point[1] for point in polygon_points],
                    height,
                    width,
                )
                st.session_state["rc_polygon_points"] = []
                st.rerun()

            polygon_image = Image.fromarray(rgb).convert("RGB")
            polygon_draw = ImageDraw.Draw(polygon_image)
            line_width = max(2, min(height, width) // 180)
            point_radius = max(3, min(height, width) // 100)
            for idx, item in enumerate(st.session_state["rc_regions"]):
                region = item["region"]
                color_values = PALETTE[idx % len(PALETTE)]
                color = tuple(int(value) for value in color_values)
                if region.get("type") in {"lasso", "polygon"}:
                    saved_points = list(zip(region.get("x", []), region.get("y", [])))
                    if len(saved_points) >= 3:
                        polygon_draw.line(
                            saved_points + [saved_points[0]],
                            fill=color,
                            width=line_width,
                        )
                else:
                    r0, r1, c0, c1 = region["roi"]
                    polygon_draw.rectangle(
                        (c0, r0, c1, r1), outline=color, width=line_width
                    )
            if len(polygon_points) >= 2:
                polygon_draw.line(
                    polygon_points, fill="#ffd54f", width=line_width
                )
            for point_index, (point_x, point_y) in enumerate(polygon_points, 1):
                polygon_draw.ellipse(
                    (
                        point_x - point_radius,
                        point_y - point_radius,
                        point_x + point_radius,
                        point_y + point_radius,
                    ),
                    fill="#ffd54f",
                    outline="#111111",
                )
                polygon_draw.text(
                    (point_x + point_radius, point_y - point_radius),
                    str(point_index),
                    fill="#ffffff",
                )
            polygon_click = streamlit_image_coordinates(
                polygon_image,
                height=680,
                key=f"rc_polygon_image_{st.session_state.get('rc_file', '')}",
                cursor="crosshair",
            )
            if polygon_click:
                click_id = polygon_click.get("unix_time")
                if click_id != st.session_state.get("rc_polygon_last_click"):
                    display_width = max(1, int(polygon_click.get("width", width)))
                    display_height = max(1, int(polygon_click.get("height", height)))
                    point_x = np.clip(
                        float(polygon_click["x"]) * width / display_width,
                        0,
                        max(0, width - 1),
                    )
                    point_y = np.clip(
                        float(polygon_click["y"]) * height / display_height,
                        0,
                        max(0, height - 1),
                    )
                    st.session_state["rc_polygon_points"] = polygon_points + [
                        (float(point_x), float(point_y))
                    ]
                    st.session_state["rc_polygon_last_click"] = click_id
                    st.rerun()
            st.caption(f"현재 꼭짓점: **{len(polygon_points)}개** · 최소 3개")
        else:
            fig = go.Figure(go.Image(z=rgb))
            # Plotly image traces do not emit box/lasso selection events by
            # themselves. A sparse transparent grid provides selectable points.
            target_axis_points = 128
            grid_step_y = max(1, int(np.ceil(height / target_axis_points)))
            grid_step_x = max(1, int(np.ceil(width / target_axis_points)))
            grid_y = np.arange(0, height, grid_step_y)
            grid_x = np.arange(0, width, grid_step_x)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
            fig.add_trace(
                go.Scattergl(
                    x=grid_xx.ravel(),
                    y=grid_yy.ravel(),
                    mode="markers",
                    marker=dict(
                        size=max(6, min(18, max(grid_step_x, grid_step_y) + 3)),
                        color="rgba(0,0,0,0.002)",
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    name="_roi_selection_grid",
                )
            )
            for idx, item in enumerate(st.session_state["rc_regions"]):
                color = f"rgb({','.join(map(str, PALETTE[idx % len(PALETTE)]))})"
                _add_region_overlay(fig, item, color)
            fig.update_layout(
                dragmode=(
                    "zoom" if rc_mouse_mode == "🔍 확대"
                    else "lasso" if rc_mouse_mode == "✏️ Lasso ROI"
                    else "select"
                ),
                height=680,
                margin=dict(l=0, r=0, t=5, b=0),
                newselection=dict(line=dict(color="#ffd54f", width=3)),
                uirevision=(
                    f"{st.session_state.get('rc_file', '')}|"
                    f"{st.session_state.get('rc_zoom_revision', 0)}"
                ),
            )
            fig.update_xaxes(showticklabels=True, title="Column")
            fig.update_yaxes(showticklabels=True, title="Row")
            event = st.plotly_chart(
                fig,
                key="rc_region_chart",
                on_select="rerun",
                selection_mode=("box", "lasso"),
                width="stretch",
                config=ROI_PLOTLY_CONFIG,
            )
            if event is not None and hasattr(event, "selection"):
                picked = selection_to_region(event.selection, height, width)
                if picked is not None:
                    st.session_state["rc_draft"] = picked

    with right:
        draft = st.session_state.get("rc_draft")
        next_name = f"ROI {len(st.session_state['rc_regions']) + 1}"
        roi_name = st.text_input("새 ROI 이름", value=next_name, key="rc_new_name")
        if draft:
            st.write(f"선택: `{draft.get('type')}`")
            st.code(f"row {draft['roi'][0]}:{draft['roi'][1]}\ncol {draft['roi'][2]}:{draft['roi'][3]}")
            if st.button("➕ 선택 구역 추가", type="primary", width="stretch"):
                st.session_state["rc_regions"].append(
                    {"id": _region_id(), "name": roi_name.strip() or next_name, "region": dict(draft)}
                )
                st.session_state["rc_draft"] = None
                st.session_state["rc_results"] = []
                st.session_state["rc_roi_baselines"] = {}
                st.session_state["rc_roi_baseline_metrics"] = {}
                st.session_state["rc_roi_trials"] = {}
                st.session_state["rc_roi_accepted"] = {}
                st.rerun()
        else:
            st.info("왼쪽 영상에서 영역을 먼저 선택하세요.")

        with st.expander("좌표로 사각형 추가"):
            with st.form("rc_manual_roi"):
                r0 = st.number_input("row 시작", 0, height - 1, 0)
                r1 = st.number_input("row 끝", 1, height, height)
                c0 = st.number_input("col 시작", 0, width - 1, 0)
                c1 = st.number_input("col 끝", 1, width, width)
                manual_name = st.text_input("이름", value=next_name)
                if st.form_submit_button("좌표 ROI 추가"):
                    region = box_region([r0, r1, c0, c1], height, width)
                    st.session_state["rc_regions"].append(
                        {"id": _region_id(), "name": manual_name.strip() or next_name, "region": region}
                    )
                    st.session_state["rc_results"] = []
                    st.session_state["rc_roi_baselines"] = {}
                    st.session_state["rc_roi_baseline_metrics"] = {}
                    st.session_state["rc_roi_trials"] = {}
                    st.session_state["rc_roi_accepted"] = {}
                    st.rerun()

        st.markdown("#### 저장된 구역")
        if not st.session_state["rc_regions"]:
            st.caption("아직 저장된 ROI가 없습니다.")
        for idx, item in enumerate(list(st.session_state["rc_regions"])):
            a, b = st.columns([4, 1])
            a.write(f"**{idx + 1}. {item['name']}**  \n`{item['region']['type']}` · `{item['region']['roi']}`")
            if b.button("삭제", key=f"rc_del_{idx}"):
                del st.session_state["rc_regions"][idx]
                st.session_state["rc_results"] = []
                st.session_state["rc_roi_baselines"] = {}
                st.session_state["rc_roi_baseline_metrics"] = {}
                st.session_state["rc_roi_trials"] = {}
                st.session_state["rc_roi_accepted"] = {}
                st.rerun()
        if st.session_state["rc_regions"] and st.button("모든 ROI 삭제", width="stretch"):
            st.session_state["rc_regions"] = []
            st.session_state["rc_results"] = []
            st.session_state["rc_roi_baselines"] = {}
            st.session_state["rc_roi_baseline_metrics"] = {}
            st.session_state["rc_roi_trials"] = {}
            st.session_state["rc_roi_accepted"] = {}
            st.rerun()

with tab_run:
    regions = st.session_state["rc_regions"]
    if not regions:
        st.warning("먼저 ‘구역 나누기’ 탭에서 하나 이상의 ROI를 추가하세요.")
    else:
        st.markdown("### 전체 이미지 모델 1회 학습")
        st.caption(
            f"기존 분석의 **{METHODS[method]}** 방법을 전체 이미지에 적용한 뒤, "
            f"저장된 {len(regions)}개 ROI에 같은 클러스터 라벨을 적용합니다."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ROI": item["name"], "형태": item["region"]["type"],
                        "row": f"{item['region']['roi'][0]}:{item['region']['roi'][1]}",
                        "col": f"{item['region']['roi'][2]}:{item['region']['roi'][3]}",
                    }
                    for item in regions
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        if calibration_path:
            st.info("반사도 보정계수를 전체 이미지 군집화와 ROI 스펙트럼 계산에 적용합니다.")
        else:
            st.warning("보정계수가 없습니다. 선택한 기존 정규화 설정으로 처리합니다.")

        _global_work_units = _array_work_units(data.shape, data.dtype.itemsize)
        _global_work_units += 0.03 * len(regions)
        _global_timing_history = [
            record
            for record in st.session_state.get("rc_timing_history", [])
            if record.get("scope") == "global"
        ]
        _global_estimated_seconds = _estimate_seconds(
            _global_work_units,
            method,
            _global_timing_history,
        )
        if not _global_timing_history:
            # The cube is already memory-resident on this screen, so its first
            # run is materially faster than the full-field load+report pipeline.
            _global_estimated_seconds *= 0.35
        st.info(
            f"⏱ **예상 분석시간: {_format_estimate(_global_estimated_seconds)}**  ·  "
            f"로드된 배열 {data.nbytes / (1024**3):.2f} GB  ·  ROI {len(regions)}개"
        )
        _last_timing = st.session_state.get("rc_last_timing")
        if _last_timing:
            st.caption(
                f"최근 실제 소요시간: "
                f"**{_format_duration(_last_timing['elapsed_seconds'])}** · "
                f"{str(_last_timing.get('method', '')).upper()}"
            )

        if st.button("🚀 전체 클러스터링 후 ROI 스펙트럼 추출", type="primary", width="stretch"):
            _run_started = time.perf_counter()
            try:
                if method in {"supervised", "cnn"} and not labels_csv.strip():
                    raise ValueError(f"{METHODS[method]} 방법에는 라벨 CSV가 필요합니다.")
                cfg = _shared_config(
                    method, n_classes, normalize_mode,
                    ndvi_threshold, brightness_threshold, angle_threshold,
                    ae_epochs, cnn_epochs,
                    hdbscan_min_cluster_size, hdbscan_min_samples,
                )
                global_settings = {
                    "method": method,
                    "n_classes": n_classes,
                    "normalize_mode": normalize_mode,
                    "ndvi_threshold": ndvi_threshold,
                    "brightness_threshold": brightness_threshold,
                    "angle_threshold": angle_threshold,
                    "ae_epochs": ae_epochs,
                    "cnn_epochs": cnn_epochs,
                    "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
                    "hdbscan_min_samples": hdbscan_min_samples,
                    "calibration_path": calibration_path.strip(),
                    "labels_csv": labels_csv.strip(),
                    "source_file": st.session_state["rc_file"],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                progress = st.progress(0.0, text="기존 파이프라인과 동일한 전처리 적용 중")
                processed, processed_wl, value_units, calibration_info = _prepare_shared_data(
                    data, wavelengths, cfg, calibration_path,
                    st.session_state["rc_file"],
                )
                cluster_data, cluster_wl, cluster_input = _prepare_clustering_data(
                    data,
                    wavelengths,
                    cfg,
                    processed,
                    processed_wl,
                    calibration_info,
                    st.session_state["rc_file"],
                )
                progress.progress(0.25, text=f"{METHODS[method]} 전체 이미지 분석 중")
                classifier = HyperspectralClassifier(cfg)
                class_map, class_info = classifier.classify(
                    cluster_data, cluster_wl, labels_csv.strip() or None
                )
                st.session_state["rc_global_result"] = {
                    "method": method,
                    "class_info": class_info,
                    "class_map": class_map,
                    "hybrid_base_map": classifier.last_base_class_map,
                }
                global_settings["selected_calibration"] = calibration_info
                global_settings["clustering_input"] = cluster_input
                global_settings["effective_normalization"] = (
                    (calibration_info or {}).get(
                        "effective_normalization", normalize_mode
                    )
                )
                if calibration_info:
                    global_settings["calibration_path"] = str(
                        calibration_info.get("selected_profile")
                        or calibration_path.strip()
                    )

                results = []
                baselines = {}
                baseline_metrics = {}
                for idx, item in enumerate(regions):
                    progress.progress(
                        0.70 + 0.25 * idx / len(regions),
                        text=f"{item['name']} 스펙트럼 집계 중 ({idx + 1}/{len(regions)})",
                    )
                    baseline = summarize_region_from_class_map(
                            processed,
                            item["region"],
                            class_map,
                            name=item["name"],
                            wavelengths=processed_wl,
                            method=method,
                            source_scope="global",
                            value_units=value_units,
                        )
                    results.append(baseline)
                    baselines[item["id"]] = baseline
                    baseline_metrics[item["id"]] = _quality_metrics(cluster_data, baseline)
                st.session_state["rc_results"] = results
                st.session_state["rc_global_settings"] = global_settings
                st.session_state["rc_roi_baselines"] = baselines
                st.session_state["rc_roi_baseline_metrics"] = baseline_metrics
                st.session_state["rc_roi_trials"] = {item["id"]: [] for item in regions}
                st.session_state["rc_roi_accepted"] = {item["id"]: None for item in regions}
                st.session_state["rc_last_report"] = ""
                _elapsed_seconds = time.perf_counter() - _run_started
                _timing_record = {
                    "scope": "global",
                    "method": method,
                    "work_units": _global_work_units,
                    "elapsed_seconds": _elapsed_seconds,
                    "estimated_seconds": _global_estimated_seconds,
                    "roi_count": len(regions),
                }
                st.session_state["rc_last_timing"] = _timing_record
                st.session_state["rc_timing_history"].append(_timing_record)
                st.session_state["rc_timing_history"] = st.session_state[
                    "rc_timing_history"
                ][-30:]
                global_settings["timing"] = _timing_record
                _clear_roi_widget_state()
                progress.progress(1.0, text="완료")
                if calibration_info:
                    _selected_profile = calibration_info.get("selected_profile") or ""
                    _time_delta = (calibration_info.get("meta") or {}).get(
                        "white_time_delta_seconds"
                    )
                    st.info(
                        "적용 White: " + str(_selected_profile)
                        + (
                            f" · 촬영시각 차이 {float(_time_delta) / 60:.1f}분"
                            if _time_delta is not None else ""
                        )
                    )
                st.success(
                    "전체 클러스터링과 구역별 스펙트럼 추출을 완료했습니다. "
                    f"⏱ 실제 소요시간: **{_format_duration(_elapsed_seconds)}**"
                )
            except Exception:
                st.error("전체 클러스터링 또는 ROI 스펙트럼 추출에 실패했습니다.")
                st.code(traceback.format_exc(), language="python")

with tab_results:
    results = st.session_state.get("rc_results") or []
    _result_timing = st.session_state.get("rc_last_timing")
    if _result_timing and results:
        st.success(
            f"⏱ 최근 전체 분석 실제 소요시간: "
            f"**{_format_duration(_result_timing['elapsed_seconds'])}**  ·  "
            f"실행 전 예상 {_format_estimate(_result_timing.get('estimated_seconds'))}"
        )
    if not results:
        st.info("전체 클러스터링을 실행하면 구역별 결과가 여기에 표시됩니다.")
    else:
        global_result = st.session_state.get("rc_global_result")
        st.info(
            "`전역 모델` 결과는 모든 ROI에서 같은 Cluster ID를 뜻합니다. "
            "`ROI 재분석` 결과는 선택한 구역 안에서만 새로 정의된 Cluster ID입니다."
        )
        if global_result and isinstance(global_result.get("class_map"), np.ndarray):
            global_map = np.asarray(global_result["class_map"])
            class_info = list(global_result.get("class_info") or [])
            class_ids = sorted(int(v) for v in np.unique(global_map) if int(v) >= 0)
            st.markdown("### 전체 이미지 클러스터 검수")
            st.caption(
                "표시할 클러스터를 하나 또는 여러 개 선택하세요. 선택하지 않은 영역은 "
                "원본 RGB로 남으므로 잎·그림자·토양의 실제 위치와 바로 비교할 수 있습니다."
            )

            global_preview_col, global_control_col = st.columns(
                [4.2, 1.35], vertical_alignment="top"
            )
            with global_control_col:
                selected_global_ids = st.multiselect(
                    "표시할 클러스터",
                    options=class_ids,
                    default=class_ids,
                    format_func=lambda value: f"Cluster {value}",
                    key="rc_global_overlay_clusters",
                    help="하나만 남기면 해당 클러스터의 공간 분리 상태만 볼 수 있습니다.",
                )
                global_overlay_opacity = st.slider(
                    "색상 투명도",
                    min_value=0.0, max_value=1.0, value=0.60, step=0.05,
                    key="rc_global_overlay_opacity",
                    help="낮추면 원본 RGB가, 높이면 클러스터 색상이 더 선명합니다.",
                )
                global_show_boundaries = st.checkbox(
                    "흰색 경계선",
                    value=True,
                    key="rc_global_overlay_boundaries",
                )
                st.caption("선택 클러스터와 표시 설정이 이미지 바로 옆에 적용됩니다.")
            with global_preview_col:
                st.image(
                    _global_overlay(
                        rgb,
                        global_map,
                        global_overlay_opacity,
                        global_show_boundaries,
                        selected_global_ids,
                    ),
                    caption="RGB + 선택한 전체 이미지 클러스터",
                    width="stretch",
                )

            with st.expander("원본 RGB · 전체 컬러 맵 비교", expanded=False):
                original_col, map_col = st.columns(2)
                original_col.image(rgb, caption="원본 RGB", width="stretch")
                map_col.image(
                    _solid_class_map(global_map),
                    caption="전체 클러스터 컬러 맵",
                    width="stretch",
                )

            hybrid_base = global_result.get("hybrid_base_map")
            if isinstance(hybrid_base, np.ndarray):
                semantic_names = {
                    0: "배경/깊은 그림자", 1: "밝은 잎", 2: "그림자 잎", 3: "토양"
                }
                with st.expander("Hybrid 1차 의미 분리 확인", expanded=False):
                    st.caption(
                        "세부 K-means 이전 결과입니다. 밝은 잎과 그림자 잎의 위치가 "
                        "실제 영상과 맞는지 확인하세요."
                    )
                    semantic_cols = st.columns(4)
                    for semantic_id in range(4):
                        count = int(np.sum(hybrid_base == semantic_id))
                        semantic_cols[semantic_id].image(
                            _highlight_class(rgb, hybrid_base, semantic_id),
                            caption=(
                                f"{semantic_names[semantic_id]} · ID {semantic_id}\n"
                                f"{count:,} px "
                                f"({100 * count / max(1, hybrid_base.size):.1f}%)"
                            ),
                            width="stretch",
                        )
                missing_leaf_groups = [
                    semantic_names[class_id]
                    for class_id in (1, 2)
                    if not np.any(hybrid_base == class_id)
                ]
                if missing_leaf_groups:
                    st.warning(
                        "현재 설정에서는 " + ", ".join(missing_leaf_groups)
                        + " 영역이 검출되지 않았습니다. 밝기 임계값 또는 정규화 방식을 "
                        "조정한 뒤 다시 분석해 보세요."
                    )

            with st.expander("🔍 전체 이미지에서 클러스터별 단독 확인", expanded=False):
                cards = st.columns(min(3, max(1, len(class_ids))))
                for card_index, class_id in enumerate(class_ids):
                    count = int(np.sum(global_map == class_id))
                    pct = 100.0 * count / max(1, global_map.size)
                    cards[card_index % len(cards)].image(
                        _highlight_class(rgb, global_map, class_id),
                        caption=(
                            f"{_class_name(class_id, class_info)} · ID {class_id}\n"
                            f"{count:,} px ({pct:.1f}%)"
                        ),
                        width="stretch",
                    )

            st.divider()
        regions = st.session_state["rc_regions"]
        global_settings = st.session_state.get("rc_global_settings") or {}
        baselines = st.session_state.get("rc_roi_baselines") or {}
        baseline_metrics = st.session_state.get("rc_roi_baseline_metrics") or {}
        trials_by_roi = st.session_state.get("rc_roi_trials") or {}
        accepted_by_roi = st.session_state.get("rc_roi_accepted") or {}

        for idx, item in enumerate(regions):
            roi_id = item["id"]
            current_result = results[idx]
            baseline = baselines.get(roi_id, current_result)
            accepted_trial = accepted_by_roi.get(roi_id)
            status = "전역 기준" if not accepted_trial else "ROI 시험 채택"
            with st.expander(
                f"{idx + 1}. {item['name']} · {status}", expanded=idx == 0
            ):
                current_tab, trial_tab = st.tabs(["현재 채택 결과", "ROI별 재분석 설정·비교"])
                with current_tab:
                    _render_result(current_result, rgb, f"rc_current_{roi_id}")

                with trial_tab:
                    st.info(
                        "반사도 보정·정규화·불량 밴드·다운샘플링은 전역 분석값으로 고정됩니다. "
                        "아래 클러스터링 값만 이 ROI에 독립적으로 적용됩니다."
                    )
                    allowed_methods = [
                        "hybrid", "kmeans", "sam", "autoencoder", "hdbscan", "gmm", "nmf"
                    ]
                    default_method = global_settings.get("method", "kmeans")
                    if default_method not in allowed_methods:
                        default_method = "kmeans"
                    roi_method = st.selectbox(
                        "ROI 분석 방법", allowed_methods,
                        index=allowed_methods.index(default_method),
                        format_func=lambda value: METHODS[value],
                        key=f"rc_roi_cfg_method_{roi_id}",
                    )
                    left_cfg, right_cfg = st.columns(2)
                    with left_cfg:
                        roi_n_classes = st.slider(
                            "클러스터 수", 2, 20,
                            int(global_settings.get("n_classes") or 6),
                            key=f"rc_roi_cfg_n_{roi_id}",
                        )
                        roi_ndvi = st.slider(
                            "NDVI 임계값", 0.0, 1.0,
                            float(global_settings.get("ndvi_threshold", 0.15)), 0.01,
                            key=f"rc_roi_cfg_ndvi_{roi_id}",
                            disabled=roi_method != "hybrid",
                        )
                        roi_brightness = st.slider(
                            "밝기 임계값", 0.0, 0.5,
                            float(global_settings.get("brightness_threshold", 0.08)), 0.01,
                            key=f"rc_roi_cfg_bright_{roi_id}",
                            disabled=roi_method != "hybrid",
                        )
                    with right_cfg:
                        roi_angle = st.slider(
                            "SAM 각도 임계값", 0.0, 0.5,
                            float(global_settings.get("angle_threshold", 0.10)), 0.01,
                            key=f"rc_roi_cfg_angle_{roi_id}",
                            disabled=roi_method != "sam",
                        )
                        roi_hdb_size = st.slider(
                            "HDBSCAN min_cluster_size", 10, 500,
                            int(global_settings.get("hdbscan_min_cluster_size", 50)), 10,
                            key=f"rc_roi_cfg_hsize_{roi_id}",
                            disabled=roi_method != "hdbscan",
                        )
                        roi_hdb_samples = st.slider(
                            "HDBSCAN min_samples", 1, 50,
                            int(global_settings.get("hdbscan_min_samples", 5)), 1,
                            key=f"rc_roi_cfg_hsample_{roi_id}",
                            disabled=roi_method != "hdbscan",
                        )
                    roi_ae_epochs = int(global_settings.get("ae_epochs", 60))
                    if roi_method == "autoencoder":
                        roi_ae_epochs = st.slider(
                            "Autoencoder epochs", 10, 200, roi_ae_epochs, 10,
                            key=f"rc_roi_cfg_ae_{roi_id}",
                        )

                    _trial_work_units = max(
                        0.12, _array_work_units(data.shape, data.dtype.itemsize) * 0.55
                    )
                    _trial_history = [
                        record
                        for record in st.session_state.get("rc_timing_history", [])
                        if record.get("scope") == "roi_trial"
                    ]
                    _trial_estimated_seconds = _estimate_seconds(
                        _trial_work_units, roi_method, _trial_history
                    )
                    if not _trial_history:
                        _trial_estimated_seconds *= 0.35
                    st.caption(
                        f"⏱ 이 ROI 시험 예상시간: "
                        f"**{_format_estimate(_trial_estimated_seconds)}**"
                    )

                    if st.button(
                        "🧪 이 설정으로 시험 실행", type="primary", width="stretch",
                        key=f"rc_roi_trial_run_{roi_id}",
                    ):
                        _trial_started = time.perf_counter()
                        try:
                            trial_settings = {
                                "method": roi_method,
                                "n_classes": roi_n_classes,
                                "ndvi_threshold": roi_ndvi,
                                "brightness_threshold": roi_brightness,
                                "angle_threshold": roi_angle,
                                "ae_epochs": roi_ae_epochs,
                                "hdbscan_min_cluster_size": roi_hdb_size,
                                "hdbscan_min_samples": roi_hdb_samples,
                            }
                            with st.spinner(f"{item['name']} 시험 재클러스터링 중..."):
                                trial_result, trial_metrics = _run_roi_trial(
                                    data, wavelengths, item, trial_settings, global_settings
                                )
                            _trial_elapsed_seconds = time.perf_counter() - _trial_started
                            trial_id = f"{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
                            trial = {
                                "id": trial_id,
                                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "settings": trial_settings,
                                "result": trial_result,
                                "metrics": trial_metrics,
                                "elapsed_seconds": _trial_elapsed_seconds,
                                "estimated_seconds": _trial_estimated_seconds,
                            }
                            st.session_state["rc_roi_trials"].setdefault(roi_id, []).append(trial)
                            st.session_state["rc_timing_history"].append({
                                "scope": "roi_trial",
                                "method": roi_method,
                                "work_units": _trial_work_units,
                                "elapsed_seconds": _trial_elapsed_seconds,
                            })
                            st.session_state["rc_timing_history"] = st.session_state[
                                "rc_timing_history"
                            ][-30:]
                            st.session_state["rc_last_report"] = ""
                            st.rerun()
                        except Exception:
                            st.error("ROI 시험 재클러스터링에 실패했습니다.")
                            st.code(traceback.format_exc(), language="python")

                    roi_trials = trials_by_roi.get(roi_id, [])
                    if not roi_trials:
                        st.caption("아직 시험 결과가 없습니다. 설정을 바꾸고 시험 실행을 누르세요.")
                    else:
                        trial_lookup = {trial["id"]: trial for trial in roi_trials}
                        selected_trial_id = st.selectbox(
                            "비교할 시험 이력",
                            options=list(trial_lookup),
                            format_func=lambda value: (
                                f"{trial_lookup[value]['created_at']} · "
                                f"{METHODS[trial_lookup[value]['settings']['method']]} · "
                                f"K={trial_lookup[value]['settings']['n_classes']} · "
                                f"{_format_duration(trial_lookup[value].get('elapsed_seconds'))}"
                            ),
                            key=f"rc_roi_trial_select_{roi_id}",
                        )
                        selected_trial = trial_lookup[selected_trial_id]
                        candidate = selected_trial["result"]
                        if selected_trial.get("elapsed_seconds") is not None:
                            st.success(
                                f"⏱ 이 ROI 시험 실제 소요시간: "
                                f"**{_format_duration(selected_trial['elapsed_seconds'])}**  ·  "
                                f"실행 전 예상 "
                                f"{_format_estimate(selected_trial.get('estimated_seconds'))}"
                            )
                        trial_metrics = dict(selected_trial["metrics"])
                        trial_metrics["changed_pixel_fraction"] = _changed_pixel_fraction(
                            baseline, candidate
                        )
                        st.markdown("#### 전역 기준 ↔ ROI 시험 비교")
                        baseline_ids = [
                            int(value) for value in baseline.display_cluster_ids
                        ]
                        candidate_ids = [
                            int(value) for value in candidate.display_cluster_ids
                        ]
                        base_image_col, base_control_col, trial_image_col, trial_control_col = (
                            st.columns([3.2, 1.25, 3.2, 1.25], vertical_alignment="top")
                        )
                        with base_control_col:
                            selected_baseline_ids = st.multiselect(
                                "기준 표시 클러스터",
                                options=baseline_ids,
                                default=baseline_ids,
                                format_func=lambda value: f"Cluster {value}",
                                key=f"rc_baseline_overlay_clusters_{roi_id}_{selected_trial_id}",
                            )
                            baseline_opacity = st.slider(
                                "기준 투명도", 0.0, 1.0, 0.60, 0.05,
                                key=f"rc_baseline_overlay_opacity_{roi_id}_{selected_trial_id}",
                            )
                            baseline_boundaries = st.checkbox(
                                "기준 경계선", True,
                                key=f"rc_baseline_overlay_boundaries_{roi_id}_{selected_trial_id}",
                            )
                        with trial_control_col:
                            selected_candidate_ids = st.multiselect(
                                "시험 표시 클러스터",
                                options=candidate_ids,
                                default=candidate_ids,
                                format_func=lambda value: f"Cluster {value}",
                                key=f"rc_trial_overlay_clusters_{roi_id}_{selected_trial_id}",
                            )
                            candidate_opacity = st.slider(
                                "시험 투명도", 0.0, 1.0, 0.60, 0.05,
                                key=f"rc_trial_overlay_opacity_{roi_id}_{selected_trial_id}",
                            )
                            candidate_boundaries = st.checkbox(
                                "시험 경계선", True,
                                key=f"rc_trial_overlay_boundaries_{roi_id}_{selected_trial_id}",
                            )
                        with base_image_col:
                            st.image(
                                _cluster_overlay(
                                    rgb,
                                    baseline,
                                    baseline_opacity,
                                    baseline_boundaries,
                                    selected_baseline_ids,
                                ),
                                caption="전역 기준 결과", width="stretch",
                            )
                        with trial_image_col:
                            st.image(
                                _cluster_overlay(
                                    rgb,
                                    candidate,
                                    candidate_opacity,
                                    candidate_boundaries,
                                    selected_candidate_ids,
                                ),
                                caption="ROI 시험 결과", width="stretch",
                            )
                        st.dataframe(
                            _metric_frame(baseline_metrics.get(roi_id, {}), trial_metrics),
                            hide_index=True, width="stretch",
                        )
                        st.plotly_chart(
                            _comparison_figure(baseline, candidate),
                            width="stretch", key=f"rc_compare_spec_{roi_id}_{selected_trial_id}",
                        )
                        with st.expander("이 시험에 사용된 설정"):
                            st.json(selected_trial["settings"])
                        accept_col, restore_col = st.columns(2)
                        if accept_col.button(
                            "✅ 이 시험 결과 채택", type="primary", width="stretch",
                            key=f"rc_accept_{roi_id}_{selected_trial_id}",
                        ):
                            updated = list(st.session_state["rc_results"])
                            updated[idx] = candidate
                            st.session_state["rc_results"] = updated
                            st.session_state["rc_roi_accepted"][roi_id] = selected_trial_id
                            st.session_state["rc_last_report"] = ""
                            st.rerun()
                        if restore_col.button(
                            "↩ 전역 기준으로 복원", width="stretch",
                            key=f"rc_restore_{roi_id}",
                        ):
                            updated = list(st.session_state["rc_results"])
                            updated[idx] = baseline
                            st.session_state["rc_results"] = updated
                            st.session_state["rc_roi_accepted"][roi_id] = None
                            st.session_state["rc_last_report"] = ""
                            st.rerun()

        st.divider()

        if st.button("💾 전체 결과와 HTML 리포트 저장", type="primary", width="stretch"):
            try:
                roi_decisions = []
                for region_item in regions:
                    region_id = region_item["id"]
                    region_baseline = baselines.get(region_id)
                    trial_records = []
                    for trial in trials_by_roi.get(region_id, []):
                        metrics = dict(trial.get("metrics") or {})
                        if region_baseline is not None:
                            metrics["changed_pixel_fraction"] = _changed_pixel_fraction(
                                region_baseline, trial["result"]
                            )
                        trial_records.append({
                            "trial_id": trial["id"],
                            "created_at": trial["created_at"],
                            "settings": trial["settings"],
                            "metrics": metrics,
                            "elapsed_seconds": trial.get("elapsed_seconds"),
                            "estimated_seconds": trial.get("estimated_seconds"),
                            "accepted": accepted_by_roi.get(region_id) == trial["id"],
                        })
                    roi_decisions.append({
                        "roi_id": region_id,
                        "roi_name": region_item["name"],
                        "baseline_metrics": baseline_metrics.get(region_id, {}),
                        "accepted_trial_id": accepted_by_roi.get(region_id),
                        "current_result": (
                            "global_baseline"
                            if accepted_by_roi.get(region_id) is None
                            else "roi_trial"
                        ),
                        "trials": trial_records,
                    })
                cfg = {
                    "downsample": st.session_state["rc_downsample"],
                    "global_analysis": global_settings,
                    "default_clustering_scope": "global_image",
                    "roi_decisions": roi_decisions,
                }
                report = _save_report(
                    st.session_state["rc_file"], rgb, st.session_state["rc_regions"],
                    results, output_root, cfg, global_result=global_result,
                    raw_data=data, raw_wavelengths=wavelengths,
                )
                st.session_state["rc_last_report"] = str(report)
                st.success(f"저장 완료: {report}")
            except Exception:
                st.error("리포트 저장에 실패했습니다.")
                st.code(traceback.format_exc(), language="python")
        if st.session_state.get("rc_last_report"):
            report_path = Path(st.session_state["rc_last_report"])
            st.code(str(report_path))
            _report_open_col, _folder_open_col, _report_download_col = st.columns(3)
            with _report_open_col:
                if st.button("🌐 HTML 리포트 열기", width="stretch"):
                    try:
                        _open_local_path(report_path)
                        st.success("기본 웹브라우저에서 리포트를 열었습니다.")
                    except Exception as exc:
                        st.error(f"HTML 리포트를 열지 못했습니다: {exc}")
            with _folder_open_col:
                if st.button("📂 결과 폴더 열기", width="stretch"):
                    try:
                        _open_output_folder(report_path.parent)
                        st.success(f"파일 탐색기에서 열었습니다: {report_path.parent}")
                    except Exception as exc:
                        st.error(f"결과 폴더를 열지 못했습니다: {exc}")
            with _report_download_col:
                st.download_button(
                    "⬇️ HTML 리포트 다운로드",
                    data=report_path.read_bytes(),
                    file_name=report_path.name,
                    mime="text/html",
                    width="stretch",
                )

with tab_export:
    st.markdown("### Science-ready 보정·Binning BIL 생성")
    st.caption(
        "현재 화면에 로드된 다운샘플 배열이 아니라 디스크의 **원본 ENVI BIL**을 "
        "청크 단위로 읽습니다. 센서 dark와 촬영시각이 가장 가까운 White를 적용한 뒤 "
        "공간 평균 binning하여 float32 reflectance BIL/HDR로 저장합니다."
    )
    st.info(
        "원본 파일은 변경하지 않습니다. 반사도는 품질 확인을 위해 0~1로 강제 자르지 않으며, "
        "White-Dark 응답이 유효하지 않은 밴드는 NaN으로 보존합니다."
    )
    _export_source = st.session_state.get("rc_file", "")
    _local_export = bool(_export_source) and not str(_export_source).startswith("github:")
    _exp1, _exp2 = st.columns(2)
    with _exp1:
        _export_factor = st.selectbox(
            "공간 binning", [1, 2, 4, 8], index=2,
            format_func=lambda value: f"{value}×{value}",
            key="rc_export_bin_factor",
        )
    _source_path = Path(_export_source) if _local_export else Path("scene.hdr")
    _default_export = _source_path.with_name(
        f"{_source_path.stem}_bin{_export_factor}_reflectance.bil"
    )
    with _exp2:
        _export_output = st.text_input(
            "출력 BIL 경로", value=str(_default_export), key="rc_export_output"
        )
    st.code(f"원본: {_export_source}\n보정 프로파일: {calibration_path or '(미지정)'}")
    if not _local_export:
        st.warning("BIL 내보내기는 현재 로컬 ENVI 파일에서만 지원합니다.")
    if st.button(
        "🧪 보정된 binned BIL 생성", type="primary", width="stretch",
        disabled=not _local_export, key="rc_export_bil",
    ):
        try:
            if not calibration_path.strip():
                raise ValueError("왼쪽에서 White/Dark 프로파일 .npz 또는 폴더를 지정하세요.")
            from src.radiometry import export_calibrated_binned_envi

            with st.spinner(
                "원본 BIL을 청크 단위로 보정·binning 중입니다. 큰 파일은 수 분 이상 걸릴 수 있습니다..."
            ):
                _export_info = export_calibrated_binned_envi(
                    _export_source, calibration_path.strip(), _export_output,
                    bin_factor=int(_export_factor),
                )
            st.success(
                f"✅ 생성 완료: {_export_info['shape'][0]} × "
                f"{_export_info['shape'][1]} × {_export_info['shape'][2]}"
            )
            st.code(
                f"BIL: {_export_info['data_file']}\n"
                f"HDR: {_export_info['header_file']}\n"
                f"처리기록: {_export_info['manifest_file']}\n"
                f"White: {_export_info['selected_profile']}"
            )
            _delta = _export_info.get("calibration_meta", {}).get(
                "white_time_delta_seconds"
            )
            if _delta is not None:
                st.caption(f"대상 영상과 선택 White의 시간 차이: {float(_delta)/60:.1f}분")
        except Exception:
            st.error("보정된 binned BIL 생성에 실패했습니다.")
            st.code(traceback.format_exc(), language="python")
