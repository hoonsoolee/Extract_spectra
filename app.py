"""
app.py
------
Streamlit GUI for CanopySpectra.

Run with:
    python -m streamlit run app.py
"""

import importlib
import re
import sys
import traceback
import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# Make sure 'src' package is importable from this directory
sys.path.insert(0, str(Path(__file__).parent))

from src.timing import (
    estimate_seconds as _estimate_seconds,
    file_work_units as _file_work_units,
    format_duration as _format_duration,
    format_estimate as _format_estimate,
)
from src.path_picker import (
    choose_directory as _choose_directory,
    choose_file as _choose_file,
    native_dialogs_available as _native_dialogs_available,
)
from src.report_options import REPORT_PRESETS
from src.local_open import open_local_path as _open_local_path
from src.analysis_job import (
    ACTIVE_STATES as _ACTIVE_JOB_STATES,
    cancel_analysis_job as _cancel_analysis_job,
    launch_analysis_job as _launch_analysis_job,
    poll_analysis_job as _poll_analysis_job,
    read_analysis_result as _read_analysis_result,
    read_job_log as _read_job_log,
)

# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="CanopySpectra",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("analysis_job", None)
st.session_state.setdefault("run_timing_history", [])
st.session_state.setdefault("run_last_timing", None)
st.session_state.setdefault("run_last_reports", [])
st.session_state.setdefault("run_last_output_dir", "")
st.session_state.setdefault("run_last_review_dirs", [])
st.session_state.setdefault("run_last_team_packages", [])

# Replace Streamlit's filename-based navigation ("app") with research-facing
# labels that explain what each screen does.
st.markdown(
    "<style>[data-testid='stSidebarNav']{display:none}</style>",
    unsafe_allow_html=True,
)
with st.sidebar:
    st.markdown("### 🧭 분석 화면")
    st.page_link("app.py", label="🌿 전체 필드 자동 분석")
    st.page_link("pages/2_구역별_클러스터링.py", label="🗺️ ROI 구역 분석·재클러스터링")
    st.divider()

_LOCAL_HSI_EXTS = {".hdr", ".tif", ".tiff", ".h5", ".hdf5", ".mat", ".ceres"}
_ROI_PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["select2d", "lasso2d"],
}


@st.cache_data(show_spinner=False, ttl=30)
def _scan_local_hsi_files(folder: str) -> tuple[str, ...]:
    """List local hyperspectral entry files without loading their pixels."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        return ()
    return tuple(
        str(path)
        for path in sorted(
            file for file in root.rglob("*")
            if file.is_file() and file.suffix.lower() in _LOCAL_HSI_EXTS
        )
    )


def _browse_directory_into_state(
    target_key: str,
    title: str,
    *,
    set_values: dict | None = None,
) -> None:
    """Streamlit callback: choose a local folder before the script reruns."""
    try:
        selected = _choose_directory(title, st.session_state.get(target_key, ""))
        if selected:
            st.session_state[target_key] = selected
            for key, value in (set_values or {}).items():
                st.session_state[key] = value
            st.session_state["path_picker_notice"] = f"선택됨: {selected}"
    except Exception as exc:
        st.session_state["path_picker_error"] = str(exc)


def _browse_file_into_state(
    target_key: str,
    title: str,
    *,
    set_values: dict | None = None,
) -> None:
    """Streamlit callback: choose a local HSI/calibration file."""
    try:
        selected = _choose_file(
            title,
            st.session_state.get(target_key, ""),
            filetypes=(
                ("초분광/ENVI/CERES", "*.hdr *.bil *.bip *.bsq *.raw *.img *.dat *.ceres"),
                ("보정/데이터/메타데이터", "*.npz *.h5 *.hdf5 *.mat *.tif *.tiff *.csv"),
                ("모든 파일", "*.*"),
            ),
        )
        if selected:
            st.session_state[target_key] = selected
            for key, value in (set_values or {}).items():
                st.session_state[key] = value
            st.session_state["path_picker_notice"] = f"선택됨: {selected}"
    except Exception as exc:
        st.session_state["path_picker_error"] = str(exc)


def _apply_completed_analysis_job(job: dict) -> bool:
    """Move a worker result into the existing result-view session state."""
    if job.get("result_applied"):
        return True
    result = _read_analysis_result(job)
    timing_record = result.get("timing_record")
    if not result or not isinstance(timing_record, dict):
        return False

    st.session_state["run_last_timing"] = timing_record
    if float(timing_record.get("work_units") or 0.0) > 0:
        st.session_state["run_timing_history"].append(timing_record)
        st.session_state["run_timing_history"] = st.session_state[
            "run_timing_history"
        ][-30:]
    st.session_state["run_last_reports"] = list(result.get("reports") or [])
    st.session_state["run_last_output_dir"] = str(result.get("output_dir") or "")
    st.session_state["run_last_review_dirs"] = list(
        result.get("review_dirs") or []
    )
    st.session_state["run_last_team_packages"] = list(
        result.get("team_packages") or []
    )
    updated = dict(job)
    updated["result_applied"] = True
    st.session_state["analysis_job"] = updated
    return True


@st.fragment(run_every="2s")
def _render_analysis_job_status() -> None:
    """Auto-refreshing status panel; remains clickable during computation."""
    job = st.session_state.get("analysis_job")
    if not job:
        return
    state = _poll_analysis_job(job)
    state_name = str(state.get("state") or "idle")
    started_at = float(
        state.get("started_at") or state.get("launched_at")
        or job.get("launched_at") or 0.0
    )
    elapsed = max(0.0, datetime.datetime.now().timestamp() - started_at) \
        if started_at else 0.0

    if state_name in _ACTIVE_JOB_STATES:
        st.info(
            f"⏳ **분석 실행 중** · 경과 {_format_duration(elapsed)}  "
            "\n다른 탭을 확인해도 분석은 계속됩니다."
        )
        estimate = state.get("estimated_seconds") or job.get("estimated_seconds")
        if estimate:
            ratio = min(0.95, elapsed / max(float(estimate), 1.0))
            st.progress(
                ratio,
                text=(
                    f"예상 {_format_estimate(float(estimate))} · "
                    "파일 크기와 분석 단계에 따라 달라질 수 있습니다."
                ),
            )
        if st.button(
            "⏹️ 현재 분석 중지",
            type="secondary",
            use_container_width=True,
            key="stop_analysis_main",
        ):
            with st.spinner("분석 프로세스를 중지하고 있습니다..."):
                _cancel_analysis_job(job)
            st.rerun()
    elif state_name == "completed":
        if not job.get("result_applied"):
            st.rerun()
        st.success(
            "✅ 분석 완료!  ⏱ 실제 총 소요시간: "
            f"**{_format_duration(float(state.get('elapsed_seconds') or elapsed))}**"
        )
    elif state_name == "cancelled":
        st.warning(
            "⏹️ 분석이 중지되었습니다. 중지 전에 완성된 파일은 결과 폴더에 "
            "남아 있으며, 새 분석을 바로 시작할 수 있습니다."
        )
    elif state_name == "failed":
        st.error("❌ 분석 작업이 실패했습니다.")
        if state.get("error"):
            with st.expander("오류 상세", expanded=True):
                st.code(str(state["error"]), language="python")

    log_lines = _read_job_log(job)
    if log_lines:
        with st.expander(
            f"📋 실행 로그 ({len(log_lines)}개 최근 줄)",
            expanded=state_name == "failed",
        ):
            st.code("\n".join(log_lines), language="text")


# ============================================================
# Method metadata
# ============================================================

METHODS = {
    "hybrid": {
        "label":  "🌿 Hybrid  (NDVI + 밝기 + K-means)",
        "kind":   "비지도",
        "help":   (
            "NDVI로 식생 감지 → 밝기로 그림자 분리 → K-means 세분화.  \n"
            "라벨 없이 사용할 수 있는 **기본 추천 방법**입니다."
        ),
    },
    "kmeans": {
        "label":  "📊 K-Means  (비지도)",
        "kind":   "비지도",
        "help":   (
            "PCA 차원 축소 → K-means 클러스터링.  \n"
            "탐색적 분석·라벨 없는 상황에 적합합니다."
        ),
    },
    "sam": {
        "label":  "📐 SAM  (스펙트럼 각도 매핑)",
        "kind":   "비지도 / 지도",
        "help":   (
            "스펙트럼 벡터의 **각도**만 비교 → 조명·그림자 영향 없음.  \n"
            "라벨 없이도, 있어도 모두 사용 가능합니다."
        ),
    },
    "supervised": {
        "label":  "🎯 Random Forest  (지도학습)",
        "kind":   "지도",
        "help":   (
            "사용자 라벨(CSV)로 Random Forest 훈련 → 전체 픽셀 분류.  \n"
            "라벨 CSV가 반드시 필요합니다."
        ),
    },
    "autoencoder": {
        "label":  "🤖 Autoencoder  (딥러닝 비지도)",
        "kind":   "비지도",
        "help":   (
            "MLP 오토인코더로 스펙트럼 압축 → 잠재 공간 K-means.  \n"
            "PyTorch 필요 / 라벨 불필요."
        ),
    },
    "cnn": {
        "label":  "🧠 1D-CNN  (딥러닝 지도학습)",
        "kind":   "지도",
        "help":   (
            "1D 합성곱 신경망 픽셀 분류기.  \n"
            "라벨이 충분할 때 가장 높은 정확도.  \n"
            "라벨 CSV + PyTorch 필요."
        ),
    },
    "hdbscan": {
        "label":  "🔵 HDBSCAN  (밀도 기반 클러스터링)",
        "kind":   "비지도",
        "help":   (
            "계층적 밀도 기반 클러스터링 — **클러스터 수를 지정할 필요가 없습니다**.  \n"
            "알고리즘이 자동으로 클러스터 수를 결정합니다.  \n"
            "노이즈 픽셀은 Background(class 0)에 할당됩니다."
        ),
    },
    "gmm": {
        "label":  "📈 GMM  (가우시안 혼합 모델)",
        "kind":   "비지도",
        "help":   (
            "가우시안 혼합 모델을 이용한 확률적 소프트 클러스터링.  \n"
            "PCA 전처리(15 컴포넌트) 후 GMM 피팅.  \n"
            "클래스 수 슬라이더로 컴포넌트 수를 설정합니다."
        ),
    },
    "nmf": {
        "label":  "🧩 NMF  (스펙트럼 언믹싱)",
        "kind":   "비지도",
        "help":   (
            "비음수 행렬 분해 — 스펙트럼을 엔드멤버와 풍도 맵으로 분해합니다.  \n"
            "각 픽셀은 풍도가 가장 높은 엔드멤버 컴포넌트에 할당됩니다.  \n"
            "반사율 데이터는 이미 비음수이므로 전처리 불필요."
        ),
    },
}

KIND_COLOR = {"비지도": "🟢", "지도": "🔵", "비지도 / 지도": "🟡"}

# ============================================================
# Labeling tool – colour palette & helpers
# ============================================================

_DEFAULT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075",
]


def _get_display_rgb(data: np.ndarray, wavelengths) -> np.ndarray:
    """Return a (H, W, 3) uint8 RGB composite for display."""
    B = data.shape[2]
    targets = [660, 550, 450]  # R, G, B wavelengths (nm)
    channels = []
    for t in targets:
        if wavelengths:
            wl  = np.array(wavelengths)
            idx = int(np.argmin(np.abs(wl - t)))
        else:
            frac = (t - 400) / 600.0
            idx  = int(np.clip(frac * (B - 1), 0, B - 1))
        ch = data[:, :, idx].astype(np.float32)
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 > p2:
            ch = (ch - p2) / (p98 - p2)
        channels.append(np.clip(ch, 0, 1))
    return (np.stack(channels, axis=2) * 255).astype(np.uint8)


@st.cache_data(show_spinner=False)
def _load_cluster_review(review_path: str, rgb_path: str, revision: int) -> dict:
    """Load the compact class map and RGB used by the visual QC panel."""
    from PIL import Image

    del revision  # cache invalidation is carried by the file mtime value
    with np.load(review_path, allow_pickle=False) as archive:
        class_map = np.asarray(archive["class_map"], dtype=np.int32)
        class_ids = np.asarray(archive["class_ids"], dtype=np.int32)
        class_names = [str(value) for value in archive["class_names"]]
        class_colors = np.asarray(archive["class_colors"], dtype=np.uint8)
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return {
        "class_map": class_map,
        "class_ids": class_ids,
        "class_names": class_names,
        "class_colors": class_colors,
        "rgb": rgb,
    }


def _render_cluster_review(result_dir: Path, key_prefix: str) -> None:
    """Render RGB, class map, adjustable overlay, and isolated classes."""
    review_path = result_dir / "cluster_review.npz"
    rgb_path = result_dir / "cluster_review_rgb.png"
    if not review_path.is_file() or not rgb_path.is_file():
        legacy = [
            (result_dir / "rgb.png", "분석 RGB"),
            (result_dir / "cluster_map.png", "클러스터 컬러맵"),
            (result_dir / "cluster_overlay.png", "RGB + 클러스터 오버레이"),
        ]
        legacy = [(path, caption) for path, caption in legacy if path.is_file()]
        if not legacy:
            maps = sorted(result_dir.glob("class_map_*.png"))
            legacy = [(path, "클러스터 컬러맵") for path in maps[:1]]
        if legacy:
            st.markdown(f"#### {result_dir.name}")
            columns = st.columns(len(legacy))
            for column, (path, caption) in zip(columns, legacy):
                column.image(str(path), caption=caption, use_container_width=True)
            st.caption(
                "이 결과는 이전 형식입니다. 다시 분석하면 클러스터 선택·투명도·"
                "단독 이미지 기능이 포함됩니다."
            )
        return
    review = _load_cluster_review(
        str(review_path.resolve()),
        str(rgb_path.resolve()),
        max(review_path.stat().st_mtime_ns, rgb_path.stat().st_mtime_ns),
    )
    class_map = review["class_map"]
    class_ids = review["class_ids"]
    class_names = review["class_names"]
    class_colors = review["class_colors"]
    rgb = review["rgb"]
    labels = {
        int(class_id): f"{name} · ID {int(class_id)}"
        for class_id, name in zip(class_ids, class_names)
    }

    st.markdown(f"#### {result_dir.name}")
    _ctl1, _ctl2, _ctl3 = st.columns([2.4, 1, 1])
    with _ctl1:
        selected_ids = st.multiselect(
            "표시할 클러스터",
            options=[int(value) for value in class_ids],
            default=[int(value) for value in class_ids],
            format_func=lambda value: labels.get(value, f"Cluster {value}"),
            key=f"{key_prefix}_clusters",
        )
    with _ctl2:
        opacity = st.slider(
            "색상 투명도", 0.0, 1.0, 0.55, 0.05,
            key=f"{key_prefix}_opacity",
        )
    with _ctl3:
        show_boundaries = st.checkbox(
            "경계선", value=True, key=f"{key_prefix}_boundaries"
        )

    color_map = np.zeros_like(rgb)
    for class_id, color in zip(class_ids, class_colors):
        color_map[class_map == int(class_id)] = color
    selected_mask = np.isin(class_map, selected_ids)
    overlay = rgb.astype(np.float32)
    overlay[selected_mask] = (
        (1.0 - opacity) * overlay[selected_mask]
        + opacity * color_map[selected_mask].astype(np.float32)
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    if show_boundaries:
        boundary = np.zeros(class_map.shape, dtype=bool)
        boundary[:, 1:] |= class_map[:, 1:] != class_map[:, :-1]
        boundary[1:, :] |= class_map[1:, :] != class_map[:-1, :]
        overlay[boundary & selected_mask] = 255

    _img1, _img2, _img3 = st.columns(3)
    _img1.image(rgb, caption="분석 RGB", use_container_width=True)
    _img2.image(color_map, caption="클러스터 컬러맵", use_container_width=True)
    _img3.image(
        overlay,
        caption="RGB + 선택 클러스터 오버레이",
        use_container_width=True,
    )

    counts = [int(np.sum(class_map == int(value))) for value in class_ids]
    st.dataframe(
        pd.DataFrame({
            "클러스터": [labels[int(value)] for value in class_ids],
            "픽셀": counts,
            "비율 (%)": [round(100 * count / max(1, class_map.size), 2) for count in counts],
        }),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("클러스터별 단독 이미지", expanded=False):
        gray = np.mean(rgb.astype(np.float32), axis=2, keepdims=True)
        background = np.repeat(gray * 0.25, 3, axis=2).astype(np.uint8)
        columns = st.columns(min(3, max(1, len(class_ids))))
        for index, (class_id, color, name, count) in enumerate(
            zip(class_ids, class_colors, class_names, counts)
        ):
            isolated = background.copy()
            isolated[class_map == int(class_id)] = color
            columns[index % len(columns)].image(
                isolated,
                caption=(
                    f"{name} · {count:,} px "
                    f"({100 * count / max(1, class_map.size):.1f}%)"
                ),
                use_container_width=True,
            )


def _build_label_figure(
    rgb: np.ndarray,
    lbl_rows: list,
    cls_cfg: list,
) -> go.Figure:
    """
    Build an interactive Plotly figure:
      - RGB image background (px.imshow)
      - Invisible scatter grid → makes the full image area clickable
      - Coloured markers for each labelled pixel
    """
    H, W = rgb.shape[:2]

    fig = px.imshow(rgb, aspect="equal")

    # ── Invisible scatter grid for click capture ──────────────
    # Subsampled so that ~150×150 points cover the image.
    # Nearly transparent but present, so Plotly click/selection events fire.
    step = max(1, min(H, W) // 150)
    ys_g = np.arange(0, H, step)
    xs_g = np.arange(0, W, step)
    xg, yg = np.meshgrid(xs_g, ys_g)
    fig.add_trace(go.Scatter(
        x=xg.ravel().tolist(),
        y=yg.ravel().tolist(),
        mode="markers",
        marker=dict(
            size=step + 3,
            color="rgba(0,0,0,0.01)",   # nearly invisible
        ),
        showlegend=False,
        hovertemplate="row=%{y}  col=%{x}<extra></extra>",
        name="_grid",
    ))

    # ── Label markers (one trace per class for legend) ────────
    cls_map = {c["id"]: c for c in cls_cfg}
    by_class: dict = {}
    for row, col, cid in lbl_rows:
        by_class.setdefault(cid, {"xs": [], "ys": []})
        by_class[cid]["xs"].append(col)
        by_class[cid]["ys"].append(row)

    for cid, pts in by_class.items():
        c = cls_map.get(cid, {"color": "#ffffff", "name": f"Class {cid}"})
        fig.add_trace(go.Scatter(
            x=pts["xs"],
            y=pts["ys"],
            mode="markers",
            marker=dict(
                color=c["color"],
                size=12,
                line=dict(color="white", width=1.5),
                symbol="circle",
            ),
            name=c["name"],
            showlegend=True,
            hovertemplate=f"row=%{{y}}  col=%{{x}} → {c['name']}<extra></extra>",
        ))

    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=28, b=0),
        dragmode="select",
        legend=dict(
            orientation="h",
            y=1.06,
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=11),
        ),
        xaxis=dict(showticklabels=True, title="Col (pixel)"),
        yaxis=dict(showticklabels=True, title="Row (pixel)"),
    )
    return fig


_LBL_SUPPORTED_EXTS = {".hdr", ".tif", ".tiff", ".h5", ".hdf5", ".mat"}


def _do_load_file(path_str: str) -> tuple:
    """
    Load a single hyperspectral file into session state.
    Returns (H, W, B) on success; raises on failure.
    """
    from src.data_loader import HyperspectralLoader
    from src.preprocessor import Preprocessor

    _min_cfg = {
        "data": {},
        "preprocessing": {
            "normalize":          True,
            "remove_bad_bands":   True,
            "bad_band_ranges":    [[1340, 1460], [1790, 1960]],
            "smooth_spectra":     False,
            "spatial_downsample": 1,
        },
    }
    _loader = HyperspectralLoader(_min_cfg["data"])
    _prep   = Preprocessor(_min_cfg)

    _raw, _meta = _loader.load_local(path_str)
    _data, _wl  = _prep.process(_raw, _meta.get("wavelengths"))
    _rgb        = _get_display_rgb(_data, _wl)

    st.session_state["lbl_data"]      = _data
    st.session_state["lbl_wl"]        = _wl
    st.session_state["lbl_rgb"]       = _rgb
    st.session_state["lbl_file"]      = path_str
    st.session_state["lbl_rows"]      = []
    st.session_state["lbl_prev_sel"]  = None
    st.session_state["lbl_file_list"] = []

    return _data.shape


# ============================================================
# Session state – run mode file scanner
# ============================================================

if "run_scan_files" not in st.session_state:
    st.session_state["run_scan_files"] = []
st.session_state.setdefault("ceres_index", None)
st.session_state.setdefault("ceres_index_source", "")
st.session_state.setdefault("ceres_preview", None)
st.session_state.setdefault("ceres_preview_meta", None)
st.session_state.setdefault("ceres_prepared", None)

# ============================================================
# Sidebar – Settings (pipeline run tab)
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ 분석 설정")
    if st.session_state.get("path_picker_error"):
        st.error(st.session_state.pop("path_picker_error"))
    if st.session_state.get("path_picker_notice"):
        st.success(st.session_state.pop("path_picker_notice"))

    # ── Data source ─────────────────────────────────────────
    st.markdown("### 📂 데이터 소스")
    data_src = st.radio(
        "소스",
        ["로컬 폴더", "GitHub 저장소"],
        horizontal=True,
        label_visibility="collapsed",
    )

    local_folder = github_repo = github_folder = github_token = ""

    if data_src == "로컬 폴더":
        _lf1, _lf2 = st.columns([4, 1])
        with _lf1:
            local_folder = st.text_input(
                "폴더 경로",
                value="./data",
                placeholder="C:/data/field_images",
                key="local_folder_path",
            )
        with _lf2:
            st.write("")
            st.button(
                "🪟 선택",
                use_container_width=True,
                key="browse_local_folder",
                on_click=_browse_directory_into_state,
                args=("local_folder_path", "초분광 데이터 폴더 선택"),
                kwargs={"set_values": {"run_scan_files": []}},
                disabled=not _native_dialogs_available(),
                help=(
                    "Windows 탐색기에서 데이터 폴더를 선택합니다."
                    if _native_dialogs_available()
                    else "원격/비-Windows 서버에서는 경로를 직접 입력하세요."
                ),
            )
    else:
        github_repo   = st.text_input("저장소 (owner/repo)",    placeholder="username/repo")
        github_folder = st.text_input("서브폴더",               value="", placeholder="data/2024")
        github_token  = st.text_input("GitHub 토큰 (비공개용)", type="password")

    st.markdown("---")

    # ── Processing mode ───────────────────────────────────────
    st.markdown("### 🎯 처리 모드")
    run_mode = st.radio(
        "처리 모드",
        ["🔍 단일 파일 선택", "📦 전체 배치 처리"],
        horizontal=False,
        label_visibility="collapsed",
        key="run_mode_radio",
    )

    _run_single_file = None
    _selected_ceres_source = ""
    _selected_ceres_entry = None

    if run_mode == "🔍 단일 파일 선택":
        if data_src == "로컬 폴더" and local_folder:
            if st.button("📂 폴더 스캔", use_container_width=True, key="run_scan_btn"):
                _sp = Path(local_folder)
                if _sp.is_dir():
                    _scan_local_hsi_files.clear()
                    st.session_state["run_scan_files"] = list(
                        _scan_local_hsi_files(local_folder)
                    )
                    if not st.session_state["run_scan_files"]:
                        st.warning("지원 형식 파일을 찾지 못했습니다.")
                else:
                    st.warning("유효한 폴더 경로를 입력하세요.")
                    st.session_state["run_scan_files"] = []

            if st.session_state["run_scan_files"]:
                _run_single_file = st.selectbox(
                    "처리할 파일",
                    st.session_state["run_scan_files"],
                    format_func=lambda p: Path(p).name,
                    key="run_file_select",
                )
                st.caption(f"📄 {Path(_run_single_file).name}")

                if Path(_run_single_file).suffix.lower() == ".ceres":
                    from src import ceres_reader as _ceres

                    _selected_ceres_source = str(Path(_run_single_file).resolve())
                    if st.session_state.get("ceres_index_source") != _selected_ceres_source:
                        st.session_state["ceres_index"] = None
                        st.session_state["ceres_index_source"] = _selected_ceres_source
                        st.session_state["ceres_preview"] = None
                        st.session_state["ceres_preview_meta"] = None
                        st.session_state["ceres_prepared"] = None
                        st.session_state.pop("ceres_entry_key", None)

                    st.info(
                        "CERES는 먼저 내부 촬영 구간을 읽고, 선택한 항목만 미리보기/"
                        "분석용 BIL로 준비합니다. 전체 컨테이너를 RAM에 올리지 않습니다."
                    )
                    if st.button(
                        "🧭 CERES 내부 목록 읽기",
                        use_container_width=True,
                        key="ceres_scan_btn",
                    ):
                        try:
                            _cache_root = Path(
                                st.session_state.get("output_dir_path", "./output")
                            ).expanduser() / "_ceres_index"
                            with st.spinner("CERES 레코드 헤더를 읽는 중..."):
                                _index, _index_path, _reused = _ceres.load_or_build_index(
                                    _selected_ceres_source, _cache_root
                                )
                            st.session_state["ceres_index"] = _index
                            _entry_keys = [
                                item["key"] for item in _index.get("entries", [])
                            ]
                            if _entry_keys:
                                st.session_state["ceres_entry_key"] = _entry_keys[0]
                            st.success(
                                f"{len(_index['entries'])}개 항목 확인 · "
                                + ("캐시 재사용" if _reused else "새 인덱스 저장")
                            )
                        except Exception:
                            st.error("CERES 목록을 읽지 못했습니다.")
                            st.code(traceback.format_exc(), language="python")

                    _ceres_index = st.session_state.get("ceres_index")
                    if _ceres_index:
                        _ceres_entries = list(_ceres_index.get("entries") or [])
                        if not _ceres_entries:
                            st.warning(
                                "VNIR/SWIR 영상 항목을 찾지 못했습니다. "
                                "저장된 빈 인덱스는 다음 스캔에서 재사용하지 않습니다."
                            )
                            st.session_state["ceres_index"] = None
                            _ceres_index = None

                    if _ceres_index:
                        with st.expander("내부 항목 전체 보기", expanded=False):
                            st.dataframe(
                                pd.DataFrame(_ceres.index_table(_ceres_entries)),
                                hide_index=True,
                                use_container_width=True,
                            )
                        _entry_keys = [item["key"] for item in _ceres_entries]
                        if st.session_state.get("ceres_entry_key") not in _entry_keys:
                            st.session_state["ceres_entry_key"] = _entry_keys[0]
                        _entry_key = st.selectbox(
                            "열어볼 촬영 구간 / 센서",
                            _entry_keys,
                            key="ceres_entry_key",
                        )
                        _selected_ceres_entry = _ceres.entry_by_key(
                            _ceres_index, _entry_key
                        )
                        st.caption(
                            f"{_selected_ceres_entry['lines']:,} lines × "
                            f"{_selected_ceres_entry['samples']:,} samples × "
                            f"{_selected_ceres_entry['bands']:,} bands · "
                            f"선택 BIL {_selected_ceres_entry['bil_gib']:.2f} GiB"
                        )
                        _memory_rows = []
                        for _factor in (1, 2, 4, 8):
                            _estimate = _ceres.estimate_pipeline_memory_gib(
                                _selected_ceres_entry, _factor
                            )
                            _memory_rows.append({
                                "다운샘플": f"×{_factor}",
                                "float32 큐브": f"{_estimate['float32_cube_gib']:.2f} GiB",
                                "예상 피크 RAM": f"{_estimate['estimated_peak_gib']:.2f} GiB",
                            })
                        with st.expander("메모리 예상", expanded=False):
                            st.dataframe(
                                pd.DataFrame(_memory_rows),
                                hide_index=True,
                                use_container_width=True,
                            )
                            st.caption(
                                "현재 PCA/K-means 파이프라인의 보수적 추정치입니다. "
                                "원본 큐브 외에 반사율·float64·PCA 임시 배열을 포함합니다."
                            )

                        _cp1, _cp2 = st.columns(2)
                        with _cp1:
                            if st.button(
                                "👁️ 빠른 미리보기",
                                use_container_width=True,
                                key="ceres_preview_btn",
                            ):
                                try:
                                    with st.spinner("RGB 3개 밴드만 직접 읽는 중..."):
                                        _preview, _preview_meta = _ceres.read_rgb_preview(
                                            _selected_ceres_source,
                                            _selected_ceres_entry,
                                        )
                                    st.session_state["ceres_preview"] = _preview
                                    st.session_state["ceres_preview_meta"] = {
                                        **_preview_meta,
                                        "source": _selected_ceres_source,
                                        "entry_key": _entry_key,
                                    }
                                except Exception:
                                    st.error("CERES 미리보기를 만들지 못했습니다.")
                                    st.code(traceback.format_exc(), language="python")
                        with _cp2:
                            if st.button(
                                "📦 선택 항목 분석 준비",
                                use_container_width=True,
                                key="ceres_prepare_btn",
                                help="선택한 센서/구간만 uint16 BIL로 캐시합니다.",
                            ):
                                try:
                                    _demux_root = Path(
                                        st.session_state.get("output_dir_path", "./output")
                                    ).expanduser() / "_ceres_cache"
                                    with st.spinner(
                                        f"선택 항목 {_selected_ceres_entry['bil_gib']:.2f} GiB를 "
                                        "분석용 BIL로 준비 중..."
                                    ):
                                        _prepared = _ceres.export_entry_to_bil(
                                            _selected_ceres_source,
                                            _selected_ceres_entry,
                                            _demux_root,
                                        )
                                    st.session_state["ceres_prepared"] = {
                                        **_prepared,
                                        "source": _selected_ceres_source,
                                        "entry_key": _entry_key,
                                    }
                                    st.success("선택 항목만 준비되었습니다.")
                                except Exception:
                                    st.error("분석용 BIL 준비에 실패했습니다.")
                                    st.code(traceback.format_exc(), language="python")

                        _prepared = st.session_state.get("ceres_prepared") or {}
                        if (
                            _prepared.get("source") == _selected_ceres_source
                            and _prepared.get("entry_key") == _entry_key
                            and Path(_prepared.get("hdr_path", "")).is_file()
                        ):
                            _run_single_file = _prepared["hdr_path"]
                            st.success(
                                "✅ 현재 선택 항목이 일반 BIL/HDR 분석 입력으로 연결됨"
                            )
                        else:
                            _run_single_file = None
                            st.caption(
                                "분석 시작 전 ‘선택 항목 분석 준비’를 한 번 누르세요."
                            )
            else:
                st.caption("📂 스캔하여 파일을 선택하세요.")
        else:
            st.caption("로컬 폴더 모드에서 사용 가능합니다.")
    else:
        if st.session_state["run_scan_files"]:
            st.session_state["run_scan_files"] = []
        st.caption("📋 모든 파일을 순차 처리하고 파일별로 리포트를 생성합니다.")

    st.markdown("---")

    # ── Classification method ────────────────────────────────
    st.markdown("### 🧬 분류 방법")

    method = st.selectbox(
        "방법",
        list(METHODS.keys()),
        format_func=lambda k: METHODS[k]["label"],
        index=list(METHODS.keys()).index("kmeans"),
        label_visibility="collapsed",
    )

    kind          = METHODS[method]["kind"]
    needs_labels  = kind == "지도"
    needs_pytorch = method in ("autoencoder", "cnn")

    st.caption(
        f"{KIND_COLOR.get(kind, '')} {kind} "
        + ("| 🔥 PyTorch 필요" if needs_pytorch else "")
    )
    if method == "hybrid":
        st.caption(
            "클러스터링 입력: 보정파일이 있으면 반사율, 없으면 전역 배율 DN. "
            "Hybrid의 NDVI·밝기 임계값을 유지하기 위한 자동 선택입니다."
        )
    else:
        st.caption(
            "클러스터링 입력: 원본 DN의 스펙트럼 구조(기본). 같은 클러스터 마스크로 "
            "보정 전 DN과 보정 후 반사율 스펙트럼을 모두 저장합니다."
        )

    st.markdown("---")

    # ── Number of classes ────────────────────────────────────
    st.markdown("### 🔢 클래스 수")

    if method == "supervised":
        st.caption("라벨 CSV의 클래스 수를 자동으로 사용합니다.")
        n_classes = 0
    elif method == "hdbscan":
        st.caption(
            "HDBSCAN은 클러스터 수를 **자동으로** 결정합니다. "
            "슬라이더는 이 방법에서 무시됩니다."
        )
        n_classes = 0
    else:
        n_classes = st.slider(
            "클러스터(클래스) 수",
            min_value=2, max_value=20, value=6,
            label_visibility="collapsed",
        )

    st.markdown("---")

    # ── Method-specific params ───────────────────────────────
    st.markdown("### 🔧 세부 파라미터")

    ndvi_threshold       = 0.15
    brightness_threshold = 0.08
    if method == "hybrid":
        with st.expander("Hybrid 설정", expanded=True):
            ndvi_threshold = st.slider(
                "NDVI 임계값 (식생 기준)", 0.0, 1.0, 0.15, 0.01,
                help="이 값 이상의 NDVI 픽셀 = 식생으로 분류",
            )
            brightness_threshold = st.slider(
                "밝기 임계값 (그림자 기준)", 0.0, 0.5, 0.08, 0.01,
                help="평균 반사율이 이 값 미만 = 그림자로 분류",
            )

    angle_threshold = 0.10
    if method == "sam":
        with st.expander("SAM 설정", expanded=True):
            angle_threshold = st.slider(
                "각도 임계값 (radians, 0=제한없음)", 0.0, 0.5, 0.10, 0.01,
                help=(
                    "최근접 endmember와의 각도가 이 값보다 크면 "
                    "Background(0)으로 처리됩니다.\n"
                    f"현재값 ≈ {round(angle_threshold * 57.3, 1)}°"
                ),
            )

    ae_epochs  = 60
    cnn_epochs = 100
    if method == "autoencoder":
        with st.expander("Autoencoder 설정", expanded=False):
            ae_epochs = st.slider("학습 epochs", 10, 200, 60, 10)
    if method == "cnn":
        with st.expander("CNN 설정", expanded=False):
            cnn_epochs = st.slider("학습 epochs", 10, 200, 100, 10)

    hdbscan_min_cluster_size = 50
    hdbscan_min_samples      = 5
    if method == "hdbscan":
        with st.expander("HDBSCAN 설정", expanded=True):
            hdbscan_min_cluster_size = st.slider(
                "min_cluster_size", 10, 500, 50, 10,
                help=(
                    "클러스터 형성에 필요한 최소 픽셀 수. "
                    "값이 클수록 더 크고 적은 클러스터가 생성됩니다."
                ),
            )
            hdbscan_min_samples = st.slider(
                "min_samples", 1, 50, 5, 1,
                help=(
                    "클러스터링 보수성 제어. "
                    "값이 클수록 노이즈 픽셀(class 0)이 늘어납니다."
                ),
            )

    labels_csv = ""
    if needs_labels or method == "sam":
        st.markdown("---")
        lbl_header = "라벨 CSV" if needs_labels else "라벨 CSV (선택 – SAM 지도 모드)"
        st.markdown(f"### 📋 {lbl_header}")
        labels_csv = st.text_input(
            "경로 (row, col, class_id)",
            placeholder="labels.csv",
            label_visibility="collapsed",
        )
        if needs_labels and not labels_csv:
            st.warning("⚠️ 이 방법은 라벨 CSV가 필요합니다.")

    st.markdown("---")

    # ── Reflectance / normalization ─────────────────────────
    st.markdown("### 📐 반사율 처리")
    _NORM_MODES = {
        "global":   "전역 배율 (스펙트럼 형태 보존)",
        "per_band": "밴드별 스트레치 (대비 강조)",
        "none":     "정규화 안 함 (DN/보정 반사도 유지)",
    }
    _active_calibration_path = st.session_state.get("active_calibration_path", "")
    if _active_calibration_path:
        st.success(
            "자동 반사율 보정 사용 중\n\n"
            f"`{Path(_active_calibration_path).name}`"
        )
    else:
        st.caption(
            "활성 보정이 없어도 분석 시작 시 원본 폴더와 "
            "`output/calibration`에서 같은 영상 이름의 보정파일을 찾아 "
            "밴드 호환성을 확인한 뒤 자동 적용합니다."
        )

    normalize_mode = "none" if _active_calibration_path else "global"
    _manual_calibration_path = ""
    with st.expander("⚙️ 고급: 기존 보정파일·정규화 직접 설정", expanded=False):
        normalize_mode = st.selectbox(
            "정규화 방식",
            list(_NORM_MODES.keys()),
            format_func=lambda k: _NORM_MODES[k],
            index=(2 if _active_calibration_path else 0),
            help=(
                "밴드별 스트레치는 스펙트럼 형태를 바꾸므로 화면 대비 확인에만 "
                "사용하세요. 반사율 보정이 있으면 정규화 안 함이 자동 적용됩니다."
            ),
            key="advanced_normalize_mode",
        )
        _manual_calibration_path = st.text_input(
            "기존 보정 .npz 또는 White/Dark 프로파일 폴더",
            value="",
            placeholder="./calibration_profiles",
            help=(
                "일반 사용자는 입력할 필요가 없습니다. 저장해 둔 보정 결과를 "
                "직접 재사용할 때만 지정하세요."
            ),
            key="advanced_calibration_path",
        )
        if normalize_mode == "per_band":
            st.warning("밴드별 스트레치는 논문용 스펙트럼 형태를 왜곡합니다.")

    analysis_calibration_path = (
        _manual_calibration_path.strip() or _active_calibration_path
    )
    if _active_calibration_path and not _manual_calibration_path.strip():
        st.success(
            "✅ 패널 보정 탭에서 만든 보정파일을 자동 사용합니다: "
            f"{Path(_active_calibration_path).name}"
        )
    if analysis_calibration_path:
        if normalize_mode != "none":
            normalize_mode = "none"
            st.info(
                "반사율 보정이 연결되어 추가 정규화를 자동으로 끕니다."
            )

    st.markdown("---")

    # ── Large-file handling ──────────────────────────────────
    st.markdown("### ⚡ 대용량 파일")
    spatial_downsample = st.select_slider(
        "공간 다운샘플링 (1 = 원본 해상도)",
        options=[1, 2, 4, 8],
        value=1,
        help=(
            "N이면 N×N 픽셀 블록당 1개 픽셀만 읽어 메모리를 1/N²로 줄입니다. "
            "수 GB 이상 파일은 4 권장. 스펙트럼 형태는 유지되며 "
            "분류 지도 해상도만 낮아집니다."
        ),
    )

    st.markdown("---")

    # ── Selectable report builder ────────────────────────────
    st.markdown("### 📋 결과 리포트")
    _REPORT_PRESET_LABELS = {
        "quick_qc": "⚡ 빠른 필드 QC (추천)",
        "research_standard": "🔬 연구용 표준 리포트",
        "custom": "🛠️ 사용자 지정",
    }
    report_preset = st.selectbox(
        "리포트 구성",
        list(_REPORT_PRESET_LABELS),
        format_func=lambda key: _REPORT_PRESET_LABELS[key],
        key="report_preset",
        help="계산하고 HTML/PNG로 남길 결과 항목을 선택합니다.",
    )
    _report_defaults = REPORT_PRESETS[report_preset]
    report_sections = dict(_report_defaults["sections"])
    report_statistics = list(_report_defaults["spectra_statistics"])
    report_indices = list(_report_defaults["indices"])
    save_selected_images = bool(_report_defaults["save_selected_images"])
    save_daily_summary = bool(_report_defaults["daily_summary"])
    save_html_report = True
    save_spectra_csv = True
    _section_labels = {
        "rgb": "RGB 이미지",
        "false_color": "CIR 위색도",
        "spectral_indices": "선택 식생지수 이미지·요약",
        "class_map": "클러스터 맵",
        "cluster_overlay": "RGB+클러스터 오버레이",
        "per_class_images": "클러스터별 분리 이미지",
        "class_summary": "클러스터별 픽셀 통계",
        "spectral_plot": "클러스터별 스펙트럼",
        "quality_metrics": "클러스터 품질·분리도",
        "vegetation_quality": "식생 분리도 평가",
        "calibration_qc": "보정파일·유효밴드 QC",
    }

    with st.expander(
        "리포트 항목 확인·선택",
        expanded=report_preset == "custom",
    ):
        if report_preset != "custom":
            _enabled_sections = [
                _section_labels.get(key, key)
                for key, enabled in report_sections.items() if enabled
            ]
            st.caption(
                "- 포함 항목: " + ", ".join(_enabled_sections)
                + "\n- 스펙트럼 통계: " + ", ".join(report_statistics)
                + "\n- 식생지수: " + (", ".join(report_indices) or "없음")
            )
        else:
            _sc1, _sc2 = st.columns(2)
            for _section_index, (_section_key, _section_label) in enumerate(
                _section_labels.items()
            ):
                _target_column = _sc1 if _section_index % 2 == 0 else _sc2
                with _target_column:
                    report_sections[_section_key] = st.checkbox(
                        _section_label,
                        value=bool(report_sections.get(_section_key)),
                        key=f"report_section_{_section_key}",
                    )

            report_statistics = st.multiselect(
                "스펙트럼 통계",
                ["mean", "median", "std", "iqr"],
                default=report_statistics,
                key="report_statistics",
                help="std는 평균±표준편차, iqr은 25–75% 범위입니다.",
            ) or ["mean"]
            report_indices = st.multiselect(
                "식생지수",
                ["NDVI", "GNDVI", "NDRE", "PRI"],
                default=report_indices,
                key="report_indices",
                help="보정 반사율과 필요한 파장 밴드가 있을 때만 계산합니다.",
            )
            save_html_report = st.checkbox(
                "인터랙티브 HTML 리포트 저장", value=True, key="report_save_html"
            )
            save_spectra_csv = st.checkbox(
                "스펙트럼 CSV 저장", value=True, key="report_save_csv"
            )
            save_selected_images = st.checkbox(
                "선택한 이미지들을 PNG로 별도 저장",
                value=True,
                key="report_save_images",
            )
            save_daily_summary = st.checkbox(
                "배치 처리 시 하루 전체 요약 HTML·CSV 저장",
                value=True,
                key="report_daily_summary",
            )

        if report_indices:
            st.info(
                "NDVI/GNDVI/NDRE/PRI는 보정 반사율에서만 계산합니다. "
                "보정이 없거나 필요한 파장이 없으면 리포트에 계산 불가 이유를 기록합니다."
            )

    model_spectra_enabled = True
    model_spectra_per_class = 1_000
    model_spectra_save_raw = True
    with st.expander("🧬 모델 학습용 실제 스펙트럼", expanded=False):
        model_spectra_enabled = st.checkbox(
            "클러스터별 실제 픽셀 스펙트럼 저장 (.h5)",
            value=True,
            key="model_spectra_enabled",
            help=(
                "평균이나 median이 아니라 클러스터를 구성한 실제 픽셀 스펙트럼을 "
                "저장합니다. 각 영상/플랏 안의 픽셀은 서로 독립적인 플랏 라벨이 아닙니다."
            ),
        )
        model_spectra_per_class = st.number_input(
            "클러스터별 최대 스펙트럼 수",
            min_value=10,
            max_value=10_000,
            value=1_000,
            step=100,
            key="model_spectra_per_class",
            disabled=not model_spectra_enabled,
            help=(
                "각 최종 클러스터에서 고정 난수 시드로 비복원 표본추출합니다. "
                "Hybrid 분석은 sunlit/shadow/soil 기본 구분도 파일에 함께 기록합니다."
            ),
        )
        model_spectra_save_raw = st.checkbox(
            "보정/분석값과 raw DN을 함께 저장",
            value=True,
            key="model_spectra_save_raw",
            disabled=not model_spectra_enabled,
        )
        st.caption(
            "결과 폴더의 spectral_samples.h5에는 파장축, 픽셀 좌표, 최종 "
            "클러스터, Hybrid 기본 클래스, 표본가중치와 보정 이력이 포함됩니다."
        )

    with st.expander(
        "👥 팀·플랏 일일 통합 리포트",
        expanded=run_mode == "📦 전체 배치 처리",
    ):
        team_daily_enabled = st.checkbox(
            "배치 완료 후 팀용 일일 패키지 생성",
            value=True,
            disabled=run_mode != "📦 전체 배치 처리",
            key="team_daily_enabled",
            help=(
                "파일별 결과를 다시 읽어 하나의 HTML·Excel·NDVI 비교 이미지로 묶습니다. "
                "원본 초분광 큐브를 다시 RAM에 올리지 않습니다."
            ),
        )
        _default_team_name = (
            Path(local_folder).name
            if data_src == "로컬 폴더" and local_folder not in {"", ".", "./data"}
            else "Field Team"
        )
        team_name = st.text_input(
            "팀 이름",
            value=_default_team_name,
            key="team_daily_name",
            disabled=not team_daily_enabled,
        )
        measurement_date = st.date_input(
            "실제 측정일",
            value=datetime.date.today(),
            key="team_daily_date",
            disabled=not team_daily_enabled,
            help="분석 실행일이 아니라 현장에서 영상을 획득한 날짜를 선택하세요.",
        )
        _tm1, _tm2 = st.columns([4, 1])
        with _tm1:
            plot_metadata_csv = st.text_input(
                "플랏 메타데이터 CSV (선택)",
                value="",
                placeholder="filename, plot_id, treatment, genotype, replicate",
                key="team_daily_metadata_csv",
                disabled=not team_daily_enabled,
            )
        with _tm2:
            st.write("")
            st.button(
                "🪟 선택",
                use_container_width=True,
                key="browse_team_metadata_csv",
                on_click=_browse_file_into_state,
                args=("team_daily_metadata_csv", "플랏 메타데이터 CSV 선택"),
                disabled=not team_daily_enabled or not _native_dialogs_available(),
            )
        st.caption(
            "CSV가 없으면 파일명이 플랏 ID가 됩니다. CSV 열 예: "
            "`filename, plot_id, treatment, genotype, replicate, team, measurement_date`."
        )

    team_daily_enabled = bool(
        team_daily_enabled and run_mode == "📦 전체 배치 처리"
    )
    if team_daily_enabled:
        report_sections["spectral_indices"] = True
        if "NDVI" not in report_indices:
            report_indices.append("NDVI")

    st.markdown("---")

    # ── Output / misc ────────────────────────────────────────
    st.markdown("### 📁 출력")
    _of1, _of2 = st.columns([4, 1])
    with _of1:
        output_dir = st.text_input(
            "출력 폴더", value="./output", key="output_dir_path"
        )
    with _of2:
        st.write("")
        st.button(
            "🪟 선택",
            use_container_width=True,
            key="browse_output_folder",
            on_click=_browse_directory_into_state,
            args=("output_dir_path", "분석 결과 저장 폴더 선택"),
            disabled=not _native_dialogs_available(),
            help=(
                "Windows 탐색기에서 결과 저장 폴더를 선택합니다."
                if _native_dialogs_available()
                else "원격/비-Windows 서버에서는 경로를 직접 입력하세요."
            ),
        )
    file_limit = st.number_input(
        "파일 수 제한 (0 = 전체)", min_value=0, value=0, step=1,
        help="테스트 시 1~2로 제한하면 빠르게 확인할 수 있습니다.",
    )
    verbose = st.checkbox("상세 로그 (DEBUG)", value=False)

    st.markdown("---")
    _sidebar_job = st.session_state.get("analysis_job")
    _sidebar_job_state = _poll_analysis_job(_sidebar_job)
    _sidebar_job_running = _sidebar_job_state.get("state") in _ACTIVE_JOB_STATES
    run_btn = st.button(
        "🚀  분석 시작",
        type="primary",
        use_container_width=True,
        disabled=_sidebar_job_running,
    )
    if _sidebar_job_running:
        st.caption("분석이 별도 프로세스에서 실행 중입니다. 중지 버튼은 즉시 작동합니다.")
        if st.button(
            "⏹️ 실행 중지",
            use_container_width=True,
            key="stop_analysis_sidebar",
        ):
            _cancel_analysis_job(_sidebar_job)
            st.rerun()


# Runtime estimate for the currently configured local job.  GitHub jobs do not
# expose reliable payload sizes until downloaded, so they are estimated only
# after one completed run.
st.session_state.setdefault("run_timing_history", [])
st.session_state.setdefault("run_last_timing", None)
st.session_state.setdefault("run_last_reports", [])
st.session_state.setdefault("run_last_output_dir", "")
st.session_state.setdefault("run_last_review_dirs", [])
_current_analysis_job = st.session_state.get("analysis_job")
_current_analysis_state = _poll_analysis_job(_current_analysis_job)
if (
    _current_analysis_job
    and _current_analysis_state.get("state") == "completed"
    and not _current_analysis_job.get("result_applied")
):
    _apply_completed_analysis_job(_current_analysis_job)
_planned_run_files: list[str] = []
if data_src == "로컬 폴더":
    if run_mode == "🔍 단일 파일 선택" and _run_single_file:
        _planned_run_files = [str(_run_single_file)]
    elif run_mode == "📦 전체 배치 처리" and local_folder:
        _planned_run_files = [
            path for path in _scan_local_hsi_files(local_folder)
            if Path(path).suffix.lower() != ".ceres"
        ]
        if file_limit:
            _planned_run_files = _planned_run_files[: int(file_limit)]

_run_work_units = 0.0
_run_source_bytes = 0
_run_estimated_seconds: float | None = None
if _planned_run_files:
    _run_work_units, _run_source_bytes = _file_work_units(
        _planned_run_files, int(spatial_downsample)
    )
    _run_estimated_seconds = _estimate_seconds(
        _run_work_units, method, st.session_state["run_timing_history"]
    )


# ============================================================
# Main area
# ============================================================

st.markdown("# 🌿 CanopySpectra")
st.caption("From CERES to Science-Ready Field Spectra")

tab_run, tab_roi, tab_panel, tab_label = st.tabs(
    ["🚀 분석 실행", "📈 ROI 스펙트럼", "🎯 패널 보정", "🏷️ 픽셀 라벨링"]
)

# ============================================================
# Tab 1 – Run pipeline
# ============================================================

with tab_run:
    _ceres_preview_meta = st.session_state.get("ceres_preview_meta") or {}
    if (
        st.session_state.get("ceres_preview") is not None
        and _ceres_preview_meta.get("source") == _selected_ceres_source
    ):
        st.markdown("### 👁️ CERES 선택 항목 미리보기")
        st.image(
            st.session_state["ceres_preview"],
            caption=(
                f"{Path(_selected_ceres_source).name} · "
                f"{_ceres_preview_meta.get('entry_key')} · "
                f"{_ceres_preview_meta.get('preview_mode', '미리보기')} · "
                f"원본 {tuple(_ceres_preview_meta.get('source_shape', []))} → "
                f"미리보기 {tuple(_ceres_preview_meta.get('preview_shape', []))}"
            ),
            use_container_width=True,
        )
        st.caption(
            "VNIR은 가시광 RGB, SWIR은 1650/1250/1050 nm 가색을 사용합니다. "
            "3개 밴드만 읽으므로 수 MB 수준이며, CERES 전체나 "
            "전체 초분광 큐브를 RAM에 올리지 않습니다."
        )
        st.markdown("---")

    # ── Info cards ─────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        src_info   = f"`{local_folder}`" if data_src == "로컬 폴더" else f"`{github_repo}`"
        limit_info = str(int(file_limit)) + " 파일" if file_limit else "전체"
        if method == "hdbscan":
            cls_info = "자동 결정 (HDBSCAN)"
        else:
            cls_info = f"{n_classes} 클래스" if n_classes else "라벨 CSV 기준"
        st.info(
            f"**방법:** {METHODS[method]['label']}  \n"
            f"**데이터:** {src_info}  \n"
            f"**클래스:** {cls_info}  \n"
            f"**파일:** {limit_info}  ·  **출력:** `{output_dir}`"
        )

    with col_right:
        st.success(METHODS[method]["help"])

    if _run_estimated_seconds is not None:
        _source_gib = _run_source_bytes / (1024**3)
        st.info(
            f"⏱ **예상 분석시간: {_format_estimate(_run_estimated_seconds)}**  ·  "
            f"{len(_planned_run_files)}개 파일  ·  원본 {_source_gib:.2f} GB  ·  "
            f"다운샘플 ×{spatial_downsample}"
        )
        st.caption(
            "예상시간은 파일 크기·분석 방법·다운샘플링과 이 세션의 이전 실행 기록을 "
            "사용합니다. 디스크 속도와 메모리 상태에 따라 달라질 수 있습니다."
        )
    elif data_src == "GitHub 저장소":
        st.caption("⏱ GitHub 파일은 다운로드가 끝난 첫 실행부터 예상시간을 보정할 수 있습니다.")
    else:
        st.caption("⏱ 폴더를 스캔하고 처리할 파일을 선택하면 예상시간이 표시됩니다.")

    _last_run_timing = st.session_state.get("run_last_timing")
    if _last_run_timing:
        st.caption(
            f"최근 실제 소요시간: **{_format_duration(_last_run_timing['elapsed_seconds'])}**"
            f" · {_last_run_timing.get('file_count', 0)}개 파일"
            f" · {_last_run_timing.get('method', '').upper()}"
        )

    st.markdown("---")

    # ── Run ────────────────────────────────────────────────────
    _render_analysis_job_status()

    if run_btn:

        # Validate inputs
        errors = []
        if data_src == "로컬 폴더" and not local_folder:
            errors.append("로컬 폴더 경로를 입력해 주세요.")
        if data_src == "GitHub 저장소" and not github_repo:
            errors.append("GitHub 저장소를 입력해 주세요.")
        if needs_labels and not labels_csv:
            errors.append(f"{method} 방법은 라벨 CSV가 필요합니다.")
        if analysis_calibration_path and not Path(analysis_calibration_path).exists():
            errors.append("반사도 보정 .npz 또는 프로파일 폴더를 찾을 수 없습니다.")
        if run_mode == "🔍 단일 파일 선택" and not _run_single_file:
            errors.append("단일 파일 모드: 폴더를 스캔하고 파일을 선택해 주세요.")
        if team_daily_enabled and not team_name.strip():
            errors.append("팀·플랏 일일 리포트의 팀 이름을 입력해 주세요.")
        if (
            team_daily_enabled
            and plot_metadata_csv.strip()
            and not Path(plot_metadata_csv.strip()).expanduser().is_file()
        ):
            errors.append("플랏 메타데이터 CSV 파일을 찾을 수 없습니다.")

        if errors:
            for e in errors:
                st.error(f"❌ {e}")
            st.stop()

        # Build config dict
        base = max(1, n_classes // 3) if n_classes else 1
        r    = n_classes % 3           if n_classes else 0

        cfg: dict = {
            "data": {
                "local_folder": local_folder or None,
                "github": {
                    "repo":   github_repo   or None,
                    "folder": github_folder or "",
                    "token":  github_token  or None,
                },
                "supported_formats": [".hdr", ".tif", ".tiff", ".h5", ".hdf5", ".mat"],
                "cache_dir": "./cache",
            },
            "preprocessing": {
                "calibration_file": analysis_calibration_path or None,
                "auto_discover_calibration": True,
                "calibration_search_roots": [output_dir],
                "normalize":          normalize_mode != "none",
                "normalize_mode":     normalize_mode,
                "remove_bad_bands":   True,
                "bad_band_ranges":    [[1340, 1460], [1790, 1960]],
                "smooth_spectra":     False,
                "spatial_downsample": int(spatial_downsample),
            },
            "classification": {
                "method": method,
                "input_space": "auto",
                "classes": [],
                "kmeans": {
                    "n_clusters":     n_classes or 6,
                    "pca_components": 15,
                    "n_init":         10,
                    "max_iter":       300,
                    "random_state":   42,
                },
                "hybrid": {
                    "ndvi_threshold":       ndvi_threshold,
                    "brightness_threshold": brightness_threshold,
                    "kmeans_refinement":    True,
                    "n_clusters_sunlit":    base + (1 if r >= 1 else 0),
                    "n_clusters_shadow":    base + (1 if r >= 2 else 0),
                    "n_clusters_soil":      base,
                    "pca_components":       10,
                },
                "sam": {
                    "angle_threshold": angle_threshold,
                    "n_endmembers":    n_classes or 6,
                    "endmember_pca":   15,
                },
                "autoencoder": {
                    "latent_dim":    16,
                    "n_clusters":    n_classes or 6,
                    "epochs":        ae_epochs,
                    "batch_size":    1024,
                    "learning_rate": 0.001,
                    "max_pixels":    100_000,
                },
                "cnn": {
                    "epochs":        cnn_epochs,
                    "batch_size":    512,
                    "learning_rate": 0.001,
                    "test_split":    0.2,
                    "patience":      15,
                },
                "hdbscan": {
                    "min_cluster_size": hdbscan_min_cluster_size,
                    "min_samples":      hdbscan_min_samples,
                    "pca_components":   15,
                },
                "gmm": {
                    "n_components":    n_classes or 6,
                    "covariance_type": "full",
                    "max_iter":        100,
                    "pca_components":  15,
                    "random_state":    42,
                },
                "nmf": {
                    "n_components": n_classes or 6,
                    "max_iter":     500,
                    "random_state": 42,
                },
            },
            "extraction": {
                "n_neighbors": 100,
                "sample_export": {
                    "enabled": bool(model_spectra_enabled),
                    "max_per_class": int(model_spectra_per_class),
                    "random_state": 42,
                    "save_raw": bool(model_spectra_save_raw),
                },
            },
            "output": {
                "dir":                     output_dir,
                "save_classification_map": bool(report_sections.get("class_map")),
                "save_spectra_csv":        save_spectra_csv,
                "save_report":             save_html_report,
                "per_file_report":         run_mode == "📦 전체 배치 처리",
            },
            "report": {
                "title":            "CanopySpectra — Field Hyperspectral Analysis Report",
                "preset":           report_preset,
                "sections":         report_sections,
                "spectra_statistics": report_statistics,
                "indices":          report_indices,
                "save_selected_images": save_selected_images,
                "daily_summary":    save_daily_summary,
                "spectra_show_std": "std" in report_statistics,
                "lang":             "ko",
                "team_daily": {
                    "enabled":          team_daily_enabled,
                    "team_name":        team_name.strip() or "Field Team",
                    "measurement_date": measurement_date.isoformat(),
                    "metadata_csv":     plot_metadata_csv.strip(),
                },
            },
        }

        # The pipeline runs in a child process.  This is essential for a real
        # Stop button: Streamlit's page process remains free to receive clicks,
        # poll the status file, and update the log while computation continues.
        try:
            _job = _launch_analysis_job(
                cfg,
                labels_csv=labels_csv if labels_csv else None,
                file_limit=int(file_limit) if file_limit else None,
                single_file=_run_single_file,
                timing={
                    "method": method,
                    "work_units": _run_work_units,
                    "file_count": len(_planned_run_files),
                    "estimated_seconds": _run_estimated_seconds,
                    "verbose": verbose,
                },
                project_root=Path(__file__).parent,
            )
            st.session_state["analysis_job"] = _job
            st.rerun()
        except Exception:
            st.error("❌ 분석 작업을 시작하지 못했습니다.")
            st.code(traceback.format_exc(), language="python")

    # Persisted result access: clicking a button reruns Streamlit, so this must
    # live outside the one-shot `if run_btn` block.
    _last_report_paths = [
        Path(path) for path in st.session_state.get("run_last_reports", [])
        if Path(path).is_file()
    ]
    _last_output_path_text = st.session_state.get("run_last_output_dir", "")
    _last_output_path = (
        Path(_last_output_path_text)
        if _last_output_path_text else None
    )
    _review_dirs = [
        Path(path) for path in st.session_state.get("run_last_review_dirs", [])
        if Path(path).is_dir()
    ]
    if not _review_dirs:
        _review_dirs = list(dict.fromkeys(
            path.parent for path in _last_report_paths if path.parent.is_dir()
        ))
    if _review_dirs:
        st.markdown("### 🔍 클러스터링 결과 이미지 검수")
        st.caption(
            "색이 잎·그림자·밝은 반사·토양 경계와 맞는지 확인하세요. 원하는 "
            "클러스터만 선택하고 투명도를 조절할 수 있습니다."
        )
        _review_choice = st.selectbox(
            "검수할 분석 결과",
            options=[str(path) for path in _review_dirs],
            format_func=lambda value: Path(value).name,
            key="run_cluster_review_choice",
        )
        _render_cluster_review(
            Path(_review_choice),
            "run_cluster_review_" + re.sub(r"[^a-zA-Z0-9]+", "_", Path(_review_choice).name),
        )

    if _last_report_paths or (_last_output_path and _last_output_path.is_dir()):
        st.markdown("### 📄 최근 분석 결과 열기")
        _selected_report = None
        if len(_last_report_paths) > 1:
            _selected_report_text = st.selectbox(
                "열어볼 HTML 리포트",
                options=[str(path) for path in _last_report_paths],
                format_func=lambda value: (
                    f"{Path(value).parent.name} / {Path(value).name}"
                ),
                key="run_last_report_choice",
            )
            _selected_report = Path(_selected_report_text)
        elif _last_report_paths:
            _selected_report = _last_report_paths[0]
            st.caption(f"HTML 리포트: `{_selected_report}`")

        _open_col, _folder_col, _download_col = st.columns(3)
        with _open_col:
            if st.button(
                "🌐 선택한 HTML 리포트 열기",
                use_container_width=True,
                disabled=_selected_report is None,
                key="run_open_report",
            ):
                try:
                    _open_local_path(_selected_report)
                    st.success("기본 웹브라우저에서 리포트를 열었습니다.")
                except Exception as exc:
                    st.error(f"HTML 리포트를 열지 못했습니다: {exc}")
        with _folder_col:
            if st.button(
                "📂 결과 폴더 열기",
                use_container_width=True,
                disabled=not (_last_output_path and _last_output_path.is_dir()),
                key="run_open_output_folder",
            ):
                try:
                    _open_local_path(_last_output_path)
                    st.success("파일 탐색기에서 결과 폴더를 열었습니다.")
                except Exception as exc:
                    st.error(f"결과 폴더를 열지 못했습니다: {exc}")
        with _download_col:
            if _selected_report is not None:
                st.download_button(
                    "⬇️ HTML 리포트 다운로드",
                    data=_selected_report.read_bytes(),
                    file_name=_selected_report.name,
                    mime="text/html",
                    use_container_width=True,
                    key="run_download_report",
                )
            else:
                st.button(
                    "⬇️ HTML 리포트 다운로드",
                    disabled=True,
                    use_container_width=True,
                    key="run_download_report_disabled",
                )
        if not _last_report_paths:
            st.caption(
                "이번 실행에서는 HTML 리포트 저장 옵션이 꺼져 있거나 "
                "리포트가 생성되지 않았습니다. 결과 폴더는 열 수 있습니다."
            )

    _team_packages = [
        package for package in st.session_state.get("run_last_team_packages", [])
        if package.get("directory") and Path(package["directory"]).is_dir()
    ]
    if _team_packages:
        st.markdown("### 👥 팀·플랏 일일 통합 결과")
        _team_index = 0
        if len(_team_packages) > 1:
            _team_labels = [
                f"{item.get('measurement_date', '')} · {item.get('team', '')}"
                for item in _team_packages
            ]
            _team_label = st.selectbox(
                "열어볼 팀/측정일",
                _team_labels,
                key="run_team_package_choice",
            )
            _team_index = _team_labels.index(_team_label)
        _team_package = _team_packages[_team_index]
        _team_dir = Path(_team_package["directory"])
        _team_report = Path(_team_package.get("report", ""))
        _team_workbook = Path(_team_package.get("workbook", ""))
        _team_summary_csv = Path(_team_package.get("summary_csv", ""))

        _tp1, _tp2, _tp3 = st.columns(3)
        with _tp1:
            if st.button(
                "🌐 팀 일일 HTML 열기",
                use_container_width=True,
                key="open_team_daily_report",
            ):
                _open_local_path(_team_report)
        with _tp2:
            if st.button(
                "📂 팀 결과 폴더 열기",
                use_container_width=True,
                key="open_team_daily_folder",
            ):
                _open_local_path(_team_dir)
        with _tp3:
            _download_path = (
                _team_workbook if _team_workbook.is_file() else _team_summary_csv
            )
            if _download_path.is_file():
                st.download_button(
                    "⬇️ 팀 결과 Excel/CSV",
                    data=_download_path.read_bytes(),
                    file_name=_download_path.name,
                    mime=(
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        if _download_path.suffix.lower() == ".xlsx"
                        else "text/csv"
                    ),
                    use_container_width=True,
                    key="download_team_daily_results",
                )
        if _team_package.get("workbook_warning"):
            st.warning(_team_package["workbook_warning"])
        _team_visuals = [
            (_team_dir / "plots_ndvi.png", "모든 플랏 NDVI · 공통 범위 -1~1"),
            (_team_dir / "plot_ndvi_comparison.png", "플랏별 NDVI 중앙값·IQR · QC PASS만"),
        ]
        _team_visuals = [item for item in _team_visuals if item[0].is_file()]
        if _team_visuals:
            _visual_columns = st.columns(len(_team_visuals))
            for _column, (_image, _caption) in zip(_visual_columns, _team_visuals):
                _column.image(str(_image), caption=_caption, use_container_width=True)

    with st.expander("📊 Excel에서 여는 결과 CSV 설명"):
        st.markdown(
            "저장 파일은 현재 **`.xlsx` 통합문서가 아니라 Excel에서 바로 열 수 있는 "
            "UTF-8 CSV**입니다. 파일명에 붙은 접미사로 값의 단위를 구분하세요."
        )
        st.markdown(
            """
| 파일 | 용도 |
|---|---|
| `spectra_{method}_reflectance.csv` | 보정이 통과했을 때 생성되는 **과학 분석용 반사율** 스펙트럼 |
| `spectra_{method}_raw_dn.csv` | 센서 원본 DN. 보정 문제 진단과 전후 비교용 |
| `spectra_{method}_processed.csv` | 보정파일이 없을 때 정규화/전처리된 상대값. 절대 반사율이 아님 |
| `spectra_{method}.csv` | 해당 실행에서 실제로 사용·추출한 값의 대표 파일. 단위는 `value_units` 열로 확인 |
| `daily_summary_*.csv` | 하루 배치의 파일별 클래스 수, NDVI, 식생 비율, 품질지표, 처리시간 요약 |
| `all_roi_cluster_spectra*.csv` | 모든 ROI·클러스터 스펙트럼을 한 파일로 합친 결과 |
| `cluster_summary.csv` | ROI별 클러스터 픽셀 수와 면적 비율(`fraction`, 0–1) 요약 |
"""
        )
        st.markdown(
            "일반 `spectra_*` CSV는 **한 행이 한 파장**인 wide 형식이며 각 클러스터마다 "
            "`mean`, `std`, `median`, `q25`, `q75`, `mna`, `sam_avg` 열이 생깁니다. "
            "ROI의 `cluster_spectra*`는 **ROI × 클러스터 × 파장별 한 행**인 long 형식이며 "
            "`mean`, `median`, `std`, `q25`, `q75`를 저장합니다. `mna`는 값 기준, "
            "`sam_avg`는 스펙트럼 모양 기준의 대표 픽셀 평균입니다."
        )
        st.info(
            "논문용 값은 `_reflectance.csv`에서 `value_units=reflectance`, "
            "`calibration_applied=True`, `calibration_qc_status=PASS`를 우선 확인하세요. "
            "`REVIEW`는 점프·포화 등 경고를 검토한 뒤 사용하고, `FAIL`은 사용하지 않는 것이 안전합니다."
        )


# ============================================================
# Tab 2 – ROI spectrum viewer
# ============================================================

with tab_roi:
    from src import roi_utils

    st.markdown("### 📈 ROI 스펙트럼 추출")
    st.caption(
        "이미지에서 박스·올가미·클릭 Polygon으로 영역을 지정하면 그 영역 픽셀들의 "
        "평균·중간값 스펙트럼을 그래프로 보고 CSV로 저장할 수 있습니다."
    )

    _roi_defaults: dict = {
        "roi2_data":   None,   # ndarray (H, W, B)
        "roi2_wl":     None,
        "roi2_rgb":    None,
        "roi2_meta":   None,
        "roi2_file":   "",
        "roi2_region": None,
        "roi2_units":  "",
        "roi2_cal":      None,   # {"a", "b", "method", "panels", "reflectances"}
        "roi2_ref_path": "",
        "roi2_refl_txt": "0.99, 0.50, 0.25",
        "roi2_h5_path":  "",
        "roi2_cal_open": False,
        "roi2_cal_file": "",
        "roi2_cal_error": "",
        "roi2_cal_search": {},
        "roi2_show_reflectance_rgb": False,
        "roi2_zoom_region": None,
        "roi2_zoom_revision": 0,
    }
    for k, v in _roi_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Step 1: Load ──────────────────────────────────────────
    st.markdown("#### 1️⃣ 파일 로드")

    _roi_candidates: list[str] = []
    if data_src == "로컬 폴더" and local_folder:
        _roi_candidates = list(_scan_local_hsi_files(local_folder))
    if _run_single_file:
        _roi_candidates.insert(0, str(_run_single_file))
    if st.session_state.get("roi2_file") and Path(st.session_state["roi2_file"]).is_file():
        _roi_candidates.insert(0, st.session_state["roi2_file"])
    _roi_candidates = list(dict.fromkeys(_roi_candidates))
    _manual_choice = "__manual_path__"
    _roi_options = _roi_candidates + [_manual_choice]
    _preferred_roi_file = (
        str(_run_single_file)
        if _run_single_file in _roi_candidates
        else st.session_state.get("roi2_file", "")
    )
    _roi_default_index = (
        _roi_options.index(_preferred_roi_file)
        if _preferred_roi_file in _roi_options else 0
    )

    rc1, rc2, rc3, rc4 = st.columns([4, 1.4, 1, 1])
    with rc1:
        _roi_file_choice = st.selectbox(
            "초분광 파일 선택",
            options=_roi_options,
            index=_roi_default_index,
            format_func=lambda value: (
                "직접 경로 입력…" if value == _manual_choice else Path(value).name
            ),
            label_visibility="collapsed",
            key="roi2_file_choice",
        )
        if _roi_file_choice == _manual_choice:
            roi_file_input = st.text_input(
                "직접 파일 경로",
                value=st.session_state.get("roi2_manual_path", ""),
                placeholder="Z:/data/image.hdr",
                key="roi2_manual_path",
            )
        else:
            roi_file_input = _roi_file_choice
    with rc2:
        roi_ds = st.selectbox(
            "다운샘플링",
            [1, 2, 4, 8],
            index=0,
            format_func=lambda v: f"다운샘플 ×{v}",
            label_visibility="collapsed",
            key="roi2_ds",
            help="수 GB 이상 파일은 4 이상을 권장합니다. 스펙트럼 형태는 유지됩니다.",
        )
    with rc3:
        roi_load_btn = st.button("📂 로드", use_container_width=True, key="roi2_load")
    with rc4:
        st.button(
            "🪟 선택",
            use_container_width=True,
            key="roi2_native_file",
            on_click=_browse_file_into_state,
            args=("roi2_manual_path", "초분광 파일 선택"),
            kwargs={"set_values": {"roi2_file_choice": _manual_choice}},
            disabled=not _native_dialogs_available(),
            help="대용량 파일을 업로드하지 않고 Windows 경로만 선택합니다.",
        )

    if roi_file_input:
        st.caption(f"선택 파일: `{roi_file_input}`")
    if data_src == "로컬 폴더" and local_folder:
        if st.button("🔄 현재 폴더 파일 목록 새로고침", key="roi2_refresh_files"):
            _scan_local_hsi_files.clear()
            st.session_state["run_scan_files"] = list(
                _scan_local_hsi_files(local_folder)
            )
            st.rerun()
    elif not _roi_candidates:
        st.warning(
            "ROI 화면의 대용량 파일은 웹 업로드하지 않습니다. 왼쪽 데이터 소스를 "
            "로컬 폴더로 바꾸거나 ‘직접 경로 입력’을 사용하세요."
        )

    if st.session_state.get("active_calibration_path"):
        st.caption("✅ 패널 보정 탭에서 만든 반사율 보정을 자동 적용합니다.")
    else:
        st.caption(
            "ℹ️ 저장된 보정파일은 아래 `반사율 변환` 영역에서 불러올 수 있습니다. "
            "파일 로드 후 표시되는 보정 상태 배너를 확인하세요."
        )

    if roi_load_btn and roi_file_input:
        try:
            with st.spinner("초분광 파일 로딩 중... (대용량 파일은 수 분 걸릴 수 있습니다)"):
                from src.data_loader import HyperspectralLoader

                _rl = HyperspectralLoader({"spatial_downsample": int(roi_ds)})
                _rdata, _rmeta = _rl.load_local(roi_file_input)

                st.session_state["roi2_data"]   = _rdata
                st.session_state["roi2_wl"]     = _rmeta.get("wavelengths")
                st.session_state["roi2_meta"]   = _rmeta
                st.session_state["roi2_rgb"]    = roi_utils.display_rgb(
                    _rdata, _rmeta.get("wavelengths")
                )
                st.session_state["roi2_file"]   = roi_file_input
                st.session_state["roi2_region"] = None
                st.session_state["roi2_polygon_points"] = []
                st.session_state["roi2_polygon_last_click"] = None
                st.session_state["roi2_units"]  = "raw DN"
                st.session_state["roi2_cal"] = None
                st.session_state["roi2_cal_error"] = ""
                st.session_state["roi2_cal_search"] = {}
                st.session_state["roi2_show_reflectance_rgb"] = False
                st.session_state["roi2_spectrum_view"] = "원본 DN"
                st.session_state["roi2_zoom_region"] = None
                st.session_state["roi2_zoom_revision"] = (
                    int(st.session_state.get("roi2_zoom_revision", 0)) + 1
                )

                _active_roi_cal = st.session_state.get("active_calibration_path", "")
                from src import radiometry as _roi_rad

                _calibration_candidates: list[tuple[str, str]] = []
                if _active_roi_cal:
                    _calibration_candidates.append(
                        (str(_active_roi_cal), "패널 보정 탭에서 현재 활성화됨")
                    )
                for _nearby_cal in _roi_rad.discover_calibration_candidates(
                    roi_file_input,
                    search_roots=[output_dir or "./output"],
                ):
                    _nearby_path = str(_nearby_cal)
                    if all(
                        Path(_candidate[0]).resolve() != Path(_nearby_path).resolve()
                        for _candidate in _calibration_candidates
                    ):
                        _calibration_candidates.append(
                            (_nearby_path, "이미지 로드 시 자동 탐지")
                        )

                _rejected_calibrations: list[dict] = []
                _selected_calibration = None
                for _candidate_path, _candidate_origin in _calibration_candidates:
                    try:
                        _resolved_roi_cal = _roi_rad.resolve_calibration(
                            _candidate_path,
                            target_source=roi_file_input,
                            wavelengths=_rmeta.get("wavelengths"),
                        )
                        _candidate_a = np.asarray(_resolved_roi_cal["a"])
                        _candidate_b = np.asarray(_resolved_roi_cal["b"])
                        if (
                            _candidate_a.shape != (_rdata.shape[2],)
                            or _candidate_b.shape != (_rdata.shape[2],)
                        ):
                            raise ValueError(
                                f"보정계수 {_candidate_a.shape}와 영상 밴드 "
                                f"{_rdata.shape[2]}가 일치하지 않습니다."
                            )
                        if not (np.isfinite(_candidate_a) & np.isfinite(_candidate_b)).any():
                            raise ValueError("유효한 보정 밴드가 하나도 없습니다.")

                        _resolved_meta = dict(_resolved_roi_cal.get("meta") or {})
                        _resolved_meta["selection_source"] = _candidate_origin
                        st.session_state["roi2_cal"] = {
                            "a": _candidate_a,
                            "b": _candidate_b,
                            "method": _resolved_meta.get(
                                "method", "자동 패널 보정"
                            ),
                            "panels": [],
                            "reflectances": [],
                            "selected_profile": _resolved_roi_cal.get(
                                "selected_profile", _candidate_path
                            ),
                            "meta": _resolved_meta,
                        }
                        st.session_state["roi2_units"] = "reflectance"
                        st.session_state["roi2_show_reflectance_rgb"] = True
                        st.session_state["roi2_spectrum_view"] = "보정 반사율"
                        st.session_state["roi2_cal_file"] = _candidate_path
                        _selected_calibration = {
                            "path": _candidate_path,
                            "profile": st.session_state["roi2_cal"]["selected_profile"],
                            "origin": _candidate_origin,
                        }
                        break
                    except Exception as _candidate_error:
                        _rejected_calibrations.append(
                            {
                                "path": _candidate_path,
                                "error": str(_candidate_error),
                            }
                        )

                if _selected_calibration is not None:
                    st.session_state["roi2_cal_error"] = ""
                    st.session_state["roi2_cal_search"] = {
                        "status": "applied",
                        "selected": _selected_calibration,
                        "candidate_count": len(_calibration_candidates),
                        "rejected": _rejected_calibrations,
                    }
                elif _calibration_candidates:
                    _first_rejection = _rejected_calibrations[0]
                    st.session_state["roi2_cal_error"] = (
                        "보정파일을 찾았지만 현재 영상과 호환되지 않습니다: "
                        f"{Path(_first_rejection['path']).name} · "
                        f"{_first_rejection['error']}"
                    )
                    st.session_state["roi2_cal_search"] = {
                        "status": "incompatible",
                        "candidate_count": len(_calibration_candidates),
                        "rejected": _rejected_calibrations,
                    }
                else:
                    st.session_state["roi2_cal_search"] = {
                        "status": "not_found",
                        "candidate_count": 0,
                        "rejected": [],
                    }
            st.success(
                f"✅ 로드 완료  |  {_rdata.shape[0]} × {_rdata.shape[1]} px  "
                f"|  {_rdata.shape[2]} 밴드  |  다운샘플 ×{roi_ds}"
            )
        except Exception:
            st.error("❌ 파일 로드 실패")
            st.code(traceback.format_exc(), language="python")

    _rdata = st.session_state.get("roi2_data")
    _rrgb  = st.session_state.get("roi2_rgb")
    _rwl   = st.session_state.get("roi2_wl")

    if _rdata is None:
        st.info("📂 위 목록에서 파일을 선택하고 [로드]를 누르세요.")
    else:
        _rmeta = st.session_state.get("roi2_meta") or {}
        _H, _W, _B = _rdata.shape

        _full = _rmeta.get("full_shape")
        _ds   = _rmeta.get("downsample_applied", 1)
        _info = f"`{_rmeta.get('filename', '')}`  |  {_H} × {_W} px  |  {_B} 밴드"
        if _full and _ds > 1:
            _info += f"  |  원본 {_full[0]} × {_full[1]} px (다운샘플 ×{_ds})"
        if _rwl:
            _info += f"  |  {_rwl[0]:.1f}–{_rwl[-1]:.1f} nm"
        st.success(_info)

        _cal_search = st.session_state.get("roi2_cal_search") or {}
        if _cal_search.get("status") == "applied":
            _auto_selected = _cal_search.get("selected") or {}
            st.info(
                "🔎 이미지 로드 시 보정파일 확인 완료 · "
                f"{_auto_selected.get('origin', '자동 선택')} · "
                f"`{Path(str(_auto_selected.get('profile', ''))).name}`"
            )
        elif _cal_search.get("status") == "not_found":
            st.caption(
                "🔎 이미지 폴더와 출력 폴더에서 `calibration.npz` 및 "
                "현재 영상명 기반 보정파일을 확인했지만 발견되지 않았습니다."
            )
        elif _cal_search.get("status") == "incompatible":
            st.error(
                "🔎 보정파일 후보를 찾았지만 밴드 수 또는 파장축이 현재 영상과 "
                "일치하지 않아 자동 적용하지 않았습니다."
            )

        # ── Radiometric calibration (optional) ────────────────
        # Stays open once the user engages with it, so the detected panels and
        # any error stay visible across the rerun a button press triggers.
        with st.expander(
            "🎯 반사율 변환 — 보정 패널 reference 스캔 사용 (선택)",
            expanded=bool(st.session_state.get("roi2_cal_open")),
        ):
            st.caption(
                "raw DN은 반사율이 아닙니다. `DN = 반사율 × 조명 × 센서감도 + dark` 이므로 "
                "조명과 센서 감도 곡선이 스펙트럼 형태를 왜곡합니다. 패널이 찍힌 reference "
                "스캔을 지정하면 empirical line으로 이를 제거해 실제 반사율로 변환합니다."
            )

            _cal1, _cal2 = st.columns([3, 2])
            with _cal1:
                _ref_path = st.text_input(
                    "Reference 스캔 경로 (패널이 찍힌 파일)",
                    value=st.session_state.get("roi2_ref_path", ""),
                    placeholder="Z:/.../reference.vnir.hdr",
                    key="roi2_ref_path_input",
                )
            with _cal2:
                _refl_txt = st.text_input(
                    "패널 반사율 (밝은 것부터, 쉼표 구분)",
                    value=st.session_state.get("roi2_refl_txt", "0.99, 0.50, 0.25"),
                    key="roi2_refl_txt_input",
                )

            _h5_path = st.text_input(
                "HySpex .h5 경로 (선택 — 패널 탐지 정확도 향상)",
                value=st.session_state.get("roi2_h5_path", ""),
                placeholder="Z:/.../scene.hyspex.h5",
                key="roi2_h5_path_input",
                help="센서 QE 곡선을 읽어 패널의 분광 평탄도를 더 정확히 판정합니다.",
            )

            st.caption(
                "⚠️ reference 스캔은 대상 장면과 **같은 조명 조건**(같은 시각·태양각)에서 "
                "촬영된 것이어야 합니다. 시간대가 다르면 보정에 오차가 생기며 "
                "프로그램이 이를 감지할 수 없습니다."
            )

            # Reuse a calibration built in the 패널 보정 tab
            _cf1, _cf2 = st.columns([4, 1])
            with _cf1:
                _cal_file = st.text_input(
                    "보정 .npz 또는 White/Dark 프로파일 폴더",
                    value=st.session_state.get("roi2_cal_file", ""),
                    placeholder="./calibration_profiles",
                    key="roi2_cal_file_input",
                )
            with _cf2:
                st.write("")
                if st.button("📥 불러오기", use_container_width=True,
                             key="roi2_cal_load"):
                    st.session_state["roi2_cal_open"] = True
                    try:
                        from src import radiometry as _radm2
                        _radm2 = importlib.reload(_radm2)

                        _cd = _radm2.resolve_calibration(
                            _cal_file,
                            target_source=st.session_state.get("roi2_file"),
                            wavelengths=_rwl,
                        )
                        if len(_cd["a"]) != _rdata.shape[2]:
                            raise ValueError(
                                f"캘리브레이션 밴드 수({len(_cd['a'])})가 "
                                f"현재 큐브({_rdata.shape[2]})와 다릅니다."
                            )
                        st.session_state["roi2_cal"] = {
                            "a": _cd["a"], "b": _cd["b"],
                            "method": _cd["meta"].get("method", "저장된 캘리브레이션"),
                            "panels": [], "reflectances": [],
                            "selected_profile": _cd.get("selected_profile", ""),
                            "meta": _cd.get("meta", {}),
                        }
                        st.session_state["roi2_units"] = "reflectance"
                        st.session_state["roi2_cal_file"] = _cal_file
                        st.session_state["roi2_cal_error"] = ""
                        st.session_state["roi2_cal_search"] = {
                            "status": "applied",
                            "selected": {
                                "path": _cal_file,
                                "profile": _cd.get("selected_profile", _cal_file),
                                "origin": "사용자가 직접 불러옴",
                            },
                            "candidate_count": 1,
                            "rejected": [],
                        }
                        st.session_state["roi2_show_reflectance_rgb"] = True
                        st.success("✅ 캘리브레이션 적용됨")
                    except Exception as _cal_load_error:
                        st.session_state["roi2_cal_error"] = str(_cal_load_error)
                        st.error("❌ 캘리브레이션 불러오기 실패")
                        st.code(traceback.format_exc(), language="python")

            _cb1, _cb2 = st.columns(2)
            with _cb1:
                _cal_btn = st.button(
                    "🎯 패널 탐지 후 반사율 변환 적용",
                    type="primary", use_container_width=True, key="roi2_cal_btn",
                )
            with _cb2:
                if st.button("↩️ 원본 DN으로 되돌리기",
                             use_container_width=True, key="roi2_cal_reset"):
                    st.session_state["roi2_cal"] = None
                    st.session_state["roi2_units"] = "raw DN"
                    st.session_state["roi2_cal_error"] = ""
                    st.session_state["roi2_cal_search"] = {
                        "status": "manual_reset",
                        "candidate_count": 0,
                        "rejected": [],
                    }
                    st.session_state["roi2_show_reflectance_rgb"] = False
                    st.rerun()

            if _cal_btn:
                st.session_state["roi2_cal_open"] = True
                try:
                    with st.spinner("Reference 스캔 로딩 및 패널 탐지 중..."):
                        from src import radiometry as _radm
                        from src.data_loader import HyperspectralLoader as _HL

                        _prs = [float(v) for v in _refl_txt.replace(",", " ").split()]
                        if len(_prs) < 1:
                            raise ValueError("패널 반사율을 최소 1개 입력하세요.")
                        if any(not (0 < v <= 1.5) for v in _prs):
                            raise ValueError(
                                "패널 반사율은 0~1 사이 값이어야 합니다 (예: 0.99)."
                            )

                        _refd, _refm = _HL({"spatial_downsample": 1}).load_local(_ref_path)
                        if _refd.shape[2] != _rdata.shape[2]:
                            raise ValueError(
                                f"밴드 수 불일치: reference {_refd.shape[2]} vs "
                                f"대상 {_rdata.shape[2]}"
                            )

                        _qe = None
                        if _h5_path.strip():
                            _qe = _radm.load_hyspex_qe(_h5_path.strip())
                            if len(_qe) != _rdata.shape[2]:
                                st.warning(
                                    f"QE 밴드 수({len(_qe)})가 큐브({_rdata.shape[2]})와 "
                                    f"달라 무시합니다."
                                )
                                _qe = None

                        _panels = _radm.detect_panels(
                            _refd, n_panels=len(_prs), qe=_qe
                        )
                        if len(_panels) < len(_prs):
                            raise ValueError(
                                f"패널을 {len(_panels)}개만 찾았습니다 "
                                f"(반사율 {len(_prs)}개 입력). reference 파일을 "
                                f"확인하거나 반사율 개수를 맞춰주세요."
                            )

                        _dns = [p["spectrum"] for p in _panels]
                        if len(_dns) >= 2:
                            _a, _b = _radm.empirical_line_coeffs(_dns, _prs)
                            _method = "empirical line"
                        else:
                            _a = np.asarray(_prs[0]) / np.where(
                                np.abs(_dns[0]) < 1e-9, np.nan, _dns[0]
                            )
                            _b = np.zeros_like(_a)
                            _method = "flat field (패널 1장 — dark 미보정)"

                        st.session_state["roi2_cal"] = {
                            "a": np.asarray(_a), "b": np.asarray(_b),
                            "method": _method,
                            "panels": [
                                {k: p[k] for k in
                                 ("box", "n_pixels", "brightness", "flatness")}
                                for p in _panels
                            ],
                            "reflectances": _prs,
                        }
                        st.session_state["roi2_units"] = "reflectance"
                        st.session_state["roi2_cal_error"] = ""
                        st.session_state["roi2_cal_search"] = {
                            "status": "applied",
                            "selected": {
                                "path": _ref_path,
                                "profile": _ref_path,
                                "origin": "현재 세션의 Reference 패널 자동 탐지",
                            },
                            "candidate_count": 1,
                            "rejected": [],
                        }
                        st.session_state["roi2_show_reflectance_rgb"] = True
                        st.session_state["roi2_ref_path"] = _ref_path
                        st.session_state["roi2_refl_txt"] = _refl_txt
                        st.session_state["roi2_h5_path"] = _h5_path
                    st.success(f"✅ 반사율 변환 적용 ({_method})")
                except Exception as _panel_cal_error:
                    st.session_state["roi2_cal_error"] = str(_panel_cal_error)
                    st.error("❌ 반사율 변환 실패")
                    st.code(traceback.format_exc(), language="python")

            _cal = st.session_state.get("roi2_cal")
            if _cal:
                st.markdown(f"**적용 중:** {_cal['method']}")
                if _cal.get("selected_profile"):
                    st.caption(f"선택된 프로파일: `{_cal['selected_profile']}`")
                _delta = (_cal.get("meta") or {}).get("white_time_delta_seconds")
                if _delta is not None:
                    st.caption(f"대상 촬영시각과 White 간격: {float(_delta) / 60:.1f}분")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "패널": f"#{i}",
                            "반사율": _cal["reflectances"][i - 1],
                            "box [r0,r1,c0,c1]": str(p["box"]),
                            "픽셀수": f"{p['n_pixels']:,}",
                            "밝기": round(p["brightness"], 1),
                            "평탄도": round(p["flatness"], 3),
                        }
                        for i, p in enumerate(_cal["panels"], 1)
                    ]),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "평탄도가 낮을수록 분광적으로 균일한 패널입니다. "
                    "박스 위치가 실제 패널과 다르면 결과를 신뢰하지 마세요."
                )

        _cal = st.session_state.get("roi2_cal")
        if _cal is not None:
            _cal_meta = _cal.get("meta") or {}
            _cal_profile = _cal.get("selected_profile") or "현재 세션의 Reference 패널 탐지"
            _cal_profile_name = (
                Path(_cal_profile).name
                if _cal.get("selected_profile") else _cal_profile
            )
            _cal_status = (
                "✅ **반사율 보정 적용됨** — 아래 ROI 스펙트럼과 보정 CSV는 "
                "단위 없는 반사율(Reflectance)입니다.\n\n"
                f"방법: `{_cal.get('method', '저장된 캘리브레이션')}` · "
                f"프로파일: `{_cal_profile_name}`"
            )
            _white_delta = _cal_meta.get("white_time_delta_seconds")
            if _white_delta is not None:
                _cal_status += f" · 대상 영상과 White 간격: `{float(_white_delta) / 60:.1f}분`"
            st.success(_cal_status)

            _dark_source_type = _cal_meta.get("dark_source_type")
            if _dark_source_type == "measured_file":
                st.caption(f"Dark: 실측 파일 · `{Path(str(_cal_meta.get('dark_source', ''))).name}`")
            elif _dark_source_type == "synthetic_constant":
                st.warning(
                    "⚠️ 실측 Dark가 아니라 합성 상수 Dark를 사용한 보정입니다: "
                    f"DN {_cal_meta.get('manual_dark_dn', 100)}"
                )
        else:
            _cal_error = st.session_state.get("roi2_cal_error", "")
            if _cal_error:
                st.error(
                    "❌ **반사율 보정 적용 실패** — 아래 스펙트럼은 원본 DN입니다.\n\n"
                    + str(_cal_error)
                )
            else:
                st.warning(
                    "⚠️ **반사율 보정 미적용** — 아래 스펙트럼과 CSV는 원본 DN입니다. "
                    "논문용 반사율로 사용하려면 `패널 보정` 탭에서 만든 .npz를 적용하세요."
                )

        st.markdown("#### 2️⃣ ROI 지정 및 스펙트럼 확인")
        _view_control_1, _view_control_2, _view_control_3 = st.columns([2, 3, 3])
        with _view_control_1:
            _roi_wide_layout = st.toggle(
                "🖼️ ROI 이미지를 전체 폭으로 보기",
                value=True,
                key="roi2_wide_layout",
                help="켜면 이미지가 위에 크게 나오고 스펙트럼은 아래에 표시됩니다.",
            )
        with _view_control_2:
            _roi_view_height = st.slider(
                "이미지 화면 높이",
                min_value=420,
                max_value=1400,
                value=720,
                step=40,
                key="roi2_view_height",
            )
        with _view_control_3:
            if _cal is None:
                st.session_state["roi2_show_reflectance_rgb"] = False
            _show_reflectance_rgb = st.toggle(
                "🌈 보정 반사율 RGB로 보기",
                key="roi2_show_reflectance_rgb",
                disabled=_cal is None,
                help=(
                    "켜면 RGB에 해당하는 세 밴드만 반사율로 변환하여 공통 범위로 표시합니다. "
                    "스펙트럼 계산 결과에는 영향을 주지 않습니다."
                ),
            )

        _reflectance_rgb_max = 0.6
        if _show_reflectance_rgb and _cal is not None:
            _reflectance_rgb_max = st.slider(
                "반사율 RGB 밝기 범위 (0부터 최대값)",
                min_value=0.10,
                max_value=1.50,
                value=0.60,
                step=0.05,
                key="roi2_reflectance_rgb_max",
                help=(
                    "RGB 화면에만 적용되는 공통 반사율 표시 범위입니다. "
                    "스펙트럼 값과 저장되는 CSV는 바뀌지 않습니다."
                ),
            )

        # Line-scan field images can be tens of thousands of rows long. Showing
        # the whole scan at once compresses every plot into a thin strip, so an
        # elongated image opens as a manageable row/column window instead.
        _view_r0, _view_r1, _view_c0, _view_c1 = 0, _H, 0, _W
        if _H > max(1800, 3 * _W):
            _default_rows = min(_H, max(1000, 2 * _W))
            _row_range_key = f"roi2_row_range_{_H}_{_W}"
            _view_r0, _view_r1 = st.slider(
                "세로로 긴 영상 — 화면에 표시할 행(row) 구간",
                min_value=0,
                max_value=_H,
                value=(0, _default_rows),
                step=1,
                key=_row_range_key,
                help="스펙트럼 계산 좌표는 자동으로 원본 영상 좌표로 변환됩니다.",
            )
            if _view_r1 <= _view_r0:
                _view_r1 = min(_H, _view_r0 + 1)
            st.caption(
                f"현재 원본 행 {_view_r0:,}–{_view_r1:,} 표시 중 · "
                "슬라이더를 옮겨 다른 구간에서 ROI를 선택하세요."
            )
        elif _W > max(1800, 3 * _H):
            _default_cols = min(_W, max(1000, 2 * _H))
            _col_range_key = f"roi2_col_range_{_H}_{_W}"
            _view_c0, _view_c1 = st.slider(
                "가로로 긴 영상 — 화면에 표시할 열(column) 구간",
                min_value=0,
                max_value=_W,
                value=(0, _default_cols),
                step=1,
                key=_col_range_key,
                help="스펙트럼 계산 좌표는 자동으로 원본 영상 좌표로 변환됩니다.",
            )
            if _view_c1 <= _view_c0:
                _view_c1 = min(_W, _view_c0 + 1)
            st.caption(
                f"현재 원본 열 {_view_c0:,}–{_view_c1:,} 표시 중 · "
                "슬라이더를 옮겨 다른 구간에서 ROI를 선택하세요."
            )

        _base_view = (_view_r0, _view_r1, _view_c0, _view_c1)
        _zoom_region = st.session_state.get("roi2_zoom_region")
        _zoom_active = False
        if _zoom_region is not None:
            _zr0, _zr1, _zc0, _zc1 = roi_utils.box_region(
                _zoom_region, _H, _W
            )["roi"]
            _zoomed_view = (
                max(_view_r0, _zr0),
                min(_view_r1, _zr1),
                max(_view_c0, _zc0),
                min(_view_c1, _zc1),
            )
            if (
                _zoomed_view[1] > _zoomed_view[0]
                and _zoomed_view[3] > _zoomed_view[2]
                and _zoomed_view != _base_view
            ):
                _view_r0, _view_r1, _view_c0, _view_c1 = _zoomed_view
                _zoom_active = True
            else:
                st.session_state["roi2_zoom_region"] = None

        if _zoom_active:
            _base_pixels = max(
                1,
                (_base_view[1] - _base_view[0])
                * (_base_view[3] - _base_view[2]),
            )
            _zoom_pixels = max(
                1,
                (_view_r1 - _view_r0) * (_view_c1 - _view_c0),
            )
            _zoom_factor = np.sqrt(_base_pixels / _zoom_pixels)
            st.info(
                "🔍 확대 화면 유지 중 · "
                f"row `{_view_r0}:{_view_r1}`, col `{_view_c0}:{_view_c1}` · "
                f"약 `{_zoom_factor:.1f}×` 확대"
            )

        _raw_roi_view_rgb = _rrgb[_view_r0:_view_r1, _view_c0:_view_c1]
        _roi_view_rgb = _raw_roi_view_rgb
        _rgb_display_label = "원본 DN 기반 RGB · 채널별 화면 스트레치"
        if _show_reflectance_rgb and _cal is not None:
            try:
                _roi_view_rgb = roi_utils.display_reflectance_rgb(
                    _rdata[_view_r0:_view_r1, _view_c0:_view_c1, :],
                    _rwl,
                    _cal["a"],
                    _cal["b"],
                    reflectance_max=float(_reflectance_rgb_max),
                )
                _rgb_display_label = (
                    "보정 반사율 RGB · 세 채널 공통 범위 "
                    f"0–{float(_reflectance_rgb_max):.2f}"
                )
            except Exception as _rgb_cal_error:
                st.warning(
                    "보정 반사율 RGB를 만들지 못해 원본 DN RGB로 표시합니다: "
                    f"{_rgb_cal_error}"
                )
        _view_H, _view_W = _roi_view_rgb.shape[:2]
        if _roi_wide_layout:
            roi_left = st.container()
            roi_right = st.container()
        else:
            roi_left, roi_right = st.columns([3, 2])

        with roi_left:
            st.caption(f"현재 표시 영상: **{_rgb_display_label}**")
            _mouse1, _mouse2 = st.columns([3, 1])
            with _mouse1:
                _roi_mouse_mode = st.radio(
                    "마우스 조작 모드",
                    (
                        "⬚ Box ROI",
                        "✏️ Lasso ROI",
                        "🔺 Polygon 클릭 ROI",
                        "🔍 확대",
                    ),
                    horizontal=True,
                    key="roi2_mouse_mode",
                    help=(
                        "확대 모드에서 보고 싶은 영역을 사각형으로 드래그하면 그 좌표를 "
                        "확대 화면으로 저장합니다. ROI 모드로 바꿔도 확대가 유지됩니다."
                    ),
                )
            with _mouse2:
                st.write("")
                if st.button(
                    "↩️ 전체 화면",
                    use_container_width=True,
                    key="roi2_zoom_reset",
                    disabled=not _zoom_active,
                ):
                    st.session_state["roi2_zoom_region"] = None
                    st.session_state["roi2_zoom_revision"] = (
                        int(st.session_state.get("roi2_zoom_revision", 0)) + 1
                    )
                    st.rerun()
            st.caption(
                "① `🔍 확대` 선택 → ② 크게 볼 영역을 사각형으로 드래그 → "
                "③ 확대 후 Box/Lasso/Polygon으로 바꿔 ROI 선택 · "
                "Polygon은 잎 둘레의 꼭짓점을 차례로 클릭하고 완료 버튼을 누르세요."
            )
            if _roi_mouse_mode == "🔺 Polygon 클릭 ROI":
                from PIL import Image, ImageDraw

                _poly_points = list(
                    st.session_state.get("roi2_polygon_points", [])
                )
                _pc1, _pc2, _pc3 = st.columns(3)
                if _pc1.button(
                    "↶ 마지막 점 취소",
                    key="roi2_polygon_undo",
                    use_container_width=True,
                    disabled=not _poly_points,
                ):
                    st.session_state["roi2_polygon_points"] = _poly_points[:-1]
                    st.rerun()
                if _pc2.button(
                    "🗑️ 점 모두 지우기",
                    key="roi2_polygon_clear",
                    use_container_width=True,
                    disabled=not _poly_points,
                ):
                    st.session_state["roi2_polygon_points"] = []
                    st.rerun()
                if _pc3.button(
                    "✅ Polygon 완료",
                    key="roi2_polygon_finish",
                    type="primary",
                    use_container_width=True,
                    disabled=len(_poly_points) < 3,
                ):
                    st.session_state["roi2_region"] = roi_utils.polygon_region(
                        [point[0] for point in _poly_points],
                        [point[1] for point in _poly_points],
                        _H,
                        _W,
                    )
                    st.session_state["roi2_polygon_points"] = []
                    st.rerun()

                _poly_image = Image.fromarray(_roi_view_rgb).convert("RGB")
                _poly_draw = ImageDraw.Draw(_poly_image)
                _line_width = max(2, min(_view_H, _view_W) // 180)
                _point_radius = max(3, min(_view_H, _view_W) // 100)
                _current_region = st.session_state.get("roi2_region")
                if _current_region:
                    if _current_region.get("type") in {"lasso", "polygon"}:
                        _saved_points = [
                            (float(x) - _view_c0, float(y) - _view_r0)
                            for x, y in zip(
                                _current_region.get("x", []),
                                _current_region.get("y", []),
                            )
                        ]
                        if len(_saved_points) >= 3:
                            _poly_draw.line(
                                _saved_points + [_saved_points[0]],
                                fill="#00e5ff",
                                width=_line_width,
                            )
                    else:
                        _cr0, _cr1, _cc0, _cc1 = _current_region["roi"]
                        _poly_draw.rectangle(
                            (
                                _cc0 - _view_c0,
                                _cr0 - _view_r0,
                                _cc1 - _view_c0,
                                _cr1 - _view_r0,
                            ),
                            outline="#00e5ff",
                            width=_line_width,
                        )
                _local_draft = [
                    (float(x) - _view_c0, float(y) - _view_r0)
                    for x, y in _poly_points
                ]
                if len(_local_draft) >= 2:
                    _poly_draw.line(_local_draft, fill="#ffd54f", width=_line_width)
                for _point_index, (_px, _py) in enumerate(_local_draft, 1):
                    _poly_draw.ellipse(
                        (
                            _px - _point_radius,
                            _py - _point_radius,
                            _px + _point_radius,
                            _py + _point_radius,
                        ),
                        fill="#ffd54f",
                        outline="#111111",
                    )
                    _poly_draw.text(
                        (_px + _point_radius, _py - _point_radius),
                        str(_point_index),
                        fill="#ffffff",
                    )
                _click = streamlit_image_coordinates(
                    _poly_image,
                    height=int(_roi_view_height),
                    key=(
                        "roi2_polygon_image_"
                        f"{st.session_state.get('roi2_file', '')}|"
                        f"{_view_r0}:{_view_r1}:{_view_c0}:{_view_c1}"
                    ),
                    cursor="crosshair",
                )
                if _click:
                    _click_id = _click.get("unix_time")
                    if _click_id != st.session_state.get("roi2_polygon_last_click"):
                        _display_w = max(1, int(_click.get("width", _view_W)))
                        _display_h = max(1, int(_click.get("height", _view_H)))
                        _full_x = _view_c0 + np.clip(
                            float(_click["x"]) * _view_W / _display_w,
                            0,
                            max(0, _view_W - 1),
                        )
                        _full_y = _view_r0 + np.clip(
                            float(_click["y"]) * _view_H / _display_h,
                            0,
                            max(0, _view_H - 1),
                        )
                        st.session_state["roi2_polygon_points"] = (
                            _poly_points + [(float(_full_x), float(_full_y))]
                        )
                        st.session_state["roi2_polygon_last_click"] = _click_id
                        st.rerun()
                st.caption(f"현재 꼭짓점: **{len(_poly_points)}개** · 최소 3개")
            else:
                _rfig = go.Figure()
                _rfig.add_trace(go.Image(z=_roi_view_rgb))
                _current_region = st.session_state.get("roi2_region")
                if _current_region:
                    if _current_region.get("type") in {"lasso", "polygon"}:
                        _xs = [x - _view_c0 for x in _current_region.get("x", [])]
                        _ys = [y - _view_r0 for y in _current_region.get("y", [])]
                        if len(_xs) >= 3:
                            _rfig.add_shape(
                                type="path",
                                path="M " + " L ".join(
                                    f"{x},{y}" for x, y in zip(_xs, _ys)
                                ) + " Z",
                                line=dict(color="#00e5ff", width=3, dash="dash"),
                            )
                    else:
                        _cr0, _cr1, _cc0, _cc1 = _current_region.get(
                            "roi", [0, 0, 0, 0]
                        )
                        if (
                            _cr1 > _view_r0 and _cr0 < _view_r1
                            and _cc1 > _view_c0 and _cc0 < _view_c1
                        ):
                            _rfig.add_shape(
                                type="rect",
                                x0=max(_cc0, _view_c0) - _view_c0,
                                x1=min(_cc1, _view_c1) - _view_c0,
                                y0=max(_cr0, _view_r0) - _view_r0,
                                y1=min(_cr1, _view_r1) - _view_r0,
                                line=dict(color="#00e5ff", width=3, dash="dash"),
                            )
                _rfig.update_layout(
                    dragmode=(
                        "select" if _roi_mouse_mode == "🔍 확대"
                        else "lasso" if _roi_mouse_mode == "✏️ Lasso ROI"
                        else "select"
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=int(_roi_view_height),
                    newselection=dict(line=dict(color="#ffd54f", width=3)),
                    uirevision=(
                        f"{st.session_state.get('roi2_file', '')}|"
                        f"{_view_r0}:{_view_r1}:{_view_c0}:{_view_c1}|"
                        f"{_rgb_display_label}|"
                        f"{st.session_state.get('roi2_zoom_revision', 0)}"
                    ),
                )
                _rfig.update_xaxes(showticklabels=False)
                _rfig.update_yaxes(showticklabels=False)

                _revent = st.plotly_chart(
                    _rfig,
                    key=(
                        "roi2_image_chart_"
                        f"{int(st.session_state.get('roi2_zoom_revision', 0))}_"
                        f"{_roi_mouse_mode}"
                    ),
                    on_select="rerun",
                    selection_mode=(
                        ("box",)
                        if _roi_mouse_mode == "🔍 확대" else ("box", "lasso")
                    ),
                    use_container_width=True,
                    config={**_ROI_PLOTLY_CONFIG, "scrollZoom": False},
                )

                if _revent is not None and hasattr(_revent, "selection"):
                    _view_region = roi_utils.selection_to_region(
                        _revent.selection, _view_H, _view_W
                    )
                    if _view_region is not None:
                        _full_view_region = roi_utils.offset_region(
                            _view_region, _view_r0, _view_c0, _H, _W
                        )
                        if _roi_mouse_mode == "🔍 확대":
                            _new_zoom = _full_view_region["roi"]
                            if (
                                _new_zoom != st.session_state.get("roi2_zoom_region")
                                and (_new_zoom[1] - _new_zoom[0]) >= 2
                                and (_new_zoom[3] - _new_zoom[2]) >= 2
                            ):
                                st.session_state["roi2_zoom_region"] = _new_zoom
                                st.session_state["roi2_zoom_revision"] = (
                                    int(st.session_state.get("roi2_zoom_revision", 0)) + 1
                                )
                                st.rerun()
                        else:
                            st.session_state["roi2_region"] = _full_view_region

        with roi_right:
            _region = st.session_state.get("roi2_region")
            if not _region:
                st.info("⬅️ 왼쪽 이미지에서 영역을 드래그하세요.")
            else:
                _r0, _r1, _c0, _c1 = _region["roi"]
                st.write(
                    f"선택: `{_region.get('type', 'box')}`  |  "
                    f"row `{_r0}:{_r1}`, col `{_c0}:{_c1}`"
                )

                try:
                    with st.expander("좌표 직접 입력", expanded=False):
                        with st.form("roi2_manual_form"):
                            _nr0 = st.number_input("row 시작", 0, _H - 1, int(_r0), 1)
                            _nr1 = st.number_input("row 끝",   1, _H,     int(_r1), 1)
                            _nc0 = st.number_input("col 시작", 0, _W - 1, int(_c0), 1)
                            _nc1 = st.number_input("col 끝",   1, _W,     int(_c1), 1)
                            if st.form_submit_button("이 좌표로 적용"):
                                st.session_state["roi2_region"] = roi_utils.box_region(
                                    [_nr0, _nr1, _nc0, _nc1], _H, _W
                                )
                                st.rerun()

                    _raw_stats, _npix, _bounds, _rtype = roi_utils.roi_stats(
                        _rdata, _region
                    )

                    # Empirical-line calibration is affine per band, so both
                    # views can be prepared from one ROI statistics pass.
                    _calib = st.session_state.get("roi2_cal")
                    _calibrated_stats = None
                    if _calib is not None:
                        _calibrated_stats = roi_utils.apply_calibration(
                            _raw_stats, _calib["a"], _calib["b"]
                        )

                    _view_options = (
                        ["보정 반사율", "원본 DN", "원본·보정 비교"]
                        if _calibrated_stats is not None
                        else ["원본 DN"]
                    )
                    if st.session_state.get("roi2_spectrum_view") not in _view_options:
                        st.session_state["roi2_spectrum_view"] = _view_options[0]
                    _spectrum_view = st.radio(
                        "스펙트럼 데이터 선택",
                        _view_options,
                        horizontal=True,
                        key="roi2_spectrum_view",
                        help=(
                            "같은 ROI의 원본 DN과 보정 반사율을 즉시 전환합니다. "
                            "비교 모드는 반사율(왼쪽)과 DN(오른쪽)을 서로 다른 Y축에 표시합니다."
                        ),
                    )
                    _stats = (
                        _raw_stats
                        if _spectrum_view == "원본 DN"
                        else _calibrated_stats
                    )
                    if _stats is None:
                        _stats = _raw_stats

                    st.caption(
                        f"사용된 픽셀 수: {_npix:,}"
                        + (f"  ·  연결된 보정: {_calib['method']}"
                           if _calib is not None else "")
                    )

                    _has_wl = _rwl is not None and len(_rwl) == len(_raw_stats)
                    _xax   = _rwl if _has_wl else list(range(len(_raw_stats)))
                    _xttl  = "Wavelength (nm)" if _has_wl else "Band index"

                    st.markdown("##### 그래프 표시 범위")
                    _plot_mask = np.ones(len(_raw_stats), dtype=bool)
                    if _has_wl:
                        _wl_array = np.asarray(_rwl, dtype=float)
                        _finite_wl = _wl_array[np.isfinite(_wl_array)]
                        _wl_min = float(np.min(_finite_wl))
                        _wl_max = float(np.max(_finite_wl))
                        _wl_steps = np.diff(np.unique(np.sort(_finite_wl)))
                        _wl_step = float(np.median(_wl_steps)) if len(_wl_steps) else 1.0
                        _wl_step = max(0.1, min(10.0, _wl_step))
                        _quick_900 = float(
                            _finite_wl[np.argmin(np.abs(_finite_wl - 900.0))]
                        )
                        _wave_key = (
                            f"roi2_wave_range_{len(_raw_stats)}_"
                            f"{_wl_min:.2f}_{_wl_max:.2f}"
                        )
                        if _wave_key not in st.session_state:
                            st.session_state[_wave_key] = (_wl_min, _wl_max)

                        _wr1, _wr2 = st.columns(2)
                        with _wr1:
                            if st.button(
                                "⚡ 900 nm까지만 보기",
                                use_container_width=True,
                                disabled=not (_wl_min < 900.0 < _wl_max),
                                key="roi2_wave_to_900",
                            ):
                                st.session_state[_wave_key] = (_wl_min, _quick_900)
                        with _wr2:
                            if st.button(
                                "↔️ 전체 파장 복원",
                                use_container_width=True,
                                key="roi2_wave_full",
                            ):
                                st.session_state[_wave_key] = (_wl_min, _wl_max)

                        _wave_range = st.slider(
                            "표시할 파장 범위 (nm)",
                            min_value=_wl_min,
                            max_value=_wl_max,
                            step=_wl_step,
                            key=_wave_key,
                            help="그래프만 확대합니다. ROI 계산과 CSV 저장은 전체 밴드를 유지합니다.",
                        )
                        _plot_mask = (
                            (_wl_array >= float(_wave_range[0]))
                            & (_wl_array <= float(_wave_range[1]))
                        )

                    _gc1, _gc2 = st.columns(2)
                    with _gc1:
                        _show_band = st.checkbox(
                            "±1 표준편차 범위 표시", value=True, key="roi2_show_std"
                        )
                    with _gc2:
                        _y_mode = st.selectbox(
                            "Y축 표시 범위",
                            ("자동", "반사율 0–1", "직접 입력"),
                            key="roi2_y_mode",
                            help="자동은 현재 선택한 파장 구간만 기준으로 범위를 다시 맞춥니다.",
                        )

                    _y_range = None
                    if _y_mode == "반사율 0–1":
                        _y_range = [0.0, 1.0]
                    elif _y_mode == "직접 입력":
                        _yr1, _yr2 = st.columns(2)
                        with _yr1:
                            _y_min_manual = st.number_input(
                                "Y축 최소", value=0.0, format="%.4f", key="roi2_y_min"
                            )
                        with _yr2:
                            _y_max_manual = st.number_input(
                                "Y축 최대", value=1.0, format="%.4f", key="roi2_y_max"
                            )
                        if float(_y_max_manual) > float(_y_min_manual):
                            _y_range = [float(_y_min_manual), float(_y_max_manual)]
                        else:
                            st.warning("Y축 최대값은 최소값보다 커야 합니다. 현재는 자동 범위를 사용합니다.")

                    _plot_x = np.asarray(_xax)[_plot_mask]

                    def _add_roi_stat_traces(
                        figure,
                        frame,
                        *,
                        prefix,
                        mean_color,
                        median_color,
                        fill_color,
                        secondary_y=False,
                    ):
                        _mean = frame["mean"].to_numpy()[_plot_mask]
                        _median = frame["median"].to_numpy()[_plot_mask]
                        _std = frame["std"].to_numpy()[_plot_mask]
                        if _show_band:
                            figure.add_trace(
                                go.Scatter(
                                    x=list(_plot_x) + list(_plot_x)[::-1],
                                    y=list(_mean + _std) + list(_mean - _std)[::-1],
                                    fill="toself",
                                    fillcolor=fill_color,
                                    line=dict(color="rgba(0,0,0,0)"),
                                    hoverinfo="skip",
                                    name=f"{prefix} ±1 std",
                                    showlegend=False,
                                ),
                                secondary_y=secondary_y,
                            ) if getattr(figure, "_grid_ref", None) is not None else figure.add_trace(
                                go.Scatter(
                                    x=list(_plot_x) + list(_plot_x)[::-1],
                                    y=list(_mean + _std) + list(_mean - _std)[::-1],
                                    fill="toself", fillcolor=fill_color,
                                    line=dict(color="rgba(0,0,0,0)"),
                                    hoverinfo="skip", name=f"{prefix} ±1 std",
                                    showlegend=False,
                                )
                            )
                        _mean_trace = go.Scatter(
                            x=_plot_x, y=_mean, mode="lines",
                            name=f"{prefix} Mean",
                            line=dict(color=mean_color, width=2.5),
                        )
                        _median_trace = go.Scatter(
                            x=_plot_x, y=_median, mode="lines",
                            name=f"{prefix} Median",
                            line=dict(color=median_color, width=1.6, dash="dash"),
                        )
                        if getattr(figure, "_grid_ref", None) is not None:
                            figure.add_trace(_mean_trace, secondary_y=secondary_y)
                            figure.add_trace(_median_trace, secondary_y=secondary_y)
                        else:
                            figure.add_trace(_mean_trace)
                            figure.add_trace(_median_trace)

                    if _spectrum_view == "원본·보정 비교":
                        _sfig = make_subplots(specs=[[{"secondary_y": True}]])
                        _add_roi_stat_traces(
                            _sfig, _calibrated_stats,
                            prefix="Reflectance", mean_color="#1f77b4",
                            median_color="#17becf", fill_color="rgba(31,119,180,0.13)",
                            secondary_y=False,
                        )
                        _add_roi_stat_traces(
                            _sfig, _raw_stats,
                            prefix="Raw DN", mean_color="#ff7f0e",
                            median_color="#d62728", fill_color="rgba(255,127,14,0.10)",
                            secondary_y=True,
                        )
                        _sfig.update_yaxes(
                            title_text="Reflectance (unitless)",
                            range=_y_range,
                            secondary_y=False,
                        )
                        _sfig.update_yaxes(title_text="Raw DN", secondary_y=True)
                        _spectrum_title = "같은 ROI의 원본 DN ↔ 보정 반사율 비교"
                    else:
                        _sfig = go.Figure()
                        _is_reflectance_view = _spectrum_view == "보정 반사율"
                        _add_roi_stat_traces(
                            _sfig, _stats,
                            prefix="Reflectance" if _is_reflectance_view else "Raw DN",
                            mean_color="#1f77b4" if _is_reflectance_view else "#ff7f0e",
                            median_color="#d62728",
                            fill_color=(
                                "rgba(31,119,180,0.15)" if _is_reflectance_view
                                else "rgba(255,127,14,0.12)"
                            ),
                        )
                        _spectrum_title = (
                            "ROI 반사율 스펙트럼 — 보정 적용됨"
                            if _is_reflectance_view
                            else "ROI Raw DN 스펙트럼"
                        )
                        _sfig.update_yaxes(
                            title_text=(
                                "Reflectance (unitless)" if _is_reflectance_view else "Raw DN"
                            ),
                            range=(
                                _y_range
                                if _is_reflectance_view or _y_mode != "반사율 0–1"
                                else None
                            ),
                        )
                        if not _is_reflectance_view and _y_mode == "반사율 0–1":
                            st.warning("원본 DN 보기에서는 ‘반사율 0–1’ Y축 설정을 적용하지 않았습니다.")

                    _sfig.update_layout(
                        title=_spectrum_title,
                        height=420,
                        xaxis_title=_xttl,
                        margin=dict(l=50, r=55, t=45, b=45),
                        legend=dict(orientation="h", y=1.14),
                        hovermode="x unified",
                    )
                    st.plotly_chart(_sfig, use_container_width=True, key="roi2_spec_chart")

                    if _calib is not None and _calibrated_stats is not None:
                        with st.expander("🩺 보정 이상 밴드·계수 진단", expanded=False):
                            _diag = roi_utils.calibration_diagnostics(
                                _raw_stats,
                                _calibrated_stats,
                                _calib["a"],
                                _calib["b"],
                                _rwl if _has_wl else None,
                            )
                            _suspect = _diag[_diag["suspect"]]
                            _dc1, _dc2, _dc3 = st.columns(3)
                            _dc1.metric("전체 밴드", f"{len(_diag):,}")
                            _dc2.metric("진단 경고 밴드", f"{len(_suspect):,}")
                            _dc3.metric(
                                "경고 비율",
                                f"{100.0 * len(_suspect) / max(len(_diag), 1):.1f}%",
                            )
                            if len(_suspect):
                                st.warning(
                                    "자동 진단은 보정값을 수정하지 않습니다. 아래 밴드의 패널 포화, "
                                    "White/Dark 선택, 보정계수 a·b를 우선 확인하세요."
                                )
                                st.dataframe(
                                    _suspect,
                                    use_container_width=True,
                                    hide_index=True,
                                    height=min(340, 70 + 35 * len(_suspect)),
                                )
                            else:
                                st.success(
                                    "현재 ROI에서는 무효/음수 gain, 극단적 gain, "
                                    "-0.05–1.20 밖의 평균 반사율이 감지되지 않았습니다."
                                )

                            _diag_x = (
                                _diag["wavelength_nm"]
                                if "wavelength_nm" in _diag else _diag["band_index"]
                            )
                            _coef_fig = make_subplots(specs=[[{"secondary_y": True}]])
                            _coef_fig.add_trace(
                                go.Scatter(
                                    x=_diag_x, y=_diag["calibration_a"],
                                    name="a (gain)", line=dict(color="#9467bd"),
                                ),
                                secondary_y=False,
                            )
                            _coef_fig.add_trace(
                                go.Scatter(
                                    x=_diag_x, y=_diag["calibration_b"],
                                    name="b (offset)", line=dict(color="#8c564b"),
                                ),
                                secondary_y=True,
                            )
                            _coef_fig.update_yaxes(title_text="a", secondary_y=False)
                            _coef_fig.update_yaxes(title_text="b", secondary_y=True)
                            _coef_fig.update_layout(
                                title="적용된 보정계수 · R = a × DN + b",
                                height=330,
                                xaxis_title=_xttl,
                                legend=dict(orientation="h", y=1.12),
                                margin=dict(l=50, r=55, t=45, b=40),
                                hovermode="x unified",
                            )
                            st.plotly_chart(
                                _coef_fig,
                                use_container_width=True,
                                key="roi2_calibration_coefficients",
                            )

                    # Vegetation index readout — meaningful because values are raw
                    if _has_wl:
                        _wla = np.array(_rwl)
                        _ri  = int(np.argmin(np.abs(_wla - 670)))
                        _ni  = int(np.argmin(np.abs(_wla - 800)))
                        if abs(_wla[_ri] - 670) < 25 and abs(_wla[_ni] - 800) < 25:
                            _index_stats = (
                                _raw_stats
                                if _spectrum_view == "원본 DN"
                                else (
                                    _calibrated_stats
                                    if _calibrated_stats is not None
                                    else _raw_stats
                                )
                            )
                            _red = float(_index_stats["mean"].iloc[_ri])
                            _nir = float(_index_stats["mean"].iloc[_ni])
                            _den = _nir + _red
                            if _den > 1e-9:
                                m1, m2 = st.columns(2)
                                _index_suffix = (
                                    " (Raw DN 참고용)"
                                    if _spectrum_view == "원본 DN"
                                    else " (반사율)"
                                )
                                m1.metric(
                                    "NDVI" + _index_suffix,
                                    f"{(_nir - _red) / _den:.4f}",
                                )
                                m2.metric("NIR/Red" + _index_suffix, f"{_nir / _red:.3f}"
                                          if _red > 1e-9 else "—")
                                if _spectrum_view == "원본 DN":
                                    st.caption(
                                        "⚠️ Raw DN 기반 식생지수는 센서의 밴드별 감도와 Dark 영향을 "
                                        "포함하므로 논문용 지수로 사용하지 마세요."
                                    )

                    if _spectrum_view == "원본·보정 비교":
                        _prev = pd.DataFrame({
                            "raw_mean": _raw_stats["mean"],
                            "raw_median": _raw_stats["median"],
                            "reflectance_mean": _calibrated_stats["mean"],
                            "reflectance_median": _calibrated_stats["median"],
                        })
                    else:
                        _prev = _stats[["mean", "median", "std"]].copy()
                    if _has_wl:
                        _prev.insert(0, "wavelength_nm", _rwl)
                    else:
                        _prev.insert(0, "band_index", np.arange(len(_prev)))
                    st.dataframe(_prev.head(15), use_container_width=True, height=220)

                    _def_out = str(
                        Path(st.session_state["roi2_file"]).with_name(
                            Path(st.session_state["roi2_file"]).stem
                            + ("_roi_reflectance.csv" if _calib is not None else "_roi_raw_dn.csv")
                        )
                    )
                    _save_to = st.text_input(
                        "CSV 저장 경로", value=_def_out, key="roi2_save_path"
                    )
                    if st.button("💾 CSV 저장", type="primary",
                                 use_container_width=True, key="roi2_save_btn"):
                        _out = roi_utils.save_roi_csv(
                            data=_rdata,
                            wavelengths=_rwl,
                            region=_region,
                            source_file=st.session_state["roi2_file"],
                            path=_save_to,
                            value_units=st.session_state["roi2_units"],
                            calibration=((_calib["a"], _calib["b"])
                                         if _calib is not None else None),
                            calibration_meta=(
                                {
                                    "method": _calib.get("method", ""),
                                    "selected_profile": _calib.get("selected_profile", ""),
                                    "selection_source": (_calib.get("meta") or {}).get(
                                        "selection_source", ""
                                    ),
                                    "meta": _calib.get("meta", {}),
                                }
                                if _calib is not None else None
                            ),
                        )
                        if _calib is not None:
                            _raw_out = _out.with_name(
                                _out.stem + "_raw_dn" + _out.suffix
                            )
                            roi_utils.save_roi_csv(
                                data=_rdata,
                                wavelengths=_rwl,
                                region=_region,
                                source_file=st.session_state["roi2_file"],
                                path=str(_raw_out),
                                value_units="raw DN",
                                calibration=None,
                                calibration_meta={
                                    "method": _calib.get("method", ""),
                                    "selected_profile": _calib.get("selected_profile", ""),
                                    "selection_source": (_calib.get("meta") or {}).get(
                                        "selection_source", ""
                                    ),
                                    "meta": _calib.get("meta", {}),
                                },
                            )
                            st.success(
                                f"✅ 보정후: `{_out.resolve()}`\n\n"
                                f"✅ 보정전: `{_raw_out.resolve()}`"
                            )
                        else:
                            st.success(f"✅ 저장 완료: `{_out.resolve()}`")

                except Exception:
                    st.error("❌ ROI 스펙트럼 계산 실패")
                    st.code(traceback.format_exc(), language="python")


# ============================================================
# Tab 3 – Panel calibration (white / grey reference extraction)
# ============================================================

with tab_panel:
    from src import roi_utils as _ru
    from src import radiometry as _rad
    _rad = importlib.reload(_rad)

    st.markdown("### 🎯 보정 패널로 반사율 캘리브레이션 만들기")
    st.caption(
        "패널 ROI의 실제 반사도와 Dark 기준값을 등록하면 포화·노이즈를 파장별로 "
        "검사하고, 유효한 패널들을 자동 결합해 반사율 보정계수를 만듭니다."
    )

    with st.expander("📖 패널이 여러 장일 때 왜 더 정확한가?", expanded=False):
        st.markdown(
            "센서가 기록하는 값은 반사율이 아니라\n\n"
            "$$DN(\\lambda) = R(\\lambda)\\,E(\\lambda)\\,S(\\lambda) + d(\\lambda)$$\n\n"
            "입니다. $E$는 조명, $S$는 센서 감도, $d$는 dark·경로복사 같은 **더해지는 오프셋**입니다.\n\n"
            "**Sensor Dark** — 렌즈를 막고 같은 integration time·gain으로 측정한 값을 "
            "모든 영상과 패널에서 먼저 뺍니다. 패널이 한 장이어도 "
            "$R = R_{panel}(DN-D)/(DN_{panel}-D)$ 로 절대 반사율을 계산할 수 있습니다.\n\n"
            "**Dark 파일이 없을 때** — 센서 Dark가 파장별로 거의 평평하다는 사전 확인이 "
            "있다면 수동 상수값(기본 DN 100)을 임시로 사용할 수 있습니다. 이 경우 결과에 "
            "합성 Dark 사용 이력이 남습니다.\n\n"
            "**패널 2장 이상** — 밴드마다 $R=a(\\lambda)(DN-D)$를 모든 유효 패널에 "
            "동시에 맞춥니다. 밝은 패널이 포화에 가까워지면 그 가중치를 부드럽게 "
            "낮추고, 정상인 낮은 반사율 패널이 같은 회귀식을 이어받습니다. 패널 스펙트럼을 "
            "딱 잘라 붙이지 않으므로 전환 경계의 불연속을 줄일 수 있습니다.\n\n"
            "**공통 구간 검증** — 여러 패널이 모두 정상인 파장에서 각각의 보정계수가 "
            "얼마나 일치하는지 CV로 확인합니다. 차이가 크면 패널 높이·각도·오염 또는 "
            "조명 변화를 점검해야 합니다.\n\n"
            "**패널 선택 요령** — 관심 대상의 반사율 범위를 감싸도록 고르세요. "
            "식생 NIR은 40~60%까지 올라가므로 밝은 패널이 필요하고, "
            "그림자나 흙은 5~10%라 어두운 패널이 있어야 그 구간이 외삽이 아닌 내삽이 됩니다."
        )

    _pn_defaults: dict = {
        "pn_data": None, "pn_wl": None, "pn_rgb": None, "pn_meta": None,
        "pn_file": "", "pn_region": None,
        "pn_panels": [],      # [{"name","reflectance","box","type","spectrum"}]
        "pn_source_max": None,
        "pn_auto_rejected": [],
        "pn_fit": None,
        "pn_dark_mode": "수동 상수 DN",
        "pn_dark_active_mode": "",
        "pn_dark_path": "",
        "pn_dark_spectrum": None,
        "pn_dark_noise": None,
        "pn_dark_qc": None,
        "pn_preview_rgb": None,
        "pn_saved_calibration": "",
        "wd_last_profile": "",
    }
    for k, v in _pn_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Step 1: load ──────────────────────────────────────────
    st.markdown("#### 1️⃣ 패널이 찍힌 이미지 열기")
    _pn_candidates: list[str] = []
    if data_src == "로컬 폴더" and local_folder:
        _pn_candidates.extend(_scan_local_hsi_files(local_folder))
    if st.session_state.get("pn_file"):
        _pn_candidates.insert(0, st.session_state["pn_file"])
    _pn_candidates = list(dict.fromkeys(_pn_candidates))
    _pn_manual = "__manual_panel_path__"
    _pn_options = _pn_candidates + [_pn_manual]
    _pn_current = st.session_state.get("pn_file", "")
    _pn_index = (
        _pn_options.index(_pn_current)
        if _pn_current in _pn_options
        else (0 if _pn_candidates else len(_pn_options) - 1)
    )
    _p1, _p2, _p3, _p4 = st.columns([4, 1.4, 1, 1])
    with _p1:
        _pn_choice = st.selectbox(
            "패널 영상 선택",
            _pn_options,
            index=_pn_index,
            format_func=lambda value: (
                "직접 경로 입력…" if value == _pn_manual else Path(value).name
            ),
            label_visibility="collapsed",
            key="pn_file_choice",
        )
        if _pn_choice == _pn_manual:
            _pn_path = st.text_input(
                "패널 영상 직접 경로",
                value=st.session_state.get("pn_file", ""),
                placeholder="Z:/.../vnir.bil.hdr 또는 reference.hdr",
                key="pn_path_input",
            )
        else:
            _pn_path = _pn_choice
    with _p2:
        _pn_ds = st.selectbox(
            "다운샘플", [1, 2, 4, 8], index=0,
            format_func=lambda v: f"다운샘플 ×{v}",
            label_visibility="collapsed", key="pn_ds",
            help="패널은 보통 크므로 4로도 충분합니다. 작은 패널이면 1~2를 쓰세요.",
        )
    with _p3:
        _pn_load = st.button("📂 로드", use_container_width=True, key="pn_load")
    with _p4:
        st.button(
            "🪟 선택",
            use_container_width=True,
            key="pn_native_file",
            on_click=_browse_file_into_state,
            args=("pn_path_input", "보정 패널 영상 선택"),
            kwargs={"set_values": {"pn_file_choice": _pn_manual}},
            disabled=not _native_dialogs_available(),
            help="Windows 탐색기에서 패널이 찍힌 파일을 선택합니다.",
        )

    if _pn_load and _pn_path:
        try:
            with st.spinner("이미지 로딩 중..."):
                from src.data_loader import HyperspectralLoader as _HL2

                _pd, _pm = _HL2({"spatial_downsample": int(_pn_ds)}).load_local(_pn_path)
                st.session_state.update({
                    "pn_data": _pd, "pn_wl": _pm.get("wavelengths"),
                    "pn_meta": _pm, "pn_file": _pn_path,
                    "pn_rgb": _ru.display_rgb(_pd, _pm.get("wavelengths")),
                    "pn_source_max": float(np.nanmax(_pd)),
                    "pn_region": None, "pn_panels": [], "pn_auto_rejected": [],
                    "pn_fit": None, "pn_dark_spectrum": None,
                    "pn_dark_noise": None, "pn_dark_qc": None,
                    "pn_preview_rgb": None, "pn_saved_calibration": "",
                    "active_calibration_path": "",
                })
            st.success(
                f"✅ {_pd.shape[0]} × {_pd.shape[1]} px · {_pd.shape[2]} 밴드 "
                f"· 다운샘플 ×{_pn_ds}"
            )
        except Exception:
            st.error("❌ 로드 실패")
            st.code(traceback.format_exc(), language="python")

    _pdata = st.session_state.get("pn_data")
    if _pdata is None:
        st.info("📂 패널이 포함된 이미지를 열어주세요. 별도 reference 스캔도, "
                "패널이 함께 찍힌 현장 이미지도 됩니다.")
    else:
        _pwl = st.session_state["pn_wl"]
        _pH, _pW, _pB = _pdata.shape

        st.markdown("#### 2️⃣ 패널 영역 지정")
        _pm1, _pm2 = st.columns([3, 1])
        with _pm1:
            _pn_mouse_mode = st.radio(
                "패널 영상 마우스 모드",
                ("⬚ Box ROI", "✏️ Lasso ROI", "🔍 확대"),
                horizontal=True,
                key="pn_mouse_mode",
            )
        with _pm2:
            st.write("")
            if st.button("↩️ 확대 초기화", key="pn_zoom_reset", use_container_width=True):
                st.session_state["pn_zoom_revision"] = (
                    int(st.session_state.get("pn_zoom_revision", 0)) + 1
                )
                st.rerun()
        _pl, _pr = st.columns([3, 2])

        with _pl:
            st.caption("Box Select 또는 Lasso Select로 **패널 하나**를 감싸세요. "
                       "가장자리 그림자는 피하고 안쪽만 잡는 게 좋습니다.")
            _pfig = go.Figure()
            _pfig.add_trace(go.Image(z=st.session_state["pn_rgb"]))

            # Outline panels already registered
            for _i, _p in enumerate(st.session_state["pn_panels"], 1):
                _b = _p["box"]
                _pfig.add_shape(
                    type="rect", x0=_b[2], x1=_b[3], y0=_b[0], y1=_b[1],
                    line=dict(color="#00e5ff", width=2),
                )
                _pfig.add_annotation(
                    x=_b[2], y=_b[0], text=f"{_i}", showarrow=False,
                    font=dict(color="#00e5ff", size=13), yshift=10,
                )

            _pfig.update_layout(
                dragmode=(
                    "zoom" if _pn_mouse_mode == "🔍 확대"
                    else "lasso" if _pn_mouse_mode == "✏️ Lasso ROI"
                    else "select"
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=520,
                newselection=dict(line=dict(color="#ffd54f", width=3)),
                uirevision=(
                    f"{st.session_state.get('pn_file', '')}|"
                    f"{st.session_state.get('pn_zoom_revision', 0)}"
                ),
            )
            _pfig.update_xaxes(showticklabels=False)
            _pfig.update_yaxes(showticklabels=False)

            _pev = st.plotly_chart(
                _pfig, key="pn_image_chart", on_select="rerun",
                selection_mode=("box", "lasso"), use_container_width=True,
                config=_ROI_PLOTLY_CONFIG,
            )
            if _pev is not None and hasattr(_pev, "selection"):
                _rg = _ru.selection_to_region(_pev.selection, _pH, _pW)
                if _rg is not None:
                    st.session_state["pn_region"] = _rg

        with _pr:
            _reg = st.session_state.get("pn_region")
            if not _reg:
                st.info("⬅️ 이미지에서 패널 영역을 드래그하세요.")
            else:
                _b = _reg["roi"]
                st.write(f"선택 영역: row `{_b[0]}:{_b[1]}`, col `{_b[2]}:{_b[3]}`")
                try:
                    _pix, _, _ = _ru.region_pixels(_pdata, _reg)
                    _unif = float(_pix.mean(axis=1).std() /
                                  max(_pix.mean(), 1e-9))
                    _mx = float(_pix.max())
                    _sat = _rad.panel_saturation_metrics(
                        _pix,
                        observed_max=st.session_state.get("pn_source_max"),
                    )

                    _m1, _m2, _m3, _m4 = st.columns(4)
                    _m1.metric("픽셀 수", f"{len(_pix):,}")
                    _m2.metric("균일도 (CV)", f"{_unif:.3f}",
                               help="0.10 미만이면 균일하게 잘 잡은 것입니다.")
                    _m3.metric("선택 최대 DN", f"{_mx:,.0f}")
                    _m4.metric(
                        "포화 밴드",
                        f"{_sat['saturated_band_count']} / {_pB}",
                        help=(
                            f"추정 ADC 상한: {_sat.get('adc_ceiling') or '알 수 없음'} · "
                            "한 밴드에서 선택 픽셀의 1% 이상이 상한 99%에 도달하거나, "
                            "상단에서 반복되는 clipping plateau가 검출되면 제외"
                        ),
                    )
                    if _unif > 0.10:
                        st.warning(
                            "⚠️ 균일도가 낮습니다. 배경이나 그림자가 섞였을 수 있으니 "
                            "영역을 패널 안쪽으로 좁혀보세요."
                        )
                    if not _sat["usable"]:
                        _bad_indices = _sat["saturated_band_indices"]
                        _bad_labels = [
                            (
                                f"{_pwl[index]:.1f} nm"
                                if _pwl is not None and len(_pwl) == _pB
                                else f"band {index}"
                            )
                            for index in _bad_indices[:8]
                        ]
                        st.warning(
                            "⚠️ 이 패널은 일부 파장에서만 사용됩니다. "
                            f"포화 밴드 {_sat['saturated_band_count']}개가 검출되었습니다"
                            + (f" ({', '.join(_bad_labels)})" if _bad_labels else "")
                            + ". 해당 밴드의 가중치는 자동으로 0이 되고, 등록한 더 낮은 "
                            "반사율 패널이 그 구간을 담당합니다."
                        )
                    elif _sat["near_band_count"]:
                        st.warning(
                            f"⚠️ 포화 직전 밴드가 {_sat['near_band_count']}개 있습니다. "
                            "현재는 사용할 수 있지만 노출을 낮춘 reference가 더 안전합니다."
                        )

                    _pna, _pnb = st.columns([2, 1])
                    with _pna:
                        _pname = st.text_input(
                            "패널 이름", value=f"panel_{len(st.session_state['pn_panels'])+1}",
                            key="pn_name_input",
                        )
                    with _pnb:
                        _prefl = st.number_input(
                            "반사율", min_value=0.0, max_value=1.0,
                            value=0.99, step=0.01, format="%.3f",
                            key="pn_refl_input",
                            help="패널 성적서의 공칭 반사율 (예: 0.99, 0.50, 0.25)",
                        )

                    if st.button(
                        "➕ 이 영역을 패널로 추가",
                        type="primary",
                        use_container_width=True,
                        key="pn_add",
                        disabled=(_sat["saturated_band_count"] >= _pB),
                        help=(
                            "모든 밴드가 포화된 영역은 등록할 수 없습니다."
                            if _sat["saturated_band_count"] >= _pB else None
                        ),
                    ):
                        _spec = np.median(_ru.region_pixels(_pdata, _reg)[0], axis=0)
                        st.session_state["pn_panels"].append({
                            "name": _pname,
                            "reflectance": float(_prefl),
                            "box": list(_b),
                            "region": dict(_reg),
                            "uniformity": _unif,
                            "max_dn": _mx,
                            "spectrum": _spec,
                            "saturation": _sat,
                        })
                        st.session_state["pn_fit"] = None
                        st.session_state["pn_preview_rgb"] = None
                        st.session_state["pn_saved_calibration"] = ""
                        st.session_state["active_calibration_path"] = ""
                        st.session_state["pn_region"] = None
                        st.rerun()
                except Exception:
                    st.error("❌ 영역 계산 실패")
                    st.code(traceback.format_exc(), language="python")

            if st.button("🔍 패널 자동 탐지 시도", use_container_width=True,
                         key="pn_auto"):
                try:
                    with st.spinner("탐지 중..."):
                        _auto = _rad.detect_panels(_pdata, n_panels=4)
                    if not _auto:
                        st.warning("패널 후보를 찾지 못했습니다. 수동으로 지정해 주세요.")
                    else:
                        _auto_valid = []
                        _auto_rejected = []
                        for i, p in enumerate(_auto, 1):
                            _ab = p["box"]
                            _apix = _pdata[_ab[0]:_ab[1], _ab[2]:_ab[3], :].reshape(
                                -1, _pB
                            )
                            _asat = _rad.panel_saturation_metrics(
                                _apix,
                                observed_max=st.session_state.get("pn_source_max"),
                            )
                            if _asat["saturated_band_count"] >= _pB:
                                _auto_rejected.append(
                                    f"auto_{i} (전 밴드 포화)"
                                )
                                continue
                            _auto_valid.append({
                                "name": f"auto_{i}",
                                "reflectance": 0.0,
                                "box": _ab,
                                "region": {"type": "box", "roi": list(_ab)},
                                "uniformity": _rad.panel_uniformity(_pdata, _ab),
                                "max_dn": float(np.max(_apix)),
                                "spectrum": np.median(_apix, axis=0),
                                "saturation": _asat,
                            })
                        st.session_state["pn_panels"] = _auto_valid
                        st.session_state["pn_auto_rejected"] = _auto_rejected
                        st.session_state["pn_fit"] = None
                        st.session_state["pn_preview_rgb"] = None
                        st.session_state["pn_saved_calibration"] = ""
                        st.session_state["active_calibration_path"] = ""
                        if _auto_rejected:
                            st.error(
                                "⛔ 포화되어 자동 제외된 후보: "
                                + ", ".join(_auto_rejected)
                            )
                        st.warning(
                            "자동 탐지 결과입니다. **각 패널의 반사율을 아래 표에서 "
                            "직접 입력**하고, 박스가 실제 패널과 맞는지 확인하세요."
                        )
                        st.rerun()
                except Exception:
                    st.error("❌ 자동 탐지 실패")
                    st.code(traceback.format_exc(), language="python")

            if st.session_state.get("pn_auto_rejected"):
                st.error(
                    "⛔ 포화되어 자동 제외된 후보: "
                    + ", ".join(st.session_state["pn_auto_rejected"])
                )

        # ── Step 3: registered panels & calibration ───────────
        _panels = st.session_state["pn_panels"]
        if _panels:
            st.markdown("#### 3️⃣ 등록된 패널")
            st.caption(
                "반사율은 패널의 인증값을 그대로 입력합니다: 50%=0.500, 99%=0.990. "
                "프로그램은 입력값으로 절대 반사율 계수를 계산하며 50%를 99%로 "
                "임의 변경하지 않습니다."
            )

            for _panel in _panels:
                if "saturation" not in _panel:
                    _pb = _panel["box"]
                    _ppix = _pdata[
                        _pb[0]:_pb[1], _pb[2]:_pb[3], :
                    ].reshape(-1, _pB)
                    _panel["saturation"] = _rad.panel_saturation_metrics(
                        _ppix,
                        observed_max=st.session_state.get("pn_source_max"),
                    )

            _edit = st.data_editor(
                pd.DataFrame([{
                    "패널": p["name"],
                    "반사율": p["reflectance"],
                    "상태": (
                        "전 밴드 사용" if p["saturation"]["usable"]
                        else (
                            "사용 불가" if p["saturation"]["saturated_band_count"] >= _pB
                            else "부분 사용 (포화 밴드 자동 제외)"
                        )
                    ),
                    "포화 밴드": p["saturation"]["saturated_band_count"],
                    "box": str(p["box"]),
                    "평균 DN": round(float(np.mean(p["spectrum"])), 1),
                    "최대 DN": round(p["max_dn"], 1),
                    "균일도": (round(p["uniformity"], 3)
                              if p["uniformity"] == p["uniformity"] else None),
                } for p in _panels]),
                column_config={
                    "반사율": st.column_config.NumberColumn(
                        min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                    ),
                    "상태": st.column_config.TextColumn(disabled=True),
                    "포화 밴드": st.column_config.NumberColumn(disabled=True),
                    "box": st.column_config.TextColumn(disabled=True),
                    "평균 DN": st.column_config.NumberColumn(disabled=True),
                    "최대 DN": st.column_config.NumberColumn(disabled=True),
                    "균일도": st.column_config.NumberColumn(disabled=True),
                },
                hide_index=True, use_container_width=True, key="pn_editor",
            )
            _panel_values_changed = False
            for _p, (_, _row) in zip(_panels, _edit.iterrows()):
                _panel_values_changed |= (
                    float(_p["reflectance"]) != float(_row["반사율"])
                    or str(_p["name"]) != str(_row["패널"])
                )
                _p["reflectance"] = float(_row["반사율"])
                _p["name"] = str(_row["패널"])
            if _panel_values_changed:
                st.session_state["pn_fit"] = None
                st.session_state["pn_preview_rgb"] = None
                st.session_state["pn_saved_calibration"] = ""
                st.session_state["active_calibration_path"] = ""

            st.markdown("#### 4️⃣ 센서 Dark 설정")
            st.caption(
                "Dark 파일이 없으면 모든 밴드에 같은 DN을 적용할 수 있습니다. "
                "기본값은 100이며, 실측 Dark가 있으면 파일 방식이 더 정확합니다."
            )
            _dark_mode = st.radio(
                "Dark 준비 방법",
                ("수동 상수 DN", "실측 Dark 파일"),
                horizontal=True,
                key="pn_dark_mode",
                help="Dark 영상이 없으면 수동 상수 DN을 사용하세요.",
            )
            if st.session_state.get("pn_dark_active_mode") != _dark_mode:
                st.session_state["pn_fit"] = None
                st.session_state["pn_preview_rgb"] = None
                st.session_state["pn_saved_calibration"] = ""
                st.session_state["active_calibration_path"] = ""
                st.session_state["pn_dark_active_mode"] = _dark_mode

            _dark_ready = False
            if _dark_mode == "수동 상수 DN":
                _manual_dark_dn = st.number_input(
                    "모든 밴드에 적용할 Dark DN",
                    min_value=0.0,
                    value=100.0,
                    step=1.0,
                    format="%.1f",
                    key="pn_manual_dark_dn",
                    help="현재 센서에서 확인한 평균 Dark 값이 있으면 입력하세요. 기본값은 100입니다.",
                )
                _current_dark_qc = st.session_state.get("pn_dark_qc") or {}
                _manual_changed = (
                    _current_dark_qc.get("source_type") != "synthetic_constant"
                    or float(_current_dark_qc.get("constant_dn", -1.0))
                    != float(_manual_dark_dn)
                    or np.asarray(
                        st.session_state.get("pn_dark_spectrum", [])
                    ).shape != (_pB,)
                )
                if _manual_changed:
                    _dark_spec, _dark_noise, _dark_qc = _rad.constant_dark_reference(
                        _pB, float(_manual_dark_dn)
                    )
                    st.session_state.update({
                        "pn_dark_path": "",
                        "pn_dark_spectrum": _dark_spec,
                        "pn_dark_noise": _dark_noise,
                        "pn_dark_qc": _dark_qc,
                        "pn_fit": None,
                        "pn_preview_rgb": None,
                        "pn_saved_calibration": "",
                        "active_calibration_path": "",
                    })
                _dark_ready = True
                st.warning(
                    f"⚠️ 합성 Dark 사용 중: 전 밴드 DN {float(_manual_dark_dn):,.1f}. "
                    "빠른 분석용이며, 논문용 최종 처리에는 같은 설정으로 찍은 실측 Dark를 권장합니다."
                )
            else:
                _dark_candidates: list[str] = []
                if data_src == "로컬 폴더" and local_folder:
                    _dark_candidates.extend(_scan_local_hsi_files(local_folder))
                try:
                    _panel_parent = Path(st.session_state["pn_file"]).expanduser().parent
                    if _panel_parent.is_dir():
                        _dark_candidates.extend(
                            str(item) for item in _panel_parent.iterdir()
                            if item.is_file() and item.suffix.lower() in _LOCAL_HSI_EXTS
                        )
                except Exception:
                    pass
                if st.session_state.get("pn_dark_path"):
                    _dark_candidates.insert(0, st.session_state["pn_dark_path"])
                _dark_candidates = list(dict.fromkeys(_dark_candidates))
                _dark_candidates.sort(
                    key=lambda value: (
                        not any(token in Path(value).name.lower()
                                for token in ("dark", "black", "shutter")),
                        Path(value).name.lower(),
                    )
                )
                _dark_manual_path = "__manual_dark_path__"
                _dark_options = _dark_candidates + [_dark_manual_path]
                _dark_current = st.session_state.get("pn_dark_path", "")
                if _dark_current in _dark_options:
                    _dark_index = _dark_options.index(_dark_current)
                else:
                    _dark_index = 0 if _dark_candidates else len(_dark_options) - 1

                _dc1, _dc2, _dc3 = st.columns([5, 1, 1])
                with _dc1:
                    _dark_choice = st.selectbox(
                        "Sensor dark 파일",
                        _dark_options,
                        index=_dark_index,
                        format_func=lambda value: (
                            "직접 경로 입력…" if value == _dark_manual_path
                            else Path(value).name
                        ),
                        key="pn_dark_choice",
                    )
                    if _dark_choice == _dark_manual_path:
                        _dark_path = st.text_input(
                            "Dark 파일 직접 경로",
                            value=st.session_state.get("pn_dark_path", ""),
                            placeholder="D:/references/dark_20260821_090000.swir.hdr",
                            key="pn_dark_manual_path",
                        )
                    else:
                        _dark_path = _dark_choice
                with _dc2:
                    st.write("")
                    _dark_load = st.button(
                        "🌑 Dark 로드", use_container_width=True, key="pn_dark_load"
                    )
                with _dc3:
                    st.write("")
                    st.button(
                        "🪟 선택",
                        use_container_width=True,
                        key="pn_dark_native_file",
                        on_click=_browse_file_into_state,
                        args=("pn_dark_manual_path", "Sensor Dark 파일 선택"),
                        kwargs={
                            "set_values": {"pn_dark_choice": _dark_manual_path}
                        },
                        disabled=not _native_dialogs_available(),
                        help="Windows 탐색기에서 실측 Dark 파일을 선택합니다.",
                    )

                if _dark_load:
                    try:
                        if not str(_dark_path).strip():
                            raise ValueError("Sensor dark 파일을 선택하세요.")
                        with st.spinner("Sensor dark 검사 중..."):
                            from src.data_loader import HyperspectralLoader as _DarkLoader

                            _dd, _dm = _DarkLoader({
                                "spatial_downsample": max(1, int(_pn_ds))
                            }).load_local(str(_dark_path).strip())
                            if _dd.shape[2] != _pB:
                                raise ValueError(
                                    f"Dark 밴드 수({_dd.shape[2]})와 패널 영상({_pB})이 다릅니다."
                                )
                            _dwl = _dm.get("wavelengths")
                            if _pwl is not None and _dwl is not None and not np.allclose(
                                _pwl, _dwl, rtol=0, atol=1.0
                            ):
                                raise ValueError("Dark와 패널 영상의 파장축이 다릅니다.")
                            _dark_spec, _dark_qc = _rad.robust_reference_spectrum(_dd)
                            _dark_qc = dict(_dark_qc)
                            _dark_qc.update({
                                "source_type": "measured_file",
                                "source": str(_dark_path).strip(),
                            })
                        st.session_state.update({
                            "pn_dark_path": str(_dark_path).strip(),
                            "pn_dark_spectrum": _dark_spec,
                            "pn_dark_noise": np.asarray(
                                _dark_qc.get("noise_mad_by_band", np.ones(_pB)),
                                dtype=np.float32,
                            ),
                            "pn_dark_qc": _dark_qc,
                            "pn_fit": None,
                            "pn_preview_rgb": None,
                            "pn_saved_calibration": "",
                            "active_calibration_path": "",
                        })
                        st.success(
                            f"✅ Dark 로드 완료 · 중앙 DN "
                            f"{float(np.nanmedian(_dark_spec)):,.1f}"
                        )
                        st.rerun()
                    except Exception:
                        st.error("❌ Sensor dark 로드 실패")
                        st.code(traceback.format_exc(), language="python")

                _dark_qc_now = st.session_state.get("pn_dark_qc") or {}
                _dark_ready = (
                    st.session_state.get("pn_dark_spectrum") is not None
                    and _dark_qc_now.get("source_type") == "measured_file"
                )
                if _dark_ready:
                    st.success(
                        "✅ 사용 중인 Dark: "
                        f"`{Path(st.session_state['pn_dark_path']).name}` · "
                        f"픽셀 {_dark_qc_now.get('sample_pixels', 0):,}개"
                    )
                else:
                    st.info("실측 Dark 파일을 선택하고 `Dark 로드`를 누르세요.")

            _c1, _c2 = st.columns(2)
            with _c1:
                if st.button("🗑️ 패널 목록 비우기", use_container_width=True,
                             key="pn_clear"):
                    st.session_state["pn_panels"] = []
                    st.session_state["pn_fit"] = None
                    st.session_state["pn_preview_rgb"] = None
                    st.session_state["pn_saved_calibration"] = ""
                    st.session_state["active_calibration_path"] = ""
                    st.rerun()
            with _c2:
                # Key must differ from the "pn_fit" session key below — a
                # widget key overwrites that entry with the button's bool.
                _fit_btn = st.button(
                    "✨ 자동 반사율 보정 계산",
                    type="primary",
                    use_container_width=True,
                    key="pn_fit_btn",
                    disabled=not _dark_ready,
                    help=(
                        "Dark 설정을 완료하세요."
                        if not _dark_ready else None
                    ),
                )

            # Panel spectra plot
            _spfig = go.Figure()
            _xax = _pwl if (_pwl and len(_pwl) == _pB) else list(range(_pB))
            for _p in _panels:
                _spfig.add_trace(go.Scatter(
                    x=_xax, y=_p["spectrum"], mode="lines",
                    name=f"{_p['name']} (R={_p['reflectance']:.2f})",
                ))
            _spfig.update_layout(
                height=300, xaxis_title="Wavelength (nm)", yaxis_title="DN",
                margin=dict(l=50, r=10, t=30, b=40),
                legend=dict(orientation="h", y=1.15),
                title="패널 raw DN 스펙트럼",
            )
            st.plotly_chart(_spfig, use_container_width=True, key="pn_spec_chart")

            if _fit_btn:
                try:
                    _fit_panels = [
                        p for p in _panels
                        if float(p.get("reflectance", 0.0)) > 0
                        and (p.get("saturation") or {}).get(
                            "saturated_band_count", _pB
                        ) < _pB
                    ]
                    _excluded_panels = [
                        p for p in _panels
                        if not (
                            float(p.get("reflectance", 0.0)) > 0
                            and (p.get("saturation") or {}).get(
                                "saturated_band_count", _pB
                            ) < _pB
                        )
                    ]
                    if _excluded_panels:
                        st.warning(
                            "반사율이 없거나 전 밴드가 포화되어 자동 제외: "
                            + ", ".join(p["name"] for p in _excluded_panels)
                        )
                    if not _fit_panels:
                        raise ValueError(
                            "사용 가능한 패널이 없습니다. 포화되지 않은 영역을 다시 지정하세요."
                        )
                    _dark_spec = st.session_state.get("pn_dark_spectrum")
                    if _dark_spec is None or not _dark_ready:
                        raise ValueError("Dark 설정을 완료하세요.")
                    _rs = [p["reflectance"] for p in _fit_panels]
                    if any(r <= 0 for r in _rs):
                        raise ValueError("모든 패널의 반사율을 0보다 크게 입력하세요.")

                    _dns = [p["spectrum"] for p in _fit_panels]
                    _panel_weights = np.asarray([
                        (p.get("saturation") or {}).get(
                            "headroom_weight_by_band", np.ones(_pB)
                        )
                        for p in _fit_panels
                    ], dtype=np.float64)
                    _a, _bb, _q = _rad.weighted_dark_panel_calibration(
                        _dns,
                        _rs,
                        np.asarray(_dark_spec),
                        panel_band_weights=_panel_weights,
                        dark_noise=st.session_state.get("pn_dark_noise"),
                    )
                    _method = _q["method"]
                    if _q["invalid_band_count"] >= _pB:
                        raise ValueError(
                            "유효한 보정 밴드가 없습니다. 패널의 포화와 Dark 설정을 확인하세요."
                        )

                    _cal_qc = _rad.evaluate_weighted_calibration(
                        _dns,
                        _rs,
                        np.asarray(_dark_spec),
                        _a,
                        _bb,
                        fit_quality=_q,
                        panel_usable_masks=[
                            (p.get("saturation") or {}).get(
                                "usable_band_mask", np.ones(_pB, dtype=bool)
                            )
                            for p in _fit_panels
                        ],
                        panel_uniformities=[
                            p.get("uniformity") for p in _fit_panels
                        ],
                        dark_source_type=(
                            st.session_state.get("pn_dark_qc") or {}
                        ).get("source_type", "unknown"),
                        wavelengths=_pwl,
                    )

                    _preview_cube = _rad.apply_resolved_calibration(
                        _pdata, {"a": _a, "b": _bb}
                    )
                    _preview_rgb = _ru.display_rgb(_preview_cube, _pwl)
                    del _preview_cube

                    _cal_dir = Path(output_dir or "./output").expanduser() / "calibration"
                    _cal_dir.mkdir(parents=True, exist_ok=True)
                    _cal_path = _cal_dir / (
                        f"{Path(st.session_state['pn_file']).stem}"
                        "_weighted_dark_calibration.npz"
                    )
                    _weights_used = np.asarray(_q["panel_weights"])
                    _dark_qc_used = st.session_state.get("pn_dark_qc") or {}
                    _dark_source_type = _dark_qc_used.get("source_type", "unknown")
                    _dark_source = (
                        str(Path(st.session_state["pn_dark_path"]).resolve())
                        if _dark_source_type == "measured_file"
                        else f"manual constant DN {float(_dark_qc_used.get('constant_dn', 100.0)):.1f}"
                    )
                    _meta = {
                        "method": _method,
                        "formula": _q["formula"],
                        "source_image": str(Path(st.session_state["pn_file"]).resolve()),
                        "dark_source": _dark_source,
                        "dark_source_type": _dark_source_type,
                        "dark_is_measured": _dark_source_type == "measured_file",
                        "manual_dark_dn": (
                            float(_dark_qc_used.get("constant_dn"))
                            if _dark_source_type == "synthetic_constant" else None
                        ),
                        "panels": [
                            {
                                "name": panel["name"],
                                "reflectance": float(panel["reflectance"]),
                                "box": panel["box"],
                                "saturated_band_indices": (
                                    panel.get("saturation") or {}
                                ).get("saturated_band_indices", []),
                                "weight_by_band": _weights_used[index].round(6).tolist(),
                            }
                            for index, panel in enumerate(_fit_panels)
                        ],
                        "invalid_band_indices": _q["invalid_band_indices"],
                        "fallback_band_indices": _q["fallback_band_indices"],
                        "blended_band_count": _q["blended_band_count"],
                        "median_coefficient_cv": _q["median_coefficient_cv"],
                        "qc_status": _cal_qc["status"],
                        "qc_auto_apply_allowed": _cal_qc["auto_apply_allowed"],
                        "qc_summary": _cal_qc,
                    }
                    _saved_cal = _rad.save_calibration(
                        str(_cal_path), _a, _bb, wavelengths=_pwl, meta=_meta
                    )

                    st.session_state["pn_fit"] = {
                        "a": _a, "b": _bb, "quality": _q, "method": _method,
                        "reflectances": _rs,
                        "names": [p["name"] for p in _fit_panels],
                        "boxes": [p["box"] for p in _fit_panels],
                        "excluded_panels": [p["name"] for p in _excluded_panels],
                        "saved_path": _saved_cal,
                        "dark_source_type": _dark_source_type,
                        "manual_dark_dn": _dark_qc_used.get("constant_dn"),
                        "calibration_qc": _cal_qc,
                        "panel_predictions": [
                            (np.asarray(spectrum) * _a + _bb).tolist()
                            for spectrum in _dns
                        ],
                    }
                    st.session_state["pn_preview_rgb"] = _preview_rgb
                    st.session_state["pn_saved_calibration"] = _saved_cal
                    st.session_state["active_calibration_path"] = (
                        _saved_cal if _cal_qc["auto_apply_allowed"] else ""
                    )
                    if _cal_qc["status"] == "PASS":
                        st.success(
                            f"✅ QC PASS · 패널 {len(_fit_panels)}장 · "
                            f"유효 밴드 {_pB - _q['invalid_band_count']}/{_pB}"
                        )
                    elif _cal_qc["status"] == "REVIEW":
                        st.warning(
                            "⚠️ QC REVIEW · 시험 분석에는 연결했지만, 논문용 사용 전 "
                            "아래 경고와 재구성 스펙트럼을 확인하세요."
                        )
                    else:
                        st.error(
                            "⛔ QC FAIL · 파일은 점검 기록으로 저장했지만 전체 필드 "
                            "분석에는 연결하지 않았습니다. 패널 ROI/Dark를 다시 지정하세요."
                        )
                except Exception:
                    st.error("❌ 계산 실패")
                    st.code(traceback.format_exc(), language="python")

            _fit = st.session_state.get("pn_fit")
            if _fit:
                st.markdown("#### 5️⃣ 자동 반사율 보정 결과")
                _q = _fit["quality"]
                _cal_qc = _fit.get("calibration_qc") or {}
                _qa, _qb, _qc, _qd, _qe = st.columns(5)
                _qa.metric("QC 등급", _cal_qc.get("status", "미평가"))
                _qb.metric("유효 밴드", f"{_pB - _q['invalid_band_count']} / {_pB}")
                _qc.metric("다중 패널 결합", f"{_q['blended_band_count']} 밴드")
                _qd.metric("낮은 패널 대체", f"{_q['fallback_band_count']} 밴드")
                _cv_value = _q.get("median_coefficient_cv")
                _qe.metric(
                    "패널 일치도(CV)",
                    "—" if _cv_value is None else f"{100 * _cv_value:.2f}%",
                    help="공통 유효 파장에서 패널별 보정계수가 얼마나 일치하는지 나타냅니다.",
                )
                _qc_reasons = list(_cal_qc.get("severe_reasons") or []) + list(
                    _cal_qc.get("review_reasons") or []
                )
                if _qc_reasons:
                    _reason_text = "\n".join(f"- {reason}" for reason in _qc_reasons)
                    if _cal_qc.get("status") == "FAIL":
                        st.error("QC에서 자동 적용을 차단한 이유:\n\n" + _reason_text)
                    else:
                        st.warning("QC 검토 항목:\n\n" + _reason_text)
                if _q["invalid_band_count"]:
                    st.warning(
                        f"⚠️ 어떤 패널에서도 신뢰할 신호가 없었던 "
                        f"{_q['invalid_band_count']}개 밴드는 NaN으로 저장되며 분석에서 제외됩니다."
                    )
                if _fit.get("dark_source_type") == "synthetic_constant":
                    st.warning(
                        "⚠️ 이 보정은 실측 Dark가 아니라 전 밴드 상수 DN "
                        f"{float(_fit.get('manual_dark_dn', 100.0)):,.1f}을 사용했습니다. "
                        "결과 파일에도 합성 Dark 사용 이력이 저장됩니다."
                    )
                if _cv_value is not None and _cv_value > 0.05:
                    st.warning(
                        "⚠️ 패널 간 보정계수 차이가 큽니다. 패널의 높이·각도·오염 또는 "
                        "조명 변화를 확인하세요."
                    )

                _wfig = go.Figure()
                _weights_view = np.asarray(_q["panel_weights"])
                for _index, _name in enumerate(_fit["names"]):
                    _wfig.add_trace(go.Scatter(
                        x=_xax, y=_weights_view[_index], mode="lines",
                        name=f"{_name} 사용 가중치",
                    ))
                _wfig.update_layout(
                    height=300, xaxis_title="Wavelength (nm)",
                    yaxis_title="자동 사용 가중치", yaxis_range=[-0.03, 1.03],
                    margin=dict(l=50, r=10, t=35, b=40),
                    legend=dict(orientation="h", y=1.18),
                    title="파장별 패널 결합 — 포화에 가까워질수록 가중치가 부드럽게 감소",
                )
                st.plotly_chart(_wfig, use_container_width=True, key="pn_weight_chart")

                if _fit.get("panel_predictions"):
                    _rfig = go.Figure()
                    for _index, (_name, _prediction, _target) in enumerate(zip(
                        _fit["names"],
                        _fit["panel_predictions"],
                        _fit["reflectances"],
                    )):
                        _rfig.add_trace(go.Scatter(
                            x=_xax,
                            y=_prediction,
                            mode="lines",
                            name=f"{_name} 보정 결과",
                        ))
                        _rfig.add_hline(
                            y=float(_target),
                            line_dash="dot",
                            line_color="gray",
                            annotation_text=f"목표 {float(_target):.3f}",
                        )
                    _rfig.update_layout(
                        height=340,
                        xaxis_title="Wavelength (nm)",
                        yaxis_title="재구성 반사율",
                        margin=dict(l=50, r=10, t=35, b=40),
                        legend=dict(orientation="h", y=1.18),
                        title="보정파일 자기점검 — 각 패널이 입력 반사율로 복원되는가",
                    )
                    st.plotly_chart(
                        _rfig, use_container_width=True, key="pn_reconstruction_chart"
                    )

                _cfig = go.Figure()
                _cfig.add_trace(go.Scatter(x=_xax, y=_fit["a"], mode="lines",
                                           name="a (기울기)", yaxis="y1"))
                _cfig.add_trace(go.Scatter(x=_xax, y=_fit["b"], mode="lines",
                                           name="b (절편)", yaxis="y2",
                                           line=dict(dash="dash")))
                _cfig.update_layout(
                    height=320, xaxis_title="Wavelength (nm)",
                    yaxis=dict(title="a"),
                    yaxis2=dict(title="b", overlaying="y", side="right"),
                    margin=dict(l=50, r=50, t=30, b=40),
                    legend=dict(orientation="h", y=1.15),
                    title="Dark 기준 가중 보정계수  (R = a·DN + b)",
                )
                st.plotly_chart(_cfig, use_container_width=True, key="pn_coef_chart")
                st.caption(
                    "b는 선택한 실측 또는 수동 Dark에서 계산됩니다. 샘플 스펙트럼을 억지로 "
                    "평활화하지 않고 패널 가중치만 부드럽게 전환합니다."
                )

                if st.session_state.get("pn_preview_rgb") is not None:
                    _pv1, _pv2 = st.columns(2)
                    with _pv1:
                        st.image(
                            st.session_state["pn_rgb"],
                            caption="원본 DN RGB 미리보기",
                            use_container_width=True,
                        )
                    with _pv2:
                        st.image(
                            st.session_state["pn_preview_rgb"],
                            caption="반사율 보정 RGB 미리보기",
                            use_container_width=True,
                        )
                    st.caption(
                        "RGB는 화면 확인을 위한 대비 스트레치입니다. 실제 저장 스펙트럼과 "
                        "BIL에는 계산된 반사율 값이 그대로 유지됩니다."
                    )

                if (_fit.get("calibration_qc") or {}).get("auto_apply_allowed"):
                    st.success(
                        "✅ 보정파일이 저장되어 전체 필드 분석에 연결되었습니다: "
                        f"`{Path(_fit['saved_path']).resolve()}`"
                    )
                else:
                    st.error(
                        "⛔ QC FAIL 파일은 점검 기록으로만 저장되었고 자동 연결되지 "
                        f"않았습니다: `{Path(_fit['saved_path']).resolve()}`"
                    )

                _export_factor = st.selectbox(
                    "반사율 BIL 공간 binning",
                    [1, 2, 4, 8],
                    index=2,
                    format_func=lambda value: (
                        "원본 해상도" if value == 1 else f"{value}×{value} binning"
                    ),
                    key="pn_export_bin",
                    help="대용량 현장 데이터는 4×4를 기본 권장합니다.",
                )
                _is_envi_source = Path(st.session_state["pn_file"]).suffix.lower() in {
                    ".hdr", ".bil", ".bip", ".bsq", ".raw", ".img", ".dat"
                }
                if st.button(
                    "💾 현재 영상 반사율 BIL 만들기",
                    type="primary",
                    use_container_width=True,
                    key="pn_export_reflectance",
                    disabled=not _is_envi_source,
                    help=(None if _is_envi_source else "현재는 ENVI/BIL 입력만 지원합니다."),
                ):
                    try:
                        import time as _export_time

                        _reflectance_dir = (
                            Path(output_dir or "./output").expanduser() / "reflectance"
                        )
                        _reflectance_dir.mkdir(parents=True, exist_ok=True)
                        _export_stem = Path(st.session_state["pn_file"]).stem
                        _export_path = _reflectance_dir / (
                            f"{_export_stem}_reflectance_bin{_export_factor}.bil"
                        )
                        if _export_path.exists() or _export_path.with_suffix(".hdr").exists():
                            _export_path = _reflectance_dir / (
                                f"{_export_stem}_reflectance_bin{_export_factor}_"
                                f"{_export_time.strftime('%Y%m%d_%H%M%S')}.bil"
                            )
                        with st.spinner("반사율 BIL을 행 단위로 저장 중..."):
                            _exported = _rad.export_calibrated_binned_envi(
                                st.session_state["pn_file"],
                                _fit["saved_path"],
                                _export_path,
                                bin_factor=int(_export_factor),
                            )
                        st.success(
                            "✅ 반사율 BIL 저장 완료: "
                            f"`{Path(_exported['data_file']).resolve()}`"
                        )
                    except Exception:
                        st.error("❌ 반사율 BIL 저장 실패")
                        st.code(traceback.format_exc(), language="python")

    st.divider()
    _wd_advanced_box = st.expander(
        "⚙️ 고급: 여러 측정시각의 White/Dark 프로파일 관리",
        expanded=False,
    )
    _wd_advanced_box.__enter__()
    st.markdown("### 🌗 White + 센서 Dark 보정 프로파일")
    st.caption(
        "White 원스펙트럼과 렌즈를 막거나 셔터를 닫아 획득한 센서 dark current를 "
        "함께 저장합니다. 프로파일 폴더를 분석에 지정하면 대상 파일의 촬영시각과 "
        "가장 가까운 White가 자동 선택됩니다."
    )
    st.warning(
        "여기서 Dark는 검은색 패널이 아니라 반드시 같은 센서 설정으로 획득한 "
        "센서 dark current 영상이어야 합니다."
    )

    _all_registered_panels = st.session_state.get("pn_panels") or []
    _saturated_white_panels = [
        panel for panel in _all_registered_panels
        if not (panel.get("saturation") or {}).get("usable", False)
    ]
    _available_panels = [
        panel for panel in _all_registered_panels
        if (panel.get("saturation") or {}).get("usable", False)
        and float(panel.get("reflectance", 0.0)) > 0
    ]
    if _saturated_white_panels:
        st.error(
            "⛔ White 후보에서 자동 제외된 포화 패널: "
            + ", ".join(panel["name"] for panel in _saturated_white_panels)
        )
    _white_modes = ["균일한 White 영상 전체 중앙값"]
    if _available_panels and st.session_state.get("pn_data") is not None:
        _white_modes.append("현재 이미지에서 등록한 White ROI")
    _wd_white_mode = st.radio(
        "White 스펙트럼 추출 방식", _white_modes, horizontal=True,
        key="wd_white_mode",
    )

    _wd_white_path = ""
    _wd_panel_index = 0
    _selected_white_panel = None
    if _wd_white_mode == "현재 이미지에서 등록한 White ROI":
        _wd_panel_index = st.selectbox(
            "White ROI", range(len(_available_panels)),
            format_func=lambda i: (
                f"{_available_panels[i]['name']} · "
                f"R={_available_panels[i]['reflectance']:.3f}"
            ),
            key="wd_panel_index",
        )
        _selected_white_panel = _available_panels[int(_wd_panel_index)]
        st.caption("등록 ROI의 픽셀별 중앙값을 다시 계산해 White 원스펙트럼으로 저장합니다.")
    else:
        _wd_white_path = st.text_input(
            "균일한 White reference 영상 경로",
            placeholder="D:/references/white_20260821_093000.vnir.hdr",
            key="wd_white_path",
            help="프레임 전체가 같은 White reference를 측정한 영상이어야 합니다.",
        )

    _wd_dark_path = st.text_input(
        "센서 dark current 영상 경로",
        placeholder="D:/references/dark_20260821_090000.vnir.hdr",
        key="wd_dark_path",
    )
    _wd1, _wd2, _wd3 = st.columns(3)
    with _wd1:
        if _selected_white_panel is not None:
            _wd_reflectance = float(_selected_white_panel["reflectance"])
            st.metric("White 반사율 (등록값 자동 사용)", f"{_wd_reflectance:.3f}")
        else:
            _wd_reflectance = st.number_input(
                "White 반사율", 0.01, 1.20, 0.99, 0.01,
                key="wd_white_reflectance",
            )
    with _wd2:
        _wd_ds = st.selectbox(
            "Reference 읽기 다운샘플", [1, 2, 4, 8], index=2,
            key="wd_reference_ds",
            help="균일 프레임의 중앙값만 계산하므로 4를 권장합니다.",
        )
    with _wd3:
        _wd_sensor = st.text_input(
            "센서 구분", value="VNIR", key="wd_sensor",
            help="예: VNIR, SWIR. 같은 센서 프로파일만 사용하세요.",
        )

    if _selected_white_panel is not None:
        st.info(
            f"선택한 패널은 R={_wd_reflectance:.3f}로 계산합니다. "
            "50% 패널을 99%로 간주하지 않으며, 이 등록값과 센서 dark를 사용해 "
            "대상 영상을 절대 반사율 척도로 변환합니다."
        )
        if _wd_reflectance < 0.90:
            st.warning(
                f"R={_wd_reflectance:.3f}보다 밝은 대상은 패널 범위 밖 외삽입니다. "
                "계산은 가능하지만, 가능하면 포화되지 않은 더 밝은 패널도 함께 "
                "측정해 선형성을 검증하세요."
            )

    _wd_time = st.text_input(
        "White 측정시각 (선택, 비우면 파일명에서 자동 추출)",
        placeholder="2026-08-21 09:30:00",
        key="wd_white_time",
    )
    _wd4, _wd5 = st.columns(2)
    with _wd4:
        _wd_integration = st.text_input(
            "Integration time (선택)", key="wd_integration_time"
        )
    with _wd5:
        _wd_gain = st.text_input("Gain (선택)", key="wd_gain")
    _wd_profile_dir = st.text_input(
        "프로파일 저장 폴더", value="./calibration_profiles",
        key="wd_profile_dir",
    )

    if st.button(
        "💾 White/Dark 프로파일 생성", type="primary",
        use_container_width=True, key="wd_profile_save",
    ):
        try:
            from src.data_loader import HyperspectralLoader as _WDLoader

            if not _wd_dark_path.strip():
                raise ValueError("센서 dark current 영상 경로를 입력하세요.")
            _wd_loader = _WDLoader({"spatial_downsample": int(_wd_ds)})

            if _wd_white_mode == "현재 이미지에서 등록한 White ROI":
                _selected_panel = _available_panels[int(_wd_panel_index)]
                _white_pixels, _, _ = _ru.region_pixels(
                    st.session_state["pn_data"],
                    _selected_panel.get(
                        "region", {"type": "box", "roi": _selected_panel["box"]}
                    ),
                )
                _white_spec = np.median(_white_pixels, axis=0).astype(np.float32)
                _white_wl = st.session_state.get("pn_wl")
                _white_source = st.session_state.get("pn_file", "")
                _white_qc = {
                    "sample_pixels": int(len(_white_pixels)),
                    "brightness_cv": float(
                        np.std(np.mean(_white_pixels, axis=1)) /
                        max(abs(np.mean(_white_pixels)), 1e-12)
                    ),
                    "roi": _selected_panel["box"],
                }
                _white_qc.update(
                    _rad.panel_saturation_metrics(
                        _white_pixels,
                        observed_max=st.session_state.get("pn_source_max"),
                    )
                )
            else:
                if not _wd_white_path.strip():
                    raise ValueError("균일한 White reference 영상 경로를 입력하세요.")
                _white_cube, _white_meta = _wd_loader.load_local(_wd_white_path.strip())
                _white_spec, _white_qc = _rad.robust_reference_spectrum(_white_cube)
                _white_wl = _white_meta.get("wavelengths")
                _white_source = _wd_white_path.strip()

            if not _white_qc.get("usable", False):
                raise ValueError(
                    "White reference에서 포화 밴드가 "
                    f"{_white_qc.get('saturated_band_count', 0)}개 검출되어 "
                    "프로파일에서 자동 제외했습니다. 노출을 낮춰 다시 측정하거나 "
                    "더 낮은 반사율 패널을 사용하세요."
                )

            _dark_cube, _dark_meta = _wd_loader.load_local(_wd_dark_path.strip())
            _dark_spec, _dark_qc = _rad.robust_reference_spectrum(_dark_cube)
            if not _dark_qc.get("usable", False):
                raise ValueError(
                    "센서 dark 영상에서 포화가 검출되었습니다. 올바른 dark current "
                    "파일인지와 센서 설정을 확인하세요."
                )
            _dark_wl = _dark_meta.get("wavelengths")
            if len(_white_spec) != len(_dark_spec):
                raise ValueError("White와 dark 영상의 밴드 수가 다릅니다.")
            if _white_wl is not None and _dark_wl is not None and not np.allclose(
                _white_wl, _dark_wl, rtol=0, atol=1.0
            ):
                raise ValueError("White와 dark 영상의 파장축이 다릅니다.")

            _white_dt = _rad.parse_acquisition_time(_wd_time or _white_source)
            if _white_dt is None:
                raise ValueError(
                    "White 측정시각을 파일명에서 찾지 못했습니다. 직접 입력하세요."
                )
            _dark_dt = _rad.parse_acquisition_time(_wd_dark_path.strip())
            _stamp = _white_dt.strftime("%Y%m%d_%H%M%S")
            _sensor_slug = re.sub(r"[^0-9A-Za-z_-]+", "_", _wd_sensor.strip()) or "sensor"
            _profile_path = Path(_wd_profile_dir).expanduser() / (
                f"white_dark_{_sensor_slug}_{_stamp}.npz"
            )
            if _profile_path.exists():
                raise FileExistsError(f"이미 존재하는 프로파일입니다: {_profile_path}")
            _saved = _rad.save_white_dark_profile(
                _profile_path, _white_spec, _dark_spec,
                wavelengths=(_white_wl if _white_wl is not None else _dark_wl),
                white_reflectance=float(_wd_reflectance),
                white_time=_white_dt,
                dark_time=_dark_dt,
                meta={
                    "sensor": _wd_sensor.strip(),
                    "integration_time": _wd_integration.strip(),
                    "gain": _wd_gain.strip(),
                    "white_source": str(_white_source),
                    "dark_source": _wd_dark_path.strip(),
                    "white_qc": _white_qc,
                    "dark_qc": _dark_qc,
                },
            )
            st.session_state["wd_last_profile"] = _saved
            st.success(f"✅ 프로파일 저장 완료: `{Path(_saved).resolve()}`")
        except Exception:
            st.error("❌ White/Dark 프로파일 생성 실패")
            st.code(traceback.format_exc(), language="python")

    if st.session_state.get("wd_last_profile"):
        try:
            _wd_loaded = _rad.load_white_dark_profile(st.session_state["wd_last_profile"])
            _wd_x = _wd_loaded.get("wavelengths") or list(range(len(_wd_loaded["white"])))
            _wd_fig = go.Figure()
            _wd_fig.add_trace(go.Scatter(x=_wd_x, y=_wd_loaded["white"], name="White DN"))
            _wd_fig.add_trace(go.Scatter(x=_wd_x, y=_wd_loaded["dark"], name="Sensor dark DN"))
            _wd_fig.add_trace(go.Scatter(
                x=_wd_x, y=_wd_loaded["white"] - _wd_loaded["dark"],
                name="White - Dark", line=dict(dash="dash"),
            ))
            _wd_fig.update_layout(
                height=330, xaxis_title="Wavelength (nm)", yaxis_title="DN",
                title="저장된 White/Dark 보정 프로파일",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(_wd_fig, use_container_width=True, key="wd_profile_plot")
        except Exception:
            st.warning("마지막 White/Dark 프로파일 미리보기를 불러오지 못했습니다.")

    _wd_advanced_box.__exit__(None, None, None)


# ============================================================
# Tab 4 – Pixel labeling tool
# ============================================================

with tab_label:
    st.markdown("### 🏷️ 픽셀 라벨링 도구")
    st.caption(
        "초분광 이미지를 열고 픽셀을 클릭해 클래스 라벨을 지정합니다.  "
        "저장된 CSV를 **분석 실행** 탭의 라벨 CSV 경로에 입력하면 "
        "지도학습(Random Forest / 1D-CNN)에 바로 사용할 수 있습니다."
    )

    # ── Session-state defaults ────────────────────────────────
    _lbl_defaults: dict = {
        "lbl_data":          None,   # ndarray (H, W, B)
        "lbl_wl":            None,   # wavelength list
        "lbl_rgb":           None,   # ndarray (H, W, 3) uint8
        "lbl_file":          "",     # loaded file path string
        "lbl_rows":          [],     # [(row, col, class_id), ...]
        "lbl_prev_sel":      None,   # (row, col) last processed click
        "lbl_n_classes":     5,      # number of classes
        "lbl_active_cls":    0,      # currently selected class id
        "lbl_file_list":     [],     # files found when a directory is entered
        "lbl_dir_input":     "",     # last directory path entered
    }
    for k, v in _lbl_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Step 1: Load file ─────────────────────────────────────
    st.markdown("#### 1️⃣ 파일 로드")

    lcol1, lcol2 = st.columns([5, 1])
    with lcol1:
        lbl_file_input = st.text_input(
            "파일 또는 폴더 경로",
            value=st.session_state["lbl_file"] or st.session_state["lbl_dir_input"],
            placeholder="./data/image.hdr  또는  ./data  (폴더 입력 → 파일 목록 표시)",
            label_visibility="collapsed",
        )
    with lcol2:
        load_btn = st.button("📂 로드", use_container_width=True)

    if load_btn and lbl_file_input:
        _inp_path = Path(lbl_file_input)
        if _inp_path.is_dir():
            # ── Directory: scan for supported files ──────────
            _found = sorted([
                f for f in _inp_path.iterdir()
                if f.suffix.lower() in _LBL_SUPPORTED_EXTS
            ])
            if not _found:
                st.error(
                    f"❌ `{lbl_file_input}` 폴더에 지원 형식 파일이 없습니다.  \n"
                    f"지원 형식: {', '.join(sorted(_LBL_SUPPORTED_EXTS))}"
                )
                st.session_state["lbl_file_list"] = []
            else:
                st.session_state["lbl_file_list"] = [str(f) for f in _found]
                st.session_state["lbl_dir_input"] = lbl_file_input
        else:
            # ── Single file: load directly ───────────────────
            with st.spinner("파일 로딩 중..."):
                try:
                    _H, _W, _B = _do_load_file(lbl_file_input)
                    st.success(
                        f"✅ 로드 완료  |  {_H} × {_W} px  |  {_B} 밴드  "
                        f"|  {Path(lbl_file_input).name}"
                    )
                except Exception:
                    st.error("❌ 파일 로드 실패")
                    st.code(traceback.format_exc(), language="python")

    # ── File selector (shown after directory scan) ────────────
    if st.session_state["lbl_file_list"]:
        _file_list = st.session_state["lbl_file_list"]
        st.info(
            f"📁 **{len(_file_list)}개** 파일을 찾았습니다. "
            f"파일을 선택한 후 [✅ 로드] 버튼을 클릭하세요."
        )
        _fsel_c1, _fsel_c2 = st.columns([5, 1])
        with _fsel_c1:
            _sel_file = st.selectbox(
                "파일 선택",
                _file_list,
                format_func=lambda p: Path(p).name,
                label_visibility="collapsed",
                key="lbl_selectbox_file",
            )
        with _fsel_c2:
            if st.button(
                "✅ 로드", type="primary",
                use_container_width=True, key="lbl_load_sel_btn"
            ):
                with st.spinner("파일 로딩 중..."):
                    try:
                        _H, _W, _B = _do_load_file(_sel_file)
                        st.success(
                            f"✅ 로드 완료  |  {_H} × {_W} px  |  {_B} 밴드  "
                            f"|  {Path(_sel_file).name}"
                        )
                        st.rerun()
                    except Exception:
                        st.error("❌ 파일 로드 실패")
                        st.code(traceback.format_exc(), language="python")

    # ── Guard: nothing loaded yet ─────────────────────────────
    if st.session_state["lbl_data"] is None:
        if not st.session_state["lbl_file_list"]:
            st.info("⬆️ 초분광 파일 경로 또는 폴더를 입력하고 [📂 로드] 버튼을 클릭하세요.")

    else:
        # ── Step 2: Class configuration ───────────────────────
        st.divider()
        st.markdown("#### 2️⃣ 클래스 설정")

        with st.expander("클래스 수 / 이름 / 색상 편집", expanded=False):
            _n_new = st.number_input(
                "클래스 수", min_value=1, max_value=20,
                value=int(st.session_state["lbl_n_classes"]),
                step=1, key="lbl_n_classes_widget",
            )
            if int(_n_new) != st.session_state["lbl_n_classes"]:
                st.session_state["lbl_n_classes"] = int(_n_new)

            _n_cls = int(st.session_state["lbl_n_classes"])

            # Pre-initialise per-class widget keys (must happen before rendering)
            for _i in range(_n_cls):
                if f"lbl_cls_name_{_i}" not in st.session_state:
                    st.session_state[f"lbl_cls_name_{_i}"] = f"Class {_i}"
                if f"lbl_cls_color_{_i}" not in st.session_state:
                    st.session_state[f"lbl_cls_color_{_i}"] = (
                        _DEFAULT_COLORS[_i % len(_DEFAULT_COLORS)]
                    )

            _gcols = st.columns(min(_n_cls, 5))
            for _i in range(_n_cls):
                with _gcols[_i % min(_n_cls, 5)]:
                    st.text_input(
                        f"ID {_i}",
                        key=f"lbl_cls_name_{_i}",
                    )
                    st.color_picker(
                        "●",
                        key=f"lbl_cls_color_{_i}",
                        label_visibility="collapsed",
                    )

        # Build cls_cfg list from widget session state
        _n_cls = int(st.session_state["lbl_n_classes"])
        cls_cfg = [
            {
                "id":    _i,
                "name":  st.session_state.get(f"lbl_cls_name_{_i}",  f"Class {_i}"),
                "color": st.session_state.get(f"lbl_cls_color_{_i}", _DEFAULT_COLORS[_i % len(_DEFAULT_COLORS)]),
            }
            for _i in range(_n_cls)
        ]

        # ── Step 3: Interactive image labeling ────────────────
        st.divider()
        st.markdown("#### 3️⃣ 이미지 클릭 → 라벨 추가")
        st.caption(
            "이미지 위를 클릭하면 해당 픽셀 좌표가 선택한 클래스로 라벨링됩니다.  "
            "드래그·확대/축소는 좌상단 Plotly 툴바에서 **Pan** 모드로 전환 후 사용하세요."
        )

        img_col, ctrl_col = st.columns([3, 1])

        # ── Right column: class selector + counters + buttons ─
        with ctrl_col:
            st.markdown("**클래스 선택**")
            _active_idx = min(
                int(st.session_state.get("lbl_active_cls", 0)), _n_cls - 1
            )
            active_cls = st.radio(
                "현재 클래스",
                options=list(range(_n_cls)),
                format_func=lambda i: f"  {cls_cfg[i]['name']}",
                index=_active_idx,
                key="lbl_cls_radio",
                label_visibility="collapsed",
            )
            st.session_state["lbl_active_cls"] = active_cls

            st.divider()

            _total = len(st.session_state["lbl_rows"])
            st.metric("총 라벨 수", _total)
            _cnt = Counter(r[2] for r in st.session_state["lbl_rows"])
            for _c in cls_cfg:
                st.caption(f"● {_c['name']}: **{_cnt.get(_c['id'], 0)}**")

            st.divider()

            if st.button("↩️ 마지막 취소", use_container_width=True):
                if st.session_state["lbl_rows"]:
                    st.session_state["lbl_rows"].pop()
                    st.session_state["lbl_prev_sel"] = None
                    st.rerun()

            if st.button("🗑️ 전체 초기화", use_container_width=True, type="secondary"):
                st.session_state["lbl_rows"]     = []
                st.session_state["lbl_prev_sel"] = None
                st.rerun()

        # ── Left column: plotly figure ────────────────────────
        with img_col:
            _rgb_arr = st.session_state["lbl_rgb"]
            _fig     = _build_label_figure(
                _rgb_arr, st.session_state["lbl_rows"], cls_cfg
            )

            _event = st.plotly_chart(
                _fig,
                key="lbl_chart",
                on_select="rerun",
                selection_mode=("points",),
                use_container_width=True,
            )

            # ── Process click event ──────────────────────────
            if (
                _event is not None
                and hasattr(_event, "selection")
                and _event.selection.points
            ):
                _pt     = _event.selection.points[0]
                _col_px = int(round(float(_pt.get("x", 0))))
                _row_px = int(round(float(_pt.get("y", 0))))

                _H_img, _W_img = _rgb_arr.shape[:2]
                _col_px = max(0, min(_col_px, _W_img - 1))
                _row_px = max(0, min(_row_px, _H_img - 1))

                _new_sel = (_row_px, _col_px)
                if _new_sel != st.session_state.get("lbl_prev_sel"):
                    st.session_state["lbl_prev_sel"] = _new_sel
                    st.session_state["lbl_rows"].append(
                        (_row_px, _col_px, active_cls)
                    )
                    st.rerun()

        # ── Step 4: Labels table ──────────────────────────────
        if st.session_state["lbl_rows"]:
            st.divider()
            _n_lbl = len(st.session_state["lbl_rows"])
            st.markdown(f"#### 4️⃣ 라벨 목록  ({_n_lbl}개)")

            _cls_name_map = {c["id"]: c["name"] for c in cls_cfg}
            _df_lbl = pd.DataFrame([
                {
                    "row":        r,
                    "col":        c,
                    "class_id":   cid,
                    "class_name": _cls_name_map.get(cid, f"Class {cid}"),
                }
                for r, c, cid in st.session_state["lbl_rows"]
            ])
            st.dataframe(_df_lbl, use_container_width=True, height=220)

        # ── Step 5: Save CSV ──────────────────────────────────
        st.divider()
        st.markdown("#### 5️⃣ CSV 저장")
        st.caption("저장 형식: `row,col,class_id` (헤더 없음) — 지도학습 입력 형식과 동일")

        _default_csv = (
            str(Path(st.session_state["lbl_file"]).parent / "labels.csv")
            if st.session_state["lbl_file"]
            else "labels.csv"
        )
        scol1, scol2 = st.columns([5, 1])
        with scol1:
            save_path = st.text_input(
                "저장 경로",
                value=_default_csv,
                key="lbl_save_path",
                label_visibility="collapsed",
            )
        with scol2:
            save_btn = st.button("💾 저장", use_container_width=True, type="primary")

        if save_btn:
            if not st.session_state["lbl_rows"]:
                st.warning("저장할 라벨이 없습니다. 먼저 이미지를 클릭해 라벨을 추가하세요.")
            else:
                try:
                    _sp = Path(save_path)
                    _sp.parent.mkdir(parents=True, exist_ok=True)
                    _df_save = pd.DataFrame(
                        [(r, c, cid) for r, c, cid in st.session_state["lbl_rows"]],
                        columns=["row", "col", "class_id"],
                    )
                    _df_save.to_csv(_sp, index=False, header=False)
                    st.success(
                        f"✅ **{len(st.session_state['lbl_rows'])}개** 라벨 저장 완료  \n"
                        f"`{_sp.resolve()}`"
                    )
                    st.info(
                        "💡 **다음 단계**: [분석 실행] 탭 → 라벨 CSV 경로에 위 경로 입력 "
                        "→ **Random Forest** 또는 **1D-CNN**으로 분석 시작"
                    )
                except Exception as e:
                    st.error(f"저장 실패: {e}")


# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "CanopySpectra · "
    "방법: hybrid | kmeans | sam | supervised | autoencoder | cnn"
)
