"""
app.py
------
Streamlit GUI for the Hyperspectral Field Crop Analysis Pipeline.

Run with:
    python -m streamlit run app.py
"""

import logging
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Make sure 'src' package is importable from this directory
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="초분광 작물 분석",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom log handler – collects records for display
# ============================================================

class _ListLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


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
# Sidebar – Settings (pipeline run tab)
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ 분석 설정")

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
        local_folder = st.text_input(
            "폴더 경로",
            value="./data",
            placeholder="C:/data/field_images",
        )
    else:
        github_repo   = st.text_input("저장소 (owner/repo)",    placeholder="username/repo")
        github_folder = st.text_input("서브폴더",               value="", placeholder="data/2024")
        github_token  = st.text_input("GitHub 토큰 (비공개용)", type="password")

    st.markdown("---")

    # ── Classification method ────────────────────────────────
    st.markdown("### 🧬 분류 방법")

    method = st.selectbox(
        "방법",
        list(METHODS.keys()),
        format_func=lambda k: METHODS[k]["label"],
        label_visibility="collapsed",
    )

    kind          = METHODS[method]["kind"]
    needs_labels  = kind == "지도"
    needs_pytorch = method in ("autoencoder", "cnn")

    st.caption(
        f"{KIND_COLOR.get(kind, '')} {kind} "
        + ("| 🔥 PyTorch 필요" if needs_pytorch else "")
    )

    st.markdown("---")

    # ── Number of classes ────────────────────────────────────
    st.markdown("### 🔢 클래스 수")

    if method == "supervised":
        st.caption("라벨 CSV의 클래스 수를 자동으로 사용합니다.")
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

    # ── Output / misc ────────────────────────────────────────
    st.markdown("### 📁 출력")
    output_dir = st.text_input("출력 폴더", value="./output")
    file_limit = st.number_input(
        "파일 수 제한 (0 = 전체)", min_value=0, value=0, step=1,
        help="테스트 시 1~2로 제한하면 빠르게 확인할 수 있습니다.",
    )
    verbose = st.checkbox("상세 로그 (DEBUG)", value=False)

    st.markdown("---")
    run_btn = st.button("🚀  분석 시작", type="primary", use_container_width=True)


# ============================================================
# Main area
# ============================================================

st.markdown("# 🌿 초분광 작물 분석 파이프라인")
st.caption("Hyperspectral Field Crop Analysis · 자동 스펙트럼 추출 시스템")

tab_run, tab_label = st.tabs(["🚀 분석 실행", "🏷️ 픽셀 라벨링"])

# ============================================================
# Tab 1 – Run pipeline
# ============================================================

with tab_run:
    # ── Info cards ─────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        src_info   = f"`{local_folder}`" if data_src == "로컬 폴더" else f"`{github_repo}`"
        limit_info = str(int(file_limit)) + " 파일" if file_limit else "전체"
        cls_info   = f"{n_classes} 클래스" if n_classes else "라벨 CSV 기준"
        st.info(
            f"**방법:** {METHODS[method]['label']}  \n"
            f"**데이터:** {src_info}  \n"
            f"**클래스:** {cls_info}  \n"
            f"**파일:** {limit_info}  ·  **출력:** `{output_dir}`"
        )

    with col_right:
        st.success(METHODS[method]["help"])

    st.markdown("---")

    # ── Run ────────────────────────────────────────────────────
    if run_btn:

        # Validate inputs
        errors = []
        if data_src == "로컬 폴더" and not local_folder:
            errors.append("로컬 폴더 경로를 입력해 주세요.")
        if data_src == "GitHub 저장소" and not github_repo:
            errors.append("GitHub 저장소를 입력해 주세요.")
        if needs_labels and not labels_csv:
            errors.append(f"{method} 방법은 라벨 CSV가 필요합니다.")

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
                "normalize":          True,
                "remove_bad_bands":   True,
                "bad_band_ranges":    [[1340, 1460], [1790, 1960]],
                "smooth_spectra":     False,
                "spatial_downsample": 1,
            },
            "classification": {
                "method": method,
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
            },
            "output": {
                "dir":                     output_dir,
                "save_classification_map": True,
                "save_spectra_csv":        True,
                "save_report":             True,
            },
            "report": {
                "title":            "Hyperspectral Field Crop Analysis",
                "spectra_show_std": True,
            },
        }

        # Attach log handler
        log_handler = _ListLogHandler()
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                              datefmt="%H:%M:%S")
        )
        root_log = logging.getLogger()
        root_log.addHandler(log_handler)
        root_log.setLevel(logging.DEBUG if verbose else logging.INFO)

        # Execute pipeline
        pipeline_ok = False
        try:
            with st.spinner("⏳ 분석 중...  (데이터 크기에 따라 수 분이 걸릴 수 있습니다)"):
                from src.pipeline import Pipeline
                pipeline = Pipeline(cfg)
                pipeline.run(
                    labels_csv=labels_csv if labels_csv else None,
                    file_limit=int(file_limit) if file_limit else None,
                )
            pipeline_ok = True

        except Exception:
            st.error("❌ 파이프라인 오류가 발생했습니다.")
            st.code(traceback.format_exc(), language="python")

        finally:
            root_log.removeHandler(log_handler)

        # Results
        if pipeline_ok:
            st.success("✅ 분석 완료!")

            out_p = Path(output_dir)

            # Find the most recently written report_*.html
            reports = sorted(
                out_p.glob("report*.html"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if reports:
                st.markdown(
                    f"📄 **HTML 리포트:** `{reports[0].resolve()}`  \n"
                    f"브라우저에서 직접 파일을 열어 확인하세요."
                )

            # Class-map previews
            class_maps = sorted(out_p.rglob("class_map.png"))
            if class_maps:
                st.markdown("### 🗺️ 분류 맵 미리보기")
                n_cols = min(len(class_maps), 3)
                cols   = st.columns(n_cols)
                for col, img_path in zip(cols, class_maps[:3]):
                    with col:
                        st.image(
                            str(img_path),
                            caption=img_path.parent.name,
                            use_container_width=True,
                        )
                if len(class_maps) > 3:
                    st.caption(f"… 및 {len(class_maps) - 3}개 파일 더 (리포트에서 전체 확인)")

        # Log viewer
        if log_handler.lines:
            with st.expander(
                f"📋 실행 로그  ({len(log_handler.lines)} 줄)",
                expanded=not pipeline_ok,
            ):
                st.code("\n".join(log_handler.lines), language="text")


# ============================================================
# Tab 2 – Pixel labeling tool
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
    "HyperspectralPipeline · "
    "방법: hybrid | kmeans | sam | supervised | autoencoder | cnn"
)
