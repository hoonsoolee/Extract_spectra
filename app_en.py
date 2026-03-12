"""
app_en.py
---------
Streamlit GUI for the Hyperspectral Field Crop Analysis Pipeline  (English UI).

Run with:
    python -m streamlit run app_en.py
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
    page_title="Hyperspectral Crop Analysis",
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
        "label":  "🌿 Hybrid  (NDVI + Brightness + K-means)",
        "kind":   "Unsupervised",
        "help":   (
            "NDVI detects vegetation → brightness separates shadows → K-means refinement.  \n"
            "**Recommended default** — no labels required."
        ),
    },
    "kmeans": {
        "label":  "📊 K-Means  (Unsupervised)",
        "kind":   "Unsupervised",
        "help":   (
            "PCA dimensionality reduction → K-means clustering.  \n"
            "Good for exploratory analysis when no labels are available."
        ),
    },
    "sam": {
        "label":  "📐 SAM  (Spectral Angle Mapping)",
        "kind":   "Unsupervised / Supervised",
        "help":   (
            "Compares only the **angle** between spectral vectors → unaffected by illumination.  \n"
            "Works with or without labels."
        ),
    },
    "supervised": {
        "label":  "🎯 Random Forest  (Supervised)",
        "kind":   "Supervised",
        "help":   (
            "Trains a Random Forest classifier on user-provided labels (CSV) → classifies all pixels.  \n"
            "Requires a labels CSV."
        ),
    },
    "autoencoder": {
        "label":  "🤖 Autoencoder  (Deep Learning Unsupervised)",
        "kind":   "Unsupervised",
        "help":   (
            "MLP autoencoder compresses spectra → K-means in latent space.  \n"
            "Requires PyTorch · no labels needed."
        ),
    },
    "cnn": {
        "label":  "🧠 1D-CNN  (Deep Learning Supervised)",
        "kind":   "Supervised",
        "help":   (
            "1D convolutional neural network pixel classifier.  \n"
            "Highest accuracy when sufficient labels are available.  \n"
            "Requires labels CSV + PyTorch."
        ),
    },
}

KIND_COLOR = {"Unsupervised": "🟢", "Supervised": "🔵", "Unsupervised / Supervised": "🟡"}

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
            color="rgba(0,0,0,0.01)",
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

# ============================================================
# Sidebar – Settings (pipeline run tab)
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")

    # ── Data source ─────────────────────────────────────────
    st.markdown("### 📂 Data Source")
    data_src = st.radio(
        "Source",
        ["Local Folder", "GitHub Repository"],
        horizontal=True,
        label_visibility="collapsed",
    )

    local_folder = github_repo = github_folder = github_token = ""

    if data_src == "Local Folder":
        local_folder = st.text_input(
            "Folder path",
            value="./data",
            placeholder="C:/data/field_images",
        )
    else:
        github_repo   = st.text_input("Repository (owner/repo)", placeholder="username/repo")
        github_folder = st.text_input("Sub-folder",              value="", placeholder="data/2024")
        github_token  = st.text_input("GitHub Token (private repos)", type="password")

    st.markdown("---")

    # ── Processing mode ───────────────────────────────────────
    st.markdown("### 🎯 Processing Mode")
    run_mode = st.radio(
        "Processing mode",
        ["🔍 Single File", "📦 Batch (all files)"],
        horizontal=False,
        label_visibility="collapsed",
        key="run_mode_radio",
    )

    _run_single_file = None

    if run_mode == "🔍 Single File":
        if data_src == "Local Folder" and local_folder:
            if st.button("📂 Scan Folder", use_container_width=True, key="run_scan_btn"):
                _sp = Path(local_folder)
                if _sp.is_dir():
                    _exts = {".hdr", ".tif", ".tiff", ".h5", ".hdf5", ".mat"}
                    _sf = sorted([f for f in _sp.rglob("*")
                                  if f.suffix.lower() in _exts])
                    _hdr_stems = {f.stem for f in _sf if f.suffix.lower() == ".hdr"}
                    _sf = [f for f in _sf
                           if not (f.suffix.lower() in {".raw", ".bil", ".bip", ".bsq"}
                                   and f.stem in _hdr_stems)]
                    st.session_state["run_scan_files"] = [str(f) for f in sorted(set(_sf))]
                    if not st.session_state["run_scan_files"]:
                        st.warning("No supported files found in this folder.")
                else:
                    st.warning("Please enter a valid folder path.")
                    st.session_state["run_scan_files"] = []

            if st.session_state["run_scan_files"]:
                _run_single_file = st.selectbox(
                    "File to process",
                    st.session_state["run_scan_files"],
                    format_func=lambda p: Path(p).name,
                    key="run_file_select",
                )
                st.caption(f"📄 {Path(_run_single_file).name}")
            else:
                st.caption("📂 Scan to select a file.")
        else:
            st.caption("Available in Local Folder mode only.")
    else:
        if st.session_state["run_scan_files"]:
            st.session_state["run_scan_files"] = []
        st.caption("📋 Processes all files sequentially and saves one report per file.")

    st.markdown("---")

    # ── Classification method ────────────────────────────────
    st.markdown("### 🧬 Classification Method")

    method = st.selectbox(
        "Method",
        list(METHODS.keys()),
        format_func=lambda k: METHODS[k]["label"],
        label_visibility="collapsed",
    )

    kind          = METHODS[method]["kind"]
    needs_labels  = kind == "Supervised"
    needs_pytorch = method in ("autoencoder", "cnn")

    st.caption(
        f"{KIND_COLOR.get(kind, '')} {kind} "
        + ("| 🔥 PyTorch required" if needs_pytorch else "")
    )

    st.markdown("---")

    # ── Number of classes ────────────────────────────────────
    st.markdown("### 🔢 Number of Classes")

    if method == "supervised":
        st.caption("Class count is inferred automatically from the labels CSV.")
        n_classes = 0
    else:
        n_classes = st.slider(
            "Clusters / Classes",
            min_value=2, max_value=20, value=6,
            label_visibility="collapsed",
        )

    st.markdown("---")

    # ── Method-specific params ───────────────────────────────
    st.markdown("### 🔧 Parameters")

    ndvi_threshold       = 0.15
    brightness_threshold = 0.08
    if method == "hybrid":
        with st.expander("Hybrid Settings", expanded=True):
            ndvi_threshold = st.slider(
                "NDVI threshold (vegetation)", 0.0, 1.0, 0.15, 0.01,
                help="Pixels with NDVI ≥ this value are classified as vegetation.",
            )
            brightness_threshold = st.slider(
                "Brightness threshold (shadow)", 0.0, 0.5, 0.08, 0.01,
                help="Pixels with mean reflectance below this are classified as shadow.",
            )

    angle_threshold = 0.10
    if method == "sam":
        with st.expander("SAM Settings", expanded=True):
            angle_threshold = st.slider(
                "Angle threshold (radians, 0 = no limit)", 0.0, 0.5, 0.10, 0.01,
                help=(
                    "Pixels whose angle to the nearest endmember exceeds this value "
                    "are assigned to Background (0).\n"
                    f"Current value ≈ {round(angle_threshold * 57.3, 1)}°"
                ),
            )

    ae_epochs  = 60
    cnn_epochs = 100
    if method == "autoencoder":
        with st.expander("Autoencoder Settings", expanded=False):
            ae_epochs = st.slider("Training epochs", 10, 200, 60, 10)
    if method == "cnn":
        with st.expander("CNN Settings", expanded=False):
            cnn_epochs = st.slider("Training epochs", 10, 200, 100, 10)

    labels_csv = ""
    if needs_labels or method == "sam":
        st.markdown("---")
        lbl_header = "Labels CSV" if needs_labels else "Labels CSV (optional – SAM supervised mode)"
        st.markdown(f"### 📋 {lbl_header}")
        labels_csv = st.text_input(
            "Path (row, col, class_id)",
            placeholder="labels.csv",
            label_visibility="collapsed",
        )
        if needs_labels and not labels_csv:
            st.warning("⚠️ This method requires a labels CSV.")

    st.markdown("---")

    # ── Output / misc ────────────────────────────────────────
    st.markdown("### 📁 Output")
    output_dir = st.text_input("Output folder", value="./output")
    file_limit = st.number_input(
        "File limit (0 = all)", min_value=0, value=0, step=1,
        help="Set to 1–2 for a quick test run.",
    )
    verbose = st.checkbox("Verbose logging (DEBUG)", value=False)

    st.markdown("---")
    run_btn = st.button("🚀  Run Analysis", type="primary", use_container_width=True)


# ============================================================
# Main area
# ============================================================

st.markdown("# 🌿 Hyperspectral Field Crop Analysis Pipeline")
st.caption("Hyperspectral Field Crop Analysis · Automatic Spectrum Extraction System")

tab_run, tab_label = st.tabs(["🚀 Run Analysis", "🏷️ Pixel Labeling"])

# ============================================================
# Tab 1 – Run pipeline
# ============================================================

with tab_run:
    # ── Info cards ─────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        src_info   = f"`{local_folder}`" if data_src == "Local Folder" else f"`{github_repo}`"
        limit_info = str(int(file_limit)) + " file(s)" if file_limit else "all"
        cls_info   = f"{n_classes} classes" if n_classes else "inferred from labels CSV"
        st.info(
            f"**Method:** {METHODS[method]['label']}  \n"
            f"**Data:** {src_info}  \n"
            f"**Classes:** {cls_info}  \n"
            f"**Files:** {limit_info}  ·  **Output:** `{output_dir}`"
        )

    with col_right:
        st.success(METHODS[method]["help"])

    st.markdown("---")

    # ── Run ────────────────────────────────────────────────────
    if run_btn:

        # Validate inputs
        errors = []
        if data_src == "Local Folder" and not local_folder:
            errors.append("Please enter a local folder path.")
        if data_src == "GitHub Repository" and not github_repo:
            errors.append("Please enter a GitHub repository.")
        if needs_labels and not labels_csv:
            errors.append(f"Method '{method}' requires a labels CSV.")
        if run_mode == "🔍 Single File" and not _run_single_file:
            errors.append("Single File mode: scan the folder and select a file first.")

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
                "per_file_report":         run_mode == "📦 Batch (all files)",
            },
            "report": {
                "title":            "Hyperspectral Field Crop Analysis",
                "spectra_show_std": True,
                "lang":             "en",
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
        import time as _time
        pipeline_ok  = False
        _elapsed_sec = 0.0
        try:
            with st.spinner("⏳ Analysing…  (may take several minutes for large images)"):
                from src.pipeline import Pipeline
                _t_start = _time.perf_counter()
                pipeline = Pipeline(cfg)
                pipeline.run(
                    labels_csv=labels_csv if labels_csv else None,
                    file_limit=int(file_limit) if file_limit else None,
                    single_file=_run_single_file,
                )
                _elapsed_sec = _time.perf_counter() - _t_start
            pipeline_ok = True

        except Exception:
            st.error("❌ Pipeline error.")
            st.code(traceback.format_exc(), language="python")

        finally:
            root_log.removeHandler(log_handler)

        # Results
        if pipeline_ok:
            _em, _es = divmod(int(_elapsed_sec), 60)
            _elapsed_str = f"{_em}m {_es:02d}s" if _em else f"{_es}s"
            st.success(f"✅ Analysis complete!  ⏱ Total time: **{_elapsed_str}**")

            out_p = Path(output_dir)

            # Find all report_*.html files (root + per-file subdirectories)
            reports = sorted(
                out_p.rglob("report*.html"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if reports:
                if run_mode == "📦 Batch (all files)":
                    st.markdown(f"📄 **HTML Reports ({len(reports)}):** (newest first)")
                    for _rp in reports:
                        st.caption(f"  `{_rp.resolve()}`")
                    st.caption("Open any file directly in your browser to view it.")
                else:
                    st.markdown(
                        f"📄 **HTML Report:** `{reports[0].resolve()}`  \n"
                        f"Open the file directly in your browser to view it."
                    )

            # Class-map previews
            class_maps = sorted(out_p.rglob("class_map.png"))
            if class_maps:
                st.markdown("### 🗺️ Classification Map Preview")
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
                    st.caption(f"… and {len(class_maps) - 3} more file(s) — see full report")

        # Log viewer
        if log_handler.lines:
            with st.expander(
                f"📋 Run Log  ({len(log_handler.lines)} lines)",
                expanded=not pipeline_ok,
            ):
                st.code("\n".join(log_handler.lines), language="text")


# ============================================================
# Tab 2 – Pixel labeling tool
# ============================================================

with tab_label:
    st.markdown("### 🏷️ Pixel Labeling Tool")
    st.caption(
        "Open a hyperspectral image and click pixels to assign class labels.  "
        "Save the resulting CSV and use it as the Labels CSV in the **Run Analysis** tab "
        "to train a supervised classifier (Random Forest / 1D-CNN)."
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
    st.markdown("#### 1️⃣ Load File")

    lcol1, lcol2 = st.columns([5, 1])
    with lcol1:
        lbl_file_input = st.text_input(
            "File or folder path",
            value=st.session_state["lbl_file"] or st.session_state["lbl_dir_input"],
            placeholder="./data/image.hdr  or  ./data  (folder → file list shown)",
            label_visibility="collapsed",
        )
    with lcol2:
        load_btn = st.button("📂 Load", use_container_width=True)

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
                    f"❌ No supported files found in `{lbl_file_input}`.  \n"
                    f"Supported formats: {', '.join(sorted(_LBL_SUPPORTED_EXTS))}"
                )
                st.session_state["lbl_file_list"] = []
            else:
                st.session_state["lbl_file_list"] = [str(f) for f in _found]
                st.session_state["lbl_dir_input"] = lbl_file_input
        else:
            # ── Single file: load directly ───────────────────
            with st.spinner("Loading file…"):
                try:
                    _H, _W, _B = _do_load_file(lbl_file_input)
                    st.success(
                        f"✅ Loaded  |  {_H} × {_W} px  |  {_B} bands  "
                        f"|  {Path(lbl_file_input).name}"
                    )
                except Exception:
                    st.error("❌ Failed to load file")
                    st.code(traceback.format_exc(), language="python")

    # ── File selector (shown after directory scan) ────────────
    if st.session_state["lbl_file_list"]:
        _file_list = st.session_state["lbl_file_list"]
        st.info(
            f"📁 Found **{len(_file_list)}** file(s). "
            f"Select a file and click [✅ Load]."
        )
        _fsel_c1, _fsel_c2 = st.columns([5, 1])
        with _fsel_c1:
            _sel_file = st.selectbox(
                "Select file",
                _file_list,
                format_func=lambda p: Path(p).name,
                label_visibility="collapsed",
                key="lbl_selectbox_file",
            )
        with _fsel_c2:
            if st.button(
                "✅ Load", type="primary",
                use_container_width=True, key="lbl_load_sel_btn"
            ):
                with st.spinner("Loading file…"):
                    try:
                        _H, _W, _B = _do_load_file(_sel_file)
                        st.success(
                            f"✅ Loaded  |  {_H} × {_W} px  |  {_B} bands  "
                            f"|  {Path(_sel_file).name}"
                        )
                        st.rerun()
                    except Exception:
                        st.error("❌ Failed to load file")
                        st.code(traceback.format_exc(), language="python")

    # ── Guard: nothing loaded yet ─────────────────────────────
    if st.session_state["lbl_data"] is None:
        if not st.session_state["lbl_file_list"]:
            st.info("⬆️ Enter a file path or folder and click [📂 Load].")

    else:
        # ── Step 2: Class configuration ───────────────────────
        st.divider()
        st.markdown("#### 2️⃣ Class Configuration")

        with st.expander("Edit class count / names / colours", expanded=False):
            _n_new = st.number_input(
                "Number of classes", min_value=1, max_value=20,
                value=int(st.session_state["lbl_n_classes"]),
                step=1, key="lbl_n_classes_widget",
            )
            if int(_n_new) != st.session_state["lbl_n_classes"]:
                st.session_state["lbl_n_classes"] = int(_n_new)

            _n_cls = int(st.session_state["lbl_n_classes"])

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
        st.markdown("#### 3️⃣ Click Image → Add Label")
        st.caption(
            "Click anywhere on the image to label that pixel with the selected class.  "
            "Use the Plotly toolbar (top-left) to switch to **Pan** mode for zooming and panning."
        )

        img_col, ctrl_col = st.columns([3, 1])

        # ── Right column: class selector + counters + buttons ─
        with ctrl_col:
            st.markdown("**Select Class**")
            _active_idx = min(
                int(st.session_state.get("lbl_active_cls", 0)), _n_cls - 1
            )
            active_cls = st.radio(
                "Active class",
                options=list(range(_n_cls)),
                format_func=lambda i: f"  {cls_cfg[i]['name']}",
                index=_active_idx,
                key="lbl_cls_radio",
                label_visibility="collapsed",
            )
            st.session_state["lbl_active_cls"] = active_cls

            st.divider()

            _total = len(st.session_state["lbl_rows"])
            st.metric("Total labels", _total)
            _cnt = Counter(r[2] for r in st.session_state["lbl_rows"])
            for _c in cls_cfg:
                st.caption(f"● {_c['name']}: **{_cnt.get(_c['id'], 0)}**")

            st.divider()

            if st.button("↩️ Undo last", use_container_width=True):
                if st.session_state["lbl_rows"]:
                    st.session_state["lbl_rows"].pop()
                    st.session_state["lbl_prev_sel"] = None
                    st.rerun()

            if st.button("🗑️ Clear all", use_container_width=True, type="secondary"):
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
            st.markdown(f"#### 4️⃣ Label List  ({_n_lbl} entries)")

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
        st.markdown("#### 5️⃣ Save CSV")
        st.caption("Format: `row,col,class_id`  (no header) — matches the supervised learning input format")

        _default_csv = (
            str(Path(st.session_state["lbl_file"]).parent / "labels.csv")
            if st.session_state["lbl_file"]
            else "labels.csv"
        )
        scol1, scol2 = st.columns([5, 1])
        with scol1:
            save_path = st.text_input(
                "Save path",
                value=_default_csv,
                key="lbl_save_path",
                label_visibility="collapsed",
            )
        with scol2:
            save_btn = st.button("💾 Save", use_container_width=True, type="primary")

        if save_btn:
            if not st.session_state["lbl_rows"]:
                st.warning("No labels to save. Click the image first to add labels.")
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
                        f"✅ **{len(st.session_state['lbl_rows'])}** labels saved  \n"
                        f"`{_sp.resolve()}`"
                    )
                    st.info(
                        "💡 **Next step**: Go to the [Run Analysis] tab → paste this path into "
                        "Labels CSV → run with **Random Forest** or **1D-CNN**."
                    )
                except Exception as e:
                    st.error(f"Save failed: {e}")


# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "HyperspectralPipeline · "
    "Methods: hybrid | kmeans | sam | supervised | autoencoder | cnn"
)
