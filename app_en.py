"""
app_en.py
---------
English presentation of the shared CanopySpectra app.

The implementation is intentionally executed from ``app.py`` so Korean and
English users always receive the same CERES, ROI, calibration, clustering, and
report features.  ``src.english_ui`` translates display text without changing
the values used by the analysis logic.

Run with:
    python -m streamlit run app_en.py
"""

import runpy
from pathlib import Path as _SharedPath

import streamlit as _shared_st

from src.english_ui import install_english_ui as _install_english_ui


_install_english_ui()
runpy.run_path(str(_SharedPath(__file__).with_name("app.py")), run_name="__main__")
_shared_st.stop()

import logging
import sys
import traceback
import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Make sure 'src' package is importable from this directory
sys.path.insert(0, str(Path(__file__).parent))

from src.report_options import REPORT_PRESETS
from src.local_open import open_local_path as _open_local_path

# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="CanopySpectra",
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
    "hdbscan": {
        "label":  "🔵 HDBSCAN  (Density-Based)",
        "kind":   "Unsupervised",
        "help":   (
            "Hierarchical density-based clustering — **no need to set cluster count**.  \n"
            "The algorithm finds the number of clusters automatically.  \n"
            "Noise pixels are assigned to Background (class 0)."
        ),
    },
    "gmm": {
        "label":  "📈 GMM  (Gaussian Mixture Model)",
        "kind":   "Unsupervised",
        "help":   (
            "Probabilistic soft clustering via Gaussian Mixture Model.  \n"
            "PCA preprocessing (15 components) then GMM fitting.  \n"
            "Use the **Number of Classes** slider to set components."
        ),
    },
    "nmf": {
        "label":  "🧩 NMF  (Spectral Unmixing)",
        "kind":   "Unsupervised",
        "help":   (
            "Non-negative Matrix Factorization — decomposes spectra into  \n"
            "endmember components and abundance maps.  \n"
            "Each pixel is assigned to its dominant endmember component."
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
    elif method == "hdbscan":
        st.caption(
            "HDBSCAN determines the number of clusters **automatically**. "
            "The slider is ignored for this method."
        )
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

    hdbscan_min_cluster_size = 50
    hdbscan_min_samples      = 5
    if method == "hdbscan":
        with st.expander("HDBSCAN Settings", expanded=True):
            hdbscan_min_cluster_size = st.slider(
                "min_cluster_size", 10, 500, 50, 10,
                help=(
                    "Minimum number of pixels to form a cluster. "
                    "Larger values → fewer, larger clusters."
                ),
            )
            hdbscan_min_samples = st.slider(
                "min_samples", 1, 50, 5, 1,
                help=(
                    "Controls clustering conservatism. "
                    "Higher values → more noise pixels (class 0)."
                ),
            )

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

    # ── Normalization ────────────────────────────────────────
    st.markdown("### 📐 Normalization")
    _NORM_MODES = {
        "global":   "Global scale (preserves spectral shape)",
        "per_band": "Per-band stretch (maximises contrast)",
        "none":     "None (raw DN)",
    }
    normalize_mode = st.selectbox(
        "Normalization mode",
        list(_NORM_MODES.keys()),
        format_func=lambda k: _NORM_MODES[k],
        index=0,
        label_visibility="collapsed",
        help=(
            "Global scale: divides the whole cube by one number, so spectral "
            "shape and band ratios such as NDVI are preserved exactly. Use this "
            "when the extracted spectra are the product.\n\n"
            "Per-band stretch: each band gets its own gain — good contrast, but "
            "spectral shape is distorted and not comparable to reference "
            "spectra.\n\n"
            "None: keeps raw sensor DN values."
        ),
    )
    if normalize_mode == "per_band":
        st.warning(
            "⚠️ Per-band stretch applies a different gain to every band, which "
            "distorts spectral shape. Use **Global scale** if the spectra will "
            "be compared against libraries or published."
        )
    if normalize_mode == "none":
        st.info(
            "ℹ️ With raw DN the Hybrid brightness threshold (default 0.08) no "
            "longer matches the value scale. Adjust it to your data."
        )

    st.markdown("---")

    # ── Large-file handling ──────────────────────────────────
    st.markdown("### ⚡ Large Files")
    spatial_downsample = st.select_slider(
        "Spatial downsampling (1 = full resolution)",
        options=[1, 2, 4, 8],
        value=1,
        help=(
            "N keeps 1 pixel per N×N block, cutting memory to 1/N². "
            "Recommended: 4 for multi-GB files. Spectral shape is preserved; "
            "only the classification map resolution is reduced."
        ),
    )

    st.markdown("---")

    # ── Selectable report builder ────────────────────────────
    st.markdown("### 📋 Result Report")
    _REPORT_PRESET_LABELS = {
        "quick_qc": "⚡ Quick Field QC (recommended)",
        "research_standard": "🔬 Research Standard",
        "custom": "🛠️ Custom",
    }
    report_preset = st.selectbox(
        "Report preset",
        list(_REPORT_PRESET_LABELS),
        format_func=lambda key: _REPORT_PRESET_LABELS[key],
        key="report_preset",
    )
    _report_defaults = REPORT_PRESETS[report_preset]
    report_sections = dict(_report_defaults["sections"])
    report_statistics = list(_report_defaults["spectra_statistics"])
    report_indices = list(_report_defaults["indices"])
    save_selected_images = bool(_report_defaults["save_selected_images"])
    save_daily_summary = bool(_report_defaults["daily_summary"])
    save_html_report = True
    save_spectra_csv = True

    with st.expander("Review / select report contents", expanded=report_preset == "custom"):
        if report_preset != "custom":
            st.caption(
                "Sections: "
                + ", ".join(key for key, enabled in report_sections.items() if enabled)
                + "\n\nStatistics: " + ", ".join(report_statistics)
                + " · Indices: " + (", ".join(report_indices) or "none")
            )
        else:
            _section_labels = {
                "rgb": "RGB image",
                "false_color": "CIR false colour",
                "spectral_indices": "Selected index maps and summaries",
                "class_map": "Cluster map",
                "cluster_overlay": "RGB + cluster overlay",
                "per_class_images": "Per-cluster images",
                "class_summary": "Cluster pixel statistics",
                "spectral_plot": "Cluster spectra",
                "quality_metrics": "Cluster quality and separability",
                "vegetation_quality": "Vegetation separation assessment",
                "calibration_qc": "Calibration provenance and valid-band QC",
            }
            _sc1, _sc2 = st.columns(2)
            for _index, (_key, _label) in enumerate(_section_labels.items()):
                with (_sc1 if _index % 2 == 0 else _sc2):
                    report_sections[_key] = st.checkbox(
                        _label,
                        value=bool(report_sections.get(_key)),
                        key=f"report_section_{_key}",
                    )
            report_statistics = st.multiselect(
                "Spectral statistics",
                ["mean", "median", "std", "iqr"],
                default=report_statistics,
                key="report_statistics",
            ) or ["mean"]
            report_indices = st.multiselect(
                "Vegetation indices",
                ["NDVI", "GNDVI", "NDRE", "PRI"],
                default=report_indices,
                key="report_indices",
            )
            save_html_report = st.checkbox("Save interactive HTML", True, key="report_save_html")
            save_spectra_csv = st.checkbox("Save spectra CSV", True, key="report_save_csv")
            save_selected_images = st.checkbox(
                "Save selected images as PNG", True, key="report_save_images"
            )
            save_daily_summary = st.checkbox(
                "Save daily batch summary HTML and CSV", True, key="report_daily_summary"
            )
        if report_indices:
            st.info(
                "Indices are calculated only from calibrated reflectance. "
                "If calibration or required wavelengths are unavailable, the report records why."
            )

    with st.expander(
        "👥 Team / plot daily package",
        expanded=run_mode == "📦 Batch (all files)",
    ):
        team_daily_enabled = st.checkbox(
            "Create a team-facing daily package after the batch",
            value=True,
            disabled=run_mode != "📦 Batch (all files)",
            key="team_daily_enabled",
            help=(
                "Combines per-file results into one HTML, Excel workbook, and NDVI "
                "comparison without reopening the hyperspectral cubes."
            ),
        )
        team_name = st.text_input(
            "Team name",
            value="Field Team",
            key="team_daily_name",
            disabled=not team_daily_enabled,
        )
        measurement_date = st.date_input(
            "Acquisition date",
            value=datetime.date.today(),
            key="team_daily_date",
            disabled=not team_daily_enabled,
        )
        plot_metadata_csv = st.text_input(
            "Plot metadata CSV (optional)",
            value="",
            placeholder="filename, plot_id, treatment, genotype, replicate",
            key="team_daily_metadata_csv",
            disabled=not team_daily_enabled,
        )
        st.caption(
            "Without a CSV, the filename becomes the plot ID. Optional columns: "
            "filename, plot_id, treatment, genotype, replicate, team, measurement_date."
        )

    team_daily_enabled = bool(
        team_daily_enabled and run_mode == "📦 Batch (all files)"
    )
    if team_daily_enabled:
        report_sections["spectral_indices"] = True
        if "NDVI" not in report_indices:
            report_indices.append("NDVI")

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


st.session_state.setdefault("run_last_reports", [])
st.session_state.setdefault("run_last_output_dir", "")

# ============================================================
# Main area
# ============================================================

st.markdown("# 🌿 CanopySpectra")
st.caption("From CERES to Science-Ready Field Spectra")

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
        if method == "hdbscan":
            cls_info = "auto-determined (HDBSCAN)"
        else:
            cls_info = f"{n_classes} classes" if n_classes else "inferred from labels CSV"
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
        if team_daily_enabled and not team_name.strip():
            errors.append("Please enter a team name for the team/plot daily package.")
        if (
            team_daily_enabled
            and plot_metadata_csv.strip()
            and not Path(plot_metadata_csv.strip()).expanduser().is_file()
        ):
            errors.append("The plot metadata CSV file could not be found.")

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
            "output": {
                "dir":                     output_dir,
                "save_classification_map": bool(report_sections.get("class_map")),
                "save_spectra_csv":        save_spectra_csv,
                "save_report":             save_html_report,
                "per_file_report":         run_mode == "📦 Batch (all files)",
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
                "lang":             "en",
                "team_daily": {
                    "enabled":          team_daily_enabled,
                    "team_name":        team_name.strip() or "Field Team",
                    "measurement_date": measurement_date.isoformat(),
                    "metadata_csv":     plot_metadata_csv.strip(),
                },
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
        _t_wall      = _time.time()   # wall-clock start for file mtime filtering
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

            # Keep only reports produced by this run and persist the paths so
            # result-access buttons survive subsequent Streamlit reruns.
            reports = sorted(
                (
                    p for p in out_p.rglob("*report*.html")
                    if p.stat().st_mtime >= _t_wall
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            st.session_state["run_last_reports"] = [
                str(report.resolve()) for report in reports
            ]
            st.session_state["run_last_output_dir"] = str(out_p.resolve())
            st.session_state["run_last_team_packages"] = list(
                getattr(pipeline, "team_packages", [])
            )

            # Class-map previews — only files created/updated in this run
            class_maps = sorted(
                p for p in out_p.rglob("class_map.png")
                if p.stat().st_mtime >= _t_wall
            )
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

    # This panel is outside `if run_btn` so its buttons remain available after
    # their own click-triggered reruns.
    _last_report_paths = [
        Path(path) for path in st.session_state.get("run_last_reports", [])
        if Path(path).is_file()
    ]
    _last_output_path_text = st.session_state.get("run_last_output_dir", "")
    _last_output_path = (
        Path(_last_output_path_text)
        if _last_output_path_text else None
    )
    if _last_report_paths or (_last_output_path and _last_output_path.is_dir()):
        st.markdown("### 📄 Open Recent Analysis Results")
        _selected_report = None
        if len(_last_report_paths) > 1:
            _selected_report_text = st.selectbox(
                "HTML report to open",
                options=[str(path) for path in _last_report_paths],
                format_func=lambda value: (
                    f"{Path(value).parent.name} / {Path(value).name}"
                ),
                key="run_last_report_choice",
            )
            _selected_report = Path(_selected_report_text)
        elif _last_report_paths:
            _selected_report = _last_report_paths[0]
            st.caption(f"HTML report: `{_selected_report}`")

        _open_col, _folder_col, _download_col = st.columns(3)
        with _open_col:
            if st.button(
                "🌐 Open Selected HTML Report",
                use_container_width=True,
                disabled=_selected_report is None,
                key="run_open_report",
            ):
                try:
                    _open_local_path(_selected_report)
                    st.success("Opened the report in the default web browser.")
                except Exception as exc:
                    st.error(f"Could not open the HTML report: {exc}")
        with _folder_col:
            if st.button(
                "📂 Open Results Folder",
                use_container_width=True,
                disabled=not (_last_output_path and _last_output_path.is_dir()),
                key="run_open_output_folder",
            ):
                try:
                    _open_local_path(_last_output_path)
                    st.success("Opened the results folder.")
                except Exception as exc:
                    st.error(f"Could not open the results folder: {exc}")
        with _download_col:
            if _selected_report is not None:
                st.download_button(
                    "⬇️ Download HTML Report",
                    data=_selected_report.read_bytes(),
                    file_name=_selected_report.name,
                    mime="text/html",
                    use_container_width=True,
                    key="run_download_report",
                )
            else:
                st.button(
                    "⬇️ Download HTML Report",
                    disabled=True,
                    use_container_width=True,
                    key="run_download_report_disabled",
                )
        if not _last_report_paths:
            st.caption(
                "No HTML report was created in the most recent run. The report "
                "option may have been disabled; the output folder is still available."
            )

    _team_packages = [
        package for package in st.session_state.get("run_last_team_packages", [])
        if package.get("directory") and Path(package["directory"]).is_dir()
    ]
    if _team_packages:
        st.markdown("### 👥 Team / Plot Daily Results")
        _team_index = 0
        if len(_team_packages) > 1:
            _team_labels = [
                f"{item.get('measurement_date', '')} · {item.get('team', '')}"
                for item in _team_packages
            ]
            _team_label = st.selectbox(
                "Team/date package",
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
            if st.button("🌐 Open Team HTML", use_container_width=True, key="open_team_daily_report"):
                _open_local_path(_team_report)
        with _tp2:
            if st.button("📂 Open Team Folder", use_container_width=True, key="open_team_daily_folder"):
                _open_local_path(_team_dir)
        with _tp3:
            _download_path = _team_workbook if _team_workbook.is_file() else _team_summary_csv
            if _download_path.is_file():
                st.download_button(
                    "⬇️ Team Excel/CSV",
                    data=_download_path.read_bytes(),
                    file_name=_download_path.name,
                    mime=(
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        if _download_path.suffix.lower() == ".xlsx" else "text/csv"
                    ),
                    use_container_width=True,
                    key="download_team_daily_results",
                )
        if _team_package.get("workbook_warning"):
            st.warning(_team_package["workbook_warning"])
        _team_visuals = [
            (_team_dir / "plots_ndvi.png", "All plots NDVI · common -1 to 1 scale"),
            (_team_dir / "plot_ndvi_comparison.png", "Plot NDVI median/IQR · QC PASS only"),
        ]
        _team_visuals = [item for item in _team_visuals if item[0].is_file()]
        if _team_visuals:
            _visual_columns = st.columns(len(_team_visuals))
            for _column, (_image, _caption) in zip(_visual_columns, _team_visuals):
                _column.image(str(_image), caption=_caption, use_container_width=True)

    with st.expander("📊 Result CSV guide (opens in Excel)"):
        st.markdown(
            "The saved tables are currently **UTF-8 CSV files that open directly in "
            "Excel**, not a single `.xlsx` workbook. Use the filename suffix and the "
            "`value_units` column to identify the value scale."
        )
        st.markdown(
            """
| File | Purpose |
|---|---|
| `spectra_{method}_reflectance.csv` | Science-ready reflectance spectra, created when valid calibration is applied |
| `spectra_{method}_raw_dn.csv` | Original sensor DN for diagnostics and before/after comparison |
| `spectra_{method}_processed.csv` | Normalized/processed relative values when calibration is unavailable; not absolute reflectance |
| `spectra_{method}.csv` | Main values from the current run; verify its scale in `value_units` |
| `daily_summary_*.csv` | Per-file class count, NDVI, vegetation fraction, quality metrics, and elapsed time |
| `all_roi_cluster_spectra*.csv` | Combined spectra for all ROIs and clusters |
| `cluster_summary.csv` | Pixel count and area fraction (`fraction`, 0–1) for each ROI cluster |
"""
        )
        st.markdown(
            "The main `spectra_*` CSV is wide format with one row per wavelength and "
            "`mean`, `std`, `median`, `q25`, `q75`, `mna`, and `sam_avg` columns per cluster. "
            "ROI `cluster_spectra*` files are long format with one row per "
            "ROI × cluster × wavelength and store `mean`, `median`, `std`, `q25`, and `q75`. "
            "`mna` selects representative pixels by value; `sam_avg` selects by spectral shape."
        )
        st.info(
            "For publication, prefer `_reflectance.csv` and verify "
            "`value_units=reflectance`, `calibration_applied=True`, and "
            "`calibration_qc_status=PASS`. REVIEW results require inspection; avoid FAIL results."
        )


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
    "CanopySpectra · "
    "Methods: hybrid | kmeans | sam | supervised | autoencoder | cnn"
)
