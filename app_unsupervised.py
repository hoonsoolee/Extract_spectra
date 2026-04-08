"""
app_unsupervised.py
-------------------
Streamlit GUI — Unsupervised Methods Only (Hybrid / K-Means / SAM / Autoencoder).
No labels required.

Run with:
    python -m streamlit run app_unsupervised.py
"""

import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

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
# Custom log handler
# ============================================================

class _ListLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


# ============================================================
# Method metadata  (unsupervised only)
# ============================================================

METHODS = {
    "hybrid": {
        "label": "🌿 Hybrid  (NDVI + Brightness + K-means)",
        "help": (
            "NDVI detects vegetation → brightness separates shadows → K-means refinement.  \n"
            "**Recommended default** — no labels required."
        ),
    },
    "kmeans": {
        "label": "📊 K-Means",
        "help": (
            "PCA dimensionality reduction → K-means clustering.  \n"
            "Good for exploratory analysis."
        ),
    },
    "sam": {
        "label": "📐 SAM  (Spectral Angle Mapping)",
        "help": (
            "Compares only the **angle** between spectral vectors — unaffected by illumination.  \n"
            "Works without any labels."
        ),
    },
    "autoencoder": {
        "label": "🤖 Autoencoder  (Deep Learning)",
        "help": (
            "MLP autoencoder compresses spectra → K-means in latent space.  \n"
            "Requires PyTorch · no labels needed."
        ),
    },
    "hdbscan": {
        "label": "🔵 HDBSCAN  (Density-Based)",
        "help": (
            "Hierarchical density-based clustering — **no need to set cluster count**.  \n"
            "The algorithm finds the number of clusters automatically.  \n"
            "Noise pixels are assigned to Background (class 0)."
        ),
    },
    "gmm": {
        "label": "📈 GMM  (Gaussian Mixture Model)",
        "help": (
            "Probabilistic soft clustering via Gaussian Mixture Model.  \n"
            "PCA preprocessing (15 components) then GMM fitting.  \n"
            "Use the **Number of Classes** slider to set components."
        ),
    },
    "nmf": {
        "label": "🧩 NMF  (Spectral Unmixing)",
        "help": (
            "Non-negative Matrix Factorization — decomposes spectra into  \n"
            "endmember components and abundance maps.  \n"
            "Each pixel is assigned to its dominant endmember component."
        ),
    },
}

# ============================================================
# Labeling tool helpers (kept for display RGB only)
# ============================================================

def _get_display_rgb(data: np.ndarray, wavelengths) -> np.ndarray:
    B = data.shape[2]
    targets = [660, 550, 450]
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


# ============================================================
# Session state – run mode file scanner
# ============================================================

if "run_scan_files" not in st.session_state:
    st.session_state["run_scan_files"] = []

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── Data source ──────────────────────────────────────────
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
        github_folder = st.text_input("Sub-folder", value="", placeholder="data/2024")
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
                    _sf = sorted([f for f in _sp.rglob("*") if f.suffix.lower() in _exts])
                    _hdr_stems = {f.stem for f in _sf if f.suffix.lower() == ".hdr"}
                    _sf = [f for f in _sf
                           if not (f.suffix.lower() in {".raw", ".bil", ".bip", ".bsq"}
                                   and f.stem in _hdr_stems)]
                    st.session_state["run_scan_files"] = [str(f) for f in sorted(set(_sf))]
                    if not st.session_state["run_scan_files"]:
                        st.warning("No supported files found.")
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
        st.caption("📋 Processes all files sequentially, one report per file.")

    st.markdown("---")

    # ── Classification method ────────────────────────────────
    st.markdown("### 🧬 Classification Method")

    method = st.selectbox(
        "Method",
        list(METHODS.keys()),
        format_func=lambda k: METHODS[k]["label"],
        label_visibility="collapsed",
    )
    st.success(METHODS[method]["help"])

    st.markdown("---")

    # ── Number of classes ────────────────────────────────────
    st.markdown("### 🔢 Number of Classes")
    if method == "hdbscan":
        st.caption(
            "HDBSCAN determines the number of clusters **automatically**. "
            "The slider is ignored for this method."
        )
        n_classes = 6  # placeholder; not used by hdbscan
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
                "NDVI threshold (vegetation cutoff)", 0.0, 1.0, 0.15, 0.01,
                help="Pixels with NDVI ≥ this value → vegetation.",
            )
            brightness_threshold = st.slider(
                "Brightness threshold (shadow cutoff)", 0.0, 0.5, 0.08, 0.01,
                help="Pixels with mean reflectance below this → shadow.",
            )

    angle_threshold = 0.10
    if method == "sam":
        with st.expander("SAM Settings", expanded=True):
            angle_threshold = st.slider(
                "Angle threshold (radians, 0 = no limit)", 0.0, 0.5, 0.10, 0.01,
                help=f"Current value ≈ {round(0.10 * 57.3, 1)}°",
            )

    ae_epochs = 60
    if method == "autoencoder":
        with st.expander("Autoencoder Settings", expanded=False):
            ae_epochs = st.slider("Training epochs", 10, 200, 60, 10)

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

    st.markdown("---")

    # ── Output ────────────────────────────────────────────────
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

st.markdown("# 🌿 Hyperspectral Crop Analysis  —  Unsupervised")
st.caption("Automatic pixel classification · No labels required")

# ── Info banner ───────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])
with col_left:
    src_info   = f"`{local_folder}`" if data_src == "Local Folder" else f"`{github_repo}`"
    limit_info = str(int(file_limit)) + " file(s)" if file_limit else "all"
    st.info(
        f"**Method:** {METHODS[method]['label']}  \n"
        f"**Data:** {src_info}  \n"
        f"**Classes:** {n_classes}  \n"
        f"**Files:** {limit_info}  ·  **Output:** `{output_dir}`"
    )
with col_right:
    st.info(
        "ℹ️ **Unsupervised mode** — the algorithm automatically finds "
        "natural groupings in the data. No manually labelled pixels needed."
    )

st.markdown("---")

# ── Run ──────────────────────────────────────────────────────
if run_btn:

    errors = []
    if data_src == "Local Folder" and not local_folder:
        errors.append("Please enter a local folder path.")
    if data_src == "GitHub Repository" and not github_repo:
        errors.append("Please enter a GitHub repository.")
    if run_mode == "🔍 Single File" and not _run_single_file:
        errors.append("Single File mode: scan the folder and select a file first.")

    if errors:
        for e in errors:
            st.error(f"❌ {e}")
        st.stop()

    base = max(1, n_classes // 3)
    r    = n_classes % 3

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
            "method":  method,
            "classes": [],
            "kmeans": {
                "n_clusters":     n_classes,
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
                "n_endmembers":    n_classes,
                "endmember_pca":   15,
            },
            "autoencoder": {
                "latent_dim":    16,
                "n_clusters":    n_classes,
                "epochs":        ae_epochs,
                "batch_size":    1024,
                "learning_rate": 0.001,
                "max_pixels":    100_000,
            },
            "hdbscan": {
                "min_cluster_size": hdbscan_min_cluster_size,
                "min_samples":      hdbscan_min_samples,
                "pca_components":   15,
            },
            "gmm": {
                "n_components":    n_classes,
                "covariance_type": "full",
                "max_iter":        100,
                "pca_components":  15,
                "random_state":    42,
            },
            "nmf": {
                "n_components": n_classes,
                "max_iter":     500,
                "random_state": 42,
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

    log_handler = _ListLogHandler()
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    )
    root_log = logging.getLogger()
    root_log.addHandler(log_handler)
    root_log.setLevel(logging.DEBUG if verbose else logging.INFO)

    pipeline_ok = False
    import time
    t_start = time.perf_counter()

    try:
        with st.spinner("⏳ Analysing…  (may take several minutes for large images)"):
            from src.pipeline import Pipeline
            pipeline = Pipeline(cfg)
            pipeline.run(
                file_limit=int(file_limit) if file_limit else None,
                single_file=_run_single_file,
            )
        pipeline_ok = True

    except Exception:
        st.error("❌ Pipeline error.")
        st.code(traceback.format_exc(), language="python")

    finally:
        root_log.removeHandler(log_handler)

    if pipeline_ok:
        elapsed = time.perf_counter() - t_start
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        st.success(f"✅ Analysis complete!  ·  ⏱ {elapsed_str}")

        out_p = Path(output_dir)

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

        class_maps = sorted(out_p.rglob("class_map.png"))
        if class_maps:
            st.markdown("### 🗺️ Classification Map Preview")
            n_cols = min(len(class_maps), 3)
            cols   = st.columns(n_cols)
            for col, img_path in zip(cols, class_maps[:3]):
                with col:
                    st.image(str(img_path), caption=img_path.parent.name,
                             use_container_width=True)
            if len(class_maps) > 3:
                st.caption(f"… and {len(class_maps) - 3} more — see full report")

    if log_handler.lines:
        with st.expander(
            f"📋 Run Log  ({len(log_handler.lines)} lines)",
            expanded=not pipeline_ok,
        ):
            st.code("\n".join(log_handler.lines), language="text")


# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "HyperspectralPipeline · Unsupervised Edition · "
    "Methods: hybrid | kmeans | sam | autoencoder | hdbscan | gmm | nmf"
)
