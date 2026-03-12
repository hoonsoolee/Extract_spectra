# Hyperspectral Field Crop Analysis Pipeline — Usage Guide

## Overview

This tool automatically classifies pixels in hyperspectral field crop images and extracts per-class reflectance spectra. Results are saved as a CSV file and an interactive HTML report.

---

## 1. Installation

### Requirements
- Python 3.9 or later
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/hoonsoolee/Extract_spectra.git
cd Extract_spectra

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** If you plan to use the `autoencoder` or `1D-CNN` methods, PyTorch is also required:
> ```bash
> pip install torch
> ```

---

## 2. Preparing Your Data

Place your hyperspectral image files in the `data/` folder (or any folder of your choice).

**Supported formats:**

| Format | Extension |
|--------|-----------|
| ENVI | `.hdr` (+ matching `.raw` / `.bil` / `.bip` / `.bsq`) |
| GeoTIFF | `.tif` / `.tiff` |
| HDF5 | `.h5` / `.hdf5` |
| MATLAB | `.mat` |

---

## 3. Running the GUI (Recommended)

```bash
# Korean UI
python -m streamlit run app.py

# English UI
python -m streamlit run app_en.py
```

Your browser will open automatically at `http://localhost:8501`.

### Sidebar Settings

| Setting | Description |
|---------|-------------|
| **Data Source** | Choose *Local Folder* and enter the path to your data folder |
| **Processing Mode** | *Single File* — scan folder, select one file to analyse; *Batch* — process all files and save one report per file |
| **Classification Method** | See table below |
| **Number of Classes** | Number of clusters / classes to detect |
| **Output Folder** | Where results are saved (default: `./output`) |
| **File Limit** | Set to 1–2 for a quick test |

### Classification Methods

| Method | Type | Labels needed? | Notes |
|--------|------|---------------|-------|
| **Hybrid** | Unsupervised | No | **Recommended default.** NDVI → brightness → K-means |
| **K-Means** | Unsupervised | No | Exploratory analysis |
| **SAM** | Unsupervised / Supervised | Optional | Illumination-invariant |
| **Random Forest** | Supervised | **Yes** | Highest accuracy with labels |
| **Autoencoder** | Unsupervised | No | Requires PyTorch |
| **1D-CNN** | Supervised | **Yes** | Requires PyTorch |

### Output Files

After the analysis you will see:

```
output/
└── <filename>/
    ├── spectra.csv       # Per-class mean & std reflectance spectra
    ├── class_map.png     # Classification map image
    └── report_YYYYMMDD_HHMMSS.html   # Full interactive HTML report
```

Open the `.html` file in any browser to view:
- RGB / CIR composite images
- Classification map
- Per-class overlay images (colour-highlighted)
- Interactive reflectance spectra chart
- Cluster quality metrics (Silhouette, Davies-Bouldin)
- Vegetation separation accuracy (NDVI-based Recall / Precision / F1)
- **Processing time** per file

---

## 4. Pixel Labeling Tool (for Supervised Methods)

If you want to use **Random Forest** or **1D-CNN**, you need to provide labelled pixels.

1. Go to the **Pixel Labeling** tab in the GUI.
2. Enter the path to your image file (or a folder) and click **Load**.
3. Configure class names and colours.
4. Click on pixels in the image to assign class labels.
5. Click **Save** to export `labels.csv`.
6. In the **Run Analysis** tab, enter the path to `labels.csv` and select *Random Forest* or *1D-CNN*.

---

## 5. Command-Line Interface (CLI)

```bash
# Process all files in ./data with default settings
python main.py --local-folder ./data

# Single file, K-Means, 8 clusters
python main.py --local-folder ./data --method kmeans --n-clusters 8

# From a GitHub repository
python main.py --github-repo owner/repo --github-folder data/2024

# List files only (no processing)
python main.py list --local-folder ./data
```

---

## 6. Hybrid Method — Class IDs

When using the default **Hybrid** method, the following class IDs are assigned:

| ID | Class |
|----|-------|
| 0 | Background |
| 1 | Sunlit Leaves |
| 2 | Shadowed Leaves |
| 3 | Soil |
| 4 | Other |

---

## 7. Tips

- **First time?** Use *Single File* mode with the **Hybrid** method to check one image before running the full batch.
- **Processing time** is shown in the success banner and in the HTML report — use it to estimate total batch time.
- **Bad bands** around 1340–1460 nm and 1790–1960 nm are automatically removed.
- All processing runs on **CPU**. Large images (> 1000 × 1000 px) may take several minutes per file.

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Unsupported format` error | Check that the file extension is supported |
| ENVI file not loading | Make sure the `.hdr` and data file (`.raw` / `.bil` etc.) are in the same folder |
| Blank classification map | Try lowering the NDVI threshold (default 0.15) |
| Very slow processing | Enable spatial downsampling in `config.yaml` (`spatial_downsample: 2`) |

---

*Developed for field hyperspectral crop analysis. Contact the lab for questions.*
