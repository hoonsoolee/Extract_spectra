# Hyperspectral Field Crop Analysis — English Usage Guide

## 1. What the program produces

For each hyperspectral image, the workflow can:

1. load an ENVI/GeoTIFF/HDF5/MAT cube without expanding the entire file unnecessarily;
2. convert raw DN to reflectance using a compatible White/Dark or panel calibration;
3. cluster the full image once;
4. extract separate cluster spectra for every user-defined region of interest (ROI);
5. re-cluster only an ROI whose global result is unsatisfactory; and
6. export raw-DN and calibrated spectra, cluster-review images, provenance JSON, and an HTML report.

The intended scientific sequence is:

```text
ENVI BIL/HDR
  -> White/Dark or multi-panel calibration
  -> full-image clustering
  -> ROI-level spectral summaries
  -> optional ROI-only re-clustering
  -> CSV + PNG + JSON + HTML report
```

Existing output files are not retroactively recalibrated when a new
calibration profile is created. Re-run the analysis to generate new calibrated
results.

## 2. Installation

Requirements:

- Python 3.9 or later
- Git
- Sufficient local memory/storage for the chosen downsampling level

```bash
git clone https://github.com/hoonsoolee/Extract_spectra.git
cd Extract_spectra
python -m pip install -r requirements.txt
```

PyTorch is required only for the autoencoder and 1D-CNN methods. Install a CPU
or CUDA build appropriate for your workstation if those methods are needed.

## 3. Starting the web interface

Full research workflow:

```bash
python -m streamlit run app.py
```

English core-analysis and pixel-labeling interface:

```bash
python -m streamlit run app_en.py
```

Direct global-clustering/ROI interface:

```bash
python -m streamlit run app_roi_clustering.py
```

The application normally opens at `http://localhost:8501`.

The full calibration and ROI workflow currently lives in `app.py` and
`app_roi_clustering.py`; some interface labels in these two research screens
are Korean. This guide gives the matching English sequence and button meaning.

## 4. Supported inputs

| Format | Entry file |
|---|---|
| ENVI | `.hdr`, or the matching `.bil` / `.bip` / `.bsq` / `.raw` / `.img` / `.dat` |
| GeoTIFF | `.tif`, `.tiff` |
| HDF5 | `.h5`, `.hdf5` |
| MATLAB | `.mat` |

For ENVI, keep the small header and binary data file together. Examples:

```text
scene.bil
scene.bil.hdr
```

or:

```text
scene.bil
scene.hdr
```

Large Ceres acquisition files are not yet passed directly into the main web
pipeline. Demultiplex them first with `ceres_demux.py`, then analyse the
resulting ENVI file.

## 5. Recommended first run

1. Start `app.py`.
2. In the sidebar, choose a local data folder.
3. Choose **Single file**, scan the folder, and select one ENVI header/data file.
4. Start with spatial downsampling `4` for a fast review.
5. Use **Hybrid** with the default thresholds and cluster count.
6. Confirm or create a reflectance calibration before treating the spectra as scientific reflectance.
7. Run one file and review its class map and report before starting a batch.

Spatial downsampling reduces the loaded pixel count by approximately the square
of the factor while retaining the spectral bands. Use `1` for final
full-resolution processing only when memory and run time permit it.

## 6. Creating a reflectance calibration

In the **Panel Calibration** tab (`패널 보정`):

1. Load a reference-panel image or a scene containing known panels.
2. Select the inside of each panel with a box or lasso ROI.
3. Enter the certified reflectance as a fraction: `0.99`, `0.50`, `0.25`, etc.
4. Select a measured Dark image acquired with matching camera settings when available.
5. If no Dark exists, the program can use a synthetic constant Dark (default `100 DN`) for screening.
6. Run the automatic calibration calculation.

The weighted multi-panel model is:

```text
R(lambda) = a(lambda) * [DN(lambda) - Dark(lambda)]
```

Equivalently, exported coefficients can be written as:

```text
R(lambda) = a(lambda) * DN(lambda) + b(lambda)
```

The program checks saturation by band. As a bright panel approaches
saturation, its weight is reduced smoothly and a valid lower-reflectance panel
can carry the same wavelength. Bands with no usable panel information remain
invalid rather than being silently invented.

Calibration profiles are normally saved to:

```text
output/calibration/<source>_weighted_dark_calibration.npz
```

When a file is analysed, the source folder and `output/calibration` are searched
for conservatively named candidates. Band count and wavelength compatibility
are checked before automatic application. An explicit profile selected by the
user takes priority.

When calibration is applied, additional scene normalization is automatically
disabled so the physical reflectance scale is preserved.

For publication-quality work, use a measured sensor Dark with matching gain,
integration time, and temperature. A synthetic Dark is recorded explicitly in
the CSV and manifest and should not be mistaken for a measured reference.

## 7. Full-image analysis

Available methods include:

| Method | Labels required | Typical use |
|---|---:|---|
| Hybrid | No | Recommended field-crop default: NDVI, brightness, then K-Means refinement |
| K-Means | No | Exploratory clustering |
| SAM | Optional | Spectral-shape comparison |
| HDBSCAN | No | Density-based clustering and automatic cluster count |
| GMM | No | Probabilistic clusters with overlapping distributions |
| NMF | No | Exploratory spectral unmixing |
| Random Forest | Yes | Supervised classification |
| Autoencoder | No | Deep unsupervised representation + clustering |
| 1D-CNN | Yes | Deep supervised classification |

The main output folder is:

```text
output/
└── <source-name>/
    ├── report_<timestamp>_<method>.html
    ├── class_map_<method>.png
    ├── spectra_<method>.csv
    ├── spectra_<method>_reflectance.csv   # when calibration was applied
    ├── spectra_<method>_raw_dn.csv
    ├── processing_manifest.json
    ├── report_config.json
    ├── rgb.png / ndvi.png                 # when selected
    └── cluster_map.png / cluster_overlay.png
```

### Selectable result reports

- **Quick Field QC** saves RGB, NDVI, cluster map/overlay, mean and median
  spectra, calibration QC, and processing time. It skips expensive cluster
  quality calculations and is the recommended screening preset.
- **Research Standard** adds CIR, isolated cluster images, mean/median/std/IQR,
  NDVI/GNDVI/NDRE/PRI, cluster quality, and vegetation-separation assessment.
- **Custom** lets the user select image sections, spectral statistics,
  vegetation indices, HTML/CSV/PNG outputs, and a batch-level daily summary.

Indices are calculated only when calibrated reflectance and the required
wavelengths are available. Raw DN or missing bands produce an explicit
"not calculated" reason rather than an invented value. Batch mode can also
write `daily_report_<timestamp>.html` and `daily_summary_<timestamp>.csv` with
per-file calibration, NDVI, vegetation fraction, class count, quality, and time.

`spectra_<method>.csv` contains the values actually used for clustering. When
a profile was applied it is calibrated reflectance. The explicit
`_reflectance.csv` and `_raw_dn.csv` files remove ambiguity for downstream
analysis.

Each class contains:

- mean, standard deviation, median, 25th and 75th percentiles;
- Medoid-Neighbourhood Average (`mna`); and
- SAM-Neighbourhood Average (`sam_avg`).

Calibration-aware CSV files also contain:

- `value_units`
- `calibration_applied`
- `calibration_profile` and `paired_calibration_profile`
- `calibration_method` and formula
- Dark source/type and manual Dark DN, if any
- panel summary
- requested/effective normalization mode
- per-wavelength `calibration_a` and `calibration_b`

For a science-ready reflectance table, verify:

```text
value_units = reflectance
calibration_applied = true
normalization_mode = none
```

## 8. Defining ROIs and extracting spectra

The main ROI spectrum tab supports three region tools:

- **Box ROI** (`⬚ Box ROI`): drag a rectangle.
- **Lasso ROI** (`✏️ Lasso ROI`): drag a freehand outline.
- **Click Polygon ROI** (`🔺 Polygon 클릭 ROI`): click vertices around a leaf or plot.

For a polygon:

1. choose **Zoom** (`🔍 확대`) and drag the area that should fill the preview, if needed;
2. switch to **Click Polygon ROI**;
3. click at least three vertices around the target;
4. use **Undo last point** or **Clear all points** if necessary; and
5. click **Finish Polygon** (`✅ Polygon 완료`).

The selected spectrum is displayed as mean, median, and variability. The graph
can be limited by wavelength and y-axis range without changing the saved CSV.
If calibration is active, both calibrated reflectance and a raw-DN companion
CSV are saved.

With a calibration profile connected, use the buttons above the ROI chart to
switch between **Calibrated Reflectance**, **Raw DN**, and **Raw/Calibrated
Comparison**. Comparison mode uses separate left/right y-axes so the different
scales do not hide spectral-shape problems. Open **Calibration suspect-band and
coefficient diagnostics** to inspect `R = a*DN+b`, invalid or non-positive
gains, robust gain outliers, and ROI mean reflectance outside -0.05..1.20.
Diagnostics never rewrite the calibration; use them to review panel
saturation, White/Dark selection, and the affected wavelengths.

## 9. One global clustering model, separate spectra per ROI

Open the region-analysis screen from the navigation sidebar or run:

```bash
python -m streamlit run app_roi_clustering.py
```

Recommended sequence:

1. Scan the source folder and load one file, initially at downsampling `4`.
2. Confirm the green calibration banner and selected `.npz` profile.
3. In **Divide Regions** (`구역 나누기`), add named box, lasso, or polygon ROIs.
4. In **Global Clustering** (`전체 클러스터링`), choose the same method and settings used in the main analysis.
5. Run **Global clustering + ROI spectrum extraction**.
6. Inspect overlays, boundaries, isolated cluster images, and spectra.
7. If one ROI is poor, give that ROI its own threshold/cluster settings and run an ROI-only trial.
8. Compare the global baseline and ROI trial, then accept the better result.
9. Save the HTML report.

All regular ROIs share the same full-image cluster definitions. ROI-only
re-clustering is an explicit exception and is recorded in the manifest.

The region report folder contains:

```text
roi_cluster_output/
└── <source>_<timestamp>/
    ├── report.html
    ├── global_overlay.png
    ├── global_boundaries.png
    ├── global_class_map.png
    ├── global_cluster_<id>.png
    ├── all_roi_cluster_spectra.csv
    ├── all_roi_cluster_spectra_raw_dn.csv
    ├── cluster_summary.csv
    ├── analysis_manifest.json
    └── <ROI>/
        ├── cluster_map.png
        ├── cluster_boundaries.png
        ├── cluster_<id>.png
        ├── cluster_spectra.png
        ├── cluster_spectra_reflectance.csv
        ├── cluster_spectra_raw_dn.csv
        └── cluster_map.npz
```

## 10. Exporting a calibrated, binned ENVI cube

The region-analysis application includes a science-ready BIL export tab. It
reads the original ENVI file in chunks, applies the selected calibration, and
spatially bins the cube before writing a float32 reflectance BIL/HDR pair.

This export does not overwrite the original file. Invalid calibration bands
remain `NaN`, and reflectance is not forcibly clipped to `0..1`, allowing
quality problems to remain visible.

## 11. Pixel labels for supervised methods

1. Open the **Pixel Labeling** tab.
2. Load the hyperspectral file.
3. Define class names and colors.
4. Click representative pixels for each class.
5. Save `labels.csv`.
6. Provide that CSV when running Random Forest or 1D-CNN.

The expected minimum fields are row, column, and class ID.

## 12. Command-line use

```bash
# Process all supported files in a folder
python main.py --local-folder ./data

# Quick one-file test
python main.py --local-folder ./data --limit 1

# K-Means example
python main.py --local-folder ./data --method kmeans --n-clusters 8

# List files without processing
python main.py list --local-folder ./data
```

Calibration auto-discovery defaults are in `config.yaml`. For scripted final
runs, retain a processing manifest beside every result.

## 13. Troubleshooting

| Problem | What to check |
|---|---|
| `ModuleNotFoundError` | Run `python -m pip install -r requirements.txt` in the active environment. |
| ENVI file will not load | Keep the `.hdr` and binary file together and load either recognized entry. |
| Calibration is not applied | Check the green status banner, filename match, band count, wavelength axis, and profile path. |
| `calibration_applied=false` | The file is raw/normalized, not science-ready reflectance. Select or create a compatible profile and re-run. |
| Spectra look distorted | Do not use per-band normalization for scientific spectra. Verify White/Dark ROIs and saturated bands. |
| Polygon is too imprecise | Zoom first, then switch to click-polygon mode and place vertices. |
| Analysis is too slow or memory intensive | Start with spatial downsampling `4` or `8`, then reduce it for final processing. |
| Old CSV lacks calibration columns | It predates the provenance update and must be regenerated. |

## 14. Current scope

The implemented end product is the ENVI/BIL-based calibration, clustering,
ROI-spectrum, re-clustering, and report pipeline. Direct day-folder Ceres
ingestion and a configurable daily index-image package (for example, automatic
NDVI image export for every acquisition) remain the next integration stage.
