# Hyperspectral Field Crop Analysis — English Usage Guide

New users should begin with the shorter, click-by-click
[Practical Quick Start](quick_start_en.html), which includes examples for CERES,
ROI spectra, clustering review, calibration, and saved results.

## 1. What the program produces

After a run, the **Open Recent Analysis Results** panel can open the selected
HTML report in the host's default browser, open its results folder, or download
the report. Use the download action when running on a headless remote cluster.

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

Full Korean research workflow:

```bash
python -m streamlit run app.py
```

Full English research workflow with the same CERES, ROI, panel-calibration,
clustering, and reporting implementation:

```bash
python -m streamlit run app_en.py
```

From the English app, use **ROI Analysis & Re-clustering** in the navigation
sidebar. The following standalone command opens the Korean developer entry:

```bash
python -m streamlit run app_roi_clustering.py
```

The application normally opens at `http://localhost:8501`.

`app.py` and `app_en.py` execute one shared implementation. The English entry
translates presentation text only; processing options and scientific outputs
are identical in both languages.

## 4. Supported inputs

| Format | Entry file |
|---|---|
| ENVI | `.hdr`, or the matching `.bil` / `.bip` / `.bsq` / `.raw` / `.img` / `.dat` |
| GeoTIFF | `.tif`, `.tiff` |
| HDF5 | `.h5`, `.hdf5` |
| MATLAB | `.mat` |
| CERES/CBDF | `.ceres` (indexed in the web UI, then one selected sensor/segment is streamed to BIL) |

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

Large Ceres acquisition files can be indexed and previewed in the web UI. The
selected sensor/segment is then streamed to ENVI BIL for the existing analysis
pipeline; the whole container is never expanded at once.

## 5. Recommended first run

1. Start `app.py`.
2. In the sidebar, choose a local data folder.
3. Choose **Single file**, scan the folder, and select one ENVI header/data file.
4. Start with spatial downsampling `4` for a fast review.
5. Use **Hybrid** with the default thresholds and cluster count.
6. Confirm or create a reflectance calibration before treating the spectra as scientific reflectance.
7. Run one file and review its class map and report before starting a batch.

Analysis runs in a separate worker process, so the web page remains responsive.
While it is running, click **Stop Analysis** in the sidebar or run-status panel
to terminate that worker and its child processes. Files completed before the
stop remain in the output folder; do not treat a file that was being written at
the moment of cancellation as a final result. Elapsed time and the recent log
refresh about every two seconds.

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

The fitted profile is also graded `PASS`, `REVIEW`, or `FAIL` from panel
reconstruction error, ROI uniformity, coefficient agreement, adjacent-band
steps, and weight transitions. PASS/REVIEW may be connected to trial analysis;
FAIL is saved for audit but is not automatically applied. A synthetic constant
Dark receives at least REVIEW.

Calibration profiles are normally saved to:

```text
output/calibration/<source>_weighted_dark_calibration.npz
```

When a file is analysed, the source folder and `output/calibration` are searched
for conservatively named candidates. Band count, wavelength compatibility, and
QC status are checked before automatic application. An explicit profile
selected by the user takes priority, but FAIL remains blocked.

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

### Team / plot daily package

In batch mode, enable **Team / Plot Daily Package** and enter the actual
acquisition date and team name. The aggregator reuses compact per-file outputs;
it does not reopen the CERES/BIL cubes, so its additional memory cost is small.

```text
output/team_reports/<date>_<team>_<created>/
├── Team_Report.html
├── Field_Results.xlsx
├── Field_Summary.csv
├── Warnings.csv
├── plots_overview.png
├── plots_ndvi.png
├── plot_ndvi_comparison.png
├── Images/                  # copied plot images needed for sharing
└── Details/                 # copied per-plot HTML reports
```

- Every NDVI tile uses the same fixed `-1 to 1` colour scale.
- Plot medians and IQRs are compared; pixels from different plots are never pooled.
- Team statistics include only plots with `value_units=reflectance`,
  `calibration_qc_status=PASS`, and valid NDVI. REVIEW, FAIL, and uncalibrated
  plots remain visible but are listed in Warnings and excluded from pooled values.
- `Field_Results.xlsx` contains Dashboard, README, Field Summary, Cluster Summary,
  Reflectance Spectra, and Warnings sheets.

When filenames are not plot IDs, provide an optional metadata CSV:

```csv
filename,plot_id,treatment,genotype,replicate,team,measurement_date
scene_001.bil,AP3-4,Control,WT,1,Team A,2026-08-27
```

The output tables are **UTF-8 CSV files that open directly in Excel**, rather
than a single `.xlsx` workbook. Each spectral row represents one wavelength.

| File | Meaning |
|------|---------|
| `spectra_<method>_reflectance.csv` | Science-ready reflectance when valid calibration was applied |
| `spectra_<method>_raw_dn.csv` | Original sensor DN for diagnostics and before/after comparison |
| `spectra_<method>_processed.csv` | Normalized/processed relative values when no calibration is available; not absolute reflectance |
| `spectra_<method>.csv` | Main extraction from the current run; verify its scale in `value_units` |
| `daily_summary_*.csv` | Per-file NDVI, vegetation fraction, class count, quality metrics, and elapsed time |
| `all_roi_cluster_spectra*.csv` | Combined spectra for all ROIs and clusters |
| `cluster_summary.csv` | Pixel count and area fraction (`fraction`, 0–1) for each ROI cluster |

Each class contains:

- mean, standard deviation, median, 25th and 75th percentiles;
- Medoid-Neighbourhood Average (`mna`); and
- SAM-Neighbourhood Average (`sam_avg`).

ROI `cluster_spectra*` tables instead use long format with one row per
ROI × cluster × wavelength and contain `mean`, `median`, `std`, `q25`, and
`q75` for easier filtering and merging.

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
calibration_qc_status = PASS
```

`REVIEW` results require inspection of jump/saturation warnings; avoid using
`FAIL` results for scientific analysis.

RGB, cluster-map, overlay, and isolated-cluster images are shown in a larger
two-column report layout. Click any image for a full-window preview; click the
background/close button or press `Esc` to close it.

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

Open **ROI Analysis & Re-clustering** from the English navigation sidebar. The
standalone developer command below opens the Korean source page directly:

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

## 14. Browsing a CERES container safely

1. Choose **Local Folder → Single File → Scan Folder**, then select a `.ceres` file.
2. Click **Read CERES Contents**. Only record headers are scanned, producing logical
   entries such as `A/VNIR`, `A/SWIR`, and `B/VNIR`. The index is cached. Both
   the 2024 CBDF v1 and newer CBDF v2 layouts are recognized.
3. Select one entry and click **Quick Preview**. VNIR uses visible RGB and SWIR
   uses three false-color bands;
   neither the full CERES file nor a full hyperspectral cube is loaded into RAM.
4. Click **Prepare Selected Entry** only after confirming the acquisition. The software
   streams that sensor/segment alone into a uint16 BIL/HDR cache under
   `output/_ceres_cache` and connects it to the regular analysis pipeline.
5. Review the displayed peak-RAM estimates and choose spatial downsampling before running.

This release implements safe inspection, selection, preview, and selective extraction.
Fully automatic day-folder processing without materializing BIL requires a later two-pass
streaming workflow: sampled model fitting followed by chunked prediction and aggregation.

## 15. Calibration QC and clustering input

- New calibration profiles are graded `PASS`, `REVIEW`, or `FAIL` using panel
  reconstruction error, ROI uniformity, cross-panel coefficient agreement, adjacent-band
  seams, and abrupt panel-weight transitions.
- A `FAIL` profile is saved for audit but is blocked from automatic application.
- A synthetic constant Dark is explicitly recorded and receives at least `REVIEW`.
- K-Means uses raw-DN spectral structure by default. Its masks are then applied to the
  calibrated cube so both raw-DN and reflectance spectra are saved. Hybrid automatically
  uses reflectance when a valid calibration exists because its NDVI and brightness
  thresholds are defined on that scale.
- After each run, **Visual Cluster Review** shows the analysis RGB, color map, and
  adjustable overlay together. Select individual clusters, change opacity/boundaries,
  and open isolated-class images to inspect leaf, shadow, glare, and soil separation.

## 16. Current scope

The implemented product includes ENVI/BIL calibration, clustering, ROI spectra,
re-clustering, reporting, plus indexed CERES browsing and selective sensor/segment
extraction. Direct streaming day-folder CERES analysis and a configurable daily
index-image package remain the next integration stage.
