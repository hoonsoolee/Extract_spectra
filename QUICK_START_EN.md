# Practical Quick Start — Hyperspectral Field Analysis

For Korean, open the [Korean practical quick start](quick_start_ko.html).

This guide is for a new laboratory user who needs to complete a real workflow
without reading the full technical manual first.

Start the English interface:

```powershell
cd D:\Research\Hyperspectral\Extract_Spectra_v2
python -m streamlit run app_en.py
```

The browser normally opens at `http://localhost:8501`.

## Before you begin

- Keep each ENVI header and binary file together, for example `plot_01.bil.hdr`
  and `plot_01.bil`.
- Use downsampling `4` for the first inspection. Use `1` only for the final run
  when memory and processing time are acceptable.
- Raw DN is not reflectance. Publication-ready spectra require a compatible
  calibration profile with `QC = PASS`.
- Test one file before starting a whole-folder batch.

## 1. Prepare and analyse one CERES file

Example source:

```text
D:\FieldData\2026-08-21\Plot_001.ceres
```

Click through this path in the left sidebar:

```text
Local Folder
  -> select D:\FieldData\2026-08-21
  -> Single File
  -> Scan Folder
  -> Plot_001.ceres
  -> Read CERES Contents
  -> choose acquisition/sensor, for example A / VNIR
  -> Quick Preview
  -> Prepare Selected Entry
```

What happens:

1. **Read CERES Contents** reads record headers and builds a small index. It
   does not load the entire CERES file into RAM.
2. **Quick Preview** reads only three bands for an RGB or false-colour preview.
3. **Prepare Selected Entry** streams only the selected acquisition/sensor to
   an ENVI `uint16` BIL/HDR pair under `output/_ceres_cache/`.
4. Wait for the green message confirming that the prepared BIL/HDR is connected
   to the regular analysis input.
5. Choose downsampling, clustering, calibration, and report options, then click
   **Start Analysis**.

Important limitation: the current version safely prepares **one selected
CERES entry at a time**. It does not yet extract every internal entry from every
CERES file automatically. Whole-folder batch analysis should be run on prepared
BIL/HDR files.

## 2. Load a BIL or another hyperspectral file and inspect spectra

### ENVI file example

```text
D:\FieldData\Plot_001.bil
D:\FieldData\Plot_001.bil.hdr
```

Keep both files in the same folder. In folder-scan workflows, select the small
`.hdr` entry.

### Inspect an ROI spectrum

1. Open **ROI Spectra** in the main area.
2. Select the hyperspectral file. If it is not listed, choose **Enter Path
   Manually** or use **Browse**.
3. Select downsampling `4` and click **Load**.
4. Read the calibration banner:
   - **Reflectance calibration applied**: spectra can be viewed as reflectance.
   - **Raw DN**: values are sensor counts and are not absolute reflectance.
5. Use **Zoom** first if the leaf or plot is small.
6. Select an ROI with **Box**, **Lasso**, or **Click Polygon**.
7. Inspect mean, median, standard deviation, and the displayed wavelength range.
8. If calibration is active, switch among **Calibrated Reflectance**, **Raw DN**,
   and **Raw/Calibrated Comparison**.
9. Set the CSV path and click **Save CSV**. Calibrated and raw-DN companion CSVs
   are saved when calibration is active.

Example interpretation:

```text
A healthy green leaf normally has low red reflectance near 660 nm and a strong
rise toward the NIR. A sudden step near a sensor boundary or an implausible
value above 1 should trigger calibration review, not automatic smoothing.
```

## 3. Obtain and verify useful clustering

### Recommended first trial

```text
Processing Mode: Single File
Downsampling: 4
Method: K-Means
Number of clusters: 6
Report preset: Quick Field QC
```

K-Means is a good first check because it works with VNIR or SWIR data and does
not require NDVI. The default workflow clusters spectral structure from raw DN,
then applies the same cluster masks to calibrated data when extracting
reflectance spectra. **Hybrid** uses reflectance when valid calibration is
available because its NDVI and brightness thresholds are physically scaled.

After the run, open **Visual Cluster Review** and compare:

- RGB image;
- cluster colour map;
- adjustable RGB/cluster overlay;
- boundaries; and
- isolated images for individual clusters.

The clustering is useful when leaves, soil, deep shadow, and strong illumination
form spatially coherent regions. Cluster numbers have no biological meaning by
themselves; interpret each cluster from its location and spectrum.

| Problem in the review image | First adjustment |
|---|---|
| One leaf is split into too many tiny regions | Reduce the cluster count |
| Sunlit and shaded leaves are merged | Increase the cluster count, or use Hybrid and tune brightness |
| Soil and vegetation are merged in VNIR data | Use Hybrid and review the NDVI threshold |
| SWIR scene has no valid red/NIR pair | Use K-Means or SAM, not Hybrid |
| One local plot is poor but the rest is good | Open ROI Analysis & Re-clustering and retry only that ROI |
| Bright panel/glare pixels are saturated | Correct acquisition or mask them; more clusters will not restore lost signal |

For a final run, use the selected settings on the full-resolution file or batch.
Do not choose a method only because its colour map looks attractive; also check
the spectra, cluster area, and stability across similar plots.

## 4. Build and apply a White/Dark reflectance calibration

In this program, “white balance” means a wavelength-specific empirical-line
reflectance calibration:

```text
Reflectance = a(wavelength) x DN + b(wavelength)
```

### Recommended inputs

- a reference-panel image acquired close in time to the crop scene;
- certified panel reflectance values such as `0.99`, `0.50`, and `0.25`;
- a measured sensor Dark acquired with the lens covered or shutter closed; and
- the same sensor, bands, integration time, and gain as the crop acquisition.

### Build the profile

1. Open **Panel Calibration**.
2. Select the panel/reference image, start with downsampling `4`, and click
   **Load**.
3. Drag an ROI inside a clean, uniform part of a panel. Avoid panel edges,
   shadows, dirt, and glare.
4. Enter the certified reflectance as a fraction: `99% = 0.99`, `50% = 0.50`.
5. Click **Add This Region as Panel**. Repeat for each panel.
6. Under **Sensor Dark**, choose **Measured Dark File**, select the matching
   Dark image, and click **Load Dark**. Constant `100 DN` is available only for
   quick screening when no measured Dark exists.
7. Click **Calculate Automatic Reflectance Calibration**.
8. Review the reconstructed panel spectra, wavelength-dependent panel weights,
   invalid bands, and the QC grade.

The program does not change a 50% panel into 99%. It uses `0.50` as its known
reflectance. If the 99% panel is saturated only at some wavelengths, its weight
becomes zero there and a valid lower-reflectance panel can support those bands.

### QC and automatic application

| QC result | Meaning and action |
|---|---|
| `PASS` | Automatically connected and suitable for analysis after visual review |
| `REVIEW` | Connected for trial analysis; inspect warnings before scientific use |
| `FAIL` | Saved for audit but blocked from automatic whole-field application |

The profile is saved under:

```text
output/calibration/<panel-image>_weighted_dark_calibration.npz
```

It becomes active immediately when QC permits. On later runs, the program
searches the source folder and `output/calibration/` for a compatible profile.
You can also select a profile explicitly in the sidebar. After loading a crop
scene, always verify the green calibration banner and, in exported data:

```text
value_units = reflectance
calibration_applied = true
calibration_qc_status = PASS
normalization_mode = none
```

## 5. Understand and share the saved results

Typical per-file result folder:

```text
output/<source-name>/
|-- report_<timestamp>_<method>.html
|-- spectra_<method>.csv
|-- spectra_<method>_reflectance.csv
|-- spectra_<method>_raw_dn.csv
|-- spectra_<method>_processed.csv
|-- processing_manifest.json
|-- report_config.json
|-- rgb.png
|-- ndvi.png
|-- class_map_<method>.png
|-- cluster_map.png
|-- cluster_overlay.png
`-- cluster_review.npz
```

| Result | Use it for |
|---|---|
| `report_*.html` | Main visual review and the easiest file to share with the field team |
| `spectra_*_reflectance.csv` | Scientific reflectance analysis after confirming PASS calibration |
| `spectra_*_raw_dn.csv` | Diagnosing acquisition and calibration differences |
| `spectra_*_processed.csv` | Relative/normalized exploration when no reflectance calibration exists |
| `processing_manifest.json` | Reproducibility: source, calibration, normalization, method, and parameters |
| `report_config.json` | Which report preset, images, statistics, and indices were requested |
| `cluster_map/overlay/isolated images` | Checking whether clustering is spatially sensible |
| `daily_report_*.html` and `daily_summary_*.csv` | Comparing all files from one batch |
| `Field_Results.xlsx` | Team/day dashboard, field summary, cluster summary, spectra, and warnings |

For a simple research-team delivery, share:

1. the per-file or team HTML report;
2. `Field_Results.xlsx` for a day-level batch, when generated;
3. the calibrated reflectance CSV needed for later statistics; and
4. `processing_manifest.json` for scientific traceability.

Before sharing a science-ready result, confirm:

- calibration QC is `PASS`;
- the report says `Reflectance`, not `Raw DN`;
- required vegetation indices were actually calculated;
- the cluster overlay matches meaningful image regions; and
- no unexplained spectral jump or saturation warning remains.

## Common problems

| Symptom | Check |
|---|---|
| ENVI/BIL will not load | Keep the binary and `.hdr` together and select the header |
| CERES analysis cannot start | Run **Prepare Selected Entry** after selecting an internal entry |
| Reflectance banner is missing | Profile may be absent, incompatible, or QC FAIL |
| NDVI is unavailable | The image may lack calibrated red and NIR wavelengths |
| Spectrum jumps near one wavelength range | Inspect calibration QC, panel weights, saturation, and sensor boundaries |
| Run is taking too long | Use downsampling `4`, test one file, or click **Stop Analysis** |

For algorithms, thresholds, CSV column definitions, and troubleshooting details,
open [the complete English manual](manual.html) or read [USAGE_EN.md](USAGE_EN.md).
