# Hyperspectral Field Crop Analysis

Current release: **v2.0.0**

A Streamlit-based workflow for converting field hyperspectral imagery into
radiometrically traceable spectra, global cluster maps, ROI-level summaries,
and reviewable HTML reports.

The current workflow supports ENVI BIL/BIP/BSQ data, GeoTIFF, HDF5, and MATLAB
files. Large ENVI cubes are memory-mapped and can be spatially downsampled at
load time. The web UI can index, preview, and selectively demultiplex both the
2024 CBDF v1 and newer CBDF v2 variants of Ceres acquisition files.

## Main capabilities

- White/Dark or weighted multi-panel reflectance calibration
- Automatic discovery of a compatible calibration profile per image
- Raw-DN and calibrated-reflectance spectra saved side by side
- Hybrid, K-Means, SAM, HDBSCAN, GMM, NMF, Random Forest, autoencoder, and 1D-CNN methods
- One global clustering model with separate spectra for each user-defined ROI
- Box, lasso, and click-by-click polygon ROIs
- ROI-only re-clustering when the global result is locally unsatisfactory
- RGB overlays, cluster boundaries, isolated cluster images, CSV tables, and HTML reports
- Quick-QC, research-standard, and custom report presets with daily batch summaries
- Team/day plot packages with common-scale NDVI panels, QC-gated plot statistics,
  a shareable HTML report, and a multi-sheet Excel workbook
- Calibrated-reflectance / raw-DN ROI comparison with per-band coefficient diagnostics
- Calibration provenance and per-wavelength empirical-line coefficients in exported CSV files

## Quick start

```bash
git clone https://github.com/hoonsoolee/Extract_spectra.git
cd Extract_spectra
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The browser normally opens at `http://localhost:8501`.

For the full English research workflow (the same implementation and features):

```bash
python -m streamlit run app_en.py
```

The English app includes an **ROI Analysis & Re-clustering** link in its
sidebar. Korean users can also open the standalone developer entry directly:

```bash
python -m streamlit run app_roi_clustering.py
```

`app.py` and `app_en.py` share one analysis implementation. The English entry
translates only presentation text, so CERES support, calibration, ROI tools,
clustering, and reports cannot drift between languages.

## Documentation

- [English usage guide](USAGE_EN.md)
- [Korean usage guide](USAGE_KO.md)
- [Release notes](CHANGELOG.md)

## Scientific-use note

Raw DN is not reflectance. For quantitative or publication use, apply a
same-condition White/Dark or multi-panel calibration and verify that exported
tables report `value_units=reflectance`, `calibration_applied=true`, and
`normalization_mode=none`. A synthetic constant Dark is supported for rapid
screening, but a measured sensor Dark with matching acquisition settings is
recommended for final scientific results.
