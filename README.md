# Hyperspectral Field Crop Analysis

A Streamlit-based workflow for converting field hyperspectral imagery into
radiometrically traceable spectra, global cluster maps, ROI-level summaries,
and reviewable HTML reports.

The current workflow supports ENVI BIL/BIP/BSQ data, GeoTIFF, HDF5, and MATLAB
files. Large ENVI cubes are memory-mapped and can be spatially downsampled at
load time. Ceres acquisition files can first be demultiplexed with
`ceres_demux.py`.

## Main capabilities

- White/Dark or weighted multi-panel reflectance calibration
- Automatic discovery of a compatible calibration profile per image
- Raw-DN and calibrated-reflectance spectra saved side by side
- Hybrid, K-Means, SAM, HDBSCAN, GMM, NMF, Random Forest, autoencoder, and 1D-CNN methods
- One global clustering model with separate spectra for each user-defined ROI
- Box, lasso, and click-by-click polygon ROIs
- ROI-only re-clustering when the global result is locally unsatisfactory
- RGB overlays, cluster boundaries, isolated cluster images, CSV tables, and HTML reports
- Calibration provenance and per-wavelength empirical-line coefficients in exported CSV files

## Quick start

```bash
git clone https://github.com/hoonsoolee/Extract_spectra.git
cd Extract_spectra
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The browser normally opens at `http://localhost:8501`.

For the English core-analysis interface:

```bash
python -m streamlit run app_en.py
```

For the global-clustering/ROI workflow directly:

```bash
python -m streamlit run app_roi_clustering.py
```

## Documentation

- [English usage guide](USAGE_EN.md)
- [Korean usage guide](USAGE_KO.md)

## Scientific-use note

Raw DN is not reflectance. For quantitative or publication use, apply a
same-condition White/Dark or multi-panel calibration and verify that exported
tables report `value_units=reflectance`, `calibration_applied=true`, and
`normalization_mode=none`. A synthetic constant Dark is supported for rapid
screening, but a measured sensor Dark with matching acquisition settings is
recommended for final scientific results.

