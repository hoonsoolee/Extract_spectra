# Changelog

## Unreleased

### Added

- Adopted the **CanopySpectra** product name and the tagline “From CERES to
  Science-Ready Field Spectra” across the application, reports, manuals, quick
  starts, CLI, and Korean/English presentation materials.
- Optional compressed `spectral_samples.h5` output containing a reproducible,
  bounded sample of actual spectra from every final cluster, together with
  coordinates, Hybrid base classes, raw DN, sampling weights, and provenance.
- Korean and English UI controls and report/manual explanations for using the
  samples as plot-level spectral sets rather than independent pixel labels.

## v2.0.0 — 2026-08-27

### Added

- Low-memory CERES/CBDF browser with cached indexing, quick preview, and
  selective sensor/segment extraction to ENVI BIL.
- Support for both 2024 CBDF v1 and newer CBDF v2 image-record layouts.
- Global clustering with separate ROI spectra, polygon/lasso/box ROIs, and
  ROI-only re-clustering with before/after comparison.
- White/Dark and saturation-aware weighted multi-panel reflectance calibration,
  calibration provenance, QC grading, and raw/calibrated spectrum comparison.
- Configurable HTML reports, enlarged cluster-review imagery, daily summaries,
  timing estimates, and team/day plot packages with common-scale NDVI panels
  and a multi-sheet Excel workbook.
- Full English presentation of the same implementation used by the Korean UI.

### Scientific safeguards

- Science-ready team statistics require calibrated reflectance, calibration QC
  `PASS`, and a valid NDVI result.
- Raw DN, normalized values, calibrated reflectance, profile identity, and QC
  status are recorded explicitly in exported results.
- Team/day aggregation uses plots as statistical units and does not pool pixels
  from different plots.

### Current scope

- CERES files are indexed directly, but the selected sensor/segment is streamed
  to BIL before the existing full analysis pipeline runs. Direct chunked
  clustering of an entire day folder without intermediate BIL remains future
  work.
