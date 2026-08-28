# Changelog

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
