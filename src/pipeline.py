"""
pipeline.py
-----------
Main processing pipeline: orchestrates loader → preprocessor →
classifier → extractor → reporter for a batch of hyperspectral files.
"""

import logging
import json
import csv
import time
import traceback
import math
from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Dict, Any

from .data_loader import HyperspectralLoader
from .preprocessor import Preprocessor
from .classifier import HyperspectralClassifier
from .spectrum_extractor import SpectrumExtractor
from .reporter import Reporter
from .evaluator import Evaluator
from .spectral_indices import compute_selected_indices

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Batch processing pipeline for hyperspectral field images.

    Usage
    -----
    pipeline = Pipeline(config)
    pipeline.run()
    """

    def __init__(self, config: dict):
        self.config  = config
        # Give the loader the downsample factor so huge ENVI cubes are
        # subsampled at read time (memmap) instead of after a full load.
        loader_cfg = dict(config.get("data", {}))
        loader_cfg["spatial_downsample"] = \
            config.get("preprocessing", {}).get("spatial_downsample", 1)
        self.loader  = HyperspectralLoader(loader_cfg)
        self.prep    = Preprocessor(config)
        self.clf     = HyperspectralClassifier(config)
        self.extr    = SpectrumExtractor(config)
        _lang = config.get("report", {}).get("lang", "ko")
        self.reporter = Reporter(config, lang=_lang)
        self.batch_summaries: List[dict] = []
        self.team_packages: List[dict] = []
        self.last_cluster_input_space = "processed"

        out_cfg = config.get("output", {})
        self.output_dir = Path(out_cfg.get("dir", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Public
    # ============================================================

    def run(
        self,
        labels_csv: Optional[str] = None,
        file_limit: Optional[int] = None,
        single_file: Optional[str] = None,
    ) -> None:
        """
        Discover and process all files from the configured source(s).

        Parameters
        ----------
        labels_csv  : path to labelled-pixels CSV (for supervised method)
        file_limit  : stop after this many files (useful for testing)
        single_file : if given, process only this specific local file path
                      (skips discovery; file_limit is ignored)
        """
        data_cfg = self.config.get("data", {})
        out_cfg  = self.config.get("output", {})
        per_file_report = out_cfg.get("per_file_report", False)
        self.batch_summaries = []
        self.team_packages = []
        self.reporter.results.clear()

        # ── File discovery ─────────────────────────────────────────
        if single_file:
            all_tasks = [("local", Path(single_file))]
        else:
            files_local  = self._discover_local(data_cfg)
            files_github = self._discover_github(data_cfg)
            all_tasks = [("local", f) for f in files_local] + \
                        [("github", f) for f in files_github]

        if not all_tasks:
            logger.warning(
                "No files found. "
                "Set data.local_folder or data.github.repo in config.yaml"
            )
            return

        if file_limit and not single_file:
            all_tasks = all_tasks[:file_limit]

        logger.info(f"Total files to process: {len(all_tasks)}")

        ok, failed = 0, 0
        t0 = time.time()

        for i, (source, file_ref) in enumerate(all_tasks, 1):
            fname = Path(file_ref).name
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{len(all_tasks)}] {fname}  (source: {source})")
            logger.info(f"{'='*60}")

            try:
                summary = self._process_file(source, file_ref, data_cfg, labels_csv)
                ok += 1

                # ── Per-file report (batch mode with individual reports) ──
                if per_file_report and out_cfg.get("save_report", True) \
                        and self.reporter.results:
                    stem         = Path(file_ref).stem
                    ts           = time.strftime("%Y%m%d_%H%M%S")
                    method       = self.config.get("classification", {}).get("method", "unknown")
                    file_out_dir = self.output_dir / stem
                    file_out_dir.mkdir(parents=True, exist_ok=True)
                    detail_report = file_out_dir / f"report_{ts}_{method}.html"
                    self.reporter.render(detail_report)
                    summary["detail_report"] = str(detail_report.resolve())
                    self.reporter.results.clear()   # reset for next file

                self.batch_summaries.append(summary)

            except Exception as e:
                logger.error(f"FAILED: {fname}\n{traceback.format_exc()}")
                failed += 1

        elapsed = time.time() - t0
        logger.info(
            f"\nBatch complete: {ok} ok / {failed} failed / "
            f"{len(all_tasks)} total  ({elapsed:.1f}s)"
        )

        # ── Final report ───────────────────────────────────────────
        # • single_file mode  → report goes into output/{stem}/
        # • combined batch    → report goes into output/
        # • per_file_report   → already rendered above; nothing to do
        if not per_file_report and out_cfg.get("save_report", True) \
                and self.reporter.results:
            ts     = time.strftime("%Y%m%d_%H%M%S")
            method = self.config.get("classification", {}).get("method", "unknown")
            if single_file:
                stem       = Path(single_file).stem
                report_dir = self.output_dir / stem
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = report_dir / f"report_{ts}_{method}.html"
            else:
                report_path = self.output_dir / f"report_{ts}_{method}.html"
            self.reporter.render(report_path)
            if len(self.batch_summaries) == 1:
                self.batch_summaries[0]["detail_report"] = str(report_path.resolve())

        if (
            not single_file
            and self.batch_summaries
            and self.reporter.options.get("daily_summary", True)
        ):
            ts = time.strftime("%Y%m%d_%H%M%S")
            csv_path = self.output_dir / f"daily_summary_{ts}.csv"
            html_path = self.output_dir / f"daily_report_{ts}.html"
            columns = [
                "filename", "source_file", "value_units", "calibration_profile",
                "calibration_qc_status", "n_classes", "ndvi_mean", "ndvi_median",
                "ndvi_q25", "ndvi_q75", "vegetation_fraction", "silhouette",
                "davies_bouldin", "elapsed_seconds", "detail_report",
                "spectral_samples_file",
            ]
            with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.batch_summaries)
            self.reporter.render_daily_summary(self.batch_summaries, html_path)

        team_config = self.config.get("report", {}).get("team_daily", {})
        if (
            not single_file
            and self.batch_summaries
            and team_config.get("enabled", False)
        ):
            try:
                from .team_report import generate_team_daily_packages

                self.team_packages = generate_team_daily_packages(
                    self.batch_summaries,
                    self.output_dir,
                    team_config,
                )
            except Exception:
                logger.error(
                    "Team/day report generation failed\n%s", traceback.format_exc()
                )

    # ============================================================
    # Process a single file
    # ============================================================

    def _process_file(
        self,
        source: str,
        file_ref,
        data_cfg: dict,
        labels_csv: Optional[str],
    ) -> dict:
        fname    = Path(file_ref).name
        stem     = Path(file_ref).stem
        t_file   = time.time()   # wall-clock start for total elapsed

        # ---- 1. Load ----
        t0 = time.time()
        if source == "local":
            data, meta = self.loader.load_local(file_ref)
        else:
            gh = data_cfg.get("github", {})
            data, meta = self.loader.load_github(
                repo=gh["repo"],
                file_path=file_ref,
                token=gh.get("token"),
            )
        logger.info(f"  Load: {time.time()-t0:.2f}s")
        raw_data = data
        raw_wavelengths = meta.get("wavelengths")

        # ---- 2. Preprocess ----
        t0 = time.time()
        data, wavelengths = self.prep.process(
            data, meta.get("wavelengths"),
            skip_downsample=meta.get("downsample_applied", 1) > 1,
            source_path=str(file_ref),
        )
        logger.info(f"  Preprocess: {time.time()-t0:.2f}s")

        # ENVI is downsampled by the memory-mapped loader, while TIFF/HDF5/MAT
        # cubes are downsampled by the preprocessor.  Keep a spatially aligned
        # raw-DN view so the same class mask and sampled pixel coordinates can
        # be applied to both products.
        raw_analysis_data = raw_data
        spatial_factor = max(
            1,
            int(self.config.get("preprocessing", {}).get("spatial_downsample", 1)),
        )
        if raw_analysis_data.shape[:2] != data.shape[:2]:
            candidate = raw_analysis_data[::spatial_factor, ::spatial_factor, :]
            if candidate.shape[:2] != data.shape[:2]:
                raise ValueError(
                    "Raw and processed cube dimensions cannot be aligned: "
                    f"raw={raw_analysis_data.shape[:2]}, processed={data.shape[:2]}, "
                    f"downsample={spatial_factor}"
                )
            raw_analysis_data = candidate

        # ---- 3. Classify ----
        t0 = time.time()
        method = str(
            self.config.get("classification", {}).get("method", "kmeans")
        ).lower()
        requested_cluster_space = str(
            self.config.get("classification", {}).get("input_space", "auto")
        ).lower()
        if requested_cluster_space not in {"auto", "raw", "reflectance"}:
            requested_cluster_space = "auto"

        # Simple/reproducible default: use scene-independent raw spectral
        # structure for clustering, then apply the resulting masks to the
        # calibrated cube for science-ready spectra and indices. Hybrid is the
        # exception because its published thresholds are defined on NDVI and
        # reflectance brightness; when calibration is available it uses it.
        use_reflectance = requested_cluster_space == "reflectance" or (
            requested_cluster_space == "auto"
            and method == "hybrid"
            and self.prep.last_calibration_info is not None
        )
        if use_reflectance:
            cluster_data, cluster_wavelengths = data, wavelengths
            self.last_cluster_input_space = (
                "reflectance" if self.prep.last_calibration_info else "processed"
            )
        else:
            cluster_config = deepcopy(self.config)
            cluster_preprocessing = cluster_config.setdefault("preprocessing", {})
            cluster_preprocessing["calibration_file"] = None
            cluster_preprocessing["auto_discover_calibration"] = False
            cluster_preprocessing["normalize"] = True
            cluster_preprocessing["normalize_mode"] = "global"
            cluster_preprocessing["spatial_downsample"] = 1
            cluster_preprocessor = Preprocessor(cluster_config)
            cluster_data, cluster_wavelengths = cluster_preprocessor.process(
                raw_analysis_data,
                raw_wavelengths,
                skip_downsample=True,
                source_path=str(file_ref),
            )
            self.last_cluster_input_space = "raw DN (global scale)"
        logger.info(f"  Clustering input: {self.last_cluster_input_space}")
        class_map, class_info = self.clf.classify(
            cluster_data, cluster_wavelengths, labels_csv
        )
        logger.info(f"  Classify: {time.time()-t0:.2f}s")

        # ---- 3b. Quality metrics ----
        t0 = time.time()
        report_sections = self.reporter.options["sections"]
        metrics: Dict[str, Any] = {}
        if report_sections.get("quality_metrics"):
            metrics = Evaluator.unsupervised_metrics(cluster_data, class_map)
            sil = metrics.get("silhouette")
            sil_str = f"{sil:.3f}" if sil is not None else "N/A"
            logger.info(
                f"  Silhouette: {sil_str}  "
                f"DB: {metrics.get('davies_bouldin') or 'N/A'}  "
                f"-> {metrics.get('interpretation', '')}"
            )
            # Merge supervised validation accuracy if available
            if self.clf.last_val_metrics:
                metrics.update(self.clf.last_val_metrics)
                acc = self.clf.last_val_metrics.get("accuracy")
                if acc is not None:
                    logger.info(
                        f"  Supervised val accuracy: {acc:.3f}  "
                        f"F1: {self.clf.last_val_metrics.get('macro_f1', 'N/A')}"
                    )
        else:
            logger.info("  Quality metrics skipped by report selection")
        logger.info(f"  Evaluate: {time.time()-t0:.2f}s")

        # ---- 4. Extract spectra ----
        t0 = time.time()
        spectra = self.extr.extract(data, class_map, class_info, wavelengths)
        raw_spectra = self.extr.extract(
            raw_analysis_data, class_map, class_info, raw_wavelengths
        )
        logger.info(f"  Extract spectra: {time.time()-t0:.2f}s")

        # ---- 4b. Spectral separability ----
        sep = (
            Evaluator.spectral_separability(spectra)
            if report_sections.get("quality_metrics") else {}
        )

        # ---- 4c. Vegetation separation quality ----
        veg_sep = (
            Evaluator.vegetation_separation_metrics(
                data, class_map, spectra, wavelengths
            )
            if report_sections.get("vegetation_quality") else {}
        )
        if veg_sep.get("ndvi_f1") is not None:
            logger.info(
                f"  Vegetation F1: {veg_sep['ndvi_f1']:.3f}  "
                f"Recall: {veg_sep['ndvi_recall']:.3f}  "
                f"Precision: {veg_sep['ndvi_precision']:.3f}"
            )

        index_results: Dict[str, Dict[str, Any]] = {}
        if report_sections.get("spectral_indices"):
            index_results = compute_selected_indices(
                data,
                wavelengths,
                self.reporter.options.get("indices", []),
                is_reflectance=bool(self.prep.last_calibration_info),
            )

        # ---- 5. Save outputs ----
        out_cfg = self.config.get("output", {})
        method   = self.config.get("classification", {}).get("method", "unknown")
        file_out_dir = self.output_dir / stem
        file_out_dir.mkdir(parents=True, exist_ok=True)

        calibration_info = self.prep.last_calibration_info
        effective_normalization = self.prep.last_effective_normalize_mode
        analysis_value_units = (
            "reflectance"
            if calibration_info
            else "raw DN" if effective_normalization == "none"
            else f"normalized ({effective_normalization})"
        )

        if out_cfg.get("save_spectra_csv", True):
            csv_name = f"spectra_{method}.csv"
            corrected_provenance = {
                "source_file": str(file_ref),
                "value_units": analysis_value_units,
                "normalization_mode": effective_normalization,
                "calibration_info": calibration_info,
                "calibration_applied": bool(calibration_info),
                "coefficients_a": (
                    calibration_info.get("a") if calibration_info else None
                ),
                "coefficients_b": (
                    calibration_info.get("b") if calibration_info else None
                ),
            }
            raw_provenance = {
                "source_file": str(file_ref),
                "value_units": "raw DN",
                "normalization_mode": "none",
                "calibration_info": calibration_info,
                "calibration_applied": False,
            }
            # Column prefix includes both image stem and method so merged
            # CSVs from different files/methods never collide.
            self.extr.save_csv(spectra, file_out_dir / csv_name,
                               file_stem=f"{stem}_{method}",
                               provenance=corrected_provenance)
            self.extr.save_csv(
                raw_spectra,
                file_out_dir / f"spectra_{method}_raw_dn.csv",
                file_stem=f"{stem}_{method}_raw_dn",
                provenance=raw_provenance,
            )
            processed_kind = (
                "reflectance"
                if calibration_info
                else "processed"
            )
            self.extr.save_csv(
                spectra,
                file_out_dir / f"spectra_{method}_{processed_kind}.csv",
                file_stem=f"{stem}_{method}_{processed_kind}",
                provenance=corrected_provenance,
            )

        manifest = {
            "source_file": str(Path(file_ref).resolve()) if source == "local" else str(file_ref),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "method": method,
            "clustering_input": self.last_cluster_input_space,
            "normalization": self.prep.last_effective_normalize_mode,
            "requested_normalization": self.config.get("preprocessing", {}).get("normalize_mode", "global"),
            "calibration": self.prep.last_calibration_info,
            "value_units": analysis_value_units,
            "preprocessing_config": deepcopy(self.config.get("preprocessing", {})),
            "classification_config": deepcopy(self.config.get("classification", {})),
            "extraction_config": deepcopy(self.config.get("extraction", {})),
            "raw_spectra_file": (
                f"spectra_{method}_raw_dn.csv"
                if out_cfg.get("save_spectra_csv", True) else None
            ),
            "processed_spectra_file": (
                f"spectra_{method}.csv"
                if out_cfg.get("save_spectra_csv", True) else None
            ),
            "reflectance_spectra_file": (
                f"spectra_{method}_reflectance.csv"
                if self.prep.last_calibration_info
                and out_cfg.get("save_spectra_csv", True) else None
            ),
            "report_options": self.reporter.options,
            "spectral_indices": {
                name: {
                    key: value
                    for key, value in result.items()
                    if key != "values"
                }
                for name, result in index_results.items()
            },
        }

        sample_cfg = self.config.get("extraction", {}).get("sample_export", {})
        sample_export = {"enabled": bool(sample_cfg.get("enabled", False))}
        if sample_export["enabled"]:
            try:
                from .spectral_samples import export_spectral_samples

                calibration_meta = (calibration_info or {}).get("meta") or {}
                sample_export = {
                    "enabled": True,
                    "status": "completed",
                    **export_spectral_samples(
                        file_out_dir / "spectral_samples.h5",
                        analysis_data=data,
                        raw_data=raw_analysis_data,
                        class_map=class_map,
                        class_info=class_info,
                        analysis_wavelengths=wavelengths,
                        raw_wavelengths=raw_wavelengths,
                        base_class_map=getattr(self.clf, "last_base_class_map", None),
                        max_per_class=int(sample_cfg.get("max_per_class", 1_000)),
                        random_state=int(sample_cfg.get("random_state", 42)),
                        spatial_downsample=spatial_factor,
                        value_units=analysis_value_units,
                        save_raw=bool(sample_cfg.get("save_raw", True)),
                        provenance={
                            "source_file": manifest["source_file"],
                            "method": method,
                            "clustering_input": self.last_cluster_input_space,
                            "calibration_profile": (
                                (calibration_info or {}).get("selected_profile") or ""
                            ),
                            "calibration_qc_status": str(
                                calibration_meta.get("qc_status") or "UNASSESSED"
                            ).upper(),
                            "calibration": calibration_info or {},
                            "preprocessing_config": self.config.get(
                                "preprocessing", {}
                            ),
                            "classification_config": self.config.get(
                                "classification", {}
                            ),
                        },
                    ),
                }
                logger.info(
                    "  Model spectral samples saved: %s spectra",
                    f"{sample_export['n_samples']:,}",
                )
            except Exception as exc:
                sample_export = {
                    "enabled": True,
                    "status": "failed",
                    "error": str(exc),
                }
                logger.error("  Spectral sample export failed: %s", exc)
        manifest["spectral_sample_export"] = sample_export
        (file_out_dir / "processing_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (file_out_dir / "report_config.json").write_text(
            json.dumps(self.reporter.options, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if out_cfg.get("save_classification_map", True):
            self._save_class_map(
                class_map, class_info,
                file_out_dir / f"class_map_{method}.png"
            )

        cluster_review_assets = []
        team_daily_enabled = bool(
            self.config.get("report", {}).get("team_daily", {}).get("enabled", False)
        )
        if out_cfg.get("save_cluster_review", True) or team_daily_enabled:
            cluster_review_assets = self._save_cluster_review_assets(
                file_out_dir,
                data,
                wavelengths,
                class_map,
                class_info,
            )

        report_assets = self.reporter.save_selected_assets(
            file_out_dir,
            data=data,
            class_map=class_map,
            class_info=class_info,
            wavelengths=wavelengths,
            index_results=index_results,
        )
        team_ndvi_path = ""
        ndvi_values = (index_results.get("NDVI") or {}).get("values")
        if team_daily_enabled and ndvi_values is not None:
            from PIL import Image
            import numpy as np

            team_ndvi_target = file_out_dir / "team_ndvi.png"
            team_ndvi_rgb = self.reporter._index_map_array(ndvi_values)
            Image.fromarray(
                np.round(np.clip(team_ndvi_rgb, 0.0, 1.0) * 255.0).astype(np.uint8),
                mode="RGB",
            ).save(team_ndvi_target)
            team_ndvi_path = str(team_ndvi_target.resolve())
        manifest["report_assets"] = report_assets
        manifest["cluster_review_assets"] = cluster_review_assets

        # ---- 6. Add to report ----
        elapsed_sec = time.time() - t_file
        logger.info(f"  Total elapsed: {elapsed_sec:.1f}s")
        manifest["elapsed_seconds"] = elapsed_sec
        manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (file_out_dir / "processing_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if out_cfg.get("save_report", True):
            self.reporter.add_result(
                filename=fname,
                data=data,
                class_map=class_map,
                class_info=class_info,
                spectra=spectra,
                wavelengths=wavelengths,
                metadata={
                    **meta,
                    "calibration": self.prep.last_calibration_info,
                    "spectral_sample_export": sample_export,
                },
                metrics=metrics,
                separability=sep,
                veg_sep=veg_sep,
                elapsed_sec=elapsed_sec,
                index_results=index_results,
            )

        logger.info(f"  Outputs saved to: {file_out_dir}")
        calibration_info = self.prep.last_calibration_info or {}
        calibration_meta = calibration_info.get("meta") or {}
        calibration_qc_status = str(
            calibration_meta.get("qc_status") or "UNASSESSED"
        ).upper()
        ndvi = index_results.get("NDVI") or {}
        ndvi_summary = ndvi.get("summary") or {}
        profile = calibration_info.get("selected_profile") or ""
        science_ready = (
            manifest["value_units"] == "reflectance"
            and calibration_qc_status == "PASS"
        )
        overlay_image = next(
            (
                path for path in cluster_review_assets
                if Path(path).name == "cluster_review_overlay.png"
            ),
            "",
        )
        return {
            "filename": fname,
            "source_file": str(Path(file_ref).resolve()) if source == "local" else str(file_ref),
            "value_units": manifest["value_units"],
            "calibration_profile": Path(str(profile)).name if profile else "",
            "calibration_qc_status": calibration_qc_status,
            "n_classes": len(class_info),
            "ndvi_mean": ndvi_summary.get("mean", ""),
            "ndvi_median": ndvi_summary.get("median", ""),
            "ndvi_q25": ndvi_summary.get("q25", ""),
            "ndvi_q75": ndvi_summary.get("q75", ""),
            "vegetation_fraction": ndvi_summary.get("fraction_above_0_15", ""),
            "silhouette": metrics.get("silhouette", "") if metrics else "",
            "davies_bouldin": metrics.get("davies_bouldin", "") if metrics else "",
            "elapsed_seconds": elapsed_sec,
            "detail_report": "",
            "result_dir": str(file_out_dir.resolve()),
            "overlay_image": overlay_image,
            "ndvi_image": team_ndvi_path,
            "cluster_summary": [
                {
                    "class_id": int(item.get("id", index)),
                    "class_name": str(item.get("name", f"Cluster {index}")),
                    "pixel_count": int(item.get("n_pixels", 0)),
                }
                for index, item in enumerate(class_info)
            ],
            "team_spectra": (
                self._team_spectra_payload(spectra) if science_ready else []
            ),
            "cluster_review_file": str(
                (file_out_dir / "cluster_review.npz").resolve()
            ) if cluster_review_assets else "",
            "spectral_samples_file": (
                str(Path(sample_export["file"]).resolve())
                if sample_export.get("status") == "completed"
                and sample_export.get("file") else ""
            ),
        }

    # ============================================================
    # File discovery
    # ============================================================

    def _discover_local(self, data_cfg: dict) -> List[Path]:
        folder = data_cfg.get("local_folder")
        if not folder:
            return []
        try:
            files = self.loader.list_local_files(folder)
            logger.info(f"Local files found: {len(files)}")
            return files
        except FileNotFoundError as e:
            logger.error(str(e))
            return []

    def _discover_github(self, data_cfg: dict) -> List[str]:
        gh = data_cfg.get("github", {})
        repo = gh.get("repo")
        if not repo:
            return []
        try:
            files = self.loader.list_github_files(
                repo=repo,
                folder=gh.get("folder", ""),
                token=gh.get("token"),
            )
            logger.info(f"GitHub files found: {len(files)}")
            return files
        except Exception as e:
            logger.error(f"GitHub listing failed: {e}")
            return []

    # ============================================================
    # Helpers
    # ============================================================

    def _save_class_map(self, class_map, class_info, path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        H, W = class_map.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for c in class_info:
            rgb[class_map == c["id"]] = c["color"]

        fig, ax = plt.subplots(figsize=(max(6, W / 100), max(6, H / 100)), dpi=100)
        ax.imshow(rgb)
        ax.axis("off")

        # Legend patches
        import matplotlib.patches as mpatches
        patches = [
            mpatches.Patch(
                facecolor=[v / 255 for v in c["color"]],
                label=f"{c['name']} ({100*c['fraction']:.1f}%)",
            )
            for c in class_info
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=7,
                  framealpha=0.9, borderpad=0.5)
        fig.tight_layout(pad=0.3)
        fig.savefig(str(path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Class map saved: {path.name}")

    def _save_cluster_review_assets(
        self,
        output_dir: Path,
        data,
        wavelengths,
        class_map,
        class_info,
    ) -> list[str]:
        """Persist the arrays needed for an interactive visual cluster audit."""
        import numpy as np
        from PIL import Image

        output_dir.mkdir(parents=True, exist_ok=True)
        rgb = self.reporter._get_rgb_array(data, wavelengths, "rgb")
        color_map = self.reporter._make_class_map_array(class_map, class_info)
        overlay = self.reporter._make_cluster_overlay_array(
            rgb, class_map, class_info, alpha=0.55
        )

        def save_rgb(name: str, values) -> Path:
            target = output_dir / name
            encoded = np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(encoded, mode="RGB").save(target)
            return target

        rgb_path = save_rgb("cluster_review_rgb.png", rgb)
        map_path = save_rgb("cluster_review_map.png", color_map)
        overlay_path = save_rgb("cluster_review_overlay.png", overlay)
        archive_path = output_dir / "cluster_review.npz"
        np.savez_compressed(
            archive_path,
            class_map=np.asarray(class_map, dtype=np.int32),
            class_ids=np.asarray([item["id"] for item in class_info], dtype=np.int32),
            class_names=np.asarray([str(item["name"]) for item in class_info]),
            class_colors=np.asarray([item["color"] for item in class_info], dtype=np.uint8),
        )
        logger.info("  Interactive cluster review assets saved")
        return [
            str(path.resolve())
            for path in (rgb_path, map_path, overlay_path, archive_path)
        ]

    @staticmethod
    def _team_spectra_payload(spectra: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep a compact JSON-safe subset for the team/day workbook."""

        def values(array) -> list[float | None]:
            result: list[float | None] = []
            for item in array if array is not None else []:
                try:
                    number = float(item)
                except (TypeError, ValueError):
                    result.append(None)
                    continue
                result.append(number if math.isfinite(number) else None)
            return result

        payload: List[Dict[str, Any]] = []
        for index, item in enumerate(spectra):
            payload.append({
                "class_id": int(item.get("id", index)),
                "class_name": str(item.get("name", f"Cluster {index}")),
                "pixel_count": int(item.get("n_pixels", 0)),
                "wavelengths": values(item.get("wavelengths")),
                "mean": values(item.get("mean")),
                "median": values(item.get("median")),
                "q25": values(item.get("q25")),
                "q75": values(item.get("q75")),
            })
        return payload
