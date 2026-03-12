"""
pipeline.py
-----------
Main processing pipeline: orchestrates loader → preprocessor →
classifier → extractor → reporter for a batch of hyperspectral files.
"""

import logging
import time
import traceback
from pathlib import Path
from typing import Optional, List

from .data_loader import HyperspectralLoader
from .preprocessor import Preprocessor
from .classifier import HyperspectralClassifier
from .spectrum_extractor import SpectrumExtractor
from .reporter import Reporter
from .evaluator import Evaluator

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
        self.loader  = HyperspectralLoader(config.get("data", {}))
        self.prep    = Preprocessor(config)
        self.clf     = HyperspectralClassifier(config)
        self.extr    = SpectrumExtractor(config)
        _lang = config.get("report", {}).get("lang", "ko")
        self.reporter = Reporter(config, lang=_lang)

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
    ) -> None:
        """
        Discover and process all files from the configured source(s).

        Parameters
        ----------
        labels_csv  : path to labelled-pixels CSV (for supervised method)
        file_limit  : stop after this many files (useful for testing)
        """
        data_cfg = self.config.get("data", {})

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

        if file_limit:
            all_tasks = all_tasks[:file_limit]

        logger.info(f"Total files to process: {len(all_tasks)}")

        ok, failed = 0, 0
        t0 = time.time()

        for i, (source, file_ref) in enumerate(all_tasks, 1):
            fname = Path(file_ref).name if source == "local" else Path(file_ref).name
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{len(all_tasks)}] {fname}  (source: {source})")
            logger.info(f"{'='*60}")

            try:
                self._process_file(source, file_ref, data_cfg, labels_csv)
                ok += 1
            except Exception as e:
                logger.error(f"FAILED: {fname}\n{traceback.format_exc()}")
                failed += 1

        elapsed = time.time() - t0
        logger.info(
            f"\nBatch complete: {ok} ok / {failed} failed / "
            f"{len(all_tasks)} total  ({elapsed:.1f}s)"
        )

        # Generate consolidated HTML report (timestamped – never overwrites)
        out_cfg = self.config.get("output", {})
        if out_cfg.get("save_report", True) and self.reporter.results:
            ts = time.strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"report_{ts}.html"
            self.reporter.render(report_path)

    # ============================================================
    # Process a single file
    # ============================================================

    def _process_file(
        self,
        source: str,
        file_ref,
        data_cfg: dict,
        labels_csv: Optional[str],
    ) -> None:
        fname = Path(file_ref).name
        stem  = Path(file_ref).stem

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

        # ---- 2. Preprocess ----
        t0 = time.time()
        data, wavelengths = self.prep.process(data, meta.get("wavelengths"))
        logger.info(f"  Preprocess: {time.time()-t0:.2f}s")

        # ---- 3. Classify ----
        t0 = time.time()
        class_map, class_info = self.clf.classify(data, wavelengths, labels_csv)
        logger.info(f"  Classify: {time.time()-t0:.2f}s")

        # ---- 3b. Quality metrics ----
        t0 = time.time()
        metrics = Evaluator.unsupervised_metrics(data, class_map)
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
        logger.info(f"  Evaluate: {time.time()-t0:.2f}s")

        # ---- 4. Extract spectra ----
        t0 = time.time()
        spectra = self.extr.extract(data, class_map, class_info, wavelengths)
        logger.info(f"  Extract spectra: {time.time()-t0:.2f}s")

        # ---- 4b. Spectral separability ----
        sep = Evaluator.spectral_separability(spectra)

        # ---- 4c. Vegetation separation quality ----
        veg_sep = Evaluator.vegetation_separation_metrics(
            data, class_map, spectra, wavelengths
        )
        if veg_sep.get("ndvi_f1") is not None:
            logger.info(
                f"  Vegetation F1: {veg_sep['ndvi_f1']:.3f}  "
                f"Recall: {veg_sep['ndvi_recall']:.3f}  "
                f"Precision: {veg_sep['ndvi_precision']:.3f}"
            )

        # ---- 5. Save outputs ----
        out_cfg = self.config.get("output", {})
        file_out_dir = self.output_dir / stem
        file_out_dir.mkdir(parents=True, exist_ok=True)

        if out_cfg.get("save_spectra_csv", True):
            self.extr.save_csv(spectra, file_out_dir / "spectra.csv")

        if out_cfg.get("save_classification_map", True):
            self._save_class_map(class_map, class_info, file_out_dir / "class_map.png")

        # ---- 6. Add to report ----
        self.reporter.add_result(
            filename=fname,
            data=data,
            class_map=class_map,
            class_info=class_info,
            spectra=spectra,
            wavelengths=wavelengths,
            metadata=meta,
            metrics=metrics,
            separability=sep,
            veg_sep=veg_sep,
        )

        logger.info(f"  Outputs saved to: {file_out_dir}")

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
