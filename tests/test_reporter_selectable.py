import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.reporter import Reporter
from src.spectral_indices import compute_selected_indices


class SelectableReporterTests(unittest.TestCase):
    def test_quick_report_renders_selected_sections_and_assets(self):
        wavelengths = [531.0, 550.0, 570.0, 670.0, 720.0, 800.0]
        data = np.zeros((4, 5, 6), dtype=np.float32)
        data[:, :, :] = [0.20, 0.25, 0.30, 0.20, 0.35, 0.60]
        class_map = np.ones((4, 5), dtype=np.int32)
        class_map[:, 3:] = 2
        class_info = [
            {"id": 1, "name": "Leaf 1", "color": (40, 180, 80), "n_pixels": 12},
            {"id": 2, "name": "Leaf 2", "color": (230, 120, 40), "n_pixels": 8},
        ]
        spectra = []
        for info in class_info:
            spectra.append(
                {
                    **info,
                    "wavelengths": wavelengths,
                    "mean": np.array([0.20, 0.25, 0.30, 0.20, 0.35, 0.60]),
                    "median": np.array([0.20, 0.25, 0.30, 0.20, 0.35, 0.60]),
                    "std": np.full(6, 0.01),
                    "q25": np.array([0.19, 0.24, 0.29, 0.19, 0.34, 0.59]),
                    "q75": np.array([0.21, 0.26, 0.31, 0.21, 0.36, 0.61]),
                }
            )
        calibration = {
            "selected_profile": "calibration.npz",
            "calibration_type": "empirical_line_coefficients",
            "a": [0.01] * 6,
            "b": [0.0] * 6,
            "meta": {"method": "weighted multi-panel"},
        }
        indices = compute_selected_indices(
            data, wavelengths, ["NDVI"], is_reflectance=True
        )
        reporter = Reporter({"report": {"preset": "quick_qc"}}, lang="en")
        reporter.add_result(
            filename="scene.bil",
            data=data,
            class_map=class_map,
            class_info=class_info,
            spectra=spectra,
            wavelengths=wavelengths,
            metadata={"format": "ENVI", "calibration": calibration},
            elapsed_sec=1.2,
            index_results=indices,
        )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_path = root / "report.html"
            reporter.render(report_path)
            assets = reporter.save_selected_assets(
                root,
                data=data,
                class_map=class_map,
                class_info=class_info,
                wavelengths=wavelengths,
                index_results=indices,
            )
            report_text = report_path.read_text(encoding="utf-8")

            self.assertIn("RGB + Cluster Overlay", report_text)
            self.assertIn("NDVI", report_text)
            self.assertNotIn("Per-Class Classification Images", report_text)
            self.assertTrue((root / "cluster_overlay.png").exists())
            self.assertTrue((root / "ndvi.png").exists())
            self.assertGreaterEqual(len(assets), 4)


if __name__ == "__main__":
    unittest.main()
