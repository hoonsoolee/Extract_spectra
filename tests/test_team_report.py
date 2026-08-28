import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from src.team_report import (
    derive_plot_id,
    enrich_summaries,
    generate_team_daily_packages,
)


class TeamReportTests(unittest.TestCase):
    def test_plot_id_and_manifest_metadata_are_deterministic(self):
        self.assertEqual(derive_plot_id("Plot_101.A.vnir.bil.hdr"), "Plot_101.A")
        with TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "plots.csv"
            manifest.write_text(
                "filename,plot_id,treatment,replicate\n"
                "scene_01.bil,AP3-4,Control,2\n",
                encoding="utf-8",
            )
            rows, warnings = enrich_summaries(
                [{
                    "filename": "scene_01.bil",
                    "source_file": "scene_01.bil",
                    "value_units": "reflectance",
                    "calibration_qc_status": "PASS",
                    "ndvi_median": 0.71,
                    "ndvi_q25": 0.65,
                    "ndvi_q75": 0.77,
                }],
                {
                    "team_name": "Team A",
                    "measurement_date": "2026-08-27",
                    "metadata_csv": str(manifest),
                },
            )
            self.assertEqual(rows[0]["plot_id"], "AP3-4")
            self.assertEqual(rows[0]["treatment"], "Control")
            self.assertTrue(rows[0]["included_in_team_statistics"])
            self.assertAlmostEqual(rows[0]["ndvi_iqr"], 0.12)
            self.assertEqual(warnings, [])

    def test_only_pass_reflectance_is_included(self):
        summaries = [
            {
                "filename": "pass.bil",
                "value_units": "reflectance",
                "calibration_qc_status": "PASS",
                "ndvi_median": 0.7,
            },
            {
                "filename": "review.bil",
                "value_units": "reflectance",
                "calibration_qc_status": "REVIEW",
                "ndvi_median": 0.8,
            },
            {
                "filename": "raw.bil",
                "value_units": "raw DN",
                "calibration_qc_status": "UNASSESSED",
                "ndvi_median": "",
            },
        ]
        rows, warnings = enrich_summaries(
            summaries,
            {"team_name": "Team A", "measurement_date": "2026-08-27"},
        )
        self.assertEqual([row["included_in_team_statistics"] for row in rows], [True, False, False])
        self.assertEqual({item["code"] for item in warnings}, {"CALIBRATION_REVIEW", "NO_REFLECTANCE"})

    def test_package_contains_daily_visuals_and_machine_readable_csv(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            overlay = root / "overlay.png"
            ndvi = root / "ndvi.png"
            Image.new("RGB", (120, 80), (40, 130, 80)).save(overlay)
            Image.new("RGB", (120, 80), (190, 220, 90)).save(ndvi)
            summary = {
                "filename": "plot_01.bil",
                "source_file": str(root / "plot_01.bil"),
                "value_units": "reflectance",
                "calibration_qc_status": "PASS",
                "n_classes": 2,
                "ndvi_mean": 0.68,
                "ndvi_median": 0.70,
                "ndvi_q25": 0.64,
                "ndvi_q75": 0.76,
                "vegetation_fraction": 0.82,
                "silhouette": 0.41,
                "davies_bouldin": 0.88,
                "elapsed_seconds": 12.3,
                "overlay_image": str(overlay),
                "ndvi_image": str(ndvi),
                "cluster_summary": [
                    {"class_id": 0, "class_name": "Leaf", "pixel_count": 800},
                    {"class_id": 1, "class_name": "Soil", "pixel_count": 200},
                ],
                "team_spectra": [],
            }
            with patch("src.team_report._create_workbook", return_value=(False, "test fallback")):
                packages = generate_team_daily_packages(
                    [summary],
                    root / "out",
                    {"team_name": "Team A", "measurement_date": "2026-08-27"},
                )
            package = Path(packages[0]["directory"])
            self.assertTrue((package / "Team_Report.html").is_file())
            self.assertTrue((package / "Field_Summary.csv").is_file())
            self.assertTrue((package / "plots_overview.png").is_file())
            self.assertTrue((package / "plots_ndvi.png").is_file())
            self.assertTrue((package / "plot_ndvi_comparison.png").is_file())
            with (package / "Field_Summary.csv").open(encoding="utf-8-sig") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["plot_id"], "plot_01")
            self.assertEqual(row["included_in_team_statistics"], "True")


if __name__ == "__main__":
    unittest.main()
