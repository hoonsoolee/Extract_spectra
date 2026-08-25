import unittest

from src.report_options import resolve_report_options


class ReportOptionsTests(unittest.TestCase):
    def test_quick_qc_enables_overlay_and_disables_heavy_sections(self):
        resolved = resolve_report_options({"report": {"preset": "quick_qc"}})
        self.assertTrue(resolved["sections"]["cluster_overlay"])
        self.assertTrue(resolved["sections"]["spectral_indices"])
        self.assertFalse(resolved["sections"]["per_class_images"])
        self.assertEqual(resolved["spectra_statistics"], ["mean", "median"])
        self.assertEqual(resolved["indices"], ["NDVI"])

    def test_custom_overrides_are_normalized_and_serializable(self):
        resolved = resolve_report_options(
            {
                "preset": "custom",
                "sections": {"rgb": False, "false_color": True},
                "spectra_statistics": ["median", "not-real"],
                "indices": ["pri", "ndvi", "made-up"],
                "daily_summary": False,
            }
        )
        self.assertFalse(resolved["sections"]["rgb"])
        self.assertTrue(resolved["sections"]["false_color"])
        self.assertEqual(resolved["spectra_statistics"], ["median"])
        self.assertEqual(resolved["indices"], ["NDVI", "PRI"])
        self.assertFalse(resolved["daily_summary"])

    def test_legacy_flags_still_control_sections(self):
        resolved = resolve_report_options(
            {"report": {"show_rgb_composite": False, "show_spectral_plots": False}}
        )
        self.assertFalse(resolved["sections"]["rgb"])
        self.assertFalse(resolved["sections"]["spectral_plot"])


if __name__ == "__main__":
    unittest.main()
