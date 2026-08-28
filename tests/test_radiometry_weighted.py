import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.radiometry import (
    calibration_qc_status,
    constant_dark_reference,
    discover_calibration_candidates,
    evaluate_weighted_calibration,
    panel_saturation_metrics,
    weighted_dark_panel_calibration,
)


class WeightedPanelCalibrationTests(unittest.TestCase):
    def test_constant_dark_reference_is_explicitly_synthetic(self):
        dark, noise, qc = constant_dark_reference(12)

        np.testing.assert_array_equal(dark, np.full(12, 100.0, dtype=np.float32))
        np.testing.assert_array_equal(noise, np.ones(12, dtype=np.float32))
        self.assertEqual(qc["source_type"], "synthetic_constant")
        self.assertEqual(qc["constant_dn"], 100.0)
        self.assertEqual(qc["sample_pixels"], 0)

    def test_partial_saturation_uses_lower_panel_without_a_hard_seam(self):
        rng = np.random.default_rng(7)
        bands = 120
        true_a = 2e-5 * (
            1.0 + 0.08 * np.sin(np.linspace(0, 4 * np.pi, bands))
        )
        dark = 100.0 + 3.0 * np.sin(np.linspace(0, 2 * np.pi, bands))
        reflectances = [0.99, 0.50, 0.25]
        spectra = []
        weights = []

        for reflectance in reflectances:
            ideal = dark + reflectance / true_a
            pixels = ideal[None, :] + rng.normal(0, 12, (600, bands))
            if reflectance == 0.99:
                pixels[:, 45:70] = 65535
            spectra.append(np.median(pixels, axis=0))
            quality = panel_saturation_metrics(pixels, observed_max=65535)
            weights.append(quality["headroom_weight_by_band"])

        a, b, quality = weighted_dark_panel_calibration(
            spectra,
            reflectances,
            dark,
            panel_band_weights=np.asarray(weights),
            dark_noise=np.full(bands, 3.0),
        )

        self.assertEqual(quality["invalid_band_count"], 0)
        self.assertGreaterEqual(quality["fallback_band_count"], 25)
        np.testing.assert_allclose(a, true_a, rtol=3e-3, atol=0)
        np.testing.assert_allclose(b, -true_a * dark, rtol=3e-3, atol=1e-5)

    def test_saturated_bands_receive_zero_weight(self):
        pixels = np.full((500, 8), 30000.0)
        pixels[:, 2:4] = 65535.0
        quality = panel_saturation_metrics(pixels, observed_max=65535)

        self.assertEqual(quality["saturated_band_indices"], [2, 3])
        self.assertEqual(quality["headroom_weight_by_band"][2:4], [0.0, 0.0])
        self.assertTrue(quality["headroom_weight_by_band"][0] > 0.99)

    def test_isolated_hot_pixels_do_not_zero_an_entire_band(self):
        pixels = np.full((1000, 4), 3000.0)
        pixels[0, 1] = 4094.0
        quality = panel_saturation_metrics(pixels, observed_max=4094)

        self.assertNotIn(1, quality["saturated_band_indices"])
        self.assertGreater(quality["headroom_weight_by_band"][1], 0.99)
        self.assertEqual(quality["band_max_dn"][1], 4094.0)

    def test_qc_fails_when_reference_panels_disagree(self):
        bands = 20
        dark = np.full(bands, 100.0)
        panel_dns = [np.full(bands, 1000.0), np.full(bands, 900.0)]
        a = np.full(bands, 0.99 / 900.0)
        b = -a * dark
        fit_quality = {
            "median_coefficient_cv": 0.30,
            "panel_weights": np.ones((2, bands)),
        }

        qc = evaluate_weighted_calibration(
            panel_dns,
            [0.99, 0.50],
            dark,
            a,
            b,
            fit_quality=fit_quality,
            panel_usable_masks=np.ones((2, bands), dtype=bool),
            panel_uniformities=[0.02, 0.03],
            dark_source_type="measured_file",
        )

        self.assertEqual(qc["status"], "FAIL")
        self.assertFalse(qc["auto_apply_allowed"])
        self.assertGreater(qc["max_panel_mae"], 0.08)

    def test_good_measured_multi_panel_calibration_passes(self):
        bands = 30
        dark = np.full(bands, 100.0)
        true_a = np.linspace(0.0008, 0.0010, bands)
        reflectances = [0.99, 0.50]
        panel_dns = [dark + value / true_a for value in reflectances]
        fit_quality = {
            "median_coefficient_cv": 0.001,
            "panel_weights": np.ones((2, bands)),
        }

        qc = evaluate_weighted_calibration(
            panel_dns,
            reflectances,
            dark,
            true_a,
            -true_a * dark,
            fit_quality=fit_quality,
            panel_usable_masks=np.ones((2, bands), dtype=bool),
            panel_uniformities=[0.02, 0.03],
            dark_source_type="measured_file",
        )

        self.assertEqual(qc["status"], "PASS")
        self.assertTrue(qc["auto_apply_allowed"])

    def test_legacy_high_coefficient_cv_is_failed(self):
        self.assertEqual(
            calibration_qc_status({"median_coefficient_cv": 0.37}), "FAIL"
        )


class CalibrationDiscoveryTests(unittest.TestCase):
    def test_matches_direct_bil_entry_to_filename_based_profile(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "AP3-4.bil"
            source.touch()
            output = root / "output"
            calibration = output / "calibration" / (
                "AP3-4.bil_weighted_dark_calibration.npz"
            )
            calibration.parent.mkdir(parents=True)
            calibration.touch()

            found = discover_calibration_candidates(
                source, search_roots=[output]
            )

        self.assertEqual(found, [calibration.resolve()])

    def test_finds_generic_calibration_next_to_image(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "scene.vnir.hdr"
            source.touch()
            calibration = root / "calibration.npz"
            calibration.touch()

            candidates = discover_calibration_candidates(source)

        self.assertEqual(candidates, [calibration.resolve()])

    def test_matches_envi_double_suffix_to_output_calibration(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "data" / "scene.vnir.bil.hdr"
            source.parent.mkdir()
            source.touch()
            output = root / "results"
            calibration_dir = output / "calibration"
            calibration_dir.mkdir(parents=True)
            calibration = calibration_dir / "scene.vnir_weighted_dark_calibration.npz"
            calibration.touch()

            candidates = discover_calibration_candidates(
                source,
                search_roots=[output],
            )

        self.assertEqual(candidates, [calibration.resolve()])


if __name__ == "__main__":
    unittest.main()
