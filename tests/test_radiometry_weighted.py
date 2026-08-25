import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.radiometry import (
    constant_dark_reference,
    discover_calibration_candidates,
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
