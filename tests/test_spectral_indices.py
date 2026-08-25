import unittest

import numpy as np

from src.spectral_indices import compute_index, compute_selected_indices


class SpectralIndexTests(unittest.TestCase):
    def setUp(self):
        self.wavelengths = [531.0, 550.0, 570.0, 670.0, 720.0, 800.0]
        self.data = np.zeros((2, 3, len(self.wavelengths)), dtype=np.float32)
        self.data[:, :, 0] = 0.20
        self.data[:, :, 1] = 0.25
        self.data[:, :, 2] = 0.30
        self.data[:, :, 3] = 0.20
        self.data[:, :, 4] = 0.35
        self.data[:, :, 5] = 0.60

    def test_ndvi_uses_nearest_calibrated_bands(self):
        result = compute_index(
            self.data, self.wavelengths, "NDVI", is_reflectance=True
        )
        self.assertEqual(result["reason"], "")
        np.testing.assert_allclose(result["values"], 0.5, atol=1e-6)
        self.assertAlmostEqual(result["summary"]["median"], 0.5, places=6)
        self.assertEqual(result["bands"]["NIR"]["wavelength_nm"], 800.0)

    def test_raw_dn_is_not_silently_reported_as_index(self):
        result = compute_index(
            self.data, self.wavelengths, "NDVI", is_reflectance=False
        )
        self.assertIsNone(result["values"])
        self.assertIn("reflectance", result["reason"])

    def test_selected_indices_keep_requested_supported_names(self):
        results = compute_selected_indices(
            self.data,
            self.wavelengths,
            ["NDVI", "PRI"],
            is_reflectance=True,
        )
        self.assertEqual(list(results), ["NDVI", "PRI"])
        self.assertIsNotNone(results["PRI"]["values"])


if __name__ == "__main__":
    unittest.main()
