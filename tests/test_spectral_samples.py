import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.spectral_samples import export_spectral_samples


class SpectralSampleExportTests(unittest.TestCase):
    def test_exports_reproducible_grouped_pixel_spectra(self):
        analysis = np.arange(6 * 5 * 4, dtype=np.float32).reshape(6, 5, 4) / 100
        raw = analysis * 10_000
        class_map = np.zeros((6, 5), dtype=np.int32)
        class_map[:, 3:] = 1
        base_map = np.where(class_map == 0, 1, 2).astype(np.int16)
        class_info = [
            {"id": 0, "name": "Sunlit A"},
            {"id": 1, "name": "Shadow A"},
        ]
        wavelengths = [500.0, 600.0, 700.0, 800.0]

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.h5"
            second = Path(directory) / "second.h5"
            kwargs = dict(
                analysis_data=analysis,
                raw_data=raw,
                class_map=class_map,
                class_info=class_info,
                analysis_wavelengths=wavelengths,
                raw_wavelengths=wavelengths,
                base_class_map=base_map,
                max_per_class=4,
                random_state=19,
                spatial_downsample=2,
                value_units="reflectance",
                save_raw=True,
                provenance={"source_file": "plot_01.bil"},
            )
            result = export_spectral_samples(first, **kwargs)
            export_spectral_samples(second, **kwargs)

            self.assertEqual(result["n_samples"], 8)
            self.assertTrue(result["raw_values_saved"])
            with h5py.File(first, "r") as h5_first, h5py.File(second, "r") as h5_second:
                self.assertEqual(h5_first["analysis_values"].shape, (8, 4))
                self.assertEqual(h5_first["raw_values"].shape, (8, 4))
                np.testing.assert_array_equal(
                    h5_first["pixel_flat_index"][:],
                    h5_second["pixel_flat_index"][:],
                )
                np.testing.assert_array_equal(
                    h5_first["source_row"][:], h5_first["row"][:] * 2
                )
                np.testing.assert_array_equal(
                    h5_first["source_column"][:], h5_first["column"][:] * 2
                )
                np.testing.assert_array_equal(
                    h5_first["base_class_id"][:],
                    base_map.reshape(-1)[h5_first["pixel_flat_index"][:]],
                )
                self.assertEqual(h5_first.attrs["value_units"], "reflectance")
                self.assertIn("not independent", h5_first.attrs["statistical_unit_note"])
                np.testing.assert_array_equal(
                    h5_first["classes/sampled_count"][:], [4, 4]
                )
                np.testing.assert_allclose(
                    h5_first["classes/sample_weight"][:], [4.5, 3.0]
                )

    def test_can_omit_raw_values(self):
        data = np.ones((2, 2, 3), dtype=np.float32)
        class_map = np.zeros((2, 2), dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "samples.h5"
            export_spectral_samples(
                target,
                analysis_data=data,
                raw_data=data * 100,
                class_map=class_map,
                class_info=[{"id": 0, "name": "Cluster 0"}],
                max_per_class=10,
                save_raw=False,
            )
            with h5py.File(target, "r") as handle:
                self.assertNotIn("raw_values", handle)
                self.assertEqual(handle["analysis_values"].shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
