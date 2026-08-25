import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.roi_utils import (
    display_reflectance_rgb,
    offset_region,
    polygon_region,
    region_pixels,
    save_roi_csv,
    selection_to_region,
)


class OffsetRegionTests(unittest.TestCase):
    def test_zoom_box_in_cropped_view_maps_to_full_image_coordinates(self):
        selection = SimpleNamespace(
            box=[{"x": [10.0, 110.0], "y": [20.0, 120.0]}],
            lasso=[],
        )

        local_region = selection_to_region(selection, 200, 200)
        translated = offset_region(local_region, 1000, 300, 5000, 800)

        self.assertEqual(translated["roi"], [1020, 1120, 310, 410])

    def test_box_coordinates_are_translated_to_full_image(self):
        region = {"type": "box", "roi": [10, 30, 5, 25]}

        translated = offset_region(region, 1000, 200, 5000, 800)

        self.assertEqual(translated["roi"], [1010, 1030, 205, 225])

    def test_lasso_vertices_are_translated_with_preview_offset(self):
        region = {
            "type": "lasso",
            "roi": [1, 8, 2, 9],
            "x": [2.0, 9.0, 4.0],
            "y": [1.0, 3.0, 8.0],
        }

        translated = offset_region(region, 300, 20, 1000, 200)

        self.assertEqual(translated["roi"], [301, 308, 22, 29])
        self.assertEqual(translated["x"], [22.0, 29.0, 24.0])
        self.assertEqual(translated["y"], [301.0, 303.0, 308.0])

    def test_polygon_vertices_translate_and_mask_pixels(self):
        region = polygon_region([1, 4, 1], [1, 1, 4], 8, 8)
        translated = offset_region(region, 10, 20, 100, 100)
        cube = np.ones((8, 8, 2), dtype=np.float32)
        pixels, _, region_type = region_pixels(cube, region)

        self.assertEqual(region["type"], "polygon")
        self.assertEqual(translated["x"], [21.0, 24.0, 21.0])
        self.assertEqual(translated["y"], [11.0, 11.0, 14.0])
        self.assertEqual(region_type, "polygon")
        self.assertGreater(len(pixels), 0)


class ReflectanceRgbTests(unittest.TestCase):
    def test_uses_one_fixed_reflectance_scale_for_all_channels(self):
        cube = np.array([[[2.0, 5.0, 8.0]]], dtype=np.float32)
        wavelengths = [450.0, 550.0, 660.0]

        rgb = display_reflectance_rgb(
            cube,
            wavelengths,
            a=np.array([0.1, 0.1, 0.1]),
            b=np.zeros(3),
            reflectance_max=1.0,
        )

        np.testing.assert_array_equal(rgb[0, 0], [204, 128, 51])

    def test_rejects_calibration_with_wrong_band_count(self):
        cube = np.zeros((2, 2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "bands do not match"):
            display_reflectance_rgb(cube, None, [1.0, 1.0], [0.0, 0.0])


class RoiCsvCalibrationStatusTests(unittest.TestCase):
    def test_records_applied_profile_and_method(self):
        cube = np.ones((2, 2, 3), dtype=np.float32)
        region = {"type": "box", "roi": [0, 2, 0, 2]}

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "roi.csv"
            save_roi_csv(
                cube,
                [450.0, 550.0, 660.0],
                region,
                "scene.bil",
                str(output),
                value_units="reflectance",
                calibration=(np.ones(3), np.zeros(3)),
                calibration_meta={
                    "method": "weighted multi-panel",
                    "selected_profile": "calibration.npz",
                },
            )

            frame = pd.read_csv(output)

        self.assertTrue(frame["calibration_applied"].all())
        self.assertEqual(frame["calibration_method"].iloc[0], "weighted multi-panel")
        self.assertEqual(frame["calibration_profile"].iloc[0], "calibration.npz")
        np.testing.assert_allclose(frame["calibration_a"], np.ones(3))
        np.testing.assert_allclose(frame["calibration_b"], np.zeros(3))


if __name__ == "__main__":
    unittest.main()
