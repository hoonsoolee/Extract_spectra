import os
import unittest
from pathlib import Path

from src.path_picker import _initial_directory, native_dialogs_available


class PathPickerTests(unittest.TestCase):
    def test_file_initial_path_uses_its_parent_directory(self):
        source_file = Path(__file__).resolve()

        self.assertEqual(
            Path(_initial_directory(str(source_file))).resolve(),
            source_file.parent,
        )

    def test_native_availability_matches_server_operating_system(self):
        self.assertEqual(native_dialogs_available(), os.name == "nt")


if __name__ == "__main__":
    unittest.main()
