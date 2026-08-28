import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.local_open import open_local_path


class LocalOpenTests(unittest.TestCase):
    def test_rejects_missing_path(self):
        missing = Path(tempfile.gettempdir()) / "extract_spectra_missing_result.html"
        with self.assertRaises(FileNotFoundError):
            open_local_path(missing)

    def test_windows_uses_default_application(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "report.html"
            target.write_text("<html></html>", encoding="utf-8")
            with (
                mock.patch("src.local_open.os.name", "nt"),
                mock.patch("src.local_open.os.startfile", create=True) as startfile,
            ):
                resolved = open_local_path(target)

            self.assertEqual(resolved, target.resolve())
            startfile.assert_called_once_with(str(target.resolve()))


if __name__ == "__main__":
    unittest.main()
