import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ceres_demux import (
    GLOBAL_HEADER,
    TYPE_SWIR,
    TYPE_SWIR_V1,
    TYPE_VNIR,
    TYPE_VNIR_V1,
)
from src.ceres_reader import (
    entry_by_key,
    export_entry_to_bil,
    load_or_build_index,
    read_rgb_preview,
)


def _record(kind: bytes, frame: int, timestamp: int, pixels: np.ndarray) -> bytes:
    bands, samples = pixels.shape
    subheader = (
        b"\x01\x05\x02"
        + int(frame).to_bytes(2, "little")
        + b"\x00" * 6
        + int(timestamp).to_bytes(8, "little")
        + int(bands).to_bytes(8, "little")
        + int(samples).to_bytes(8, "little")
        + int(bands * samples).to_bytes(8, "little")
    )
    payload = subheader + np.asarray(pixels, dtype="<u2").tobytes()
    return b"\x00" * 4 + struct.pack("<Q", len(payload)) + kind + payload


def _record_v1(kind: bytes, frame: int, timestamp: int, pixels: np.ndarray) -> bytes:
    bands, samples = pixels.shape
    subheader = (
        b"\x05\x02"
        + int(frame).to_bytes(2, "little")
        + b"\x00" * 6
        + int(timestamp).to_bytes(8, "little")
        + int(bands).to_bytes(8, "little")
        + int(samples).to_bytes(8, "little")
        + int(bands * samples).to_bytes(8, "little")
    )
    payload = subheader + np.asarray(pixels, dtype="<u2").tobytes()
    return b"\x00" * 4 + struct.pack("<Q", len(payload)) + kind + payload


class CeresReaderTests(unittest.TestCase):
    def test_indexes_2024_cbdf_v1_records(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "sample_2024.ceres"
            pixels = np.arange(24, dtype=np.uint16).reshape(4, 6)
            source.write_bytes(
                b"\x00" * GLOBAL_HEADER
                + _record_v1(TYPE_VNIR_V1, 10, 1_720_000_000_000_000_000, pixels)
                + _record_v1(TYPE_SWIR_V1, 11, 1_720_000_000_000_000_001, pixels)
            )

            index, _, reused = load_or_build_index(source, root / "index")
            self.assertFalse(reused)
            self.assertEqual(
                [item["key"] for item in index["entries"]],
                ["A/VNIR", "A/SWIR"],
            )
            entry = entry_by_key(index, "A/VNIR")
            preview, _ = read_rgb_preview(source, entry, max_lines=10, max_samples=10)
            self.assertEqual(preview.shape, (1, 6, 3))

    def test_index_preview_and_selective_export(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "sample.ceres"
            rows = []
            for frame in (1, 2):
                pixels = np.vstack([
                    np.arange(8) + band * 100 + frame
                    for band in range(6)
                ])
                rows.append(_record(TYPE_VNIR, frame, 1_700_000_000_000_000_000 + frame, pixels))
            rows.append(_record(TYPE_SWIR, 3, 1_700_000_000_000_000_003, np.ones((4, 5))))
            source.write_bytes(b"\x00" * GLOBAL_HEADER + b"".join(rows))

            index, cache_path, reused = load_or_build_index(source, root / "index")
            self.assertFalse(reused)
            self.assertTrue(cache_path.is_file())
            self.assertEqual([item["key"] for item in index["entries"]], ["A/VNIR", "A/SWIR"])

            cached, _, reused = load_or_build_index(source, root / "index")
            self.assertTrue(reused)
            entry = entry_by_key(cached, "A/VNIR")
            preview, meta = read_rgb_preview(source, entry, max_lines=10, max_samples=10)
            self.assertEqual(preview.shape, (2, 8, 3))
            self.assertEqual(meta["source_shape"], [2, 8, 6])

            exported = export_entry_to_bil(source, entry, root / "bil")
            self.assertEqual(Path(exported["bil_path"]).stat().st_size, entry["bil_bytes"])
            self.assertTrue(Path(exported["hdr_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
