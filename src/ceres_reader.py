"""Low-memory browsing and selective extraction for CERES/CBDF containers.

The container is indexed from record headers only. RGB previews read three
bands directly from selected frames, and a full ENVI BIL is materialized only
when the user explicitly prepares one sensor/segment for the existing
analysis pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ceres_demux import collect_frames, split_segments, wavelengths_from, write_hdr


INDEX_VERSION = 1


def _fingerprint(path: Path, gap_threshold: int) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{gap_threshold}|{INDEX_VERSION}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _segment_record(
    source: Path,
    sensor: str,
    letter: str,
    frames: list[dict],
) -> dict[str, Any]:
    first = frames[0]
    last = frames[-1]
    frame_numbers = [int(item["frame_no"]) for item in frames]
    line_bytes = int(first["bytes"])
    total_bytes = line_bytes * len(frames)
    wavelengths = wavelengths_from(source.parent, sensor, int(first["bands"]))
    return {
        "key": f"{letter}/{sensor}",
        "segment": letter,
        "sensor": sensor,
        "lines": len(frames),
        "samples": int(first["samples"]),
        "bands": int(first["bands"]),
        "line_bytes": line_bytes,
        "bil_bytes": total_bytes,
        "bil_gib": total_bytes / 1024**3,
        "float32_cube_gib": total_bytes * 2 / 1024**3,
        "start_time": datetime.fromtimestamp(first["ts_ns"] / 1e9).isoformat(
            timespec="seconds"
        ),
        "duration_seconds": float((last["ts_ns"] - first["ts_ns"]) / 1e9),
        "first_frame": min(frame_numbers),
        "last_frame": max(frame_numbers),
        "dropped_frames": max(frame_numbers) - min(frame_numbers) + 1 - len(frames),
        "wavelengths": [float(value) for value in wavelengths],
        "frames": [
            {
                "frame_no": int(item["frame_no"]),
                "ts_ns": int(item["ts_ns"]),
                "pix_off": int(item["pix_off"]),
                "bytes": int(item["bytes"]),
            }
            for item in frames
        ],
    }


def scan_ceres(path: str | Path, *, gap_threshold: int = 30) -> dict[str, Any]:
    """Scan record headers and return logical sensor/segment entries."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".ceres" or not source.is_file():
        raise ValueError(f"Not a readable CERES file: {source}")
    frames_by_sensor, other_records = collect_frames(str(source))
    entries: list[dict[str, Any]] = []
    for sensor in ("VNIR", "SWIR"):
        for letter, frames in split_segments(
            frames_by_sensor.get(sensor, []), int(gap_threshold)
        ):
            entries.append(_segment_record(source, sensor, letter, frames))
    if not entries:
        raise ValueError(
            "No VNIR/SWIR image records were found in this CERES file. "
            "The file may be incomplete or use an unsupported CBDF schema."
        )
    return {
        "index_version": INDEX_VERSION,
        "source_path": str(source),
        "source_size": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "gap_threshold": int(gap_threshold),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "entries": entries,
        "other_record_counts": other_records,
    }


def load_or_build_index(
    path: str | Path,
    cache_dir: str | Path,
    *,
    gap_threshold: int = 30,
) -> tuple[dict[str, Any], Path, bool]:
    """Load a valid cached index or scan and persist a new one."""
    source = Path(path).expanduser().resolve()
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{source.stem}_{_fingerprint(source, gap_threshold)}.json"
    if cache_path.is_file():
        try:
            index = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                index.get("index_version") == INDEX_VERSION
                and index.get("source_size") == source.stat().st_size
                and index.get("source_mtime_ns") == source.stat().st_mtime_ns
                and bool(index.get("entries"))
            ):
                return index, cache_path, True
        except (OSError, ValueError, TypeError):
            pass
    index = scan_ceres(source, gap_threshold=gap_threshold)
    cache_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index, cache_path, False


def entry_by_key(index: dict[str, Any], key: str | None) -> dict[str, Any]:
    if not key:
        raise KeyError("CERES entry was not selected")
    for entry in index.get("entries", []):
        if entry.get("key") == key:
            return entry
    raise KeyError(f"CERES entry not found: {key}")


def estimate_pipeline_memory_gib(entry: dict[str, Any], downsample: int = 1) -> dict:
    """Conservative estimate for the current in-memory PCA/clustering pipeline."""
    factor = max(1, int(downsample))
    pixels = math.ceil(entry["lines"] / factor) * math.ceil(
        entry["samples"] / factor
    )
    bands = int(entry["bands"])
    float_cube = pixels * bands * 4
    # Raw float cube + processed cube + float64 classifier matrix, with modest
    # room for band slicing, PCA scores and temporary arrays.
    peak = float_cube * 5.2 + pixels * min(15, bands) * 8
    return {
        "downsample": factor,
        "pixels": int(pixels),
        "float32_cube_gib": float_cube / 1024**3,
        "estimated_peak_gib": peak / 1024**3,
    }


def read_rgb_preview(
    path: str | Path,
    entry: dict[str, Any],
    *,
    max_lines: int = 700,
    max_samples: int = 1200,
) -> tuple[np.ndarray, dict]:
    """Read only three bands from a selected CERES entry for a fast preview."""
    source = Path(path).expanduser().resolve()
    frames = entry["frames"]
    lines = int(entry["lines"])
    samples = int(entry["samples"])
    bands = int(entry["bands"])
    wavelengths = np.asarray(entry.get("wavelengths") or [], dtype=np.float64)
    if wavelengths.shape == (bands,):
        if float(np.nanmin(wavelengths)) > 800:
            targets = (1650, 1250, 1050)
            preview_mode = "SWIR false color"
        else:
            targets = (660, 550, 450)
            preview_mode = "visible RGB"
        band_indices = [
            int(np.argmin(np.abs(wavelengths - target))) for target in targets
        ]
    else:
        targets = None
        preview_mode = "relative-band preview"
        band_indices = [
            int(np.clip(round(frac * (bands - 1)), 0, bands - 1))
            for frac in (0.44, 0.25, 0.08)
        ]
    line_stride = max(1, math.ceil(lines / max(1, int(max_lines))))
    sample_stride = max(1, math.ceil(samples / max(1, int(max_samples))))
    selected_frames = frames[::line_stride]
    width = math.ceil(samples / sample_stride)
    rgb = np.empty((len(selected_frames), width, 3), dtype=np.float32)
    with source.open("rb") as stream:
        for row, frame in enumerate(selected_frames):
            for channel, band in enumerate(band_indices):
                stream.seek(int(frame["pix_off"]) + band * samples * 2)
                values = np.frombuffer(stream.read(samples * 2), dtype="<u2")
                if values.size != samples:
                    raise IOError("Unexpected end of CERES image record")
                rgb[row, :, channel] = values[::sample_stride]
    for channel in range(3):
        values = rgb[:, :, channel]
        low, high = np.percentile(values, [2, 98])
        if high > low:
            rgb[:, :, channel] = (values - low) / (high - low)
        else:
            rgb[:, :, channel] = 0.0
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), {
        "line_stride": line_stride,
        "sample_stride": sample_stride,
        "band_indices": band_indices,
        "preview_mode": preview_mode,
        "target_wavelengths_nm": list(targets) if targets else None,
        "wavelengths_nm": (
            [float(wavelengths[index]) for index in band_indices]
            if wavelengths.shape == (bands,) else None
        ),
        "source_shape": [lines, samples, bands],
        "preview_shape": list(rgb.shape),
    }


def export_entry_to_bil(
    path: str | Path,
    entry: dict[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stream one selected sensor/segment to a uint16 ENVI BIL cache."""
    source = Path(path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    name = f"{source.stem}.{entry['segment']}.{str(entry['sensor']).lower()}.bil"
    bil_path = output_root / name
    hdr_path = bil_path.with_suffix(".hdr")
    expected = int(entry["bil_bytes"])
    if bil_path.is_file() and hdr_path.is_file() and bil_path.stat().st_size == expected:
        return {
            "bil_path": str(bil_path),
            "hdr_path": str(hdr_path),
            "bytes": expected,
            "reused": True,
        }
    free = shutil.disk_usage(output_root).free
    if free < int(expected * 1.05):
        raise OSError(
            f"Not enough free disk space: need about {expected * 1.05 / 1024**3:.2f} GiB, "
            f"available {free / 1024**3:.2f} GiB"
        )
    if bil_path.exists() and not overwrite:
        raise FileExistsError(
            f"Incomplete/different cache already exists: {bil_path}. "
            "Remove it or explicitly overwrite it."
        )
    temporary = bil_path.with_suffix(bil_path.suffix + ".partial")
    with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
        for frame in entry["frames"]:
            input_stream.seek(int(frame["pix_off"]))
            remaining = int(frame["bytes"])
            while remaining:
                block = input_stream.read(min(16 * 1024 * 1024, remaining))
                if not block:
                    raise IOError("Unexpected end of CERES image record")
                output_stream.write(block)
                remaining -= len(block)
    if bil_path.exists():
        bil_path.unlink()
    os.replace(temporary, bil_path)
    write_hdr(
        str(hdr_path),
        int(entry["lines"]),
        int(entry["samples"]),
        int(entry["bands"]),
        entry["wavelengths"],
    )
    return {
        "bil_path": str(bil_path),
        "hdr_path": str(hdr_path),
        "bytes": expected,
        "reused": False,
    }


def index_table(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact rows suitable for a UI table."""
    return [
        {
            "항목": entry["key"],
            "라인": entry["lines"],
            "샘플": entry["samples"],
            "밴드": entry["bands"],
            "예상 BIL (GiB)": round(float(entry["bil_gib"]), 3),
            "길이 (초)": round(float(entry["duration_seconds"]), 1),
            "시작": entry["start_time"],
        }
        for entry in entries
    ]
