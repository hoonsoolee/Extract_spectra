"""Lightweight runtime estimates for local hyperspectral analyses.

The estimate is intentionally expressed as a range.  Disk speed, available
RAM, algorithm convergence, and other jobs on the machine can change runtime
substantially.  Recent timings from the current Streamlit session can be used
to calibrate the default rate without writing user telemetry to disk.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


_METHOD_FACTORS = {
    "hybrid": 1.0,
    "kmeans": 1.0,
    "sam": 1.15,
    "supervised": 1.35,
    "autoencoder": 2.5,
    "cnn": 3.0,
    "hdbscan": 2.4,
    "gmm": 1.8,
    "nmf": 2.2,
}


def format_duration(seconds: float | int | None) -> str:
    """Return a compact Korean duration string."""
    if seconds is None or not np.isfinite(float(seconds)):
        return "산정 불가"
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes:02d}분"
    if minutes:
        return f"{minutes}분 {secs:02d}초"
    return f"{secs}초"


def format_estimate(seconds: float | int | None) -> str:
    """Format a deliberately broad expected-runtime interval."""
    if seconds is None or not np.isfinite(float(seconds)):
        return "첫 실행 후 산정 가능"
    center = max(5.0, float(seconds))
    lower = max(3.0, center * 0.65)
    upper = max(lower + 2.0, center * 1.55)
    return f"약 {format_duration(lower)}–{format_duration(upper)}"


def source_payload_bytes(path: str | Path) -> int:
    """Return the data payload size, resolving an ENVI header companion."""
    source = Path(path)
    if not source.is_file():
        return 0
    if source.suffix.lower() != ".hdr":
        return int(source.stat().st_size)

    candidates: list[Path] = []
    try:
        header_text = source.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"(?im)^\s*data\s+file\s*=\s*\{?([^}\r\n]+)", header_text)
        if match:
            declared = match.group(1).strip().strip('"\'')
            candidates.append(source.parent / declared)
    except OSError:
        pass

    # image.bil.hdr -> image.bil, and image.hdr -> image.bil/.raw/...
    candidates.append(Path(str(source)[:-4]))
    candidates.extend(
        source.with_suffix(ext)
        for ext in (".bil", ".bip", ".bsq", ".raw", ".img", ".dat")
    )
    for candidate in candidates:
        if candidate.is_file() and candidate != source:
            return int(candidate.stat().st_size)
    return int(source.stat().st_size)


def file_work_units(paths: Iterable[str | Path], downsample: int = 1) -> tuple[float, int]:
    """Return (work units, source bytes) for one or more local files."""
    path_list = list(paths)
    total_bytes = sum(source_payload_bytes(path) for path in path_list)
    factor = max(1, int(downsample))
    effective_gib = total_bytes / (1024**3) / (factor * factor)
    # Per-file fixed cost covers imports, model setup, plots, metrics and report.
    return max(0.0, effective_gib) + 0.25 * len(path_list), total_bytes


def array_work_units(shape: Sequence[int], itemsize: int = 4) -> float:
    """Estimate work units for an already loaded H×W×B array."""
    if len(shape) < 3:
        return 0.25
    nbytes = int(np.prod(shape, dtype=np.int64)) * max(1, int(itemsize))
    return nbytes / (1024**3) + 0.25


def estimate_seconds(
    work_units: float,
    method: str,
    history: Sequence[Mapping[str, object]] | None = None,
) -> float:
    """Estimate runtime, preferring recent same-method measurements."""
    default_rate = 75.0 * _METHOD_FACTORS.get(str(method), 1.2)
    rates: list[float] = []
    for record in list(history or [])[-12:]:
        if str(record.get("method")) != str(method):
            continue
        try:
            units = float(record.get("work_units", 0.0))
            elapsed = float(record.get("elapsed_seconds", 0.0))
        except (TypeError, ValueError):
            continue
        if units > 0 and elapsed > 0:
            rates.append(elapsed / units)
    rate = float(np.median(rates[-7:])) if rates else default_rate
    rate = float(np.clip(rate, 5.0, 900.0))
    return max(5.0, float(work_units) * rate)

