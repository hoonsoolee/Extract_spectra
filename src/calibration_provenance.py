"""Compact, repeatable calibration provenance for exported spectrum tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _json_scalar(value: Any) -> str:
    """Serialize a small value without leaking huge per-band QC arrays."""
    if value in (None, ""):
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _panel_summary(meta: Mapping[str, Any]) -> tuple[int, str]:
    panels = meta.get("panels") or []
    compact = []
    for panel in panels:
        if not isinstance(panel, Mapping):
            continue
        compact.append(
            {
                key: panel.get(key)
                for key in ("name", "reflectance", "box", "source")
                if panel.get(key) not in (None, "")
            }
        )
    return len(compact), _json_scalar(compact)


def add_calibration_provenance(
    frame: pd.DataFrame,
    *,
    source_file: str = "",
    value_units: str = "",
    normalization_mode: str = "",
    calibration_info: Optional[Mapping[str, Any]] = None,
    calibration_applied: bool = False,
    coefficients_a: Optional[Sequence[float]] = None,
    coefficients_b: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Add human-readable calibration audit columns to a spectrum DataFrame.

    One row represents one wavelength, so empirical-line coefficients ``a``
    and ``b`` are stored beside that wavelength.  The same profile is recorded
    as ``paired_calibration_profile`` in the raw-DN companion CSV, while
    ``calibration_applied`` remains false there.
    """
    out = frame.copy()
    info = dict(calibration_info or {})
    meta = info.get("meta") if isinstance(info.get("meta"), Mapping) else {}
    selected_profile = str(info.get("selected_profile") or "")
    panel_count, panels_json = _panel_summary(meta)

    applied_profile = selected_profile if calibration_applied else ""
    paired_profile = selected_profile if selected_profile else ""
    columns: list[tuple[str, Any]] = [
        ("source_file", str(Path(source_file).resolve()) if source_file else ""),
        ("value_units", value_units),
        ("normalization_mode", normalization_mode),
        ("calibration_applied", bool(calibration_applied)),
        ("calibration_profile", applied_profile),
        ("paired_calibration_profile", paired_profile),
        ("calibration_selection", str(info.get("selection_source") or "")),
        ("calibration_type", str(info.get("calibration_type") or "")),
        ("calibration_method", str(meta.get("method") or "")),
        ("calibration_formula", str(meta.get("formula") or "")),
        ("calibration_source_image", str(meta.get("source_image") or "")),
        ("dark_source_type", str(meta.get("dark_source_type") or "")),
        ("dark_source", str(meta.get("dark_source") or "")),
        ("manual_dark_dn", meta.get("manual_dark_dn", "")),
        ("white_time", str(meta.get("white_time") or info.get("white_time") or "")),
        ("calibration_panel_count", panel_count),
        ("calibration_panels_json", panels_json),
    ]

    # Insert in reverse so the final CSV reads in the order above.
    for name, value in reversed(columns):
        if name in out:
            out[name] = value
        else:
            out.insert(0, name, value)

    for name, values in (
        ("calibration_a", coefficients_a),
        ("calibration_b", coefficients_b),
    ):
        if values is None or not calibration_applied:
            out[name] = np.nan
            continue
        array = np.asarray(values, dtype=float)
        if array.ndim == 1 and len(array) and len(out) % len(array) == 0:
            out[name] = np.tile(array, len(out) // len(array))
        elif array.shape != (len(out),):
            out[name] = np.nan
        else:
            out[name] = array
    return out
