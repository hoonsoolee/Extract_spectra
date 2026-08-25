"""Science-aware vegetation index calculation for calibrated HSI cubes."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np


INDEX_DEFINITIONS: dict[str, dict[str, Any]] = {
    "NDVI": {
        "bands": (("NIR", 800.0, 25.0), ("Red", 670.0, 25.0)),
        "formula": "(NIR - Red) / (NIR + Red)",
    },
    "GNDVI": {
        "bands": (("NIR", 800.0, 25.0), ("Green", 550.0, 25.0)),
        "formula": "(NIR - Green) / (NIR + Green)",
    },
    "NDRE": {
        "bands": (("NIR", 790.0, 25.0), ("RedEdge", 720.0, 25.0)),
        "formula": "(NIR - RedEdge) / (NIR + RedEdge)",
    },
    "PRI": {
        "bands": (("R531", 531.0, 15.0), ("R570", 570.0, 15.0)),
        "formula": "(R531 - R570) / (R531 + R570)",
    },
}


def _nearest_band(wavelengths: np.ndarray, target: float, tolerance: float) -> tuple[int, float]:
    finite = np.flatnonzero(np.isfinite(wavelengths))
    if not len(finite):
        raise ValueError("No finite wavelengths are available")
    idx = int(finite[np.argmin(np.abs(wavelengths[finite] - target))])
    actual = float(wavelengths[idx])
    if abs(actual - target) > tolerance:
        raise ValueError(
            f"required {target:.0f}±{tolerance:.0f} nm, nearest band is {actual:.1f} nm"
        )
    return idx, actual


def compute_index(
    data: np.ndarray,
    wavelengths: Optional[Iterable[float]],
    name: str,
    *,
    is_reflectance: bool,
) -> dict[str, Any]:
    """Compute one index and return values, band provenance, and summary.

    Publication-facing indices are intentionally refused when the cube is not
    calibrated reflectance. This prevents raw DN values from being silently
    reported as physical vegetation indices.
    """

    key = str(name).upper()
    if key not in INDEX_DEFINITIONS:
        return {"name": key, "values": None, "reason": "unsupported index"}
    if not is_reflectance:
        return {
            "name": key,
            "values": None,
            "reason": "calibrated reflectance is required",
            "formula": INDEX_DEFINITIONS[key]["formula"],
        }
    if data.ndim != 3:
        return {"name": key, "values": None, "reason": "data must be H×W×bands"}
    if wavelengths is None:
        return {"name": key, "values": None, "reason": "wavelength metadata is missing"}

    wavelength_array = np.asarray(list(wavelengths), dtype=float)
    if wavelength_array.shape != (data.shape[2],):
        return {
            "name": key,
            "values": None,
            "reason": "wavelength count does not match the data bands",
        }

    definition = INDEX_DEFINITIONS[key]
    try:
        selected = [
            (label, *_nearest_band(wavelength_array, target, tolerance))
            for label, target, tolerance in definition["bands"]
        ]
    except ValueError as exc:
        return {
            "name": key,
            "values": None,
            "reason": str(exc),
            "formula": definition["formula"],
        }

    first = np.asarray(data[:, :, selected[0][1]], dtype=np.float32)
    second = np.asarray(data[:, :, selected[1][1]], dtype=np.float32)
    denominator = first + second
    valid = np.isfinite(first) & np.isfinite(second) & (np.abs(denominator) > 1e-8)
    values = np.full(first.shape, np.nan, dtype=np.float32)
    values[valid] = (first[valid] - second[valid]) / denominator[valid]
    finite_values = values[np.isfinite(values)]
    if not len(finite_values):
        return {
            "name": key,
            "values": None,
            "reason": "no valid pixels after denominator and finite-value checks",
            "formula": definition["formula"],
        }

    summary = {
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
        "std": float(np.std(finite_values)),
        "q25": float(np.percentile(finite_values, 25)),
        "q75": float(np.percentile(finite_values, 75)),
        "valid_fraction": float(len(finite_values) / values.size),
    }
    if key == "NDVI":
        summary["fraction_above_0_15"] = float(np.mean(finite_values > 0.15))

    return {
        "name": key,
        "values": values,
        "reason": "",
        "formula": definition["formula"],
        "bands": {
            label: {"index": int(index), "wavelength_nm": float(actual)}
            for label, index, actual in selected
        },
        "summary": summary,
    }


def compute_selected_indices(
    data: np.ndarray,
    wavelengths: Optional[Iterable[float]],
    names: Iterable[str],
    *,
    is_reflectance: bool,
) -> dict[str, dict[str, Any]]:
    """Compute a deterministic ordered collection of requested indices."""

    return {
        str(name).upper(): compute_index(
            data,
            wavelengths,
            str(name),
            is_reflectance=is_reflectance,
        )
        for name in names
    }
