"""
roi_utils.py
------------
Shared helpers for ROI (box / lasso / click-polygon) spectrum extraction.

Used by both the standalone ROI app (app_roi_spectra.py) and the ROI tab in
the main app, so the two always compute spectra the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- #
# Region geometry
# ---------------------------------------------------------------- #

def box_region(roi: Sequence[int], H: int, W: int) -> dict:
    """Clamp a [r0, r1, c0, c1] box to the image and return a region dict."""
    r0, r1, c0, c1 = (int(v) for v in roi)
    r0, r1 = sorted((max(0, r0), min(H, r1)))
    c0, c1 = sorted((max(0, c0), min(W, c1)))
    return {"type": "box", "roi": [r0, r1, c0, c1]}


def polygon_region(xs: Sequence[float], ys: Sequence[float], H: int, W: int) -> dict:
    """Build a clamped click-polygon region from image x/y coordinates."""
    x_values = [float(np.clip(value, 0, max(0, W - 1))) for value in xs]
    y_values = [float(np.clip(value, 0, max(0, H - 1))) for value in ys]
    if len(x_values) < 3 or len(x_values) != len(y_values):
        raise ValueError("폴리곤 ROI는 꼭짓점이 3개 이상 필요합니다.")
    c0 = int(max(0, np.floor(min(x_values))))
    c1 = int(min(W, np.ceil(max(x_values)) + 1))
    r0 = int(max(0, np.floor(min(y_values))))
    r1 = int(min(H, np.ceil(max(y_values)) + 1))
    return {
        "type": "polygon",
        "x": x_values,
        "y": y_values,
        "roi": [r0, r1, c0, c1],
    }


def selection_to_region(selection, H: int, W: int) -> Optional[dict]:
    """Convert a Streamlit/Plotly selection event into a region dict."""
    box = getattr(selection, "box", None)
    if box:
        b = box[0]
        xs = sorted([b["x"][0], b["x"][1]])
        ys = sorted([b["y"][0], b["y"][1]])
        c0, c1 = int(max(0, xs[0])), int(min(W, xs[1]))
        r0, r1 = int(max(0, ys[0])), int(min(H, ys[1]))
        return box_region([r0, r1, c0, c1], H, W)

    lasso = getattr(selection, "lasso", None)
    if lasso:
        l = lasso[0]
        xs = [float(x) for x in l.get("x", [])]
        ys = [float(y) for y in l.get("y", [])]
        if len(xs) >= 3 and len(xs) == len(ys):
            c0 = int(max(0, np.floor(min(xs))))
            c1 = int(min(W, np.ceil(max(xs)) + 1))
            r0 = int(max(0, np.floor(min(ys))))
            r1 = int(min(H, np.ceil(max(ys)) + 1))
            return {"type": "lasso", "x": xs, "y": ys, "roi": [r0, r1, c0, c1]}

    return None


def offset_region(
    region: dict,
    row_offset: int,
    col_offset: int,
    H: int,
    W: int,
) -> dict:
    """Translate a region selected in a cropped preview to full-image coordinates."""
    translated = dict(region)
    r0, r1, c0, c1 = region.get("roi", [0, 0, 0, 0])
    translated["roi"] = box_region(
        [
            int(r0) + int(row_offset),
            int(r1) + int(row_offset),
            int(c0) + int(col_offset),
            int(c1) + int(col_offset),
        ],
        H,
        W,
    )["roi"]
    if region.get("type") in {"lasso", "polygon"}:
        translated["x"] = [
            float(value) + int(col_offset) for value in region.get("x", [])
        ]
        translated["y"] = [
            float(value) + int(row_offset) for value in region.get("y", [])
        ]
    return translated


# ---------------------------------------------------------------- #
# Pixel extraction and statistics
# ---------------------------------------------------------------- #

def region_pixels(data: np.ndarray, region: dict) -> tuple[np.ndarray, list[int], str]:
    """Return (pixels (N, B), [r0, r1, c0, c1], region_type) for *region*."""
    H, W, B = data.shape
    region_type = region.get("type", "box")
    r0, r1, c0, c1 = box_region(region.get("roi", [0, H, 0, W]), H, W)["roi"]
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"잘못된 ROI입니다: {region}")

    if region_type in {"lasso", "polygon"}:
        from matplotlib.path import Path as MplPath

        xs = np.asarray(region.get("x", []), dtype=float)
        ys = np.asarray(region.get("y", []), dtype=float)
        if len(xs) < 3 or len(xs) != len(ys):
            raise ValueError("올가미/폴리곤 ROI 좌표가 충분하지 않습니다.")

        yy, xx = np.mgrid[r0:r1, c0:c1]
        points = np.column_stack([(xx.ravel() + 0.5), (yy.ravel() + 0.5)])
        polygon = MplPath(np.column_stack([xs, ys]))
        mask = polygon.contains_points(points).reshape((r1 - r0), (c1 - c0))
        pixels = data[r0:r1, c0:c1, :][mask].reshape(-1, B).astype(np.float64)
    else:
        pixels = data[r0:r1, c0:c1, :].reshape(-1, B).astype(np.float64)

    if len(pixels) == 0:
        raise ValueError("ROI 안에 선택된 픽셀이 없습니다.")

    return pixels, [r0, r1, c0, c1], region_type


def roi_stats(data: np.ndarray, region: dict) -> tuple[pd.DataFrame, int, list[int], str]:
    """Per-band summary statistics over the pixels inside *region*."""
    pixels, bounds, region_type = region_pixels(data, region)
    df = pd.DataFrame(
        {
            "mean":   np.mean(pixels, axis=0),
            "median": np.median(pixels, axis=0),
            "std":    np.std(pixels, axis=0),
            "q25":    np.percentile(pixels, 25, axis=0),
            "q75":    np.percentile(pixels, 75, axis=0),
        }
    )
    return df, len(pixels), bounds, region_type


def apply_calibration(df: pd.DataFrame, a, b) -> pd.DataFrame:
    """
    Convert ROI statistics from DN to reflectance using empirical-line
    coefficients: R = a·DN + b.

    The transform is affine per band, so it can be applied to the summary
    statistics directly instead of to the whole cube — mean/median/quantiles
    map through unchanged, and the spread scales by |a|. For a multi-GB cube
    that difference is the difference between instant and out of memory.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(df) or len(b) != len(df):
        raise ValueError(
            f"Calibration has {len(a)} bands, ROI statistics have {len(df)}"
        )

    out = df.copy()
    for col in ("mean", "median", "q25", "q75"):
        if col in out:
            out[col] = out[col].to_numpy() * a + b
    if "std" in out:
        out["std"] = out["std"].to_numpy() * np.abs(a)

    # A negative slope would invert the quartiles; restore their order.
    if np.any(a < 0) and {"q25", "q75"} <= set(out.columns):
        q25 = np.minimum(out["q25"].to_numpy(), out["q75"].to_numpy())
        q75 = np.maximum(out["q25"].to_numpy(), out["q75"].to_numpy())
        out["q25"], out["q75"] = q25, q75

    return out


def calibration_diagnostics(
    raw_stats: pd.DataFrame,
    calibrated_stats: pd.DataFrame,
    a,
    b,
    wavelengths=None,
) -> pd.DataFrame:
    """Build a per-band audit table for an ROI calibration.

    Flags are deliberately conservative: invalid/non-positive coefficients,
    robust gain outliers, or an ROI mean reflectance outside the broad
    physically plausible range -0.05..1.20.  The function diagnoses but never
    edits coefficients or spectra.
    """
    coeff_a = np.asarray(a, dtype=np.float64)
    coeff_b = np.asarray(b, dtype=np.float64)
    bands = len(raw_stats)
    if len(calibrated_stats) != bands or coeff_a.shape != (bands,) or coeff_b.shape != (bands,):
        raise ValueError("Raw, calibrated, and coefficient band counts must match")

    raw_mean = raw_stats["mean"].to_numpy(dtype=float)
    calibrated_mean = calibrated_stats["mean"].to_numpy(dtype=float)
    finite = (
        np.isfinite(coeff_a)
        & np.isfinite(coeff_b)
        & np.isfinite(raw_mean)
        & np.isfinite(calibrated_mean)
    )
    nonpositive = finite & (coeff_a <= 0)
    reflectance_range = finite & (
        (calibrated_mean < -0.05) | (calibrated_mean > 1.20)
    )

    gain_outlier = np.zeros(bands, dtype=bool)
    positive = finite & (coeff_a > 0)
    if int(positive.sum()) >= 7:
        log_gain = np.log10(coeff_a[positive])
        center = float(np.median(log_gain))
        mad = float(np.median(np.abs(log_gain - center)))
        if mad > 1e-12:
            robust_z = np.abs(0.67448975 * (log_gain - center) / mad)
            gain_outlier[np.flatnonzero(positive)] = robust_z > 6.0

    flags = []
    for index in range(bands):
        current = []
        if not finite[index]:
            current.append("invalid coefficient/value")
        if nonpositive[index]:
            current.append("non-positive gain")
        if gain_outlier[index]:
            current.append("gain outlier")
        if reflectance_range[index]:
            current.append("ROI reflectance outside -0.05..1.20")
        flags.append("; ".join(current))

    frame = pd.DataFrame(
        {
            "band_index": np.arange(bands),
            "raw_mean": raw_mean,
            "calibrated_mean": calibrated_mean,
            "calibration_a": coeff_a,
            "calibration_b": coeff_b,
            "suspect": (~finite) | nonpositive | gain_outlier | reflectance_range,
            "diagnostic": flags,
        }
    )
    if wavelengths is not None and len(wavelengths) == bands:
        frame.insert(1, "wavelength_nm", np.asarray(wavelengths, dtype=float))
    return frame


def save_roi_csv(
    data: np.ndarray,
    wavelengths,
    region: dict,
    source_file: str,
    path: str,
    value_units: str = "",
    calibration: Optional[tuple] = None,
    calibration_meta: Optional[dict] = None,
) -> Path:
    """
    Write the ROI spectrum statistics to *path* and return the written path.

    Pass *calibration* as (a, b) to write reflectance instead of raw DN.
    """
    df, n_pixels, bounds, region_type = roi_stats(data, region)
    if calibration is not None:
        df = apply_calibration(df, calibration[0], calibration[1])
    if wavelengths is not None and len(wavelengths) == len(df):
        df.insert(0, "wavelength_nm", wavelengths)
    else:
        df.insert(0, "band_index", np.arange(len(df)))

    r0, r1, c0, c1 = bounds
    if region_type in {"lasso", "polygon"}:
        df.insert(0, "roi_polygon_y", ";".join(f"{v:.3f}" for v in region.get("y", [])))
        df.insert(0, "roi_polygon_x", ";".join(f"{v:.3f}" for v in region.get("x", [])))
    else:
        df.insert(0, "roi_polygon_y", "")
        df.insert(0, "roi_polygon_x", "")
    df.insert(0, "roi_col_max", c1)
    df.insert(0, "roi_col_min", c0)
    df.insert(0, "roi_row_max", r1)
    df.insert(0, "roi_row_min", r0)
    df.insert(0, "roi_type", region_type)
    df.insert(0, "n_pixels", n_pixels)
    if value_units:
        df.insert(0, "value_units", value_units)
    df.insert(0, "calibration_applied", calibration is not None)
    df.insert(
        0,
        "calibration_profile",
        str((calibration_meta or {}).get("selected_profile", "")),
    )
    df.insert(
        0,
        "calibration_method",
        str((calibration_meta or {}).get("method", "")),
    )
    df.insert(0, "source_file", Path(source_file).name)

    # Keep the spatial ROI columns above, and append the same radiometric
    # provenance schema used by the global and region-clustering exporters.
    from .calibration_provenance import add_calibration_provenance

    supplied_meta = dict(calibration_meta or {})
    if isinstance(supplied_meta.get("meta"), dict):
        calibration_info = supplied_meta
    else:
        calibration_info = {
            "selected_profile": supplied_meta.get("selected_profile", ""),
            "calibration_type": supplied_meta.get("calibration_type", ""),
            "selection_source": supplied_meta.get("selection_source", ""),
            "meta": supplied_meta,
        }
    df = add_calibration_provenance(
        df,
        source_file=source_file,
        value_units=value_units,
        normalization_mode="none",
        calibration_info=calibration_info,
        calibration_applied=calibration is not None,
        coefficients_a=(calibration[0] if calibration is not None else None),
        coefficients_b=(calibration[1] if calibration is not None else None),
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------- #
# Display helper
# ---------------------------------------------------------------- #

def display_rgb(data: np.ndarray, wavelengths) -> np.ndarray:
    """
    Build an 8-bit RGB preview for on-screen ROI picking.

    The per-channel percentile stretch here is for *display only* and never
    touches the array the spectra are computed from.
    """
    B = data.shape[2]
    channels = []

    for target in (660, 550, 450):
        if wavelengths is not None and len(wavelengths):
            wl = np.asarray(wavelengths, dtype=float)
            idx = int(np.argmin(np.abs(wl - target)))
        else:
            frac = (target - 400) / 600.0
            idx = int(np.clip(frac * (B - 1), 0, B - 1))

        ch = data[:, :, idx].astype(np.float32)
        finite = np.isfinite(ch)
        if not finite.any():
            channels.append(np.zeros_like(ch, dtype=np.uint8))
            continue
        p2, p98 = np.nanpercentile(ch, 2), np.nanpercentile(ch, 98)
        if p98 <= p2:
            channels.append(np.zeros_like(ch, dtype=np.uint8))
        else:
            scaled = np.nan_to_num((ch - p2) / (p98 - p2), nan=0.0)
            channels.append(
                np.clip(scaled * 255, 0, 255).astype(np.uint8)
            )

    return np.dstack(channels)


def display_reflectance_rgb(
    data: np.ndarray,
    wavelengths,
    a,
    b,
    *,
    reflectance_max: float = 0.6,
    gamma: float = 1.0,
) -> np.ndarray:
    """Build a fixed-scale RGB preview after per-band reflectance calibration.

    Only the three display bands are converted, so this remains practical for
    large cubes.  Unlike :func:`display_rgb`, all three channels use the same
    0..``reflectance_max`` scale.  An independent percentile stretch would
    cancel a positive affine calibration and make raw-DN and reflectance
    previews look deceptively identical.
    """
    if data.ndim != 3:
        raise ValueError("Hyperspectral data must have shape (H, W, bands)")
    if not np.isfinite(reflectance_max) or float(reflectance_max) <= 0:
        raise ValueError("reflectance_max must be a finite positive number")
    if not np.isfinite(gamma) or float(gamma) <= 0:
        raise ValueError("gamma must be a finite positive number")

    bands = data.shape[2]
    coeff_a = np.asarray(a, dtype=np.float32)
    coeff_b = np.asarray(b, dtype=np.float32)
    if coeff_a.shape != (bands,) or coeff_b.shape != (bands,):
        raise ValueError("Calibration bands do not match hyperspectral data")

    valid = np.isfinite(coeff_a) & np.isfinite(coeff_b)
    if not valid.any():
        raise ValueError("Calibration contains no valid RGB display bands")

    valid_indices = np.flatnonzero(valid)
    wavelength_array = None
    if wavelengths is not None and len(wavelengths) == bands:
        wavelength_array = np.asarray(wavelengths, dtype=float)

    channels = []
    for target in (660, 550, 450):
        if wavelength_array is not None:
            finite_candidates = valid_indices[np.isfinite(wavelength_array[valid_indices])]
            if not len(finite_candidates):
                raise ValueError("Calibration has no valid bands with finite wavelengths")
            index = int(
                finite_candidates[
                    np.argmin(np.abs(wavelength_array[finite_candidates] - target))
                ]
            )
        else:
            fraction = (target - 400) / 600.0
            nominal_index = int(np.clip(fraction * (bands - 1), 0, bands - 1))
            index = int(valid_indices[np.argmin(np.abs(valid_indices - nominal_index))])

        reflectance = (
            np.asarray(data[:, :, index], dtype=np.float32) * coeff_a[index]
            + coeff_b[index]
        )
        scaled = np.nan_to_num(
            reflectance / float(reflectance_max),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        scaled = np.clip(scaled, 0.0, 1.0)
        if float(gamma) != 1.0:
            scaled = np.power(scaled, 1.0 / float(gamma))
        channels.append(np.round(scaled * 255.0).astype(np.uint8))

    return np.dstack(channels)
