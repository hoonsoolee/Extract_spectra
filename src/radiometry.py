"""
radiometry.py
-------------
Convert raw sensor DN to reflectance.

Background
----------
A raw HySpex frame is not reflectance. To a good approximation

    DN(x, λ)  =  R(x, λ) · E(λ) · S(λ)  +  d(λ)

where R is surface reflectance, E the illumination (solar irradiance reaching
the target), S the sensor response (quantum efficiency × responsivity × gain ×
integration time) and d a dark offset. E and S are unknown per-band multipliers
that bend every measured spectrum away from its true shape — which is why raw
DN does not look like a leaf, and why a per-band contrast stretch (which
divides them out by accident, using scene statistics) *looks* better while
being scene-dependent and not reproducible.

Panels of known reflectance measured in the same scene let us cancel E·S
properly. Two methods are provided:

single-panel flat field
    R(x, λ) = DN(x, λ) / DN_panel(λ) · R_panel(λ)
    One panel. Assumes the dark offset is negligible relative to the signal.

empirical line (recommended when ≥2 panels are available)
    Least-squares fit, per band, of known panel reflectance against measured
    panel DN:  R = a(λ)·DN + b(λ). The intercept absorbs the dark offset and
    any additive path radiance, so no separate dark correction is needed.

Both cancel E(λ)·S(λ) exactly, so the result is comparable across images and
against spectral libraries.
"""

from __future__ import annotations

import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

WHITE_DARK_PROFILE_TYPE = "sensor_dark_white_v1"


# ---------------------------------------------------------------- #
# Panel spectra
# ---------------------------------------------------------------- #

def panel_spectrum(
    data: np.ndarray,
    box: Sequence[int],
    trim_percentile: float = 10.0,
) -> np.ndarray:
    """
    Mean DN spectrum of a panel region.

    Parameters
    ----------
    data            : (H, W, B) cube
    box             : [r0, r1, c0, c1] bounding box of the panel
    trim_percentile : discard the darkest and brightest this-% of panel pixels
                      before averaging, so shadowed edges and specular hot
                      spots do not bias the reference. Set to 0 to disable.
    """
    r0, r1, c0, c1 = (int(v) for v in box)
    patch = data[r0:r1, c0:c1, :].reshape(-1, data.shape[2]).astype(np.float64)
    if patch.size == 0:
        raise ValueError(f"Empty panel box: {list(box)}")

    if trim_percentile > 0 and len(patch) > 20:
        b = patch.mean(axis=1)
        lo = np.percentile(b, trim_percentile)
        hi = np.percentile(b, 100.0 - trim_percentile)
        keep = (b >= lo) & (b <= hi)
        if keep.sum() >= 10:
            patch = patch[keep]

    logger.info(f"  Panel box {list(box)}: {len(patch):,} pixels averaged")
    return patch.mean(axis=0)


def panel_uniformity(data: np.ndarray, box: Sequence[int]) -> float:
    """
    Coefficient of variation of panel brightness.

    A well-exposed, evenly lit panel sits well below ~0.10. Larger values mean
    the box includes background, shadow or saturation and should be tightened.
    """
    r0, r1, c0, c1 = (int(v) for v in box)
    b = data[r0:r1, c0:c1, :].mean(axis=2)
    return float(b.std() / max(b.mean(), 1e-9))


def check_saturation(
    data: np.ndarray,
    box: Sequence[int],
    max_dn: float = 65535.0,
    frac_limit: float = 0.01,
) -> float:
    """Fraction of panel samples at/above 99% of *max_dn* (saturated)."""
    r0, r1, c0, c1 = (int(v) for v in box)
    patch = data[r0:r1, c0:c1, :]
    frac = float((patch >= 0.99 * max_dn).mean())
    if frac > frac_limit:
        logger.warning(
            f"  Panel box {list(box)} is {frac:.1%} saturated — "
            f"reflectance derived from it will be wrong."
        )
    return frac


def infer_dn_ceiling(observed_max: float) -> float | None:
    """Infer a plausible ADC ceiling from common integer sensor bit depths."""
    value = float(observed_max)
    if not np.isfinite(value) or value <= 1.5:
        return None
    for ceiling in (255.0, 1023.0, 2047.0, 4095.0, 8191.0,
                    16383.0, 32767.0, 65535.0, 131071.0):
        if value <= ceiling * 1.001:
            return ceiling
    bits = int(np.ceil(np.log2(value + 1.0)))
    return float(2**bits - 1)


def panel_saturation_metrics(
    samples: np.ndarray,
    *,
    observed_max: float | None = None,
    max_dn: float | None = None,
    saturation_level: float = 0.99,
    band_fraction_limit: float = 0.01,
) -> dict:
    """Return band-aware saturation QC for a panel or White selection.

    A band is rejected when at least ``band_fraction_limit`` of its selected
    pixels lie within the top one percent of the inferred digital range.  The
    check is intentionally conservative: a clipped White reference cannot
    provide a valid denominator for that wavelength.
    """
    array = np.asarray(samples)
    if array.ndim == 3:
        array = array.reshape(-1, array.shape[2])
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("Panel samples must have shape pixels x bands")
    array = np.asarray(array, dtype=np.float64)
    finite_max = np.nanmax(array) if array.size else np.nan
    reference_max = max(
        float(finite_max) if np.isfinite(finite_max) else 0.0,
        float(observed_max) if observed_max is not None and np.isfinite(observed_max) else 0.0,
    )
    ceiling = float(max_dn) if max_dn is not None else infer_dn_ceiling(reference_max)
    if ceiling is None:
        usable_mask = np.isfinite(array).any(axis=0)
        return {
            "usable": True,
            "adc_ceiling": None,
            "threshold_dn": None,
            "overall_fraction": 0.0,
            "saturated_band_count": 0,
            "saturated_band_indices": [],
            "near_band_count": 0,
            "near_band_indices": [],
            "usable_band_mask": usable_mask.astype(bool).tolist(),
            "headroom_weight_by_band": usable_mask.astype(float).tolist(),
            "band_max_dn": np.nanmax(array, axis=0).astype(float).tolist(),
            "band_peak_ratio": [None] * array.shape[1],
            "reason": "DN ceiling unavailable (non-integer-scale data)",
        }

    threshold = float(saturation_level) * ceiling
    near_threshold = 0.95 * ceiling
    finite = np.isfinite(array)
    denominator = np.maximum(finite.sum(axis=0), 1)
    saturated_fraction = ((array >= threshold) & finite).sum(axis=0) / denominator
    near_fraction = ((array >= near_threshold) & finite).sum(axis=0) / denominator
    band_max = np.nanmax(array, axis=0)
    peak_hits = np.isclose(array, band_max[None, :], rtol=0.0, atol=0.25) & finite
    peak_counts = peak_hits.sum(axis=0)
    peak_fraction = peak_counts / denominator
    # Some cameras clip below the nominal storage-type maximum. A broad,
    # perfectly flat upper plateau is an additional clipping signature.
    plateau_mask = (
        (peak_counts >= 5)
        & (peak_fraction >= 0.05)
        & (band_max >= 0.70 * ceiling)
    )
    saturated_mask = (
        saturated_fraction >= float(band_fraction_limit)
    ) | plateau_mask
    saturated = np.flatnonzero(saturated_mask)
    near = np.flatnonzero(
        (near_fraction >= float(band_fraction_limit))
        & ~saturated_mask
    )
    overall = float(np.sum((array >= threshold) & finite) / max(1, finite.sum()))
    # Do not hard-switch panels at the first saturated band.  Fade the bright
    # panel out while it approaches the non-linear end of the sensor range,
    # then give fully clipped bands zero weight.  A lower-reflectance panel can
    # therefore carry the common calibration model smoothly through the region.
    peak_ratio = band_max / ceiling
    fade_start = min(0.85, float(saturation_level) - 0.01)
    fade_end = max(fade_start + 1e-6, float(saturation_level))
    headroom = np.clip((fade_end - peak_ratio) / (fade_end - fade_start), 0.0, 1.0)
    # Smoothstep avoids a kink at both ends of the transition.
    headroom = headroom * headroom * (3.0 - 2.0 * headroom)
    headroom[saturated_mask | ~np.isfinite(band_max)] = 0.0
    return {
        "usable": len(saturated) == 0,
        "adc_ceiling": ceiling,
        "threshold_dn": threshold,
        "overall_fraction": overall,
        "saturated_band_count": int(len(saturated)),
        "saturated_band_indices": saturated.astype(int).tolist(),
        "plateau_band_count": int(np.sum(plateau_mask)),
        "plateau_band_indices": np.flatnonzero(plateau_mask).astype(int).tolist(),
        "near_band_count": int(len(near)),
        "near_band_indices": near.astype(int).tolist(),
        "usable_band_mask": (~saturated_mask).astype(bool).tolist(),
        "headroom_weight_by_band": headroom.astype(float).tolist(),
        "band_max_dn": band_max.astype(float).tolist(),
        "band_peak_ratio": peak_ratio.astype(float).tolist(),
        "reason": (
            "usable" if len(saturated) == 0
            else f"{len(saturated)} band(s) exceeded the saturation limit"
        ),
    }


def weighted_dark_panel_calibration(
    panel_dns: Sequence[np.ndarray],
    panel_reflectances: Sequence[float | np.ndarray],
    dark_dn: np.ndarray,
    *,
    panel_band_weights: Optional[np.ndarray] = None,
    dark_noise: Optional[np.ndarray] = None,
    min_snr: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit one continuous dark-referenced calibration from multiple panels.

    The physical model after dark subtraction is ``R = a * (DN - dark)``.
    Each panel/band observation receives a weight.  Saturated observations
    have weight zero; observations approaching saturation fade out smoothly;
    and weak observations are down-weighted from their dark-reference SNR.
    This avoids splicing independently calibrated spectra at a hard band
    boundary while still allowing a lower-reflectance panel to carry bands in
    which a brighter panel clipped.

    ``panel_reflectances`` may contain scalar nominal reflectances or certified
    per-band curves.  The latter should be preferred for publication-quality
    work.
    """
    if not panel_dns:
        raise ValueError("At least one calibration panel is required")
    if len(panel_dns) != len(panel_reflectances):
        raise ValueError("panel_dns and panel_reflectances must be the same length")

    X_raw = np.asarray(panel_dns, dtype=np.float64)
    if X_raw.ndim != 2:
        raise ValueError("Panel spectra must have shape panels x bands")
    n_panels, n_bands = X_raw.shape
    dark = np.asarray(dark_dn, dtype=np.float64)
    if dark.shape != (n_bands,):
        raise ValueError(
            f"Dark spectrum has {dark.size} bands; panels have {n_bands}"
        )

    R = np.empty_like(X_raw)
    for index, reflectance in enumerate(panel_reflectances):
        value = np.asarray(reflectance, dtype=np.float64)
        if value.ndim == 0:
            R[index, :] = float(value)
        elif value.shape == (n_bands,):
            R[index, :] = value
        else:
            raise ValueError(
                "Each panel reflectance must be scalar or match the band count"
            )

    if panel_band_weights is None:
        weights = np.ones_like(X_raw, dtype=np.float64)
    else:
        weights = np.asarray(panel_band_weights, dtype=np.float64).copy()
        if weights.shape != X_raw.shape:
            raise ValueError(
                "panel_band_weights must have shape panels x bands"
            )
        weights = np.clip(weights, 0.0, 1.0)

    signal = X_raw - dark[None, :]
    if dark_noise is None:
        noise = np.ones(n_bands, dtype=np.float64)
    else:
        noise = np.asarray(dark_noise, dtype=np.float64)
        if noise.shape != (n_bands,):
            raise ValueError("dark_noise must match the band count")
        # Integer DN often produces a zero MAD in quiet bands. One DN is a
        # conservative floor that keeps the SNR finite and reproducible.
        noise = np.maximum(np.abs(noise), 1.0)

    snr = signal / noise[None, :]
    snr_start = max(1.0, float(min_snr) * 0.5)
    snr_end = max(snr_start + 1e-6, float(min_snr))
    snr_weight = np.clip((snr - snr_start) / (snr_end - snr_start), 0.0, 1.0)
    snr_weight = snr_weight * snr_weight * (3.0 - 2.0 * snr_weight)
    weights *= snr_weight
    finite = np.isfinite(signal) & np.isfinite(R) & (signal > 0)
    weights[~finite] = 0.0

    denominator = np.sum(weights * signal * signal, axis=0)
    numerator = np.sum(weights * signal * R, axis=0)
    valid = np.isfinite(denominator) & (denominator > 1e-12) & (weights.sum(axis=0) > 0)
    a = np.full(n_bands, np.nan, dtype=np.float64)
    a[valid] = numerator[valid] / denominator[valid]
    b = -a * dark

    prediction = a[None, :] * signal
    residual = R - prediction
    weight_sum = weights.sum(axis=0)
    rmse = np.full(n_bands, np.nan, dtype=np.float64)
    nonzero = weight_sum > 0
    rmse[nonzero] = np.sqrt(
        np.sum(weights[:, nonzero] * residual[:, nonzero] ** 2, axis=0)
        / weight_sum[nonzero]
    )

    individual_a = np.divide(
        R,
        signal,
        out=np.full_like(R, np.nan),
        where=finite,
    )
    mean_individual = np.divide(
        np.sum(weights * np.nan_to_num(individual_a), axis=0),
        weight_sum,
        out=np.full(n_bands, np.nan),
        where=weight_sum > 0,
    )
    coefficient_delta = np.where(
        weights > 0,
        individual_a - mean_individual[None, :],
        0.0,
    )
    spread = np.divide(
        np.sum(weights * coefficient_delta ** 2, axis=0),
        weight_sum,
        out=np.full(n_bands, np.nan),
        where=weight_sum > 0,
    )
    coefficient_cv = np.sqrt(spread) / np.maximum(np.abs(mean_individual), 1e-12)
    effective_count = np.sum(weights > 1e-6, axis=0).astype(int)

    nominal = np.asarray([
        float(np.nanmedian(np.asarray(value, dtype=np.float64)))
        for value in panel_reflectances
    ])
    brightest_index = int(np.nanargmax(nominal))
    fallback = (
        (weights[brightest_index] <= 1e-6)
        & (effective_count > 0)
    )

    qc = {
        "method": "dark-referenced weighted multi-panel",
        "formula": "R=a*(DN-dark); weighted across valid panel observations",
        "panel_weights": weights,
        "panel_snr": snr,
        "valid_band_mask": valid,
        "invalid_band_indices": np.flatnonzero(~valid).astype(int).tolist(),
        "invalid_band_count": int((~valid).sum()),
        "effective_panel_count": effective_count,
        "blended_band_count": int(np.sum(effective_count >= 2)),
        "fallback_band_count": int(np.sum(fallback)),
        "fallback_band_indices": np.flatnonzero(fallback).astype(int).tolist(),
        "brightest_panel_index": brightest_index,
        "weighted_rmse": rmse,
        "coefficient_cv": coefficient_cv,
        "median_coefficient_cv": (
            float(np.nanmedian(coefficient_cv[effective_count >= 2]))
            if np.any(effective_count >= 2) else None
        ),
    }
    return a.astype(np.float32), b.astype(np.float32), qc


def constant_dark_reference(
    n_bands: int,
    dark_dn: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create an explicitly synthetic, spectrally flat Dark reference.

    This is a fallback for acquisitions without a measured sensor-Dark frame.
    A one-DN noise floor is returned so downstream SNR weighting remains finite.
    The QC metadata deliberately labels the reference as synthetic so exported
    calibration files cannot be mistaken for measured-Dark processing.
    """
    bands = int(n_bands)
    value = float(dark_dn)
    if bands <= 0:
        raise ValueError("n_bands must be greater than zero")
    if not np.isfinite(value) or value < 0:
        raise ValueError("dark_dn must be a finite value greater than or equal to zero")

    dark = np.full(bands, value, dtype=np.float32)
    noise = np.ones(bands, dtype=np.float32)
    qc = {
        "usable": True,
        "source_type": "synthetic_constant",
        "constant_dn": value,
        "sample_pixels": 0,
        "noise_mad_by_band": noise.copy(),
        "warning": (
            "Synthetic constant Dark was used because no measured Dark frame "
            "was supplied."
        ),
    }
    return dark, noise, qc


# ---------------------------------------------------------------- #
# Conversion
# ---------------------------------------------------------------- #

def flat_field_reflectance(
    data: np.ndarray,
    panel_dn: np.ndarray,
    panel_reflectance: float | np.ndarray = 0.99,
    clip: Optional[tuple[float, float]] = (0.0, 1.5),
) -> np.ndarray:
    """
    Single-panel flat-field conversion: R = DN / DN_panel · R_panel.

    panel_reflectance may be a scalar (spectrally flat panel) or a per-band
    array of the panel's certified reflectance.
    """
    panel_dn = np.asarray(panel_dn, dtype=np.float64)
    if panel_dn.shape[0] != data.shape[2]:
        raise ValueError(
            f"Panel spectrum has {panel_dn.shape[0]} bands, cube has {data.shape[2]}"
        )

    safe = np.where(np.abs(panel_dn) < 1e-9, np.nan, panel_dn)
    refl = data / safe * np.asarray(panel_reflectance, dtype=np.float64)

    n_bad = int(np.isnan(safe).sum())
    if n_bad:
        logger.warning(f"  {n_bad} bands had ~zero panel signal and became NaN")

    if clip is not None:
        refl = np.clip(refl, clip[0], clip[1])
    return refl.astype(np.float32)


def empirical_line_coeffs(
    panel_dns: Sequence[np.ndarray],
    panel_reflectances: Sequence[float | np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-band least-squares fit R = a·DN + b across two or more panels.

    Returns (a, b), each shaped (B,). The intercept b absorbs the dark offset
    and additive path radiance.
    """
    if len(panel_dns) < 2:
        raise ValueError("Empirical line needs at least 2 panels")
    if len(panel_dns) != len(panel_reflectances):
        raise ValueError("panel_dns and panel_reflectances must be the same length")

    X = np.asarray(panel_dns, dtype=np.float64)            # (P, B)
    B = X.shape[1]
    R = np.empty_like(X)
    for i, r in enumerate(panel_reflectances):
        R[i, :] = np.asarray(r, dtype=np.float64) if np.ndim(r) else float(r)

    xm = X.mean(axis=0)
    rm = R.mean(axis=0)
    var = ((X - xm) ** 2).sum(axis=0)
    var = np.where(var < 1e-12, np.nan, var)
    a = ((X - xm) * (R - rm)).sum(axis=0) / var
    b = rm - a * xm

    n_bad = int(np.isnan(a).sum())
    if n_bad:
        logger.warning(f"  {n_bad}/{B} bands had no DN spread across panels")

    # Fit quality
    pred = a[None, :] * X + b[None, :]
    ss_res = ((R - pred) ** 2).sum(axis=0)
    ss_tot = ((R - rm) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)
    logger.info(
        f"  Empirical line over {len(panel_dns)} panels: "
        f"median R² = {np.nanmedian(r2):.4f}"
    )
    return a, b


def empirical_line_reflectance(
    data: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    clip: Optional[tuple[float, float]] = (0.0, 1.5),
) -> np.ndarray:
    """Apply empirical-line coefficients to a cube."""
    refl = data * a + b
    if clip is not None:
        refl = np.clip(refl, clip[0], clip[1])
    return refl.astype(np.float32)


# ---------------------------------------------------------------- #
# Panel detection in a reference scan
# ---------------------------------------------------------------- #

def detect_panels(
    data: np.ndarray,
    n_panels: int = 3,
    qe: Optional[np.ndarray] = None,
    min_pixels: int = 60,
    flatness_limit: float = 0.60,
) -> list[dict]:
    """
    Find calibration panels in a reference scan.

    Panels are bright, spatially compact and spectrally flat, and a grey-scale
    set forms distinct brightness levels. Returns up to *n_panels* regions
    sorted brightest-first, each as
    ``{"box": [r0, r1, c0, c1], "n_pixels": int, "brightness": float,
       "flatness": float, "spectrum": ndarray}``.

    Flatness is the coefficient of variation of the QE-corrected spectrum, so a
    spectrally neutral panel scores low. Pass *qe* when available; without it
    the sensor response is left in and the threshold is far less meaningful.

    Detection is a starting point, not an oracle — always eyeball the boxes
    before trusting the reflectance they produce.
    """
    try:
        from scipy import ndimage
    except ImportError:
        raise ImportError("scipy is required for panel detection: pip install scipy")

    H, W, B = data.shape
    cube = data.astype(np.float32)
    if qe is not None:
        qe = np.asarray(qe, dtype=np.float32)
        if qe.shape[0] != B:
            raise ValueError(f"QE has {qe.shape[0]} bands, cube has {B}")
        cube = cube / np.where(np.abs(qe) < 1e-12, np.nan, qe)

    bright = np.nanmean(cube, axis=2)
    cv = np.nanstd(cube, axis=2) / np.maximum(bright, 1e-9)

    def blobs_above(pct: float) -> list[dict]:
        """Panel-like blobs among pixels brighter than the *pct* percentile."""
        mask = bright > np.nanpercentile(bright, pct)
        if qe is not None:
            # Only meaningful once the sensor response is divided out.
            mask &= cv < flatness_limit
        lab, n = ndimage.label(mask)
        out: list[dict] = []
        for k in range(1, n + 1):
            m = lab == k
            npx = int(m.sum())
            if npx < min_pixels:
                continue
            rr, cc = np.where(m)
            r0, r1 = rr.min(), rr.max() + 1
            c0, c1 = cc.min(), cc.max() + 1
            # Reject stringy blobs: a panel fills most of its bounding box.
            if npx < 0.5 * (r1 - r0) * (c1 - c0):
                continue
            # Shrink to the central 60% so edges and shadowed borders stay
            # out of the reference spectrum.
            dr, dc = (r1 - r0) * 0.2, (c1 - c0) * 0.2
            box = [int(r0 + dr), int(np.ceil(r1 - dr)),
                   int(c0 + dc), int(np.ceil(c1 - dc))]
            if box[1] <= box[0] or box[3] <= box[2]:
                box = [int(r0), int(r1), int(c0), int(c1)]

            spec = panel_spectrum(data, box)
            sq = spec / qe if qe is not None else spec
            out.append({
                "box": box,
                "n_pixels": npx,
                "brightness": float(np.nanmean(bright[m])),
                "flatness": float(np.std(sq / np.mean(sq))),
                "spectrum": spec,
            })
        return out

    # A fixed brightness cut fails as soon as the panels cover more or less of
    # the frame than assumed — a grey-scale set spans a wide brightness range,
    # so the darkest panel easily falls under a high threshold. Walk the
    # threshold down and stop as soon as the whole set separates out.
    found: list[dict] = []
    for pct in (90, 80, 70, 60, 50, 40, 30):
        cand = blobs_above(pct)
        if len(cand) > len(found):
            found = cand
        if len(found) >= n_panels:
            logger.info(f"  Panels separated at the {pct}th brightness percentile")
            break

    if not found:
        logger.warning("  No panel-like regions found")
        return []

    found.sort(key=lambda d: d["brightness"], reverse=True)
    if len(found) > n_panels:
        found = found[:n_panels]

    logger.info(f"  Detected {len(found)} panel candidate(s)")
    for i, p in enumerate(found, 1):
        logger.info(
            f"    #{i} box={p['box']} px={p['n_pixels']:,} "
            f"brightness={p['brightness']:.1f} flatness={p['flatness']:.3f}"
        )
    return found


def reflectance_from_reference(
    data: np.ndarray,
    reference: np.ndarray,
    panel_reflectances: Sequence[float],
    qe: Optional[np.ndarray] = None,
    boxes: Optional[Sequence[Sequence[int]]] = None,
) -> tuple[np.ndarray, dict]:
    """
    Convert *data* to reflectance using panels measured in a *reference* scan.

    Give explicit *boxes* (one per entry of panel_reflectances, brightest
    first) when you have them; otherwise the panels are auto-detected.

    Returns (reflectance cube, info dict). With two or more panels this uses
    the empirical line, so the dark offset is absorbed by the fitted intercept
    and no dark frame is needed.

    Caveat: reference and target must share the same illumination. A reference
    taken at a different time or sun angle introduces an error this cannot
    detect or correct.
    """
    if reference.shape[2] != data.shape[2]:
        raise ValueError(
            f"Reference has {reference.shape[2]} bands, target has {data.shape[2]}"
        )

    if boxes is not None:
        if len(boxes) != len(panel_reflectances):
            raise ValueError("boxes and panel_reflectances must be the same length")
        panels = [{"box": list(b), "spectrum": panel_spectrum(reference, b)}
                  for b in boxes]
    else:
        panels = detect_panels(reference, n_panels=len(panel_reflectances), qe=qe)
        if len(panels) < len(panel_reflectances):
            raise ValueError(
                f"Found only {len(panels)} panel(s) but {len(panel_reflectances)} "
                f"reflectance value(s) were given. Pass explicit boxes."
            )

    dns = [p["spectrum"] for p in panels]
    info: dict = {"boxes": [p["box"] for p in panels],
                  "panel_reflectances": list(panel_reflectances)}

    if len(dns) >= 2:
        a, b = empirical_line_coeffs(dns, panel_reflectances)
        refl = empirical_line_reflectance(data, a, b)
        info.update({"method": "empirical_line", "a": a, "b": b})
    else:
        refl = flat_field_reflectance(data, dns[0], panel_reflectances[0])
        info["method"] = "flat_field"
        logger.warning(
            "  Only one panel: dark offset is uncorrected. Two or more panels "
            "give a far more accurate result."
        )

    return refl, info


# ---------------------------------------------------------------- #
# Fit diagnostics
# ---------------------------------------------------------------- #

def empirical_line_fit_quality(
    panel_dns: Sequence[np.ndarray],
    panel_reflectances: Sequence[float | np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
) -> dict:
    """
    Per-band goodness of fit for an empirical line.

    R² near 1 in every band means the panels behave linearly, which is the
    whole assumption behind the method. Bands where it dips are usually
    saturated on the bright panel or buried in noise on the dark one — check
    those before trusting reflectance there.

    With exactly two panels a straight line always passes through both points,
    so R² is 1 by construction and says nothing; residuals only become
    informative from three panels up.
    """
    X = np.asarray(panel_dns, dtype=np.float64)                    # (P, B)
    R = np.empty_like(X)
    for i, r in enumerate(panel_reflectances):
        R[i, :] = np.asarray(r, dtype=np.float64) if np.ndim(r) else float(r)

    pred = a[None, :] * X + b[None, :]
    resid = R - pred
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((R - R.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)

    return {
        "r2": r2,
        "max_abs_residual": np.abs(resid).max(axis=0),
        "rmse": np.sqrt((resid ** 2).mean(axis=0)),
        "median_r2": float(np.nanmedian(r2)),
        "min_r2": float(np.nanmin(r2)),
        "n_panels": len(panel_dns),
        "informative": len(panel_dns) >= 3,
    }


# ---------------------------------------------------------------- #
# Persistence
# ---------------------------------------------------------------- #

def save_calibration(path: str, a, b, wavelengths=None, meta: Optional[dict] = None) -> str:
    """Write empirical-line coefficients to a .npz so they can be reused."""
    import json
    from pathlib import Path as _P

    p = _P(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        a=np.asarray(a),
        b=np.asarray(b),
        wavelengths=(np.asarray(wavelengths) if wavelengths is not None
                     else np.array([])),
        meta=np.array(json.dumps(meta or {}, ensure_ascii=False)),
    )
    logger.info(f"  Calibration saved: {p}")
    return str(p)


def load_calibration(path: str) -> dict:
    """Read coefficients written by :func:`save_calibration`."""
    with np.load(path, allow_pickle=False) as z:
        # A direct White/Dark profile can still be used anywhere that expects
        # legacy a/b coefficients. Directory-based nearest-White selection is
        # handled by resolve_calibration().
        if "profile_type" in z.files:
            profile = _profile_from_npz(z, str(path))
            return _profile_coefficients(profile)
        wl = z["wavelengths"] if "wavelengths" in z.files else np.array([])
        return {
            "a": np.asarray(z["a"]),
            "b": np.asarray(z["b"]),
            "wavelengths": (wl.tolist() if wl.size else None),
            "meta": json.loads(str(z["meta"])) if "meta" in z.files else {},
            "calibration_type": "empirical_line_coefficients",
            "selected_profile": str(Path(path).resolve()),
        }


def parse_acquisition_time(value: str | Path | None) -> Optional[datetime]:
    """Infer acquisition time from an explicit ISO string or filename."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    # Explicit ISO strings from the UI take priority.
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass

    name = Path(raw).name
    for digits, fmt in ((14, "%Y%m%d%H%M%S"), (12, "%Y%m%d%H%M")):
        for match in re.finditer(rf"(?<!\d)(\d{{{digits}}})(?!\d)", name):
            try:
                parsed = datetime.strptime(match.group(1), fmt)
            except ValueError:
                continue
            if 1990 <= parsed.year <= 2100:
                return parsed

    # Deliberately do not fall back to filesystem modified time. Copying a
    # Ceres/ENVI file changes mtime and could silently select the wrong White.
    return None


def discover_calibration_candidates(
    source_path: str | Path,
    *,
    search_roots: Optional[Sequence[str | Path]] = None,
) -> list[Path]:
    """Find conservatively named calibration files near one source image.

    The search is deliberately narrow: the source directory, its
    ``calibration`` child, and explicitly supplied output roots.  Only a
    generic ``calibration.npz`` or filenames tied to the source stem are
    returned.  Broad ``*.npz`` scans could silently select another sensor,
    date, or field and are therefore avoided.
    """
    source = Path(source_path).expanduser()
    stems = [source.stem]
    # The panel tab names profiles from the displayed source filename. When a
    # binary ENVI entry (``scene.bil``) is loaded directly that intentionally
    # produces ``scene.bil_weighted_dark_calibration.npz``.
    if source.suffix.lower() != ".hdr" and source.name not in stems:
        stems.insert(0, source.name)
    # ENVI headers are often named ``scene.bil.hdr`` while outputs use
    # ``scene_weighted_dark_calibration.npz``.
    nested_stem = Path(source.stem).stem
    if nested_stem and nested_stem not in stems:
        stems.append(nested_stem)

    directories = [source.parent, source.parent / "calibration"]
    for root_value in search_roots or []:
        if not str(root_value).strip():
            continue
        root = Path(root_value).expanduser()
        directories.extend([root / "calibration", root])

    filenames: list[str] = []
    for stem in stems:
        filenames.extend(
            [
                f"{stem}_weighted_dark_calibration.npz",
                f"{stem}_calibration.npz",
                f"{stem}.calibration.npz",
            ]
        )
    filenames.append("calibration.npz")

    candidates: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
        for filename in filenames:
            candidate = directory / filename
            if not candidate.is_file():
                continue
            identity = str(candidate.resolve()).casefold()
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(candidate.resolve())
    return candidates


def robust_reference_spectrum(
    data: np.ndarray,
    *,
    max_pixels: int = 200_000,
) -> tuple[np.ndarray, dict]:
    """Return a robust per-band median from a uniform White or dark frame."""
    if data.ndim != 3:
        raise ValueError(f"Expected H x W x bands reference cube, got {data.shape}")
    flat = np.asarray(data).reshape(-1, data.shape[2])
    if len(flat) > max_pixels:
        at = np.linspace(0, len(flat) - 1, max_pixels, dtype=np.int64)
        flat = flat[at]
    finite = np.all(np.isfinite(flat), axis=1)
    flat = np.asarray(flat[finite], dtype=np.float64)
    if not len(flat):
        raise ValueError("Reference image contains no finite spectra")
    spectrum = np.median(flat, axis=0)
    band_mad = np.median(np.abs(flat - spectrum[None, :]), axis=0)
    robust_noise = 1.4826 * band_mad
    brightness = np.mean(flat, axis=1)
    qc = {
        "sample_pixels": int(len(flat)),
        "brightness_median": float(np.median(brightness)),
        "brightness_cv": float(np.std(brightness) / max(abs(np.mean(brightness)), 1e-12)),
        "max_dn": float(np.max(flat)),
        "noise_mad_by_band": robust_noise.astype(float).tolist(),
    }
    qc.update(
        panel_saturation_metrics(
            flat,
            observed_max=float(np.nanmax(data)),
        )
    )
    return spectrum.astype(np.float32), qc


def save_white_dark_profile(
    path: str | Path,
    white: np.ndarray,
    dark: np.ndarray,
    *,
    wavelengths: Optional[Sequence[float]] = None,
    white_reflectance: float | Sequence[float] = 0.99,
    white_time: str | datetime | None = None,
    dark_time: str | datetime | None = None,
    meta: Optional[dict] = None,
) -> str:
    """Persist raw White and sensor-dark spectra plus acquisition provenance."""
    white_arr = np.asarray(white, dtype=np.float32)
    dark_arr = np.asarray(dark, dtype=np.float32)
    if white_arr.ndim != 1 or dark_arr.shape != white_arr.shape:
        raise ValueError("White and dark spectra must be one-dimensional and equal length")
    reflectance = np.asarray(white_reflectance, dtype=np.float32)
    if reflectance.ndim == 0:
        reflectance = np.full(white_arr.shape, float(reflectance), dtype=np.float32)
    if reflectance.shape != white_arr.shape:
        raise ValueError("White reflectance must be scalar or match the spectrum bands")
    if np.any(~np.isfinite(reflectance)) or np.any(reflectance <= 0):
        raise ValueError("White reflectance must contain finite positive values")

    def iso(value: str | datetime | None) -> str:
        parsed = value if isinstance(value, datetime) else parse_acquisition_time(value)
        return parsed.isoformat(timespec="seconds") if parsed is not None else ""

    p = Path(path)
    if p.suffix.lower() != ".npz":
        p = p.with_suffix(".npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        profile_type=np.array(WHITE_DARK_PROFILE_TYPE),
        white=white_arr,
        dark=dark_arr,
        white_reflectance=reflectance,
        wavelengths=(np.asarray(wavelengths, dtype=np.float64)
                     if wavelengths is not None else np.array([])),
        white_time=np.array(iso(white_time)),
        dark_time=np.array(iso(dark_time)),
        meta=np.array(json.dumps(meta or {}, ensure_ascii=False)),
    )
    logger.info(f"  White/Dark calibration profile saved: {p}")
    return str(p)


def _profile_from_npz(z, source_path: str) -> dict:
    kind = str(z["profile_type"])
    if kind != WHITE_DARK_PROFILE_TYPE:
        raise ValueError(f"Unsupported calibration profile type: {kind}")
    wl = np.asarray(z["wavelengths"], dtype=np.float64)
    return {
        "profile_type": kind,
        "white": np.asarray(z["white"], dtype=np.float32),
        "dark": np.asarray(z["dark"], dtype=np.float32),
        "white_reflectance": np.asarray(z["white_reflectance"], dtype=np.float32),
        "wavelengths": wl.tolist() if wl.size else None,
        "white_time": str(z["white_time"]) if "white_time" in z.files else "",
        "dark_time": str(z["dark_time"]) if "dark_time" in z.files else "",
        "meta": json.loads(str(z["meta"])) if "meta" in z.files else {},
        "source_path": str(Path(source_path).resolve()),
    }


def load_white_dark_profile(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return _profile_from_npz(z, str(path))


def _profile_coefficients(profile: dict) -> dict:
    white = np.asarray(profile["white"], dtype=np.float64)
    dark = np.asarray(profile["dark"], dtype=np.float64)
    rho = np.asarray(profile["white_reflectance"], dtype=np.float64)
    denominator = white - dark
    # A near-zero or negative White-Dark response cannot produce defensible
    # reflectance. Preserve it as NaN rather than silently clipping.
    scale = max(float(np.nanmedian(np.abs(denominator))), 1.0)
    valid = np.isfinite(denominator) & (denominator > scale * 1e-8)
    a = np.full(white.shape, np.nan, dtype=np.float32)
    b = np.full(white.shape, np.nan, dtype=np.float32)
    a[valid] = (rho[valid] / denominator[valid]).astype(np.float32)
    b[valid] = (-dark[valid] * a[valid]).astype(np.float32)
    meta = dict(profile.get("meta") or {})
    meta.update({
        "method": "sensor dark + nearest white",
        "formula": "reflectance=(raw-dark)/(white-dark)*white_reflectance",
        "white_time": profile.get("white_time", ""),
        "dark_time": profile.get("dark_time", ""),
        "invalid_bands": int((~valid).sum()),
    })
    return {
        "a": a,
        "b": b,
        "wavelengths": profile.get("wavelengths"),
        "meta": meta,
        "calibration_type": WHITE_DARK_PROFILE_TYPE,
        "selected_profile": profile.get("source_path", ""),
        "white": white.astype(np.float32),
        "dark": dark.astype(np.float32),
        "white_reflectance": rho.astype(np.float32),
    }


def resolve_calibration(
    profile_or_directory: str | Path,
    *,
    target_source: str | Path | None = None,
    wavelengths: Optional[Sequence[float]] = None,
) -> dict:
    """Resolve legacy coefficients or the time-nearest White/Dark profile."""
    root = Path(profile_or_directory).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Calibration path not found: {root}")

    if root.is_file():
        resolved = load_calibration(str(root))
    else:
        target_time = parse_acquisition_time(target_source)
        if target_time is None:
            raise ValueError(
                "Target acquisition time could not be determined; choose one profile file "
                "instead of a profile directory."
            )
        candidates = []
        for candidate in sorted(root.rglob("*.npz")):
            try:
                profile = load_white_dark_profile(candidate)
            except Exception:
                continue
            profile_wl = profile.get("wavelengths")
            if profile_wl is not None and wavelengths is not None:
                profile_wl_arr = np.asarray(profile_wl, dtype=np.float64)
                target_wl_arr = np.asarray(wavelengths, dtype=np.float64)
                if (
                    profile_wl_arr.shape != target_wl_arr.shape
                    or not np.allclose(profile_wl_arr, target_wl_arr, rtol=0, atol=1.0)
                ):
                    continue
            white_time = parse_acquisition_time(profile.get("white_time"))
            if white_time is None:
                white_time = parse_acquisition_time(candidate)
            if white_time is not None:
                candidates.append((abs((white_time - target_time).total_seconds()), profile))
        if not candidates:
            raise ValueError(f"No timestamped White/Dark profiles found in: {root}")
        delta_seconds, selected = min(candidates, key=lambda pair: pair[0])
        resolved = _profile_coefficients(selected)
        resolved["meta"]["target_time"] = target_time.isoformat(timespec="seconds")
        resolved["meta"]["white_time_delta_seconds"] = float(delta_seconds)

    cal_wl = resolved.get("wavelengths")
    if cal_wl is not None and wavelengths is not None:
        cal_wl_arr = np.asarray(cal_wl, dtype=np.float64)
        scene_wl_arr = np.asarray(wavelengths, dtype=np.float64)
        if (
            cal_wl_arr.shape != scene_wl_arr.shape
            or not np.allclose(cal_wl_arr, scene_wl_arr, rtol=0, atol=1.0)
        ):
            raise ValueError(
                "Calibration wavelengths do not match the target image. "
                "Use profiles from the same sensor and band configuration."
            )
    return resolved


def apply_resolved_calibration(data: np.ndarray, calibration: dict) -> np.ndarray:
    """Apply R=a*DN+b without clipping scientifically useful out-of-range values."""
    a = np.asarray(calibration["a"], dtype=np.float32)
    b = np.asarray(calibration["b"], dtype=np.float32)
    if a.shape != (data.shape[-1],) or b.shape != a.shape:
        raise ValueError("Calibration bands do not match data")
    return np.asarray(data, dtype=np.float32) * a + b


def export_calibrated_binned_envi(
    source_path: str | Path,
    profile_or_directory: str | Path,
    output_path: str | Path,
    *,
    bin_factor: int = 4,
    chunk_output_rows: int = 32,
) -> dict:
    """Stream a full ENVI cube into a spatially binned float32 reflectance BIL."""
    try:
        import spectral
    except ImportError as exc:
        raise ImportError("spectral is required to export ENVI BIL files") from exc

    source = Path(source_path)
    if source.suffix.lower() == ".hdr":
        image = spectral.open_image(str(source))
    else:
        header_candidates = [source.with_suffix(".hdr"), Path(str(source) + ".hdr")]
        header = next((item for item in header_candidates if item.is_file()), None)
        if header is None:
            raise FileNotFoundError(f"ENVI header not found for: {source}")
        image = spectral.envi.open(str(header), str(source))

    factor = int(bin_factor)
    if factor < 1:
        raise ValueError("bin_factor must be at least 1")
    mm = image.open_memmap(interleave="bip")
    height, width, bands = mm.shape
    out_height, out_width = height // factor, width // factor
    if out_height < 1 or out_width < 1:
        raise ValueError("Binning factor is larger than the input image")
    wavelengths = (
        list(image.bands.centers)
        if image.bands is not None and image.bands.centers is not None else None
    )
    calibration = resolve_calibration(
        profile_or_directory, target_source=source, wavelengths=wavelengths
    )

    output = Path(output_path)
    if output.suffix.lower() == ".hdr":
        output = output.with_suffix(".bil")
    elif output.suffix.lower() != ".bil":
        output = output.with_suffix(".bil")
    header_out = output.with_suffix(".hdr")
    manifest_out = output.with_suffix(".calibration.json")
    if output.exists() or header_out.exists() or manifest_out.exists():
        raise FileExistsError(
            f"Output already exists: {output}, {header_out}, or {manifest_out}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    a = np.asarray(calibration["a"], dtype=np.float32)
    b = np.asarray(calibration["b"], dtype=np.float32)
    with output.open("xb") as stream:
        for out_start in range(0, out_height, max(1, int(chunk_output_rows))):
            out_end = min(out_height, out_start + max(1, int(chunk_output_rows)))
            in_start, in_end = out_start * factor, out_end * factor
            raw = np.asarray(mm[in_start:in_end, :out_width * factor, :], dtype=np.float32)
            corrected = raw * a + b
            binned = np.nanmean(
                corrected.reshape(out_end - out_start, factor, out_width, factor, bands),
                axis=(1, 3),
            ).astype("<f4", copy=False)
            # ENVI BIL ordering is line, band, sample.
            binned.transpose(0, 2, 1).tofile(stream)

    wl_text = ""
    if wavelengths is not None:
        wl_text = "\nwavelength = {" + ", ".join(f"{v:.8g}" for v in wavelengths) + "}"
    description = (
        "Science-ready spatially binned reflectance; "
        f"factor={factor}; calibration={Path(calibration['selected_profile']).name}"
    )
    header_out.write_text(
        "ENVI\n"
        f"description = {{{description}}}\n"
        f"samples = {out_width}\nlines = {out_height}\nbands = {bands}\n"
        "header offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
        "interleave = bil\nbyte order = 0\n"
        "data units = Reflectance"
        "\nwavelength units = Nanometers"
        f"{wl_text}\n",
        encoding="ascii",
    )
    manifest = {
        "source_file": str(source.resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_data": str(output.resolve()),
        "output_header": str(header_out.resolve()),
        "shape": [out_height, out_width, bands],
        "spatial_bin_factor": factor,
        "output_dtype": "float32",
        "output_units": "reflectance",
        "formula": "reflectance=(raw-dark)/(white-dark)*white_reflectance",
        "selected_profile": calibration["selected_profile"],
        "calibration_meta": calibration.get("meta", {}),
    }
    manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "data_file": str(output.resolve()),
        "header_file": str(header_out.resolve()),
        "manifest_file": str(manifest_out.resolve()),
        "shape": [out_height, out_width, bands],
        "bin_factor": factor,
        "selected_profile": calibration["selected_profile"],
        "calibration_meta": calibration.get("meta", {}),
    }


# ---------------------------------------------------------------- #
# Sensor-only fallback
# ---------------------------------------------------------------- #

def qe_correct(data: np.ndarray, qe: np.ndarray) -> np.ndarray:
    """
    Divide out the sensor quantum-efficiency curve.

    Removes the instrument's spectral bias but NOT the illumination spectrum,
    so the result is a relative quantity, not reflectance. Useful only when no
    panel is available in the scene.
    """
    qe = np.asarray(qe, dtype=np.float64)
    if qe.shape[0] != data.shape[2]:
        raise ValueError(f"QE has {qe.shape[0]} bands, cube has {data.shape[2]}")
    safe = np.where(np.abs(qe) < 1e-12, np.nan, qe)
    return (data / safe).astype(np.float32)


def load_hyspex_qe(h5_path: str, sensor: str = "vnir") -> np.ndarray:
    """Read the quantum-efficiency curve from a HySpex .hyspex.h5 file."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        return f[f"{sensor}/intrinsics/quantum_efficiency"][:].astype(np.float64)
