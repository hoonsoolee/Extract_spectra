"""
spectrum_extractor.py
---------------------
Extract representative spectra (mean ± std, percentiles, and robust
neighborhood averages) for each class in the classification map.

Per-class output columns (CSV):
  mean     – arithmetic mean over all class pixels
  std      – standard deviation  (spread / noise indicator)
  median   – per-band median     (robust but not a real pixel)
  q25      – 25th percentile
  q75      – 75th percentile
  mna      – Medoid-Neighbourhood Average: mean of the N pixels
             closest to the median in Euclidean spectral distance
             (outlier-free, still a real-pixel average)
  sam_avg  – SAM-Neighbourhood Average: same idea but distance is
             the Spectral Angle (illumination-invariant), so pixels
             are selected by spectral *shape* similarity to the median.
             Best suited for Vcmax / biochemical trait matching.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class SpectrumExtractor:
    """
    For each unique class ID in *class_map*, gather all pixel spectra and
    compute summary statistics.

    Output structure (per class):
    {
        "id":           int,
        "name":         str,
        "color":        [R, G, B],
        "n_pixels":     int,
        "fraction":     float,
        "wavelengths":  list[float] or None,
        "mean":         np.ndarray (B,),
        "std":          np.ndarray (B,),
        "median":       np.ndarray (B,),
        "q25":          np.ndarray (B,),
        "q75":          np.ndarray (B,),
        "min":          np.ndarray (B,),
        "max":          np.ndarray (B,),
        "mna":          np.ndarray (B,),   # Medoid-Neighbourhood Average
        "sam_avg":      np.ndarray (B,),   # SAM-Neighbourhood Average
    }
    """

    def __init__(self, config: dict):
        self.cfg = config
        ext_cfg = config.get("extraction", {})
        self._n_neighbors = int(ext_cfg.get("n_neighbors", 100))

    # ============================================================
    # Public
    # ============================================================

    def extract(
        self,
        data: np.ndarray,
        class_map: np.ndarray,
        class_info: List[Dict[str, Any]],
        wavelengths: Optional[List[float]] = None,
        max_pixels_per_class: int = 50_000,
    ) -> List[Dict[str, Any]]:
        """
        Parameters
        ----------
        data                  : (H, W, B) float32
        class_map             : (H, W) int32
        class_info            : list from classifier.classify()
        wavelengths           : list of nm values or None
        max_pixels_per_class  : cap pixel sample to avoid memory issues

        Returns
        -------
        List of class dicts with added spectral statistics.
        """
        results = []
        for cinfo in class_info:
            cid  = cinfo["id"]
            n_px = cinfo["n_pixels"]
            name = cinfo["name"]

            mask   = class_map == cid
            pixels = data[mask]                       # (N, B)

            # Random subsample for large classes
            if len(pixels) > max_pixels_per_class:
                rng = np.random.default_rng(seed=42)
                idx = rng.choice(len(pixels), size=max_pixels_per_class, replace=False)
                pixels = pixels[idx]
                logger.info(f"  {name}: sampled {max_pixels_per_class:,} / {n_px:,} pixels")

            if len(pixels) == 0:
                logger.warning(f"  {name}: 0 pixels - skipping")
                continue

            stats = self._compute_stats(pixels, n_neighbors=self._n_neighbors)
            entry = {**cinfo, "wavelengths": wavelengths, **stats}
            results.append(entry)
            logger.info(
                f"  {name} (id={cid}): {n_px:,} px | "
                f"mean reflectance = {stats['mean'].mean():.3f} +/- {stats['std'].mean():.3f} | "
                f"mna = {stats['mna'].mean():.3f} | sam_avg = {stats['sam_avg'].mean():.3f}"
            )

        return results

    def save_csv(
        self,
        spectra: List[Dict[str, Any]],
        path: str | Path,
    ) -> None:
        """Save spectra to a wide-format CSV file.

        Column order per class:
          mean, std, median, q25, q75, mna, sam_avg
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not spectra:
            logger.warning("  No spectra to save")
            return

        n_bands = len(spectra[0]["mean"])
        wl = spectra[0]["wavelengths"]

        if wl is not None:
            index_col  = wl
            index_name = "wavelength_nm"
        else:
            index_col  = list(range(n_bands))
            index_name = "band_index"

        rows = {}
        for s in spectra:
            name = s["name"].replace(" ", "_")
            rows[f"{name}_mean"]    = s["mean"]
            rows[f"{name}_std"]     = s["std"]
            rows[f"{name}_median"]  = s["median"]
            rows[f"{name}_q25"]     = s["q25"]
            rows[f"{name}_q75"]     = s["q75"]
            rows[f"{name}_mna"]     = s["mna"]
            rows[f"{name}_sam_avg"] = s["sam_avg"]

        df = pd.DataFrame(rows, index=index_col)
        df.index.name = index_name
        df.to_csv(path)
        logger.info(f"  Spectra CSV saved: {path}")

    # ============================================================
    # Private
    # ============================================================

    @staticmethod
    def _compute_stats(
        pixels: np.ndarray,
        n_neighbors: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Compute spectral statistics over pixel matrix (N, B).

        Parameters
        ----------
        pixels       : (N, B) array of reflectance values
        n_neighbors  : number of nearest pixels used for MNA / SAM-avg
                       (capped to N when N < n_neighbors)
        """
        pixels = pixels.astype(np.float64)
        N      = len(pixels)
        k      = min(n_neighbors, N)        # can't pick more than we have

        median_spec = np.median(pixels, axis=0)   # (B,)

        # ── Medoid-Neighbourhood Average (Euclidean) ───────────────
        # Select k pixels closest to the median in L2 distance.
        # Robust to outliers; more stable than the median itself because
        # it averages real pixel vectors instead of per-band statistics.
        euc_dist = np.linalg.norm(pixels - median_spec, axis=1)   # (N,)
        euc_idx  = np.argpartition(euc_dist, k - 1)[:k]           # top-k indices
        mna      = np.mean(pixels[euc_idx], axis=0).astype(np.float32)

        # ── SAM-Neighbourhood Average (Spectral Angle) ─────────────
        # Selects k pixels with the smallest spectral angle relative to
        # the median. Because SAM ignores absolute brightness, pixels are
        # chosen purely by spectral *shape* similarity — illumination
        # (sunlit vs shaded) does not affect selection.
        # Recommended when matching against leaf-level Vcmax / N / Chl
        # measurements where biochemistry (shape) matters, not brightness.
        dot        = np.sum(pixels * median_spec, axis=1)              # (N,)
        norm_px    = np.linalg.norm(pixels, axis=1)                    # (N,)
        norm_med   = np.linalg.norm(median_spec)
        denom      = norm_px * norm_med
        denom      = np.where(denom < 1e-12, 1e-12, denom)            # avoid /0
        sam_dist   = np.arccos(np.clip(dot / denom, -1.0, 1.0))       # (N,) radians
        sam_idx    = np.argpartition(sam_dist, k - 1)[:k]
        sam_avg    = np.mean(pixels[sam_idx], axis=0).astype(np.float32)

        return {
            "mean":    np.mean(pixels, axis=0).astype(np.float32),
            "std":     np.std(pixels,  axis=0).astype(np.float32),
            "median":  median_spec.astype(np.float32),
            "q25":     np.percentile(pixels, 25, axis=0).astype(np.float32),
            "q75":     np.percentile(pixels, 75, axis=0).astype(np.float32),
            "min":     np.min(pixels,  axis=0).astype(np.float32),
            "max":     np.max(pixels,  axis=0).astype(np.float32),
            "mna":     mna,
            "sam_avg": sam_avg,
        }
