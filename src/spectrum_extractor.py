"""
spectrum_extractor.py
---------------------
Extract representative spectra (mean ± std, percentiles) for each
class in the classification map.
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
    }
    """

    def __init__(self, config: dict):
        self.cfg = config

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
            cid   = cinfo["id"]
            n_px  = cinfo["n_pixels"]
            name  = cinfo["name"]

            mask  = class_map == cid
            pixels = data[mask]                       # (N, B)

            # Random subsample for large classes
            if len(pixels) > max_pixels_per_class:
                rng = np.random.default_rng(seed=42)
                idx = rng.choice(len(pixels), size=max_pixels_per_class, replace=False)
                pixels = pixels[idx]
                logger.info(f"  {name}: sampled {max_pixels_per_class:,} / {n_px:,} pixels")

            if len(pixels) == 0:
                logger.warning(f"  {name}: 0 pixels – skipping")
                continue

            stats = self._compute_stats(pixels)
            entry = {**cinfo, "wavelengths": wavelengths, **stats}
            results.append(entry)
            logger.info(
                f"  {name} (id={cid}): {n_px:,} px | "
                f"mean reflectance = {stats['mean'].mean():.3f} ± {stats['std'].mean():.3f}"
            )

        return results

    def save_csv(
        self,
        spectra: List[Dict[str, Any]],
        path: str | Path,
    ) -> None:
        """Save spectra to a wide-format CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not spectra:
            logger.warning("  No spectra to save")
            return

        n_bands = len(spectra[0]["mean"])
        wl = spectra[0]["wavelengths"]

        if wl is not None:
            index_col = wl
            index_name = "wavelength_nm"
        else:
            index_col = list(range(n_bands))
            index_name = "band_index"

        rows = {}
        for s in spectra:
            name = s["name"].replace(" ", "_")
            rows[f"{name}_mean"]   = s["mean"]
            rows[f"{name}_std"]    = s["std"]
            rows[f"{name}_median"] = s["median"]
            rows[f"{name}_q25"]    = s["q25"]
            rows[f"{name}_q75"]    = s["q75"]

        df = pd.DataFrame(rows, index=index_col)
        df.index.name = index_name
        df.to_csv(path)
        logger.info(f"  Spectra CSV saved: {path}")

    # ============================================================
    # Private
    # ============================================================

    @staticmethod
    def _compute_stats(pixels: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute spectral statistics over pixel matrix (N, B)."""
        pixels = pixels.astype(np.float64)
        return {
            "mean":   np.mean(pixels, axis=0).astype(np.float32),
            "std":    np.std(pixels, axis=0).astype(np.float32),
            "median": np.median(pixels, axis=0).astype(np.float32),
            "q25":    np.percentile(pixels, 25, axis=0).astype(np.float32),
            "q75":    np.percentile(pixels, 75, axis=0).astype(np.float32),
            "min":    np.min(pixels, axis=0).astype(np.float32),
            "max":    np.max(pixels, axis=0).astype(np.float32),
        }
