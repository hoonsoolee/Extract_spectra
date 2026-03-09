"""
preprocessor.py
---------------
Preprocessing steps for raw hyperspectral cubes.
"""

import logging
import numpy as np
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Applies standard preprocessing to a hyperspectral cube (H × W × B).

    Steps (all optional, controlled by config):
      1. Scale raw DN values to physical reflectance
      2. Remove noisy / water-absorption bands
      3. Clip & normalize to [0, 1]
      4. Savitzky-Golay spectral smoothing
      5. Spatial downsampling
    """

    def __init__(self, config: dict):
        self.cfg = config.get("preprocessing", {})
        self.wl_cfg = config.get("wavelengths", {})

    # ---------------------------------------------------------- #
    # Public
    # ---------------------------------------------------------- #

    def process(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Optional[List[float]]]:
        """
        Run all preprocessing steps.

        Parameters
        ----------
        data        : (H, W, B) float32 array
        wavelengths : list of wavelengths in nm (or None)

        Returns
        -------
        processed_data : (H, W, B') float32
        wavelengths    : updated wavelength list (or None)
        """
        H, W, B = data.shape
        logger.info(f"  Preprocessing: input shape {data.shape}")

        # Resolve wavelengths from config if not in file
        wavelengths = self._resolve_wavelengths(wavelengths, B)

        # 1. Scale DN → reflectance
        scale = self.cfg.get("data_scale")
        if scale:
            data = data / float(scale)
            logger.info(f"  Scaled by 1/{scale}")

        # 2. Remove bad bands
        if self.cfg.get("remove_bad_bands", True) and wavelengths is not None:
            data, wavelengths = self._remove_bad_bands(data, wavelengths)
            logger.info(f"  After bad-band removal: {data.shape[2]} bands")

        # 3. Normalize to [0, 1]
        if self.cfg.get("normalize", True):
            data = self._normalize(data)
            logger.info("  Normalized to [0, 1]")

        # 4. Spectral smoothing
        if self.cfg.get("smooth_spectra", False):
            data = self._smooth_spectra(data)
            logger.info("  Spectral smoothing applied")

        # 5. Spatial downsampling
        factor = int(self.cfg.get("spatial_downsample", 1))
        if factor > 1:
            data = data[::factor, ::factor, :]
            logger.info(f"  Downsampled ×{factor}: {data.shape}")

        logger.info(f"  Preprocessing done: output shape {data.shape}")
        return data, wavelengths

    # ---------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------- #

    def _resolve_wavelengths(
        self,
        wavelengths: Optional[List[float]],
        n_bands: int,
    ) -> Optional[List[float]]:
        """Use config-defined wavelengths if file metadata is missing."""
        if wavelengths is not None:
            return wavelengths

        explicit = self.wl_cfg.get("bands")
        if explicit:
            if len(explicit) != n_bands:
                logger.warning(
                    f"  Config wavelength list length ({len(explicit)}) != "
                    f"n_bands ({n_bands}). Ignoring."
                )
                return None
            return list(explicit)

        start = self.wl_cfg.get("start")
        end = self.wl_cfg.get("end")
        if start is not None and end is not None:
            wl = list(np.linspace(float(start), float(end), n_bands))
            logger.info(f"  Wavelengths inferred from config: {start}–{end} nm")
            return wl

        logger.warning(
            "  No wavelength info available. Some features will be limited."
        )
        return None

    def _remove_bad_bands(
        self,
        data: np.ndarray,
        wavelengths: List[float],
    ) -> Tuple[np.ndarray, List[float]]:
        """Remove bands in water-absorption ranges specified in config."""
        bad_ranges = self.cfg.get("bad_band_ranges", [])
        if not bad_ranges:
            return data, wavelengths

        wl_arr = np.array(wavelengths)
        keep_mask = np.ones(len(wavelengths), dtype=bool)
        for lo, hi in bad_ranges:
            keep_mask &= ~((wl_arr >= lo) & (wl_arr <= hi))

        data = data[:, :, keep_mask]
        wavelengths = wl_arr[keep_mask].tolist()
        return data, wavelengths

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Percentile-based normalization to [0, 1] (robust to outliers)."""
        flat = data.reshape(-1, data.shape[2])
        # Per-band 1st and 99th percentile stretch
        lo = np.percentile(flat, 1, axis=0)   # shape (B,)
        hi = np.percentile(flat, 99, axis=0)

        # Avoid division by zero
        range_ = hi - lo
        range_[range_ == 0] = 1.0

        data = (data - lo) / range_
        data = np.clip(data, 0.0, 1.0)
        return data.astype(np.float32)

    def _smooth_spectra(self, data: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter along the spectral dimension."""
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            logger.warning("  scipy not available, skipping spectral smoothing")
            return data

        window = int(self.cfg.get("smooth_window", 7))
        polyorder = int(self.cfg.get("smooth_polyorder", 2))

        # Ensure window is odd and > polyorder
        if window % 2 == 0:
            window += 1
        if window <= polyorder:
            window = polyorder + 2
            if window % 2 == 0:
                window += 1

        H, W, B = data.shape
        flat = data.reshape(-1, B)  # (H*W, B)
        smoothed = savgol_filter(flat, window_length=window, polyorder=polyorder, axis=1)
        return np.clip(smoothed, 0.0, 1.0).reshape(H, W, B).astype(np.float32)

    # ---------------------------------------------------------- #
    # Utility: band-index lookup
    # ---------------------------------------------------------- #

    @staticmethod
    def find_band(
        wavelengths: Optional[List[float]],
        target_nm: float,
        tolerance_nm: float = 20.0,
    ) -> Optional[int]:
        """
        Return index of the band closest to *target_nm*.
        Returns None if wavelengths are not available or no band is within tolerance.
        """
        if wavelengths is None:
            return None
        wl = np.array(wavelengths)
        diffs = np.abs(wl - target_nm)
        idx = int(np.argmin(diffs))
        if diffs[idx] <= tolerance_nm:
            return idx
        return None

    @staticmethod
    def band_by_fraction(n_bands: int, fraction: float) -> int:
        """Fallback: return band index at *fraction* (0.0–1.0) of the band range."""
        return int(np.clip(fraction * (n_bands - 1), 0, n_bands - 1))
