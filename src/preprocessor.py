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
        self.last_calibration_info = None
        self.last_effective_normalize_mode = str(
            self.cfg.get("normalize_mode", "global")
        ).lower()

    # ---------------------------------------------------------- #
    # Public
    # ---------------------------------------------------------- #

    def process(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]] = None,
        skip_downsample: bool = False,
        source_path: Optional[str] = None,
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

        # Optional empirical-line / flat-field calibration saved by the panel
        # correction tab. Apply before band removal and normalization so the
        # whole-image app and ROI app use the same radiometric order.
        calibration_file = self.cfg.get("calibration_file")
        calibration_selection = "explicit" if calibration_file else ""
        rejected_candidates: list[dict] = []
        if (
            not calibration_file
            and source_path
            and self.cfg.get("auto_discover_calibration", True)
            and not str(source_path).startswith("github:")
        ):
            from .radiometry import (
                discover_calibration_candidates,
                resolve_calibration,
            )

            candidates = discover_calibration_candidates(
                source_path,
                search_roots=self.cfg.get("calibration_search_roots", []),
            )
            for candidate in candidates:
                try:
                    # Validate sensor/band compatibility before accepting the
                    # first conservatively named candidate.
                    resolve_calibration(
                        str(candidate),
                        target_source=source_path,
                        wavelengths=wavelengths,
                    )
                    calibration_file = str(candidate)
                    calibration_selection = "auto_discovered"
                    logger.info(f"  Auto-discovered calibration: {candidate}")
                    break
                except Exception as exc:
                    rejected_candidates.append(
                        {"path": str(candidate), "reason": str(exc)}
                    )
                    logger.warning(
                        f"  Ignored incompatible calibration {candidate}: {exc}"
                    )
        self.last_calibration_info = None
        calibration_a = calibration_b = None
        if calibration_file:
            from .radiometry import resolve_calibration

            calibration = resolve_calibration(
                calibration_file,
                target_source=source_path,
                wavelengths=wavelengths,
            )
            a = np.asarray(calibration["a"], dtype=np.float32)
            b = np.asarray(calibration["b"], dtype=np.float32)
            if a.shape != (B,) or b.shape != (B,):
                raise ValueError(
                    f"Calibration bands ({len(a)}, {len(b)}) do not match data ({B})"
                )
            data = np.asarray(data, dtype=np.float32) * a + b
            valid_calibration_bands = np.isfinite(a) & np.isfinite(b)
            if not valid_calibration_bands.any():
                raise ValueError("Calibration contains no valid White-Dark bands")
            if not valid_calibration_bands.all():
                removed = int((~valid_calibration_bands).sum())
                data = data[:, :, valid_calibration_bands]
                a = a[valid_calibration_bands]
                b = b[valid_calibration_bands]
                if wavelengths is not None:
                    wavelengths = np.asarray(wavelengths)[valid_calibration_bands].tolist()
                logger.warning(
                    f"  Removed {removed} invalid White-Dark calibration band(s)"
                )
            self.last_calibration_info = {
                "selected_profile": calibration.get("selected_profile"),
                "calibration_type": calibration.get("calibration_type"),
                "meta": calibration.get("meta", {}),
                "selection_source": calibration_selection,
                "rejected_candidates": rejected_candidates,
            }
            calibration_a, calibration_b = a, b
            logger.info(f"  Reflectance calibration applied: {calibration_file}")

        # 2. Remove bad bands
        if self.cfg.get("remove_bad_bands", True) and wavelengths is not None:
            keep_mask = self._bad_band_keep_mask(wavelengths)
            data = data[:, :, keep_mask]
            wavelengths = np.asarray(wavelengths)[keep_mask].tolist()
            if calibration_a is not None:
                calibration_a = np.asarray(calibration_a)[keep_mask]
                calibration_b = np.asarray(calibration_b)[keep_mask]
            logger.info(f"  After bad-band removal: {data.shape[2]} bands")

        # 3. Normalize to [0, 1]
        requested_mode = str(self.cfg.get("normalize_mode", "global")).lower()
        self.last_effective_normalize_mode = requested_mode
        if self.cfg.get("normalize", True):
            # Calibration already puts the cube on a physical reflectance
            # scale. A later scene normalization would destroy that scale and
            # make the exported values no longer publication-ready.
            mode = "none" if self.last_calibration_info else requested_mode
            self.last_effective_normalize_mode = mode
            data = self._normalize(data, mode)
            if mode == "none" and self.last_calibration_info and requested_mode != "none":
                logger.info(
                    "  Scene normalization skipped to preserve calibrated reflectance"
                )
            else:
                logger.info(f"  Normalized to [0, 1] (mode={mode})")

        if self.last_calibration_info is not None:
            self.last_calibration_info.update(
                {
                    "requested_normalization": requested_mode,
                    "effective_normalization": self.last_effective_normalize_mode,
                    "value_units": "reflectance",
                    "wavelengths": (
                        list(wavelengths) if wavelengths is not None else None
                    ),
                    "a": np.asarray(calibration_a, dtype=float).tolist(),
                    "b": np.asarray(calibration_b, dtype=float).tolist(),
                }
            )

        # 4. Spectral smoothing
        if self.cfg.get("smooth_spectra", False):
            data = self._smooth_spectra(data)
            logger.info("  Spectral smoothing applied")

        # 5. Spatial downsampling (skipped if the loader already applied it)
        factor = 1 if skip_downsample else int(self.cfg.get("spatial_downsample", 1))
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
        keep_mask = self._bad_band_keep_mask(wavelengths)

        data = data[:, :, keep_mask]
        wavelengths = wl_arr[keep_mask].tolist()
        return data, wavelengths

    def _bad_band_keep_mask(self, wavelengths: List[float]) -> np.ndarray:
        """Return the same bad-band mask for data and calibration coefficients."""
        wl_arr = np.asarray(wavelengths, dtype=float)
        keep_mask = np.ones(len(wl_arr), dtype=bool)
        for lo, hi in self.cfg.get("bad_band_ranges", []):
            keep_mask &= ~((wl_arr >= lo) & (wl_arr <= hi))
        return keep_mask

    def _normalize(self, data: np.ndarray, mode: str = "global") -> np.ndarray:
        """
        Scale values into [0, 1].

        mode="global"   : one scalar divisor for the whole cube, with no offset
                          subtracted. Every band is multiplied by the same
                          number, so the *shape* of each pixel spectrum — and
                          any band ratio such as NDVI — is preserved exactly
                          (an offset would not preserve ratios, which is why
                          none is applied here). This is the correct choice
                          when the extracted spectra are the scientific product.
        mode="per_band" : independent 1–99 percentile stretch per band. Maximises
                          per-band contrast for display/clustering but rescales
                          every band by a different gain, which changes spectral
                          shape and band ratios. Not comparable to reference
                          spectra.
        mode="none"     : leave values untouched (raw DN).
        """
        if mode == "none":
            return data.astype(np.float32)

        # Sample at most ~2M pixels for percentile estimation; a full-cube
        # percentile on a multi-GB array is slow and needs a large temp copy.
        flat = data.reshape(-1, data.shape[2])
        if flat.shape[0] > 2_000_000:
            step = flat.shape[0] // 2_000_000 + 1
            sample = flat[::step]
        else:
            sample = flat

        if mode == "per_band":
            lo = np.percentile(sample, 1, axis=0)   # shape (B,)
            hi = np.percentile(sample, 99, axis=0)
            range_ = hi - lo
            range_[range_ == 0] = 1.0
            data = (data - lo) / range_
        else:  # "global" — pure scaling, no offset, so ratios survive
            hi = float(np.percentile(sample, 99)) or 1.0
            data = data / hi

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
