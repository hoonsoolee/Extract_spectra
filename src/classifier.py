"""
classifier.py
-------------
Hyperspectral pixel classification.

Methods
-------
hybrid   : Spectral-index rules (NDVI + brightness) → K-means refinement.
           Best when no training labels are available.
           Interpretable and fast to tune.

kmeans   : Pure unsupervised K-means on PCA-reduced spectra.
           Good for exploratory analysis; user must label clusters afterwards.

supervised : Random Forest / SVM trained on user-supplied labels (CSV).
             Requires a labelled-pixels CSV with columns:
               row, col, class_id
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Class-ID constants used by hybrid method
BACKGROUND    = 0
SUNLIT_LEAF   = 1
SHADOW_LEAF   = 2
SOIL          = 3
OTHER         = 4
# ------------------------------------------------------------------ #


class HyperspectralClassifier:

    def __init__(self, config: dict):
        self.cfg = config.get("classification", {})
        self.classes = self.cfg.get("classes", [])

    # ============================================================
    # Public entry point
    # ============================================================

    def classify(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]] = None,
        labels_csv: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Parameters
        ----------
        data         : (H, W, B) preprocessed reflectance cube [0, 1]
        wavelengths  : list of wavelengths in nm
        labels_csv   : path to labelled-pixels CSV (supervised only)

        Returns
        -------
        class_map    : (H, W) int32 array, 0 = background
        class_info   : list of dicts {id, name, color, n_pixels, fraction}
        """
        method = self.cfg.get("method", "hybrid")
        logger.info(f"  Classification method: {method}")

        if method == "hybrid":
            class_map = self._classify_hybrid(data, wavelengths)
        elif method == "kmeans":
            class_map = self._classify_kmeans(data)
        elif method == "supervised":
            class_map = self._classify_supervised(data, wavelengths, labels_csv)
        else:
            raise ValueError(f"Unknown classification method: {method}")

        class_info = self._compute_class_info(class_map)
        return class_map.astype(np.int32), class_info

    # ============================================================
    # Method 1 – Hybrid (spectral indices + K-means refinement)
    # ============================================================

    def _classify_hybrid(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]],
    ) -> np.ndarray:
        H, W, B = data.shape
        cfg = self.cfg.get("hybrid", {})
        class_map = np.zeros((H, W), dtype=np.int32)

        # --- 1. Compute spectral indices ---
        brightness = self._brightness(data)
        ndvi = self._ndvi(data, wavelengths, cfg)
        shadow_mask = self._shadow_mask(data, wavelengths, brightness, cfg)

        # --- 2. Rule-based segmentation ---
        if ndvi is not None:
            veg_thresh = float(cfg.get("ndvi_threshold", 0.15))
            veg_mask = ndvi > veg_thresh
            logger.info(
                f"  NDVI veg mask: {veg_mask.sum():,} pixels "
                f"({100 * veg_mask.mean():.1f}%)"
            )
        else:
            # Fallback: use upper-NIR brightness as rough vegetation proxy
            logger.warning("  NDVI unavailable – using NIR-brightness as vegetation proxy")
            nir_idx = self._find_band_fraction(B, 0.6)  # ~70% of spectrum range
            nir = data[:, :, nir_idx]
            veg_mask = nir > np.percentile(nir, 40)

        # Initial segments
        class_map[veg_mask & ~shadow_mask]  = SUNLIT_LEAF   # 1
        class_map[veg_mask & shadow_mask]   = SHADOW_LEAF   # 2
        class_map[~veg_mask & ~shadow_mask] = SOIL          # 3
        class_map[~veg_mask & shadow_mask]  = BACKGROUND    # 0

        logger.info(
            f"  Initial segments – "
            f"sunlit: {(class_map==SUNLIT_LEAF).sum():,}, "
            f"shadow: {(class_map==SHADOW_LEAF).sum():,}, "
            f"soil: {(class_map==SOIL).sum():,}, "
            f"bg: {(class_map==BACKGROUND).sum():,}"
        )

        # --- 3. Optional K-means refinement within each segment ---
        if cfg.get("kmeans_refinement", True):
            class_map = self._refine_with_kmeans(data, class_map, cfg)

        return class_map

    # ============================================================
    # Method 2 – Pure K-means
    # ============================================================

    def _classify_kmeans(self, data: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        cfg = self.cfg.get("kmeans", {})
        H, W, B = data.shape
        n_clusters  = int(cfg.get("n_clusters", 6))
        n_pca       = int(cfg.get("pca_components", 15))
        n_init      = int(cfg.get("n_init", 10))
        max_iter    = int(cfg.get("max_iter", 300))
        random_state = int(cfg.get("random_state", 42))

        flat = data.reshape(-1, B).astype(np.float64)

        # PCA
        n_pca = min(n_pca, B, flat.shape[0] - 1)
        logger.info(f"  K-means: PCA {B}→{n_pca} components")
        pca = PCA(n_components=n_pca, random_state=random_state)
        flat_pca = pca.fit_transform(flat)
        explained = pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA explained variance: {100*explained:.1f}%")

        # K-means
        logger.info(f"  K-means: {n_clusters} clusters")
        km = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        labels = km.fit_predict(flat_pca)
        class_map = labels.reshape(H, W) + 1  # 1-indexed
        return class_map

    # ============================================================
    # Method 3 – Supervised (Random Forest)
    # ============================================================

    def _classify_supervised(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]],
        labels_csv: Optional[str],
    ) -> np.ndarray:
        """
        Train Random Forest on labelled pixels, then predict all pixels.

        labels_csv format (no header):  row,col,class_id
        """
        if labels_csv is None:
            raise ValueError(
                "supervised method requires --labels-csv with columns: row,col,class_id"
            )
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.decomposition import PCA

        H, W, B = data.shape
        df = pd.read_csv(labels_csv, names=["row", "col", "class_id"])
        rows = df["row"].astype(int).values
        cols = df["col"].astype(int).values
        y    = df["class_id"].astype(int).values

        # Feature matrix for labelled pixels
        X_train = data[rows, cols, :]          # (N_labels, B)

        # Optional PCA
        n_pca = min(30, B, len(df) - 1)
        pca = PCA(n_components=n_pca, random_state=42)
        X_train_pca = pca.fit_transform(X_train)

        logger.info(
            f"  Supervised RF: {len(df)} labelled pixels, "
            f"{len(np.unique(y))} classes"
        )
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train_pca, y)

        # Predict all pixels
        flat = data.reshape(-1, B)
        flat_pca = pca.transform(flat)
        pred = rf.predict(flat_pca)
        class_map = pred.reshape(H, W)

        logger.info("  Supervised classification done")
        return class_map

    # ============================================================
    # K-means refinement helper
    # ============================================================

    def _refine_with_kmeans(
        self,
        data: np.ndarray,
        class_map: np.ndarray,
        cfg: dict,
    ) -> np.ndarray:
        """Run K-means within each segment to find sub-classes."""
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        n_pca = int(cfg.get("pca_components", 10))
        H, W, B = data.shape
        refined = class_map.copy()

        cluster_cfg = {
            SUNLIT_LEAF:  ("n_clusters_sunlit",  3),
            SHADOW_LEAF:  ("n_clusters_shadow",   2),
            SOIL:         ("n_clusters_soil",     3),
        }

        next_id = int(class_map.max()) + 1

        for seg_id, (key, default_k) in cluster_cfg.items():
            mask = class_map == seg_id
            n_px = mask.sum()
            k = int(cfg.get(key, default_k))
            if n_px < k * 10:
                logger.warning(
                    f"  Segment {seg_id}: only {n_px} pixels, "
                    f"skipping K-means refinement"
                )
                continue

            pixels = data[mask]                        # (N, B)
            n_pca_use = min(n_pca, B, n_px - 1)

            if n_pca_use >= 2:
                pca = PCA(n_components=n_pca_use, random_state=42)
                pixels_red = pca.fit_transform(pixels)
            else:
                pixels_red = pixels

            km = KMeans(n_clusters=k, n_init=5, random_state=42)
            sub_labels = km.fit_predict(pixels_red)

            # Assign new IDs
            new_ids = next_id + sub_labels
            refined[mask] = new_ids
            logger.info(
                f"  Segment {seg_id}: refined into {k} sub-clusters "
                f"(ids {next_id}–{next_id+k-1})"
            )
            next_id += k

        return refined

    # ============================================================
    # Spectral indices
    # ============================================================

    def _brightness(self, data: np.ndarray) -> np.ndarray:
        """Mean reflectance across all bands → (H, W)."""
        return np.mean(data, axis=2)

    def _ndvi(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]],
        cfg: dict,
    ) -> Optional[np.ndarray]:
        """NDVI = (NIR - Red) / (NIR + Red)."""
        red_nm = float(cfg.get("ndvi_red_nm", 670))
        nir_nm = float(cfg.get("ndvi_nir_nm", 800))
        B = data.shape[2]

        if wavelengths is not None:
            red_idx = self._find_band(wavelengths, red_nm)
            nir_idx = self._find_band(wavelengths, nir_nm)
        else:
            # Fallback: approximate positions for visible-NIR sensor
            red_idx = self._find_band_fraction(B, 0.50)  # ~middle of range
            nir_idx = self._find_band_fraction(B, 0.75)

        if red_idx is None or nir_idx is None:
            return None

        red = data[:, :, red_idx].astype(np.float64)
        nir = data[:, :, nir_idx].astype(np.float64)
        ndvi = (nir - red) / (nir + red + 1e-9)
        return ndvi.astype(np.float32)

    def _shadow_mask(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]],
        brightness: np.ndarray,
        cfg: dict,
    ) -> np.ndarray:
        """Return boolean mask: True = shadow pixels."""
        method = cfg.get("shadow_method", "brightness")
        thresh = float(cfg.get("brightness_threshold", 0.08))

        if method == "brightness":
            mask = brightness < thresh
        elif method == "ratio" and wavelengths is not None:
            # Shadow ratio: (NIR - Blue) / (NIR + Blue)
            blue_nm = 480
            nir_nm  = 800
            b_idx   = self._find_band(wavelengths, blue_nm)
            n_idx   = self._find_band(wavelengths, nir_nm)
            if b_idx is not None and n_idx is not None:
                blue = data[:, :, b_idx].astype(np.float64)
                nir  = data[:, :, n_idx].astype(np.float64)
                ratio = (nir - blue) / (nir + blue + 1e-9)
                ratio_thresh = float(cfg.get("shadow_ratio_threshold", 0.05))
                mask = ratio < ratio_thresh
            else:
                logger.warning("  Shadow ratio: bands not found, falling back to brightness")
                mask = brightness < thresh
        else:
            mask = brightness < thresh

        logger.info(
            f"  Shadow mask ({method}): {mask.sum():,} pixels "
            f"({100 * mask.mean():.1f}%)"
        )
        return mask

    # ============================================================
    # Utilities
    # ============================================================

    @staticmethod
    def _find_band(wavelengths: List[float], target_nm: float, tol: float = 25.0) -> Optional[int]:
        wl = np.array(wavelengths)
        diff = np.abs(wl - target_nm)
        idx = int(np.argmin(diff))
        return idx if diff[idx] <= tol else None

    @staticmethod
    def _find_band_fraction(n_bands: int, fraction: float) -> int:
        return int(np.clip(fraction * (n_bands - 1), 0, n_bands - 1))

    def _compute_class_info(
        self,
        class_map: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Return per-class statistics."""
        total = class_map.size
        unique_ids = sorted(np.unique(class_map).tolist())

        # Build id→name/color lookup from config
        id_lookup = {c["id"]: c for c in self.classes}

        info = []
        for cid in unique_ids:
            n_px = int((class_map == cid).sum())
            cfg_entry = id_lookup.get(cid, {})
            info.append({
                "id":       cid,
                "name":     cfg_entry.get("name", f"Cluster {cid}"),
                "color":    cfg_entry.get("color", [128, 128, 128]),
                "n_pixels": n_px,
                "fraction": n_px / total,
            })

        return info
