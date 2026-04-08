# -*- coding: utf-8 -*-
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

hdbscan  : Hierarchical density-based clustering (HDBSCAN).
           No need to specify cluster count; the algorithm determines it
           automatically. Noise pixels (label=-1) are mapped to class 0.

gmm      : Gaussian Mixture Model (soft clustering).
           Probabilistic assignments via sklearn GaussianMixture.
           Uses PCA preprocessing same as kmeans (15 components).

nmf      : Non-negative Matrix Factorization (spectral unmixing).
           Each pixel is assigned to the component with highest activation.
           Data must be non-negative (reflectance values already are).

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

# High-contrast default colour palette (RGB 0-255).
# Used when a class ID has no colour configured in config.yaml.
# First 5 entries match the hybrid method class IDs for intuitive colours.
_CLASS_PALETTE = [
    [40,  40,  40 ],  # 0  Background   – near-black
    [60,  200, 60 ],  # 1  Sunlit Leaf  – bright green
    [20,  100, 20 ],  # 2  Shadow Leaf  – dark green
    [160, 100, 40 ],  # 3  Soil         – brown
    [70,  130, 180],  # 4  Other        – steel blue
    [220, 20,  60 ],  # 5               – crimson
    [255, 165, 0  ],  # 6               – orange
    [138, 43,  226],  # 7               – violet
    [0,   206, 209],  # 8               – dark turquoise
    [255, 105, 180],  # 9               – hot pink
    [255, 215, 0  ],  # 10              – gold
    [30,  144, 255],  # 11              – dodger blue
    [255, 69,  0  ],  # 12              – red-orange
    [144, 238, 144],  # 13              – light green
    [128, 0,   128],  # 14              – purple
    [0,   128, 128],  # 15              – teal
    [255, 0,   0  ],  # 16              – red
    [0,   0,   205],  # 17              – medium blue
    [50,  205, 50 ],  # 18              – lime green
    [255, 255, 0  ],  # 19              – yellow
]
# ------------------------------------------------------------------ #


class HyperspectralClassifier:

    def __init__(self, config: dict):
        self.cfg = config.get("classification", {})
        self.classes = self.cfg.get("classes", [])
        # Populated by supervised/cnn methods; read by Pipeline after classify()
        self.last_val_metrics: dict = {}

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
        elif method == "sam":
            class_map = self._classify_sam(data, wavelengths, labels_csv)
        elif method == "supervised":
            class_map = self._classify_supervised(data, wavelengths, labels_csv)
        elif method == "autoencoder":
            class_map = self._classify_autoencoder(data)
        elif method == "cnn":
            class_map = self._classify_cnn(data, labels_csv)
        elif method == "hdbscan":
            class_map = self._classify_hdbscan(data)
        elif method == "gmm":
            class_map = self._classify_gmm(data)
        elif method == "nmf":
            class_map = self._classify_nmf(data)
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

        n_cls_lbl = len(np.unique(y))
        logger.info(
            f"  Supervised RF: {len(df)} labelled pixels, "
            f"{n_cls_lbl} classes"
        )

        # ── Validation split: 20 % held-out for accuracy estimation ──
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score

        val_acc = val_f1 = None
        n_val = 0
        if len(df) >= 10 and n_cls_lbl >= 2:
            try:
                stratify = y if np.min(np.bincount(y)) >= 2 else None
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train_pca, y, test_size=0.2,
                    random_state=42, stratify=stratify,
                )
                rf_val = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
                rf_val.fit(X_tr, y_tr)
                y_val_pred = rf_val.predict(X_val)
                val_acc = float(accuracy_score(y_val, y_val_pred))
                val_f1  = float(f1_score(y_val, y_val_pred,
                                         average="macro", zero_division=0))
                n_val   = len(y_val)
                logger.info(
                    f"  RF validation  accuracy={val_acc:.3f}  "
                    f"macro-F1={val_f1:.3f}  (n_val={n_val})"
                )
            except Exception as e:
                logger.warning(f"  Validation split failed: {e}")

        self.last_val_metrics = {
            "method":    "supervised",
            "accuracy":  val_acc,
            "macro_f1":  val_f1,
            "n_train":   len(df) - n_val,
            "n_val":     n_val,
        }

        # ── Final model trained on ALL labelled pixels ────────────
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
    # Method 3b – Spectral Angle Mapper (SAM)
    # ============================================================

    def _classify_sam(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]],
        labels_csv: Optional[str],
    ) -> np.ndarray:
        """
        Spectral Angle Mapper (SAM) classification.

        Measures the angle between each pixel spectrum and a set of
        reference (endmember) spectra in n-dimensional feature space.
        Because only the *direction* matters, SAM is invariant to
        per-pixel brightness differences (illumination / shadow effects).

        Modes
        -----
        Supervised  : --labels CSV (row,col,class_id)
                      Mean spectrum of each class label = endmember.
        Unsupervised: No labels needed.
                      K-means cluster centres = endmembers.

        Assignment
        ----------
        Each pixel is assigned to the class whose endmember forms the
        smallest spectral angle.  Pixels with angle > angle_threshold
        are marked as background (class 0 = unclassified).
        """
        cfg       = self.cfg.get("sam", {})
        H, W, B   = data.shape
        threshold = float(cfg.get("angle_threshold", 0.10))   # radians

        # ---- 1. Obtain reference endmembers ----
        if labels_csv is not None:
            import pandas as pd
            df        = pd.read_csv(labels_csv, names=["row", "col", "class_id"])
            class_ids = sorted(df["class_id"].unique().tolist())
            endmembers = []
            for cid in class_ids:
                sub  = df[df["class_id"] == cid]
                rows = sub["row"].astype(int).values
                cols = sub["col"].astype(int).values
                px   = data[rows, cols, :]          # (N_label, B)
                endmembers.append(px.mean(axis=0))  # mean spectrum
            endmembers = np.stack(endmembers)       # (n_classes, B)
            logger.info(
                f"  SAM supervised: {len(class_ids)} endmembers "
                f"(class IDs: {class_ids})"
            )
        else:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA

            n_em  = int(cfg.get("n_endmembers", 6))
            n_pca = int(cfg.get("endmember_pca", 15))
            flat  = data.reshape(-1, B).astype(np.float32)

            # Optional PCA before K-means
            if n_pca > 0:
                n_pca = min(n_pca, B, flat.shape[0] - 1)
                pca   = PCA(n_components=n_pca, random_state=42)
                flat_r = pca.fit_transform(flat)
            else:
                flat_r = flat

            km   = KMeans(n_clusters=n_em, n_init=5, random_state=42)
            km.fit(flat_r)

            # Recover endmembers in original spectral space:
            # use the mean spectrum of each K-means cluster
            labels_km = km.labels_
            endmembers = np.stack([
                data.reshape(-1, B)[labels_km == k].mean(axis=0)
                for k in range(n_em)
            ])
            class_ids = list(range(1, n_em + 1))
            logger.info(
                f"  SAM unsupervised: {n_em} K-means endmembers "
                f"(pca_components={n_pca})"
            )

        # ---- 2. Compute spectral angles ----
        flat = data.reshape(-1, B).astype(np.float64)   # (N, B)
        em   = endmembers.astype(np.float64)             # (C, B)

        # L2-normalise both sets of vectors
        flat_n = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9)
        em_n   = em   / (np.linalg.norm(em,   axis=1, keepdims=True) + 1e-9)

        # Dot-product → cosine → angle   shape: (N, C)
        cos_sim = flat_n @ em_n.T
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        angles  = np.arccos(cos_sim)

        # ---- 3. Assign class ----
        best_idx   = np.argmin(angles, axis=1)                         # (N,)
        best_angle = angles[np.arange(len(angles)), best_idx]          # (N,)

        class_ids_arr = np.array(class_ids, dtype=np.int32)
        pred          = class_ids_arr[best_idx]

        # Reject pixels beyond threshold
        if threshold > 0:
            rejected = best_angle > threshold
            pred[rejected] = 0
            logger.info(
                f"  SAM angle_threshold={threshold:.3f} rad "
                f"({np.degrees(threshold):.1f} deg) -> "
                f"{rejected.sum():,} px unclassified "
                f"({100 * rejected.mean():.1f}%)"
            )

        class_map = pred.reshape(H, W)
        return class_map

    # ============================================================
    # Method 4 – Autoencoder + K-means (unsupervised deep)
    # ============================================================

    def _classify_autoencoder(self, data: np.ndarray) -> np.ndarray:
        from .deep_classifier import DeepClassifier
        cfg       = self.cfg.get("autoencoder", {})
        n_clusters = int(cfg.get("n_clusters", 8))
        dc        = DeepClassifier({"classification": self.cfg})
        return dc.classify_autoencoder(data, n_clusters).astype(np.int32)

    # ============================================================
    # Method 5 – 1D-CNN (supervised deep)
    # ============================================================

    def _classify_cnn(
        self,
        data: np.ndarray,
        labels_csv: Optional[str],
    ) -> np.ndarray:
        from .deep_classifier import DeepClassifier
        dc = DeepClassifier({"classification": self.cfg})
        return dc.classify_cnn(
            data,
            n_classes=self.cfg.get("cnn", {}).get("n_classes", 4),
            labels_csv=labels_csv,
        ).astype(np.int32)

    # ============================================================
    # Method 6 – HDBSCAN (hierarchical density-based clustering)
    # ============================================================

    def _classify_hdbscan(self, data: np.ndarray) -> np.ndarray:
        """
        HDBSCAN clustering on PCA-reduced spectra.

        The number of clusters is determined automatically by the algorithm.
        Noise pixels (HDBSCAN label = -1) are reassigned to class 0
        (Background).  All detected clusters are 1-indexed.
        """
        from sklearn.decomposition import PCA

        cfg = self.cfg.get("hdbscan", {})
        H, W, B = data.shape
        min_cluster_size = int(cfg.get("min_cluster_size", 50))
        min_samples      = int(cfg.get("min_samples", 5))
        n_pca            = int(cfg.get("pca_components", 15))

        flat = data.reshape(-1, B).astype(np.float64)
        n_pca = min(n_pca, B, flat.shape[0] - 1)
        logger.info(f"  HDBSCAN: PCA {B}→{n_pca} components")
        pca = PCA(n_components=n_pca, random_state=42)
        flat_pca = pca.fit_transform(flat)
        explained = pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA explained variance: {100*explained:.1f}%")

        # Try sklearn >= 1.3 first, then fall back to hdbscan package
        import warnings
        try:
            from sklearn.cluster import HDBSCAN as _HDBSCAN
            hdb = _HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                copy=True,  # suppress FutureWarning in sklearn>=1.10
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = hdb.fit_predict(flat_pca)
        except ImportError:
            try:
                import hdbscan as _hdbscan_pkg
                hdb = _hdbscan_pkg.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    labels = hdb.fit_predict(flat_pca)
            except ImportError:
                raise ImportError(
                    "HDBSCAN requires scikit-learn >= 1.3  or  the 'hdbscan' package.\n"
                    "Install with: pip install hdbscan"
                )

        n_noise = int((labels == -1).sum())
        n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
        logger.info(
            f"  HDBSCAN: {n_clusters} clusters found, "
            f"{n_noise:,} noise pixels → class 0"
        )

        # Shift so cluster 0 → 1, noise (-1) → 0
        class_labels = np.where(labels == -1, 0, labels + 1)
        return class_labels.reshape(H, W)

    # ============================================================
    # Method 7 – GMM (Gaussian Mixture Model)
    # ============================================================

    def _classify_gmm(self, data: np.ndarray) -> np.ndarray:
        """
        Gaussian Mixture Model clustering on PCA-reduced spectra.

        Uses the same PCA preprocessing as K-Means (15 components by
        default).  Each pixel is hard-assigned to its most probable
        component.  Components are 1-indexed.
        """
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture

        cfg = self.cfg.get("gmm", {})
        H, W, B = data.shape
        n_components    = int(cfg.get("n_components", self.cfg.get("kmeans", {}).get("n_clusters", 6)))
        covariance_type = str(cfg.get("covariance_type", "full"))
        max_iter        = int(cfg.get("max_iter", 100))
        random_state    = int(cfg.get("random_state", 42))
        n_pca           = int(cfg.get("pca_components", 15))

        flat = data.reshape(-1, B).astype(np.float64)
        n_pca = min(n_pca, B, flat.shape[0] - 1)
        logger.info(f"  GMM: PCA {B}→{n_pca} components")
        pca = PCA(n_components=n_pca, random_state=random_state)
        flat_pca = pca.fit_transform(flat)
        explained = pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA explained variance: {100*explained:.1f}%")

        logger.info(
            f"  GMM: {n_components} components, "
            f"covariance_type='{covariance_type}', max_iter={max_iter}"
        )
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
        )
        labels = gmm.fit_predict(flat_pca)
        logger.info(f"  GMM converged: {gmm.converged_}")
        class_map = labels.reshape(H, W) + 1  # 1-indexed
        return class_map

    # ============================================================
    # Method 8 – NMF (Non-negative Matrix Factorization)
    # ============================================================

    def _classify_nmf(self, data: np.ndarray) -> np.ndarray:
        """
        NMF-based spectral unmixing.

        Each pixel is assigned to the component (endmember) with the
        highest activation (abundance coefficient).  Components are
        1-indexed.  Data must be non-negative — reflectance values
        after normalization are already in [0, 1].
        """
        import warnings
        from sklearn.decomposition import NMF

        cfg = self.cfg.get("nmf", {})
        H, W, B = data.shape
        n_components = int(cfg.get("n_components", self.cfg.get("kmeans", {}).get("n_clusters", 6)))
        max_iter     = int(cfg.get("max_iter", 500))
        random_state = int(cfg.get("random_state", 42))

        flat = data.reshape(-1, int(B)).astype(np.float64)

        # NMF requires non-negative input; clip for safety
        flat = np.clip(flat, 0.0, None)

        logger.info(f"  NMF: {n_components} components, max_iter={max_iter}")
        nmf = NMF(
            n_components=n_components,
            max_iter=max_iter,
            random_state=random_state,
        )
        # W_mat: (N_pixels, n_components) — activation / abundance matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress ConvergenceWarning
            W_mat = nmf.fit_transform(flat)

        logger.info(f"  NMF reconstruction error: {nmf.reconstruction_err_:.4f}")

        # Hard assignment: pixel -> component with highest abundance
        labels = np.argmax(W_mat, axis=1)
        class_map = labels.reshape(int(H), int(W)) + 1  # 1-indexed
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
                f"(ids {next_id}-{next_id+k-1})"
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
        for rank, cid in enumerate(unique_ids):
            n_px = int((class_map == cid).sum())
            cfg_entry = id_lookup.get(cid, {})
            # Auto-assign a distinct palette colour when none is configured.
            # Use cid-based index first (preserves hybrid method semantics),
            # then fall back to rank order so colours never collide.
            default_color = _CLASS_PALETTE[cid % len(_CLASS_PALETTE)]
            info.append({
                "id":       cid,
                "name":     cfg_entry.get("name", f"Cluster {cid}"),
                "color":    cfg_entry.get("color", default_color),
                "n_pixels": n_px,
                "fraction": n_px / total,
            })

        return info
