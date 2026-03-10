"""
evaluator.py
------------
Quality metrics for hyperspectral classification.

방법별 정확도 표현
------------------
지도학습 (supervised / cnn)
  - Validation Accuracy / F1 Score
  - 클래스별 Precision, Recall, F1
  - Confusion Matrix

비지도학습 (kmeans / hybrid / autoencoder)
  - Silhouette Score  : 클러스터 분리도 (-1~1, 높을수록 좋음)
  - Davies-Bouldin    : 클러스터 내 분산 대비 클러스터 간 거리 (낮을수록 좋음)
  - 스펙트럼 분리도   : 클래스 평균 스펙트럼 간 유클리드 거리 행렬

공통
  - 클래스별 픽셀 분포 균일도 (CV)
  - 훈련 Loss 곡선 (AE / CNN)
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class Evaluator:

    # ---------------------------------------------------------- #
    # Unsupervised quality metrics
    # ---------------------------------------------------------- #

    @staticmethod
    def unsupervised_metrics(
        data: np.ndarray,
        class_map: np.ndarray,
        max_sample: int = 30_000,
    ) -> Dict[str, Any]:
        """
        Silhouette Score, Davies-Bouldin Index on a pixel sample.

        Returns
        -------
        {
          "silhouette":      float,   # -1~1, higher = better separation
          "davies_bouldin":  float,   # lower = better
          "n_classes":       int,
          "interpretation":  str,     # human-readable
        }
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        H, W, B   = data.shape
        flat      = data.reshape(-1, B).astype(np.float32)
        labels    = class_map.flatten()
        unique    = np.unique(labels)

        if len(unique) < 2:
            return {"silhouette": None, "davies_bouldin": None,
                    "n_classes": 1, "interpretation": "단일 클래스 - 평가 불가"}

        # Subsample for speed
        rng = np.random.default_rng(42)
        if len(flat) > max_sample:
            idx   = rng.choice(len(flat), size=max_sample, replace=False)
            flat  = flat[idx]
            labels = labels[idx]

        try:
            sil = float(silhouette_score(flat, labels, metric="euclidean",
                                         sample_size=min(10_000, len(flat))))
        except Exception:
            sil = None

        try:
            db  = float(davies_bouldin_score(flat, labels))
        except Exception:
            db  = None

        # Interpretation
        if sil is None:
            interp = "계산 실패"
        elif sil >= 0.7:
            interp = "매우 좋음 (클러스터가 뚜렷하게 분리됨)"
        elif sil >= 0.5:
            interp = "좋음"
        elif sil >= 0.25:
            interp = "보통 (클러스터 경계 다소 모호)"
        else:
            interp = "미흡 (클러스터 수 조정 또는 다른 방법 고려)"

        return {
            "silhouette":     sil,
            "davies_bouldin": db,
            "n_classes":      len(unique),
            "interpretation": interp,
        }

    # ---------------------------------------------------------- #
    # Vegetation (leaf) separation quality
    # ---------------------------------------------------------- #

    @staticmethod
    def _identify_leaf_classes(
        spectra: List[Dict[str, Any]],
        wavelengths: Optional[List[float]] = None,
    ) -> List[int]:
        """
        Auto-detect leaf/vegetation class IDs.

        Priority:
          1. Name contains a leaf keyword  (e.g. "leaf", "sunlit", "잎")
          2. Mean spectrum has NDVI > 0.15  (NIR ≫ Red → vegetation shape)
        """
        leaf_keywords = {
            "leaf", "sunlit", "shadow", "잎", "식물", "vegetation",
            "veg", "plant", "foliage", "canopy", "crop",
        }

        ids: List[int] = []

        # ── pass 1: name-based ──────────────────────────────────
        for s in spectra:
            if any(kw in s["name"].lower() for kw in leaf_keywords):
                ids.append(s["id"])

        if ids:
            return ids

        # ── pass 2: spectral NDVI of mean spectrum ──────────────
        wl = np.array(wavelengths) if wavelengths else None
        for s in spectra:
            mean = np.asarray(s["mean"], dtype=np.float32)
            B    = len(mean)
            if B < 4:
                continue
            if wl is not None:
                red_idx = int(np.argmin(np.abs(wl - 670)))
                nir_idx = int(np.argmin(np.abs(wl - 800)))
            else:
                red_idx = max(0, int(B * 0.25))
                nir_idx = min(B - 1, int(B * 0.65))
            red   = float(mean[red_idx])
            nir   = float(mean[nir_idx])
            denom = nir + red
            if denom > 1e-6 and (nir - red) / denom > 0.15:
                ids.append(s["id"])

        return ids

    @staticmethod
    def vegetation_separation_metrics(
        data: np.ndarray,
        class_map: np.ndarray,
        spectra: List[Dict[str, Any]],
        wavelengths: Optional[List[float]] = None,
        ndvi_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        """
        식생(잎) 클래스가 다른 클래스로부터 얼마나 잘 분리되어 있는지 평가.

        두 가지 관점:
          A) NDVI 기반 식생 검출 정확도
             - GT: NDVI > ndvi_threshold 픽셀 = "실제 식생"
             - Pred: 잎 클래스로 분류된 픽셀
             → Recall(검출률), Precision(정밀도), F1

          B) 잎 ↔ 비잎 클래스 간 스펙트럼 거리
             - 잎 클래스 평균 스펙트럼과 비잎 클래스 평균 스펙트럼 간 Euclidean 거리

        Returns
        -------
        {
          "leaf_ids"          : list[int],
          "leaf_names"        : list[str],
          "ndvi_recall"       : float or None,
          "ndvi_precision"    : float or None,
          "ndvi_f1"           : float or None,
          "ndvi_gt_pixels"    : int,
          "leaf_pred_pixels"  : int,
          "separation_bars"   : list[{label, leaf, other, distance}],
          "min_separation"    : float or None,
          "mean_separation"   : float or None,
          "note"              : str or None,
        }
        """
        H, W, B = data.shape

        # ── 1. Identify leaf classes ────────────────────────────
        leaf_ids = Evaluator._identify_leaf_classes(spectra, wavelengths)
        if not leaf_ids:
            return {
                "leaf_ids": [], "leaf_names": [],
                "ndvi_recall": None, "ndvi_precision": None, "ndvi_f1": None,
                "ndvi_gt_pixels": 0, "leaf_pred_pixels": 0,
                "separation_bars": [], "min_separation": None,
                "mean_separation": None,
                "note": (
                    "잎 클래스를 자동으로 감지하지 못했습니다.  "
                    "클래스 이름에 'leaf'/'잎' 등을 포함하거나 "
                    "NIR 밴드(~800 nm)가 있으면 자동 감지됩니다."
                ),
            }

        leaf_spec_map  = {s["id"]: s for s in spectra if s["id"] in leaf_ids}
        other_spectra  = [s for s in spectra if s["id"] not in leaf_ids]
        leaf_names     = [leaf_spec_map[i]["name"] for i in leaf_ids
                          if i in leaf_spec_map]

        # ── 2. NDVI-based recall / precision / F1 ───────────────
        ndvi_recall = ndvi_precision = ndvi_f1 = None
        ndvi_gt_px = leaf_pred_px = 0

        wl = np.array(wavelengths) if wavelengths else None
        if wl is not None:
            red_idx = int(np.argmin(np.abs(wl - 670)))
            nir_idx = int(np.argmin(np.abs(wl - 800)))
        else:
            red_idx = max(0, int(B * 0.25))
            nir_idx = min(B - 1, int(B * 0.65))

        try:
            red    = data[:, :, red_idx].astype(np.float32)
            nir    = data[:, :, nir_idx].astype(np.float32)
            denom  = nir + red
            ndvi   = np.where(denom > 1e-6, (nir - red) / denom, 0.0)

            gt_veg    = ndvi > ndvi_threshold
            pred_leaf = np.isin(class_map, leaf_ids)

            ndvi_gt_px   = int(gt_veg.sum())
            leaf_pred_px = int(pred_leaf.sum())

            if ndvi_gt_px > 0 and leaf_pred_px > 0:
                tp = int((gt_veg & pred_leaf).sum())
                ndvi_recall    = tp / ndvi_gt_px
                ndvi_precision = tp / leaf_pred_px
                denom_f1 = ndvi_recall + ndvi_precision
                ndvi_f1  = (2 * ndvi_recall * ndvi_precision / denom_f1
                            if denom_f1 > 0 else 0.0)
        except Exception as e:
            logger.warning(f"  Vegetation NDVI metrics failed: {e}")

        # ── 3. Leaf ↔ other class spectral distances ─────────────
        bars: list = []
        for lid in leaf_ids:
            ls = leaf_spec_map.get(lid)
            if ls is None:
                continue
            lm = np.asarray(ls["mean"])
            for os_ in other_spectra:
                dist = float(np.linalg.norm(lm - np.asarray(os_["mean"])))
                bars.append({
                    "label":    f"{ls['name']} ↔ {os_['name']}",
                    "leaf":     ls["name"],
                    "other":    os_["name"],
                    "distance": round(dist, 4),
                })

        bars.sort(key=lambda x: x["distance"])
        all_dists = [b["distance"] for b in bars]

        return {
            "leaf_ids":         leaf_ids,
            "leaf_names":       leaf_names,
            "ndvi_recall":      ndvi_recall,
            "ndvi_precision":   ndvi_precision,
            "ndvi_f1":          ndvi_f1,
            "ndvi_gt_pixels":   ndvi_gt_px,
            "leaf_pred_pixels": leaf_pred_px,
            "separation_bars":  bars,
            "min_separation":   float(min(all_dists)) if all_dists else None,
            "mean_separation":  float(np.mean(all_dists)) if all_dists else None,
            "note":             None,
        }

    # ---------------------------------------------------------- #
    # Spectral separability matrix
    # ---------------------------------------------------------- #

    @staticmethod
    def spectral_separability(
        spectra: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute pairwise Euclidean distance between class mean spectra.
        Higher distance = more spectrally distinct classes.

        Returns
        -------
        {
          "names":  list[str],
          "matrix": np.ndarray (N, N),   # symmetric distance matrix
        }
        """
        if len(spectra) < 2:
            return {"names": [], "matrix": np.array([])}

        names = [s["name"] for s in spectra]
        means = np.stack([s["mean"] for s in spectra])   # (N_classes, B)

        N = len(means)
        mat = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                mat[i, j] = float(np.linalg.norm(means[i] - means[j]))

        return {"names": names, "matrix": mat}

    # ---------------------------------------------------------- #
    # Supervised accuracy metrics (CNN / RF)
    # ---------------------------------------------------------- #

    @staticmethod
    def supervised_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Accuracy, per-class F1, confusion matrix.

        Returns
        -------
        {
          "accuracy":       float,
          "macro_f1":       float,
          "per_class":      list[{name, precision, recall, f1, support}],
          "confusion_matrix": np.ndarray,
          "class_names":    list[str],
        }
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score,
            recall_score, confusion_matrix,
        )

        classes   = sorted(np.unique(np.concatenate([y_true, y_pred])))
        n         = len(classes)
        names     = class_names or [str(c) for c in classes]

        acc     = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0))
        prec    = precision_score(y_true, y_pred, average=None,
                                   labels=classes, zero_division=0)
        rec     = recall_score(y_true, y_pred, average=None,
                                labels=classes, zero_division=0)
        f1s     = f1_score(y_true, y_pred, average=None,
                            labels=classes, zero_division=0)
        cm      = confusion_matrix(y_true, y_pred, labels=classes)
        support = cm.sum(axis=1)

        per_class = [
            {
                "name":      names[i] if i < len(names) else str(classes[i]),
                "precision": float(prec[i]),
                "recall":    float(rec[i]),
                "f1":        float(f1s[i]),
                "support":   int(support[i]),
            }
            for i in range(n)
        ]

        return {
            "accuracy":        acc,
            "macro_f1":        macro_f1,
            "per_class":       per_class,
            "confusion_matrix": cm,
            "class_names":     [p["name"] for p in per_class],
        }
