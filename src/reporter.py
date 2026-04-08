"""
reporter.py
-----------
Generates a self-contained HTML report for one or multiple processed images.
Includes:
  - RGB composite & False Color Infrared (CIR) images
  - Classification map (all classes combined)
  - Per-class individual overlay images
  - Interactive spectral plots (Plotly)
  - Per-class statistics table

Language support: pass lang="ko" (default) or lang="en" to Reporter().
"""

import base64
import io
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ===================================================================
# Translation dictionaries
# ===================================================================

_T_KO: Dict[str, str] = {
    "html_lang":          "ko",
    "generated_at":       "생성 시각",
    "files_processed":    "처리 파일 수",
    "toc":                "목차",
    "resolution":         "해상도",
    "bands":              "밴드 수",
    "total_pixels":       "총 픽셀",
    "n_classes":          "클래스 수",
    "format":             "형식",
    "image_overview":     "📷 이미지 개요",
    "rgb_composite":      "RGB 합성",
    "cir_falsecolor":     "CIR 위색도 (NIR-R-G)",
    "class_map":          "통합 분류 맵",
    "per_class_images":   "🗂️ 클래스별 분류 이미지",
    "per_class_caption":  "각 클래스 픽셀을 해당 색상으로 강조, 나머지는 그레이스케일로 표시",
    "class_summary":      "📊 분류 요약",
    "spectra_chart":      "📈 클래스별 반사율 스펙트럼",
    "quality_title":      "&#128202; 품질 평가 (Quality Assessment)",
    "veg_sep_title":      "&#127807; 식생 분리도 평가 (Vegetation Separation)",
    "col_class":          "클래스",
    "col_id":             "ID",
    "col_pixels":         "픽셀 수",
    "col_ratio":          "비율",
    "sep_caption":        "▶ 클래스 평균 스펙트럼 간 유클리드 거리 — 값이 클수록 클래스가 잘 구별됩니다.",
    "install_plotly":     "pip install plotly 후 표시됩니다.",
    "no_quality":         "품질 지표를 계산할 수 없습니다.",
    "val_accuracy":       "검증 정확도 (Validation Accuracy)",
    "train_val_px":       "학습: {n_tr}px  /  검증: {n_val}px",
    "train_pixels":       "학습 픽셀 수",
    "val_pixels":         "검증 픽셀 수",
    "val_accuracy_lbl":   "검증 정확도",
    "cluster_quality":    "클러스터 품질 점수",
    "silhouette_note":    "Silhouette 기반 (−1→0 % ~ +1→100 %)",
    "n_classes_lbl":      "클래스 수",
    "no_veg_data":        "식생 분리도 데이터가 없습니다.",
    "detected_leaf":      "감지된 잎 클래스",
    "none_detected":      "감지 못 함",
    "ndvi_gt_note":       "▶ NDVI &gt; 0.15 픽셀을 식생 기준(Ground Truth)으로 사용",
    "recall_lbl":         "검출률 (Recall)",
    "recall_sub":         "GT 식생 {gt_px:,}px 중 검출",
    "precision_lbl":      "정밀도 (Precision)",
    "precision_sub":      "예측 잎 {leaf_px:,}px 중 정확",
    "f1_lbl":             "F1 점수",
    "f1_sub":             "Recall × Precision 조화평균",
    "no_ndvi":            "NIR/Red 밴드 정보가 없어 NDVI 정확도를 계산하지 못했습니다.",
    "sep_dist_caption":   "▶ 잎 클래스 ↔ 비잎 클래스 간 평균 스펙트럼 거리 — 값이 클수록 잘 분리됩니다.",
    "sep_dist_mean":      "평균 {mean:.4f}",
    "sep_dist_xaxis":     "스펙트럼 유클리드 거리",
    "min_sep":            "최소 분리 거리 (가장 가까운 쌍)",
    "mean_sep":           "평균 분리 거리",
    "heatmap_caption":    "pip install plotly 후 분리도 히트맵이 표시됩니다.",
    "dist_label":         "거리",
    "sep_matrix_title":   "스펙트럼 분리도 행렬 (클래스 평균 스펙트럼 간 유클리드 거리)",
    "axis_class":         "클래스",
    "no_chart":           "pip install plotly 후 차트가 표시됩니다.",
    "wavelength_nm":      "파장 (nm)",
    "band_index":         "밴드 인덱스",
    "reflectance":        "반사율",
    "processing_time":    "처리 시간",
    # ── 품질 해석 문구 ──
    "interp_title":         "📋 결과 해석",
    "quality_what":         "이 점수는 클래스들이 스펙트럼 공간에서 얼마나 잘 분리됐는지를 나타냅니다. 높을수록 각 클래스의 픽셀들이 서로 비슷하고 다른 클래스와는 확실히 다르다는 의미입니다.",
    "sil_what":             "Silhouette Score란?",
    "sil_explain":          "각 픽셀이 자기 클래스에 얼마나 잘 속하는지를 -1~+1로 표현합니다. +1에 가까울수록 완벽한 분류, 0은 경계선에 위치, -1은 잘못 분류됐다는 의미입니다.",
    "db_what":              "Davies-Bouldin Index란?",
    "db_explain":           "클러스터 내 분산 대비 클러스터 간 거리의 비율입니다. 낮을수록 좋습니다. 1.0 이하이면 양호, 2.0 이상이면 클래스 간 겹침이 심한 편입니다.",
    "quality_action_good":  "✅ 분류 결과를 신뢰할 수 있습니다. 클래스별 이미지와 스펙트럼 차트를 확인하여 각 클래스의 의미를 해석하세요.",
    "quality_action_fair":  "⚠️ 일부 클래스 간 겹침이 있습니다. 클래스 수를 조정하거나 다른 분류 방법을 시도해보세요.",
    "quality_action_poor":  "❌ 클래스 분리가 충분하지 않습니다. 파라미터를 조정하거나 다른 방법을 사용해보세요.",
    # ── 식생 분리도 해석 문구 ──
    "veg_interp_title":     "📋 결과 해석",
    "veg_what":             "NDVI &gt; 0.15인 픽셀을 '실제 식생'으로 보고, 알고리즘이 잎 클래스로 분류한 픽셀과 얼마나 일치하는지를 검증합니다.",
    "recall_what":          "Recall(검출률)이 낮으면?",
    "recall_action":        "잎 픽셀이 토양·배경 클래스에 섞여 있습니다. NDVI 임계값을 낮추거나(예: 0.15→0.10) 클래스 수를 늘려보세요.",
    "precision_what":       "Precision(정밀도)이 낮으면?",
    "precision_action":     "비식생 픽셀(토양, 경로 등)이 잎 클래스로 잘못 분류됩니다. NDVI 임계값을 높이거나(예: 0.15→0.20) 밝기 임계값을 조정해보세요.",
    "f1_what":              "F1 점수 해석",
    "f1_action_good":       "✅ 식생 분리가 잘 됐습니다. 잎 클래스를 분석에 바로 활용할 수 있습니다.",
    "f1_action_fair":       "⚠️ 식생 분리가 보통 수준입니다. 위의 Recall/Precision을 확인하고 파라미터를 조정하세요.",
    "f1_action_poor":       "❌ 식생 분리가 좋지 않습니다. Hybrid 방법의 임계값을 조정하거나 다른 방법을 시도해보세요.",
}

_T_EN: Dict[str, str] = {
    "html_lang":          "en",
    "generated_at":       "Generated",
    "files_processed":    "Files processed",
    "toc":                "Table of Contents",
    "resolution":         "Resolution",
    "bands":              "Bands",
    "total_pixels":       "Total Pixels",
    "n_classes":          "Classes",
    "format":             "Format",
    "image_overview":     "📷 Image Overview",
    "rgb_composite":      "RGB Composite",
    "cir_falsecolor":     "CIR False Color (NIR-R-G)",
    "class_map":          "Classification Map",
    "per_class_images":   "🗂️ Per-Class Classification Images",
    "per_class_caption":  "Each class highlighted in its assigned colour; remaining pixels shown in greyscale",
    "class_summary":      "📊 Classification Summary",
    "spectra_chart":      "📈 Reflectance Spectra by Class",
    "quality_title":      "&#128202; Quality Assessment",
    "veg_sep_title":      "&#127807; Vegetation Separation Assessment",
    "col_class":          "Class",
    "col_id":             "ID",
    "col_pixels":         "Pixels",
    "col_ratio":          "Ratio",
    "sep_caption":        "▶ Euclidean distances between class mean spectra — larger values indicate better class separation.",
    "install_plotly":     "Install plotly to display this section.",
    "no_quality":         "Quality metrics unavailable.",
    "val_accuracy":       "Validation Accuracy",
    "train_val_px":       "Train: {n_tr}px  /  Val: {n_val}px",
    "train_pixels":       "Training Pixels",
    "val_pixels":         "Validation Pixels",
    "val_accuracy_lbl":   "Validation Accuracy",
    "cluster_quality":    "Cluster Quality Score",
    "silhouette_note":    "Silhouette-based (−1→0 % ~ +1→100 %)",
    "n_classes_lbl":      "Classes",
    "no_veg_data":        "No vegetation separation data available.",
    "detected_leaf":      "Detected Leaf Classes",
    "none_detected":      "None detected",
    "ndvi_gt_note":       "▶ NDVI &gt; 0.15 pixels used as vegetation ground truth",
    "recall_lbl":         "Recall",
    "recall_sub":         "{gt_px:,} GT vegetation px detected",
    "precision_lbl":      "Precision",
    "precision_sub":      "{leaf_px:,} predicted leaf px correct",
    "f1_lbl":             "F1 Score",
    "f1_sub":             "Harmonic mean of Recall &amp; Precision",
    "no_ndvi":            "NIR/Red band information unavailable — NDVI accuracy could not be computed.",
    "sep_dist_caption":   "▶ Mean spectral distance between leaf ↔ non-leaf classes — larger values indicate better separation.",
    "sep_dist_mean":      "Mean {mean:.4f}",
    "sep_dist_xaxis":     "Spectral Euclidean Distance",
    "min_sep":            "Min Separation Distance (closest pair)",
    "mean_sep":           "Mean Separation Distance",
    "heatmap_caption":    "Install plotly to display separability heatmap.",
    "dist_label":         "Distance",
    "sep_matrix_title":   "Spectral Separability Matrix (Euclidean distance between class mean spectra)",
    "axis_class":         "Class",
    "no_chart":           "Install plotly to display chart.",
    "wavelength_nm":      "Wavelength (nm)",
    "band_index":         "Band Index",
    "reflectance":        "Reflectance",
    "processing_time":    "Processing Time",
    # ── Quality interpretation strings ──
    "interp_title":         "📋 Interpretation",
    "quality_what":         "This score reflects how well the classes are separated in spectral space. Higher values mean pixels within each class are spectrally similar to each other and distinct from other classes.",
    "sil_what":             "What is Silhouette Score?",
    "sil_explain":          "Measures how well each pixel fits its assigned class on a scale of -1 to +1. Near +1 = well classified, 0 = on the boundary between classes, -1 = likely misclassified.",
    "db_what":              "What is Davies-Bouldin Index?",
    "db_explain":           "Ratio of within-cluster scatter to between-cluster distance. Lower is better. Below 1.0 is good; above 2.0 indicates significant class overlap.",
    "quality_action_good":  "✅ The classification result is reliable. Check the per-class images and spectral chart to interpret the meaning of each class.",
    "quality_action_fair":  "⚠️ Some class overlap detected. Try adjusting the number of classes or switching to a different method.",
    "quality_action_poor":  "❌ Class separation is insufficient. Adjust parameters or try a different classification method.",
    # ── Vegetation separation interpretation strings ──
    "veg_interp_title":     "📋 Interpretation",
    "veg_what":             "Pixels with NDVI &gt; 0.15 are treated as ground-truth vegetation. These metrics measure how closely the algorithm's leaf classes match that reference.",
    "recall_what":          "Low Recall?",
    "recall_action":        "Leaf pixels are being placed in soil or background classes. Try lowering the NDVI threshold (e.g. 0.15 → 0.10) or increasing the number of classes.",
    "precision_what":       "Low Precision?",
    "precision_action":     "Non-vegetation pixels (soil, paths, etc.) are being classified as leaves. Try raising the NDVI threshold (e.g. 0.15 → 0.20) or adjusting the brightness threshold.",
    "f1_what":              "F1 Score",
    "f1_action_good":       "✅ Vegetation separation is good. The leaf classes can be used directly for spectral analysis.",
    "f1_action_fair":       "⚠️ Vegetation separation is moderate. Check Recall and Precision above and adjust parameters.",
    "f1_action_poor":       "❌ Vegetation separation is poor. Adjust the Hybrid thresholds or try a different method.",
}


# ===================================================================
# HTML TEMPLATE
# ===================================================================

_HTML_HEAD = """<!DOCTYPE html>
<html lang="{html_lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #222; }}
  .container {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
  .header {{ background: linear-gradient(135deg, #1a3a5c, #2e7d62); color: #fff;
             padding: 28px 32px; border-radius: 12px; margin-bottom: 24px; }}
  .header h1 {{ font-size: 26px; font-weight: 700; }}
  .header p  {{ margin-top: 6px; opacity: .85; font-size: 13px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 24px; margin-bottom: 20px;
           box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  .card h2 {{ font-size: 18px; color: #1a3a5c; border-bottom: 2px solid #2e7d62;
              padding-bottom: 8px; margin-bottom: 16px; }}
  /* Overview images: max 3 columns */
  .img-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
               gap: 16px; }}
  /* Per-class images: larger grid for easier visual inspection */
  .class-img-grid {{ display: grid;
                     grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
                     gap: 18px; }}
  .img-card {{ text-align: center; }}
  .img-card img {{ width: 100%; border-radius: 6px; border: 1px solid #ddd; }}
  .img-card .img-title {{
    margin-top: 6px; font-size: 13px; font-weight: 600; color: #333; }}
  .img-card .img-sub {{
    font-size: 11px; color: #777; margin-top: 2px; }}
  .class-card {{
    background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px;
    padding: 12px; text-align: center; transition: box-shadow .2s; }}
  .class-card:hover {{ box-shadow: 0 4px 14px rgba(0,0,0,.14); }}
  .class-card img {{ width: 100%; border-radius: 4px; }}
  .class-card .cls-name {{
    margin-top: 10px; font-size: 14px; font-weight: 700; }}
  .class-card .cls-stats {{
    font-size: 12px; color: #666; margin-top: 4px; }}
  .color-dot {{ display: inline-block; width: 10px; height: 10px;
                border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #1a3a5c; color: #fff; padding: 10px 12px; text-align: left; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f7faf9; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
            color: #fff; font-size: 11px; font-weight: 600; }}
  .stat-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
               gap: 12px; margin-bottom: 16px; }}
  .stat-box {{ background: #f7faf9; border: 1px solid #dde; border-radius: 8px;
               padding: 14px; text-align: center; }}
  .stat-val {{ font-size: 22px; font-weight: 700; color: #1a3a5c; }}
  .stat-lbl {{ font-size: 11px; color: #888; margin-top: 4px; }}
  .file-block {{ border-left: 4px solid #2e7d62; padding-left: 16px; margin-bottom: 40px; }}
  .file-block > h2 {{ color: #2e7d62; font-size: 17px; margin-bottom: 12px; }}
  .toc {{ background: #f7faf9; border: 1px solid #dde; border-radius: 8px;
          padding: 14px 20px; margin-bottom: 24px; }}
  .toc h3 {{ margin-bottom: 8px; color: #444; font-size: 14px; }}
  .toc ul {{ list-style: none; padding: 0; }}
  .toc li {{ padding: 2px 0; }}
  .toc a  {{ color: #2e7d62; text-decoration: none; font-size: 13px; }}
  .toc a:hover {{ text-decoration: underline; }}
  footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
"""

_HTML_FOOT = """
<footer>
  Generated by HyperspectralPipeline &nbsp;|&nbsp; {timestamp}
</footer>
</div></body></html>
"""


# ===================================================================
# Reporter class
# ===================================================================

class Reporter:
    """Accumulate per-file results and render a final HTML report."""

    def __init__(self, config: dict, lang: str = "ko"):
        self.cfg    = config
        self.rcfg   = config.get("report", {})
        self.lang   = lang
        self._labels: Dict[str, str] = _T_EN if lang == "en" else _T_KO
        self.results: List[Dict[str, Any]] = []

    def _t(self, key: str, **kwargs) -> str:
        """Return translated string for *key*, optionally formatted with kwargs."""
        s = self._labels.get(key, key)
        return s.format(**kwargs) if kwargs else s

    # ---------------------------------------------------------- #
    # Public
    # ---------------------------------------------------------- #

    def add_result(
        self,
        filename: str,
        data: np.ndarray,
        class_map: np.ndarray,
        class_info: List[Dict[str, Any]],
        spectra: List[Dict[str, Any]],
        wavelengths: Optional[List[float]],
        metadata: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        separability: Optional[Dict[str, Any]] = None,
        veg_sep: Optional[Dict[str, Any]] = None,
        elapsed_sec: Optional[float] = None,
    ) -> None:
        self.results.append(dict(
            filename=filename, data=data, class_map=class_map,
            class_info=class_info, spectra=spectra,
            wavelengths=wavelengths, metadata=metadata,
            metrics=metrics, separability=separability,
            veg_sep=veg_sep, elapsed_sec=elapsed_sec,
        ))

    def render(self, output_path: str | Path) -> None:
        """Write the complete HTML report to *output_path*."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        title     = self.rcfg.get("title", "Hyperspectral Analysis Report")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_parts = [
            _HTML_HEAD.format(title=title, html_lang=self._t("html_lang")),
            f'<div class="header"><h1>{title}</h1>'
            f'<p>{self._t("generated_at")}: {timestamp} &nbsp;|&nbsp; '
            f'{self._t("files_processed")}: {len(self.results)}</p></div>',
        ]

        # Table of contents
        toc_items = "\n".join(
            f'<li><a href="#file-{i}">{r["filename"]}</a></li>'
            for i, r in enumerate(self.results)
        )
        html_parts.append(
            f'<div class="toc"><h3>{self._t("toc")}</h3><ul>{toc_items}</ul></div>'
        )

        for i, result in enumerate(self.results):
            html_parts.append(self._render_file_block(i, result))

        html_parts.append(_HTML_FOOT.format(timestamp=timestamp))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))
        logger.info(f"Report saved: {output_path}")

    # ---------------------------------------------------------- #
    # Per-file block
    # ---------------------------------------------------------- #

    def _render_file_block(self, idx: int, result: Dict[str, Any]) -> str:
        fname      = result["filename"]
        data       = result["data"]
        class_map  = result["class_map"]
        class_info = result["class_info"]
        spectra    = result["spectra"]
        wl         = result["wavelengths"]
        meta       = result["metadata"]

        H, W, B = data.shape
        n_px    = H * W

        # Elapsed time formatting
        elapsed_sec = result.get("elapsed_sec")
        if elapsed_sec is not None:
            m, s = divmod(int(elapsed_sec), 60)
            elapsed_str = f"{m}m {s:02d}s" if m else f"{s}s"
        else:
            elapsed_str = "—"

        # Pre-compute RGB array once (reused for per-class overlays)
        rgb_arr = self._get_rgb_array(data, wl, mode="rgb")

        parts = [
            f'<div class="file-block" id="file-{idx}">',
            f'<h2>📁 {fname}</h2>',

            # ── Summary stats ─────────────────────────────────────
            '<div class="stat-row">',
            self._stat_box(self._t("resolution"),      f"{W} × {H}"),
            self._stat_box(self._t("bands"),            str(B)),
            self._stat_box(self._t("total_pixels"),     f"{n_px:,}"),
            self._stat_box(self._t("n_classes"),        str(len(class_info))),
            self._stat_box(self._t("format"),           meta.get("format", "—")),
            self._stat_box(self._t("processing_time"),  elapsed_str),
            '</div>',

            # ── Overview images (RGB / CIR / Classification map) ──
            '<div class="card">',
            f'<h2>{self._t("image_overview")}</h2>',
            '<div class="img-grid">',
            self._img_card(self._t("rgb_composite"),
                           self._array_to_b64(rgb_arr)),
            self._img_card(self._t("cir_falsecolor"),
                           self._array_to_b64(self._get_rgb_array(data, wl, "cir"))),
            self._img_card(self._t("class_map"),
                           self._make_class_map_img(class_map, class_info)),
            '</div></div>',

            # ── Per-class images ───────────────────────────────────
            '<div class="card">',
            f'<h2>{self._t("per_class_images")}</h2>',
            f'<p style="font-size:12px;color:#888;margin-bottom:14px;">'
            f'{self._t("per_class_caption")}</p>',
            self._render_per_class_images(data, class_map, class_info, rgb_arr, n_px),
            '</div>',

            # ── Classification summary table ───────────────────────
            '<div class="card">',
            f'<h2>{self._t("class_summary")}</h2>',
            self._class_table(class_info, n_px),
            '</div>',

            # ── Spectral chart ─────────────────────────────────────
            '<div class="card">',
            f'<h2>{self._t("spectra_chart")}</h2>',
            self._spectra_plot_html(spectra, wl),
            '</div>',

            # ── Quality assessment ─────────────────────────────────
            '<div class="card">',
            f'<h2>{self._t("quality_title")}</h2>',
            self._render_quality_section(result.get("metrics"), result.get("separability")),
            '</div>',

            # ── Vegetation separation ──────────────────────────────
            '<div class="card">',
            f'<h2>{self._t("veg_sep_title")}</h2>',
            self._render_veg_separation_card(result.get("veg_sep")),
            '</div>',

            '</div>',  # file-block
        ]
        return "\n".join(parts)

    # ---------------------------------------------------------- #
    # Per-class overlay images
    # ---------------------------------------------------------- #

    def _render_per_class_images(
        self,
        data: np.ndarray,
        class_map: np.ndarray,
        class_info: List[Dict[str, Any]],
        rgb_arr: np.ndarray,
        total_px: int,
    ) -> str:
        """Build a grid of per-class highlight images (larger for visual inspection)."""
        cards = []
        for cinfo in sorted(class_info, key=lambda x: -x["n_pixels"]):
            b64   = self._make_single_class_img(class_map, cinfo, rgb_arr)
            r, g, b_ = cinfo["color"]
            pct   = 100.0 * cinfo["n_pixels"] / total_px
            dot   = f'<span class="color-dot" style="background:rgb({r},{g},{b_})"></span>'
            cards.append(
                f'<div class="class-card">'
                f'<img src="data:image/png;base64,{b64}" alt="{cinfo["name"]}">'
                f'<div class="cls-name">{dot}{cinfo["name"]}</div>'
                f'<div class="cls-stats">'
                f'ID {cinfo["id"]} &nbsp;|&nbsp; '
                f'{cinfo["n_pixels"]:,} px &nbsp;|&nbsp; {pct:.1f}%'
                f'</div></div>'
            )
        return f'<div class="class-img-grid">{"".join(cards)}</div>'

    def _make_single_class_img(
        self,
        class_map: np.ndarray,
        cinfo: Dict[str, Any],
        rgb_arr: np.ndarray,
    ) -> str:
        """
        Highlight one class in its class color on a darkened grayscale background.
        Rendered at high resolution (720 px) for clear visual inspection.
        """
        # Grayscale background (darkened to 30%)
        gray = np.mean(rgb_arr, axis=2, keepdims=True)
        bg   = np.repeat(gray * 0.30, 3, axis=2)

        # Overlay class pixels in class color
        result = bg.copy()
        mask   = class_map == cinfo["id"]
        r, g, b_ = [c / 255.0 for c in cinfo["color"]]
        result[mask, 0] = r
        result[mask, 1] = g
        result[mask, 2] = b_

        return self._array_to_b64(np.clip(result, 0, 1))

    # ---------------------------------------------------------- #
    # RGB / class-map image helpers
    # ---------------------------------------------------------- #

    def _get_rgb_array(
        self,
        data: np.ndarray,
        wavelengths: Optional[List[float]],
        mode: str = "rgb",
    ) -> np.ndarray:
        """Return a normalised (H, W, 3) float32 array for RGB or CIR composite."""
        rcfg = self.rcfg
        B    = data.shape[2]

        if mode == "rgb":
            targets = [
                rcfg.get("rgb_red_nm",   660),
                rcfg.get("rgb_green_nm", 550),
                rcfg.get("rgb_blue_nm",  450),
            ]
        else:   # CIR: NIR / Red / Green
            targets = [800, 660, 550]

        channels = []
        for t in targets:
            if wavelengths:
                wl  = np.array(wavelengths)
                idx = int(np.argmin(np.abs(wl - t)))
            else:
                frac = (t - 400) / 600.0
                idx  = int(np.clip(frac * (B - 1), 0, B - 1))
            channels.append(data[:, :, idx])

        rgb = np.stack(channels, axis=2).astype(np.float32)
        lo, hi = rcfg.get("rgb_percentile_stretch", [2, 98])
        for c in range(3):
            p_lo = np.percentile(rgb[:, :, c], lo)
            p_hi = np.percentile(rgb[:, :, c], hi)
            if p_hi > p_lo:
                rgb[:, :, c] = (rgb[:, :, c] - p_lo) / (p_hi - p_lo)
        return np.clip(rgb, 0, 1)

    def _make_class_map_img(
        self,
        class_map: np.ndarray,
        class_info: List[Dict[str, Any]],
    ) -> str:
        """Full classification map with all classes coloured."""
        H, W = class_map.shape
        rgb  = np.zeros((H, W, 3), dtype=np.uint8)
        for cinfo in class_info:
            rgb[class_map == cinfo["id"]] = cinfo["color"]
        return self._array_to_b64(rgb.astype(np.float32) / 255.0)

    @staticmethod
    def _array_to_b64(arr: np.ndarray) -> str:
        """
        Convert (H, W, 3) float [0,1] to base64-encoded PNG.
        Uses figsize=6×6 at dpi=120 → 720×720 px for crisp display at large sizes.
        """
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        ax.imshow(arr)
        ax.axis("off")
        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=120)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    @staticmethod
    def _img_card(title: str, b64: str) -> str:
        return (
            f'<div class="img-card">'
            f'<img src="data:image/png;base64,{b64}" alt="{title}">'
            f'<p class="img-title">{title}</p></div>'
        )

    # ---------------------------------------------------------- #
    # Classification summary table
    # ---------------------------------------------------------- #

    def _class_table(
        self,
        class_info: List[Dict[str, Any]],
        total_px: int,
    ) -> str:
        rows = ""
        for c in sorted(class_info, key=lambda x: -x["n_pixels"]):
            r, g, b_ = c["color"]
            badge = (
                f'<span class="badge" style="background:rgb({r},{g},{b_})">'
                f'{c["name"]}</span>'
            )
            pct = 100.0 * c["n_pixels"] / total_px
            bar_w = max(1, int(pct))
            bar = (
                f'<div style="background:rgb({r},{g},{b_});height:8px;'
                f'width:{bar_w}%;border-radius:4px;display:inline-block;'
                f'vertical-align:middle;margin-left:6px;"></div>'
            )
            rows += (
                f'<tr><td>{badge}</td>'
                f'<td>{c["id"]}</td>'
                f'<td>{c["n_pixels"]:,}</td>'
                f'<td>{pct:.2f}% {bar}</td></tr>'
            )
        return (
            "<table>"
            "<thead><tr>"
            f"<th>{self._t('col_class')}</th>"
            f"<th>{self._t('col_id')}</th>"
            f"<th>{self._t('col_pixels')}</th>"
            f"<th>{self._t('col_ratio')}</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    # ---------------------------------------------------------- #
    # Quality assessment section
    # ---------------------------------------------------------- #

    def _render_quality_section(
        self,
        metrics: Optional[Dict[str, Any]],
        sep: Optional[Dict[str, Any]],
    ) -> str:
        """
        Render quality assessment:
          - Supervised  → Validation Accuracy gauge + Macro-F1 stat box
          - Unsupervised→ Cluster Quality Score gauge (silhouette → 0-100 %)
          Both include the spectral separability heatmap.
        """
        try:
            import plotly.graph_objects as go  # noqa: F401
            has_plotly = True
        except ImportError:
            has_plotly = False

        parts = []

        if metrics:
            is_supervised = (
                metrics.get("method") in ("supervised", "cnn")
                and metrics.get("accuracy") is not None
            )
            if is_supervised:
                parts += self._render_supervised_quality(metrics, has_plotly)
            else:
                parts += self._render_unsupervised_quality(metrics, has_plotly)

        # ── Spectral separability heatmap (always shown) ──────────
        if sep and len(sep.get("names", [])) >= 2:
            parts.append(
                f'<p style="font-size:12px;color:#666;margin:16px 0 8px 0;">'
                f'{self._t("sep_caption")}</p>'
            )
            parts.append(
                self._separability_heatmap_html(sep) if has_plotly
                else f'<p><em>{self._t("install_plotly")}</em></p>'
            )

        if not parts:
            parts.append(f'<p style="color:#999;">{self._t("no_quality")}</p>')

        return "\n".join(parts)

    # ---- Supervised quality (Accuracy / F1) ----------------------

    def _render_supervised_quality(
        self,
        metrics: Dict[str, Any],
        has_plotly: bool,
    ) -> list:
        import plotly.graph_objects as go

        acc    = metrics["accuracy"]
        f1     = metrics.get("macro_f1")
        n_tr   = metrics.get("n_train", "?")
        n_val  = metrics.get("n_val",   "?")
        acc_pct = acc * 100

        if acc_pct >= 90:   gauge_color = "#27ae60"
        elif acc_pct >= 75: gauge_color = "#2980b9"
        elif acc_pct >= 60: gauge_color = "#f39c12"
        else:               gauge_color = "#e74c3c"

        parts = []

        if has_plotly:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(acc_pct, 1),
                number={"suffix": "%", "font": {"size": 44, "color": gauge_color}},
                title={"text": (
                    f"{self._t('val_accuracy')}"
                    "<br><span style='font-size:0.75em;color:#888;'>"
                    f"{self._t('train_val_px', n_tr=n_tr, n_val=n_val)}</span>"
                )},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar":  {"color": gauge_color, "thickness": 0.3},
                    "bgcolor": "#f8f8f8",
                    "borderwidth": 1,
                    "bordercolor": "#ddd",
                    "steps": [
                        {"range": [0,  60], "color": "#ffebee"},
                        {"range": [60, 75], "color": "#fff9c4"},
                        {"range": [75, 90], "color": "#e8f5e9"},
                        {"range": [90, 100],"color": "#c8e6c9"},
                    ],
                    "threshold": {
                        "line": {"color": "#1a7a3a", "width": 3},
                        "thickness": 0.75, "value": 90,
                    },
                },
            ))
            fig.update_layout(
                height=280, margin=dict(l=30, r=30, t=100, b=10),
                paper_bgcolor="white",
            )
            parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
        else:
            parts.append(
                f'<div class="stat-box">'
                f'<div class="stat-val" style="color:{gauge_color}">'
                f'{acc_pct:.1f}%</div>'
                f'<div class="stat-lbl">{self._t("val_accuracy_lbl")}</div></div>'
            )

        f1_text = f"{f1*100:.1f}%" if f1 is not None else "N/A"
        parts += [
            '<div class="stat-row" style="margin-top:12px;">',
            (f'<div class="stat-box"><div class="stat-val">{f1_text}</div>'
             f'<div class="stat-lbl">Macro F1-Score</div></div>'),
            (f'<div class="stat-box"><div class="stat-val">{n_tr}</div>'
             f'<div class="stat-lbl">{self._t("train_pixels")}</div></div>'),
            (f'<div class="stat-box"><div class="stat-val">{n_val}</div>'
             f'<div class="stat-lbl">{self._t("val_pixels")}</div></div>'),
            '</div>',
        ]
        return parts

    # ---- Unsupervised quality (Silhouette → 0-100 %) ─────────────

    def _render_unsupervised_quality(
        self,
        metrics: Dict[str, Any],
        has_plotly: bool,
    ) -> list:
        import plotly.graph_objects as go

        sil    = metrics.get("silhouette")
        db     = metrics.get("davies_bouldin")
        interp = metrics.get("interpretation", "")
        n_cls  = metrics.get("n_classes", "?")

        quality = (float(sil) + 1) / 2 * 100 if sil is not None else None

        if quality is None:
            gauge_color = "#999999"
        elif quality >= 85:   gauge_color = "#27ae60"
        elif quality >= 62.5: gauge_color = "#2980b9"
        elif quality >= 37.5: gauge_color = "#f39c12"
        else:                 gauge_color = "#e74c3c"

        parts = []

        if has_plotly and quality is not None:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(quality, 1),
                number={"suffix": "%", "font": {"size": 44, "color": gauge_color}},
                title={"text": (
                    f"{self._t('cluster_quality')}"
                    "<br><span style='font-size:0.75em;color:#888;'>"
                    f"{self._t('silhouette_note')}</span>"
                )},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar":  {"color": gauge_color, "thickness": 0.3},
                    "bgcolor": "#f8f8f8",
                    "borderwidth": 1,
                    "bordercolor": "#ddd",
                    "steps": [
                        {"range": [0,   37.5], "color": "#ffebee"},
                        {"range": [37.5, 62.5],"color": "#fff9c4"},
                        {"range": [62.5, 85],  "color": "#e8f5e9"},
                        {"range": [85,  100],  "color": "#c8e6c9"},
                    ],
                    "threshold": {
                        "line": {"color": "#1a7a3a", "width": 3},
                        "thickness": 0.75, "value": 85,
                    },
                },
            ))
            fig.update_layout(
                height=280, margin=dict(l=30, r=30, t=100, b=10),
                paper_bgcolor="white",
            )
            parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
        elif sil is not None:
            parts.append(
                f'<div class="stat-box">'
                f'<div class="stat-val" style="color:{gauge_color}">'
                f'{quality:.1f}%</div>'
                f'<div class="stat-lbl">{self._t("cluster_quality")}</div></div>'
            )

        db_text  = f"{db:.3f}"  if db  is not None else "N/A"
        sil_text = f"{sil:.3f}" if sil is not None else "N/A"
        parts += [
            '<div class="stat-row" style="margin-top:12px;">',
            (f'<div class="stat-box"><div class="stat-val">{sil_text}</div>'
             f'<div class="stat-lbl">Silhouette Score (−1~1)</div></div>'),
            (f'<div class="stat-box"><div class="stat-val">{db_text}</div>'
             f'<div class="stat-lbl">Davies-Bouldin Index ↓</div></div>'),
            (f'<div class="stat-box"><div class="stat-val">{n_cls}</div>'
             f'<div class="stat-lbl">{self._t("n_classes_lbl")}</div></div>'),
            '</div>',
        ]
        if interp:
            parts.append(
                f'<p style="margin:8px 0 0 0;">'
                f'<span class="badge" style="background:{gauge_color};'
                f'font-size:13px;padding:5px 14px;">{interp}</span></p>'
            )

        # ── Interpretation box ─────────────────────────────────────
        if quality is not None:
            if quality >= 85:
                action_key = "quality_action_good"
                action_bg  = "#e8f5e9"
                action_bd  = "#a5d6a7"
            elif quality >= 50:
                action_key = "quality_action_fair"
                action_bg  = "#fff9c4"
                action_bd  = "#f9e04b"
            else:
                action_key = "quality_action_poor"
                action_bg  = "#ffebee"
                action_bd  = "#ef9a9a"

            parts.append(f"""
<div style="margin-top:18px;padding:14px 16px;background:#f5f5f5;
            border-radius:8px;border-left:4px solid #2980b9;">
  <div style="font-weight:700;font-size:13px;margin-bottom:8px;color:#1a1a1a;">
    {self._t('interp_title')}
  </div>
  <p style="margin:0 0 10px 0;font-size:12.5px;color:#333;line-height:1.6;">
    {self._t('quality_what')}
  </p>
  <div style="background:{action_bg};border:1px solid {action_bd};
              border-radius:6px;padding:8px 12px;font-size:12.5px;
              color:#333;margin-bottom:10px;">
    {self._t(action_key)}
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:12px;">
    <tr>
      <td style="padding:6px 8px;background:#fff;border:1px solid #e0e0e0;
                 vertical-align:top;width:50%;">
        <strong>{self._t('sil_what')}</strong><br>
        <span style="color:#555;line-height:1.5;">{self._t('sil_explain')}</span>
      </td>
      <td style="padding:6px 8px;background:#fff;border:1px solid #e0e0e0;
                 vertical-align:top;width:50%;">
        <strong>{self._t('db_what')}</strong><br>
        <span style="color:#555;line-height:1.5;">{self._t('db_explain')}</span>
      </td>
    </tr>
  </table>
</div>""")

        return parts

    # ---- Vegetation separation card ──────────────────────────────

    def _render_veg_separation_card(
        self,
        veg_sep: Optional[Dict[str, Any]],
    ) -> str:
        if not veg_sep:
            return f'<p style="color:#999;">{self._t("no_veg_data")}</p>'

        note = veg_sep.get("note")
        if note:
            return f'<p style="color:#888;">{note}</p>'

        try:
            import plotly.graph_objects as go
        except ImportError:
            return f'<p><em>{self._t("install_plotly")}</em></p>'

        parts = []

        leaf_names   = veg_sep.get("leaf_names", [])
        recall       = veg_sep.get("ndvi_recall")
        precision    = veg_sep.get("ndvi_precision")
        f1           = veg_sep.get("ndvi_f1")
        ndvi_gt_px   = veg_sep.get("ndvi_gt_pixels", 0)
        leaf_pred_px = veg_sep.get("leaf_pred_pixels", 0)
        bars         = veg_sep.get("separation_bars", [])
        min_sep      = veg_sep.get("min_separation")
        mean_sep     = veg_sep.get("mean_separation")

        # ── Detected leaf classes ──────────────────────────────────
        leaf_badge = ", ".join(
            f'<span class="badge" style="background:#2e7d62;padding:3px 10px;">'
            f'{n}</span>' for n in leaf_names
        ) or f'<em>{self._t("none_detected")}</em>'
        parts.append(
            f'<p style="margin-bottom:14px;font-size:13px;">'
            f'<strong>{self._t("detected_leaf")}:</strong> {leaf_badge}</p>'
        )

        # ── NDVI-based accuracy (Recall / Precision / F1) ──────────
        if recall is not None:
            def _pct_bar(val: float, label: str, sub: str) -> str:
                pct   = round(val * 100, 1)
                width = max(2, round(pct))
                if pct >= 85:   c = "#27ae60"
                elif pct >= 65: c = "#2980b9"
                elif pct >= 45: c = "#f39c12"
                else:           c = "#e74c3c"
                return (
                    f'<div class="stat-box">'
                    f'<div class="stat-val" style="color:{c};font-size:28px;">'
                    f'{pct}%</div>'
                    f'<div style="background:#eee;border-radius:4px;height:8px;'
                    f'margin:6px 0 4px;">'
                    f'<div style="background:{c};width:{width}%;height:8px;'
                    f'border-radius:4px;"></div></div>'
                    f'<div class="stat-lbl">{label}<br><small>{sub}</small></div>'
                    f'</div>'
                )

            parts += [
                f'<p style="font-size:12px;color:#555;margin-bottom:8px;">'
                f'{self._t("ndvi_gt_note")}</p>',
                '<div class="stat-row">',
                _pct_bar(recall,    self._t("recall_lbl"),
                         self._t("recall_sub",    gt_px=ndvi_gt_px)),
                _pct_bar(precision, self._t("precision_lbl"),
                         self._t("precision_sub", leaf_px=leaf_pred_px)),
                _pct_bar(f1,        self._t("f1_lbl"),
                         self._t("f1_sub")),
                '</div>',
            ]
        else:
            parts.append(
                f'<p style="color:#aaa;font-size:12px;">{self._t("no_ndvi")}</p>'
            )

        # ── Vegetation interpretation box ──────────────────────────
        if recall is not None and f1 is not None:
            if f1 >= 0.85:
                f1_key = "f1_action_good"
                f1_bg  = "#e8f5e9"
                f1_bd  = "#a5d6a7"
            elif f1 >= 0.60:
                f1_key = "f1_action_fair"
                f1_bg  = "#fff9c4"
                f1_bd  = "#f9e04b"
            else:
                f1_key = "f1_action_poor"
                f1_bg  = "#ffebee"
                f1_bd  = "#ef9a9a"

            low_recall    = recall    < 0.65
            low_precision = precision is not None and precision < 0.65

            hint_rows = ""
            if low_recall:
                hint_rows += f"""
      <tr>
        <td style="padding:6px 8px;background:#fff3e0;border:1px solid #ffe0b2;
                   vertical-align:top;width:50%;">
          <strong>{self._t('recall_what')}</strong><br>
          <span style="color:#555;line-height:1.5;">{self._t('recall_action')}</span>
        </td>"""
            if low_precision:
                hint_rows += f"""
        <td style="padding:6px 8px;background:#fff3e0;border:1px solid #ffe0b2;
                   vertical-align:top;width:50%;">
          <strong>{self._t('precision_what')}</strong><br>
          <span style="color:#555;line-height:1.5;">{self._t('precision_action')}</span>
        </td>
      </tr>"""
            elif low_recall:
                hint_rows += "</tr>"

            if not low_recall and not low_precision:
                hint_rows = ""

            table_html = (
                f'<table style="width:100%;border-collapse:collapse;'
                f'font-size:12px;margin-top:8px;">{hint_rows}</table>'
                if hint_rows else ""
            )

            parts.append(f"""
<div style="margin-top:18px;padding:14px 16px;background:#f5f5f5;
            border-radius:8px;border-left:4px solid #2e7d62;">
  <div style="font-weight:700;font-size:13px;margin-bottom:8px;color:#1a1a1a;">
    {self._t('veg_interp_title')}
  </div>
  <p style="margin:0 0 10px 0;font-size:12.5px;color:#333;line-height:1.6;">
    {self._t('veg_what')}
  </p>
  <div style="background:{f1_bg};border:1px solid {f1_bd};
              border-radius:6px;padding:8px 12px;font-size:12.5px;
              color:#333;margin-bottom:6px;">
    <strong>{self._t('f1_what')}:</strong> {self._t(f1_key)}
  </div>
  {table_html}
</div>""")

        # ── Leaf ↔ non-leaf spectral distance bar chart ────────────
        if bars:
            parts.append(
                f'<p style="font-size:12px;color:#555;margin:18px 0 6px;">'
                f'{self._t("sep_dist_caption")}</p>'
            )

            labels    = [b["label"]    for b in bars]
            distances = [b["distance"] for b in bars]
            mean_d    = float(np.mean(distances)) if distances else 0
            colors    = ["#27ae60" if d >= mean_d else "#f39c12" for d in distances]

            fig_bar = go.Figure(go.Bar(
                x=distances, y=labels, orientation="h",
                marker_color=colors,
                text=[f"{d:.4f}" for d in distances],
                textposition="outside",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_bar.add_vline(
                x=mean_d, line_dash="dot", line_color="#888",
                annotation_text=self._t("sep_dist_mean", mean=mean_d),
                annotation_position="top right",
                annotation_font_size=11,
            )
            fig_bar.update_layout(
                height=max(200, 40 * len(bars) + 80),
                margin=dict(l=10, r=80, t=20, b=30),
                xaxis_title=self._t("sep_dist_xaxis"),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="#fafafa",
                paper_bgcolor="#ffffff",
                showlegend=False,
            )
            parts.append(fig_bar.to_html(full_html=False, include_plotlyjs=False))

            parts += [
                '<div class="stat-row" style="margin-top:10px;">',
                (f'<div class="stat-box"><div class="stat-val">'
                 f'{min_sep:.4f}</div>'
                 f'<div class="stat-lbl">{self._t("min_sep")}</div></div>'),
                (f'<div class="stat-box"><div class="stat-val">'
                 f'{mean_sep:.4f}</div>'
                 f'<div class="stat-lbl">{self._t("mean_sep")}</div></div>'),
                '</div>',
            ]

        return "\n".join(parts)

    def _separability_heatmap_html(
        self,
        sep: Dict[str, Any],
    ) -> str:
        """Plotly heatmap of pairwise spectral distances between class means."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return f"<p><em>{self._t('heatmap_caption')}</em></p>"

        names  = sep["names"]
        matrix = sep["matrix"]
        z    = np.round(matrix, 4).tolist()
        text = [[f"{v:.3f}" for v in row] for row in z]

        fig = go.Figure(data=go.Heatmap(
            z=z, x=names, y=names,
            text=text, texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale="Viridis",
            colorbar=dict(title=self._t("dist_label")),
        ))
        fig.update_layout(
            title=dict(
                text=self._t("sep_matrix_title"),
                font=dict(size=13),
            ),
            xaxis_title=self._t("axis_class"),
            yaxis_title=self._t("axis_class"),
            height=max(320, 60 * len(names) + 120),
            margin=dict(l=120, r=20, t=60, b=80),
            plot_bgcolor="#fafafa",
            paper_bgcolor="#ffffff",
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # ---------------------------------------------------------- #
    # Plotly spectral chart
    # ---------------------------------------------------------- #

    _DASH_STYLES = [
        "solid", "dash", "dot", "dashdot", "longdash", "longdashdot",
    ]

    def _spectra_plot_html(
        self,
        spectra: List[Dict[str, Any]],
        wavelengths: Optional[List[float]],
    ) -> str:
        try:
            import plotly.graph_objects as go
        except ImportError:
            return f"<p><em>{self._t('no_chart')}</em></p>"

        show_std = self.rcfg.get("spectra_show_std", True)
        fig = go.Figure()

        for i, s in enumerate(spectra):
            name  = s["name"]
            r, g, b_ = s["color"]
            color_rgb = f"rgb({r},{g},{b_})"
            dash  = self._DASH_STYLES[i % len(self._DASH_STYLES)]
            wl    = s.get("wavelengths") or list(range(len(s["mean"])))
            mean  = s["mean"].tolist()
            std   = s["std"].tolist()

            fig.add_trace(go.Scatter(
                x=wl, y=mean, name=name, mode="lines",
                line=dict(color=color_rgb, width=2.5, dash=dash),
            ))

            if show_std:
                upper = [m + s_ for m, s_ in zip(mean, std)]
                lower = [m - s_ for m, s_ in zip(mean, std)]
                fig.add_trace(go.Scatter(
                    x=wl + wl[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b_},0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{name} ±std",
                    showlegend=False,
                    hoverinfo="skip",
                ))

        x_label = self._t("wavelength_nm") if wavelengths else self._t("band_index")
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=self._t("reflectance"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc", borderwidth=1,
            ),
            margin=dict(l=50, r=20, t=40, b=50),
            hovermode="x unified",
            height=460,
            plot_bgcolor="#fafafa",
            paper_bgcolor="#ffffff",
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # ---------------------------------------------------------- #
    # Utility
    # ---------------------------------------------------------- #

    @staticmethod
    def _stat_box(label: str, value: str) -> str:
        return (
            f'<div class="stat-box">'
            f'<div class="stat-val">{value}</div>'
            f'<div class="stat-lbl">{label}</div>'
            f'</div>'
        )
