"""Report presets and normalized, serializable report configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


SECTION_KEYS = (
    "rgb",
    "false_color",
    "spectral_indices",
    "class_map",
    "cluster_overlay",
    "per_class_images",
    "class_summary",
    "spectral_plot",
    "quality_metrics",
    "vegetation_quality",
    "calibration_qc",
)

STATISTIC_KEYS = ("mean", "median", "std", "iqr")
INDEX_KEYS = ("NDVI", "GNDVI", "NDRE", "PRI")


REPORT_PRESETS: dict[str, dict[str, Any]] = {
    "quick_qc": {
        "sections": {
            "rgb": True,
            "false_color": False,
            "spectral_indices": True,
            "class_map": True,
            "cluster_overlay": True,
            "per_class_images": False,
            "class_summary": True,
            "spectral_plot": True,
            "quality_metrics": False,
            "vegetation_quality": False,
            "calibration_qc": True,
        },
        "spectra_statistics": ["mean", "median"],
        "indices": ["NDVI"],
        "save_selected_images": True,
        "daily_summary": True,
    },
    "research_standard": {
        "sections": {key: True for key in SECTION_KEYS},
        "spectra_statistics": ["mean", "median", "std", "iqr"],
        "indices": ["NDVI", "GNDVI", "NDRE", "PRI"],
        "save_selected_images": True,
        "daily_summary": True,
    },
    "custom": {
        "sections": {
            "rgb": True,
            "false_color": False,
            "spectral_indices": True,
            "class_map": True,
            "cluster_overlay": True,
            "per_class_images": False,
            "class_summary": True,
            "spectral_plot": True,
            "quality_metrics": False,
            "vegetation_quality": False,
            "calibration_qc": True,
        },
        "spectra_statistics": ["mean", "median"],
        "indices": ["NDVI"],
        "save_selected_images": True,
        "daily_summary": True,
    },
}


_LEGACY_SECTION_KEYS = {
    "show_rgb_composite": "rgb",
    "show_false_color": "false_color",
    "show_spectral_plots": "spectral_plot",
    "show_statistics": "class_summary",
    "show_confusion_info": "quality_metrics",
}


def _bool_map(values: Mapping[str, Any] | None, defaults: Mapping[str, bool]) -> dict[str, bool]:
    supplied = dict(values or {})
    return {
        key: bool(supplied.get(key, defaults.get(key, False)))
        for key in SECTION_KEYS
    }


def resolve_report_options(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return one complete report configuration.

    ``config`` may be the full application config or only its ``report``
    mapping.  Older YAML switches remain supported, while new callers should
    use ``preset``, ``sections``, ``spectra_statistics`` and ``indices``.
    The returned dictionary contains only JSON-serializable values so it can
    be written directly into the processing manifest.
    """

    supplied = dict(config or {})
    rcfg = dict(supplied.get("report", supplied))
    preset_name = str(rcfg.get("preset") or "research_standard").strip().lower()
    if preset_name not in REPORT_PRESETS:
        preset_name = "research_standard"

    resolved = deepcopy(REPORT_PRESETS[preset_name])
    legacy_overrides: dict[str, bool] = {}
    for legacy_key, section_key in _LEGACY_SECTION_KEYS.items():
        if legacy_key in rcfg:
            legacy_overrides[section_key] = bool(rcfg[legacy_key])

    section_overrides = dict(rcfg.get("sections") or {})
    section_overrides = {**legacy_overrides, **section_overrides}
    resolved["sections"] = _bool_map(
        section_overrides,
        resolved["sections"],
    )

    stats = rcfg.get("spectra_statistics", resolved["spectra_statistics"])
    if isinstance(stats, str):
        stats = [stats]
    resolved["spectra_statistics"] = [
        key for key in STATISTIC_KEYS if key in set(stats or [])
    ]
    if not resolved["spectra_statistics"]:
        resolved["spectra_statistics"] = ["mean"]

    indices = rcfg.get("indices", resolved["indices"])
    if isinstance(indices, str):
        indices = [indices]
    requested_indices = {str(value).upper() for value in (indices or [])}
    resolved["indices"] = [
        key for key in INDEX_KEYS if key in requested_indices
    ]

    resolved["save_selected_images"] = bool(
        rcfg.get("save_selected_images", resolved["save_selected_images"])
    )
    resolved["daily_summary"] = bool(
        rcfg.get("daily_summary", resolved["daily_summary"])
    )
    resolved["preset"] = preset_name
    return resolved
