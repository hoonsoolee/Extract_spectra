"""English presentation layer for the single-source Streamlit application.

The analysis logic lives in ``app.py`` and ``app_roi_clustering.py``.  This
module translates Streamlit display arguments while leaving option values,
session-state keys, file paths, and processing decisions unchanged.  Korean
and English users therefore execute exactly the same implementation.
"""

from __future__ import annotations

import copy
import json
import re
from functools import wraps
from pathlib import Path
from typing import Any, Callable


_HANGUL = re.compile(r"[가-힣]")
_TRANSLATIONS_PATH = Path(__file__).with_name("translations_ko_en.json")
_ORIGINALS: dict[str, Callable[..., Any]] = {}
_TRANSLATIONS: dict[str, str] = {}
_REPLACEMENTS: list[tuple[str, str]] = []

_REVIEWED_OVERRIDES = {
    "소스": "Source",
    "방법": "Method",
    "**방법:** ": "**Method:** ",
    "  \n**데이터:** ": "  \n**Data:** ",
    "  \n**클래스:** ": "  \n**Classes:** ",
    "  \n**파일:** ": "  \n**Files:** ",
    "  ·  **출력:** `": "  ·  **Output:** `",
    "비지도": "Unsupervised",
    "폴더 경로": "Folder Path",
    "로컬 폴더": "Local Folder",
    "GitHub 저장소": "GitHub Repository",
    "처리 모드": "Processing Mode",
    "파일 선택": "Select File",
    "출력 폴더": "Output Folder",
    "정규화 방식": "Normalization Method",
    "반사도": "Reflectance",
    "반사율": "Reflectance",
    "보정 반사율": "Calibrated Reflectance",
    "패널 영상 선택": "Select Panel Image",
    "보정 패널 영상 선택": "Select Calibration Panel Image",
    "상세 로그 (DEBUG)": "Detailed Log (DEBUG)",
    "📂 로드": "📂 Load",
    "🪟 선택": "🪟 Browse",
    "📂 폴더 스캔": "📂 Scan Folder",
    "📂 선택 파일 로드": "📂 Load Selected File",
    "🪟 파일": "🪟 Browse File",
    "처리할 파일": "File to Process",
    "열어볼 촬영 구간 / 센서": "Acquisition Segment / Sensor",
    "🧭 CERES 내부 목록 읽기": "🧭 Read CERES Contents",
    "👁️ 빠른 미리보기": "👁️ Quick Preview",
    "📦 선택 항목 분석 준비": "📦 Prepare Selected Entry",
    "🚀 분석 실행": "🚀 Run Analysis",
    "🚀  분석 시작": "🚀 Start Analysis",
    "🌿 전체 필드 자동 분석": "🌿 Whole-field Analysis",
    "🗺️ ROI 구역 분석·재클러스터링": "🗺️ ROI Analysis & Re-clustering",
    "구역별 초분광 클러스터링": "ROI Hyperspectral Clustering",
    "🗺️ 구역별 분석 설정": "🗺️ ROI Analysis Settings",
    "전역 클러스터링 · 구역별 스펙트럼": (
        "Global Clustering · ROI-level Spectra"
    ),
    "📈 ROI 스펙트럼": "📈 ROI Spectra",
    "🎯 패널 보정": "🎯 Panel Calibration",
    "🏷️ 픽셀 라벨링": "🏷️ Pixel Labeling",
    "배치 완료 후 팀용 일일 패키지 생성": (
        "Create Team Daily Package After Batch"
    ),
    "👥 팀·플랏 일일 통합 리포트": "👥 Team / Plot Daily Package",
    "### 👥 팀·플랏 일일 통합 결과": "### 👥 Team / Plot Daily Results",
    "플랏 메타데이터 CSV (선택)": "Plot Metadata CSV (Optional)",
    "플랏 메타데이터 CSV 선택": "Select Plot Metadata CSV",
}


def _polish_english(value: str) -> str:
    polished = value.replace("\u200b", "")
    for source, target in (
        ("sauce", "Source"),
        ("reflectivity", "reflectance"),
        ("Reflectivity", "Reflectance"),
        ("correction file", "calibration file"),
        ("Correction file", "Calibration file"),
        ("correction profile", "calibration profile"),
        ("Correction Profile", "Calibration Profile"),
        ("correction coefficient", "calibration coefficient"),
        ("Correction coefficient", "Calibration coefficient"),
        ("effective band", "valid band"),
        ("Panel video", "Panel image"),
        ("panel video", "panel image"),
        ("ground truth Dark", "measured Dark"),
        ("after deployment is complete", "after the batch completes"),
        ("Absolutely not reflective", "Not absolute reflectance"),
    ):
        polished = polished.replace(source, target)
    return polished


def translate_text(value: Any) -> Any:
    if not isinstance(value, str) or not _HANGUL.search(value):
        return value
    exact = _TRANSLATIONS.get(value)
    if exact:
        return _polish_english(exact)
    translated = value
    for source, target in _REPLACEMENTS:
        if source in translated:
            translated = translated.replace(source, target)
    return _polish_english(translated)


def _translate_help(kwargs: dict[str, Any]) -> dict[str, Any]:
    updated = dict(kwargs)
    for key in ("help", "placeholder", "caption", "text", "page_title"):
        if key in updated:
            updated[key] = translate_text(updated[key])
    return updated


def _text_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        translated_args = tuple(translate_text(value) for value in args)
        return function(*translated_args, **_translate_help(kwargs))

    return wrapped


def _label_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(label: Any, *args: Any, **kwargs: Any) -> Any:
        return function(translate_text(label), *args, **_translate_help(kwargs))

    return wrapped


def _options_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(label: Any, options: Any, *args: Any, **kwargs: Any) -> Any:
        updated = _translate_help(kwargs)
        original_format = updated.get("format_func", str)

        def english_format(value: Any) -> str:
            return str(translate_text(original_format(value)))

        updated["format_func"] = english_format
        return function(translate_text(label), options, *args, **updated)

    return wrapped


def _tabs_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(tabs: Any, *args: Any, **kwargs: Any) -> Any:
        return function([translate_text(item) for item in tabs], *args, **kwargs)

    return wrapped


def _image_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(image: Any, *args: Any, **kwargs: Any) -> Any:
        updated = dict(kwargs)
        caption = updated.get("caption")
        if isinstance(caption, list):
            updated["caption"] = [translate_text(item) for item in caption]
        elif caption is not None:
            updated["caption"] = translate_text(caption)
        return function(image, *args, **updated)

    return wrapped


def _page_link_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(page: Any, *args: Any, **kwargs: Any) -> Any:
        original_page = page
        page_text = str(page).replace("\\", "/")
        if page_text == "app.py":
            page = "app_en.py"
        elif page_text.endswith("pages/2_구역별_클러스터링.py"):
            page = "pages/3_ROI_Analysis.py"
        updated = _translate_help(kwargs)
        if "label" in updated:
            updated["label"] = translate_text(updated["label"])
        try:
            return function(page, *args, **updated)
        except KeyError:
            # AppTest can execute a page without registering its normal main
            # script.  Falling back keeps that isolated smoke test usable;
            # a normal ``streamlit run app_en.py`` session uses English links.
            try:
                return function(original_page, *args, **updated)
            except KeyError:
                return None

    return wrapped


def _dataframe_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(data: Any = None, *args: Any, **kwargs: Any) -> Any:
        try:
            display = data.copy()
            display.columns = [translate_text(str(column)) for column in display.columns]
        except Exception:
            display = data
        return function(display, *args, **_translate_help(kwargs))

    return wrapped


def _plotly_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(figure: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            display = copy.deepcopy(figure)
            layout = display.layout
            if layout.title and layout.title.text:
                layout.title.text = translate_text(layout.title.text)
            for axis_name in ("xaxis", "yaxis", "yaxis2", "coloraxis"):
                axis = getattr(layout, axis_name, None)
                if axis is not None and getattr(axis, "title", None):
                    axis.title.text = translate_text(axis.title.text)
            for annotation in layout.annotations or ():
                annotation.text = translate_text(annotation.text)
            for trace in display.data:
                if getattr(trace, "name", None):
                    trace.name = translate_text(trace.name)
        except Exception:
            display = figure
        return function(display, *args, **_translate_help(kwargs))

    return wrapped


def install_english_ui() -> None:
    """Patch Streamlit display calls once for an English view of the shared app."""
    import streamlit as st

    if getattr(st, "_hyperspectral_english_ui", False):
        return
    global _TRANSLATIONS, _REPLACEMENTS
    _TRANSLATIONS = json.loads(_TRANSLATIONS_PATH.read_text(encoding="utf-8"))
    _TRANSLATIONS.update(_REVIEWED_OVERRIDES)
    _REPLACEMENTS = sorted(
        (
            (source, target)
            for source, target in _TRANSLATIONS.items()
            if source.strip() and len(source) >= 2 and source != target
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    _ORIGINALS["set_page_config"] = st.set_page_config

    @wraps(st.set_page_config)
    def english_page_config(*args: Any, **kwargs: Any) -> Any:
        return _ORIGINALS["set_page_config"](*args, **_translate_help(kwargs))

    st.set_page_config = english_page_config

    for name in (
        "markdown", "caption", "info", "warning", "error", "success",
        "header", "subheader", "title", "write", "toast",
    ):
        if hasattr(st, name):
            _ORIGINALS[name] = getattr(st, name)
            setattr(st, name, _text_wrapper(getattr(st, name)))

    for name in (
        "button", "checkbox", "toggle", "text_input", "text_area",
        "number_input", "slider", "file_uploader", "download_button",
        "date_input", "time_input", "color_picker", "metric", "expander",
        "spinner", "form_submit_button",
    ):
        if hasattr(st, name):
            _ORIGINALS[name] = getattr(st, name)
            setattr(st, name, _label_wrapper(getattr(st, name)))

    for name in ("radio", "selectbox", "multiselect", "select_slider"):
        if hasattr(st, name):
            _ORIGINALS[name] = getattr(st, name)
            setattr(st, name, _options_wrapper(getattr(st, name)))

    _ORIGINALS["tabs"] = st.tabs
    st.tabs = _tabs_wrapper(st.tabs)
    _ORIGINALS["image"] = st.image
    st.image = _image_wrapper(st.image)
    _ORIGINALS["page_link"] = st.page_link
    st.page_link = _page_link_wrapper(st.page_link)
    _ORIGINALS["dataframe"] = st.dataframe
    st.dataframe = _dataframe_wrapper(st.dataframe)
    _ORIGINALS["data_editor"] = st.data_editor
    st.data_editor = _dataframe_wrapper(st.data_editor)
    _ORIGINALS["plotly_chart"] = st.plotly_chart
    st.plotly_chart = _plotly_wrapper(st.plotly_chart)
    st._hyperspectral_english_ui = True
