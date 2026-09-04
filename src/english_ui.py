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
    # Dynamic fragments used inside f-strings.  These need natural spacing
    # because the numeric value is inserted before the fragment at runtime.
    "  |  원본 ": "  |  Original ",
    "  ·  원본 ": "  ·  source ",
    " · 원본 ": " · Original ",
    " 밴드": " bands",
    " 밴드  |  ": " bands  |  ",
    " 줄)": " lines)",
    " 클래스": " classes",
    " 파일": " files",
    "개 최근 줄)": " recent lines)",
    "개 파일  ·  원본 ": " files  ·  source ",
    "개 파일 · ": " files · ",
    "개 항목 확인 · ": " entries found · ",
    "개 ROI에 같은 클러스터 라벨을 적용합니다.": (
        " ROIs using the same cluster labels."
    ),
    "개 밴드는 NaN으로 저장되며 분석에서 제외됩니다.": (
        " bands will be stored as NaN and excluded from analysis."
    ),
    "개 입력). reference 파일을 확인하거나 반사율 개수를 맞춰주세요.": (
        " entered). Check the reference file or match the number of reflectance values."
    ),
    "개 있습니다. 현재는 사용할 수 있지만 노출을 낮춘 reference가 더 안전합니다.": (
        " detected. The current reference is usable, but a lower-exposure reference is safer."
    ),
    "현재는 사용할 수 있지만 노출을 낮춘 reference가 더 안전합니다.": (
        "The current reference is usable, but a lower-exposure reference is safer."
    ),
    "reference 파일을 확인하거나 반사율 개수를 맞춰주세요.": (
        "Check the reference file or match the number of reflectance values."
    ),
    "개** · 최소 3개": "** · minimum 3",
    "개** 파일을 찾았습니다. 파일을 선택한 후 [✅ 로드] 버튼을 클릭하세요.": (
        "** files found. Select a file, then click [✅ Load]."
    ),
    "개** 라벨 저장 완료  \n`": "** labels saved  \n`",
    "개)": ")",
    "개가 검출되었습니다": " detected",
    "개 검출되어 프로파일에서 자동 제외했습니다. 노출을 낮춰 다시 측정하거나 더 낮은 반사율 패널을 사용하세요.": (
        " detected and automatically excluded from the profile. Reduce exposure or use a lower-reflectance panel."
    ),
    "개만 찾았습니다 (반사율 ": " panels were found (reflectance values: ",
    # Frequently reused scientific/UI terms that were awkward in the initial
    # machine-generated catalog.
    "  \n**클래스:** ": "  \n**Classes:** ",
    " (반사율)": " (reflectance)",
    " (전 밴드 포화)": " (all bands saturated)",
    "**방법:** ": "**Method:** ",
    "CIR 위색도": "CIR false-color image",
    "RGB 이미지": "RGB Image",
    "선택 식생지수 이미지·요약": "Selected Vegetation-index Images and Summary",
    "클러스터 맵": "Cluster Map",
    "RGB+클러스터 오버레이": "RGB + Cluster Overlay",
    "클러스터별 픽셀 통계": "Per-cluster Pixel Statistics",
    "클러스터별 스펙트럼": "Per-cluster Spectra",
    "보정파일·유효밴드 QC": "Calibration and Valid-band QC",
    "포함 항목: ": "Includes: ",
    "스펙트럼 통계: ": "Spectrum statistics: ",
    " · 식생지수: ": " · Vegetation indices: ",
    "ROI 시험 채택": "Adopt ROI Trial",
    "메모리 예상": "Memory Estimate",
    "분석시간": "analysis time",
    "로드된 배열 ": "loaded array ",
    "⚠️ 이 패널은 일부 파장에서만 사용됩니다. ": (
        "⚠️ This panel is used only at wavelengths where it remains valid. "
    ),
    "미평가": "Not Evaluated",
    "반사율": "reflectance",
    "반사도": "reflectance",
    "비지도 / 지도": "Unsupervised / Supervised",
    "비지도": "Unsupervised",
    "지도": "Supervised",
    "상태": "Status",
    "소스": "Source",
    "시험 C": "Trial C",
    "시험 경계선": "Trial Boundaries",
    "없음": "None",
    "완료": "Complete",
    "원본: ": "Source: ",
    "원본 DN": "Raw DN",
    "원본 RGB": "Raw-DN RGB",
    "원본·보정 비교": "Raw/Calibrated Comparison",
    "유효 밴드": "Valid Bands",
    "전체": "All",
    "전체 밴드": "All Bands",
    "최근 실제 소요시간: **": "Recent actual runtime: **",
    "클래스": "Classes",
    "패널 영상 선택": "Select Panel Image",
    "패널 영상 마우스 모드": "Panel Image Mouse Mode",
    "패널 영상 직접 경로": "Panel Image Path",
    "현재 표시 영상: **": "Current Display Image: **",
    "🌑 Dark 로드": "🌑 Load Dark",
    "📂 결과 폴더 열기": "📂 Open Results Folder",
    "📥 불러오기": "📥 Load",
    "🛠️ 사용자 지정": "🛠️ Custom",
    "🪟 파일": "🪟 Browse File",
    "전체 필드 화면은 `streamlit run app.py`로 실행합니다.": (
        "Run the whole-field screen with `streamlit run app_en.py`."
    ),
    "⏱ GitHub 파일은 다운로드가 끝난 첫 실행부터 예상시간을 보정할 수 있습니다.": (
        "⏱ Runtime estimates become available after the first downloaded GitHub file is analyzed."
    ),
    "⏱ 이 ROI 시험 예상시간: **": "⏱ Estimated runtime for this ROI trial: **",
    "⏱ 이 ROI 시험 실제 소요시간: **": "⏱ Actual runtime for this ROI trial: **",
    "⏱ 최근 전체 분석 실제 소요시간: **": "⏱ Most recent full-analysis runtime: **",
    "**  ·  실행 전 예상 ": "** · estimated before run: ",
    "저장 파일은 현재 **`.xlsx` 통합문서가 아니라 Excel에서 바로 열 수 있는 UTF-8 CSV**입니다. 파일명에 붙은 접미사로 값의 단위를 구분하세요.": (
        "Results are saved as **UTF-8 CSV files**, not as an `.xlsx` workbook. "
        "Excel can open them directly; use each filename suffix to identify the value units."
    ),
    "일반 `spectra_*` CSV는 **한 행이 한 파장**인 wide 형식이며 각 클러스터마다 `mean`, `std`, `median`, `q25`, `q75`, `mna`, `sam_avg` 열이 생깁니다. ROI의 `cluster_spectra*`는 **ROI × 클러스터 × 파장별 한 행**인 long 형식이며 `mean`, `median`, `std`, `q25`, `q75`를 저장합니다. `mna`는 값 기준, `sam_avg`는 스펙트럼 모양 기준의 대표 픽셀 평균입니다.": (
        "Standard `spectra_*` CSV files use **one row per wavelength** (wide format), "
        "with `mean`, `std`, `median`, `q25`, `q75`, `mna`, and `sam_avg` columns "
        "for each cluster. ROI `cluster_spectra*` files use **one row per ROI × cluster × "
        "wavelength** (long format) and store `mean`, `median`, `std`, `q25`, and `q75`. "
        "`mna` is a value-ranked representative-pixel mean; `sam_avg` is a "
        "spectral-shape-ranked representative-pixel mean."
    ),
    "논문용 값은 `_reflectance.csv`에서 `value_units=reflectance`, `calibration_applied=True`, `calibration_qc_status=PASS`를 우선 확인하세요. `REVIEW`는 점프·포화 등 경고를 검토한 뒤 사용하고, `FAIL`은 사용하지 않는 것이 안전합니다.": (
        "For publication-ready values, first confirm `value_units=reflectance`, "
        "`calibration_applied=True`, and `calibration_qc_status=PASS` in `_reflectance.csv`. "
        "Use `REVIEW` only after checking discontinuity and saturation warnings; exclude `FAIL`."
    ),
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
        ("Actual time required", "Actual runtime"),
        ("Actual time spent", "Actual runtime"),
        ("Estimated time for this ROI exam", "Estimated runtime for this ROI trial"),
        ("ROI exam", "ROI trial"),
        ("exam border", "trial boundaries"),
        ("Original/Retouched Comparison", "Raw/Calibrated Comparison"),
        ("false chromaticity", "false-color image"),
        ("Vertically long video", "Vertically long image"),
        ("Horizontally long video", "Horizontally long image"),
        ("current original row", "Current source row"),
        ("Current Source Column", "Current source column"),
        ("Not rated", "Not evaluated"),
        ("Flat ID", "Plot ID"),
        ("Sensor source DN", "Raw sensor DN"),
    ):
        polished = polished.replace(source, target)
    # Values inserted into Korean f-strings can otherwise produce text such as
    # ``6class``.  Keep scientific units compact and count nouns readable.
    polished = re.sub(r"(?<=\d)(classes|class|files|file|bands|band|pixels|pixel|ROIs|ROI)\b", r" \1", polished)
    for plural, singular in (
        ("classes", "class"),
        ("files", "file"),
        ("bands", "band"),
        ("pixels", "pixel"),
        ("ROIs", "ROI"),
    ):
        polished = re.sub(rf"\b1 {plural}\b", f"1 {singular}", polished)
    polished = re.sub(r"\s+:(?=\s|about)", ":", polished)
    polished = re.sub(r":\s*about\s+", ": about ", polished)
    polished = re.sub(r"(?<=\d)(GB|GiB|MB|MiB)\b", r" \1", polished)
    return polished


def _translate_dynamic_patterns(value: str) -> str:
    """Translate Korean duration/count expressions assembled at runtime."""
    number = r"\d+(?:\.\d+)?"
    translated = value
    translated = re.sub(
        r"기존 분석의 \*\*(.+?)\*\* 방법을 전체 이미지에 적용한 뒤, 저장된 (\d[\d,]*)개 ROI에 같은 클러스터 라벨을 적용합니다\.",
        lambda match: (
            f"Apply the **{match.group(1)}** method from the existing analysis "
            f"to the whole image, then apply the same cluster labels to "
            f"{match.group(2)} saved ROIs."
        ),
        translated,
    )
    translated = re.sub(
        r"포화 밴드 (\d[\d,]*)개가 검출되었습니다",
        lambda match: f"{match.group(1)} saturated bands detected",
        translated,
    )
    translated = re.sub(
        r"포화 직전 밴드가 (\d[\d,]*)개 있습니다\.",
        lambda match: f"{match.group(1)} near-saturation bands detected.",
        translated,
    )
    translated = re.sub(
        r"(\d[\d,]*)개 밴드는 NaN으로 저장되며 분석에서 제외됩니다\.",
        lambda match: (
            f"{match.group(1)} bands will be stored as NaN and excluded from analysis."
        ),
        translated,
    )
    translated = re.sub(
        r"저장된 (\d[\d,]*)개 ROI에 같은 클러스터 라벨을 적용합니다\.",
        lambda match: (
            f"The same cluster labels will be applied to {match.group(1)} saved ROIs."
        ),
        translated,
    )
    translated = re.sub(
        r"패널을 (\d[\d,]*)개만 찾았습니다",
        lambda match: f"Only {match.group(1)} panels were found",
        translated,
    )
    translated = re.sub(
        r"반사율 (\d[\d,]*)개 입력",
        lambda match: f"{match.group(1)} reflectance values entered",
        translated,
    )
    translated = re.sub(
        rf"({number})시간\s*({number})분",
        lambda match: f"{match.group(1)}h {match.group(2)}m",
        translated,
    )
    translated = re.sub(
        rf"({number})분\s*({number})초",
        lambda match: f"{match.group(1)}m {match.group(2)}s",
        translated,
    )
    translated = re.sub(rf"({number})시간", lambda match: f"{match.group(1)}h", translated)
    translated = re.sub(rf"({number})분", lambda match: f"{match.group(1)}m", translated)
    translated = re.sub(rf"({number})초", lambda match: f"{match.group(1)}s", translated)
    translated = re.sub(r"약\s+(?=\d)", "about ", translated)
    translated = re.sub(r"(\d[\d,]*)개\s+ROI", r"\1 ROIs", translated)
    translated = re.sub(r"(\d[\d,]*)개\s+파일", r"\1 files", translated)
    translated = re.sub(r"(\d[\d,]*)개\s+밴드", r"\1 bands", translated)
    translated = re.sub(r"(\d[\d,]*)개\s+픽셀", r"\1 pixels", translated)
    translated = re.sub(r"(\d[\d,]*)개\s+클래스", r"\1 classes", translated)
    translated = re.sub(r"픽셀\s+(\d[\d,]*)개", r"\1 pixels", translated)
    translated = re.sub(r"ROI\s+(\d[\d,]*)개", r"\1 ROIs", translated)
    return translated


def _load_translations() -> None:
    global _TRANSLATIONS, _REPLACEMENTS
    if _TRANSLATIONS:
        return
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


def translate_text(value: Any) -> Any:
    if not isinstance(value, str) or not _HANGUL.search(value):
        return value
    _load_translations()
    exact = _TRANSLATIONS.get(value)
    if exact:
        return _polish_english(exact)
    translated = _translate_dynamic_patterns(value)
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


def _progress_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(value: Any, *args: Any, **kwargs: Any) -> Any:
        updated = dict(kwargs)
        if "text" in updated:
            updated["text"] = translate_text(updated["text"])
        return function(value, *args, **updated)

    return wrapped


def _metric_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(label: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        updated = _translate_help(kwargs)
        if "delta" in updated:
            updated["delta"] = translate_text(updated["delta"])
        return function(
            translate_text(label), translate_text(value), *args, **updated
        )

    return wrapped


def _code_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(body: Any, *args: Any, **kwargs: Any) -> Any:
        language = kwargs.get("language")
        if language is None and args:
            language = args[0]
        display = body
        if str(language or "text").lower() not in {"python", "py"}:
            display = translate_text(body)
        return function(display, *args, **_translate_help(kwargs))

    return wrapped


def _delta_wrapper(
    function: Callable[..., Any],
    wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> Callable[..., Any]:
    """Apply an existing wrapper to methods on columns/placeholders/sidebar."""
    @wraps(function)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = function.__get__(self, type(self))
        return wrapper_factory(bound)(*args, **kwargs)

    return wrapped


def _patch_delta_generator() -> None:
    """Translate calls made through ``st.columns()``, placeholders, and forms."""
    from streamlit.delta_generator import DeltaGenerator

    groups = (
        (
            (
                "markdown", "caption", "info", "warning", "error", "success",
                "header", "subheader", "title", "write", "toast",
            ),
            _text_wrapper,
        ),
        (
            (
                "button", "checkbox", "toggle", "text_input", "text_area",
                "number_input", "slider", "file_uploader", "download_button",
                "date_input", "time_input", "color_picker", "expander",
                "spinner", "form_submit_button", "status", "popover",
            ),
            _label_wrapper,
        ),
        (
            ("radio", "selectbox", "multiselect", "select_slider", "pills", "segmented_control"),
            _options_wrapper,
        ),
        (("tabs",), _tabs_wrapper),
        (("image",), _image_wrapper),
        (("page_link",), _page_link_wrapper),
        (("dataframe", "data_editor"), _dataframe_wrapper),
        (("plotly_chart",), _plotly_wrapper),
        (("progress",), _progress_wrapper),
        (("metric",), _metric_wrapper),
        (("code",), _code_wrapper),
    )
    for names, wrapper_factory in groups:
        for name in names:
            if not hasattr(DeltaGenerator, name):
                continue
            key = f"DeltaGenerator.{name}"
            original = getattr(DeltaGenerator, name)
            _ORIGINALS[key] = original
            setattr(
                DeltaGenerator,
                name,
                _delta_wrapper(original, wrapper_factory),
            )


def install_english_ui() -> None:
    """Patch Streamlit display calls once for an English view of the shared app."""
    import streamlit as st

    if getattr(st, "_hyperspectral_english_ui", False):
        return
    _load_translations()

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
        "spinner", "form_submit_button", "status", "popover",
    ):
        if hasattr(st, name):
            _ORIGINALS[name] = getattr(st, name)
            setattr(st, name, _label_wrapper(getattr(st, name)))

    for name in (
        "radio", "selectbox", "multiselect", "select_slider", "pills",
        "segmented_control",
    ):
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
    _ORIGINALS["progress"] = st.progress
    st.progress = _progress_wrapper(st.progress)
    _ORIGINALS["metric"] = st.metric
    st.metric = _metric_wrapper(st.metric)
    _ORIGINALS["code"] = st.code
    st.code = _code_wrapper(st.code)
    _patch_delta_generator()
    st._hyperspectral_english_ui = True
