import ast
import json
import re
import unittest
from pathlib import Path

from src.english_ui import _delta_wrapper, _progress_wrapper, translate_text


ROOT = Path(__file__).resolve().parents[1]
HANGUL = re.compile(r"[가-힣]")


class EnglishUiTests(unittest.TestCase):
    def test_translation_catalog_covers_shared_ui_literals(self):
        catalog = json.loads(
            (ROOT / "src" / "translations_ko_en.json").read_text(encoding="utf-8")
        )
        literals = set()
        for relative in ("app.py", "app_roi_clustering.py"):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                value = getattr(node, "value", None)
                if isinstance(value, str) and HANGUL.search(value):
                    literals.add(value)
        self.assertEqual(literals - set(catalog), set())

    def test_static_english_catalog_contains_no_hangul(self):
        catalog = json.loads(
            (ROOT / "src" / "translations_ko_en.json").read_text(encoding="utf-8")
        )
        untranslated = {
            source: target
            for source, target in catalog.items()
            if HANGUL.search(str(target))
        }
        self.assertEqual(untranslated, {})

    def test_dynamic_runtime_and_count_strings_are_natural_english(self):
        cases = {
            "⏱ 예상 분석시간: 약 2분 05초–4분 58초": (
                "⏱ Estimated analysis time: about 2m 05s–4m 58s"
            ),
            "최근 실제 소요시간: **1시간 03분** · 6개 파일": (
                "Recent actual runtime: **1h 03m** · 6 files"
            ),
            "**클래스:** 6 클래스": "**Classes:** 6 classes",
            "현재 꼭짓점: **4개** · 최소 3개": (
                "Current vertex: **4** · minimum 3"
            ),
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                translated = translate_text(source)
                self.assertEqual(translated, expected)
                self.assertIsNone(HANGUL.search(translated))

    def test_dynamic_progress_messages_contain_no_hangul(self):
        messages = (
            "기존 파이프라인과 동일한 전처리 적용 중",
            "📊 K-Means 전체 이미지 분석 중",
            "Plot 12 스펙트럼 집계 중 (2/5)",
            "예상 약 12초–29초 · 파일 크기와 분석 단계에 따라 달라질 수 있습니다.",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertIsNone(HANGUL.search(translate_text(message)))

    def test_composed_shared_ui_messages_contain_no_hangul(self):
        """Exercise every f-string with representative runtime values."""
        for relative in ("app.py", "app_roi_clustering.py"):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.JoinedStr):
                    continue
                sample = "".join(
                    part.value
                    if isinstance(part, ast.Constant)
                    and isinstance(part.value, str)
                    else "2"
                    for part in node.values
                )
                if not HANGUL.search(sample):
                    continue
                with self.subTest(file=relative, line=node.lineno):
                    self.assertIsNone(HANGUL.search(translate_text(sample)))

    def test_progress_text_is_translated_for_module_and_container_calls(self):
        captured = []

        def module_progress(value, **kwargs):
            captured.append((value, kwargs.get("text")))

        _progress_wrapper(module_progress)(
            0.25, text="기존 파이프라인과 동일한 전처리 적용 중"
        )

        class FakeContainer:
            pass

        def container_progress(self, value, **kwargs):
            captured.append((value, kwargs.get("text")))

        wrapped_container_progress = _delta_wrapper(
            container_progress, _progress_wrapper
        )
        wrapped_container_progress(
            FakeContainer(), 0.75, text="Plot 12 스펙트럼 집계 중 (2/5)"
        )

        self.assertEqual([value for value, _ in captured], [0.25, 0.75])
        for _, text in captured:
            self.assertIsNone(HANGUL.search(text))


if __name__ == "__main__":
    unittest.main()
