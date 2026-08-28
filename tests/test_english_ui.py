import ast
import json
import re
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
