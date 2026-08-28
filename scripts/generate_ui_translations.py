"""Generate the static Korean-to-English UI dictionary used by app_en.py.

This is a release-maintenance helper, not a runtime dependency.  It extracts
Korean string literals from the shared Korean source and translates only new
entries, preserving reviewed translations already present in the JSON file.
"""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

from deep_translator import GoogleTranslator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "translations_ko_en.json"
SOURCES = [ROOT / "app.py", ROOT / "app_roi_clustering.py"]


def extract_strings() -> list[str]:
    values: set[str] = set()
    for source in SOURCES:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            value = getattr(node, "value", None)
            if isinstance(value, str) and re.search(r"[가-힣]", value):
                values.add(value)
    return sorted(values, key=lambda value: (len(value), value))


def main() -> None:
    existing = (
        json.loads(OUTPUT.read_text(encoding="utf-8"))
        if OUTPUT.is_file()
        else {}
    )
    pending = [value for value in extract_strings() if value not in existing]
    translator = GoogleTranslator(source="ko", target="en")
    print(f"Existing: {len(existing)}; pending: {len(pending)}")
    for offset in range(0, len(pending), 20):
        batch = pending[offset : offset + 20]
        try:
            translated = translator.translate_batch(batch)
        except Exception as exc:
            print(f"Batch {offset // 20 + 1} failed: {exc}; retrying individually")
            translated = []
            for value in batch:
                try:
                    translated.append(translator.translate(value))
                except Exception:
                    translated.append(value)
                time.sleep(0.15)
        existing.update(zip(batch, translated))
        OUTPUT.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Translated {min(offset + len(batch), len(pending))}/{len(pending)}")
        time.sleep(0.25)


if __name__ == "__main__":
    main()
