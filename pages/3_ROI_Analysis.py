"""English view of the shared ROI analysis and re-clustering page."""

from pathlib import Path

from src.english_ui import install_english_ui


install_english_ui()
_page = Path(__file__).resolve().parents[1] / "app_roi_clustering.py"
exec(compile(_page.read_text(encoding="utf-8"), str(_page), "exec"), globals())
