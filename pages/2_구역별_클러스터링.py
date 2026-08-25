"""Always execute the latest ROI page source on every Streamlit rerun.

Using ``from app_roi_clustering import *`` leaves the module cached in a
long-running Streamlit process, so browser refreshes can keep showing an older
ROI implementation. Reading and executing the source makes code fixes take
effect without restarting PowerShell.
"""

from pathlib import Path

_page = Path(__file__).resolve().parents[1] / "app_roi_clustering.py"
if not _page.is_file():
    raise FileNotFoundError(f"ROI clustering web page not found: {_page}")

exec(compile(_page.read_text(encoding="utf-8"), str(_page), "exec"), globals())
