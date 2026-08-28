"""Open local result files or folders with the host operating system."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def open_local_path(path: str | Path) -> Path:
    """Open an existing local path in its default application.

    This is intended for a locally running Streamlit server.  A headless
    cluster job cannot open the user's desktop, so it raises a clear error and
    the UI can offer a download instead.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    if os.name == "nt":
        os.startfile(str(resolved))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(resolved)])
    else:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError(
                "This Streamlit server has no desktop session. "
                "Download the HTML report or use the server file manager."
            )
        subprocess.Popen(["xdg-open", str(resolved)])

    return resolved
