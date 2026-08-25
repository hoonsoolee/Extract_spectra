"""Native Windows file/folder dialogs for a locally hosted Streamlit app.

Browser upload controls cannot reveal an existing local path and are unsuitable
for multi-gigabyte hyperspectral files.  When Streamlit is running on the same
Windows workstation as the user, tkinter can safely return a path without
uploading the file.  Headless/Linux servers keep using the normal text fields.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def native_dialogs_available() -> bool:
    """Return whether the Streamlit server is running on Windows."""
    return os.name == "nt"


def _initial_directory(value: str | None) -> str:
    candidate = Path(value or ".").expanduser()
    try:
        candidate = candidate.resolve()
    except OSError:
        candidate = Path.cwd()
    if candidate.is_file():
        candidate = candidate.parent
    while not candidate.is_dir() and candidate != candidate.parent:
        candidate = candidate.parent
    return str(candidate if candidate.is_dir() else Path.cwd())


def _make_root():
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover - depends on local Python build
        raise RuntimeError("이 Python 환경에 Windows 탐색기 기능(tkinter)이 없습니다.") from exc

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        return root
    except Exception as exc:  # pragma: no cover - depends on desktop session
        raise RuntimeError(
            "Windows 탐색기 창을 열 수 없습니다. 경로를 직접 입력해 주세요."
        ) from exc


def choose_directory(title: str, initial_path: str = "") -> str:
    """Open a native folder dialog and return the chosen path, or ``''``."""
    if not native_dialogs_available():
        raise RuntimeError("Windows 로컬 실행에서만 탐색기 창을 열 수 있습니다.")
    from tkinter import filedialog

    root = _make_root()
    try:
        selected = filedialog.askdirectory(
            parent=root,
            title=title,
            initialdir=_initial_directory(initial_path),
            mustexist=True,
        )
        return str(Path(selected)) if selected else ""
    finally:
        root.destroy()


def choose_file(
    title: str,
    initial_path: str = "",
    filetypes: Iterable[tuple[str, str]] | None = None,
) -> str:
    """Open a native file dialog and return the chosen path, or ``''``."""
    if not native_dialogs_available():
        raise RuntimeError("Windows 로컬 실행에서만 탐색기 창을 열 수 있습니다.")
    from tkinter import filedialog

    root = _make_root()
    try:
        selected = filedialog.askopenfilename(
            parent=root,
            title=title,
            initialdir=_initial_directory(initial_path),
            filetypes=list(filetypes or [("모든 파일", "*.*")]),
        )
        return str(Path(selected)) if selected else ""
    finally:
        root.destroy()
