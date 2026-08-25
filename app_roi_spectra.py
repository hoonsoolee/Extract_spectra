"""
app_roi_spectra.py
------------------
Small Streamlit app for extracting mean and median spectra from a user-selected
ROI box.

Run with:
    python -m streamlit run app_roi_spectra.py
"""

from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))


st.set_page_config(
    page_title="ROI 스펙트럼 추출",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _open_native_file_dialog(title: str, initial_path: str = "") -> str:
    initial_dir = Path(initial_path).expanduser()
    if initial_dir.is_file():
        initial_dir = initial_dir.parent
    if not initial_dir.exists():
        initial_dir = Path.cwd()

    if sys.platform.startswith("win"):
        ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.StartPosition = 'CenterScreen'
$owner.Width = 1
$owner.Height = 1
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
$owner.Show()
$owner.Activate()
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = {title!r}
$dialog.InitialDirectory = {str(initial_dir)!r}
$dialog.Filter = 'Hyperspectral / ENVI (*.hdr;*.raw;*.bil;*.bip;*.bsq)|*.hdr;*.raw;*.bil;*.bip;*.bsq|Raster / data files (*.tif;*.tiff;*.h5;*.hdf5;*.mat)|*.tif;*.tiff;*.h5;*.hdf5;*.mat|All files (*.*)|*.*'
$dialog.CheckFileExists = $true
$dialog.Multiselect = $false
$result = $dialog.ShowDialog($owner)
$owner.Dispose()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $dialog.FileName
}}
"""
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Sta", "-Command", ps_script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            msg = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"파일 선택창을 열 수 없습니다: {msg}")
        return completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise RuntimeError(f"파일 선택창을 열 수 없습니다: {e}") from e

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    try:
        path = filedialog.askopenfilename(
            parent=root,
            title=title,
            initialdir=str(initial_dir),
            filetypes=[
                ("Hyperspectral / ENVI", "*.hdr *.raw *.bil *.bip *.bsq"),
                ("Raster / data files", "*.tif *.tiff *.h5 *.hdf5 *.mat"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    return str(path) if path else ""


from src.roi_utils import (
    box_region as _box_region,
    display_rgb as _get_display_rgb,
    roi_stats as _roi_stats,
    save_roi_csv as _save_roi_csv,
    selection_to_region as _selection_to_region,
)


def _load_cube(path: str, downsample: int = 1) -> tuple[np.ndarray, list[float] | None, dict]:
    from src.data_loader import HyperspectralLoader

    loader = HyperspectralLoader(
        {"cache_dir": "./cache", "spatial_downsample": int(downsample)}
    )
    data, meta = loader.load_local(path)
    return data, meta.get("wavelengths"), meta


def _default_roi_output_path(source_file: str) -> str:
    p = Path(source_file)
    return str(p.with_name(f"{p.stem}_roi_spectra.csv"))




defaults = {
    "roi_file": "",
    "roi_file_input": "",
    "roi_file_pending": "",
    "roi_data": None,
    "roi_wl": None,
    "roi_meta": None,
    "roi_rgb": None,
    "roi_region": None,
    "roi_box": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.get("roi_file_pending"):
    st.session_state["roi_file"] = st.session_state["roi_file_pending"]
    st.session_state["roi_file_input"] = st.session_state["roi_file_pending"]
    st.session_state["roi_file_pending"] = ""


with st.sidebar:
    st.markdown("## ROI 스펙트럼 추출")

    c1, c2 = st.columns([4, 1])
    with c1:
        file_path = st.text_input(
            "초분광 파일",
            value=st.session_state["roi_file"],
            placeholder="C:/data/image.hdr",
            key="roi_file_input",
        )
    with c2:
        st.write("")
        if st.button("찾기", use_container_width=True):
            try:
                picked = _open_native_file_dialog(
                    "ROI 스펙트럼을 추출할 파일 선택",
                    st.session_state.get("roi_file_input", ""),
                )
                if picked:
                    st.session_state["roi_file"] = picked
                    st.session_state["roi_file_pending"] = picked
                    st.rerun()
            except Exception as e:
                st.error(str(e))

    roi_downsample = st.select_slider(
        "공간 다운샘플링 (1 = 원본 해상도)",
        options=[1, 2, 4, 8],
        value=1,
        help=(
            "N이면 N×N 픽셀 블록당 1개만 읽어 메모리를 1/N²로 줄입니다. "
            "수 GB 이상 파일은 4 이상을 권장합니다. "
            "각 픽셀의 스펙트럼은 그대로 유지됩니다."
        ),
    )

    if st.button("파일 로드", type="primary", use_container_width=True):
        try:
            with st.spinner("초분광 파일 로딩 중... (대용량 파일은 수 분 걸릴 수 있습니다)"):
                data, wl, meta = _load_cube(file_path, downsample=int(roi_downsample))
                st.session_state["roi_file"] = file_path
                st.session_state["roi_data"] = data
                st.session_state["roi_wl"] = wl
                st.session_state["roi_meta"] = meta
                st.session_state["roi_rgb"] = _get_display_rgb(data, wl)
                st.session_state["roi_region"] = None
                st.session_state["roi_box"] = None
                st.session_state["roi_save_path"] = _default_roi_output_path(file_path)
            st.success(f"로드 완료: {data.shape[0]} x {data.shape[1]} x {data.shape[2]}")
        except Exception:
            st.error("파일 로드 실패")
            st.code(traceback.format_exc(), language="python")

    st.divider()
    default_out = "roi_spectrum.csv"
    if st.session_state.get("roi_file"):
        default_out = _default_roi_output_path(st.session_state["roi_file"])
    save_path = st.text_input("CSV 저장 경로", value=default_out, key="roi_save_path")


st.markdown("# ROI 평균/중간값 스펙트럼 추출")
st.caption("이미지에서 박스 또는 올가미(lasso)로 영역을 지정하면 그 영역 픽셀들의 mean/median spectrum을 CSV로 저장합니다.")

data = st.session_state.get("roi_data")
rgb = st.session_state.get("roi_rgb")
wl = st.session_state.get("roi_wl")

if data is None or rgb is None:
    st.info("왼쪽 사이드바에서 초분광 파일을 선택하고 **파일 로드**를 누르세요.")
else:
    meta = st.session_state.get("roi_meta") or {}
    H, W, B = data.shape
    st.success(
        f"파일: `{meta.get('filename', Path(st.session_state['roi_file']).name)}` | "
        f"크기: {H} x {W} px | 밴드: {B}"
    )

    left, right = st.columns([3, 2])

    with left:
        st.markdown("### 1. ROI 지정")
        st.caption("상단 Plotly 도구에서 Box Select 또는 Lasso Select를 선택해 영역을 그리세요.")
        fig = go.Figure()
        fig.add_trace(go.Image(z=rgb))
        fig.update_layout(
            dragmode="select",
            margin=dict(l=0, r=0, t=0, b=0),
            height=620,
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        event = st.plotly_chart(
            fig,
            key="roi_box_chart",
            on_select="rerun",
            selection_mode=("box", "lasso"),
            use_container_width=True,
        )

        if event is not None and hasattr(event, "selection"):
            region = _selection_to_region(event.selection, H, W)
            if region is not None:
                st.session_state["roi_region"] = region
                st.session_state["roi_box"] = region.get("roi")

    with right:
        st.markdown("### 2. 스펙트럼 확인 및 저장")
        region = st.session_state.get("roi_region")
        if not region:
            st.info("왼쪽 이미지에서 박스 또는 올가미로 영역을 드래그하세요.")
        else:
            r0, r1, c0, c1 = region.get("roi")
            st.write(
                f"선택 ROI: `{region.get('type', 'box')}` | "
                f"row `{r0}:{r1}`, col `{c0}:{c1}`"
            )

            try:
                with st.expander("좌표 직접 조정", expanded=False):
                    with st.form("roi_manual_form"):
                        nr0 = st.number_input("row 시작", 0, H - 1, int(r0), 1)
                        nr1 = st.number_input("row 끝", 1, H, int(r1), 1)
                        nc0 = st.number_input("col 시작", 0, W - 1, int(c0), 1)
                        nc1 = st.number_input("col 끝", 1, W, int(c1), 1)
                        if st.form_submit_button("이 좌표로 박스 ROI 적용"):
                            st.session_state["roi_region"] = _box_region(
                                [nr0, nr1, nc0, nc1], H, W
                            )
                            st.session_state["roi_box"] = st.session_state["roi_region"]["roi"]
                            st.rerun()

                stats_df, n_pixels, bounds, region_type = _roi_stats(data, region)
                st.caption(f"계산에 사용된 픽셀 수: {n_pixels:,}")
                x_axis = wl if wl is not None and len(wl) == len(stats_df) else list(range(len(stats_df)))
                x_title = "Wavelength (nm)" if wl is not None and len(wl) == len(stats_df) else "Band index"

                spec_fig = go.Figure()
                spec_fig.add_trace(
                    go.Scatter(x=x_axis, y=stats_df["mean"], mode="lines", name="Mean")
                )
                spec_fig.add_trace(
                    go.Scatter(x=x_axis, y=stats_df["median"], mode="lines", name="Median")
                )
                spec_fig.update_layout(
                    height=360,
                    xaxis_title=x_title,
                    yaxis_title="Value",
                    margin=dict(l=40, r=10, t=30, b=40),
                    legend=dict(orientation="h", y=1.08),
                )
                st.plotly_chart(spec_fig, use_container_width=True)

                preview = stats_df[["mean", "median"]].copy()
                if wl is not None and len(wl) == len(preview):
                    preview.insert(0, "wavelength_nm", wl)
                else:
                    preview.insert(0, "band_index", np.arange(len(preview)))
                st.dataframe(preview.head(20), use_container_width=True, height=260)

                if st.button("CSV 저장", type="primary", use_container_width=True):
                    out = _save_roi_csv(
                        data=data,
                        wavelengths=wl,
                        region=region,
                        source_file=st.session_state["roi_file"],
                        path=save_path,
                    )
                    st.success(f"저장 완료: `{out.resolve()}`")
                    st.caption(f"저장된 픽셀 수: {n_pixels:,}")
            except Exception:
                st.error("ROI 스펙트럼 계산 실패")
                st.code(traceback.format_exc(), language="python")
