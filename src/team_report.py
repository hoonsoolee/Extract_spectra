"""Team/day aggregation for plot-level hyperspectral batch results.

This module deliberately works from the compact per-file outputs produced by
``Pipeline``.  It never reopens the hyperspectral cubes, so creating a team
report does not add meaningful RAM pressure to a CERES/BIL batch run.
"""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps


logger = logging.getLogger(__name__)

# Keep JSON/workbook construction comfortably below Excel's row ceiling and
# prevent the compact aggregation stage from becoming a new RAM bottleneck.
_SPECTRA_ROW_LIMIT = 250_000

_SUMMARY_COLUMNS = [
    "measurement_date",
    "team",
    "plot_id",
    "filename",
    "treatment",
    "genotype",
    "replicate",
    "value_units",
    "calibration_qc_status",
    "included_in_team_statistics",
    "ndvi_mean",
    "ndvi_median",
    "ndvi_q25",
    "ndvi_q75",
    "ndvi_iqr",
    "vegetation_fraction",
    "n_classes",
    "silhouette",
    "davies_bouldin",
    "elapsed_seconds",
    "detail_report",
    "spectral_samples_file",
    "source_file",
]


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _slug(value: str, fallback: str = "unassigned") -> str:
    text = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", str(value).strip())
    return text.strip("._-") or fallback


def derive_plot_id(filename: str) -> str:
    """Derive a readable plot identifier without guessing experimental labels."""
    name = Path(str(filename)).name
    lowered = name.lower()
    for suffix in (".bil.hdr", ".bip.hdr", ".bsq.hdr", ".raw.hdr", ".img.hdr"):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    else:
        name = Path(name).stem
    name = re.sub(r"(?i)(?:[._-](?:vnir|swir|nir|rgb))$", "", name)
    return name or Path(str(filename)).stem


def _normalise_manifest_row(row: dict[str, Any]) -> dict[str, str]:
    aliases = {
        "file": "filename",
        "file_name": "filename",
        "source": "source_file",
        "date": "measurement_date",
        "measurement_day": "measurement_date",
        "team_name": "team",
        "plot": "plot_id",
        "plotid": "plot_id",
        "rep": "replicate",
    }
    clean: dict[str, str] = {}
    for key, value in row.items():
        normal = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
        clean[aliases.get(normal, normal)] = str(value or "").strip()
    return clean


def load_plot_manifest(path: str | Path | None) -> list[dict[str, str]]:
    """Load an optional filename-to-plot metadata CSV."""
    if not path:
        return []
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"Plot metadata CSV not found: {target}")
    with target.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [_normalise_manifest_row(dict(row)) for row in reader]
    if rows and not any(
        row.get("filename") or row.get("source_file") or row.get("plot_id")
        for row in rows
    ):
        raise ValueError(
            "Plot metadata CSV needs filename, source_file, or plot_id column."
        )
    return rows


def _manifest_match(
    summary: dict[str, Any], rows: Iterable[dict[str, str]]
) -> dict[str, str]:
    filename = str(summary.get("filename") or "")
    source_file = str(summary.get("source_file") or "")
    candidates = {
        filename.casefold(),
        Path(filename).stem.casefold(),
        Path(source_file).name.casefold(),
        Path(source_file).stem.casefold(),
        source_file.casefold(),
    }
    for row in rows:
        identifiers = {
            str(row.get("filename") or "").casefold(),
            Path(str(row.get("filename") or "")).stem.casefold(),
            str(row.get("source_file") or "").casefold(),
            Path(str(row.get("source_file") or "")).name.casefold(),
            str(row.get("plot_id") or "").casefold(),
        }
        identifiers.discard("")
        if candidates & identifiers:
            return row
    return {}


def enrich_summaries(
    summaries: Iterable[dict[str, Any]], team_config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Attach plot metadata and explicit science-ready inclusion flags."""
    manifest_rows = load_plot_manifest(team_config.get("metadata_csv"))
    default_team = str(team_config.get("team_name") or "Unassigned Team").strip()
    default_date = str(team_config.get("measurement_date") or "").strip()
    enriched: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []

    for source in summaries:
        row = dict(source)
        mapping = _manifest_match(row, manifest_rows)
        row["measurement_date"] = (
            mapping.get("measurement_date") or default_date or "Unspecified date"
        )
        row["team"] = mapping.get("team") or default_team
        row["plot_id"] = (
            mapping.get("plot_id")
            or derive_plot_id(str(row.get("filename") or row.get("source_file") or "plot"))
        )
        row["treatment"] = mapping.get("treatment", "")
        row["genotype"] = mapping.get("genotype", "")
        row["replicate"] = mapping.get("replicate", "")

        qc = str(row.get("calibration_qc_status") or "UNASSESSED").upper()
        units = str(row.get("value_units") or "")
        is_reflectance = units.strip().lower() == "reflectance"
        ndvi_median = _finite_number(row.get("ndvi_median"))
        science_ready = is_reflectance and qc == "PASS"
        included = science_ready and ndvi_median is not None
        row["calibration_qc_status"] = qc
        row["included_in_team_statistics"] = included
        q25 = _finite_number(row.get("ndvi_q25"))
        q75 = _finite_number(row.get("ndvi_q75"))
        row["ndvi_iqr"] = q75 - q25 if q25 is not None and q75 is not None else None

        if not is_reflectance:
            warnings.append({
                "measurement_date": row["measurement_date"],
                "team": row["team"],
                "plot_id": row["plot_id"],
                "severity": "EXCLUDED",
                "code": "NO_REFLECTANCE",
                "message": "Calibrated reflectance is unavailable; NDVI is not pooled.",
            })
        elif qc != "PASS":
            warnings.append({
                "measurement_date": row["measurement_date"],
                "team": row["team"],
                "plot_id": row["plot_id"],
                "severity": "EXCLUDED" if qc == "FAIL" else "REVIEW",
                "code": f"CALIBRATION_{qc}",
                "message": "Calibration QC is not PASS; excluded from team statistics.",
            })
        elif ndvi_median is None:
            warnings.append({
                "measurement_date": row["measurement_date"],
                "team": row["team"],
                "plot_id": row["plot_id"],
                "severity": "EXCLUDED",
                "code": "NDVI_UNAVAILABLE",
                "message": "NDVI could not be calculated from the available wavelengths.",
            })
        if not mapping and manifest_rows:
            warnings.append({
                "measurement_date": row["measurement_date"],
                "team": row["team"],
                "plot_id": row["plot_id"],
                "severity": "INFO",
                "code": "METADATA_NOT_MATCHED",
                "message": "No metadata CSV row matched this file; filename-derived plot ID used.",
            })
        enriched.append(row)
    return enriched, warnings


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_tile(path: str | Path | None, size: tuple[int, int]) -> Image.Image | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    try:
        with Image.open(target) as image:
            return ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    except Exception:
        logger.warning("Could not load team-report image: %s", target)
        return None


def _contact_sheet(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    image_key: str,
    title: str,
    columns: int = 3,
) -> bool:
    available = [row for row in rows if row.get(image_key) and Path(row[image_key]).is_file()]
    if not available:
        return False
    cell_w, cell_h = 540, 400
    header_h = 70
    n_rows = math.ceil(len(available) / columns)
    canvas = Image.new("RGB", (cell_w * columns, header_h + cell_h * n_rows), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((22, 20), title, fill="#173F35", font=font)
    for index, row in enumerate(available):
        x = (index % columns) * cell_w
        y = header_h + (index // columns) * cell_h
        image = _load_tile(row.get(image_key), (cell_w - 34, cell_h - 76))
        if image is not None:
            px = x + (cell_w - image.width) // 2
            py = y + 34 + (cell_h - 76 - image.height) // 2
            canvas.paste(image, (px, py))
        qc = str(row.get("calibration_qc_status") or "UNASSESSED")
        ndvi = _finite_number(row.get("ndvi_median"))
        label = f"{row['plot_id']}  |  QC {qc}"
        if ndvi is not None:
            label += f"  |  NDVI median {ndvi:.3f}"
        draw.text((x + 16, y + 10), label, fill="#1E2E29", font=font)
        draw.rectangle((x + 6, y + 3, x + cell_w - 7, y + cell_h - 7), outline="#CBD8D2", width=2)
    canvas.save(output_path, quality=92)
    return True


def _ndvi_comparison(rows: list[dict[str, Any]], output_path: Path) -> bool:
    plotted = [row for row in rows if row.get("included_in_team_statistics")]
    plotted = [row for row in plotted if _finite_number(row.get("ndvi_median")) is not None]
    if not plotted:
        return False
    plotted.sort(key=lambda row: str(row.get("plot_id", "")))
    labels = [str(row["plot_id"]) for row in plotted]
    median = [_finite_number(row["ndvi_median"]) for row in plotted]
    q25 = [_finite_number(row.get("ndvi_q25")) for row in plotted]
    q75 = [_finite_number(row.get("ndvi_q75")) for row in plotted]
    low = [m - q if q is not None else 0.0 for m, q in zip(median, q25)]
    high = [q - m if q is not None else 0.0 for m, q in zip(median, q75)]
    width = max(9.0, min(24.0, 0.48 * len(labels) + 4.0))
    fig, ax = plt.subplots(figsize=(width, 5.6), dpi=150)
    ax.errorbar(
        range(len(labels)), median, yerr=[low, high], fmt="o", color="#247A5A",
        ecolor="#8BBEAA", capsize=3, markersize=5, linewidth=1.2,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("NDVI")
    ax.set_title("Plot-level NDVI median and IQR (calibration QC PASS only)")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(axis="y", color="#DDE7E2", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _expand_cluster_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        total = sum(int(item.get("pixel_count") or 0) for item in row.get("cluster_summary", []))
        for item in row.get("cluster_summary", []):
            count = int(item.get("pixel_count") or 0)
            expanded.append({
                "measurement_date": row["measurement_date"],
                "team": row["team"],
                "plot_id": row["plot_id"],
                "filename": row.get("filename", ""),
                "class_id": item.get("class_id"),
                "class_name": item.get("class_name", ""),
                "pixel_count": count,
                "fraction": count / total if total else None,
            })
    return expanded


def _expand_spectra_rows(
    rows: list[dict[str, Any]], limit: int = _SPECTRA_ROW_LIMIT
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("included_in_team_statistics"):
            continue
        for series in row.get("team_spectra", []):
            wavelengths = series.get("wavelengths") or []
            stats = {key: series.get(key) or [] for key in ("mean", "median", "q25", "q75")}
            for index, wavelength in enumerate(wavelengths):
                if len(expanded) >= limit:
                    return expanded
                expanded.append({
                    "measurement_date": row["measurement_date"],
                    "team": row["team"],
                    "plot_id": row["plot_id"],
                    "filename": row.get("filename", ""),
                    "class_id": series.get("class_id"),
                    "class_name": series.get("class_name", ""),
                    "pixel_count": series.get("pixel_count"),
                    "wavelength_nm": _finite_number(wavelength),
                    "mean": _finite_number(stats["mean"][index]) if index < len(stats["mean"]) else None,
                    "median": _finite_number(stats["median"][index]) if index < len(stats["median"]) else None,
                    "q25": _finite_number(stats["q25"][index]) if index < len(stats["q25"]) else None,
                    "q75": _finite_number(stats["q75"][index]) if index < len(stats["q75"]) else None,
                })
    return expanded


def _find_artifact_runtime() -> tuple[Path, Path] | None:
    node_text = os.environ.get("HYPERSPECTRAL_NODE_EXECUTABLE", "").strip()
    modules_text = os.environ.get("HYPERSPECTRAL_NODE_MODULES", "").strip()
    if node_text and modules_text:
        node, modules = Path(node_text), Path(modules_text)
        if node.is_file() and modules.is_dir():
            return node, modules

    local_modules = Path(__file__).resolve().parents[1] / "node_modules"
    local_node = shutil.which("node")
    if local_node and (local_modules / "@oai" / "artifact-tool").exists():
        return Path(local_node), local_modules

    runtime_root = (
        Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime"
        / "dependencies" / "node"
    )
    node = runtime_root / "bin" / ("node.exe" if os.name == "nt" else "node")
    modules = runtime_root / "node_modules"
    if node.is_file() and (modules / "@oai" / "artifact-tool").exists():
        return node, modules
    return None


def _link_node_modules(link_path: Path, target: Path) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    else:
        link_path.symlink_to(target, target_is_directory=True)


def _create_workbook(payload: dict[str, Any], output_path: Path) -> tuple[bool, str]:
    runtime = _find_artifact_runtime()
    if runtime is None:
        return False, (
            "Excel builder is unavailable. HTML and CSV were saved; configure "
            "HYPERSPECTRAL_NODE_EXECUTABLE and HYPERSPECTRAL_NODE_MODULES for XLSX."
        )
    node, modules = runtime
    builder_source = Path(__file__).resolve().parents[1] / "scripts" / "build_team_workbook.mjs"
    if not builder_source.is_file():
        return False, f"Workbook builder not found: {builder_source}"

    with tempfile.TemporaryDirectory(prefix="hyperspectral_team_xlsx_") as temporary:
        work = Path(temporary)
        shutil.copy2(builder_source, work / builder_source.name)
        _link_node_modules(work / "node_modules", modules)
        payload_path = work / "team_report.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            [str(node), str(work / builder_source.name), str(payload_path), str(output_path)],
            cwd=str(work),
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if result.returncode != 0:
            return False, (result.stderr.strip() or result.stdout.strip())[-2000:]
    return output_path.is_file(), ""


def _relative_href(path: str | Path | None, root: Path) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        try:
            return Path(os.path.relpath(Path(path).resolve(), root.resolve())).as_posix()
        except Exception:
            return ""


def _stage_share_assets(rows: list[dict[str, Any]], package_dir: Path) -> None:
    """Copy visible images and self-contained detail HTML into the share package."""
    images_dir = package_dir / "Images"
    details_dir = package_dir / "Details"
    images_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        stem = f"{index:03d}_{_slug(str(row.get('plot_id') or 'plot'))}"
        for key, suffix in (("overlay_image", "overlay"), ("ndvi_image", "ndvi")):
            source_text = str(row.get(key) or "")
            source = Path(source_text) if source_text else None
            if source is None or not source.is_file():
                continue
            target = images_dir / f"{stem}_{suffix}{source.suffix.lower() or '.png'}"
            shutil.copy2(source, target)
            row[key] = str(target.resolve())
        detail_text = str(row.get("detail_report") or "")
        detail = Path(detail_text) if detail_text else None
        if detail is not None and detail.is_file():
            target = details_dir / f"{stem}_report.html"
            shutil.copy2(detail, target)
            row["detail_report"] = str(target.resolve())


def _render_html(
    rows: list[dict[str, Any]],
    warnings: list[dict[str, str]],
    package_dir: Path,
    *,
    team: str,
    measurement_date: str,
    workbook_available: bool,
    workbook_warning: str,
) -> Path:
    included = [row for row in rows if row.get("included_in_team_statistics")]
    plot_ndvi = [_finite_number(row.get("ndvi_median")) for row in included]
    plot_ndvi = [value for value in plot_ndvi if value is not None]
    mean_ndvi = sum(plot_ndvi) / len(plot_ndvi) if plot_ndvi else None
    summary_rows = []
    cards = []
    for row in sorted(rows, key=lambda item: str(item.get("plot_id", ""))):
        included_text = "Yes" if row.get("included_in_team_statistics") else "No"
        ndvi = _finite_number(row.get("ndvi_median"))
        veg = _finite_number(row.get("vegetation_fraction"))
        detail = _relative_href(row.get("detail_report"), package_dir)
        detail_link = f'<a href="{html.escape(detail)}">open</a>' if detail else ""
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['plot_id']))}</td>"
            f"<td>{html.escape(str(row.get('calibration_qc_status', '')))}</td>"
            f"<td>{included_text}</td>"
            f"<td>{'' if ndvi is None else f'{ndvi:.4f}'}</td>"
            f"<td>{'' if veg is None else f'{100 * veg:.1f}%'}</td>"
            f"<td>{html.escape(str(row.get('n_classes', '')))}</td>"
            f"<td>{detail_link}</td></tr>"
        )
        plot_images = []
        for key, label in (("overlay_image", "RGB + clusters"), ("ndvi_image", "NDVI (-1 to 1)")):
            href = _relative_href(row.get(key), package_dir)
            if href:
                plot_images.append(
                    f'<figure><img src="{html.escape(href)}" alt="{html.escape(label)}">'
                    f'<figcaption>{html.escape(label)}</figcaption></figure>'
                )
        cards.append(
            f'<section class="plot-card"><h3>{html.escape(str(row["plot_id"]))}</h3>'
            f'<p>Calibration QC: <b>{html.escape(str(row.get("calibration_qc_status", "")))}</b>'
            f' · Included: <b>{included_text}</b>'
            + (f' · NDVI median: <b>{ndvi:.4f}</b>' if ndvi is not None else "")
            + f'</p><div class="image-grid">{"".join(plot_images)}</div></section>'
        )

    warning_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['plot_id'])}</td>"
        f"<td>{html.escape(item['severity'])}</td>"
        f"<td>{html.escape(item['code'])}</td>"
        f"<td>{html.escape(item['message'])}</td></tr>"
        for item in warnings
    ) or '<tr><td colspan="4">No warnings</td></tr>'
    workbook_link = (
        '<a class="button" href="Field_Results.xlsx">Download Field_Results.xlsx</a>'
        if workbook_available
        else '<span class="warning">XLSX unavailable; use Field_Summary.csv.</span>'
    )
    if workbook_warning:
        workbook_link += f'<p class="warning">{html.escape(workbook_warning)}</p>'
    overview = "plots_overview.png" if (package_dir / "plots_overview.png").is_file() else ""
    ndvi_sheet = "plots_ndvi.png" if (package_dir / "plots_ndvi.png").is_file() else ""
    ndvi_compare = "plot_ndvi_comparison.png" if (package_dir / "plot_ndvi_comparison.png").is_file() else ""
    combined_images = "".join(
        f'<figure><img src="{name}" alt="{label}"><figcaption>{label}</figcaption></figure>'
        for name, label in (
            (overview, "All plots: RGB + clusters"),
            (ndvi_sheet, "All plots: NDVI with a common -1 to 1 scale"),
            (ndvi_compare, "Plot-level NDVI median and IQR"),
        )
        if name
    )

    report_path = package_dir / "Team_Report.html"
    report_path.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(team)} daily field report</title>
<style>*{{box-sizing:border-box}}body{{font-family:Segoe UI,Arial,sans-serif;background:#f3f6f4;color:#20332b;margin:0}}
.container{{max-width:1800px;margin:auto;padding:22px}}header{{background:linear-gradient(135deg,#173f35,#2d8064);color:white;padding:28px;border-radius:12px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}}.kpi,.card,.plot-card{{background:white;border-radius:10px;padding:18px;box-shadow:0 2px 8px #0001}}
.kpi b{{display:block;font-size:25px;color:#176848}}table{{width:100%;border-collapse:collapse}}th{{background:#173f35;color:#fff;text-align:left;padding:9px}}td{{padding:8px;border-bottom:1px solid #e2e9e5}}
.image-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}figure{{margin:0;text-align:center}}img{{width:100%;height:auto;border:1px solid #d5dfda;border-radius:6px;cursor:zoom-in}}figcaption{{font-size:12px;color:#65766e;margin-top:5px}}
.plot-card{{margin-top:18px}}.button{{display:inline-block;background:#176848;color:#fff;text-decoration:none;padding:10px 15px;border-radius:7px}}.warning{{color:#a33}}
.modal{{display:none;position:fixed;z-index:9999;inset:0;background:#000d;align-items:center;justify-content:center}}.modal img{{max-width:96vw;max-height:92vh;width:auto;background:white}}
@media(max-width:900px){{.kpis,.image-grid{{grid-template-columns:1fr}}}}</style></head><body><div class="container">
<header><h1>{html.escape(team)} · {html.escape(measurement_date)}</h1><p>Plot-level team report · only calibrated reflectance with QC PASS is pooled.</p></header>
<div class="kpis"><div class="kpi"><b>{len(rows)}</b>Total plots</div><div class="kpi"><b>{len(included)}</b>Included</div><div class="kpi"><b>{len(rows)-len(included)}</b>Review/excluded</div><div class="kpi"><b>{'—' if mean_ndvi is None else f'{mean_ndvi:.3f}'}</b>Mean of plot NDVI medians</div></div>
<div class="card"><h2>Downloads</h2>{workbook_link} <a class="button" href="Field_Summary.csv">Field_Summary.csv</a></div>
<div class="card"><h2>Daily visual comparison</h2><div class="image-grid">{combined_images}</div></div>
<div class="card"><h2>Plot summary</h2><table><thead><tr><th>Plot</th><th>Calibration QC</th><th>Included</th><th>NDVI median</th><th>Vegetation</th><th>Classes</th><th>Detail</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div>
<div class="card"><h2>Warnings</h2><table><thead><tr><th>Plot</th><th>Severity</th><th>Code</th><th>Message</th></tr></thead><tbody>{warning_rows}</tbody></table></div>
<h2>Individual plots</h2>{''.join(cards)}</div><div id="modal" class="modal"><img alt="Expanded plot image"></div>
<script>const m=document.getElementById('modal'),mi=m.querySelector('img');document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{{mi.src=i.src;m.style.display='flex'}});m.onclick=()=>m.style.display='none';document.addEventListener('keydown',e=>{{if(e.key==='Escape')m.style.display='none'}});</script></body></html>""",
        encoding="utf-8",
    )
    return report_path


def generate_team_daily_packages(
    summaries: Iterable[dict[str, Any]],
    output_root: str | Path,
    team_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create one compact delivery package per measurement-date/team group."""
    enriched, warnings = enrich_summaries(summaries, team_config)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in enriched:
        grouped.setdefault((str(row["measurement_date"]), str(row["team"])), []).append(row)

    root = Path(output_root).expanduser().resolve() / "team_reports"
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    packages: list[dict[str, Any]] = []
    for (measurement_date, team), rows in grouped.items():
        package_dir = root / f"{_slug(measurement_date)}_{_slug(team)}_{stamp}"
        package_dir.mkdir(parents=True, exist_ok=False)
        _stage_share_assets(rows, package_dir)
        row_warnings = [
            item for item in warnings
            if item["measurement_date"] == measurement_date and item["team"] == team
        ]
        _write_csv(package_dir / "Field_Summary.csv", rows, _SUMMARY_COLUMNS)
        _contact_sheet(
            rows, package_dir / "plots_overview.png", image_key="overlay_image",
            title=f"{team} · {measurement_date} · RGB + cluster overlay",
        )
        _contact_sheet(
            rows, package_dir / "plots_ndvi.png", image_key="ndvi_image",
            title=f"{team} · {measurement_date} · NDVI (common scale -1 to 1)",
        )
        _ndvi_comparison(rows, package_dir / "plot_ndvi_comparison.png")

        cluster_rows = _expand_cluster_rows(rows)
        spectra_rows = _expand_spectra_rows(rows)
        if sum(
            len(series.get("wavelengths") or [])
            for row in rows if row.get("included_in_team_statistics")
            for series in row.get("team_spectra", [])
        ) > len(spectra_rows):
            row_warnings.append({
                "measurement_date": measurement_date,
                "team": team,
                "plot_id": "ALL",
                "severity": "INFO",
                "code": "EXCEL_ROW_LIMIT",
                "message": (
                    "Reflectance_Spectra was capped at "
                    f"{_SPECTRA_ROW_LIMIT:,} rows to bound memory use."
                ),
            })
        _write_csv(
            package_dir / "Warnings.csv",
            row_warnings,
            ["measurement_date", "team", "plot_id", "severity", "code", "message"],
        )
        payload = {
            "meta": {
                "team": team,
                "measurement_date": measurement_date,
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "inclusion_rule": "value_units=reflectance AND calibration_qc_status=PASS AND NDVI available",
            },
            "summaries": [{key: row.get(key) for key in _SUMMARY_COLUMNS} for row in rows],
            "cluster_rows": cluster_rows,
            "spectra_rows": spectra_rows,
            "warnings": row_warnings,
        }
        workbook_path = package_dir / "Field_Results.xlsx"
        workbook_available, workbook_warning = _create_workbook(payload, workbook_path)
        report_path = _render_html(
            rows,
            row_warnings,
            package_dir,
            team=team,
            measurement_date=measurement_date,
            workbook_available=workbook_available,
            workbook_warning=workbook_warning,
        )
        packages.append({
            "measurement_date": measurement_date,
            "team": team,
            "directory": str(package_dir),
            "report": str(report_path),
            "workbook": str(workbook_path) if workbook_available else "",
            "summary_csv": str(package_dir / "Field_Summary.csv"),
            "workbook_warning": workbook_warning,
        })
        logger.info("Team/day package saved: %s", package_dir)
    return packages
