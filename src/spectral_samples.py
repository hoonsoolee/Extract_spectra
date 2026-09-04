"""Reproducible export of actual pixel spectra for later model development.

The summary CSVs intentionally contain class-level statistics.  This module
adds a compact, grouped HDF5 product that preserves a bounded number of the
actual spectra behind those summaries.  Samples remain nested inside their
source image/plot; downstream code must not treat pixels sharing one plot
label as independent biological replicates.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np


SCHEMA_VERSION = "1.0"
_HYBRID_BASE_NAMES = {
    -1: "Unavailable",
    0: "Background",
    1: "Sunlit Leaf",
    2: "Shadow Leaf",
    3: "Soil",
    4: "Other",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    return str(value)


def _wavelength_axis(values: Optional[Iterable[float]], bands: int) -> np.ndarray:
    if values is None:
        return np.arange(bands, dtype=np.float64)
    axis = np.asarray(list(values), dtype=np.float64)
    if axis.shape != (bands,):
        return np.arange(bands, dtype=np.float64)
    return axis


def _dataset_options(rows: int, bands: int) -> dict[str, Any]:
    """Compression/chunk options that also work for small sample groups."""
    if rows <= 0 or bands <= 0:
        return {}
    return {
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
        "chunks": (min(256, rows), bands),
    }


def _unique_class_ids(labels: np.ndarray, block_size: int = 1_000_000) -> list[int]:
    """Find class IDs without asking ``np.unique`` to copy a huge map."""
    values: set[int] = set()
    for start in range(0, labels.size, block_size):
        values.update(int(item) for item in np.unique(labels[start:start + block_size]))
    return sorted(values)


def _sample_class_positions(
    labels: np.ndarray,
    class_id: int,
    limit: int,
    rng: np.random.Generator,
    block_size: int = 1_000_000,
) -> tuple[np.ndarray, int]:
    """Uniformly sample matching flat indices with bounded temporary memory."""
    population = 0
    for start in range(0, labels.size, block_size):
        population += int(
            np.count_nonzero(labels[start:start + block_size] == class_id)
        )
    if population == 0:
        return np.empty(0, dtype=np.int64), 0

    take = min(population, limit)
    if take == population:
        selected_ranks = np.arange(population, dtype=np.int64)
    else:
        selected_ranks = np.sort(
            rng.choice(population, size=take, replace=False).astype(np.int64)
        )

    chosen = np.empty(take, dtype=np.int64)
    seen = 0
    for start in range(0, labels.size, block_size):
        block = labels[start:start + block_size]
        local_positions = np.flatnonzero(block == class_id)
        block_count = int(len(local_positions))
        if block_count:
            left = int(np.searchsorted(selected_ranks, seen, side="left"))
            right = int(
                np.searchsorted(selected_ranks, seen + block_count, side="left")
            )
            if right > left:
                local_ranks = selected_ranks[left:right] - seen
                chosen[left:right] = start + local_positions[local_ranks]
            seen += block_count
        if seen >= population:
            break
    return chosen, population


def export_spectral_samples(
    output_path: str | Path,
    *,
    analysis_data: np.ndarray,
    raw_data: Optional[np.ndarray],
    class_map: np.ndarray,
    class_info: Iterable[Mapping[str, Any]],
    analysis_wavelengths: Optional[Iterable[float]] = None,
    raw_wavelengths: Optional[Iterable[float]] = None,
    base_class_map: Optional[np.ndarray] = None,
    max_per_class: int = 1_000,
    random_state: int = 42,
    spatial_downsample: int = 1,
    value_units: str = "processed",
    save_raw: bool = True,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Save uniformly sampled real spectra from every final class.

    Sampling is deterministic for a fixed class map and random seed.  Each
    selected row retains its final cluster, optional Hybrid base class, pixel
    coordinate and inverse-probability weight.  The latter allows downstream
    aggregation to recover the original class proportions when classes are
    capped at different sample counts.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError("h5py is required to export spectral sample sets") from exc

    analysis = np.asarray(analysis_data)
    labels = np.asarray(class_map)
    if analysis.ndim != 3:
        raise ValueError(f"analysis_data must be H x W x bands, got {analysis.shape}")
    if labels.shape != analysis.shape[:2]:
        raise ValueError(
            f"class_map shape {labels.shape} does not match data {analysis.shape[:2]}"
        )
    raw = None if raw_data is None else np.asarray(raw_data)
    if raw is not None and (raw.ndim != 3 or raw.shape[:2] != analysis.shape[:2]):
        raise ValueError("raw_data must share the analysis_data spatial dimensions")
    base = None if base_class_map is None else np.asarray(base_class_map)
    if base is not None and base.shape != labels.shape:
        raise ValueError("base_class_map must match class_map")
    limit = max(1, int(max_per_class))
    factor = max(1, int(spatial_downsample))

    analysis_flat = analysis.reshape(-1, analysis.shape[2])
    raw_flat = raw.reshape(-1, raw.shape[2]) if raw is not None else None
    label_flat = labels.reshape(-1)
    base_flat = base.reshape(-1) if base is not None else None
    width = int(analysis.shape[1])

    selected_blocks: list[np.ndarray] = []
    final_ids: list[np.ndarray] = []
    base_ids: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    class_rows: list[dict[str, Any]] = []

    info_by_id = {int(item["id"]): dict(item) for item in class_info}
    for class_id in _unique_class_ids(label_flat):
        seed = np.random.SeedSequence([int(random_state), class_id & 0xFFFFFFFF])
        rng = np.random.default_rng(seed)
        chosen, population = _sample_class_positions(
            label_flat, class_id, limit, rng
        )
        if population == 0:
            continue

        # Invalid pixels cannot support a scientific model.  Keep the original
        # population count and record the actual retained count for audit.
        finite = np.all(np.isfinite(analysis_flat[chosen]), axis=1)
        if raw_flat is not None and save_raw:
            finite &= np.all(np.isfinite(raw_flat[chosen]), axis=1)
        chosen = chosen[finite]
        if not len(chosen):
            continue

        selected_blocks.append(chosen.astype(np.int64, copy=False))
        final_ids.append(np.full(len(chosen), class_id, dtype=np.int32))
        if base_flat is None:
            base_values = np.full(len(chosen), -1, dtype=np.int16)
        else:
            base_values = np.asarray(base_flat[chosen], dtype=np.int16)
        base_ids.append(base_values)
        sample_weight = population / float(len(chosen))
        weights.append(np.full(len(chosen), sample_weight, dtype=np.float32))

        info = info_by_id.get(class_id, {})
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": str(info.get("name", f"Cluster {class_id}")),
                "population_count": population,
                "sampled_count": int(len(chosen)),
                "sample_weight": sample_weight,
            }
        )

    if not selected_blocks:
        raise ValueError("No finite classified spectra were available for export")

    selected = np.concatenate(selected_blocks)
    class_ids = np.concatenate(final_ids)
    hybrid_base_ids = np.concatenate(base_ids)
    sample_weights = np.concatenate(weights)
    rows = (selected // width).astype(np.int32)
    cols = (selected % width).astype(np.int32)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        with h5py.File(partial, "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            handle.attrs["value_units"] = str(value_units)
            handle.attrs["sampling_strategy"] = "uniform_random_without_replacement_per_final_class"
            handle.attrs["max_per_class"] = limit
            handle.attrs["random_state"] = int(random_state)
            handle.attrs["spatial_downsample"] = factor
            handle.attrs["n_plot_groups"] = 1
            handle.attrs["n_samples"] = int(len(selected))
            handle.attrs["statistical_unit_note"] = (
                "Rows are nested pixel observations from one source image/plot, "
                "not independent plot-level labels."
            )
            handle.attrs["provenance_json"] = json.dumps(
                dict(provenance or {}),
                ensure_ascii=False,
                default=_json_default,
            )

            handle.create_dataset(
                "analysis_values",
                data=np.asarray(analysis_flat[selected], dtype=np.float32),
                **_dataset_options(len(selected), analysis.shape[2]),
            )
            handle.create_dataset(
                "analysis_wavelength_nm",
                data=_wavelength_axis(analysis_wavelengths, analysis.shape[2]),
            )
            if raw_flat is not None and save_raw:
                handle.create_dataset(
                    "raw_values",
                    data=np.asarray(raw_flat[selected], dtype=np.float32),
                    **_dataset_options(len(selected), raw.shape[2]),
                )
                handle.create_dataset(
                    "raw_wavelength_nm",
                    data=_wavelength_axis(raw_wavelengths, raw.shape[2]),
                )

            handle.create_dataset("class_id", data=class_ids)
            handle.create_dataset("base_class_id", data=hybrid_base_ids)
            handle.create_dataset("row", data=rows)
            handle.create_dataset("column", data=cols)
            handle.create_dataset("source_row", data=rows.astype(np.int64) * factor)
            handle.create_dataset("source_column", data=cols.astype(np.int64) * factor)
            handle.create_dataset("pixel_flat_index", data=selected)
            handle.create_dataset("sample_weight", data=sample_weights)

            classes = handle.create_group("classes")
            classes.create_dataset(
                "class_id",
                data=np.asarray([item["class_id"] for item in class_rows], dtype=np.int32),
            )
            classes.create_dataset(
                "class_name",
                data=np.asarray([item["class_name"] for item in class_rows], dtype=object),
                dtype=string_dtype,
            )
            classes.create_dataset(
                "population_count",
                data=np.asarray([item["population_count"] for item in class_rows], dtype=np.int64),
            )
            classes.create_dataset(
                "sampled_count",
                data=np.asarray([item["sampled_count"] for item in class_rows], dtype=np.int32),
            )
            classes.create_dataset(
                "sample_weight",
                data=np.asarray([item["sample_weight"] for item in class_rows], dtype=np.float32),
            )
            base_classes = handle.create_group("hybrid_base_classes")
            base_classes.create_dataset(
                "class_id", data=np.asarray(sorted(_HYBRID_BASE_NAMES), dtype=np.int16)
            )
            base_classes.create_dataset(
                "class_name",
                data=np.asarray(
                    [_HYBRID_BASE_NAMES[key] for key in sorted(_HYBRID_BASE_NAMES)],
                    dtype=object,
                ),
                dtype=string_dtype,
            )
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    os.replace(partial, target)
    return {
        "file": str(target.resolve()),
        "schema_version": SCHEMA_VERSION,
        "n_samples": int(len(selected)),
        "max_per_class": limit,
        "n_classes": len(class_rows),
        "value_units": str(value_units),
        "raw_values_saved": bool(raw_flat is not None and save_raw),
        "sampling_strategy": "uniform_random_without_replacement_per_final_class",
        "classes": class_rows,
    }
