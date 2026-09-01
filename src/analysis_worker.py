"""Command-line worker used by :mod:`src.analysis_job`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from .analysis_job import read_json, write_json_atomic


def _write_status(spec: dict, state: str, message: str, **extra) -> None:
    payload = {
        "job_id": spec.get("job_id"),
        "state": state,
        "message": message,
        "pid": os.getpid(),
        "launched_at": spec.get("launched_at"),
        **extra,
    }
    write_json_atomic(spec["status_path"], payload)


def run_worker(spec_path: str | Path) -> int:
    path = Path(spec_path)
    spec = read_json(path)
    if not spec:
        raise ValueError(f"Invalid or missing analysis job spec: {path}")

    # The config may contain an access token.  Keep it in memory only after the
    # worker starts instead of leaving it in the temporary spec file.
    path.unlink(missing_ok=True)
    cancel_path = Path(spec["cancel_path"])
    started_at = time.time()
    _write_status(
        spec,
        "running",
        "초분광 파일을 분석하고 있습니다.",
        started_at=started_at,
    )

    logging.basicConfig(
        level=logging.DEBUG if spec.get("timing", {}).get("verbose") else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    try:
        if cancel_path.exists():
            _write_status(
                spec,
                "cancelled",
                "분석을 시작하기 전에 중지되었습니다.",
                started_at=started_at,
                finished_at=time.time(),
            )
            return 2

        from .pipeline import Pipeline

        pipeline = Pipeline(spec["config"])
        pipeline.run(
            labels_csv=spec.get("labels_csv") or None,
            file_limit=spec.get("file_limit") or None,
            single_file=spec.get("single_file") or None,
        )
        elapsed = time.time() - started_at
        output_dir = Path(spec["config"].get("output", {}).get("dir", "./output"))
        launched_at = float(spec.get("launched_at") or started_at)
        reports = sorted(
            (
                item.resolve()
                for item in output_dir.rglob("*report*.html")
                if item.is_file() and item.stat().st_mtime >= launched_at - 1.0
            ),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        review_dirs = [
            str(Path(summary["result_dir"]).resolve())
            for summary in pipeline.batch_summaries
            if summary.get("result_dir") and Path(summary["result_dir"]).is_dir()
        ]
        timing_input = spec.get("timing", {})
        timing_record = {
            "method": timing_input.get("method", ""),
            "work_units": float(timing_input.get("work_units") or 0.0),
            "elapsed_seconds": elapsed,
            "file_count": int(
                timing_input.get("file_count") or len(pipeline.batch_summaries)
            ),
            "estimated_seconds": timing_input.get("estimated_seconds"),
        }
        result = {
            "timing_record": timing_record,
            "reports": [str(item) for item in reports],
            "output_dir": str(output_dir.resolve()),
            "review_dirs": review_dirs,
            "team_packages": list(getattr(pipeline, "team_packages", [])),
            "batch_summaries": list(pipeline.batch_summaries),
        }
        write_json_atomic(spec["result_path"], result)
        _write_status(
            spec,
            "completed",
            "분석이 완료되었습니다.",
            started_at=started_at,
            finished_at=time.time(),
            elapsed_seconds=elapsed,
        )
        return 0
    except Exception:
        error_text = traceback.format_exc()
        logging.error("Analysis worker failed\n%s", error_text)
        _write_status(
            spec,
            "failed",
            "분석 중 오류가 발생했습니다.",
            started_at=started_at,
            finished_at=time.time(),
            error=error_text,
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    return run_worker(args.spec)


if __name__ == "__main__":
    raise SystemExit(main())
