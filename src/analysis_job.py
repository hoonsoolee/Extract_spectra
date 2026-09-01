"""Background-process support for long Streamlit analysis runs.

Streamlit executes a page synchronously.  A normal ``Pipeline.run`` call blocks
the same session that must receive a Stop button click.  This module launches
the pipeline in a separate Python process and exchanges small JSON status files
with the web UI, keeping the page responsive without loading image data twice.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


ACTIVE_STATES = {"queued", "running", "cancelling"}
TERMINAL_STATES = {"completed", "failed", "cancelled"}


def _json_default(value: Any) -> Any:
    """Convert common scientific/path values to JSON-safe primitives."""
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


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON without exposing readers to a partially written file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def launch_analysis_job(
    config: dict[str, Any],
    *,
    labels_csv: str | None,
    file_limit: int | None,
    single_file: str | None,
    timing: dict[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    """Launch one pipeline worker and return serializable job metadata."""
    root = Path(project_root).resolve()
    job_id = uuid.uuid4().hex
    jobs_root = Path(tempfile.gettempdir()) / "hyperspectral_analysis_jobs"
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    spec_path = job_dir / "spec.json"
    status_path = job_dir / "status.json"
    result_path = job_dir / "result.json"
    log_path = job_dir / "analysis.log"
    cancel_path = job_dir / "cancel.requested"
    launched_at = time.time()

    spec = {
        "job_id": job_id,
        "config": config,
        "labels_csv": labels_csv,
        "file_limit": file_limit,
        "single_file": single_file,
        "timing": timing,
        "launched_at": launched_at,
        "status_path": str(status_path),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "cancel_path": str(cancel_path),
    }
    write_json_atomic(spec_path, spec)
    write_json_atomic(
        status_path,
        {
            "job_id": job_id,
            "state": "queued",
            "message": "분석 작업을 준비하고 있습니다.",
            "launched_at": launched_at,
        },
    )

    creationflags = 0
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True

    command = [
        sys.executable,
        "-m",
        "src.analysis_worker",
        "--spec",
        str(spec_path),
    ]
    worker_environment = os.environ.copy()
    worker_environment["PYTHONUTF8"] = "1"
    worker_environment["PYTHONIOENCODING"] = "utf-8"
    try:
        with log_path.open("ab") as log_stream:
            process = subprocess.Popen(
                command,
                cwd=str(root),
                env=worker_environment,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                close_fds=True,
                **popen_kwargs,
            )
    except Exception:
        # A spec can contain a GitHub token, so never leave it behind when the
        # worker could not start and therefore could not delete it itself.
        spec_path.unlink(missing_ok=True)
        raise

    job = {
        "job_id": job_id,
        "pid": int(process.pid),
        "job_dir": str(job_dir),
        "status_path": str(status_path),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "cancel_path": str(cancel_path),
        "spec_path": str(spec_path),
        "launched_at": launched_at,
        "estimated_seconds": timing.get("estimated_seconds"),
        "result_applied": False,
    }
    return job


def poll_analysis_job(job: dict[str, Any] | None) -> dict[str, Any]:
    """Read current job state and detect workers that exited unexpectedly."""
    if not job:
        return {"state": "idle"}
    state = read_json(job.get("status_path", ""))
    if not state:
        state = {
            "job_id": job.get("job_id"),
            "state": "queued",
            "launched_at": job.get("launched_at"),
        }

    if state.get("state") in ACTIVE_STATES:
        pid = int(state.get("pid") or job.get("pid") or 0)
        # Give the worker a short startup window before treating a missing PID
        # as a crash.
        age = time.time() - float(job.get("launched_at") or time.time())
        if age > 2.0 and not _process_exists(pid):
            state = {
                **state,
                "state": "failed",
                "finished_at": time.time(),
                "message": "분석 작업 프로세스가 예기치 않게 종료되었습니다.",
            }
            write_json_atomic(job["status_path"], state)
    return state


def cancel_analysis_job(job: dict[str, Any]) -> dict[str, Any]:
    """Stop exactly the worker process belonging to ``job`` and its children."""
    state = poll_analysis_job(job)
    if state.get("state") not in ACTIVE_STATES:
        return state

    Path(job["cancel_path"]).touch()
    cancelling = {
        **state,
        "state": "cancelling",
        "message": "사용자 요청으로 분석을 중지하고 있습니다.",
        "cancel_requested_at": time.time(),
    }
    write_json_atomic(job["status_path"], cancelling)

    pid = int(state.get("pid") or job.get("pid") or 0)
    if pid > 0 and _process_exists(pid):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    # Do not overwrite a completion record if the worker finished in the tiny
    # interval between the state check and the termination request.
    latest = read_json(job["status_path"])
    if latest.get("state") in {"completed", "failed"}:
        return latest

    cancelled = {
        **cancelling,
        "state": "cancelled",
        "message": "사용자가 분석을 중지했습니다.",
        "finished_at": time.time(),
    }
    write_json_atomic(job["status_path"], cancelled)
    if job.get("spec_path"):
        Path(job["spec_path"]).unlink(missing_ok=True)
    return cancelled


def read_job_log(job: dict[str, Any] | None, max_lines: int = 160) -> list[str]:
    """Return the most recent worker log lines without loading a huge log."""
    if not job:
        return []
    path = Path(job.get("log_path", ""))
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max(1, int(max_lines)):]


def read_analysis_result(job: dict[str, Any] | None) -> dict[str, Any]:
    if not job:
        return {}
    return read_json(job.get("result_path", ""))
