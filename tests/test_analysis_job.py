import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis_job import (
    cancel_analysis_job,
    poll_analysis_job,
    read_analysis_result,
    read_json,
    write_json_atomic,
)
from src.analysis_worker import run_worker


class AnalysisJobTests(unittest.TestCase):
    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            write_json_atomic(path, {"state": "running", "value": 3})
            self.assertEqual(read_json(path)["value"], 3)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_poll_marks_an_exited_worker_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "status.json"
            write_json_atomic(
                status_path,
                {"state": "running", "pid": 999_999_999},
            )
            job = {
                "status_path": str(status_path),
                "pid": 999_999_999,
                "launched_at": time.time() - 10,
            }
            state = poll_analysis_job(job)
            self.assertEqual(state["state"], "failed")

    def test_cancel_records_cancelled_when_worker_has_not_started(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "status.json"
            cancel_path = root / "cancel.requested"
            spec_path = root / "spec.json"
            spec_path.write_text("{}", encoding="utf-8")
            write_json_atomic(status_path, {"state": "queued"})
            job = {
                "status_path": str(status_path),
                "cancel_path": str(cancel_path),
                "spec_path": str(spec_path),
                "pid": 999_999_999,
                "launched_at": time.time(),
            }
            state = cancel_analysis_job(job)
            self.assertEqual(state["state"], "cancelled")
            self.assertTrue(cancel_path.exists())
            self.assertFalse(spec_path.exists())

    def test_cancel_terminates_the_owned_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = (
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                **kwargs,
            )
            status_path = root / "status.json"
            cancel_path = root / "cancel.requested"
            spec_path = root / "spec.json"
            spec_path.write_text("{}", encoding="utf-8")
            write_json_atomic(
                status_path,
                {"state": "running", "pid": process.pid},
            )
            job = {
                "status_path": str(status_path),
                "cancel_path": str(cancel_path),
                "spec_path": str(spec_path),
                "pid": process.pid,
                "launched_at": time.time(),
            }
            try:
                state = cancel_analysis_job(job)
                return_code = process.wait(timeout=8)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

            self.assertEqual(state["state"], "cancelled")
            self.assertNotEqual(return_code, 0)

    def test_worker_returns_existing_result_contract(self):
        class FakePipeline:
            def __init__(self, config):
                self.config = config
                self.batch_summaries = []
                self.team_packages = []

            def run(self, **_kwargs):
                result_dir = Path(self.config["output"]["dir"]) / "sample"
                result_dir.mkdir(parents=True, exist_ok=True)
                report = result_dir / "report_test.html"
                report.write_text("<html></html>", encoding="utf-8")
                self.batch_summaries = [{"result_dir": str(result_dir)}]
                self.team_packages = [{"directory": str(result_dir)}]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "output"
            spec_path = root / "spec.json"
            status_path = root / "status.json"
            result_path = root / "result.json"
            spec = {
                "job_id": "test-job",
                "config": {"output": {"dir": str(output_dir)}},
                "labels_csv": None,
                "file_limit": 1,
                "single_file": None,
                "timing": {
                    "method": "kmeans",
                    "work_units": 2.5,
                    "file_count": 1,
                    "estimated_seconds": 12.0,
                },
                "launched_at": time.time(),
                "status_path": str(status_path),
                "result_path": str(result_path),
                "log_path": str(root / "analysis.log"),
                "cancel_path": str(root / "cancel.requested"),
            }
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            with patch("src.pipeline.Pipeline", FakePipeline):
                exit_code = run_worker(spec_path)

            self.assertEqual(exit_code, 0)
            self.assertFalse(spec_path.exists())
            self.assertEqual(read_json(status_path)["state"], "completed")
            result = read_analysis_result({"result_path": str(result_path)})
            self.assertEqual(result["timing_record"]["method"], "kmeans")
            self.assertEqual(len(result["reports"]), 1)
            self.assertEqual(len(result["review_dirs"]), 1)


if __name__ == "__main__":
    unittest.main()
