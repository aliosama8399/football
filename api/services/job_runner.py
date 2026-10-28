"""
Background Pipeline Execution & Autonomous Scheduler Service
=============================================================
Thread-safe background execution manager and autonomous scheduler for the
Football Data-to-Fine-Tuning & Continuous Evaluation pipeline.

Capabilities:
  1. Full 8-stage pipeline execution (Ingest -> Preprocess -> Tactics/Store Sync
     -> Teacher LLM -> Split -> GNN Train -> ONNX Export -> API Hot-Reload).
  2. Standalone Continuous Scrape + Store Sync + Evaluation job (`trigger_eval_job`).
  3. Startup Auto-Bootstrap: automatically exports missing ONNX models or builds
     missing processed datasets/tactics in the background when the server starts.
  4. Autonomous Periodic Scheduler: runs background incremental ingestion, knowledge
     store synchronization, model hot-reload, and evaluation on a configurable interval
     without requiring manual CLI intervention.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class JobRunRecord:
    job_id: str
    status: str  # IDLE, RUNNING, SUCCESS, FAILED, CANCELLED
    step: int
    step_name: str
    progress_pct: int
    started_at: str
    ended_at: Optional[str] = None
    duration_s: float = 0.0
    error: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)


class PipelineJobManager:
    """Singleton manager tracking and executing pipeline background jobs and periodic schedules."""

    _instance: Optional["PipelineJobManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PipelineJobManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.state_lock = threading.Lock()
        self.active_job: Optional[JobRunRecord] = None
        self.job_history: deque[JobRunRecord] = deque(maxlen=30)
        self.active_process: Optional[subprocess.Popen] = None
        self.active_thread: Optional[threading.Thread] = None
        self.log_buffer: deque[str] = deque(maxlen=1000)

        # Reference to FastAPI app for direct in-process singleton hot-reloads
        self.app_ref: Any = None

        # Autonomous Background Scheduler state
        env_enabled = os.environ.get("AUTO_SCHEDULER_ENABLED", "true").strip().lower()
        self.scheduler_enabled: bool = env_enabled in ("1", "true", "yes", "on")
        try:
            self.scheduler_interval_hours: float = max(
                0.5, float(os.environ.get("AUTO_SCHEDULER_INTERVAL_HOURS", "24"))
            )
        except ValueError:
            self.scheduler_interval_hours = 24.0
        self.last_scheduled_run: Optional[str] = None
        self.next_scheduled_run: Optional[str] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_stop_event = threading.Event()

    def start_auto_scheduler(self, app: Any = None) -> None:
        """
        Attaches the FastAPI app reference, runs non-blocking startup auto-bootstrap
        checks, and starts the periodic background scheduler daemon thread.
        """
        if app is not None:
            self.app_ref = app

        # Launch startup self-healing bootstrap check in background
        bootstrap_thread = threading.Thread(
            target=self._startup_auto_bootstrap,
            daemon=True,
            name="StartupAutoBootstrap",
        )
        bootstrap_thread.start()

        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            return

        self._scheduler_stop_event.clear()
        if self.scheduler_enabled:
            self.next_scheduled_run = (
                datetime.now() + timedelta(hours=self.scheduler_interval_hours)
            ).isoformat()

        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="AutonomousPipelineScheduler",
        )
        self._scheduler_thread.start()
        logger.info(
            "Autonomous MLOps Scheduler initialized (enabled=%s, interval=%.1fh)",
            self.scheduler_enabled,
            self.scheduler_interval_hours,
        )

    def stop_auto_scheduler(self) -> None:
        """Gracefully stops the background scheduler daemon."""
        self._scheduler_stop_event.set()

    def get_scheduler_status(self) -> Dict[str, Any]:
        with self.state_lock:
            return {
                "enabled": self.scheduler_enabled,
                "interval_hours": self.scheduler_interval_hours,
                "last_scheduled_run": self.last_scheduled_run,
                "next_scheduled_run": self.next_scheduled_run if self.scheduler_enabled else None,
                "is_job_running": self.active_job is not None and self.active_job.status == "RUNNING",
            }

    def configure_scheduler(
        self, enabled: Optional[bool] = None, interval_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        with self.state_lock:
            if enabled is not None:
                self.scheduler_enabled = bool(enabled)
            if interval_hours is not None:
                self.scheduler_interval_hours = max(0.5, float(interval_hours))
            if self.scheduler_enabled:
                self.next_scheduled_run = (
                    datetime.now() + timedelta(hours=self.scheduler_interval_hours)
                ).isoformat()
            else:
                self.next_scheduled_run = None
        logger.info(
            "Updated Autonomous Scheduler config: enabled=%s, interval=%.1fh",
            self.scheduler_enabled,
            self.scheduler_interval_hours,
        )
        return self.get_scheduler_status()

    def _startup_auto_bootstrap(self) -> None:
        """
        Self-healing startup check:
        1. If TEA-GNN PyTorch weights exist (`gnn_tea-gnn_tuned.pt` or `gnn_tea-gnn.pt`)
           but `models/onnx/tea_gnn.onnx` is missing, automatically export to ONNX.
        2. If `data/processed/processed_matches.csv` or `rag/knowledge_base/team_tactics.json`
           is missing, automatically trigger preprocessing & tactics extraction.
        """
        time.sleep(3.0)  # Let FastAPI complete startup logging first
        try:
            onnx_path = BASE_DIR / "models" / "onnx" / "tea_gnn.onnx"
            pt_tuned = BASE_DIR / "models" / "weights" / "gnn_tea-gnn_tuned.pt"
            pt_base = BASE_DIR / "models" / "weights" / "gnn_tea-gnn.pt"
            if not onnx_path.exists() and (pt_tuned.exists() or pt_base.exists()):
                logger.info("[Auto-Bootstrap] Exporting missing ONNX model from TEA-GNN checkpoint...")
                export_script = BASE_DIR / "models" / "export_gnn.py"
                if export_script.exists():
                    subprocess.run(
                        [sys.executable, str(export_script), "--dynamic"],
                        cwd=str(BASE_DIR),
                        timeout=180,
                        check=False,
                    )
                    self._hot_reload_in_process()

            processed_csv = BASE_DIR / "data" / "processed" / "processed_matches.csv"
            tactics_json = BASE_DIR / "rag" / "knowledge_base" / "team_tactics.json"
            if not processed_csv.exists() and not self.is_running():
                logger.info("[Auto-Bootstrap] processed_matches.csv missing; launching Step 2 & 3...")
                self.trigger_pipeline(skip_scrape=True, sync_db=True, specific_step=2)
            elif not tactics_json.exists() and processed_csv.exists() and not self.is_running():
                logger.info("[Auto-Bootstrap] team_tactics.json missing; launching Step 3...")
                self.trigger_pipeline(skip_scrape=True, sync_db=True, specific_step=3)
        except Exception as e:
            logger.warning("[Auto-Bootstrap] Non-fatal warning during startup check: %s", e)

    def _scheduler_loop(self) -> None:
        """Daemon loop that checks every 30s whether a scheduled pipeline run is due."""
        while not self._scheduler_stop_event.wait(30.0):
            if not self.scheduler_enabled:
                continue

            now = datetime.now()
            should_run = False
            with self.state_lock:
                if self.next_scheduled_run:
                    try:
                        next_dt = datetime.fromisoformat(self.next_scheduled_run)
                        if now >= next_dt:
                            should_run = True
                    except Exception:
                        self.next_scheduled_run = (
                            now + timedelta(hours=self.scheduler_interval_hours)
                        ).isoformat()
                else:
                    self.next_scheduled_run = (
                        now + timedelta(hours=self.scheduler_interval_hours)
                    ).isoformat()

            if should_run:
                if not self.is_running():
                    logger.info("[Auto-Scheduler] Launching scheduled incremental scrape + sync + eval job...")
                    try:
                        self.trigger_eval_job(rag_limit=10, pred_limit=15, skip_scrape=False)
                        with self.state_lock:
                            self.last_scheduled_run = now.isoformat()
                    except Exception as e:
                        logger.error("[Auto-Scheduler] Failed to launch scheduled job: %s", e)
                with self.state_lock:
                    self.next_scheduled_run = (
                        datetime.now() + timedelta(hours=self.scheduler_interval_hours)
                    ).isoformat()

    def _hot_reload_in_process(self) -> None:
        """Directly reloads FastAPI in-memory singletons after any background job completes."""
        try:
            from api.dependencies import reload_all_singletons

            info = reload_all_singletons(self.app_ref)
            self._log_line(f"[JOB MANAGER] In-process API singletons hot-reloaded: {info}")
        except Exception as e:
            self._log_line(f"[JOB MANAGER] Note: In-process reload skipped/failed: {e}")

    def is_running(self) -> bool:
        with self.state_lock:
            return self.active_job is not None and self.active_job.status == "RUNNING"

    def get_status(self) -> Dict[str, Any]:
        with self.state_lock:
            if not self.active_job:
                return {
                    "status": "IDLE",
                    "job_id": None,
                    "step": 0,
                    "step_name": "None",
                    "progress_pct": 0,
                    "started_at": None,
                    "elapsed_s": 0.0,
                    "recent_logs": list(self.log_buffer)[-20:],
                }

            elapsed = 0.0
            if self.active_job.started_at:
                try:
                    start_dt = datetime.fromisoformat(self.active_job.started_at)
                    elapsed = round((datetime.now() - start_dt).total_seconds(), 1)
                except Exception:
                    elapsed = 0.0

            return {
                "status": self.active_job.status,
                "job_id": self.active_job.job_id,
                "step": self.active_job.step,
                "step_name": self.active_job.step_name,
                "progress_pct": self.active_job.progress_pct,
                "started_at": self.active_job.started_at,
                "ended_at": self.active_job.ended_at,
                "elapsed_s": elapsed,
                "error": self.active_job.error,
                "parameters": self.active_job.parameters,
                "recent_logs": list(self.log_buffer)[-30:],
            }

    def get_logs(self, limit: int = 200) -> List[str]:
        with self.state_lock:
            return list(self.log_buffer)[-limit:]

    def get_history(self) -> List[Dict[str, Any]]:
        with self.state_lock:
            history = []
            for j in reversed(self.job_history):
                history.append({
                    "job_id": j.job_id,
                    "status": j.status,
                    "started_at": j.started_at,
                    "ended_at": j.ended_at,
                    "duration_s": j.duration_s,
                    "error": j.error,
                    "parameters": j.parameters,
                })
            return history

    def cancel_job(self) -> bool:
        with self.state_lock:
            if not self.active_job or self.active_job.status != "RUNNING":
                return False

            if self.active_process and self.active_process.poll() is None:
                try:
                    self.active_process.terminate()
                    time.sleep(1)
                    if self.active_process.poll() is None:
                        self.active_process.kill()
                except Exception as e:
                    logger.error("Error killing pipeline process: %s", e)

            self.active_job.status = "CANCELLED"
            self.active_job.ended_at = datetime.now().isoformat()
            self._log_line("[JOB MANAGER] Job was cancelled by user request.")
            return True

    def trigger_pipeline(
        self,
        provider: str = "ollama",
        model_name: str = "",
        gnn_model: str = "TEA-GNN",
        gnn_epochs: int = 100,
        skip_scrape: bool = False,
        force_scrape: bool = False,
        sync_db: bool = True,
        teacher_samples: int = 0,
        specific_step: Optional[int] = None,
    ) -> JobRunRecord:
        """Launches the 8-stage pipeline (or a single step) in a background worker thread."""
        with self.state_lock:
            if self.active_job and self.active_job.status == "RUNNING":
                raise RuntimeError(f"A pipeline job ({self.active_job.job_id}) is already currently running!")

            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.log_buffer.clear()

            params = {
                "job_type": "master_pipeline",
                "provider": provider,
                "model_name": model_name,
                "gnn_model": gnn_model,
                "gnn_epochs": gnn_epochs,
                "skip_scrape": skip_scrape,
                "force_scrape": force_scrape,
                "sync_db": sync_db,
                "teacher_samples": teacher_samples,
                "specific_step": specific_step,
            }

            self.active_job = JobRunRecord(
                job_id=job_id,
                status="RUNNING",
                step=specific_step or 1,
                step_name="Starting Pipeline Worker...",
                progress_pct=5,
                started_at=datetime.now().isoformat(),
                parameters=params,
            )

        thread = threading.Thread(
            target=self._run_job_worker,
            args=(job_id, params),
            daemon=True,
            name=f"Worker-{job_id}",
        )
        self.active_thread = thread
        thread.start()

        return self.active_job

    def trigger_eval_job(
        self,
        rag_limit: int = 10,
        pred_limit: int = 15,
        skip_scrape: bool = False,
    ) -> JobRunRecord:
        """
        Launches `data/auto_scrape_and_eval.py` in a background worker thread to perform
        incremental scraping, multi-store sync, API hot-reload, and live RAG/Prediction evaluation.
        """
        with self.state_lock:
            if self.active_job and self.active_job.status == "RUNNING":
                raise RuntimeError(f"A job ({self.active_job.job_id}) is already currently running!")

            job_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.log_buffer.clear()

            params = {
                "job_type": "auto_scrape_and_eval",
                "rag_limit": rag_limit,
                "pred_limit": pred_limit,
                "skip_scrape": skip_scrape,
            }

            self.active_job = JobRunRecord(
                job_id=job_id,
                status="RUNNING",
                step=1,
                step_name="Starting Auto-Scrape & Continuous Eval...",
                progress_pct=10,
                started_at=datetime.now().isoformat(),
                parameters=params,
            )

        thread = threading.Thread(
            target=self._run_eval_worker,
            args=(job_id, params),
            daemon=True,
            name=f"EvalWorker-{job_id}",
        )
        self.active_thread = thread
        thread.start()

        return self.active_job

    def _log_line(self, line: str):
        cleaned = line.rstrip()
        if cleaned:
            self.log_buffer.append(cleaned)
            logger.info(cleaned)

    def _run_job_worker(self, job_id: str, params: Dict[str, Any]):
        start_time = time.time()
        self._log_line("==================================================")
        self._log_line(f"  STARTING PIPELINE JOB [{job_id}]")
        self._log_line(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log_line(f"  Parameters: {params}")
        self._log_line("==================================================")

        run_script = BASE_DIR / "data" / "run_pipeline.py"
        if not run_script.exists():
            with self.state_lock:
                if self.active_job:
                    self.active_job.status = "FAILED"
                    self.active_job.error = f"Pipeline script not found: {run_script}"
                    self.active_job.ended_at = datetime.now().isoformat()
            return

        cmd = [sys.executable, "-u", str(run_script)]

        specific_step = params.get("specific_step")
        step_skip_flags = {
            1: "--skip-scrape",
            2: "--skip-preprocess",
            3: "--skip-tactics",
            4: "--skip-teacher",
            5: "--skip-split",
            6: "--skip-gnn",
            7: "--skip-export",
            8: "--skip-reload",
        }
        if specific_step:
            for s, flag in step_skip_flags.items():
                if s != specific_step:
                    cmd.append(flag)
            if specific_step == 3 and params.get("sync_db", True):
                cmd.append("--sync-db")
        else:
            if params.get("skip_scrape"):
                cmd.append("--skip-scrape")
            if params.get("force_scrape"):
                cmd.append("--force-scrape")
            if params.get("sync_db", True):
                cmd.append("--sync-db")

        cmd.extend(["--provider", params.get("provider", "ollama")])
        if params.get("model_name"):
            cmd.extend(["--model-name", params["model_name"]])
        if params.get("teacher_samples", 0) > 0:
            cmd.extend(["--teacher-samples", str(params["teacher_samples"])])
        if params.get("gnn_model"):
            cmd.extend(["--gnn-model", params["gnn_model"]])
        if params.get("gnn_epochs"):
            cmd.extend(["--gnn-epochs", str(params["gnn_epochs"])])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1,
            )
            with self.state_lock:
                self.active_process = process

            step_names = {
                1: "Incremental Ingestion",
                2: "Preprocessing & Features",
                3: "Tactics & Multi-Store Sync",
                4: "Teacher LLM Generation",
                5: "Train/Val Re-split",
                6: "GNN Model Retraining",
                7: "ONNX Model Export",
                8: "API Singleton Hot-Reload",
            }

            for line in process.stdout:
                line_str = line.strip()
                if not line_str:
                    continue
                self._log_line(line_str)

                for s_num, s_name in step_names.items():
                    if f"[STEP {s_num}/8]" in line_str or f"[STEP {s_num}/6]" in line_str:
                        with self.state_lock:
                            if self.active_job:
                                self.active_job.step = s_num
                                self.active_job.step_name = s_name
                                self.active_job.progress_pct = int((s_num / 8.0) * 92)

            process.wait()
            ret_code = process.returncode
            duration = round(time.time() - start_time, 1)

            if ret_code == 0:
                self._hot_reload_in_process()

            with self.state_lock:
                if self.active_job and self.active_job.status != "CANCELLED":
                    if ret_code == 0:
                        self.active_job.status = "SUCCESS"
                        self.active_job.progress_pct = 100
                        self.active_job.step_name = "Completed Successfully"
                        self._log_line(f"[JOB MANAGER] Pipeline job [{job_id}] completed successfully in {duration}s.")
                    else:
                        self.active_job.status = "FAILED"
                        self.active_job.error = f"Process exited with non-zero code: {ret_code}"
                        self._log_line(f"[JOB MANAGER] Pipeline job [{job_id}] failed (exit code {ret_code}).")

                    self.active_job.ended_at = datetime.now().isoformat()
                    self.active_job.duration_s = duration
                    self.active_job.logs = list(self.log_buffer)
                    self.job_history.append(self.active_job)

        except Exception as e:
            logger.error("Job execution exception: %s", e, exc_info=True)
            with self.state_lock:
                if self.active_job:
                    self.active_job.status = "FAILED"
                    self.active_job.error = str(e)
                    self.active_job.ended_at = datetime.now().isoformat()
                    self.active_job.duration_s = round(time.time() - start_time, 1)
                    self.job_history.append(self.active_job)

    def _run_eval_worker(self, job_id: str, params: Dict[str, Any]):
        start_time = time.time()
        self._log_line("==================================================")
        self._log_line(f"  STARTING AUTO-SCRAPE & EVAL JOB [{job_id}]")
        self._log_line(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log_line(f"  Parameters: {params}")
        self._log_line("==================================================")

        eval_script = BASE_DIR / "data" / "auto_scrape_and_eval.py"
        if not eval_script.exists():
            with self.state_lock:
                if self.active_job:
                    self.active_job.status = "FAILED"
                    self.active_job.error = f"Eval script not found: {eval_script}"
                    self.active_job.ended_at = datetime.now().isoformat()
            return

        cmd = [
            sys.executable,
            "-u",
            str(eval_script),
            "--rag-limit",
            str(params.get("rag_limit", 10)),
            "--pred-limit",
            str(params.get("pred_limit", 15)),
        ]
        if params.get("skip_scrape"):
            cmd.append("--skip-scrape")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1,
            )
            with self.state_lock:
                self.active_process = process

            for line in process.stdout:
                line_str = line.strip()
                if not line_str:
                    continue
                self._log_line(line_str)
                if "STAGE 1" in line_str:
                    with self.state_lock:
                        if self.active_job:
                            self.active_job.step = 1
                            self.active_job.step_name = "Incremental Scrape & Preprocess"
                            self.active_job.progress_pct = 25
                elif "STAGE 1.5" in line_str:
                    with self.state_lock:
                        if self.active_job:
                            self.active_job.step = 2
                            self.active_job.step_name = "Syncing DB, Neo4j, FAISS & pgvector"
                            self.active_job.progress_pct = 55
                elif "STAGE 2" in line_str:
                    with self.state_lock:
                        if self.active_job:
                            self.active_job.step = 3
                            self.active_job.step_name = "Running RAG & Prediction Evaluations"
                            self.active_job.progress_pct = 80

            process.wait()
            ret_code = process.returncode
            duration = round(time.time() - start_time, 1)

            if ret_code == 0:
                self._hot_reload_in_process()

            with self.state_lock:
                if self.active_job and self.active_job.status != "CANCELLED":
                    if ret_code == 0:
                        self.active_job.status = "SUCCESS"
                        self.active_job.progress_pct = 100
                        self.active_job.step_name = "Auto-Scrape & Eval Complete"
                        self._log_line(f"[JOB MANAGER] Eval job [{job_id}] completed in {duration}s.")
                    else:
                        self.active_job.status = "FAILED"
                        self.active_job.error = f"Process exited with code {ret_code}"
                        self._log_line(f"[JOB MANAGER] Eval job [{job_id}] failed (code {ret_code}).")

                    self.active_job.ended_at = datetime.now().isoformat()
                    self.active_job.duration_s = duration
                    self.active_job.logs = list(self.log_buffer)
                    self.job_history.append(self.active_job)
        except Exception as e:
            logger.error("Eval job exception: %s", e, exc_info=True)
            with self.state_lock:
                if self.active_job:
                    self.active_job.status = "FAILED"
                    self.active_job.error = str(e)
                    self.active_job.ended_at = datetime.now().isoformat()
                    self.active_job.duration_s = round(time.time() - start_time, 1)
                    self.job_history.append(self.active_job)


job_manager = PipelineJobManager()
