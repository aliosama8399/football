"""
Pipeline Background Job & Autonomous Scheduler API Routes
=========================================================
Provides REST endpoints for on-demand triggering, autonomous scheduler management,
status polling, log viewing, and lifecycle management of the automated
8-stage data-to-fine-tuning pipeline and continuous evaluation loop.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from api.auth import decode_access_token
from api.services.job_runner import job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Automated Pipeline Jobs"])


async def _verify_supervisor_or_internal(request: Request) -> None:
    """
    Guard mutating job endpoints: allows authenticated supervisors (Bearer JWT),
    valid `X-Internal-Token` headers, or trusted local loopback automation requests.
    """
    internal_expected = os.environ.get("INTERNAL_ADMIN_TOKEN", "")
    internal_header = request.headers.get("X-Internal-Token", "")
    if internal_expected and internal_header == internal_expected:
        return

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token_str = auth_header.split(" ", 1)[1].strip()
        payload = decode_access_token(token_str)
        if payload and payload.get("sub"):
            role = payload.get("role")
            if role == "supervisor":
                return
            from api.dependencies import get_db
            from api.database import User
            from sqlalchemy.future import select

            async for db in get_db():
                stmt = select(User).where(User.username == payload["sub"])
                user = (await db.execute(stmt)).scalar_one_or_none()
                if user and user.is_active and user.role == "supervisor":
                    return
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Supervisor privileges required to manage pipeline jobs.",
                )

    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Supervisor authentication required to trigger or modify pipeline jobs.",
    )


class PipelineTriggerRequest(BaseModel):
    provider: str = Field(default="ollama", description="Teacher LLM provider ('ollama', 'gemini', 'vllm')")
    model_name: str = Field(default="", description="Specific model name (e.g. 'qwen3.5:0.8b')")
    gnn_model: str = Field(default="TEA-GNN", description="GNN architecture ('TEA-GNN', 'Hybrid', 'all')")
    gnn_epochs: int = Field(default=100, ge=1, le=1000, description="Training epochs for GNN")
    skip_scrape: bool = Field(default=False, description="Skip incremental data collection")
    force_scrape: bool = Field(default=False, description="Force re-downloading all seasons")
    sync_db: bool = Field(default=True, description="Sync tactics & matches to PostgreSQL, Neo4j, FAISS & pgvector")
    teacher_samples: int = Field(default=0, ge=0, description="Limit teacher LLM annotations (0 = all new)")
    trigger_airflow: bool = Field(default=False, description="Trigger via Apache Airflow REST API if available")


class EvalTriggerRequest(BaseModel):
    rag_limit: int = Field(default=10, ge=1, le=100, description="Number of RAG golden samples to evaluate")
    pred_limit: int = Field(default=15, ge=1, le=200, description="Number of prediction matchups to evaluate")
    skip_scrape: bool = Field(default=False, description="Skip incremental scraping before evaluation")


class SchedulerConfigRequest(BaseModel):
    enabled: Optional[bool] = Field(default=None, description="Enable or disable autonomous periodic runs")
    interval_hours: Optional[float] = Field(default=None, ge=0.5, le=168.0, description="Interval in hours between runs")


class JobStatusResponse(BaseModel):
    status: str
    job_id: Optional[str] = None
    step: int
    step_name: str
    progress_pct: int
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    elapsed_s: float
    error: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    recent_logs: List[str] = []


def _trigger_airflow_dag(dag_id: str = "football_master_pipeline", conf: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Attempt triggering the Airflow DAG via Airflow REST API."""
    airflow_urls = [
        "http://airflow-webserver:8080/api/v1",
        "http://localhost:8080/api/v1",
        "http://127.0.0.1:8080/api/v1",
    ]
    payload = json.dumps({"conf": conf or {}}).encode("utf-8")

    for base_url in airflow_urls:
        url = f"{base_url}/dags/{dag_id}/dagRuns"
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Basic YWlyZmxvdzphaXJmbG93",  # airflow:airflow
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status in (200, 201):
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue

    raise ConnectionError("Could not connect to Airflow REST API on port 8080.")


@router.post("/pipeline/trigger", response_model=Dict[str, Any])
async def trigger_pipeline_job(
    payload: PipelineTriggerRequest,
    _: None = Depends(_verify_supervisor_or_internal),
):
    """
    Trigger the automated 8-stage data-to-fine-tuning pipeline.
    Runs asynchronously and provides a job_id for status tracking.
    """
    if payload.trigger_airflow:
        try:
            result = _trigger_airflow_dag(
                dag_id="football_master_pipeline",
                conf=payload.model_dump(),
            )
            return {
                "triggered_via": "apache_airflow",
                "dag_run_id": result.get("dag_run_id"),
                "state": result.get("state"),
                "execution_date": result.get("execution_date"),
                "message": "Airflow DAG football_master_pipeline triggered successfully.",
                "web_ui": "http://localhost:8080",
            }
        except Exception as e:
            logger.warning("Airflow trigger failed (%s); falling back to native background worker.", e)

    try:
        job = job_manager.trigger_pipeline(
            provider=payload.provider,
            model_name=payload.model_name,
            gnn_model=payload.gnn_model,
            gnn_epochs=payload.gnn_epochs,
            skip_scrape=payload.skip_scrape,
            force_scrape=payload.force_scrape,
            sync_db=payload.sync_db,
            teacher_samples=payload.teacher_samples,
        )
        return {
            "triggered_via": "background_worker",
            "job_id": job.job_id,
            "status": job.status,
            "step": job.step,
            "step_name": job.step_name,
            "started_at": job.started_at,
            "message": "Background 8-stage pipeline job launched. Poll /api/v1/jobs/pipeline/status for live progress.",
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {e}")


@router.post("/pipeline/step/{step_id}", response_model=Dict[str, Any])
async def trigger_single_step(
    step_id: int,
    provider: str = "ollama",
    gnn_model: str = "TEA-GNN",
    gnn_epochs: int = 100,
    sync_db: bool = True,
    _: None = Depends(_verify_supervisor_or_internal),
):
    """
    Trigger an individual pipeline step in isolation:
      1: Incremental Ingestion
      2: Preprocessing & Features
      3: Team Tactics & Multi-Store Sync (PostgreSQL, Neo4j, FAISS, pgvector)
      4: Teacher LLM Generation
      5: Train/Val Re-splitting
      6: GNN Retraining with MLflow
      7: ONNX Model Export
      8: API Singleton Hot-Reload
    """
    if step_id not in range(1, 9):
        raise HTTPException(status_code=400, detail="step_id must be an integer between 1 and 8.")

    try:
        job = job_manager.trigger_pipeline(
            provider=provider,
            gnn_model=gnn_model,
            gnn_epochs=gnn_epochs,
            sync_db=sync_db,
            specific_step=step_id,
        )
        return {
            "job_id": job.job_id,
            "step": step_id,
            "status": job.status,
            "message": f"Step {step_id} launched in background.",
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/eval/trigger", response_model=Dict[str, Any])
async def trigger_eval_job(
    payload: EvalTriggerRequest,
    _: None = Depends(_verify_supervisor_or_internal),
):
    """
    Trigger the automated incremental scrape + knowledge store sync + live
    RAG/Prediction evaluation pipeline (`data/auto_scrape_and_eval.py`).
    """
    try:
        job = job_manager.trigger_eval_job(
            rag_limit=payload.rag_limit,
            pred_limit=payload.pred_limit,
            skip_scrape=payload.skip_scrape,
        )
        return {
            "job_id": job.job_id,
            "status": job.status,
            "step_name": job.step_name,
            "started_at": job.started_at,
            "message": "Auto-Scrape & Continuous Evaluation job launched in background.",
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scheduler/status", response_model=Dict[str, Any])
async def get_scheduler_status():
    """Return the autonomous periodic scheduler configuration and next run timestamp."""
    return job_manager.get_scheduler_status()


@router.post("/scheduler/config", response_model=Dict[str, Any])
async def update_scheduler_config(
    payload: SchedulerConfigRequest,
    _: None = Depends(_verify_supervisor_or_internal),
):
    """Enable/disable the autonomous periodic scheduler or update its interval."""
    return job_manager.configure_scheduler(
        enabled=payload.enabled,
        interval_hours=payload.interval_hours,
    )


@router.get("/pipeline/status", response_model=JobStatusResponse)
async def get_pipeline_status():
    """Poll live execution status, current step, percent completion, and recent log lines."""
    return job_manager.get_status()


@router.get("/pipeline/logs")
async def get_pipeline_logs(limit: int = Query(default=200, ge=10, le=1000)):
    """Retrieve raw tail of real-time log output from the running job."""
    return {
        "is_running": job_manager.is_running(),
        "total_lines": len(job_manager.get_logs(limit)),
        "logs": job_manager.get_logs(limit),
    }


@router.get("/pipeline/history")
async def get_pipeline_history():
    """Retrieve execution history of previous pipeline runs."""
    return {
        "history": job_manager.get_history(),
    }


@router.post("/pipeline/cancel")
async def cancel_pipeline_job(_: None = Depends(_verify_supervisor_or_internal)):
    """Cancel currently executing pipeline job."""
    cancelled = job_manager.cancel_job()
    if not cancelled:
        raise HTTPException(status_code=400, detail="No active pipeline job currently running to cancel.")
    return {"message": "Job cancellation signal sent."}
