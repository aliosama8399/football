"""
Apache Airflow Multi-Task Modular DAG: Football Data Preparation Pipeline (GNN & LLM Datasets)
==============================================================================================
Production DAG orchestrating data preparation for both GNN and LLM fine-tuning datasets:
  [Task 1] Incremental Data Ingestion (Finished <= 2526 + Active 2627)
  [Task 2] Preprocessing & Feature Engineering (GNN Dataset: data/processed/processed_matches.csv)
  [Task 3] Teacher LLM Tactical Annotation via Ollama (LLM Dataset: data/finetune/football_tactical.jsonl)
  [Task 4] Team Tactics Extraction & Knowledge/Vector Store Sync (Neo4j, Postgres, FAISS, pgvector)
  [Task 5] Fine-Tuning Dataset Re-splitting (data/finetune/football_train.json & football_val.json)
  [Task 6] Trigger Downstream GNN MLflow Training & Production Promotion DAG (football_gnn_mlflow_pipeline)

Schedule: Daily at 02:00 UTC (0 2 * * *)
Airflow Web UI: http://localhost:8080
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

GPU_WORKER_CONTAINER = os.environ.get("GPU_WORKER_CONTAINER", "football-auto-eval")
BASE_DIR = os.environ.get("PROJECT_ROOT", "/workspace")
PYTHON_BIN = f"PYTHONPATH={BASE_DIR} {os.environ.get('PYTHON_BIN', 'python')}"


def run_teacher_in_gpu_worker(**context):
    """
    Triggers Step 4 (GNN prediction + Teacher LLM) on the GPU worker (FastAPI) and streams logs.
    Fails loudly if the worker encounters any GNN/LLM error or produces 0 analyses.
    """
    import json
    import time
    import urllib.parse
    import urllib.request

    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    api_url = os.environ.get("API_URL", "http://fastapi:8001").rstrip("/")
    provider = conf.get("provider", "ollama")
    model_name = conf.get("model_name")
    if not model_name:
        try:
            import yaml

            cfg_file = os.path.join(BASE_DIR, "models", "llm_config.yaml")
            if os.path.exists(cfg_file):
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                model_name = cfg.get("providers", {}).get(provider, {}).get("default_model", "")
        except Exception as e:
            print(f"[Airflow -> GPU Worker] Note: Could not read default_model from yaml: {e}")
    if not model_name:
        model_name = "gemma4:31b-cloud" if provider == "ollama" else ""

    # Default to 5 samples per run to ensure stable incremental progress unless overridden
    teacher_samples = int(conf["teacher_samples"]) if "teacher_samples" in conf else 5

    params = urllib.parse.urlencode({
        "provider": provider,
        "model_name": model_name,
        "teacher_samples": teacher_samples,
    })
    trigger_url = f"{api_url}/api/v1/jobs/pipeline/step/4?{params}"
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Token": "football_internal_secret_token_2026",
    }

    print(f"[Airflow -> GPU Worker] Triggering Step 4 via {trigger_url}...")
    try:
        req = urllib.request.Request(trigger_url, data=b"{}", headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            job_id = resp_data.get("job_id")
            print(f"[Airflow -> GPU Worker] Job launched: {job_id} (status: {resp_data.get('status')})")
    except Exception as e:
        raise RuntimeError(f"Failed to trigger Step 4 on GPU worker ({api_url}): {e}")

    status_url = f"{api_url}/api/v1/jobs/pipeline/status"
    seen_logs = 0
    start_time = time.time()
    max_wait = 12 * 3600  # Up to 12 hours for large annotation runs

    while time.time() - start_time < max_wait:
        time.sleep(5)
        try:
            req = urllib.request.Request(status_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                status_data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  [Warn] Status poll temporary error: {e}")
            continue

        logs = status_data.get("recent_logs", [])
        if len(logs) > seen_logs:
            for line in logs[seen_logs:]:
                print(f"[Worker] {line}")
            seen_logs = len(logs)

        st = status_data.get("status", "UNKNOWN")
        if st == "SUCCESS":
            print(f"[Airflow -> GPU Worker] Step 4 completed successfully! Duration: {time.time() - start_time:.1f}s")
            return
        elif st in ("FAILED", "CANCELLED"):
            err = status_data.get("error", "Unknown worker error")
            raise RuntimeError(f"Step 4 failed on GPU worker with status {st}: {err}")

    raise TimeoutError(f"Step 4 timed out after {max_wait}s on GPU worker ({api_url})")


default_args = {
    "owner": "football-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(hours=3),
}

with DAG(
    dag_id="football_master_pipeline",
    default_args=default_args,
    description="Data Preparation DAG: Ingestion -> GNN Features -> Teacher LLM -> Tactics/KB Sync -> LLM Splits -> Triggers GNN MLflow DAG",
    schedule_interval="0 2 * * *",  # Daily at 02:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["football", "mlops", "data-prep", "llm-teacher", "finetune", "pipeline"],
    doc_md=__doc__,
) as dag:

    # ── Task 1: Incremental Ingestion ─────────────────────────────────────────
    task_ingest = BashOperator(
        task_id="step1_incremental_ingest",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/run_pipeline.py "
            "--skip-preprocess --skip-tactics --skip-teacher --skip-split --skip-gnn --skip-export --skip-reload"
        ),
        doc_md="Fetches finished seasons (<= 2526) from disk cache and active season (2627) incrementally.",
    )

    # ── Task 2: Preprocessing & Feature Engineering (GNN Dataset) ─────────────
    task_preprocess = BashOperator(
        task_id="step2_preprocess_features",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/run_pipeline.py "
            "--skip-scrape --skip-tactics --skip-teacher --skip-split --skip-gnn --skip-export --skip-reload"
        ),
        doc_md="Rebuilds processed_matches.csv (GNN dataset) with rolling form, Elo, H2H, Referee strictness, and xG features.",
    )

    # ── Task 3: Team Tactics Grounding & Knowledge/Vector Store Refresh ───────
    task_tactics = BashOperator(
        task_id="step3_team_tactics_kb",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/run_pipeline.py "
            "--skip-scrape --skip-preprocess --skip-teacher --skip-split --skip-gnn --skip-export --skip-reload --sync-db"
        ),
        doc_md="Extracts team tactics into team_tactics.json and synchronizes to Neo4j, PostgreSQL, FAISS, and pgvector.",
    )

    # ── Task 4: Teacher LLM Tactical Generation (GNN + Ollama in GPU worker) ──
    task_teacher_llm = PythonOperator(
        task_id="step4_teacher_llm_annotation",
        python_callable=run_teacher_in_gpu_worker,
        execution_timeout=timedelta(hours=12),
        doc_md="Runs TEA-GNN prediction + teacher LLM (Ollama, model from llm_config.yaml) for "
               "un-annotated matches inside the GPU worker container.",
    )

    # ── Task 5: Train/Val Dataset Re-splitting (LLM Dataset) ──────────────────
    task_resplit = BashOperator(
        task_id="step5_resplit_finetune_data",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/run_pipeline.py "
            "--skip-scrape --skip-preprocess --skip-tactics --skip-teacher --skip-gnn --skip-export --skip-reload"
        ),
        doc_md="Verifies/re-splits clean tactical records into train/val sets (data/finetune & models/).",
    )

    # ── Task 6: Trigger GNN MLflow Training & Production Promotion DAG ────────
    trigger_gnn_mlflow_dag = TriggerDagRunOperator(
        task_id="trigger_gnn_mlflow_dag",
        trigger_dag_id="football_gnn_mlflow_pipeline",
        conf={
            "gnn_model": "{{ dag_run.conf.get('gnn_model', 'TEA-GNN') if dag_run and dag_run.conf else 'TEA-GNN' }}",
            "gnn_epochs": "{{ dag_run.conf.get('gnn_epochs', 100) if dag_run and dag_run.conf else 100 }}",
            "experiment": "{{ dag_run.conf.get('experiment', 'football-gnn-training') if dag_run and dag_run.conf else 'football-gnn-training' }}",
            "selection_metric": "{{ dag_run.conf.get('selection_metric', 'composite') if dag_run and dag_run.conf else 'composite' }}",
            "judge": "{{ dag_run.conf.get('judge', 'heuristic') if dag_run and dag_run.conf else 'heuristic' }}",
            "eval_samples": "{{ dag_run.conf.get('eval_samples', 15) if dag_run and dag_run.conf else 15 }}",
        },
        wait_for_completion=False,
        reset_dag_run=True,
        doc_md="Triggers football_gnn_mlflow_pipeline once GNN and LLM datasets are prepared.",
    )

    # ── Sequential DAG Dependency Flow ────────────────────────────────────────
    # Teacher runs BEFORE tactics so team_tactics.json ingests the new LLM analyses.
    (
        task_ingest
        >> task_preprocess
        >> task_teacher_llm
        >> task_tactics
        >> task_resplit
        >> trigger_gnn_mlflow_dag
    )
