"""
Apache Airflow Multi-Task Modular DAG: Football Master Pipeline
================================================================
Production DAG orchestrating the 8 sequential stages of the automated pipeline:
  [Task 1] Incremental Data Ingestion (Finished <= 2526 + Active 2627)
  [Task 2] Preprocessing & Feature Engineering
  [Task 3] Team Tactics Extraction & Knowledge/Vector Store Sync (Neo4j, Postgres, FAISS, pgvector)
  [Task 4] Teacher LLM Tactical Annotation (Ollama)
  [Task 5] Fine-Tuning Dataset Re-splitting (data/finetune & models/)
  [Task 6] GNN Retraining with MLflow Tracking
  [Task 7] GNN ONNX Runtime Export & Parity Validation
  [Task 8] Live FastAPI Hot-Reload & Continuous Evaluation Suite

Schedule: Daily at 02:00 UTC (0 2 * * *)
Airflow Web UI: http://localhost:8080
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Default task arguments
default_args = {
    "owner": "football-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(hours=3),
}

# Base project path inside container or host
BASE_DIR = os.environ.get("PROJECT_ROOT", "/opt/airflow")
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")

with DAG(
    dag_id="football_master_pipeline",
    default_args=default_args,
    description="Production 8-task pipeline: Ingestion -> Preprocessing -> Tactics/Vector/KG -> Teacher LLM -> Splits -> GNN MLflow -> ONNX Export -> API Reload & Eval",
    schedule_interval="0 2 * * *",  # Daily at 02:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["football", "mlops", "gnn", "onnx", "llm-teacher", "finetune", "pipeline"],
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

    # ── Task 2: Preprocessing & Feature Engineering ───────────────────────────
    task_preprocess = BashOperator(
        task_id="step2_preprocess_features",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/run_pipeline.py "
            "--skip-scrape --skip-tactics --skip-teacher --skip-split --skip-gnn --skip-export --skip-reload"
        ),
        doc_md="Rebuilds processed_matches.csv with rolling form, Elo, H2H, chronological Referee strictness, and xG features.",
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

    # ── Task 4: Teacher LLM Tactical Generation (Ollama) ──────────────────────
    task_teacher_llm = BashOperator(
        task_id="step4_teacher_llm_annotation",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/run_pipeline.py "
            "--skip-scrape --skip-preprocess --skip-tactics --skip-split --skip-gnn --skip-export --skip-reload "
            "--provider {{ dag_run.conf.get('provider', 'ollama') }} "
            "--model-name '{{ dag_run.conf.get('model_name', '') }}' "
            "--teacher-samples {{ dag_run.conf.get('teacher_samples', 0) }}"
        ),
        doc_md="Queries local Ollama Teacher LLM to produce grounded tactical analyses for un-annotated matches.",
    )

    # ── Task 5: Train/Val Dataset Re-splitting ─────────────────────────────────
    task_resplit = BashOperator(
        task_id="step5_resplit_finetune_data",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/run_pipeline.py "
            "--skip-scrape --skip-preprocess --skip-tactics --skip-teacher --skip-gnn --skip-export --skip-reload"
        ),
        doc_md="Re-splits clean tactical records into train/val sets and mirrors to data/finetune & models/.",
    )

    # ── Task 6: Retrain GNN Model with MLflow Tracking ────────────────────────
    task_train_gnn = BashOperator(
        task_id="step6_retrain_gnn_mlflow",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/run_pipeline.py "
            "--skip-scrape --skip-preprocess --skip-tactics --skip-teacher --skip-split --skip-export --skip-reload "
            "--gnn-model {{ dag_run.conf.get('gnn_model', 'TEA-GNN') }} "
            "--gnn-epochs {{ dag_run.conf.get('gnn_epochs', 100) }} "
            "--experiment {{ dag_run.conf.get('experiment', 'football-gnn-training') }}"
        ),
        doc_md="Trains GNN match prediction model and logs metrics, confusion matrix, and model weights to MLflow.",
    )

    # ── Task 7: Export GNN Checkpoint to ONNX Runtime ─────────────────────────
    task_export_onnx = BashOperator(
        task_id="step7_export_gnn_onnx",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} models/export_gnn.py --dynamic"
        ),
        doc_md="Exports newly trained TEA-GNN checkpoint to ONNX Runtime format and validates PyTorch-ONNX parity.",
    )

    # ── Task 8: Hot-Reload API & Run Continuous Evaluation ────────────────────
    task_reload_and_eval = BashOperator(
        task_id="step8_reload_api_and_eval",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/auto_scrape_and_eval.py --now --skip-scrape --skip-sync "
            "--judge {{ dag_run.conf.get('judge', 'heuristic') }} "
            "--samples {{ dag_run.conf.get('eval_samples', 15) }}"
        ),
        doc_md="Triggers POST /api/v1/admin/reload on the FastAPI backend and runs continuous RAG & Match Prediction evaluations.",
    )

    # ── Sequential DAG Dependency Flow ────────────────────────────────────────
    (
        task_ingest
        >> task_preprocess
        >> task_tactics
        >> task_teacher_llm
        >> task_resplit
        >> task_train_gnn
        >> task_export_onnx
        >> task_reload_and_eval
    )
