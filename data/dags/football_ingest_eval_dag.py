"""
Apache Airflow DAG: Football Match Ingestion & Continuous Evaluation
====================================================================
Lightweight daily DAG that orchestrates:
  1. Incremental match scraping, chronological feature preprocessing, and team tactics extraction
  2. Synchronization of PostgreSQL, Neo4j Knowledge Graph, FAISS, and pgvector indexes
  3. Live FastAPI singleton hot-reload (POST /api/v1/admin/reload) and continuous RAG + Prediction evaluation
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "football-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

BASE_DIR = os.environ.get("PROJECT_ROOT", "/opt/airflow")
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")

with DAG(
    "football_scrape_and_continuous_eval",
    default_args=default_args,
    description="Modular match scrape, KG/Vector store sync, API hot-reload, and continuous evaluation pipeline",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["football", "mlops", "eval", "scraping"],
    doc_md=__doc__,
) as dag:

    scrape_and_preprocess = BashOperator(
        task_id="scrape_and_preprocess",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/run_pipeline.py "
            "--skip-teacher --skip-split --skip-gnn --skip-export --skip-reload"
        ),
        doc_md="Runs incremental football-data.co.uk & Understat xG scraping, preprocess.py, and extract_tactics.py.",
    )

    sync_kg_and_vectors = BashOperator(
        task_id="sync_kg_and_vectors",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/run_pipeline.py "
            "--skip-scrape --skip-preprocess --skip-teacher --skip-split --skip-gnn --skip-export --skip-reload --sync-db"
        ),
        doc_md="Synchronizes updated matches and tactical profiles to PostgreSQL, Neo4j, FAISS, and pgvector.",
    )

    reload_api_and_run_eval = BashOperator(
        task_id="reload_api_and_run_eval",
        bash_command=(
            f"cd {BASE_DIR} && "
            f"{PYTHON_BIN} data/auto_scrape_and_eval.py --now --skip-scrape --skip-sync "
            "--judge {{ dag_run.conf.get('judge', 'heuristic') }} "
            "--samples {{ dag_run.conf.get('samples', 15) }}"
        ),
        doc_md="Hot-reloads FastAPI singletons and executes continuous RAG and Match Prediction evaluation suites.",
    )

    scrape_and_preprocess >> sync_kg_and_vectors >> reload_api_and_run_eval
