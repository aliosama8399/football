"""
Apache Airflow DAG: Football RAG Evaluation (LLM-as-a-Judge)
============================================================
Independent, self-scheduled DAG delegating execution to the GPU Worker:
  [Task 1] Hot-Reload Live FastAPI Singletons (POST /api/v1/admin/reload) so latest
           production GNN/SLM models and knowledge stores are active
  [Task 2] Run RAG & Match Prediction Evaluation with LLM-as-a-Judge on Worker:
           delegates to Worker (POST /api/v1/jobs/eval/trigger) and streams real-time
           RAGAS faithfulness, relevancy, precision/recall, and grounding logs.

Schedule: Daily at 04:00 UTC (0 4 * * *).
Manual trigger conf (optional):
  {"judge": "gemini", "samples": 15}
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    # Standalone mock classes allowing syntax verification and CI linting outside Airflow
    class DAG:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class _BaseOp:
        def __init__(self, *args, **kwargs):
            pass

        def __rshift__(self, other):
            return other

    class PythonOperator(_BaseOp):  # type: ignore[no-redef]
        pass


BASE_DIR = os.environ.get("PROJECT_ROOT")
if not BASE_DIR or not os.path.exists(BASE_DIR):
    BASE_DIR = str(Path(__file__).resolve().parent.parent.parent)


def reload_api_singletons(**_context) -> dict:
    """Task 1: Hot-reload FastAPI in-memory singletons (RAG system, GNN/SLM providers, stores)."""
    import json
    import urllib.request

    api_url = os.environ.get("API_URL", "http://bento:8001").rstrip("/")
    reload_url = f"{api_url}/api/v1/admin/reload"
    print(f"[RAG Eval] Requesting API singleton reload at {reload_url}...")
    try:
        req = urllib.request.Request(
            reload_url,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"[OK] Reload succeeded ({resp.status}): {body[:500]}")
            return {"status": resp.status, "body": json.loads(body) if body.startswith("{") else body}
    except Exception as e:
        print(f"[WARN] API hot-reload note (service unreachable or restarting): {e}")
        return {"status": "skipped", "error": str(e)}


def run_rag_eval_on_worker(**context) -> dict:
    """Task 2: Dispatch RAG evaluation with LLM-as-a-judge to worker and stream live logs."""
    import json
    import time
    import urllib.request

    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    judge = str(conf.get("judge", "gemini"))
    try:
        samples = int(conf.get("samples", conf.get("rag_samples", 15)))
    except (ValueError, TypeError):
        samples = 15

    api_url = os.environ.get("API_URL", "http://bento:8001").rstrip("/")
    trigger_url = f"{api_url}/api/v1/jobs/eval/trigger"
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Token": "football_internal_secret_token_2026",
    }
    payload = json.dumps(
        {
            "skip_scrape": True,
            "skip_sync": True,
            "judge": judge,
            "samples": samples,
        }
    ).encode("utf-8")

    print(f"[Airflow -> Worker] Dispatching RAG evaluation job (judge={judge}, samples={samples}) to {trigger_url}...")
    try:
        req = urllib.request.Request(trigger_url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            job_id = resp_data.get("job_id")
            print(f"[Airflow -> Worker] Eval job launched: {job_id} (status: {resp_data.get('status')})")
    except Exception as e:
        print(f"[Airflow -> Worker] Note: Could not reach worker API ({e}); falling back to local runner...")
        import subprocess

        cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "data", "auto_scrape_and_eval.py"),
            "--now",
            "--skip-scrape",
            "--skip-sync",
            "--judge",
            judge,
            "--samples",
            str(samples),
        ]
        res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, check=True)
        print(res.stdout)
        return {"status": "SUCCESS", "mode": "local_fallback"}

    status_url = f"{api_url}/api/v1/jobs/pipeline/status"
    seen_logs = 0
    start_time = time.time()
    max_wait = 3600  # 1 hour timeout for LLM judge evaluation

    while time.time() - start_time < max_wait:
        time.sleep(4)
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
            elapsed = time.time() - start_time
            print(f"[Airflow -> Worker] RAG Evaluation completed successfully in {elapsed:.1f}s!")
            return status_data
        elif st in ("FAILED", "CANCELLED"):
            err = status_data.get("error", "Unknown worker error")
            raise RuntimeError(f"RAG Evaluation failed on worker with status {st}: {err}")

    raise TimeoutError(f"RAG Evaluation timed out after {max_wait}s on worker ({api_url})")


default_args = {
    "owner": "football-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

with DAG(
    dag_id="football_rag_eval",
    default_args=default_args,
    description="Independent RAG evaluation with LLM-as-a-judge: delegates to worker and streams real-time RAGAS metrics",
    schedule_interval="0 4 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["football", "mlops", "eval", "rag", "llm-judge", "ragas"],
    doc_md=__doc__,
) as dag:

    # ── Task 1: Hot-Reload Live FastAPI Singletons ───────────────────────────
    task_reload_api = PythonOperator(
        task_id="reload_api_singletons",
        python_callable=reload_api_singletons,
        execution_timeout=timedelta(minutes=5),
        doc_md="Hot-reloads FastAPI singletons so the latest production models and knowledge stores are served.",
    )

    # ── Task 2: Continuous RAG Evaluation with LLM-as-a-Judge ────────────────
    task_eval_rag_judge = PythonOperator(
        task_id="evaluate_rag_with_llm_judge",
        python_callable=run_rag_eval_on_worker,
        execution_timeout=timedelta(hours=1),
        doc_md="Delegates RAG evaluation to worker container via POST /api/v1/jobs/eval/trigger and streams real-time "
               "RAGAS (faithfulness, relevancy, precision, recall) and numeric grounding logs.",
    )

    task_reload_api >> task_eval_rag_judge
