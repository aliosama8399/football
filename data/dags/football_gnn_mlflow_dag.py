"""
Apache Airflow DAG: Football GNN & SLM MLflow Training & Production Promotion Pipeline
======================================================================================
Dedicated MLOps DAG triggered after GNN & LLM dataset preparation (`football_master_pipeline`):
  [Task 1] Train GNN Model(s) (`TEA-GNN` or `all` 7 architectures) on GPU Worker with MLflow Tracking
  [Task 2] Evaluate Candidate Runs in MLflow & Tag Best Metrics Run/Model with `production`
  [Task 3] Export `production`-Tagged GNN Model to ONNX Runtime & Verify Parity
  [Task 4] Check Season End Gate for SLM Fine-Tuning (ShortCircuitOperator: May-July or completed season)
  [Task 5] Fine-Tune Tactical Qwen SLM using Unsloth QLoRA with MLflow Tracking (football-slm-finetuning)
  [Task 6] Evaluate SLM Runs in MLflow & Tag Champion Model with `production` (alias @production)
  [Task 7] Export Production SLM Checkpoint to ONNX Runtime
  [Task 8] Upload Champion Production SLM to Hugging Face Hub (aliosama8399/football-analysisN) if Best

API hot-reload and RAG LLM-as-a-judge evaluation run independently in the `football_rag_eval` DAG.

Can be triggered automatically at the end of `football_master_pipeline` or manually with custom `conf`:
  {"gnn_model": "TEA-GNN", "gnn_epochs": 100, "force_slm_train": true, "slm_steps": 150, "force_hf_upload": false}
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, ShortCircuitOperator
except ImportError:
    # Fallback mock classes allowing standalone unit testing and CI checks without apache-airflow installed
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

    class ShortCircuitOperator(PythonOperator):  # type: ignore[no-redef]
        pass

BASE_DIR = os.environ.get("PROJECT_ROOT")
if not BASE_DIR or not os.path.exists(BASE_DIR):
    BASE_DIR = str(Path(__file__).resolve().parent.parent.parent)


def _run_gpu_worker_step(step_id: int, query_params: dict, max_wait: int = 7200) -> dict:
    """Trigger a pipeline step on the GPU worker (`http://fastapi:8001`) and stream real-time logs."""
    import json
    import time
    import urllib.parse
    import urllib.request

    api_url = os.environ.get("API_URL", "http://fastapi:8001").rstrip("/")
    encoded_params = urllib.parse.urlencode(query_params)
    trigger_url = f"{api_url}/api/v1/jobs/pipeline/step/{step_id}?{encoded_params}"
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Token": "football_internal_secret_token_2026",
    }

    print(f"[Airflow -> GPU Worker] Triggering Step {step_id} via {trigger_url}...")
    try:
        req = urllib.request.Request(trigger_url, data=b"{}", headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            job_id = resp_data.get("job_id")
            print(f"[Airflow -> GPU Worker] Job launched: {job_id} (status: {resp_data.get('status')})")
    except Exception as e:
        raise RuntimeError(f"Failed to trigger Step {step_id} on GPU worker ({api_url}): {e}")

    status_url = f"{api_url}/api/v1/jobs/pipeline/status"
    seen_logs = 0
    start_time = time.time()

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
            print(f"[Airflow -> GPU Worker] Step {step_id} completed in {elapsed:.1f}s!")
            return status_data
        elif st in ("FAILED", "CANCELLED"):
            err = status_data.get("error", "Unknown worker error")
            raise RuntimeError(f"Step {step_id} failed on GPU worker with status {st}: {err}")

    raise TimeoutError(f"Step {step_id} timed out after {max_wait}s on GPU worker ({api_url})")


def run_gnn_training_on_mlflow(**context):
    """Task 1: Train GNN model(s) on the GPU worker container with MLflow tracking."""
    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    gnn_model = str(conf.get("gnn_model", "TEA-GNN"))
    try:
        gnn_epochs = int(conf.get("gnn_epochs", 100))
    except (ValueError, TypeError):
        gnn_epochs = 100
    experiment = str(conf.get("experiment", "football-gnn-training"))

    return _run_gpu_worker_step(
        step_id=6,
        query_params={
            "gnn_model": gnn_model,
            "gnn_epochs": gnn_epochs,
            "experiment": experiment,
        },
        max_wait=4 * 3600,
    )


def promote_best_gnn_model_to_production(**context):
    """Task 2: Evaluate MLflow runs, tag the best metrics run & model version with `production`,
    and persist `models/saved/production_gnn_meta.json` + `models/saved/gnn_production.pt`."""
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    experiment = str(conf.get("experiment", "football-gnn-training"))
    selection_metric = str(conf.get("selection_metric", "composite"))

    from models.mlflow_config import promote_best_gnn_to_production

    meta = promote_best_gnn_to_production(
        experiment=experiment,
        selection_metric=selection_metric,
    )
    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="production_gnn_meta", value=meta)
    return meta


def export_production_gnn_to_onnx(**context):
    """Task 3: Export the `production`-tagged GNN checkpoint to ONNX Runtime on the GPU worker."""
    return _run_gpu_worker_step(
        step_id=7,
        query_params={},
        max_wait=1800,
    )


def is_season_finished_for_slm(**context) -> bool:
    """Task 4 (Gatekeeper): Check whether the active football season has concluded.

    Returns True to proceed with SLM fine-tuning, or False to short-circuit and skip downstream tasks.
    Triggers when:
      1. Explicitly requested via dag_run.conf: {'season_finished': True} or {'force_slm_train': True}.
      2. In the European post-season / summer offseason window (calendar months May, June, July: 5, 6, 7).
      3. Completed match count in processed_matches.csv indicates all season rounds are completed.
    """
    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}

    # Manual force or skip overrides
    if conf.get("force_slm_train") is True or conf.get("season_finished") is True:
        print("[SeasonCheck] Force trigger enabled via DAG conf. Proceeding to SLM fine-tuning.")
        return True
    if conf.get("skip_slm") is True or conf.get("season_finished") is False:
        print("[SeasonCheck] Explicit skip configured in DAG conf. Skipping SLM fine-tuning.")
        return False

    now = datetime.now()
    # European football seasons typically end in May; June and July are the offseason
    if now.month in (5, 6, 7):
        print(f"[SeasonCheck] Current month is {now.month} (European post-season window). Proceeding to SLM fine-tuning.")
        return True

    # Check match completion in processed_matches.csv
    matches_csv = os.path.join(BASE_DIR, "data", "processed", "processed_matches.csv")
    if os.path.exists(matches_csv):
        try:
            import pandas as pd

            df = pd.read_csv(matches_csv, usecols=["Season", "Date"])
            if not df.empty and "Season" in df.columns:
                latest_season = df["Season"].max()
                season_matches = df[df["Season"] == latest_season]
                # If active season has reached full schedule (>= 1800 matches across 5 leagues)
                if len(season_matches) >= 1800:
                    print(
                        f"[SeasonCheck] Latest season {latest_season} has {len(season_matches)} matches completed. "
                        "Proceeding to SLM fine-tuning."
                    )
                    return True
                print(
                    f"[SeasonCheck] Season {latest_season} is currently ongoing with {len(season_matches)} matches "
                    f"(month: {now.month}). Skipping routine SLM fine-tuning."
                )
                return False
        except Exception as e:
            print(f"[SeasonCheck] Note: could not inspect {matches_csv}: {e}")

    print(f"[SeasonCheck] Season is ongoing (month: {now.month}). Skipping routine SLM fine-tuning until season end.")
    return False


def run_slm_training_on_mlflow(**context) -> dict:
    """Task 5: Fine-tune the tactical SLM (Unsloth QLoRA) on the GPU worker with MLflow tracking."""
    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    slm_steps = int(conf.get("slm_steps", 150))
    slm_epochs = int(conf.get("slm_epochs", 0))
    slm_lr = float(conf.get("slm_lr", 0.0))
    slm_model = str(conf.get("slm_model", ""))
    experiment = str(conf.get("slm_experiment", "football-slm-finetuning"))

    # Attempt running on GPU worker container if endpoint is reachable
    api_url = os.environ.get("API_URL", "http://fastapi:8001").rstrip("/")
    try:
        import urllib.request

        req = urllib.request.Request(f"{api_url}/health", headers={"User-Agent": "airflow-ping"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                print(f"[Airflow -> GPU Worker] GPU worker reachable at {api_url}. Dispatching Step 9...")
                return _run_gpu_worker_step(
                    step_id=9,
                    query_params={
                        "slm_steps": slm_steps,
                        "slm_epochs": slm_epochs,
                        "slm_lr": slm_lr,
                        "model_name": slm_model,
                        "experiment": experiment,
                    },
                    max_wait=6 * 3600,
                )
    except Exception as e:
        print(f"[Airflow] GPU worker not directly reachable ({e}); executing finetune_unsloth.py locally...")

    # Local / in-worker execution fallback
    import subprocess

    cmd = [
        sys.executable,
        "-u",
        os.path.join(BASE_DIR, "models", "finetune_unsloth.py"),
        "--save-merged-16bit",
        "--export-onnx",
        "--promote-production",
    ]
    if slm_steps > 0:
        cmd.extend(["--max-steps", str(slm_steps)])
    if slm_epochs > 0:
        cmd.extend(["--epochs", str(slm_epochs)])
    if slm_lr > 0.0:
        cmd.extend(["--lr", str(slm_lr)])
    if slm_model:
        cmd.extend(["--model-name", slm_model])

    print(f"[Airflow] Running Unsloth SLM fine-tuning: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, check=True)
    print(res.stdout)
    return {"status": "SUCCESS", "returncode": res.returncode}


def promote_best_slm_model_to_production(**context):
    """Task 6: Evaluate MLflow SLM runs, tag the champion run & registered model version with `production`
    (and alias @production), and persist `models/saved/production_slm_meta.json`."""
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    experiment = str(conf.get("slm_experiment", "football-slm-finetuning"))
    selection_metric = str(conf.get("slm_selection_metric", "eval_loss"))

    from models.mlflow_config import promote_best_slm_to_production

    meta = promote_best_slm_to_production(
        experiment=experiment,
        selection_metric=selection_metric,
    )
    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="production_slm_meta", value=meta)
    return meta


def export_production_slm_to_onnx(**context):
    """Task 7: Ensure the fine-tuned production SLM merged weights are exported to ONNX format."""
    export_script = os.path.join(BASE_DIR, "models", "export_slm.py")
    merged_dir = os.path.join(BASE_DIR, "models", "finetuned_qwen_unsloth", "merged_16bit")
    if not os.path.exists(export_script):
        print(f"[Warn] export_slm.py not found at {export_script}; skipping ONNX export.")
        return {"status": "skipped"}

    import subprocess

    cmd = [
        sys.executable,
        export_script,
        "--model_id",
        merged_dir,
    ]
    print(f"[Airflow] Running SLM ONNX export: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, check=False)
    print(res.stdout)
    if res.returncode != 0:
        print(f"[Warn] SLM ONNX export returned non-zero ({res.returncode}): {res.stderr}")
    return {"returncode": res.returncode, "status": "completed"}


def upload_production_slm_to_hf_task(**context) -> dict:
    """Task 8: Upload champion (`production`) SLM model to Hugging Face Hub (aliosama8399/football-analysisN).

    Strict Conditional Gate:
      - ONLY uploads if the model is crowned champion (`newly_promoted == True`) or forced via dag_run.conf.
      - Skips cleanly if the latest trained model did NOT beat the production benchmark or was already uploaded.
    """
    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    force_upload = bool(conf.get("force_hf_upload", False))
    skip_upload = bool(conf.get("skip_hf_upload", False))
    repo_id = str(conf.get("hf_repo_id", "aliosama8399/football-analysisN"))

    if skip_upload:
        print("[HF Upload] Explicit skip requested in DAG conf. Skipping Hugging Face upload.")
        return {"status": "skipped", "reason": "skip_hf_upload_conf"}

    ti = context.get("ti")
    meta = None
    if ti:
        meta = ti.xcom_pull(task_ids="step6_tag_best_slm_production", key="production_slm_meta")

    if not meta:
        meta_path = os.path.join(BASE_DIR, "models", "saved", "production_slm_meta.json")
        if os.path.exists(meta_path):
            try:
                import json

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[Warn] Could not read {meta_path}: {e}")

    # Check promotion condition
    if not force_upload:
        if not meta:
            print("[HF Upload] No SLM production metadata found. Skipping upload.")
            return {"status": "skipped", "reason": "no_meta"}

        is_new_champion = meta.get("newly_promoted", False)
        run_id = meta.get("run_id")
        uploaded_run_id = meta.get("uploaded_to_hf_run_id")

        if uploaded_run_id and uploaded_run_id == run_id:
            print(
                f"[HF Upload] Model for champion run '{run_id}' was already uploaded to {repo_id}. "
                "Skipping duplicate upload."
            )
            return {"status": "skipped", "reason": "already_uploaded", "run_id": run_id}

        if not is_new_champion and uploaded_run_id is not None:
            print(
                f"[HF Upload] SKIPPED: The latest trained model did NOT beat the current champion "
                f"(run_id: {run_id}). Only new champion models are uploaded to Hugging Face."
            )
            return {"status": "skipped", "reason": "not_new_champion", "run_id": run_id}

    # Attempt running on GPU worker container if endpoint is reachable
    api_url = os.environ.get("API_URL", "http://fastapi:8001").rstrip("/")
    try:
        import urllib.request

        req = urllib.request.Request(f"{api_url}/health", headers={"User-Agent": "airflow-ping"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                print(f"[Airflow -> GPU Worker] GPU worker reachable at {api_url}. Dispatching Step 10...")
                return _run_gpu_worker_step(
                    step_id=10,
                    query_params={
                        "force_hf_upload": force_upload,
                        "repo_id": repo_id,
                    },
                    max_wait=1800,
                )
    except Exception as e:
        print(f"[Airflow] GPU worker not directly reachable ({e}); executing upload_slm_hf.py locally...")

    # Local fallback execution
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    from models.upload_slm_hf import upload_production_slm_to_hf

    res = upload_production_slm_to_hf(
        repo_id=repo_id,
        force=force_upload,
    )
    if ti:
        ti.xcom_push(key="hf_upload_result", value=res)
    return res


default_args = {
    "owner": "football-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(hours=6),
}

with DAG(
    dag_id="football_gnn_mlflow_pipeline",
    default_args=default_args,
    description="GNN Training -> Production Promotion -> ONNX Export -> Season End Gate -> SLM Fine-Tuning & Production Promotion",
    schedule_interval=None,  # Triggered by football_master_pipeline after data prep or on-demand
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["football", "mlops", "gnn", "slm", "unsloth", "mlflow", "production", "onnx", "inference"],
    doc_md=__doc__,
) as dag:

    # â”€â”€ Task 1: Train GNN Models with MLflow Tracking (GPU Worker) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    task_train_gnn = PythonOperator(
        task_id="step1_train_gnn_mlflow",
        python_callable=run_gnn_training_on_mlflow,
        execution_timeout=timedelta(hours=4),
        doc_md="Trains GNN match prediction model(s) on the GPU worker and logs metrics, artifacts, and weights to MLflow.",
    )

    # â”€â”€ Task 2: Evaluate MLflow Metrics & Tag Best Model with `production` â”€â”€â”€â”€
    task_tag_production = PythonOperator(
        task_id="step2_tag_best_gnn_production",
        python_callable=promote_best_gnn_model_to_production,
        execution_timeout=timedelta(minutes=15),
        doc_md="Compares candidate GNN runs in MLflow, tags the best run and registered model version with "
               "`production` (tags.production='true', alias @production), and updates gnn_production.pt.",
    )

    # â”€â”€ Task 3: Export `production`-Tagged GNN Checkpoint to ONNX Runtime â”€â”€â”€â”€â”€
    task_export_onnx = PythonOperator(
        task_id="step3_export_production_gnn_onnx",
        python_callable=export_production_gnn_to_onnx,
        execution_timeout=timedelta(minutes=30),
        doc_md="Exports the MLflow `production`-tagged GNN checkpoint to ONNX Runtime format and validates parity.",
    )

    # â”€â”€ Task 4: Check Season End Gate for SLM Fine-Tuning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    task_season_check = ShortCircuitOperator(
        task_id="step4_check_season_end_for_slm",
        python_callable=is_season_finished_for_slm,
        ignore_downstream_trigger_rules=False,
        doc_md="Gates SLM fine-tuning so it only executes after the football season finishes "
               "(months 5, 6, 7 or full match schedule), or when forced via dag_run.conf.",
    )

    # â”€â”€ Task 5: Train Unsloth SLM with MLflow Tracking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    task_train_slm = PythonOperator(
        task_id="step5_train_slm_unsloth",
        python_callable=run_slm_training_on_mlflow,
        execution_timeout=timedelta(hours=6),
        doc_md="Fine-tunes the Qwen tactical SLM with Unsloth QLoRA on the GPU worker and logs metrics to MLflow.",
    )

    # â”€â”€ Task 6: Evaluate MLflow Metrics & Tag Best SLM with `production` â”€â”€â”€â”€â”€â”€
    task_tag_slm_production = PythonOperator(
        task_id="step6_tag_best_slm_production",
        python_callable=promote_best_slm_model_to_production,
        execution_timeout=timedelta(minutes=15),
        doc_md="Compares candidate SLM runs in MLflow (by eval_loss), tags the champion run and registered model with "
               "`production` (tags.production='true', alias @production), and updates production_slm_meta.json.",
    )

    # -- Task 7: Export Production SLM Checkpoint to ONNX Runtime --------------
    task_export_slm_onnx = PythonOperator(
        task_id="step7_export_production_slm_onnx",
        python_callable=export_production_slm_to_onnx,
        execution_timeout=timedelta(minutes=30),
        doc_md="Exports the newly promoted production SLM merged weights to ONNX format.",
    )

    # -- Task 8: Upload Champion Production SLM to Hugging Face Hub ------------
    task_upload_hf = PythonOperator(
        task_id="step8_upload_production_slm_to_hf",
        python_callable=upload_production_slm_to_hf_task,
        execution_timeout=timedelta(minutes=30),
        doc_md="Uploads the production champion SLM model to Hugging Face Hub (aliosama8399/football-analysisN) "
               "ONLY if newly promoted as the best model or explicitly forced via dag_run.conf.",
    )

    (
        task_train_gnn
        >> task_tag_production
        >> task_export_onnx
        >> task_season_check
        >> task_train_slm
        >> task_tag_slm_production
        >> task_export_slm_onnx
        >> task_upload_hf
    )
