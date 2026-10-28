"""
Automated End-to-End Football ML Pipeline
=========================================
Automates the full 8-stage flow:
  [STEP 1/8] Incremental match data collection (finished season 2526 + active season 2627)
  [STEP 2/8] Data preprocessing and feature engineering (data/preprocess.py)
  [STEP 3/8] Team tactics extraction and knowledge/vector store sync (rag/extract_tactics.py + DB/KG/Vectors)
  [STEP 4/8] Teacher LLM tactical generation via Ollama (data/build_finetune_dataset.py)
  [STEP 5/8] Re-splitting train/val dataset for SLM fine-tuning
  [STEP 6/8] GNN model training with MLflow tracking (models/train_gnn.py)
  [STEP 7/8] GNN ONNX model export & parity validation (models/export_gnn.py --dynamic)
  [STEP 8/8] Live FastAPI singleton hot-reload (POST /api/v1/admin/reload)

Usage:
  # Run entire pipeline with Ollama and MLflow tracking (default)
  python data/run_pipeline.py

  # Test run: dry-run teacher LLM and fast 5-epoch GNN training
  python data/run_pipeline.py --teacher-samples 5 --gnn-epochs 5

  # Skip scraping (if raw CSVs already present)
  python data/run_pipeline.py --skip-scrape

  # Run only through fine-tuning data prep (skip GNN training)
  python data/run_pipeline.py --skip-gnn
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError

# Ensure stdout/stderr handle UTF-8 cleanly on Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (pipeline) %(message)s"
)
logger = logging.getLogger("football_pipeline")


def _print_step_banner(step_num: int, total_steps: int, title: str, icon: str = ">>"):
    divider = "=" * 78
    print(f"\n{divider}")
    print(f" {icon} [STEP {step_num}/{total_steps}] {title.upper()}")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{divider}\n")


def _run_subcommand(cmd: list[str], desc: str, env: dict | None = None) -> bool:
    """Run an external python or CLI process, streaming output in real-time."""
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info("Executing: %s", cmd_str)
    start_t = time.time()
    
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env=run_env,
            check=True
        )
        elapsed = time.time() - start_t
        logger.info("[OK] %s completed in %.1f seconds (code %d)", desc, elapsed, proc.returncode)
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_t
        logger.error("[FAIL] %s failed after %.1f seconds with exit code %d", desc, elapsed, e.returncode)
        return False
    except Exception as e:
        logger.error("[FAIL] %s encountered unexpected error: %s", desc, e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def step1_data_collection(force: bool = False) -> bool:
    """Incremental fetch of finished and current football seasons."""
    from data.collectors.football_data_uk import download_all_data
    from data.collectors.understat_scraper import main as run_understat_scraper

    logger.info("Running incremental data download from football-data.co.uk...")
    df_fduk = download_all_data(output_dir="data/raw", incremental=not force, force=force)
    logger.info("Football-data.co.uk ingest complete: %d matches in combined set.", len(df_fduk) if df_fduk is not None else 0)

    logger.info("Running Understat xG collector via soccerdata (incremental=%s)...", not force)
    try:
        run_understat_scraper(incremental=not force, force=force)
        logger.info("Understat xG ingest complete.")
    except Exception as e:
        logger.warning("Understat collector encountered non-fatal error: %s", e)

    return True


def step2_preprocessing() -> bool:
    """Preprocess raw matches into ML-ready features (data/processed/processed_matches.csv)."""
    preprocess_script = BASE_DIR / "data" / "preprocess.py"
    if not preprocess_script.exists():
        logger.error("data/preprocess.py not found!")
        return False

    return _run_subcommand(
        [sys.executable, str(preprocess_script)],
        desc="Data Preprocessing & Feature Engineering"
    )


def step3_team_tactics(sync_db: bool = False) -> bool:
    """Update team_tactics.json with grounded stats and optionally sync all DB/KG/Vector stores."""
    tactics_script = BASE_DIR / "rag" / "extract_tactics.py"
    if not tactics_script.exists():
        logger.error("rag/extract_tactics.py not found!")
        return False

    ok = _run_subcommand(
        [sys.executable, str(tactics_script)],
        desc="Team Tactics Grounded Extraction"
    )
    if not ok:
        return False

    if sync_db:
        logger.info("Syncing updated tactical profiles & matches to Knowledge Graph, DB & Vector Indexes...")
        neo4j_script = BASE_DIR / "rag" / "build_neo4j_kg.py"
        pg_script = BASE_DIR / "rag" / "build_postgres_db.py"
        faiss_script = BASE_DIR / "rag" / "build_faiss_index.py"
        pgvector_script = BASE_DIR / "rag" / "build_pgvector_index.py"

        if pg_script.exists():
            _run_subcommand([sys.executable, str(pg_script)], desc="PostgreSQL DB Tactics & Match Sync")
        if neo4j_script.exists():
            _run_subcommand([sys.executable, str(neo4j_script)], desc="Neo4j KG Tactics Sync")
        if faiss_script.exists():
            _run_subcommand([sys.executable, str(faiss_script)], desc="FAISS Vector Store Rebuild")
        if pgvector_script.exists():
            _run_subcommand([sys.executable, str(pgvector_script), "--rebuild"], desc="pgvector Index Rebuild")

    return True


def step4_teacher_tactics(
    provider: str = "ollama",
    model_name: str = "",
    samples: int = 0,
    seasons: list[int] | None = None,
    dry_run: bool = False
) -> bool:
    """Call Teacher LLM (Ollama) to generate tactical analyses for un-annotated matches."""
    dataset_script = BASE_DIR / "data" / "build_finetune_dataset.py"
    if not dataset_script.exists():
        logger.error("data/build_finetune_dataset.py not found!")
        return False

    cmd = [
        sys.executable,
        str(dataset_script),
        "--provider", provider,
        "--incremental",
    ]
    if model_name:
        cmd.extend(["--model-name", model_name])
    if samples > 0:
        cmd.extend(["--max-samples", str(samples)])
    if seasons:
        cmd.append("--seasons")
        cmd.extend(str(s) for s in seasons)
    if dry_run:
        cmd.append("--dry-run")

    return _run_subcommand(
        cmd,
        desc=f"Teacher LLM Tactical Generation ({provider})"
    )


def step5_resplit_dataset() -> bool:
    """Rebuild and verify train/val splits in data/finetune/ and models/."""
    jsonl_path = BASE_DIR / "data" / "finetune" / "football_tactical.jsonl"
    if not jsonl_path.exists():
        logger.warning("No tactical dataset found at %s; skipping split rebuild.", jsonl_path)
        return False

    from data.build_finetune_dataset import rebuild_splits
    import json

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    train_n, val_n = rebuild_splits(records, output_dir=BASE_DIR / "data" / "finetune")
    logger.info("Dataset splits rebuilt: %d training, %d validation records.", train_n, val_n)
    return train_n > 0


def step6_train_gnn(
    model: str = "TEA-GNN",
    epochs: int = 100,
    experiment: str = "football-gnn-training"
) -> bool:
    """Train GNN models and log parameters, metrics, and artifacts to MLflow."""
    train_script = BASE_DIR / "models" / "train_gnn.py"
    if not train_script.exists():
        logger.error("models/train_gnn.py not found!")
        return False

    cmd = [
        sys.executable,
        str(train_script),
        "--model", model,
        "--epochs", str(epochs),
        "--experiment", experiment
    ]
    return _run_subcommand(cmd, desc=f"GNN Training ({model}) with MLflow")


def step7_export_gnn_onnx(dynamic: bool = True) -> bool:
    """Export newly trained TEA-GNN checkpoint to ONNX Runtime format and verify parity."""
    export_script = BASE_DIR / "models" / "export_gnn.py"
    if not export_script.exists():
        logger.error("models/export_gnn.py not found!")
        return False

    cmd = [sys.executable, str(export_script)]
    if dynamic:
        cmd.append("--dynamic")
    return _run_subcommand(cmd, desc="GNN ONNX Export & Parity Verification")


def step8_reload_api() -> bool:
    """Hot-reload live FastAPI singletons so newly exported ONNX/KG/Vector stores take effect."""
    api_url = os.environ.get("API_URL", "http://localhost:8001").rstrip("/")
    reload_url = f"{api_url}/api/v1/admin/reload"
    logger.info("Notifying live FastAPI service at %s to hot-reload singletons...", reload_url)
    try:
        req = urlrequest.Request(reload_url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        with urlrequest.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            logger.info("[OK] API hot-reloaded (%d): %s", resp.status, body)
            return resp.status == 200
    except URLError as e:
        logger.warning("[WARN] Live API at %s not reachable (skipping hot-reload): %s", reload_url, e)
        return True
    except Exception as e:
        logger.warning("[WARN] Error calling %s: %s", reload_url, e)
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Automated Football Ingestion, Tactics, Fine-Tuning & ONNX Export Pipeline"
    )
    # Step toggles
    parser.add_argument("--skip-scrape",     action="store_true", help="Skip Step 1 (incremental scraping)")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip Step 2 (feature preprocessing)")
    parser.add_argument("--skip-tactics",    action="store_true", help="Skip Step 3 (team tactics extraction)")
    parser.add_argument("--skip-teacher",    action="store_true", help="Skip Step 4 (teacher LLM annotation)")
    parser.add_argument("--skip-split",      action="store_true", help="Skip Step 5 (dataset train/val split)")
    parser.add_argument("--skip-gnn",        action="store_true", help="Skip Step 6 (GNN training with MLflow)")
    parser.add_argument("--skip-export",     action="store_true", help="Skip Step 7 (GNN ONNX export)")
    parser.add_argument("--skip-reload",     action="store_true", help="Skip Step 8 (Live API hot-reload)")

    # Data options
    parser.add_argument("--force-scrape",    action="store_true", help="Force redownload of all seasons from scratch")
    parser.add_argument("--sync-db",         action="store_true", help="Sync updated tactics & matches to Neo4j, Postgres, FAISS & pgvector")

    # Teacher LLM options
    parser.add_argument("--provider",        type=str, default="ollama", choices=["ollama", "gemini", "vllm", "openai"],
                        help="Teacher LLM provider (default: ollama)")
    parser.add_argument("--model-name",      type=str, default="", help="Specific teacher model name (e.g. qwen3.5:0.8b)")
    parser.add_argument("--teacher-samples", type=int, default=0,
                        help="Number of matches to annotate with Teacher LLM (0 = all new)")
    parser.add_argument("--teacher-seasons",  nargs="+", type=int, default=None,
                        help="Specific season codes for teacher annotation (default: all un-annotated seasons dynamically)")
    parser.add_argument("--dry-run-teacher", action="store_true", help="Dry-run teacher LLM prompt generation without API calls")

    # GNN options
    parser.add_argument("--gnn-model",       type=str, default="TEA-GNN",
                        help="GNN model architecture (default: TEA-GNN, or 'all', 'Hybrid', 'EdgeConv')")
    parser.add_argument("--gnn-epochs",      type=int, default=100, help="Number of training epochs for GNN (default: 100)")
    parser.add_argument("--experiment",      type=str, default="football-gnn-training",
                        help="MLflow experiment name (default: football-gnn-training)")

    # Automation / Scheduling options
    parser.add_argument("--interval-hours",  type=float, default=0.0,
                        help="Automated continuous loop: run every N hours (e.g. 24.0). 0 = single immediate run.")

    args = parser.parse_args()

    if args.interval_hours > 0:
        logger.info("Starting automated pipeline daemon (interval: %.1f hours). Press Ctrl+C to stop.", args.interval_hours)
        sleep_sec = args.interval_hours * 3600.0
        while True:
            try:
                run_single_cycle(args)
            except KeyboardInterrupt:
                logger.info("Automated pipeline interrupted by user. Exiting.")
                break
            except Exception as e:
                logger.error("Automated cycle encountered error: %s", e)
            logger.info("Automated daemon sleeping %.1f hours until next scheduled cycle...", args.interval_hours)
            time.sleep(sleep_sec)
    else:
        run_single_cycle(args)


def run_single_cycle(args):
    total_steps = 8
    pipeline_start = time.time()

    from data._config import get_active_season, get_latest_completed_season
    active_season = get_active_season()
    completed_season = get_latest_completed_season()

    print("\n" + "#" * 78)
    print("  FOOTBALL AUTOMATED DATA-TO-FINE-TUNING & ONNX PIPELINE")
    print(f"  Execution Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Active Season:      {active_season} (Latest Completed: {completed_season})")
    print(f"  Teacher Provider:   {args.provider} ({args.model_name or 'default'})")
    print(f"  GNN Architecture:   {args.gnn_model} ({args.gnn_epochs} epochs, MLflow -> ONNX)")
    print("#" * 78)

    # ── Step 1: Incremental Data Collection ──
    _print_step_banner(1, total_steps, f"Incremental Data Ingestion (Finished <= {completed_season} + Active {active_season})", icon="[1]")
    if not args.skip_scrape:
        ok = step1_data_collection(force=args.force_scrape)
        if not ok:
            logger.error("Step 1 failed! Halting pipeline.")
            sys.exit(1)
    else:
        logger.info("Skipping Step 1 as requested (--skip-scrape).")

    # ── Step 2: Preprocessing & Feature Engineering ──
    _print_step_banner(2, total_steps, "Preprocessing & Feature Engineering", icon="[2]")
    if not args.skip_preprocess:
        ok = step2_preprocessing()
        if not ok:
            logger.error("Step 2 failed! Halting pipeline.")
            sys.exit(1)
    else:
        logger.info("Skipping Step 2 as requested (--skip-preprocess).")

    # ── Step 3: Team Tactics & Knowledge Store Refresh ──
    _print_step_banner(3, total_steps, "Team Tactics Extraction & Knowledge/Vector Store Sync", icon="[3]")
    if not args.skip_tactics:
        ok = step3_team_tactics(sync_db=args.sync_db)
        if not ok:
            logger.warning("Step 3 encountered warnings.")
    else:
        logger.info("Skipping Step 3 as requested (--skip-tactics).")

    # ── Step 4: Teacher LLM Tactical Annotation ──
    _print_step_banner(4, total_steps, f"Teacher LLM Tactical Annotation ({args.provider})", icon="[4]")
    if not args.skip_teacher:
        ok = step4_teacher_tactics(
            provider=args.provider,
            model_name=args.model_name,
            samples=args.teacher_samples,
            seasons=args.teacher_seasons,
            dry_run=args.dry_run_teacher
        )
        if not ok:
            logger.warning("Step 4 encountered errors during teacher generation.")
    else:
        logger.info("Skipping Step 4 as requested (--skip-teacher).")

    # ── Step 5: Train/Val Dataset Re-Splitting ──
    _print_step_banner(5, total_steps, "Re-splitting Train/Val Dataset for Fine-Tuning", icon="[5]")
    if not args.skip_split:
        ok = step5_resplit_dataset()
        if not ok:
            logger.warning("Step 5 split rebuild finished with warnings.")
    else:
        logger.info("Skipping Step 5 as requested (--skip-split).")

    # ── Step 6: Train GNN with MLflow ──
    _print_step_banner(6, total_steps, f"Train GNN Model ({args.gnn_model}) with MLflow Tracking", icon="[6]")
    if not args.skip_gnn:
        ok = step6_train_gnn(
            model=args.gnn_model,
            epochs=args.gnn_epochs,
            experiment=args.experiment
        )
        if not ok:
            logger.warning("Step 6 GNN training finished with warnings.")
    else:
        logger.info("Skipping Step 6 as requested (--skip-gnn).")

    # ── Step 7: Export GNN to ONNX Runtime ──
    _print_step_banner(7, total_steps, "Export TEA-GNN Checkpoint to ONNX Runtime (--dynamic)", icon="[7]")
    if not args.skip_export and not args.skip_gnn:
        ok = step7_export_gnn_onnx(dynamic=True)
        if not ok:
            logger.warning("Step 7 ONNX export finished with warnings.")
    else:
        logger.info("Skipping Step 7 ONNX export (--skip-export or --skip-gnn).")

    # ── Step 8: Hot-Reload Live FastAPI Service ──
    _print_step_banner(8, total_steps, "Hot-Reload Live FastAPI Singletons (/api/v1/admin/reload)", icon="[8]")
    if not args.skip_reload:
        step8_reload_api()
    else:
        logger.info("Skipping Step 8 API reload (--skip-reload).")

    total_time = time.time() - pipeline_start
    print("\n" + "#" * 78)
    print("  [SUCCESS] FULL PIPELINE COMPLETED!")
    print(f"  Total Duration: {total_time / 60:.2f} minutes ({total_time:.1f}s)")
    print(f"  Finished at:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 78 + "\n")


if __name__ == "__main__":
    main()
