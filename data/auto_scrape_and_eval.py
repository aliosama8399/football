"""
Lightweight Automated Data Scraping & Continuous Evaluation Runner
===================================================================
Replaces heavyweight Apache Airflow DAGs with an efficient, Python-native
automation script that:
  1. Periodically scrapes new match results (football-data.co.uk & Understat incremental)
  2. Optionally refreshes per-match player stats (player_match_stats.py)
  3. Updates raw/processed datasets and team_tactics.json
  4. Synchronizes PostgreSQL, Neo4j Knowledge Graph, FAISS, and pgvector indexes
  5. Hot-reloads the running FastAPI service (POST /api/v1/admin/reload)
  6. Automatically triggers both RAG evaluation (eval/run_rag_eval.py) and
     Match Prediction evaluation (eval/run_prediction_eval.py)
  7. Streams updated benchmarks to Prometheus and Grafana in real-time

Usage:
  # One-shot immediate scrape + store sync + reload + eval
  python data/auto_scrape_and_eval.py --now

  # Run eval only without scraping
  python data/auto_scrape_and_eval.py --now --skip-scrape

  # Continuous loop (e.g., every 12 hours)
  python data/auto_scrape_and_eval.py --interval-hours 12
"""

import os
import sys
import time
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from urllib import request as urlrequest
from urllib.error import URLError

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (auto_scrape_eval) %(message)s"
)
logger = logging.getLogger("auto_scrape_eval")


def run_data_scrape(quick: bool = False, with_players: bool = False) -> bool:
    """Run data collectors to ingest latest match scores and stats."""
    logger.info("=== STEP 1: Running Automated Match Data Scraping ===")
    try:
        from data.collectors.football_data_uk import download_all_data
        from data.collectors.understat_scraper import main as run_understat
        from data._config import load_config

        logger.info("Downloading latest match data from football-data.co.uk (incremental)...")
        df = download_all_data(incremental=True)
        logger.info("Successfully ingested %d matches from football-data.co.uk.", len(df) if df is not None else 0)

        logger.info("Fetching Understat xG metrics (incremental)...")
        try:
            run_understat(incremental=True, force=False)
            logger.info("Understat xG ingestion complete.")
        except Exception as e:
            logger.warning("Understat collector encountered non-fatal error: %s", e)

        if with_players:
            player_script = BASE_DIR / "data" / "collectors" / "player_match_stats.py"
            if player_script.exists():
                cfg = load_config()
                leagues_csv = ",".join(cfg.get("leagues", {}).keys()) or "E0,SP1,D1,I1,F1"
                latest_season = str(cfg.get("soccerdata_seasons", ["2526"])[-1])
                logger.info("Refreshing per-match player stats for %s (season %s)...", leagues_csv, latest_season)
                res = subprocess.run(
                    [sys.executable, str(player_script), "--leagues", leagues_csv, "--seasons", latest_season],
                    cwd=str(BASE_DIR),
                    env=os.environ,
                )
                if res.returncode != 0:
                    logger.warning("Player match stats collector exited with code %d (continuing).", res.returncode)

        # Run preprocessing to update team form, xG, and processed features
        preprocess_script = BASE_DIR / "data" / "preprocess.py"
        if preprocess_script.exists():
            logger.info("Running data/preprocess.py to refresh tabular features...")
            subprocess.run([sys.executable, str(preprocess_script)], cwd=str(BASE_DIR), env=os.environ, check=True)

        # Update team tactics knowledge base
        tactics_script = BASE_DIR / "rag" / "extract_tactics.py"
        if tactics_script.exists():
            logger.info("Running rag/extract_tactics.py to refresh team_tactics.json...")
            subprocess.run([sys.executable, str(tactics_script)], cwd=str(BASE_DIR), env=os.environ, check=True)

        return True
    except Exception as e:
        logger.error("Data scraping, preprocessing, or tactics update failed: %s", e)
        return False


def sync_knowledge_stores(skip_vectors: bool = False) -> bool:
    """Synchronize PostgreSQL, Neo4j Knowledge Graph, FAISS, and pgvector indexes."""
    logger.info("=== STEP 1.5: Synchronizing Relational, Graph & Vector Knowledge Stores ===")
    all_ok = True

    sync_jobs = [
        ("PostgreSQL DB Sync", [sys.executable, str(BASE_DIR / "rag" / "build_postgres_db.py")]),
        ("Neo4j KG Sync", [sys.executable, str(BASE_DIR / "rag" / "build_neo4j_kg.py")]),
    ]
    if not skip_vectors:
        sync_jobs.extend([
            ("FAISS Vector Index Rebuild", [sys.executable, str(BASE_DIR / "rag" / "build_faiss_index.py")]),
            ("pgvector Index Rebuild", [sys.executable, str(BASE_DIR / "rag" / "build_pgvector_index.py"), "--rebuild"]),
        ])

    for label, cmd in sync_jobs:
        script_path = Path(cmd[1])
        if not script_path.exists():
            continue
        logger.info("Running %s...", label)
        try:
            res = subprocess.run(cmd, cwd=str(BASE_DIR), env=os.environ, capture_output=True, text=True)
            if res.returncode == 0:
                logger.info("[OK] %s completed.", label)
            else:
                logger.warning("[WARN] %s exited with code %d: %s", label, res.returncode, (res.stderr or res.stdout)[-300:])
                all_ok = False
        except Exception as e:
            logger.warning("[WARN] %s failed: %s", label, e)
            all_ok = False

    return all_ok


def trigger_api_reload() -> bool:
    """Notify the live FastAPI service to hot-reload its RAG, GNN, and KB singletons."""
    api_url = os.environ.get("API_URL", "http://localhost:8001").rstrip("/")
    reload_url = f"{api_url}/api/v1/admin/reload"
    logger.info("Triggering live API hot-reload at %s...", reload_url)
    try:
        req = urlrequest.Request(reload_url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        with urlrequest.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            logger.info("API reload response (%d): %s", resp.status, body)
            return resp.status == 200
    except URLError as e:
        logger.warning("Could not reach live API at %s for hot-reload (service may be offline): %s", reload_url, e)
        return False
    except Exception as e:
        logger.warning("Unexpected error calling %s: %s", reload_url, e)
        return False


def run_evaluations(judge: str = "heuristic", samples: int = 15) -> bool:
    """Run both RAG evaluation and Match Prediction evaluation suites."""
    logger.info("=== STEP 2: Running Automated Continuous Evaluation Suite ===")
    rag_script = BASE_DIR / "eval" / "run_rag_eval.py"
    pred_script = BASE_DIR / "eval" / "run_prediction_eval.py"

    all_success = True

    # 1. RAG evaluation
    if rag_script.exists():
        logger.info("Launching RAG evaluation suite (judge=%s, samples=%d)...", judge, samples)
        try:
            cmd = [sys.executable, str(rag_script), "--judge", judge, "--samples", str(samples)]
            res = subprocess.run(cmd, cwd=str(BASE_DIR), env=os.environ, capture_output=True, text=True)
            if res.returncode == 0:
                logger.info("RAG evaluation completed successfully.")
            else:
                logger.warning("RAG evaluation returned code %d:\n%s", res.returncode, res.stderr[-500:])
                all_success = False
        except Exception as e:
            logger.error("Failed running RAG evaluation: %s", e)
            all_success = False

    # 2. Prediction evaluation
    if pred_script.exists():
        logger.info("Launching Match Prediction evaluation suite (samples=%d)...", samples)
        try:
            cmd = [sys.executable, str(pred_script), "--samples", str(samples)]
            res = subprocess.run(cmd, cwd=str(BASE_DIR), env=os.environ, capture_output=True, text=True)
            if res.returncode == 0:
                logger.info("Match prediction evaluation completed successfully.")
            else:
                logger.warning("Prediction evaluation returned code %d:\n%s", res.returncode, res.stderr[-500:])
                all_success = False
        except Exception as e:
            logger.error("Failed running prediction evaluation: %s", e)
            all_success = False

    return all_success


def execute_cycle(
    skip_scrape: bool = False,
    skip_sync: bool = False,
    skip_vectors: bool = False,
    with_players: bool = False,
    judge: str = "heuristic",
    samples: int = 15,
):
    """Execute a single pipeline cycle."""
    start_time = time.time()
    logger.info("Starting pipeline execution cycle at %s", datetime.now().isoformat())

    if not skip_scrape:
        scrape_ok = run_data_scrape(with_players=with_players)
        if not scrape_ok:
            logger.warning("Scrape step encountered warnings; continuing...")
    else:
        logger.info("Skipping scrape step as requested.")

    if not skip_sync and not skip_scrape:
        sync_knowledge_stores(skip_vectors=skip_vectors)
    elif skip_sync:
        logger.info("Skipping knowledge store synchronization as requested.")

    trigger_api_reload()

    eval_ok = run_evaluations(judge=judge, samples=samples)
    duration = time.time() - start_time
    logger.info("Pipeline cycle completed in %.2f seconds (eval_ok=%s).", duration, eval_ok)


def main():
    parser = argparse.ArgumentParser(description="Automated Scrape & Continuous Evaluation Pipeline")
    parser.add_argument("--now", action="store_true", help="Run a single cycle immediately and exit")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip data scraping and run evaluation only")
    parser.add_argument("--skip-sync", action="store_true", help="Skip Postgres/Neo4j/Vector store synchronization")
    parser.add_argument("--skip-vectors", action="store_true", help="Sync Postgres/Neo4j only; skip FAISS/pgvector rebuild")
    parser.add_argument("--with-players", action="store_true", help="Also refresh Understat per-match player stats")
    parser.add_argument("--judge", default="heuristic", choices=["heuristic", "vllm", "gemini"],
                        help="LLM judge mode for RAG evaluation (default: heuristic for 0-quota)")
    parser.add_argument("--samples", type=int, default=15, help="Number of evaluation samples to evaluate (default: 15)")
    parser.add_argument("--interval-hours", type=float, default=24.0,
                        help="Hours between automated pipeline runs (default: 24.0)")

    args = parser.parse_args()

    if args.now:
        execute_cycle(
            skip_scrape=args.skip_scrape,
            skip_sync=args.skip_sync,
            skip_vectors=args.skip_vectors,
            with_players=args.with_players,
            judge=args.judge,
            samples=args.samples,
        )
        return

    logger.info("Starting continuous runner (interval: %.1f hours)... Press Ctrl+C to stop.", args.interval_hours)
    sleep_seconds = args.interval_hours * 3600.0

    while True:
        try:
            execute_cycle(
                skip_scrape=args.skip_scrape,
                skip_sync=args.skip_sync,
                skip_vectors=args.skip_vectors,
                with_players=args.with_players,
                judge=args.judge,
                samples=args.samples,
            )
            logger.info("Sleeping for %.1f hours until next cycle...", args.interval_hours)
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logger.info("Received interrupt. Stopping runner.")
            break
        except Exception as e:
            logger.error("Unexpected error in runner loop: %s", e)
            time.sleep(60.0)


if __name__ == "__main__":
    main()
