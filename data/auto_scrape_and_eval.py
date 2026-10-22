"""
Lightweight Automated Data Scraping & Continuous Evaluation Runner
===================================================================
Replaces heavyweight Apache Airflow DAGs with an efficient, Python-native
automation script that:
  1. Periodically scrapes new match results (football-data.co.uk & Understat)
  2. Updates raw/processed datasets without external orchestrator overhead
  3. Automatically triggers both RAG evaluation (eval/run_rag_eval.py) and
     Match Prediction evaluation (eval/run_prediction_eval.py)
  4. Streams updated benchmarks to Prometheus and Grafana in real-time

Usage:
  # One-shot immediate scrape + eval
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


def run_data_scrape(quick: bool = False) -> bool:
    """Run data collectors to ingest latest match scores and stats."""
    logger.info("=== STEP 1: Running Automated Match Data Scraping ===")
    try:
        from data.collect_all import collect_football_data_uk
        logger.info("Downloading latest match data from football-data.co.uk...")
        df = collect_football_data_uk()
        logger.info("Successfully fetched %d matches from football-data.co.uk.", len(df) if df is not None else 0)

        # Run preprocessing to update team form, xG, and processed features
        preprocess_script = BASE_DIR / "data" / "preprocess.py"
        if preprocess_script.exists():
            logger.info("Running data/preprocess.py to refresh tabular features...")
            subprocess.run([sys.executable, str(preprocess_script)], cwd=str(BASE_DIR), env=os.environ, check=True)
        return True
    except Exception as e:
        logger.error("Data scraping or preprocessing failed: %s", e)
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


def execute_cycle(skip_scrape: bool = False, judge: str = "heuristic", samples: int = 15):
    """Execute a single pipeline cycle."""
    start_time = time.time()
    logger.info("Starting pipeline execution cycle at %s", datetime.now().isoformat())

    if not skip_scrape:
        scrape_ok = run_data_scrape()
        if not scrape_ok:
            logger.warning("Scrape step encountered warnings; continuing to evaluation...")
    else:
        logger.info("Skipping scrape step as requested.")

    eval_ok = run_evaluations(judge=judge, samples=samples)
    duration = time.time() - start_time
    logger.info("Pipeline cycle completed in %.2f seconds (eval_ok=%s).", duration, eval_ok)


def main():
    parser = argparse.ArgumentParser(description="Automated Scrape & Continuous Evaluation Pipeline")
    parser.add_argument("--now", action="store_true", help="Run a single cycle immediately and exit")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip data scraping and run evaluation only")
    parser.add_argument("--judge", default="heuristic", choices=["heuristic", "vllm", "gemini"],
                        help="LLM judge mode for RAG evaluation (default: heuristic for 0-quota)")
    parser.add_argument("--samples", type=int, default=15, help="Number of evaluation samples to evaluate (default: 15)")
    parser.add_argument("--interval-hours", type=float, default=24.0,
                        help="Hours between automated pipeline runs (default: 24.0)")

    args = parser.parse_args()

    if args.now:
        execute_cycle(skip_scrape=args.skip_scrape, judge=args.judge, samples=args.samples)
        return

    logger.info("Starting continuous runner (interval: %.1f hours)... Press Ctrl+C to stop.", args.interval_hours)
    sleep_seconds = args.interval_hours * 3600.0

    while True:
        try:
            execute_cycle(skip_scrape=args.skip_scrape, judge=args.judge)
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
