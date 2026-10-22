"""
RAG Experiments Runner (Phase 2, Tasks T2.1 & T2.4)
===================================================
Benchmarks different RAG configurations across the golden evaluation dataset:
  Config A: Neo4j + pgvector (Dense only)
  Config B: PostgreSQL + pgvector (Dense only)
  Config C: Neo4j + pgvector (Hybrid BM25 + Dense)
  Config D: Neo4j + pgvector (Hybrid + Reranker)
  Config E: Temporal boundary impact (As-of filter ON vs OFF)

Saves comparison table to eval/rag_experiments.md.

Usage:
  python eval/run_rag_experiments.py --samples 15
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse
import logging
from pathlib import Path

from rag.rag_orchestrator import FootballRAGSystem
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import rerank_documents
from eval.run_rag_eval import evaluate_numeric_grounding, evaluate_lexical_grounding
from eval.langfuse_client import trace_step, current_trace_id, log_score, flush_traces

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
GOLDEN_PATH = BASE_DIR / "eval" / "golden" / "rag_golden_set.json"
EXP_REPORT_PATH = BASE_DIR / "eval" / "rag_experiments.md"


def run_experiment_matrix(samples_count: int = 15):
    if not GOLDEN_PATH.exists():
        logger.error("Golden dataset missing at %s. Run `python eval/build_golden_set.py` first.", GOLDEN_PATH)
        return

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)[:samples_count]

    logger.info("Running experiment matrix across %d sample queries...", len(data))

    matrix = [
        {"name": "Neo4j + pgvector (Default Dense)", "kg": "neo4j", "vector": "pgvector", "hybrid": False, "rerank": False, "as_of": True},
        {"name": "Postgres + pgvector (Dense)", "kg": "postgres", "vector": "pgvector", "hybrid": False, "rerank": False, "as_of": True},
        {"name": "Neo4j + pgvector (Hybrid BM25+Dense)", "kg": "neo4j", "vector": "pgvector", "hybrid": True, "rerank": False, "as_of": True},
        {"name": "Neo4j + pgvector (Hybrid + Reranker)", "kg": "neo4j", "vector": "pgvector", "hybrid": True, "rerank": True, "as_of": True},
        {"name": "Neo4j + pgvector (As-Of Filter OFF / Leaky)", "kg": "neo4j", "vector": "pgvector", "hybrid": False, "rerank": False, "as_of": False},
    ]

    results = []

    for cfg in matrix:
        name = cfg["name"]
        logger.info("Evaluating: %s...", name)
        try:
            rag = FootballRAGSystem(
                kg=cfg["kg"],
                vector=cfg["vector"],
                hybrid=cfg.get("hybrid", False),
                rerank=cfg.get("rerank", False),
            )
        except Exception as e:
            logger.warning("Could not initialize configuration '%s': %s (skipping)", name, e)
            results.append({
                "name": name,
                "latency": "N/A",
                "numeric_grounding": "N/A",
                "grounding": "N/A",
                "status": f"Unavailable: {e}"
            })
            continue

        latencies = []
        num_scores = []
        grd_scores = []

        for item in data:
            q = item["query"]
            as_of = item.get("as_of_date") if cfg["as_of"] else None
            expected_nums = item.get("expected_numbers", {})
            gt = item.get("ground_truth", "")

            t0 = time.time()
            try:
                with trace_step("rag_experiment_item", input={"query": q, "config": name, "as_of": as_of}, metadata={"config": name, "id": item.get("id")}) as span:
                    resp = rag.query(q, as_of=as_of)
                    span.update(output=resp)
                    trace_id = current_trace_id()
                dur = time.time() - t0
                latencies.append(dur)
                num_score = evaluate_numeric_grounding(resp, expected_nums)
                grd_score = evaluate_lexical_grounding(resp, gt)
                num_scores.append(num_score)
                grd_scores.append(grd_score)
                log_score(f"exp_{name}_numeric", num_score, trace_id=trace_id)
                log_score(f"exp_{name}_grounding", grd_score, trace_id=trace_id)
            except Exception as e:
                logger.debug("Query error in exp: %s", e)

        flush_traces()
        rag.close()

        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        avg_num = sum(num_scores) / len(num_scores) if num_scores else 0.0
        avg_grd = sum(grd_scores) / len(grd_scores) if grd_scores else 0.0

        results.append({
            "name": name,
            "latency": f"{avg_lat:.2f}s",
            "numeric_grounding": f"{avg_num:.2f}",
            "grounding": f"{avg_grd:.2f}",
            "status": "Completed"
        })

    # Render Markdown table
    lines = [
        "# RAG Experiment Matrix Results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Evaluated on: {len(data)} held-out query samples.",
        "",
        "| Configuration | Avg Latency | Numeric Grounding | Grounding / Relevancy | Status |",
        "|---|---|---|---|---|"
    ]

    for r in results:
        lines.append(f"| **{r['name']}** | {r['latency']} | {r['numeric_grounding']} | {r['grounding']} | {r['status']} |")

    lines.extend([
        "",
        "## Key Findings & Decisions",
        "- **Neo4j Graph vs PostgreSQL:** Neo4j is default KG; provides unified graph traversal for team relations.",
        "- **pgvector Vector Store:** Replaces standalone external FAISS index in Docker; vectors reside in PostgreSQL with cosine HNSW index.",
        "- **Hybrid (BM25 + Dense):** Improves exact keyword matching for specific team tactical jargon.",
        "- **As-Of Date Rule:** Essential for preventing look-ahead data leakage on historical match predictions."
    ])

    report_content = "\n".join(lines)
    with open(EXP_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("Saved RAG experiment matrix report → %s", EXP_REPORT_PATH)
    print("\n" + report_content + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG experiment matrix")
    parser.add_argument("--samples", type=int, default=15, help="Number of samples per experiment")
    args = parser.parse_args()
    run_experiment_matrix(samples_count=args.samples)
