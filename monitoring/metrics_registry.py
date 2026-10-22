"""
Central Prometheus Metrics Registry for Football RAG & Prediction System
=========================================================================
Maintains Prometheus metrics for:
  - RAGAS evaluation scores (faithfulness, answer relevancy, context precision, context recall)
  - Match prediction performance (accuracy, Brier score, log loss, consensus rate)
  - Live API operational metrics (query counts, latencies, error rates)

Includes disk persistence (monitoring/latest_metrics.json) so benchmarks persist
across FastAPI restarts and are always visible to Prometheus scraping.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_CACHE_PATH = BASE_DIR / "monitoring" / "latest_metrics.json"

# ── RAGAS Gauges ─────────────────────────────────────────────────────────────
ragas_faithfulness = Gauge(
    "ragas_faithfulness",
    "RAGAS Faithfulness score (0.0 - 1.0): factual consistency of generated answers",
)
ragas_answer_relevancy = Gauge(
    "ragas_answer_relevancy",
    "RAGAS Answer Relevancy score (0.0 - 1.0): pertinence to the user question",
)
ragas_context_precision = Gauge(
    "ragas_context_precision",
    "RAGAS Context Precision score (0.0 - 1.0): retriever ranking quality of relevant facts",
)
ragas_context_recall = Gauge(
    "ragas_context_recall",
    "RAGAS Context Recall score (0.0 - 1.0): coverage of ground-truth facts in retrieved context",
)
rag_numeric_grounding = Gauge(
    "rag_numeric_grounding",
    "Exact numeric fact grounding score (0.0 - 1.0)",
)
rag_eval_latency_seconds = Gauge(
    "rag_eval_latency_seconds",
    "Average RAG query latency in seconds during evaluation",
)

# ── Match Prediction Gauges ──────────────────────────────────────────────────
prediction_gnn_accuracy = Gauge(
    "prediction_gnn_accuracy",
    "Expert 1 (GNN) 3-way match outcome prediction accuracy (0.0 - 1.0)",
)
prediction_slm_accuracy = Gauge(
    "prediction_slm_accuracy",
    "Expert 2 (SLM) 3-way match outcome prediction accuracy (0.0 - 1.0)",
)
prediction_ensemble_accuracy = Gauge(
    "prediction_ensemble_accuracy",
    "Hybrid Ensemble 3-way match outcome prediction accuracy (0.0 - 1.0)",
)
prediction_brier_score = Gauge(
    "prediction_brier_score",
    "Multi-class Brier Score across [P(H), P(D), P(A)] vs actual result (lower is better, random=0.667)",
)
prediction_log_loss = Gauge(
    "prediction_log_loss",
    "Multi-class cross-entropy log loss vs actual outcome (lower is better, random=1.098)",
)
prediction_consensus_rate = Gauge(
    "prediction_consensus_rate",
    "Agreement rate between Expert 1 (GNN) and Expert 2 (SLM) (0.0 - 1.0)",
)
prediction_tactical_grounding = Gauge(
    "prediction_tactical_grounding",
    "Tactical stat grounding score in match prediction narratives (0.0 - 1.0)",
)
prediction_eval_latency_seconds = Gauge(
    "prediction_eval_latency_seconds",
    "Average latency per match prediction in seconds during evaluation",
)

# ── Metadata & Operational Metrics ───────────────────────────────────────────
eval_last_run_timestamp = Gauge(
    "football_eval_last_run_timestamp_seconds",
    "Epoch timestamp of the most recent evaluation run",
    ["suite"],
)
api_rag_queries_total = Counter(
    "football_rag_queries_total",
    "Total number of RAG queries executed by the API",
    ["status"],
)
api_rag_query_latency_seconds = Histogram(
    "football_rag_query_latency_seconds",
    "Histogram of RAG query latency in seconds",
    buckets=[0.5, 1.0, 2.0, 3.5, 5.0, 10.0, 20.0, 30.0],
)
api_predictions_total = Counter(
    "football_predictions_total",
    "Total number of match predictions served by the API",
    ["mode"],
)

# ── Online Live Stream Drift & Evidence Gauges / Counters ───────────────────
football_online_evidence_score = Gauge(
    "football_online_evidence_score",
    "Rolling average evidence relevance/similarity score from online RAG queries (0.0 - 1.0)",
)
football_online_faithfulness_score = Gauge(
    "football_online_faithfulness_score",
    "Rolling average factual and numeric grounding score for online assistant responses (0.0 - 1.0)",
)
football_online_query_drift_score = Gauge(
    "football_online_query_drift_score",
    "Statistical drift score for incoming query length/vocabulary shift (0.0 - 1.0)",
)
football_online_prediction_entropy = Gauge(
    "football_online_prediction_entropy",
    "Normalized outcome entropy of recent predictions (0.0=degenerate/skewed, 1.0=balanced)",
)
football_drift_detected_total = Counter(
    "football_drift_detected_total",
    "Total number of drift anomalies detected across the 5 dimensions",
    ["drift_type"],
)


def load_cached_metrics():
    """Load latest metrics from disk on application startup."""
    if not METRICS_CACHE_PATH.exists():
        return
    try:
        with open(METRICS_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # RAGAS
        if "ragas_faithfulness" in data:
            ragas_faithfulness.set(data["ragas_faithfulness"])
        if "ragas_answer_relevancy" in data:
            ragas_answer_relevancy.set(data["ragas_answer_relevancy"])
        if "ragas_context_precision" in data:
            ragas_context_precision.set(data["ragas_context_precision"])
        if "ragas_context_recall" in data:
            ragas_context_recall.set(data["ragas_context_recall"])
        if "numeric_grounding" in data:
            rag_numeric_grounding.set(data["numeric_grounding"])
        if "rag_latency_seconds" in data:
            rag_eval_latency_seconds.set(data["rag_latency_seconds"])

        # Predictions
        if "gnn_accuracy" in data:
            prediction_gnn_accuracy.set(data["gnn_accuracy"])
        if "slm_accuracy" in data:
            prediction_slm_accuracy.set(data["slm_accuracy"])
        if "ensemble_accuracy" in data:
            prediction_ensemble_accuracy.set(data["ensemble_accuracy"])
        if "brier_score" in data:
            prediction_brier_score.set(data["brier_score"])
        if "log_loss" in data:
            prediction_log_loss.set(data["log_loss"])
        if "consensus_rate" in data:
            prediction_consensus_rate.set(data["consensus_rate"])
        if "tactical_grounding" in data:
            prediction_tactical_grounding.set(data["tactical_grounding"])
        if "prediction_latency_seconds" in data:
            prediction_eval_latency_seconds.set(data["prediction_latency_seconds"])
        if "last_eval_timestamp" in data:
            eval_last_run_timestamp.labels(suite="all").set(data["last_eval_timestamp"])

        # Online Streaming Metrics & Drift
        if "online_evidence_score" in data:
            football_online_evidence_score.set(data["online_evidence_score"])
        if "online_faithfulness_score" in data:
            football_online_faithfulness_score.set(data["online_faithfulness_score"])
        if "online_query_drift_score" in data:
            football_online_query_drift_score.set(data["online_query_drift_score"])
        if "online_prediction_entropy" in data:
            football_online_prediction_entropy.set(data["online_prediction_entropy"])

        logger.info("Loaded cached evaluation metrics into Prometheus registry from %s", METRICS_CACHE_PATH)
    except Exception as e:
        logger.warning("Failed to load cached metrics: %s", e)


def record_online_rag_drift_metrics(
    evidence_score: float,
    faithfulness_score: float,
    query_drift_score: float,
    drift_type: Optional[str] = None,
):
    """Update Prometheus Gauges for live online RAG queries and drift detection."""
    football_online_evidence_score.set(evidence_score)
    football_online_faithfulness_score.set(faithfulness_score)
    football_online_query_drift_score.set(query_drift_score)
    if drift_type:
        football_drift_detected_total.labels(drift_type=drift_type).inc()

    save_cached_metrics({
        "online_evidence_score": evidence_score,
        "online_faithfulness_score": faithfulness_score,
        "online_query_drift_score": query_drift_score,
        "last_online_rag_timestamp": time.time(),
    })


def record_online_prediction_drift_metrics(
    entropy: float,
    drift_type: Optional[str] = None,
):
    """Update Prometheus Gauges for live match prediction distributions and drift."""
    football_online_prediction_entropy.set(entropy)
    if drift_type:
        football_drift_detected_total.labels(drift_type=drift_type).inc()

    save_cached_metrics({
        "online_prediction_entropy": entropy,
        "last_online_prediction_timestamp": time.time(),
    })


def save_cached_metrics(updates: dict):
    """Save updated metrics to disk for persistence across restarts."""
    METRICS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if METRICS_CACHE_PATH.exists():
        try:
            with open(METRICS_CACHE_PATH, "r", encoding="utf-8") as f:
                current = json.load(f)
        except Exception:
            current = {}
    current.update(updates)
    current["last_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(METRICS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)


def record_rag_evaluation_metrics(
    avg_latency: float,
    avg_numeric: float,
    avg_grounding: float,
    ragas_scores: Optional[dict] = None,
):
    """Update Prometheus Gauges for RAG & RAGAS evaluation."""
    now = time.time()
    rag_eval_latency_seconds.set(avg_latency)
    rag_numeric_grounding.set(avg_numeric)
    eval_last_run_timestamp.labels(suite="rag").set(now)

    updates = {
        "rag_latency_seconds": avg_latency,
        "numeric_grounding": avg_numeric,
        "grounding_relevancy": avg_grounding,
        "last_rag_eval_timestamp": now,
        "last_eval_timestamp": now,
    }

    if ragas_scores:
        if "faithfulness" in ragas_scores:
            val = float(ragas_scores["faithfulness"])
            ragas_faithfulness.set(val)
            updates["ragas_faithfulness"] = val
        if "answer_relevancy" in ragas_scores:
            val = float(ragas_scores["answer_relevancy"])
            ragas_answer_relevancy.set(val)
            updates["ragas_answer_relevancy"] = val
        if "context_precision" in ragas_scores:
            val = float(ragas_scores["context_precision"])
            ragas_context_precision.set(val)
            updates["ragas_context_precision"] = val
        if "context_recall" in ragas_scores:
            val = float(ragas_scores["context_recall"])
            ragas_context_recall.set(val)
            updates["ragas_context_recall"] = val

    save_cached_metrics(updates)


def record_prediction_evaluation_metrics(
    gnn_acc: float,
    slm_acc: float,
    ens_acc: float,
    brier: float,
    log_loss: float,
    consensus: float,
    tactical: float,
    latency: float,
):
    """Update Prometheus Gauges for match prediction evaluation."""
    now = time.time()
    prediction_gnn_accuracy.set(gnn_acc)
    prediction_slm_accuracy.set(slm_acc)
    prediction_ensemble_accuracy.set(ens_acc)
    prediction_brier_score.set(brier)
    prediction_log_loss.set(log_loss)
    prediction_consensus_rate.set(consensus)
    prediction_tactical_grounding.set(tactical)
    prediction_eval_latency_seconds.set(latency)
    eval_last_run_timestamp.labels(suite="prediction").set(now)

    save_cached_metrics({
        "gnn_accuracy": gnn_acc,
        "slm_accuracy": slm_acc,
        "ensemble_accuracy": ens_acc,
        "brier_score": brier,
        "log_loss": log_loss,
        "consensus_rate": consensus,
        "tactical_grounding": tactical,
        "prediction_latency_seconds": latency,
        "last_prediction_eval_timestamp": now,
        "last_eval_timestamp": now,
    })


def get_prometheus_metrics_bytes() -> bytes:
    """Generate Prometheus scrape payload."""
    return generate_latest(REGISTRY)


def send_live_metric(metric_type: str, data: dict, api_url: Optional[str] = None) -> bool:
    """
    Stream live metrics over HTTP directly to the running FastAPI server.
    Enables real-time gauge and card updates in Grafana while test suites are running.
    """
    import urllib.request
    target = api_url or os.getenv("API_URL", "http://localhost:8001")
    url = f"{target.rstrip('/')}/api/v1/monitoring/live-metrics"
    payload = json.dumps({"type": metric_type, **data}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            return resp.status == 200
    except Exception:
        return False


# Initialize from cache on module import
load_cached_metrics()
