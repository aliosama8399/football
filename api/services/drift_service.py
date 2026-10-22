"""
Online 5-Dimensional Drift Detection & Real-time Evidence Evaluator
===================================================================
Runs alongside live user requests (POST /api/v1/chat, POST /api/v1/predictions)
without adding user-facing latency.

Monitors 5 dimensions:
  1. Covariate / Input Query Drift (Query length, token complexity, out-of-domain)
  2. Prediction / Output Prior Drift (H/D/A distribution shift & normalized entropy)
  3. Retrieval / Evidence Drift (Average similarity / relevance of retrieved chunks)
  4. Faithfulness / Hallucination Drift (Numeric fact grounding against retrieved sources)
  5. Concept / Supervisor Override Drift (Human expert overrides & outcome disagreement)

Updates Prometheus metrics and cached disk state automatically.
"""

import re
import math
import time
import logging
from collections import deque
from typing import Dict, List, Optional, Any

from monitoring.metrics_registry import (
    record_online_rag_drift_metrics,
    record_online_prediction_drift_metrics,
    football_drift_detected_total,
)

logger = logging.getLogger("drift_service")

# Common football terminology tokens for out-of-domain detection
FOOTBALL_LEXICON = {
    "goal", "goals", "assist", "score", "match", "vs", "against", "win", "draw",
    "loss", "defeat", "table", "league", "points", "premier", "laliga", "serie",
    "bundesliga", "ligue", "champions", "arsenal", "chelsea", "liverpool", "manchester",
    "city", "united", "tottenham", "real", "barcelona", "madrid", "bayern", "juventus",
    "inter", "milan", "psg", "tactics", "xg", "xg_conceded", "clean", "sheet",
    "defense", "attack", "midfield", "striker", "coach", "manager", "form",
    "fixture", "injury", "suspension", "referee", "penalty", "card", "derby"
}


class OnlineDriftEvaluator:
    """
    Stateful sliding-window drift evaluator for streaming RAG and Match Predictions.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size

        # Sliding windows for online statistics
        self.query_lengths = deque(maxlen=window_size)
        self.evidence_scores = deque(maxlen=window_size)
        self.faithfulness_scores = deque(maxlen=window_size)
        self.prediction_outcomes = deque(maxlen=window_size)

        # Baseline expectations
        self.baseline_query_len_mean = 45.0
        self.baseline_query_len_std = 25.0
        self.evidence_threshold = 0.65
        self.faithfulness_threshold = 0.70

    # ─────────────────────────────────────────────────────────────────────────
    # 1. & 3. & 4. RAG Live Stream Evaluator (Input, Evidence, Faithfulness)
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_rag_turn(
        self,
        query: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single live RAG turn:
          - Covariate drift (query length / out-of-domain terms)
          - Retrieval / Evidence score (average source similarity)
          - Faithfulness score (numeric & fact grounding against retrieved text)
        """
        detected_drifts = []

        # ── 1. Covariate / Input Query Drift ──
        q_clean = query.strip()
        q_len = len(q_clean)
        self.query_lengths.append(q_len)

        # Tokenize query to check domain relevance
        tokens = set(re.findall(r"[a-zA-Z]{3,}", q_clean.lower()))
        domain_matches = tokens.intersection(FOOTBALL_LEXICON)

        # Compute query drift score based on length Z-score and domain match
        rolling_mean_len = sum(self.query_lengths) / max(1, len(self.query_lengths))
        len_shift = abs(rolling_mean_len - self.baseline_query_len_mean) / max(10.0, self.baseline_query_len_std)
        query_drift_score = min(1.0, len_shift * 0.3)

        if len(tokens) >= 3 and len(domain_matches) == 0:
            # Query is completely out of domain
            query_drift_score = max(query_drift_score, 0.85)
            detected_drifts.append("covariate_out_of_domain")
            logger.warning("[Drift] Out-of-domain query detected: '%s'", q_clean[:60])
        elif len_shift > 2.0:
            detected_drifts.append("covariate_query_length_shift")
            logger.info("[Drift] Covariate query length shift detected (mean=%.1f)", rolling_mean_len)

        # ── 3. Retrieval / Evidence Drift ──
        evidence_score = 0.0
        combined_source_text = ""
        if sources:
            scores = []
            for s in sources:
                score_val = s.get("score") or s.get("similarity") or s.get("relevance")
                if score_val is not None:
                    try:
                        scores.append(float(score_val))
                    except (ValueError, TypeError):
                        pass
                text_val = s.get("text") or s.get("content") or s.get("snippet") or ""
                combined_source_text += f" {text_val}"

            if scores:
                evidence_score = sum(scores) / len(scores)
            else:
                # If no explicit score field, normalize hit count
                evidence_score = min(1.0, 0.5 + 0.1 * len(sources))
        else:
            evidence_score = 0.2  # Unretrieved / zero context

        self.evidence_scores.append(evidence_score)
        rolling_evidence = sum(self.evidence_scores) / max(1, len(self.evidence_scores))

        if rolling_evidence < self.evidence_threshold:
            detected_drifts.append("retrieval_evidence_decay")
            logger.warning("[Drift] Retrieval evidence score decay: rolling=%.3f < threshold=%.2f",
                           rolling_evidence, self.evidence_threshold)

        # ── 4. Faithfulness / Hallucination Drift (Numeric Grounding) ──
        # Extract numeric facts from the generated answer (e.g., scores "2-1", "3.4", "85%")
        answer_numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", answer)
        if answer_numbers:
            grounded = sum(1 for num in answer_numbers if num in combined_source_text)
            faithfulness_score = grounded / len(answer_numbers)
        else:
            # If no numbers, compute lexical overlap with sources
            ans_tokens = set(re.findall(r"[a-zA-Z]{4,}", answer.lower()))
            src_tokens = set(re.findall(r"[a-zA-Z]{4,}", combined_source_text.lower()))
            overlap = len(ans_tokens.intersection(src_tokens))
            faithfulness_score = min(1.0, overlap / max(1, len(ans_tokens) * 0.4))

        self.faithfulness_scores.append(faithfulness_score)
        rolling_faithfulness = sum(self.faithfulness_scores) / max(1, len(self.faithfulness_scores))

        if rolling_faithfulness < self.faithfulness_threshold:
            detected_drifts.append("faithfulness_hallucination")
            logger.warning("[Drift] Faithfulness/hallucination drift: rolling=%.3f < threshold=%.2f",
                           rolling_faithfulness, self.faithfulness_threshold)

        # Record to Prometheus Gauges & Counters
        primary_drift = detected_drifts[0] if detected_drifts else None
        record_online_rag_drift_metrics(
            evidence_score=round(rolling_evidence, 3),
            faithfulness_score=round(rolling_faithfulness, 3),
            query_drift_score=round(query_drift_score, 3),
            drift_type=primary_drift,
        )

        return {
            "query_drift_score": query_drift_score,
            "evidence_score": rolling_evidence,
            "faithfulness_score": rolling_faithfulness,
            "detected_drifts": detected_drifts,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Match Prediction Prior & Output Drift
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_prediction_turn(
        self,
        home_team: str,
        away_team: str,
        predicted_result: str,
        probabilities: Optional[Dict[str, float]] = None,
        source: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        Evaluate a single live match prediction:
          - Output distribution shift (e.g. extreme home win bias)
          - Normalized Shannon entropy across [P(H), P(D), P(A)]
        """
        detected_drifts = []

        # Standardize predicted result
        pred_label = predicted_result.upper().strip()
        if "HOME" in pred_label or pred_label == "H":
            outcome = "H"
        elif "AWAY" in pred_label or pred_label == "A":
            outcome = "A"
        else:
            outcome = "D"

        self.prediction_outcomes.append(outcome)

        # Compute outcome entropy over rolling window
        counts = {"H": 0, "D": 0, "A": 0}
        for o in self.prediction_outcomes:
            counts[o] = counts.get(o, 0) + 1

        total = len(self.prediction_outcomes)
        entropy = 0.0
        if total >= 5:
            for outcome_key, count in counts.items():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            # Normalized entropy: max entropy for 3 classes is log2(3) ~ 1.585
            normalized_entropy = entropy / math.log2(3)
        else:
            normalized_entropy = 1.0

        # Check for extreme bias (e.g., Home wins > 75% or Draws < 5% in rolling window)
        h_ratio = counts["H"] / max(1, total)
        if total >= 15 and (h_ratio > 0.75 or normalized_entropy < 0.60):
            detected_drifts.append("prediction_output_distribution_shift")
            logger.warning("[Drift] Prediction output distribution skew: H_ratio=%.2f, entropy=%.2f",
                           h_ratio, normalized_entropy)

        # Record to Prometheus
        primary_drift = detected_drifts[0] if detected_drifts else None
        record_online_prediction_drift_metrics(
            entropy=round(normalized_entropy, 3),
            drift_type=primary_drift,
        )

        return {
            "normalized_entropy": round(normalized_entropy, 3),
            "outcome_counts": counts,
            "detected_drifts": detected_drifts,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Concept Drift / Supervisor Override
    # ─────────────────────────────────────────────────────────────────────────
    def record_supervisor_override(
        self,
        match_id: str,
        original_prediction: str,
        override_prediction: str,
        reason: Optional[str] = None,
    ):
        """
        Record a human supervisor override indicating concept drift or model failure.
        """
        logger.warning(
            "[Drift] Supervisor override recorded on match '%s': '%s' -> '%s' (Reason: %s)",
            match_id, original_prediction, override_prediction, reason or "None"
        )
        football_drift_detected_total.labels(drift_type="concept_supervisor_override").inc()


# Global singleton instance
drift_evaluator = OnlineDriftEvaluator(window_size=30)
