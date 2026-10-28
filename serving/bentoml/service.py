"""
Unified BentoML Service for Football RAG & Match Prediction SaaS
================================================================
Wraps the full FastAPI application with BentoML serving infrastructure:
  1. Mounts FastAPI via @bentoml.asgi_app(fastapi_app, path="/")
     Preserves all REST routes, GraphQL (/graphql), UI (/ui), and Prometheus (/metrics).
  2. Exposes adaptive batching endpoint predict_match_batch for GNN high-throughput predictions
     using ONNXPredictionProvider (with GNNPredictionProvider fallback) and vectorized
     predict_batch(matchups) execution.
  3. Provides declarative packaging for containerization (bentofile.yaml).
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import bentoml
from api.main import app as fastapi_app

logger = logging.getLogger("bentoml.service")


@bentoml.asgi_app(fastapi_app, path="/")
@bentoml.service(
    name="football_rag_service",
    traffic={
        "timeout": 300,
        "max_concurrency": 64,
    },
    resources={
        "cpu": "2",
    },
)
class FootballBentoService:
    """
    Unified BentoML Service wrapping the entire Football RAG & Prediction backend.
    """

    def __init__(self):
        logger.info("Initializing FootballBentoService (BentoML %s)...", bentoml.__version__)
        self._gnn_provider = None

    @property
    def gnn_provider(self):
        if self._gnn_provider is None:
            # Prefer ONNX Runtime provider for low-latency adaptive batching, fallback to PyG GNN
            try:
                from rag.providers.gnn_provider_onnx import ONNXPredictionProvider
                provider = ONNXPredictionProvider()
                provider.load()
                if provider._loaded:
                    self._gnn_provider = provider
                    logger.info("ONNX GNN prediction provider loaded successfully in BentoML.")
                    return self._gnn_provider
            except Exception as onnx_err:
                logger.info("ONNX GNN provider unavailable in BentoML (%s); falling back to PyG GNN.", onnx_err)

            try:
                from rag.providers.gnn_provider import GNNPredictionProvider
                provider = GNNPredictionProvider()
                provider.load()
                self._gnn_provider = provider
                logger.info("PyG GNN prediction provider loaded successfully in BentoML.")
            except Exception as e:
                logger.warning("GNN Provider lazy load warning in BentoML service: %s", e)
        return self._gnn_provider

    @bentoml.api(batchable=True, batch_dim=0)
    def predict_match_batch(self, matchups: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Adaptive Batching Prediction API.
        BentoML automatically collects individual match prediction calls arriving concurrently
        into a single batch and executes a vectorized batch forward pass via predict_batch().

        Input: list of {"home_team": str, "away_team": str}
        """
        provider = self.gnn_provider
        if provider and provider._loaded:
            if hasattr(provider, "predict_batch"):
                batch_preds = provider.predict_batch(matchups)
            else:
                batch_preds = [
                    provider.predict(m.get("home_team", ""), m.get("away_team", ""))
                    for m in matchups
                ]
            results = []
            for item, pred in zip(matchups, batch_preds):
                results.append(pred or {
                    "home_team": item.get("home_team", ""),
                    "away_team": item.get("away_team", ""),
                    "predicted_result": "D",
                    "probabilities": {"H": 0.33, "D": 0.34, "A": 0.33},
                    "source": "fallback",
                })
            return results

        return [
            {
                "home_team": item.get("home_team", ""),
                "away_team": item.get("away_team", ""),
                "predicted_result": "D",
                "probabilities": {"H": 0.33, "D": 0.34, "A": 0.33},
                "source": "fallback",
            }
            for item in matchups
        ]

    @bentoml.api
    def service_status(self) -> Dict[str, Any]:
        """BentoML health and metadata inspection probe."""
        provider = self.gnn_provider
        return {
            "service": "football_rag_service",
            "bentoml_version": bentoml.__version__,
            "gnn_loaded": bool(provider and provider._loaded),
            "gnn_backend": type(provider).__name__ if provider else None,
            "fastapi_mounted": True,
            "status": "healthy"
        }
