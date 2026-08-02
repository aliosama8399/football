"""
Prediction Model Provider
==========================
Pluggable slot for the "Expert 1" prediction model, mirroring the KG/vector
provider pattern. `BasePredictionProvider` is the ABC; `GNNPredictionProvider`
is the first concrete implementation (TEA-GNN, the #1 tuned model).

Usage:
    provider = GNNPredictionProvider()
    provider.load()
    result = provider.predict("Arsenal", "Chelsea")
    # -> {"predicted_result": "H"|"D"|"A", "probabilities": {"H":..,"D":..,"A":..}}
    #    or None if the model is not loaded / teams unknown / prediction fails.

None-safe: if the checkpoint file is missing, `load()` logs a warning and
leaves the provider unloaded; `predict()` returns None (never raises).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from models.gnn_models import get_model
from data.graph_builder import FootballGraphBuilder

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CFG_PATH = BASE_DIR / "models" / "llm_config.yaml"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_PATH = "models/saved/gnn_tea-gnn_tuned.pt"


def _load_gnn_model_path() -> str:
    """Read rag.gnn_model_path from llm_config.yaml (with default)."""
    try:
        if CFG_PATH.exists():
            with open(CFG_PATH) as f:
                return yaml.safe_load(f).get("rag", {}).get("gnn_model_path", DEFAULT_MODEL_PATH)
    except Exception as e:
        logger.warning("Could not read rag.gnn_model_path from config: %s", e)
    return DEFAULT_MODEL_PATH


class BasePredictionProvider(ABC):
    """Strategy for the Expert 1 match-prediction model."""

    @abstractmethod
    def load(self) -> bool:
        """Load the model + graph. Returns True on success, False otherwise."""

    @abstractmethod
    def predict(self, home_team: str, away_team: str) -> Optional[dict]:
        """Return structured prediction or None if unavailable."""


class GNNPredictionProvider(BasePredictionProvider):
    """
    TEA-GNN expert. Loads the tuned graph neural network checkpoint and the
    match graph once, then answers (home, away) matchups.

    This consolidates the two divergent code paths that previously lived in
    rag_orchestrator.py (`_get_gnn_prediction` -> EdgeConv checkpoint, and
    `get_gnn_prediction_structured` -> TEA-GNN checkpoint) into a single
    provider reading ONE canonical path (`rag.gnn_model_path`).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else Path(_load_gnn_model_path())
        if not self.model_path.is_absolute():
            self.model_path = BASE_DIR / self.model_path
        self.model = None
        self.graph = None
        self.builder = None
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return True

        if not self.model_path.exists():
            logger.warning(
                "GNN model file not found at %s. Train the model or set "
                "rag.gnn_model_path in llm_config.yaml.", self.model_path
            )
            return False

        try:
            builder = FootballGraphBuilder(
                data_path=str(BASE_DIR / "data" / "processed" / "processed_matches.csv"))
            graph = builder.build_train_test_graphs()

            checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)
            bp = checkpoint.get("best_params", {})
            model_name = checkpoint.get("model_name", "TEA-GNN")

            nf = graph["num_node_features"]
            ef = graph["num_edge_features"]
            kwargs = dict(hidden_dim=bp.get("hidden_dim", 64),
                          dropout=bp.get("dropout", 0.3))
            is_tea_gnn = "TEA-GNN" in model_name
            if is_tea_gnn:
                kwargs["heads"] = bp.get("heads", 4)
                kwargs["num_leagues"] = graph.get("num_leagues", 5)
                kwargs["use_cross_league"] = bp.get("use_cross_league", True)

            model = get_model(model_name, nf, ef, **kwargs)
            model.load_state_dict(checkpoint["model_state"])
            model = model.to(DEVICE).eval()

            graph["is_tea_gnn"] = is_tea_gnn

            self.model = model
            self.graph = graph
            self.builder = builder
            self._loaded = True
            logger.info(
                "Loaded GNN (Expert 1): %s (nf=%d, ef=%d) from %s",
                model_name, nf, ef, self.model_path
            )
            return True
        except Exception as e:
            logger.error("Failed to load GNN (Expert 1): %s", e)
            return False

    def predict(self, home_team: str, away_team: str) -> Optional[dict]:
        if not self._loaded and not self.load():
            return None

        try:
            home_idx = self.builder.team_to_idx.get(home_team)
            away_idx = self.builder.team_to_idx.get(away_team)
            if home_idx is None or away_idx is None:
                logger.warning("Teams not found in GNN graph for %s vs %s",
                               home_team, away_team)
                return None

            result = self._predict_indices(home_idx, away_idx)
            return result
        except Exception as e:
            logger.error("GNN Prediction failed: %s", e)
            return None

    def predict_indices(self, home_idx: int, away_idx: int) -> Optional[dict]:
        """Predict for already-resolved node indices (used by callers that have the graph)."""
        if not self._loaded and not self.load():
            return None
        return self._predict_indices(home_idx, away_idx)

    def _predict_indices(self, home_idx: int, away_idx: int) -> Optional[dict]:
        x  = self.graph.get("x_test", self.graph["x"]).to(DEVICE)
        ei = self.graph["edge_index"].to(DEVICE)
        ea = self.graph["edge_attr"].to(DEVICE)
        is_tea_gnn = self.graph.get("is_tea_gnn", False)

        with torch.no_grad():
            if is_tea_gnn:
                edge_time = self.graph.get("edge_time")
                league_id = self.graph.get("league_id")
                if edge_time is not None:
                    edge_time = edge_time.to(DEVICE)
                if league_id is not None:
                    league_id = league_id.to(DEVICE)
                out = self.model(x, ei, ea, edge_time=edge_time, league_id=league_id)
            else:
                out = self.model(x, ei, ea)

        src_mask = ei[0] == home_idx
        dst_mask = ei[1] == away_idx
        match_edges = (src_mask & dst_mask).nonzero(as_tuple=True)[0]
        if len(match_edges) == 0:
            logger.warning("Match edge not found for idx %d vs %d", home_idx, away_idx)
            return None

        edge_idx = match_edges[-1].item()
        logits   = out[edge_idx]
        probs    = F.softmax(logits, dim=0).cpu().numpy()
        pred_class = logits.argmax().item()

        class_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        return {
            "predicted_result": class_map[pred_class],
            "probabilities": {
                "A": float(probs[0]),
                "D": float(probs[1]),
                "H": float(probs[2]),
            },
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded
