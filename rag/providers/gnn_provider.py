"""
Prediction Model Provider
==========================
Pluggable slot for the "Expert 1" prediction model, mirroring the KG/vector
provider pattern. `BasePredictionProvider` is the ABC; `GNNPredictionProvider`
is the PyTorch implementation (TEA-GNN, the #1 tuned model).

Supports:
  - Single matchup prediction (`predict(home_team, away_team)`)
  - Cross-league & unplayed matchup prediction via synthetic pre-match query edges
  - Vectorized batch prediction (`predict_batch(matchups)`)
  - Hot-reloading (`reload()`)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
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
    """Read rag.gnn_model_path from llm_config.yaml (with fallback to newest .pt checkpoint)."""
    try:
        if CFG_PATH.exists():
            with open(CFG_PATH) as f:
                p = yaml.safe_load(f).get("rag", {}).get("gnn_model_path", DEFAULT_MODEL_PATH)
                if p and str(p).endswith(".pt"):
                    return p
    except Exception as e:
        logger.warning("Could not read rag.gnn_model_path from config: %s", e)

    tuned = BASE_DIR / "models" / "saved" / "gnn_tea-gnn_tuned.pt"
    base = BASE_DIR / "models" / "saved" / "gnn_tea-gnn.pt"
    if tuned.exists() and base.exists():
        return str(base if base.stat().st_mtime > tuned.stat().st_mtime else tuned)
    if base.exists():
        return str(base)
    return DEFAULT_MODEL_PATH


class BasePredictionProvider(ABC):
    """Strategy for the Expert 1 match-prediction model."""

    @abstractmethod
    def load(self) -> bool:
        """Load the model + graph. Returns True on success, False otherwise."""

    @abstractmethod
    def predict(self, home_team: str, away_team: str) -> Optional[dict]:
        """Return structured prediction or None if unavailable."""

    def predict_batch(self, matchups: list[dict]) -> list[dict]:
        """Predict a batch of matchups [{'home_team': ..., 'away_team': ...}]."""
        results = []
        for m in matchups:
            ht = m.get("home_team", "")
            at = m.get("away_team", "")
            pred = self.predict(ht, at)
            if pred is None:
                results.append({
                    "home_team": ht,
                    "away_team": at,
                    "predicted_result": "Draw",
                    "probabilities": {"H": 0.33, "D": 0.34, "A": 0.33},
                    "source": "fallback",
                })
            else:
                results.append({
                    "home_team": ht,
                    "away_team": at,
                    **pred,
                    "source": pred.get("source", "live_model"),
                })
        return results

    def reload(self) -> bool:
        """Reload model weights and graph snapshot."""
        return self.load()


class GNNPredictionProvider(BasePredictionProvider):
    """
    TEA-GNN expert (PyTorch). Loads the tuned graph neural network checkpoint and the
    match graph once, then answers single or batched (home, away) matchups, including
    unplayed and cross-league fixtures via synthetic pre-match query edges.
    """

    def __init__(self, model_path: Optional[str] = None):
        resolved = model_path if (model_path and str(model_path).endswith(".pt")) else _load_gnn_model_path()
        self.model_path = Path(resolved)
        if not self.model_path.is_absolute():
            self.model_path = BASE_DIR / self.model_path
        self.model = None
        self.graph = None
        self.builder = None
        self._loaded = False

    def reload(self) -> bool:
        self._loaded = False
        self.model = None
        self.graph = None
        self.builder = None
        return self.load()

    def load(self) -> bool:
        if self._loaded:
            return True

        if not self.model_path.exists():
            alt = BASE_DIR / "models" / "saved" / "gnn_tea-gnn.pt"
            if alt.exists():
                self.model_path = alt
            else:
                logger.warning(
                    "GNN model file not found at %s. Train the model or set "
                    "rag.gnn_model_path in llm_config.yaml.", self.model_path
                )
                return False

        try:
            builder = FootballGraphBuilder(
                data_path=str(BASE_DIR / "data" / "processed" / "processed_matches.csv")
            )
            graph = builder.build_train_test_graphs()

            checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)
            bp = checkpoint.get("best_params", {})
            model_name = checkpoint.get("model_name", "TEA-GNN")

            nf = graph["num_node_features"]
            ef = graph["num_edge_features"]
            hidden_dim = bp.get("hidden_dim", checkpoint.get("hidden_dim", 64))
            dropout = bp.get("dropout", checkpoint.get("dropout", 0.3))
            kwargs = dict(hidden_dim=hidden_dim, dropout=dropout)

            is_tea_gnn = "TEA-GNN" in model_name
            if is_tea_gnn:
                state_keys = checkpoint.get("model_state", {}).keys()
                has_cross = any("cross_league." in k for k in state_keys)
                kwargs["heads"] = bp.get("heads", checkpoint.get("heads", 4))
                kwargs["num_leagues"] = graph.get("num_leagues", 5)
                kwargs["use_cross_league"] = bp.get("use_cross_league", has_cross)

            model = get_model(model_name, nf, ef, **kwargs)
            model.load_state_dict(checkpoint["model_state"], strict=False)
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
                logger.warning("Teams not found in GNN graph for %s vs %s", home_team, away_team)
                return None

            return self._predict_batch_indices([home_idx], [away_idx])[0]
        except Exception as e:
            logger.error("GNN Prediction failed: %s", e)
            return None

    def predict_indices(self, home_idx: int, away_idx: int) -> Optional[dict]:
        """Predict for already-resolved node indices."""
        if not self._loaded and not self.load():
            return None
        return self._predict_batch_indices([home_idx], [away_idx])[0]

    def predict_batch(self, matchups: list[dict]) -> list[dict]:
        """Vectorized batch prediction in a single GNN forward pass."""
        if not matchups:
            return []
        if not self._loaded and not self.load():
            return super().predict_batch(matchups)

        valid_positions = []
        home_indices = []
        away_indices = []
        results: list[Optional[dict]] = [None] * len(matchups)

        for idx, m in enumerate(matchups):
            ht = m.get("home_team", "")
            at = m.get("away_team", "")
            hi = self.builder.team_to_idx.get(ht)
            ai = self.builder.team_to_idx.get(at)
            if hi is not None and ai is not None:
                valid_positions.append(idx)
                home_indices.append(hi)
                away_indices.append(ai)
            else:
                results[idx] = {
                    "home_team": ht,
                    "away_team": at,
                    "predicted_result": "Draw",
                    "probabilities": {"H": 0.33, "D": 0.34, "A": 0.33},
                    "source": "fallback",
                }

        if valid_positions:
            try:
                batch_preds = self._predict_batch_indices(home_indices, away_indices)
                for pos, pred in zip(valid_positions, batch_preds):
                    m = matchups[pos]
                    results[pos] = {
                        "home_team": m.get("home_team", ""),
                        "away_team": m.get("away_team", ""),
                        **pred,
                        "source": "live_model",
                    }
            except Exception as e:
                logger.error("GNN batch prediction failed: %s", e)
                return super().predict_batch(matchups)

        return [r for r in results if r is not None]

    def _predict_batch_indices(self, home_indices: list[int], away_indices: list[int]) -> list[dict]:
        """
        Score one or more (home_idx, away_idx) pairs by appending synthetic pre-match
        query edges at edge_time=0.0, supporting unplayed and cross-league matchups.
        """
        x = self.graph.get("x_test", self.graph["x"]).to(DEVICE)
        base_ei = self.graph["edge_index"].to(DEVICE)
        base_ea = self.graph["edge_attr"].to(DEVICE)
        is_tea_gnn = self.graph.get("is_tea_gnn", False)

        batch_size = len(home_indices)
        query_ei = torch.tensor([home_indices, away_indices], dtype=torch.long, device=DEVICE)
        query_ea_np = self.builder.build_query_edge_attr(home_indices, away_indices, x)
        query_ea = torch.tensor(query_ea_np, dtype=torch.float32, device=DEVICE)

        ei = torch.cat([base_ei, query_ei], dim=1)
        ea = torch.cat([base_ea, query_ea], dim=0)

        with torch.no_grad():
            if is_tea_gnn:
                base_et = self.graph.get("edge_time")
                league_id = self.graph.get("league_id")
                if base_et is not None:
                    query_et = torch.zeros(batch_size, dtype=torch.float32, device=DEVICE)
                    edge_time = torch.cat([base_et.to(DEVICE), query_et], dim=0)
                else:
                    edge_time = None
                if league_id is not None:
                    league_id = league_id.to(DEVICE)
                out = self.model(x, ei, ea, edge_time=edge_time, league_id=league_id)
            else:
                out = self.model(x, ei, ea)

        query_logits = out[-batch_size:]
        probs_batch = F.softmax(query_logits, dim=-1).cpu().numpy()
        preds_batch = query_logits.argmax(dim=-1).cpu().numpy()

        class_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        outputs = []
        for i in range(batch_size):
            probs = probs_batch[i]
            pred_class = int(preds_batch[i])
            outputs.append({
                "predicted_result": class_map[pred_class],
                "probabilities": {
                    "A": float(probs[0]),
                    "D": float(probs[1]),
                    "H": float(probs[2]),
                },
                "source": "live_model",
            })
        return outputs

    @property
    def is_loaded(self) -> bool:
        return self._loaded
