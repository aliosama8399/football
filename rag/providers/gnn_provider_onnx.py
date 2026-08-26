"""
ONNX Prediction Provider (Expert 1)
===================================
Replaces GNNPredictionProvider (torch) with onnxruntime inference.
Artifact:  models/export/gnn/tea_gnn.onnx  (+ tea_gnn_io.json)
Export:    conda run -n football python models/export_gnn.py

None-safe: if artifact missing, load() returns False and predict() returns None
(never raises), matching GNNPredictionProvider behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CFG_PATH = BASE_DIR / "models" / "llm_config.yaml"
DEFAULT_ONNX_PATH = BASE_DIR / "models" / "export" / "gnn" / "tea_gnn.onnx"

# Reuse the same ABC as torch provider for drop-in replacement
from rag.providers.gnn_provider import BasePredictionProvider
from data.graph_builder import FootballGraphBuilder


def _resolve_onnx_path() -> Path:
    env_path = os.getenv("FOOTBALL_GNN_MODEL_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else BASE_DIR / p
    try:
        if CFG_PATH.exists():
            with open(CFG_PATH) as f:
                p = yaml.safe_load(f).get("rag", {}).get("gnn_model_path", "")
                if p:
                    pp = Path(p)
                    if not pp.is_absolute():
                        pp = BASE_DIR / pp
                    return pp
    except Exception as e:
        logger.warning("Could not read rag.gnn_model_path: %s", e)
    return DEFAULT_ONNX_PATH


class ONNXPredictionProvider(BasePredictionProvider):
    """TEA-GNN expert via ONNX Runtime (replaces torch)."""

    def __init__(self, model_path: Optional[str | Path] = None):
        p = Path(model_path) if model_path else _resolve_onnx_path()
        if not p.is_absolute():
            p = BASE_DIR / p
        self.model_path = p
        self.session = None
        self.graph = None
        self.builder = None
        self._loaded = False
        self._io_spec = None

    def load(self) -> bool:
        if self._loaded:
            return True

        if not self.model_path.exists():
            logger.warning(
                "ONNX GNN not found at %s. Run: conda run -n football python models/export_gnn.py",
                self.model_path,
            )
            return False

        try:
            import onnxruntime as ort

            # Load IO spec alongside ONNX (for input names/dtypes)
            io_path = self.model_path.parent / "tea_gnn_io.json"
            if io_path.exists():
                self._io_spec = json.loads(io_path.read_text())

            # Build graph (same builder as torch provider)
            builder = FootballGraphBuilder(
                data_path=str(BASE_DIR / "data" / "processed" / "processed_matches.csv")
            )
            graph = builder.build_train_test_graphs()

            sess = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

            self.builder = builder
            self.graph = graph
            self.session = sess
            # need is_tea_gnn flag for edge_time/league_id handling parity with torch provider
            self.graph["is_tea_gnn"] = True
            self._loaded = True
            logger.info("Loaded ONNX GNN (Expert 1) from %s", self.model_path)
            return True
        except Exception as e:
            logger.error("Failed to load ONNX GNN: %s", e)
            return False

    def predict(self, home_team: str, away_team: str) -> Optional[dict]:
        if not self._loaded and not self.load():
            return None
        try:
            hi = self.builder.team_to_idx.get(home_team)
            ai = self.builder.team_to_idx.get(away_team)
            if hi is None or ai is None:
                logger.warning("Teams not found in ONNX GNN graph: %s vs %s", home_team, away_team)
                return None
            return self._predict_indices(hi, ai)
        except Exception as e:
            logger.error("ONNX GNN predict failed: %s", e)
            return None

    def predict_indices(self, home_idx: int, away_idx: int) -> Optional[dict]:
        if not self._loaded and not self.load():
            return None
        return self._predict_indices(home_idx, away_idx)

    def _predict_indices(self, home_idx: int, away_idx: int) -> Optional[dict]:
        import torch.nn.functional as F
        import torch

        # Prepare inputs (same as torch provider, but as numpy for ORT)
        x = self.graph.get("x_test", self.graph["x"]).numpy().astype(np.float32)
        edge_index = self.graph["edge_index"].numpy().astype(np.int64)
        edge_attr = self.graph["edge_attr"].numpy().astype(np.float32)
        edge_time = self.graph.get("edge_time")
        league_id = self.graph.get("league_id")
        edge_time_np = edge_time.numpy().astype(np.float32) if edge_time is not None else np.zeros(edge_index.shape[1], dtype=np.float32)
        league_id_np = league_id.numpy().astype(np.int64) if league_id is not None else np.zeros(x.shape[0], dtype=np.int64)

        ort_inputs = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_time": edge_time_np,
            "league_id": league_id_np,
        }

        # Only feed inputs that the ONNX graph expects (handles dynamic vs static export)
        sess_inputs = {k: v for k, v in ort_inputs.items() if k in [i.name for i in self.session.get_inputs()]}

        logits_all = self.session.run(["logits"], sess_inputs)[0]  # [E,3]

        # Find match edge (last occurrence of home->away)
        ei = self.graph["edge_index"].numpy()
        mask = (ei[0] == home_idx) & (ei[1] == away_idx)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            logger.warning("Match edge not found for idx %d vs %d", home_idx, away_idx)
            return None
        eidx = int(idxs[-1])
        logits = logits_all[eidx]
        # softmax
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        pred = int(np.argmax(logits))
        cmap = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        return {
            "predicted_result": cmap[pred],
            "probabilities": {"A": float(probs[0]), "D": float(probs[1]), "H": float(probs[2])},
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded
