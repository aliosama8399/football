"""
ONNX Prediction Provider (Expert 1)
===================================
Replaces GNNPredictionProvider (torch) with onnxruntime inference.
Artifact:  models/export/gnn/tea_gnn.onnx  (+ tea_gnn_io.json)
Export:    python models/export_gnn.py --dynamic

Features:
  - Auto-exports `tea_gnn.onnx` if missing and a `.pt` checkpoint exists
  - Scores unplayed & cross-league matchups via synthetic pre-match query edges
  - Vectorized `predict_batch(matchups)` in a single ONNX Runtime session call
  - Hot-reloadable via `reload()`
"""

import sys
import os
import subprocess
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

from rag.providers.gnn_provider import BasePredictionProvider
from data.graph_builder import FootballGraphBuilder


def _resolve_onnx_path() -> Path:
    env_path = os.getenv("FOOTBALL_GNN_MODEL_PATH", "").strip()
    if env_path and env_path.endswith(".onnx"):
        p = Path(env_path)
        return p if p.is_absolute() else BASE_DIR / p
    try:
        if CFG_PATH.exists():
            with open(CFG_PATH) as f:
                p = yaml.safe_load(f).get("rag", {}).get("gnn_model_path", "")
                if p and str(p).endswith(".onnx"):
                    pp = Path(p)
                    if not pp.is_absolute():
                        pp = BASE_DIR / pp
                    return pp
    except Exception as e:
        logger.warning("Could not read rag.gnn_model_path: %s", e)
    return DEFAULT_ONNX_PATH


class ONNXPredictionProvider(BasePredictionProvider):
    """TEA-GNN expert via ONNX Runtime with synthetic query-edge & batch support."""

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
        self._is_dynamic_edges = False

    def reload(self) -> bool:
        self._loaded = False
        self.session = None
        self.graph = None
        self.builder = None
        return self.load()

    def _auto_export_if_missing(self) -> bool:
        """Automatically export tea_gnn.onnx if missing on disk and a .pt checkpoint exists."""
        export_script = BASE_DIR / "models" / "export_gnn.py"
        if not export_script.exists():
            return False
        logger.info("Auto-exporting missing ONNX GNN artifact via models/export_gnn.py --dynamic...")
        try:
            res = subprocess.run(
                [sys.executable, str(export_script), "--dynamic"],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if res.returncode == 0 and self.model_path.exists():
                logger.info("Auto-exported ONNX GNN to %s", self.model_path)
                return True
            logger.warning("Auto-export exited with %d: %s", res.returncode, (res.stderr or res.stdout)[-300:])
        except Exception as e:
            logger.warning("Auto-export of ONNX GNN failed: %s", e)
        return False

    def load(self) -> bool:
        if self._loaded:
            return True

        if not self.model_path.exists():
            if not self._auto_export_if_missing():
                logger.warning(
                    "ONNX GNN not found at %s. Run: python models/export_gnn.py --dynamic",
                    self.model_path,
                )
                return False

        try:
            import onnxruntime as ort

            io_path = self.model_path.parent / "tea_gnn_io.json"
            if io_path.exists():
                self._io_spec = json.loads(io_path.read_text())

            builder = FootballGraphBuilder(
                data_path=str(BASE_DIR / "data" / "processed" / "processed_matches.csv")
            )
            graph = builder.build_train_test_graphs()

            avail_providers = ort.get_available_providers()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail_providers else ["CPUExecutionProvider"]
            sess = ort.InferenceSession(str(self.model_path), providers=providers)

            # Inspect whether edge_index second dimension is dynamic
            self._is_dynamic_edges = False
            for inp in sess.get_inputs():
                if inp.name == "edge_index" and len(inp.shape) >= 2:
                    if not isinstance(inp.shape[1], int):
                        self._is_dynamic_edges = True

            self.builder = builder
            self.graph = graph
            self.session = sess
            self.graph["is_tea_gnn"] = True
            self._loaded = True
            logger.info(
                "Loaded ONNX GNN (Expert 1) from %s (dynamic_edges=%s, providers=%s)",
                self.model_path,
                self._is_dynamic_edges,
                sess.get_providers(),
            )
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
            return self._predict_batch_indices([hi], [ai])[0]
        except Exception as e:
            logger.error("ONNX GNN predict failed: %s", e)
            return None

    def predict_indices(self, home_idx: int, away_idx: int) -> Optional[dict]:
        if not self._loaded and not self.load():
            return None
        return self._predict_batch_indices([home_idx], [away_idx])[0]

    def predict_batch(self, matchups: list[dict]) -> list[dict]:
        """Vectorized batch prediction in a single ONNX Runtime session call."""
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
                logger.error("ONNX GNN batch predict failed: %s", e)
                return super().predict_batch(matchups)

        return [r for r in results if r is not None]

    def _predict_batch_indices(self, home_indices: list[int], away_indices: list[int]) -> list[dict]:
        """
        Score one or more (home_idx, away_idx) pairs using synthetic pre-match query edges
        at edge_time=0.0. Works seamlessly with both dynamic-axis and static-axis ONNX exports,
        and supports cross-league / previously unplayed matchups.
        """
        x = self.graph.get("x_test", self.graph["x"]).numpy().astype(np.float32)
        base_ei = self.graph["edge_index"].numpy().astype(np.int64)
        base_ea = self.graph["edge_attr"].numpy().astype(np.float32)
        edge_time = self.graph.get("edge_time")
        league_id = self.graph.get("league_id")
        base_et = edge_time.numpy().astype(np.float32) if edge_time is not None else np.zeros(base_ei.shape[1], dtype=np.float32)
        league_id_np = league_id.numpy().astype(np.int64) if league_id is not None else np.zeros(x.shape[0], dtype=np.int64)

        batch_size = len(home_indices)
        query_ei = np.array([home_indices, away_indices], dtype=np.int64)
        query_ea = self.builder.build_query_edge_attr(home_indices, away_indices, x)
        query_et = np.zeros(batch_size, dtype=np.float32)

        if self._is_dynamic_edges:
            ei = np.concatenate([base_ei, query_ei], axis=1)
            ea = np.concatenate([base_ea, query_ea], axis=0)
            et = np.concatenate([base_et, query_et], axis=0)
        else:
            # Static ONNX graph: place synthetic query edges in the final `batch_size` slots
            ei = base_ei.copy()
            ea = base_ea.copy()
            et = base_et.copy()
            ei[:, -batch_size:] = query_ei
            ea[-batch_size:, :] = query_ea
            et[-batch_size:] = query_et

        ort_inputs = {
            "x": x,
            "edge_index": ei,
            "edge_attr": ea,
            "edge_time": et,
            "league_id": league_id_np,
        }
        expected_names = {i.name for i in self.session.get_inputs()}
        sess_inputs = {k: v for k, v in ort_inputs.items() if k in expected_names}

        logits_all = self.session.run(["logits"], sess_inputs)[0]  # [E, 3]
        query_logits = logits_all[-batch_size:]

        cmap = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        outputs = []
        for i in range(batch_size):
            logits = query_logits[i]
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            pred = int(np.argmax(logits))
            outputs.append({
                "predicted_result": cmap[pred],
                "probabilities": {"A": float(probs[0]), "D": float(probs[1]), "H": float(probs[2])},
                "source": "live_model",
            })
        return outputs

    @property
    def is_loaded(self) -> bool:
        return self._loaded
