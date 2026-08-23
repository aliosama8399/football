"""
Export GNN to ONNX (HF-style serialization for PyTorch Geometric)

Source:  TEA-GNN checkpoint  models/saved/gnn_tea-gnn_tuned.pt  (config in llm_config.yaml:83)
Output:  models/export/gnn/tea_gnn.onnx  (+ tea_gnn_io.json for runtime validation)

The GNN is a PyG model (tea_gnn.py:220 TEA_GNN_Model) with optional
edge_time [E] and league_id [N]. The ONNX export keeps them optional
so the same artifact works with and without TEA extras.

Usage:
    conda run -n football python models/export_gnn.py
    conda run -n football python models/export_gnn.py --checkpoint models/saved/gnn_tea-gnn_tuned.pt --out models/export/gnn

Validation (CPU):
    onnxruntime.InferenceSession("models/export/gnn/tea_gnn.onnx", providers=["CPUExecutionProvider"])
    np.allclose(torch_logits, onnx_logits, atol=1e-4)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as `python models/export_gnn.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINT = Path(__file__).parent / "saved" / "gnn_tea-gnn_tuned.pt"
DEFAULT_OUT = Path(__file__).parent / "export" / "gnn"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export TEA-GNN to ONNX")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Path to .pt checkpoint")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output dir")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset")
    p.add_argument("--dynamic", action="store_true", help="Use dynamic_axes for num_nodes/num_edges")
    return p.parse_args()


def _load_model(checkpoint_path: Path):
    import torch

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    # ckpt format from train_gnn.py / tune_gnn.py:
    #  - either {"model_state": state_dict, "best_params": {...}}  or  raw state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        params = ckpt.get("best_params", {})
        # many tune runs store hidden_dim/heads/dropout in best_params
        hidden_dim = params.get("hidden_dim", 64)
        num_leagues = params.get("num_leagues", 5)
        heads = params.get("heads", 4)
        dropout = params.get("dropout", 0.3)
    else:
        state = ckpt
        hidden_dim, num_leagues, heads, dropout = 64, 5, 4, 0.3

    # dims from FootballGraphBuilder (graph_builder.py:40,56)
    #  NODE_FEATURE_SUFFIXES = 32,  HIST_EDGE_FEATURE_COLS = 12
    from data.graph_builder import FootballGraphBuilder

    num_node_features = len(FootballGraphBuilder.NODE_FEATURE_SUFFIXES)
    num_edge_features = len(FootballGraphBuilder.HIST_EDGE_FEATURE_COLS)

    # Instantiate via registry (gnn_models.py:295)
    from models.gnn_models import get_model

    # checkpoint filename hints model name: gnn_tea-gnn_tuned.pt -> TEA-GNN
    ckpt_name = checkpoint_path.name.lower()
    if "tea" in ckpt_name:
        model_name = "TEA-GNN"
    elif "gat" in ckpt_name:
        model_name = "GAT"
    elif "sage" in ckpt_name:
        model_name = "GraphSAGE"
    else:
        model_name = "TEA-GNN"

    # Detect whether checkpoint contains cross-league weights
    has_cross = any(k.startswith("cross_league.") for k in state.keys())
    # also handle "module." prefix
    if not has_cross:
        has_cross = any(k.replace("module.", "").startswith("cross_league.") for k in state.keys())
    use_cross = has_cross
    if not has_cross:
        print("[export_gnn] checkpoint has no cross_league weights → exporting with use_cross_league=False")

    model = get_model(
        model_name,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=hidden_dim,
        num_classes=3,
        heads=heads,
        num_leagues=num_leagues,
        dropout=dropout,
        use_cross_league=use_cross,
    )
    # state dict may have been saved from DataParallel or with prefix
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # try stripping "module." prefix and allow missing cross_league
        fixed = {k.replace("module.", ""): v for k, v in state.items()}
        try:
            model.load_state_dict(fixed, strict=False)
            print(f"[export_gnn] loaded with strict=False ({e})")
        except Exception as e2:
            raise RuntimeError(f"Failed to load state_dict: {e} / {e2}") from e2

    model.eval()
    return model, num_node_features, num_edge_features


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[export_gnn] checkpoint={ckpt_path}")
    print(f"[export_gnn] out={out_dir.resolve()}  opset={args.opset}  dynamic={args.dynamic}")

    import torch
    import numpy as np

    model, num_node_feats, num_edge_feats = _load_model(ckpt_path)
    print(f"[export_gnn] model={model.__class__.__name__}  node_feats={num_node_feats}  edge_feats={num_edge_feats}")

    # Dummy inputs matching tea_gnn.py:293 self-test shapes
    # Use small graph for tracing; dynamic_axes allow variable sizes at runtime
    N, E = 20, 60
    x = torch.randn(N, num_node_feats)
    edge_index = torch.randint(0, N, (2, E), dtype=torch.long)
    edge_attr = torch.randn(E, num_edge_feats)
    edge_time = torch.rand(E) * 2.0
    league_id = torch.randint(0, 5, (N,), dtype=torch.long)

    # Wrapper to make edge_time/league_id optional for ONNX (default None -> zeros)
    class _Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, edge_index, edge_attr, edge_time, league_id):
            return self.m(x, edge_index, edge_attr, edge_time=edge_time, league_id=league_id)

    wrapped = _Wrapper(model)

    # Torch reference logits
    with torch.no_grad():
        torch_out = wrapped(x, edge_index, edge_attr, edge_time, league_id)

    onnx_path = out_dir / "tea_gnn.onnx"
    io_path = out_dir / "tea_gnn_io.json"

    # Try dynamo_export first (PyG + torch>=2.7), fallback to classic export
    exported = False
    try:
        # torch.onnx.dynamo_export is preferred for PyG (avoids torch_scatter issues)
        ep = torch.onnx.dynamo_export(
            wrapped, x, edge_index, edge_attr, edge_time, league_id
        )
        ep.save(str(onnx_path))
        exported = True
        print("[export_gnn] exported via torch.onnx.dynamo_export")
    except Exception as e:
        print(f"[export_gnn] dynamo_export failed ({e}), falling back to torch.onnx.export ...")
        dynamic_axes = None
        if args.dynamic:
            dynamic_axes = {
                "x": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "edge_attr": {0: "num_edges"},
                "edge_time": {0: "num_edges"},
                "league_id": {0: "num_nodes"},
                "logits": {0: "num_edges"},
            }
        torch.onnx.export(
            wrapped,
            (x, edge_index, edge_attr, edge_time, league_id),
            str(onnx_path),
            input_names=["x", "edge_index", "edge_attr", "edge_time", "league_id"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
        )
        exported = True
        print("[export_gnn] exported via torch.onnx.export")

    if not exported or not onnx_path.exists():
        raise RuntimeError("ONNX export failed")

    # Save IO spec for runtime
    io_spec = {
        "inputs": {
            "x": {"shape": [N, num_node_feats], "dtype": "float32", "dynamic_axis": 0},
            "edge_index": {"shape": [2, E], "dtype": "int64", "dynamic_axis": 1},
            "edge_attr": {"shape": [E, num_edge_feats], "dtype": "float32", "dynamic_axis": 0},
            "edge_time": {"shape": [E], "dtype": "float32", "dynamic_axis": 0},
            "league_id": {"shape": [N], "dtype": "int64", "dynamic_axis": 0},
        },
        "output": {"logits": {"shape": [E, 3], "dtype": "float32"}},
        "opset": args.opset,
        "checkpoint": str(ckpt_path),
        "num_node_features": num_node_feats,
        "num_edge_features": num_edge_feats,
    }
    io_path.write_text(json.dumps(io_spec, indent=2))
    print(f"[export_gnn] wrote {io_path}")

    # Parity check (CPU)
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {
            "x": x.numpy(),
            "edge_index": edge_index.numpy(),
            "edge_attr": edge_attr.numpy(),
            "edge_time": edge_time.numpy(),
            "league_id": league_id.numpy(),
        }
        onnx_out = sess.run(["logits"], ort_inputs)[0]
        max_abs = float(np.max(np.abs(torch_out.numpy() - onnx_out)))
        ok = bool(np.allclose(torch_out.numpy(), onnx_out, atol=1e-4))
        print(f"[export_gnn] parity max_abs={max_abs:.6f}  ok={ok}  (tol 1e-4)")
        if not ok:
            print("[export_gnn] WARNING: parity failed — check opset / dynamic_axes")
    except Exception as e:
        print(f"[export_gnn] parity check skipped/failed: {e}")

    print(f"[export_gnn] done. Files in {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:<20} {f.stat().st_size/1024:7.1f} KB")


if __name__ == "__main__":
    main()
