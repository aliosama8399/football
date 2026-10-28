"""
Export GNN to ONNX (HF-style serialization for PyTorch Geometric)

Source:  TEA-GNN checkpoint (newest of models/saved/gnn_tea-gnn_tuned.pt or gnn_tea-gnn.pt)
Output:  models/export/gnn/tea_gnn.onnx  (+ tea_gnn_io.json for runtime validation)

The GNN is a PyG model (tea_gnn.py:220 TEA_GNN_Model) with optional
edge_time [E] and league_id [N]. Dynamic axes are enabled by default so
synthetic pre-match query edges and variable batch sizes work seamlessly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAVED_DIR = Path(__file__).parent / "saved"
DEFAULT_OUT = Path(__file__).parent / "export" / "gnn"


def _resolve_default_checkpoint() -> Path:
    """Select the newest TEA-GNN checkpoint between tuned and standard training outputs."""
    tuned = SAVED_DIR / "gnn_tea-gnn_tuned.pt"
    base = SAVED_DIR / "gnn_tea-gnn.pt"
    if tuned.exists() and base.exists():
        return base if base.stat().st_mtime > tuned.stat().st_mtime else tuned
    if base.exists():
        return base
    return tuned


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export TEA-GNN to ONNX")
    p.add_argument("--checkpoint", default="", help="Path to .pt checkpoint (defaults to newest TEA-GNN .pt)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output dir")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset")
    p.add_argument("--dynamic", action="store_true", default=True, help="Use dynamic_axes for num_nodes/num_edges (default: True)")
    p.add_argument("--no-dynamic", action="store_true", help="Disable dynamic_axes")
    return p.parse_args()


def _load_model(checkpoint_path: Path):
    import torch

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        params = ckpt.get("best_params", {})
        hidden_dim = params.get("hidden_dim", ckpt.get("hidden_dim", 64))
        num_leagues = params.get("num_leagues", ckpt.get("num_leagues", 5))
        heads = params.get("heads", ckpt.get("heads", 4))
        dropout = params.get("dropout", ckpt.get("dropout", 0.3))
    else:
        state = ckpt
        hidden_dim, num_leagues, heads, dropout = 64, 5, 4, 0.3

    from data.graph_builder import FootballGraphBuilder

    num_node_features = len(FootballGraphBuilder.NODE_FEATURE_SUFFIXES)
    num_edge_features = len(FootballGraphBuilder.PRE_MATCH_EDGE_FEATURE_COLS)

    from models.gnn_models import get_model

    ckpt_name = checkpoint_path.name.lower()
    if "tea" in ckpt_name:
        model_name = "TEA-GNN"
    elif "gat" in ckpt_name:
        model_name = "GAT"
    elif "sage" in ckpt_name:
        model_name = "GraphSAGE"
    else:
        model_name = "TEA-GNN"

    has_cross = any(k.startswith("cross_league.") or k.replace("module.", "").startswith("cross_league.") for k in state.keys())
    use_cross = has_cross
    if not has_cross:
        print("[export_gnn] checkpoint has no cross_league weights -> exporting with use_cross_league=False")

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
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
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
    use_dynamic = args.dynamic and not args.no_dynamic
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _resolve_default_checkpoint()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[export_gnn] checkpoint={ckpt_path}")
    print(f"[export_gnn] out={out_dir.resolve()}  opset={args.opset}  dynamic={use_dynamic}")

    import torch
    import numpy as np

    model, num_node_feats, num_edge_feats = _load_model(ckpt_path)
    print(f"[export_gnn] model={model.__class__.__name__}  node_feats={num_node_feats}  edge_feats={num_edge_feats}")

    N, E = 20, 60
    x = torch.randn(N, num_node_feats)
    edge_index = torch.randint(0, N, (2, E), dtype=torch.long)
    edge_attr = torch.randn(E, num_edge_feats)
    edge_time = torch.rand(E) * 2.0
    league_id = torch.randint(0, 5, (N,), dtype=torch.long)

    class _Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, edge_index, edge_attr, edge_time, league_id):
            return self.m(x, edge_index, edge_attr, edge_time=edge_time, league_id=league_id)

    wrapped = _Wrapper(model)

    with torch.no_grad():
        torch_out = wrapped(x, edge_index, edge_attr, edge_time, league_id)

    onnx_path = out_dir / "tea_gnn.onnx"
    io_path = out_dir / "tea_gnn_io.json"

    dynamic_axes = None
    if use_dynamic:
        dynamic_axes = {
            "x": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "edge_attr": {0: "num_edges"},
            "edge_time": {0: "num_edges"},
            "league_id": {0: "num_nodes"},
            "logits": {0: "num_edges"},
        }

    # Prefer torch.onnx.export when dynamic_axes are requested so variable edge counts work reliably
    exported = False
    if not use_dynamic:
        try:
            ep = torch.onnx.dynamo_export(wrapped, x, edge_index, edge_attr, edge_time, league_id)
            ep.save(str(onnx_path))
            exported = True
            print("[export_gnn] exported via torch.onnx.dynamo_export")
        except Exception as e:
            print(f"[export_gnn] dynamo_export failed ({e}), falling back to torch.onnx.export ...")

    if not exported:
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
        print("[export_gnn] exported via torch.onnx.export (dynamic_axes enabled)" if use_dynamic else "[export_gnn] exported via torch.onnx.export")

    if not exported or not onnx_path.exists():
        raise RuntimeError("ONNX export failed")

    io_spec = {
        "inputs": {
            "x": {"shape": [N, num_node_feats], "dtype": "float32", "dynamic_axis": 0 if use_dynamic else None},
            "edge_index": {"shape": [2, E], "dtype": "int64", "dynamic_axis": 1 if use_dynamic else None},
            "edge_attr": {"shape": [E, num_edge_feats], "dtype": "float32", "dynamic_axis": 0 if use_dynamic else None},
            "edge_time": {"shape": [E], "dtype": "float32", "dynamic_axis": 0 if use_dynamic else None},
            "league_id": {"shape": [N], "dtype": "int64", "dynamic_axis": 0 if use_dynamic else None},
        },
        "output": {"logits": {"shape": [E, 3], "dtype": "float32"}},
        "opset": args.opset,
        "dynamic": use_dynamic,
        "checkpoint": str(ckpt_path),
        "num_node_features": num_node_feats,
        "num_edge_features": num_edge_feats,
    }
    io_path.write_text(json.dumps(io_spec, indent=2))
    print(f"[export_gnn] wrote {io_path}")

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
    except Exception as e:
        print(f"[export_gnn] parity check skipped/failed: {e}")

    print(f"[export_gnn] done. Files in {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:<20} {f.stat().st_size/1024:7.1f} KB")


if __name__ == "__main__":
    main()
