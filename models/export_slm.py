"""
Export finetuned SLM to ONNX (HF Serialization docs)
=====================================================
Source:  aliosama8399/football-analysisN  (Qwen3-0.6B, merged finetune, BF16)
Docs:    https://huggingface.co/docs/transformers/serialization  →  Optimum ONNX

This script follows the doc's programmatic path:
    ORTModelForCausalLM.from_pretrained(..., export=True) + save_pretrained
and the CLI equivalent:
    optimum-cli export onnx --model <id> --task text-generation-with-past <out>

No quantization. Tokenizer is exported alongside the model (required for
ORTModelForCausalLM.from_pretrained to work — see explanation in plan).

Usage:
    conda run -n football python models/export_slm.py
    conda run -n football python models/export_slm.py --model_id aliosama8399/football-analysisN --out models/export/slm
    optimum-cli export onnx --model aliosama8399/football-analysisN --task text-generation-with-past models/export/slm

Output (gitignored, ~1.2 GB):
    models/export/slm/model.onnx
    models/export/slm/config.json
    models/export/slm/tokenizer.json, tokenizer_config.json, special_tokens_map.json, vocab.json
    models/export/slm/chat_template.jinja  (if present)

Load test:
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("models/export/slm", trust_remote_code=True)
    m = ORTModelForCausalLM.from_pretrained("models/export/slm")
    inputs = tok("Compare Bournemouth and Wolves tactically", return_tensors="pt")
    out = m.generate(**inputs, max_new_tokens=32)
    print(tok.batch_decode(out, skip_special_tokens=True))
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# ── Qwen3 GQA fix (must run BEFORE optimum loads the model config) ───────────
# optimum 2.1.0 maps "qwen3" -> NormalizedTextConfig (missing num_key_value_heads),
# so the dummy past_key_values generator computes head_dim = hidden/heads = 64
# instead of the real head_dim = 128 -> export fails with
# "past_key_values.N.key Got 64 Expected 128". Registering the GQA-aware
# normalizer fixes KV shapes and lets the decoder merge succeed.
from optimum.utils.normalized_config import (
    NormalizedConfigManager,
    NormalizedTextConfigWithGQA,
)

NormalizedConfigManager._conf["qwen3"] = NormalizedTextConfigWithGQA

# ── MemoryError fix for tied-weight dedup post-process ────────────────────────
# optimum's check_and_save_model() calls model.ByteSize() which serializes the
# whole ~2.4GB proto in RAM -> MemoryError before its own external-data branch.
# Replace it with a version that always saves with external data (no ByteSize).
import onnx as _onnx


def _check_and_save_model_no_bytesize(model, save_path):
    save_path = Path(save_path).as_posix()
    external_file_name = os.path.basename(save_path) + "_data"
    if os.path.isfile(save_path):
        os.remove(save_path)
    _onnx.save(
        model,
        save_path,
        save_as_external_data=True,
        location=external_file_name,
        all_tensors_to_one_file=True,
        convert_attribute=True,
        size_threshold=100,
    )


import optimum.onnx.graph_transformations as _gt

_gt.check_and_save_model = _check_and_save_model_no_bytesize


DEFAULT_MODEL_ID = "aliosama8399/football-analysisN"
DEFAULT_OUT = Path(__file__).parent / "export" / "slm"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export finetuned Qwen SLM to ONNX via Optimum")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="HF model id or local path")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory (will be created)")
    p.add_argument("--task", default="text-generation-with-past", help="Optimum task (default: text-generation-with-past for KV-cache)")
    p.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"],
                   help="Export dtype. fp16 halves the protobuf size (~1.5GB) so the tied-weight "
                        "dedup post-process survives onnx.checker.MAXIMUM_PROTOBUF (2GB) without MemoryError.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Token mirrors hf_provider.py: config hf_token -> env HUGGINGFACE_HUB_TOKEN
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip() or None

    print(f"[export_slm] model_id={args.model_id}")
    print(f"[export_slm] task={args.task}")
    print(f"[export_slm] dtype={args.dtype}")
    print(f"[export_slm] out={out.resolve()}")

    from transformers import AutoTokenizer
    from optimum.exporters.onnx import main_export

    print("[export_slm] exporting via optimum main_export (same path as optimum-cli)...")
    # NOTE: called directly instead of ORTModelForCausalLM.from_pretrained(export=True)
    # because from_pretrained does NOT forward `dtype` to main_export (silently fp32).
    main_export(
        model_name_or_path=args.model_id,
        output=str(out),
        task=args.task,
        dtype=args.dtype,
        trust_remote_code=True,
        token=hf_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer.save_pretrained(out)

    # Verify expected files
    expected = ["model.onnx", "config.json", "tokenizer.json"]
    missing = [f for f in expected if not (out / f).exists()]
    if missing:
        print(f"[export_slm] WARNING: missing {missing}")
    else:
        print("[export_slm] export complete. Files:")
        for f in sorted(out.iterdir()):
            if f.is_file():
                mb = f.stat().st_size / 1_048_576
                print(f"  {f.name:<30} {mb:7.2f} MB")

    print("[export_slm] quick load test (CPU, with embed_size_per_head patch)...")
    try:
        from optimum.onnxruntime import ORTModelForCausalLM

        tok2 = AutoTokenizer.from_pretrained(str(out), trust_remote_code=True)
        m2 = ORTModelForCausalLM.from_pretrained(str(out))
        head_dim = getattr(m2.config, "head_dim", None)
        if getattr(m2, "can_use_cache", False) and head_dim:
            m2.embed_size_per_head = head_dim
        inputs = tok2("Hello", return_tensors="pt")
        gen = m2.generate(**inputs, max_new_tokens=8)
        print("[export_slm] generate OK:", tok2.batch_decode(gen, skip_special_tokens=True)[0][:120])
    except Exception as e:
        print("[export_slm] load/generate test failed (model still exported, but check env):", e)


if __name__ == "__main__":
    main()
