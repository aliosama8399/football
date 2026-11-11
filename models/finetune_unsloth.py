"""
Professional Football Tactical SLM Fine-Tuning Script (Unsloth & Unsloth Studio)
=================================================================================
Replaces legacy notebooks (`models/finetune_qwen25.ipynb` & `models/finetune_smollm2.ipynb`)
with a production-grade CLI pipeline powered by the latest Unsloth & Unsloth Studio config schema.

Hardware-Tuned for NVIDIA GTX 1660 Ti (6 GB VRAM, Turing sm_75):
  1. Prevents Windows Shared GPU Memory spillover (which slows training 10x-25x)
     by bounding `per_device_train_batch_size=2` + `gradient_accumulation_steps=4`.
  2. Forces `lora_dropout=0.0` & `bias="none"` to unlock Unsloth's fused Triton QLoRA kernels (~2x speedup).
  3. Enforces native Turing `float16` (`fp16=True, bf16=False`) since sm_75 lacks hardware BF16.
  4. Enables `use_gradient_checkpointing="unsloth"` (async host-RAM activation offloading).
  5. Uses `optim="adamw_8bit"`, `group_by_length=True`, and `max_length=1024`.
  6. Caps intermediate evaluation samples (`--max-eval-samples 50`) so eval steps take seconds.

Usage:
  # 1. Dry-run to inspect dataset, Unsloth Studio config & GTX 1660 Ti 6GB tuning (no GPU training):
  python models/finetune_unsloth.py --dry-run

  # 2. Train with the GTX 1660 Ti 6GB Unsloth Studio preset (fast ~15-25 min run):
  python models/finetune_unsloth.py --studio-config models/unsloth_studio_1660ti.yaml

  # 3. Train using any existing Unsloth Studio YAML in models/ (auto-tuned for 6GB VRAM):
  python models/finetune_unsloth.py --studio-config models/Qwen3-0.6B_qlora_dataset_2026-04-08-07-03-35.yaml

  # 4. Train SmolLM2 or Qwen2.5 and export merged 16-bit + GGUF for Ollama:
  python models/finetune_unsloth.py --model-name unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit --save-merged-16bit --save-gguf q4_k_m
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Reduce CUDA memory fragmentation on 6 GB VRAM GPUs before torch initializes
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Prevent Unsloth Zoo from offloading FP32 activations in BFloat16 on Turing (sm_75 / GTX 1660 Ti lacks native BF16)
os.environ.setdefault("UNSLOTH_OFFLOAD_FP32", "exact")

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (unsloth-finetune) %(message)s",
)
logger = logging.getLogger("unsloth_finetune")

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert football tactical analyst. "
    "Analyse the match data provided, predict the most likely outcome, "
    "and deliver a detailed tactical report covering both teams' "
    "strengths, weaknesses, and key strategic factors."
)

DEFAULT_TEST_PROMPT = """Analyze the upcoming match: Manchester City (Home) vs Arsenal (Away).

Manchester City recent form (last 5 matches):
  - Points per match (last 5): 2.20
  - Goals scored per match (last 5): 3.20
  - Goals conceded per match (last 5): 0.80
  - Shots on target per match (last 5): 7.40

Arsenal recent form (last 5 matches):
  - Points per match (last 5): 1.80
  - Goals scored per match (last 5): 2.10
  - Goals conceded per match (last 5): 1.20
  - Shots on target per match (last 5): 5.80

Head-to-Head: Total meetings: 10, Home wins: 4, Away wins: 2, Draws: 4

Prediction: Home Win (Home 58.3% | Draw 24.1% | Away 17.6%)"""


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION SCHEMA (Compatible with Unsloth Studio YAML)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FinetuneConfig:
    """Unified configuration matching Unsloth Studio YAML + CLI overrides."""

    # Model
    model_name_or_path: str = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    chat_template: str = "qwen-2.5"
    load_in_4bit: bool = True
    dtype: Optional[str] = "float16"

    # Dataset
    jsonl_path: str = "data/finetune/football_tactical.jsonl"
    train_json: str = "data/finetune/football_train.json"
    val_json: str = "data/finetune/football_val.json"
    val_split: float = 0.10
    max_eval_samples: int = 50

    # Training (Unsloth Studio `training:` block)
    max_seq_length: int = 1024
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    max_steps: int = 150
    save_steps: int = 50
    eval_steps: float = 50
    weight_decay: float = 0.01
    random_seed: int = 3407
    packing: bool = False
    group_by_length: bool = True
    train_on_completions: bool = True
    gradient_checkpointing: str = "unsloth"
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    output_dir: str = "models/finetuned_qwen_unsloth"
    resume_from_checkpoint: Optional[str] = None

    @property
    def max_length(self) -> int:
        return self.max_seq_length

    @max_length.setter
    def max_length(self, val: int) -> None:
        self.max_seq_length = int(val)

    # LoRA (Unsloth Studio `lora:` block)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_rslora: bool = True
    use_loftq: bool = False
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True

    @classmethod
    def from_studio_yaml(cls, yaml_path: Path) -> "FinetuneConfig":
        """Load configuration from an Unsloth Studio YAML file."""
        cfg = cls()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Unsloth Studio config not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        model_block = raw.get("model", {})
        ds_block = raw.get("dataset", {})
        train_block = raw.get("training", {})
        lora_block = raw.get("lora", {})

        # Infer model name from YAML filename if not in `model:` block (Unsloth Studio convention)
        stem = yaml_path.stem.lower()
        if "model_name_or_path" in model_block:
            cfg.model_name_or_path = str(model_block["model_name_or_path"])
        elif "qwen3.5-0.8b" in stem:
            cfg.model_name_or_path = "Qwen/Qwen3.5-0.8B"
        elif "qwen3-0.6b" in stem:
            cfg.model_name_or_path = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
        elif "smollm2" in stem:
            cfg.model_name_or_path = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            cfg.chat_template = "chatml"

        for k, v in model_block.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
        for k, v in ds_block.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
        for k, v in train_block.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
        for k, v in lora_block.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)

        return cfg

    def to_studio_yaml_dict(self) -> dict[str, Any]:
        """Serialize back to the Unsloth Studio YAML structure."""
        return {
            "model": {
                "model_name_or_path": self.model_name_or_path,
                "chat_template": self.chat_template,
                "load_in_4bit": self.load_in_4bit,
                "dtype": self.dtype,
            },
            "dataset": {
                "jsonl_path": self.jsonl_path,
                "train_json": self.train_json,
                "val_json": self.val_json,
                "val_split": self.val_split,
                "max_eval_samples": self.max_eval_samples,
            },
            "training": {
                "max_seq_length": self.max_seq_length,
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "warmup_steps": self.warmup_steps,
                "max_steps": self.max_steps,
                "save_steps": self.save_steps,
                "eval_steps": self.eval_steps,
                "weight_decay": self.weight_decay,
                "random_seed": self.random_seed,
                "packing": self.packing,
                "group_by_length": self.group_by_length,
                "train_on_completions": self.train_on_completions,
                "gradient_checkpointing": self.gradient_checkpointing,
                "optim": self.optim,
                "lr_scheduler_type": self.lr_scheduler_type,
                "output_dir": self.output_dir,
            },
            "lora": {
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": self.target_modules,
                "use_rslora": self.use_rslora,
                "use_loftq": self.use_loftq,
                "finetune_vision_layers": self.finetune_vision_layers,
                "finetune_language_layers": self.finetune_language_layers,
                "finetune_attention_modules": self.finetune_attention_modules,
                "finetune_mlp_modules": self.finetune_mlp_modules,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GTX 1660 Ti (6 GB VRAM) HARDWARE AUTO-TUNER
# ═══════════════════════════════════════════════════════════════════════════════


def detect_gpu_and_tune_config(cfg: FinetuneConfig, auto_tune_6gb: bool = True) -> dict[str, Any]:
    """Detect GPU capabilities and apply memory/speed tuning for 6 GB GPUs (GTX 1660 Ti)."""
    gpu_info: dict[str, Any] = {
        "cuda_available": False,
        "device_name": "CPU",
        "vram_gb": 0.0,
        "compute_capability": (0, 0),
        "supports_bfloat16": False,
        "is_6gb_constrained": False,
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 2)
            cc = (props.major, props.minor)
            gpu_info.update(
                {
                    "cuda_available": True,
                    "device_name": props.name,
                    "vram_gb": vram_gb,
                    "compute_capability": cc,
                    "supports_bfloat16": cc[0] >= 8,  # Ampere+ (sm_80+)
                    "is_6gb_constrained": vram_gb <= 6.5,
                }
            )
    except Exception as e:
        logger.debug("Torch GPU check skipped: %s", e)

    if not auto_tune_6gb:
        return gpu_info

    # Apply GTX 1660 Ti / 6GB VRAM optimizations automatically
    if gpu_info["is_6gb_constrained"] or not gpu_info["supports_bfloat16"]:
        logger.info(
            "Applying GTX 1660 Ti 6GB Turbo Profile (Device: %s, VRAM: %.2f GB, CC: %s)",
            gpu_info["device_name"],
            gpu_info["vram_gb"],
            gpu_info["compute_capability"],
        )

        # 1. Force lora_dropout = 0.0 to activate Unsloth's fused Triton kernels
        if cfg.lora_dropout > 0.0:
            logger.info(
                "  [Speed Fix] Overriding lora_dropout %.3f -> 0.0 (enables Unsloth fast Triton QLoRA kernels)",
                cfg.lora_dropout,
            )
            cfg.lora_dropout = 0.0

        # 2. For long sequences (>2000 tokens), use micro-batch=1; otherwise cap at 2
        total_eff_batch = cfg.batch_size * cfg.gradient_accumulation_steps
        if cfg.max_seq_length > 2000:
            cfg.batch_size = 1
            cfg.gradient_accumulation_steps = max(8, total_eff_batch)
            logger.info(
                "  [VRAM Fix] Configured micro_batch=1, grad_accum=%d for max_seq_length=%d on 6GB GPU",
                cfg.gradient_accumulation_steps,
                cfg.max_seq_length,
            )
        elif cfg.batch_size > 2:
            cfg.batch_size = 2
            cfg.gradient_accumulation_steps = max(4, total_eff_batch // 2)
            logger.info(
                "  [VRAM Fix] Adjusted batch_size=2, gradient_accumulation_steps=%d (prevents PCIe RAM spill on 6GB VRAM)",
                cfg.gradient_accumulation_steps,
            )

        # 3. Cap max_seq_length to 2560 on 6GB GPU to prevent OOM
        if cfg.max_seq_length > 2560:
            logger.info(
                "  [VRAM Fix] Capped max_seq_length %d -> 2560 for 6GB VRAM safety",
                cfg.max_seq_length,
            )
            cfg.max_seq_length = 2560

        # 4. Turing (CC 7.5) requires FP16, not BF16
        if not gpu_info["supports_bfloat16"]:
            cfg.dtype = "float16"

        # 5. Ensure Unsloth gradient checkpointing & 8-bit AdamW
        cfg.gradient_checkpointing = "unsloth"
        cfg.optim = "adamw_8bit"
        cfg.group_by_length = True

        # 6. Fix fractional eval_steps (e.g. 0.1 -> 50 steps) so eval doesn't run 10x
        if isinstance(cfg.eval_steps, float) and cfg.eval_steps < 1.0:
            cfg.eval_steps = 50
            logger.info("  [Speed Fix] Adjusted fractional eval_steps -> 50 steps")

    return gpu_info


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET PREPARATION (JSONL -> ShareGPT Train/Val JSON)
# ═══════════════════════════════════════════════════════════════════════════════


def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure records contain both 'messages' and 'conversations' keys."""
    role_map_to_sharegpt = {"user": "human", "assistant": "gpt", "system": "system"}
    role_map_to_chat = {"human": "user", "gpt": "assistant", "system": "system", "bot": "assistant"}

    normalized: list[dict[str, Any]] = []
    for rec in records:
        item = dict(rec)
        if "messages" in item and isinstance(item["messages"], list):
            if "conversations" not in item:
                item["conversations"] = [
                    {
                        "from": role_map_to_sharegpt.get(m.get("role", "user"), m.get("role", "user")),
                        "value": m.get("content", ""),
                    }
                    for m in item["messages"]
                    if isinstance(m, dict)
                ]
        elif "conversations" in item and isinstance(item["conversations"], list):
            if "messages" not in item:
                item["messages"] = [
                    {
                        "role": role_map_to_chat.get(m.get("from", "human"), "user"),
                        "content": m.get("value", m.get("text", "")),
                    }
                    for m in item["conversations"]
                    if isinstance(m, dict)
                ]
        normalized.append(item)
    return normalized


def prepare_sharegpt_splits(
    cfg: FinetuneConfig,
    rebuild_splits: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load or build ShareGPT train/val splits from `football_tactical.jsonl`."""
    jsonl_path = BASE_DIR / cfg.jsonl_path
    train_path = BASE_DIR / cfg.train_json
    val_path = BASE_DIR / cfg.val_json

    if not rebuild_splits and train_path.exists() and val_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_ds = json.load(f)
        with open(val_path, "r", encoding="utf-8") as f:
            val_ds = json.load(f)
        if train_ds and val_ds:
            train_ds = _normalize_records(train_ds)
            val_ds = _normalize_records(val_ds)
            logger.info(
                "Loaded existing splits: Train=%d (%s) | Val=%d (%s)",
                len(train_ds),
                train_path.name,
                len(val_ds),
                val_path.name,
            )
            return train_ds, val_ds

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Source tactical JSONL not found: {jsonl_path}")

    role_map = {"user": "human", "assistant": "gpt", "system": "system"}
    records: list[dict[str, Any]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec.get("messages", [])
            if not messages:
                continue
            conversations = [
                {"from": role_map.get(m["role"], m["role"]), "value": m["content"]}
                for m in messages
                if "role" in m and "content" in m
            ]
            if conversations and conversations[0]["from"] != "system":
                conversations.insert(0, {"from": "system", "value": DEFAULT_SYSTEM_PROMPT})
            normalized_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if "role" in m and "content" in m
            ]
            if normalized_messages and normalized_messages[0]["role"] != "system":
                normalized_messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
            records.append({
                "messages": normalized_messages,
                "conversations": conversations,
            })

    if not records:
        raise RuntimeError(f"Zero valid records parsed from {jsonl_path}")

    random.Random(cfg.random_seed).shuffle(records)
    val_n = max(1, int(len(records) * cfg.val_split))
    val_ds = records[:val_n]
    train_ds = records[val_n:]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_ds, f, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_ds, f, ensure_ascii=False)

    logger.info("Built ShareGPT splits from %d records: Train=%d | Val=%d", len(records), len(train_ds), len(val_ds))
    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════════════════════
# UNSLOTH TRAINING & EXPORT PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def run_unsloth_training(
    cfg: FinetuneConfig,
    gpu_info: dict[str, Any],
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    save_lora: bool = True,
    save_merged_16bit: bool = False,
    save_gguf: str = "",
    export_onnx: bool = False,
    test_inference: bool = False,
    promote_production: bool = False,
    selection_metric: str = "eval_loss",
) -> dict[str, Any]:
    """Execute QLoRA fine-tuning using Unsloth FastLanguageModel and TRL SFTTrainer."""
    import enum  # noqa: I001
    import torch  # noqa: I001
    import torch.utils._pytree as _pytree  # noqa: I001

    # PyTorch 2.7+ natively supports Enum subclasses in torch.compile; skip redundant registration
    _orig_reg = getattr(_pytree, "register_constant", None)
    if _orig_reg is not None and not getattr(_orig_reg, "_enum_guarded", False):
        def _safe_register_constant(cls):
            if isinstance(cls, type) and issubclass(cls, enum.Enum):
                return cls
            return _orig_reg(cls)
        _safe_register_constant._enum_guarded = True  # type: ignore[attr-defined]
        _pytree.register_constant = _safe_register_constant

    # CRITICAL: unsloth MUST be imported before trl/transformers/peft so SFTTrainer is patched into UnslothSFTTrainer
    from unsloth import FastLanguageModel  # noqa: I001
    import transformers  # noqa: I001
    _orig_tok_from_pretrained = transformers.AutoTokenizer.from_pretrained
    if not getattr(_orig_tok_from_pretrained, "_mistral_regex_guarded", False):
        def _safe_tok_from_pretrained(*args, **kwargs):
            if "fix_mistral_regex" not in kwargs:
                kwargs["fix_mistral_regex"] = True
            return _orig_tok_from_pretrained(*args, **kwargs)
        _safe_tok_from_pretrained._mistral_regex_guarded = True
        transformers.AutoTokenizer.from_pretrained = _safe_tok_from_pretrained

    from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only  # noqa: I001
    from datasets import Dataset  # noqa: I001
    from trl import SFTConfig, SFTTrainer  # noqa: I001

    torch.cuda.empty_cache()

    use_bf16 = bool(gpu_info.get("supports_bfloat16", False)) and cfg.dtype == "bfloat16"
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if not use_bf16:
        # Enforce exact FP32 offload mode in unsloth_zoo gradient checkpointing to prevent rogue BF16 activations
        os.environ["UNSLOTH_OFFLOAD_FP32"] = "exact"
        try:
            import unsloth_zoo.gradient_checkpointing as unsloth_gc

            unsloth_gc.FP32_OFFLOAD_MODE = "exact"
            unsloth_gc.FP32_OFFLOAD_EXACT = True
        except Exception:
            pass

        # Defensively patch GradScaler to guard against any rogue BF16 params/grads on Turing (sm_75)
        try:
            from torch.amp.grad_scaler import GradScaler

            orig_unscale = GradScaler._unscale_grads_

            def safe_unscale(self, optimizer, inv_scale, found_inf, allow_fp16):
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.dtype == torch.bfloat16:
                            param.data = param.data.to(torch.float32)
                        if param.grad is not None and param.grad.dtype == torch.bfloat16:
                            param.grad = param.grad.to(torch.float32)
                return orig_unscale(self, optimizer, inv_scale, found_inf, allow_fp16)

            GradScaler._unscale_grads_ = safe_unscale
        except Exception:
            pass

    logger.info(
        "Loading base model via Unsloth: %s (4-bit=%s, dtype=%s, max_seq_len=%d)",
        cfg.model_name_or_path,
        cfg.load_in_4bit,
        torch_dtype,
        cfg.max_seq_length,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name_or_path,
        max_seq_length=cfg.max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=cfg.load_in_4bit,
        trust_remote_code=True,
    )
    if hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, dict):
        tokenizer.init_kwargs["fix_mistral_regex"] = True
    setattr(tokenizer, "fix_mistral_regex", True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing=cfg.gradient_checkpointing,
        random_state=cfg.random_seed,
        use_rslora=cfg.use_rslora,
        loftq_config=None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=cfg.chat_template,
    )

    # Cap validation set for fast intermediate eval steps on 6GB GPU
    if cfg.max_eval_samples > 0 and len(val_records) > cfg.max_eval_samples:
        logger.info(
            "Subsampling validation set %d -> %d records for fast eval on 6GB GPU",
            len(val_records),
            cfg.max_eval_samples,
        )
        val_records = val_records[: cfg.max_eval_samples]

    raw_train_ds = Dataset.from_list(train_records)
    raw_val_ds = Dataset.from_list(val_records)
    try:
        if "conversations" in raw_train_ds.column_names:
            raw_train_ds = standardize_sharegpt(raw_train_ds)
        if "conversations" in raw_val_ds.column_names:
            raw_val_ds = standardize_sharegpt(raw_val_ds)
    except Exception as e:
        logger.debug("standardize_sharegpt skipped: %s", e)

    def _format_prompts(batch: dict[str, Any]) -> dict[str, list[str]]:
        texts: list[str] = []
        if "messages" in batch:
            raw_convos = batch["messages"]
        elif "conversations" in batch:
            raw_convos = batch["conversations"]
        elif "text" in batch:
            return {"text": [str(t) for t in batch["text"]]}
        else:
            first_val = next(iter(batch.values()), [])
            raw_convos = [[] for _ in range(len(first_val))] if isinstance(first_val, list) else []

        role_map = {"human": "user", "gpt": "assistant", "system": "system", "bot": "assistant"}
        for convo in raw_convos:
            if not isinstance(convo, list):
                texts.append("")
                continue
            normalized_msgs: list[dict[str, str]] = []
            for m in convo:
                if not isinstance(m, dict):
                    continue
                if "from" in m:
                    role = role_map.get(m.get("from", "human"), "user")
                    content = m.get("value", m.get("text", m.get("content", "")))
                else:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                normalized_msgs.append({"role": str(role), "content": str(content)})

            if normalized_msgs:
                formatted = tokenizer.apply_chat_template(
                    normalized_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(formatted)
            else:
                texts.append("")
        return {"text": texts}

    train_dataset = raw_train_ds.map(_format_prompts, batched=True, desc="Formatting train prompts")
    val_dataset = raw_val_ds.map(_format_prompts, batched=True, desc="Formatting val prompts")

    out_dir = BASE_DIR / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure SFTConfig dynamically matching installed TRL signature
    sft_config_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    sft_kwargs: dict[str, Any] = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "warmup_steps": cfg.warmup_steps,
        "learning_rate": cfg.learning_rate,
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "logging_steps": 10,
        "optim": cfg.optim,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "seed": cfg.random_seed,
        "save_strategy": "steps",
        "save_steps": int(cfg.save_steps),
        "save_total_limit": 2,
        "eval_strategy": "steps" if cfg.eval_steps and cfg.eval_steps > 0 else "no",
        "eval_steps": int(cfg.eval_steps) if cfg.eval_steps and cfg.eval_steps > 0 else None,
        "dataset_text_field": "text",
        "dataset_num_proc": 1,  # Safe on Windows (avoids multiprocessing spawn hang)
        "packing": cfg.packing,
        "group_by_length": cfg.group_by_length and not cfg.packing,
        "report_to": "none",
    }
    if "max_length" in sft_config_params:
        sft_kwargs["max_length"] = cfg.max_seq_length
    elif "max_seq_length" in sft_config_params:
        sft_kwargs["max_seq_length"] = cfg.max_seq_length

    if cfg.max_steps and cfg.max_steps > 0:
        sft_kwargs["max_steps"] = int(cfg.max_steps)
    else:
        sft_kwargs["num_train_epochs"] = float(cfg.num_epochs)

    filtered_sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in sft_config_params}

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(**filtered_sft_kwargs),
    )

    if not use_bf16:
        # Guard against TRL 1.15+ SFTTrainer.__init__ converting quantized PEFT adapters to bfloat16 on non-BF16 GPUs
        for param in model.parameters():
            if param.requires_grad and param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float32)

    if cfg.train_on_completions:
        try:
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
            logger.info("Enabled Unsloth train_on_responses_only (assistant-only loss masking).")
        except Exception as e:
            logger.warning("Could not apply train_on_responses_only (%s); continuing standard SFT.", e)

    # Track run in MLflow if available
    mlflow_mod = None
    try:
        from models.mlflow_config import safe_end_run, setup_mlflow

        safe_end_run()
        mlflow_mod = setup_mlflow("football-slm-finetuning")
        mlflow_mod.start_run(run_name=f"Unsloth-{Path(cfg.model_name_or_path).name}")
        mlflow_mod.log_params(
            {
                "model": cfg.model_name_or_path,
                "max_length": cfg.max_length,
                "batch_size": cfg.batch_size,
                "grad_accum": cfg.gradient_accumulation_steps,
                "lr": cfg.learning_rate,
                "max_steps": cfg.max_steps,
                "num_epochs": cfg.num_epochs,
                "lora_r": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "lora_dropout": cfg.lora_dropout,
                "gpu": gpu_info.get("device_name", "unknown"),
                "vram_gb": gpu_info.get("vram_gb", 0.0),
            }
        )
    except Exception as e:
        logger.debug("MLflow tracking skipped: %s", e)

    start_t = time.time()
    logger.info("Starting Unsloth QLoRA training...")
    train_stats = trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    elapsed_s = round(time.time() - start_t, 1)

    peak_vram_gb = round(torch.cuda.max_memory_reserved() / (1024**3), 3) if torch.cuda.is_available() else 0.0
    train_loss = float(getattr(train_stats, "training_loss", 0.0))

    logger.info(
        "[OK] Training finished in %.1fs | Training Loss: %.4f | Peak Reserved VRAM: %.2f GB",
        elapsed_s,
        train_loss,
        peak_vram_gb,
    )

    eval_loss = None
    if val_records and len(val_records) > 0:
        logger.info("Evaluating fine-tuned model on validation split (%d samples)...", len(val_records))
        try:
            eval_metrics = trainer.evaluate()
            if "eval_loss" in eval_metrics:
                eval_loss = float(eval_metrics["eval_loss"])
                logger.info("[OK] Validation evaluation finished | Eval Loss: %.4f", eval_loss)
        except Exception as eval_err:
            logger.warning("Validation evaluation failed: %s", eval_err)

    if mlflow_mod:
        try:
            metrics_payload = {
                "training_loss": train_loss,
                "train_runtime_s": elapsed_s,
                "peak_vram_gb": peak_vram_gb,
            }
            if eval_loss is not None:
                metrics_payload["eval_loss"] = eval_loss
            mlflow_mod.log_metrics(metrics_payload)
            mlflow_mod.end_run()
        except Exception:
            pass

    if save_lora:
        lora_dir = out_dir / "lora_adapter"
        model.save_pretrained(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))
        logger.info("Saved LoRA adapter & tokenizer to %s", lora_dir)

    if save_merged_16bit:
        merged_dir = out_dir / "merged_16bit"
        logger.info("Merging and saving 16-bit weights to %s...", merged_dir)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    if save_gguf:
        gguf_dir = out_dir / "gguf"
        logger.info("Exporting GGUF (%s) for Ollama to %s...", save_gguf, gguf_dir)
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=save_gguf)

    if export_onnx:
        merged_dir = out_dir / "merged_16bit"
        if not merged_dir.exists():
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        export_script = BASE_DIR / "models" / "export_slm.py"
        if export_script.exists():
            subprocess.run(
                [sys.executable, str(export_script), "--model_id", str(merged_dir)],
                cwd=str(BASE_DIR),
                check=False,
            )

    if promote_production:
        logger.info("Evaluating MLflow runs to promote champion SLM to 'production'...")
        try:
            from models.mlflow_config import promote_best_slm_to_production

            promote_best_slm_to_production(
                experiment="football-slm-finetuning",
                selection_metric=selection_metric,
            )
        except Exception as e:
            logger.warning("SLM champion production promotion failed: %s", e)

    if test_inference:
        logger.info("Running post-training tactical analysis smoke test...")
        FastLanguageModel.for_inference(model)
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": DEFAULT_TEST_PROMPT},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=384,
                temperature=0.6,
                do_sample=True,
                use_cache=True,
            )
        decoded = tokenizer.batch_decode(outputs[:, inputs.shape[1] :], skip_special_tokens=True)[0]
        print("\n" + "=" * 70)
        print("  SAMPLE FINE-TUNED TACTICAL ANALYSIS")
        print("=" * 70)
        print(decoded.strip())
        print("=" * 70 + "\n")

    return {
        "training_loss": train_loss,
        "eval_loss": eval_loss,
        "elapsed_s": elapsed_s,
        "peak_vram_gb": peak_vram_gb,
        "output_dir": str(out_dir),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Football Tactical SLM using Unsloth & Unsloth Studio (GTX 1660 Ti 6GB tuned)"
    )
    p.add_argument(
        "--studio-config",
        type=str,
        default="models/unsloth_studio_1660ti.yaml",
        help="Path to Unsloth Studio YAML config (default: models/unsloth_studio_1660ti.yaml)",
    )
    p.add_argument("--model-name", type=str, default="", help="Override HuggingFace / Unsloth model ID")
    p.add_argument("--chat-template", type=str, default="", help="Override chat template (e.g. 'qwen-2.5', 'chatml')")
    p.add_argument("--output-dir", type=str, default="", help="Override checkpoint output directory")
    p.add_argument(
        "--max-seq-length",
        "--max-length",
        dest="max_seq_length",
        type=int,
        default=0,
        help="Override max sequence length (default: 1024 for 6GB GPU)",
    )
    p.add_argument("--batch-size", type=int, default=0, help="Override per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=0, help="Override gradient accumulation steps")
    p.add_argument("--max-steps", type=int, default=-1, help="Override max_steps (0 = use epochs)")
    p.add_argument("--epochs", type=int, default=0, help="Override num_epochs")
    p.add_argument("--lr", type=float, default=0.0, help="Override learning rate")
    p.add_argument("--max-eval-samples", type=int, default=0, help="Override validation subset cap")
    p.add_argument("--rebuild-splits", action="store_true", help="Force rebuilding train/val JSON splits from JSONL")
    p.add_argument("--no-auto-tune", action="store_true", help="Disable automatic GTX 1660 Ti 6GB VRAM tuning")
    p.add_argument(
        "--save-merged-16bit",
        "--export-merged-16bit",
        dest="save_merged_16bit",
        action="store_true",
        help="Save merged 16-bit weights after training",
    )
    p.add_argument(
        "--save-gguf",
        "--export-gguf",
        dest="save_gguf",
        type=str,
        default="",
        help="Export GGUF with specified quant (e.g. 'q4_k_m')",
    )
    p.add_argument("--export-onnx", action="store_true", help="Export merged model to ONNX via models/export_slm.py")
    p.add_argument("--test-inference", action="store_true", help="Run tactical analysis prompt test after training")
    p.add_argument(
        "--promote-production",
        action="store_true",
        help="Evaluate all SLM runs in MLflow and tag/alias the champion as 'production'",
    )
    p.add_argument(
        "--selection-metric",
        type=str,
        default="eval_loss",
        help="Metric to determine champion SLM in MLflow (default: eval_loss)",
    )
    p.add_argument("--export-studio-yaml", type=str, default="", help="Write resolved Unsloth Studio YAML to path")
    p.add_argument("--launch-studio", action="store_true", help="Launch Unsloth Studio UI server")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset splits, GPU profile, and Unsloth Studio config without running training",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.launch_studio:
        logger.info("Launching Unsloth Studio UI...")
        subprocess.run([sys.executable, "-m", "unsloth", "studio"], check=False)
        return

    yaml_path = Path(args.studio_config)
    if not yaml_path.is_absolute():
        yaml_path = BASE_DIR / yaml_path

    if yaml_path.exists():
        logger.info("Loading Unsloth Studio config from %s", yaml_path)
        cfg = FinetuneConfig.from_studio_yaml(yaml_path)
    else:
        logger.warning("Studio config %s not found; using built-in GTX 1660 Ti 6GB defaults.", yaml_path)
        cfg = FinetuneConfig()

    # Apply CLI overrides
    if args.model_name:
        cfg.model_name_or_path = args.model_name
    if args.chat_template:
        cfg.chat_template = args.chat_template
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.max_seq_length > 0:
        cfg.max_seq_length = args.max_seq_length
    if args.batch_size > 0:
        cfg.batch_size = args.batch_size
    if args.grad_accum > 0:
        cfg.gradient_accumulation_steps = args.grad_accum
    if args.max_steps >= 0:
        cfg.max_steps = args.max_steps
    if args.epochs > 0:
        cfg.num_epochs = args.epochs
    if args.lr > 0.0:
        cfg.learning_rate = args.lr
    if args.max_eval_samples > 0:
        cfg.max_eval_samples = args.max_eval_samples

    gpu_info = detect_gpu_and_tune_config(cfg, auto_tune_6gb=not args.no_auto_tune)

    if args.export_studio_yaml:
        out_yaml = Path(args.export_studio_yaml)
        if not out_yaml.is_absolute():
            out_yaml = BASE_DIR / out_yaml
        out_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(out_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg.to_studio_yaml_dict(), f, sort_keys=False)
        logger.info("Exported tuned Unsloth Studio YAML to %s", out_yaml)

    train_ds, val_ds = prepare_sharegpt_splits(cfg, rebuild_splits=args.rebuild_splits)

    eff_batch = cfg.batch_size * cfg.gradient_accumulation_steps
    steps_per_epoch = max(1, len(train_ds) // eff_batch)
    planned_steps = cfg.max_steps if (cfg.max_steps and cfg.max_steps > 0) else (steps_per_epoch * max(1, cfg.num_epochs))

    print("\n" + "=" * 74)
    print("  UNSLOTH / UNSLOTH STUDIO FINE-TUNING PLAN (GTX 1660 Ti 6GB TUNED)")
    print("=" * 74)
    print(f"  GPU Detected:     {gpu_info['device_name']} ({gpu_info['vram_gb']} GB VRAM, CC {gpu_info['compute_capability']})")
    print(f"  Base Model:       {cfg.model_name_or_path}")
    print(f"  Chat Template:    {cfg.chat_template} | 4-bit QLoRA: {cfg.load_in_4bit} | Dtype: {cfg.dtype}")
    print(f"  LoRA Config:      r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout} (Triton Fast Kernels: ON)")
    print(f"  Batching:         micro_batch={cfg.batch_size} x grad_accum={cfg.gradient_accumulation_steps} (effective={eff_batch})")
    print(f"  Sequence Length:  {cfg.max_seq_length} | Grad Checkpointing: {cfg.gradient_checkpointing} | Optim: {cfg.optim}")
    print(f"  Dataset Splits:   Train={len(train_ds)} | Val={len(val_ds)} (Eval cap={cfg.max_eval_samples})")
    print(f"  Training Steps:   {planned_steps} steps (~{steps_per_epoch} steps/epoch)")
    print(f"  Output Directory: {cfg.output_dir}")
    print("=" * 74 + "\n")

    if args.dry_run:
        logger.info("[DRY RUN] Configuration and dataset verified. Exiting without starting training.")
        return

    run_unsloth_training(
        cfg=cfg,
        gpu_info=gpu_info,
        train_records=train_ds,
        val_records=val_ds,
        save_lora=True,
        save_merged_16bit=args.save_merged_16bit,
        save_gguf=args.save_gguf,
        export_onnx=args.export_onnx,
        test_inference=args.test_inference,
        promote_production=args.promote_production,
        selection_metric=args.selection_metric,
    )


if __name__ == "__main__":
    main()

