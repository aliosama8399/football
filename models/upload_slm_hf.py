"""
Champion SLM Hugging Face Upload Utility
=========================================
Publishes the MLflow `production`-tagged tactical Small Language Model (SLM)
to Hugging Face Hub (default: `aliosama8399/football-analysisN`).

Conditional Logic:
  - ONLY uploads if the model is crowned champion (`production=True`) AND
    is newly promoted or has not yet been uploaded for this champion run_id.
  - Can be forced via `--force` or DAG conf `{"force_hf_upload": true}`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("upload_slm_hf")

DEFAULT_REPO_ID = "aliosama8399/football-analysis-PROD"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "finetuned_qwen_unsloth" / "merged_16bit"
PRODUCTION_META_PATH = BASE_DIR / "models" / "saved" / "production_slm_meta.json"
MAX_ACCEPTABLE_EVAL_LOSS = 2.5
RELEASE_AUDIT_LOG_PATH = BASE_DIR / "models" / "saved" / "slm_release_audit.jsonl"


def _compute_file_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 checksum streaming 8 MB chunks."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_retry(fn: Callable[[], T], max_attempts: int = 3, base_delay: float = 1.0) -> T:
    """Execute a transient network operation with exponential backoff."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("Attempt %s/%s failed (%s); retrying in %.1fs...", attempt, max_attempts, exc, delay)
            time.sleep(delay)
    raise last_exc  # pragma: no cover


def build_release_manifest(repo_id: str, meta: Dict[str, Any], weights_file: Path) -> Dict[str, Any]:
    """Construct an immutable release manifest with SHA-256 artifact checksums."""
    return {
        "schema_version": "1.0",
        "repo_id": repo_id,
        "run_id": meta.get("run_id", "unknown"),
        "previous_run_id": meta.get("previous_run_id"),
        "model_name": meta.get("model_name", "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"),
        "metrics": meta.get("metrics", {}),
        "weights_filename": weights_file.name,
        "weights_size_bytes": weights_file.stat().st_size,
        "weights_sha256": _compute_file_sha256(weights_file),
        "promoted_at": meta.get("promoted_at"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _append_release_audit_log(entry: Dict[str, Any]) -> None:
    """Append an immutable JSONL release governance audit record."""
    try:
        RELEASE_AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RELEASE_AUDIT_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Could not write release audit log %s: %s", RELEASE_AUDIT_LOG_PATH, exc)


def _load_env_token() -> Optional[str]:
    """Retrieve HF_TOKEN from environment or `.env` file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()

    env_path = BASE_DIR / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        except Exception as e:
            logger.debug("Failed reading .env for HF_TOKEN: %s", e)
    return None


def generate_model_card(
    repo_id: str,
    meta: Dict[str, Any],
    base_model: str = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
) -> str:
    """Generate a GitHub Flavored Markdown Model Card (README.md) with MLOps metadata."""
    run_id = meta.get("run_id", "unknown")
    metrics = meta.get("metrics", {})
    eval_loss = metrics.get("eval_loss", "N/A")
    train_loss = metrics.get("training_loss", "N/A")
    promoted_at = meta.get("promoted_at", datetime.now(timezone.utc).isoformat())
    model_name = meta.get("model_name", base_model)

    return f"""---
base_model: {base_model}
tags:
- text-generation-inference
- transformers
- unsloth
- qwen3
- football-analysis
- tactical-reasoning
- soccer-analytics
- mlops-production
license: apache-2.0
language:
- en
pipeline_tag: text-generation
---

# ⚽ Football Tactical Reasoning SLM (Production Champion)

Fine-tuned tactical domain expert model automatically trained, evaluated, and promoted to **Production Champion** status via the automated MLOps pipeline.

## 🏆 Champion Production Metadata
- **Hugging Face Repository:** `{repo_id}`
- **MLflow Model Registry:** `football-slm-production@production`
- **MLflow Champion Run ID:** `{run_id}`
- **Base Architecture:** `{model_name}`
- **Validation Loss (`eval_loss`):** `{eval_loss}`
- **Training Loss (`training_loss`):** `{train_loss}`
- **Promoted Timestamp:** `{promoted_at}`
- **Fine-Tuning Framework:** [Unsloth AI](https://github.com/unslothai/unsloth) (FastLanguageModel)

## 📖 Model Summary
This model is specialized in structured football match analysis, temporal graph neural network explanation interpretation, tactical pressing setups, and matchup scenario predictions.

## 🚀 Quickstart Inference (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "Analyze Arsenal's high-press defensive structure against Manchester City's build-up phase."
messages = [
    {{"role": "system", "content": "You are a professional tactical football performance analyst."}},
    {{"role": "user", "content": prompt}},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```
"""


def upload_production_slm_to_hf(
    repo_id: str = DEFAULT_REPO_ID,
    model_dir: Optional[Path | str] = None,
    force: bool = False,
    dry_run: bool = False,
    commit_message: Optional[str] = None,
    max_eval_loss: float = MAX_ACCEPTABLE_EVAL_LOSS,
) -> Dict[str, Any]:
    """Upload the production champion SLM model to Hugging Face Hub.

    Args:
        repo_id: Hugging Face repo ID (e.g. 'aliosama8399/football-analysis-PROD').
        model_dir: Directory containing merged 16-bit weights and tokenizer files.
        force: If True, bypass conditional promotion and SLA quality gates.
        dry_run: If True, validate condition and files without executing the network upload.
        commit_message: Optional custom Git commit message for Hugging Face Hub.
        max_eval_loss: Maximum acceptable validation loss allowed by production SLA.

    Returns:
        Dict with execution summary (status, repo_id, url, run_id, reason).
    """
    model_path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    logger.info("Evaluating champion SLM status for Hugging Face upload to: %s", repo_id)

    # 1. Inspect production metadata
    if not PRODUCTION_META_PATH.exists():
        msg = f"Cannot upload: Production metadata not found at {PRODUCTION_META_PATH}. Train and promote SLM first."
        logger.warning(msg)
        return {"status": "skipped", "reason": "missing_production_meta", "message": msg}

    try:
        meta = json.loads(PRODUCTION_META_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        msg = f"Failed to parse production metadata {PRODUCTION_META_PATH}: {e}"
        logger.error(msg)
        return {"status": "error", "reason": "unreadable_meta", "message": msg}

    if not meta.get("production") and meta.get("tag") != "production":
        msg = "The model in metadata is NOT tagged as 'production'. Upload aborted."
        logger.warning(msg)
        return {"status": "skipped", "reason": "not_production", "message": msg}

    run_id = meta.get("run_id", "unknown")
    metrics = meta.get("metrics", {})
    eval_loss = metrics.get("eval_loss", "N/A")
    uploaded_run_id = meta.get("uploaded_to_hf_run_id")
    newly_promoted = meta.get("newly_promoted", False)

    # 2. Verify champion conditional upload & SLA quality gates
    if not force:
        if uploaded_run_id == run_id:
            msg = (
                f"Production model (Run ID: {run_id}, eval_loss: {eval_loss}) was ALREADY uploaded to "
                f"Hugging Face ({repo_id}) at {meta.get('hf_uploaded_at')}. Skipping redundant upload."
            )
            logger.info(msg)
            return {
                "status": "skipped",
                "reason": "already_uploaded",
                "run_id": run_id,
                "repo_id": repo_id,
                "message": msg,
            }

        if not newly_promoted and uploaded_run_id is not None:
            msg = (
                f"Skipping upload: Latest run is NOT a newly crowned champion (current champion is {run_id} "
                f"with eval_loss={eval_loss}). Upload only triggers when a new best model is promoted."
            )
            logger.info(msg)
            return {
                "status": "skipped",
                "reason": "not_newly_promoted",
                "run_id": run_id,
                "repo_id": repo_id,
                "message": msg,
            }

        raw_eval = metrics.get("eval_loss")
        sla_valid = False
        try:
            numeric_eval = float(raw_eval)
            sla_valid = math.isfinite(numeric_eval) and 0.0 < numeric_eval <= max_eval_loss
        except (TypeError, ValueError):
            sla_valid = False

        if not sla_valid:
            msg = (
                f"SLA Quality Gate failed: eval_loss={eval_loss} exceeds max_eval_loss={max_eval_loss} "
                f"or is invalid for Run ID {run_id}. Blocking production release."
            )
            logger.warning(msg)
            return {
                "status": "skipped",
                "reason": "sla_quality_gate_failed",
                "eval_loss": eval_loss,
                "max_eval_loss": max_eval_loss,
                "run_id": run_id,
                "repo_id": repo_id,
                "message": msg,
            }

    # 3. Verify model directory, weights, config JSON & SHA-256 artifact integrity
    if not model_path.exists():
        msg = f"Model directory {model_path} does not exist. Ensure merged_16bit weights are exported."
        logger.error(msg)
        return {"status": "error", "reason": "missing_weights_dir", "message": msg}

    safetensors_file = model_path / "model.safetensors"
    pytorch_bin_file = model_path / "pytorch_model.bin"
    active_weights_file = safetensors_file if safetensors_file.exists() else pytorch_bin_file
    config_file = model_path / "config.json"

    if not active_weights_file.exists():
        msg = f"No model weights (model.safetensors or pytorch_model.bin) found in {model_path}."
        logger.error(msg)
        return {"status": "error", "reason": "missing_weights_file", "message": msg}

    if not config_file.exists():
        msg = f"config.json missing in {model_path}."
        logger.error(msg)
        return {"status": "error", "reason": "missing_config", "message": msg}

    try:
        if active_weights_file.stat().st_size <= 0:
            raise ValueError(f"Weights file {active_weights_file} is empty (0 bytes).")
        json.loads(config_file.read_text(encoding="utf-8"))
    except Exception as exc:
        msg = f"Artifact integrity check failed in {model_path}: {exc}"
        logger.error(msg)
        return {"status": "error", "reason": "corrupt_artifact", "message": msg}

    # 4. Prepare / update Model Card (README.md) & release_manifest.json
    readme_path = model_path / "README.md"
    try:
        readme_content = generate_model_card(repo_id=repo_id, meta=meta)
        readme_path.write_text(readme_content, encoding="utf-8")
        logger.info("Updated model card README.md with production champion metrics.")
    except Exception as e:
        logger.warning("Could not write README.md: %s; proceeding with existing card.", e)

    manifest = build_release_manifest(repo_id=repo_id, meta=meta, weights_file=active_weights_file)
    manifest_path = model_path / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 5. Check Token
    token = _load_env_token()
    if not token:
        msg = "HF_TOKEN not found in environment or .env file. Please authenticate with Hugging Face."
        logger.error(msg)
        return {"status": "error", "reason": "missing_hf_token", "message": msg}

    if dry_run:
        logger.info("[Dry Run] Validation successful. Would upload %s to %s (Run ID: %s)", model_path, repo_id, run_id)
        return {
            "status": "dry_run_success",
            "repo_id": repo_id,
            "run_id": run_id,
            "model_dir": str(model_path),
            "eval_loss": eval_loss,
            "manifest": manifest,
        }

    # 6. Upload via huggingface_hub with exponential backoff retries & immutable tag
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

        user_info = _with_retry(api.whoami)
        logger.info("Authenticated as Hugging Face user: %s", user_info.get("name"))

        logger.info("Ensuring Hugging Face repository exists: %s", repo_id)
        _with_retry(lambda: api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False))

        commit_msg = (
            commit_message
            or f"Promote champion SLM model to production [Run: {run_id[:8]} | Eval Loss: {eval_loss}]"
        )
        logger.info("Uploading folder %s to Hugging Face repo %s ...", model_path, repo_id)
        commit_info = _with_retry(
            lambda: api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_msg,
                ignore_patterns=["*.tmp", "*.pyc", "__pycache__/*", ".git/*"],
            )
        )

        tag_name = f"prod-{str(run_id)[:8]}"
        try:
            api.create_tag(
                repo_id=repo_id,
                tag=tag_name,
                repo_type="model",
                tag_message=commit_msg,
                exist_ok=True,
            )
            logger.info("Tagged Hugging Face release revision: %s", tag_name)
        except Exception as tag_exc:
            logger.warning("Could not create HF release tag '%s': %s", tag_name, tag_exc)

        repo_url = f"https://huggingface.co/{repo_id}"
        uploaded_at = datetime.now(timezone.utc).isoformat()
        commit_oid = str(getattr(commit_info, "oid", commit_info))

        # 7. Update production metadata & append immutable audit trail
        meta["uploaded_to_hf_run_id"] = run_id
        meta["hf_repo_id"] = repo_id
        meta["hf_repo_url"] = repo_url
        meta["hf_uploaded_at"] = uploaded_at
        meta["hf_commit_oid"] = commit_oid
        meta["hf_release_tag"] = tag_name
        meta["weights_sha256"] = manifest["weights_sha256"]
        PRODUCTION_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        _append_release_audit_log(
            {
                "event": "slm_production_release",
                "repo_id": repo_id,
                "run_id": run_id,
                "previous_run_id": meta.get("previous_run_id"),
                "eval_loss": eval_loss,
                "weights_sha256": manifest["weights_sha256"],
                "hf_release_tag": tag_name,
                "hf_commit_oid": commit_oid,
                "uploaded_at": uploaded_at,
            }
        )

        print("\n" + "=" * 70)
        print("  [Hugging Face Hub Upload] Champion Model Successfully Published!")
        print(f"  Repo:       {repo_url}")
        print(f"  Run ID:     {run_id} (tag: {tag_name})")
        print(f"  Eval Loss:  {eval_loss}")
        print(f"  SHA-256:    {manifest['weights_sha256'][:16]}...")
        print(f"  Commit:     {commit_msg}")
        print(f"  Timestamp:  {uploaded_at}")
        print("=" * 70 + "\n")

        return {
            "status": "success",
            "repo_id": repo_id,
            "url": repo_url,
            "run_id": run_id,
            "release_tag": tag_name,
            "weights_sha256": manifest["weights_sha256"],
            "uploaded_at": uploaded_at,
            "commit": commit_oid,
        }

    except Exception as e:
        msg = f"Hugging Face upload failed: {e}"
        logger.error(msg, exc_info=True)
        return {"status": "error", "reason": "upload_exception", "message": msg}


def main():
    parser = argparse.ArgumentParser(
        description="Upload champion production SLM model to Hugging Face Hub (aliosama8399/football-analysisN)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.environ.get("HF_SLM_REPO_ID", DEFAULT_REPO_ID),
        help=f"Hugging Face repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help=f"Path to merged model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force upload even if model was previously uploaded or not newly promoted",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate metadata, model files, and credentials without uploading",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message for Hugging Face Hub",
    )
    args = parser.parse_args()

    res = upload_production_slm_to_hf(
        repo_id=args.repo_id,
        model_dir=args.model_dir,
        force=args.force,
        dry_run=args.dry_run,
        commit_message=args.commit_message,
    )

    if res.get("status") in ("success", "dry_run_success", "skipped"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

