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
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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
) -> Dict[str, Any]:
    """Upload the production champion SLM model to Hugging Face Hub.

    Args:
        repo_id: Hugging Face repo ID (e.g. 'aliosama8399/football-analysisN').
        model_dir: Directory containing merged 16-bit weights and tokenizer files.
        force: If True, bypass the 'newly_promoted' or 'already_uploaded' checks.
        dry_run: If True, validate condition and files without executing the network upload.
        commit_message: Optional custom Git commit message for Hugging Face Hub.

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

    # 2. Verify champion conditional upload criteria
    if not force:
        # Check if this exact run was already uploaded
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

        # Check if the model is newly crowned champion
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

    # 3. Verify model directory and weights exist
    if not model_path.exists():
        msg = f"Model directory {model_path} does not exist. Ensure merged_16bit weights are exported."
        logger.error(msg)
        return {"status": "error", "reason": "missing_weights_dir", "message": msg}

    weights_file = model_path / "model.safetensors"
    config_file = model_path / "config.json"
    if not weights_file.exists() and not (model_path / "pytorch_model.bin").exists():
        msg = f"No model weights (model.safetensors or pytorch_model.bin) found in {model_path}."
        logger.error(msg)
        return {"status": "error", "reason": "missing_weights_file", "message": msg}

    if not config_file.exists():
        msg = f"config.json missing in {model_path}."
        logger.error(msg)
        return {"status": "error", "reason": "missing_config", "message": msg}

    # 4. Prepare / update Model Card (README.md)
    readme_path = model_path / "README.md"
    try:
        readme_content = generate_model_card(repo_id=repo_id, meta=meta)
        readme_path.write_text(readme_content, encoding="utf-8")
        logger.info("Updated model card README.md with production champion metrics.")
    except Exception as e:
        logger.warning("Could not write README.md: %e; proceeding with existing card.", e)

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
        }

    # 6. Upload via huggingface_hub
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

        # Confirm identity
        user_info = api.whoami()
        logger.info("Authenticated as Hugging Face user: %s", user_info.get("name"))

        # Create repo if not already existing
        logger.info("Ensuring Hugging Face repository exists: %s", repo_id)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

        commit_msg = (
            commit_message
            or f"Promote champion SLM model to production [Run: {run_id[:8]} | Eval Loss: {eval_loss}]"
        )
        logger.info("Uploading folder %s to Hugging Face repo %s ...", model_path, repo_id)
        commit_info = api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg,
            ignore_patterns=["*.tmp", "*.pyc", "__pycache__/*", ".git/*"],
        )

        repo_url = f"https://huggingface.co/{repo_id}"
        uploaded_at = datetime.now(timezone.utc).isoformat()

        # 7. Update production metadata with upload receipt
        meta["uploaded_to_hf_run_id"] = run_id
        meta["hf_repo_id"] = repo_id
        meta["hf_repo_url"] = repo_url
        meta["hf_uploaded_at"] = uploaded_at
        meta["hf_commit_oid"] = str(getattr(commit_info, "oid", commit_info))
        PRODUCTION_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("\n" + "=" * 70)
        print("  [Hugging Face Hub Upload] Champion Model Successfully Published!")
        print(f"  Repo:       {repo_url}")
        print(f"  Run ID:     {run_id}")
        print(f"  Eval Loss:  {eval_loss}")
        print(f"  Commit:     {commit_msg}")
        print(f"  Timestamp:  {uploaded_at}")
        print("=" * 70 + "\n")

        return {
            "status": "success",
            "repo_id": repo_id,
            "url": repo_url,
            "run_id": run_id,
            "uploaded_at": uploaded_at,
            "commit": str(getattr(commit_info, "oid", commit_info)),
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

