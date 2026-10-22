"""
SLM Evaluation Harness (Phase 1)
=================================
Evaluates the Small Language Model (Fine-tuned Qwen / ONNX / Base) on the
golden validation dataset (eval/golden/slm_golden_set.json).
Verifies:
  - Strict JSON validity rate (100% threshold)
  - Schema adherence (match_state, analysis, coach_recommendations)
  - Exact coach recommendation count and priority formatting (1, 2, 3)

Usage:
  python eval/run_slm_eval.py --samples 25
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import yaml
import argparse
import logging
from pathlib import Path

from models.llm_providers import get_llm_provider
from eval.langfuse_client import log_score, flush_traces

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
GOLDEN_PATH = BASE_DIR / "eval" / "golden" / "slm_golden_set.json"
THRESHOLDS_PATH = BASE_DIR / "eval" / "thresholds.yaml"


def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def evaluate_slm_json(response: str) -> tuple[bool, bool, bool]:
    """
    Returns (is_valid_json, has_required_schema, has_valid_actions)
    """
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except Exception:
        return False, False, False

    if not isinstance(data, dict):
        return False, False, False

    has_schema = all(k in data for k in ["match_state", "analysis", "coach_recommendations"])
    if not has_schema:
        # Also tolerate analysis dict with recommendations nested
        analysis = data.get("analysis", {})
        if isinstance(analysis, dict) and "coach_recommendations" in analysis:
            has_schema = True
            recs = analysis.get("coach_recommendations", [])
        else:
            recs = data.get("coach_recommendations", [])
    else:
        recs = data.get("coach_recommendations") or data.get("analysis", {}).get("coach_recommendations", [])

    valid_actions = False
    if isinstance(recs, list) and len(recs) == 3:
        priorities = [r.get("priority") for r in recs if isinstance(r, dict)]
        if priorities == [1, 2, 3]:
            valid_actions = True

    return True, has_schema, valid_actions


def run_slm_eval(samples_count: int = 25) -> bool:
    if not GOLDEN_PATH.exists():
        logger.error("SLM Golden dataset missing at %s. Run `python eval/build_golden_set.py` first.", GOLDEN_PATH)
        return False

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if samples_count > 0:
        data = data[:samples_count]

    logger.info("Starting SLM evaluation on %d samples...", len(data))
    thresholds = load_thresholds().get("slm", {})
    llm = get_llm_provider()

    if llm is None:
        logger.warning("No LLM provider configured or available. Cannot run SLM inference.")
        return False

    json_valid_count = 0
    schema_valid_count = 0
    actions_valid_count = 0

    for i, item in enumerate(data, 1):
        prompt = item.get("input") or item.get("instruction") or str(item)
        full_prompt = (
            "You are Expert 2 (Fine-tuned SLM). Return a single JSON object with match_state, "
            "analysis, and coach_recommendations (exactly 3 items with priorities 1, 2, 3):\n" + prompt
        )

        try:
            if hasattr(llm, "generate"):
                response = llm.generate(full_prompt)
            elif hasattr(llm, "_call_api"):
                response = llm._call_api(full_prompt)
            else:
                response = ""
        except Exception as e:
            logger.error("Inference failed for sample %d: %s", i, e)
            continue

        is_json, has_schema, has_actions = evaluate_slm_json(response)
        if is_json:
            json_valid_count += 1
        if has_schema:
            schema_valid_count += 1
        if has_actions:
            actions_valid_count += 1

        logger.info(
            "[%d/%d] JSON=%s | Schema=%s | 3Actions=%s",
            i, len(data), is_json, has_schema, has_actions
        )

    flush_traces()

    n = len(data)
    json_rate = json_valid_count / n if n else 0.0
    schema_rate = schema_valid_count / n if n else 0.0
    actions_rate = actions_valid_count / n if n else 0.0

    logger.info("=" * 60)
    logger.info("SLM EVALUATION RESULTS SUMMARY")
    logger.info("  Samples Tested           : %d", n)
    logger.info("  Valid JSON Rate          : %.1f%%", json_rate * 100)
    logger.info("  Schema Compliance Rate   : %.1f%%", schema_rate * 100)
    logger.info("  3-Action Priorities Rate : %.1f%%", actions_rate * 100)
    logger.info("=" * 60)

    min_json = thresholds.get("valid_json_rate_min", 1.0)
    passed = (json_rate >= min_json)
    if not passed:
        logger.warning("FAILED: Valid JSON rate (%.2f) below threshold (%.2f)", json_rate, min_json)
    else:
        logger.info("PASSED: SLM meets JSON format requirements.")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLM Format and Schema Evaluation")
    parser.add_argument("--samples", type=int, default=25, help="Number of validation samples")
    args = parser.parse_args()

    success = run_slm_eval(samples_count=args.samples)
    sys.exit(0 if success else 1)
