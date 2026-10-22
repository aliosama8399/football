"""
RAG Evaluation Harness (Phase 1)
=================================
Runs automated evaluations against the golden evaluation set (eval/golden/rag_golden_set.json).
Computes exact numeric accuracy, lexical/semantic grounding, latency, and RAGAS metrics.
Logs traces and scores to Langfuse and checks against eval/thresholds.yaml.

Usage:
  python eval/run_rag_eval.py --samples 20
  python eval/run_rag_eval.py --samples 50 --judge api
"""

import sys, os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import warnings
import re
import time
import json
import yaml
import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rag.rag_orchestrator import FootballRAGSystem
from eval.langfuse_client import log_score, flush_traces, trace_step, current_trace_id, langfuse_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_ENV_PATH, override=False)
except Exception:
    pass

BASE_DIR = Path(__file__).parent.parent
GOLDEN_PATH = Path(os.getenv("GOLDEN_SET_PATH", BASE_DIR / "eval" / "golden" / "rag_golden_set.json"))
THRESHOLDS_PATH = Path(os.getenv("THRESHOLDS_PATH", BASE_DIR / "eval" / "thresholds.yaml"))
CONFIG_PATH = Path(os.getenv("LLM_CONFIG_PATH", BASE_DIR / "models" / "llm_config.yaml"))


def load_llm_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def resolve_judge_model(provider: str, explicit_model: str | None = None) -> str:
    """
    Dynamically loads the judge model with priority:
      1. Explicit CLI argument (--judge-model)
      2. Environment variable (JUDGE_MODEL or <PROVIDER>_MODEL, e.g. GEMINI_MODEL)
      3. models/llm_config.yaml (providers.<provider>.default_model)
      4. Fallback defaults
    """
    if explicit_model:
        return explicit_model.strip()

    prov_key = "gemini" if provider in ("google", "gemini") else provider

    # 1. Environment variable priority
    env_model = (
        os.getenv("JUDGE_MODEL")
        or os.getenv(f"{prov_key.upper()}_MODEL")
        or os.getenv(f"{provider.upper()}_MODEL")
    )
    if env_model and env_model.strip():
        return env_model.strip()

    # 2. Config file
    cfg = load_llm_config()
    model = cfg.get("providers", {}).get(prov_key, {}).get("default_model")
    if model and str(model).strip():
        return str(model).strip()

    # 3. Fallback
    if prov_key == "gemini":
        return "gemini-3.8-flash"
    elif prov_key == "openai":
        return "gpt-4o-mini"
    return "default"


def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def extract_numbers(text: str) -> set[int]:
    matches = re.findall(r"\b\d+\b", text)
    return {int(m) for m in matches}


def evaluate_numeric_grounding(response: str, expected_numbers: dict) -> float:
    if not expected_numbers:
        return 1.0
    text_numbers = extract_numbers(response)
    expected_vals = set(expected_numbers.values())
    if not expected_vals:
        return 1.0
    matches = expected_vals.intersection(text_numbers)
    return len(matches) / len(expected_vals)


def evaluate_lexical_grounding(response: str, ground_truth: str) -> float:
    resp_words = set(re.findall(r"\w+", response.lower()))
    gt_words = set(re.findall(r"\w+", ground_truth.lower()))
    if not gt_words:
        return 1.0
    overlap = resp_words.intersection(gt_words)
    return min(1.0, len(overlap) / (len(gt_words) * 0.5))


def run_ragas_evaluation(
    eval_records: list[dict],
    judge_provider: str = "google",
    judge_model: str | None = None,
) -> dict:
    """
    Run RAGAS evaluation on collected samples using official LLM judge.
    Evaluates faithfulness, answer_relevancy, context_precision, and context_recall.
    Uses modern Ragas 0.4 native providers (llm_factory, GoogleEmbeddings/OpenAIEmbeddings).
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
    except ImportError:
        logger.warning(
            "\n" + "="*70 + "\n"
            "[!] RAGAS dependencies not installed in this environment.\n"
            "    To compute official RAGAS metrics, install them via:\n"
            "        pip install ragas datasets google-genai openai\n"
            + "="*70
        )
        return {}

    judge_model = resolve_judge_model(judge_provider, judge_model)
    evaluator_llm = None
    evaluator_embeddings = None

    if judge_provider in ("google", "gemini"):
        # Prevent double-key warning if both GEMINI_API_KEY and GOOGLE_API_KEY are set
        google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_API_KEY"):
            os.environ.pop("GOOGLE_API_KEY", None)

        if not google_api_key:
            logger.warning("GEMINI_API_KEY / GOOGLE_API_KEY not found in environment; check your .env file.")
        else:
            try:
                from google import genai
                from ragas.llms import llm_factory
                from ragas.embeddings import GoogleEmbeddings

                logger.info("Configuring Google API judge: model=%s (from llm_config.yaml)", judge_model)
                client = genai.Client(api_key=google_api_key)
                try:
                    evaluator_llm = llm_factory(judge_model, provider="google", client=client)
                except Exception as e:
                    logger.warning("Judge model %s unavailable (%s), falling back to gemini-2.5-flash", judge_model, e)
                    judge_model = "gemini-2.5-flash"
                    evaluator_llm = llm_factory(judge_model, provider="google", client=client)

                embed_model_name = "models/gemini-embedding-001"
                try:
                    evaluator_embeddings = GoogleEmbeddings(client=client, model=embed_model_name)
                except Exception:
                    embed_model_name = "models/text-embedding-004"
                    evaluator_embeddings = GoogleEmbeddings(client=client, model=embed_model_name)

                # Add LangChain-compatible aliases for Ragas metric evaluators
                if not hasattr(evaluator_embeddings, "embed_query") and hasattr(evaluator_embeddings, "embed_text"):
                    evaluator_embeddings.embed_query = evaluator_embeddings.embed_text
                if not hasattr(evaluator_embeddings, "embed_documents") and hasattr(evaluator_embeddings, "embed_texts"):
                    evaluator_embeddings.embed_documents = evaluator_embeddings.embed_texts
                if not hasattr(evaluator_embeddings, "aembed_query") and hasattr(evaluator_embeddings, "aembed_text"):
                    evaluator_embeddings.aembed_query = evaluator_embeddings.aembed_text
                if not hasattr(evaluator_embeddings, "aembed_documents") and hasattr(evaluator_embeddings, "aembed_texts"):
                    evaluator_embeddings.aembed_documents = evaluator_embeddings.aembed_texts

                logger.info("Google Gemini (%s) judge & %s initialized successfully.", judge_model, embed_model_name)
            except Exception as e:
                logger.warning("Failed to initialize Google Gemini judge: %s. Falling back to default evaluation.", e)

    elif judge_provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment; check your .env file.")
        else:
            try:
                from openai import OpenAI
                from ragas.llms import llm_factory
                from ragas.embeddings import OpenAIEmbeddings

                logger.info("Configuring OpenAI API judge: model=%s", judge_model)
                client = OpenAI(api_key=openai_api_key)
                evaluator_llm = llm_factory(judge_model, provider="openai", client=client)
                evaluator_embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")
                logger.info("OpenAI (%s) judge & text-embedding-3-small initialized successfully.", judge_model)
            except Exception as e:
                logger.warning("Failed to initialize OpenAI judge: %s.", e)

    questions = [r["question"] for r in eval_records]
    answers = [r["answer"] for r in eval_records]
    contexts = [r["contexts"] if r["contexts"] else ["No context retrieved."] for r in eval_records]
    ground_truths = [r["ground_truth"] for r in eval_records]

    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(dataset_dict)

    logger.info("Evaluating RAGAS metrics with %s judge...", judge_provider.upper())
    try:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        eval_kwargs = {"dataset": dataset, "metrics": metrics}
        if evaluator_llm is not None:
            eval_kwargs["llm"] = evaluator_llm
        if evaluator_embeddings is not None:
            eval_kwargs["embeddings"] = evaluator_embeddings

        results = evaluate(**eval_kwargs)

        scores = {}
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            val = None
            if hasattr(results, "_repr_dict") and m in results._repr_dict:
                val = results._repr_dict[m]
            elif hasattr(results, "_scores_dict") and m in results._scores_dict:
                vals = [x for x in results._scores_dict[m] if x is not None and not (isinstance(x, float) and math.isnan(x))]
                val = sum(vals) / len(vals) if vals else None
            elif hasattr(results, "get"):
                val = results.get(m)

            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                val_flt = float(val)
                scores[m] = val_flt
                log_score(f"ragas_{m}", val_flt)

        # Log item-level scores to Langfuse traces when available
        try:
            df = results.to_pandas()
            for idx, r in enumerate(eval_records):
                tid = r.get("trace_id")
                if tid and idx < len(df):
                    row = df.iloc[idx]
                    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                        if m in row:
                            val_item = row[m]
                            if val_item is not None and not str(val_item).lower() == "nan":
                                log_score(f"ragas_{m}", float(val_item), trace_id=tid)
        except Exception as e:
            logger.debug("Item-level RAGAS score logging note: %s", e)

        return scores
    except Exception as e:
        logger.error("RAGAS execution error: %s", e)
        return {}


def run_eval(
    samples_count: int = 20,
    judge_mode: str = "google",
    judge_model: str | None = None,
    enable_ragas: bool = False,
) -> bool:
    judge_model = resolve_judge_model(judge_mode, judge_model)
    if not GOLDEN_PATH.exists():
        logger.error("Golden dataset not found at %s. Run `python eval/build_golden_set.py` first.", GOLDEN_PATH)
        return False

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if samples_count > 0:
        data = data[:samples_count]
    if not data:
        logger.error("Golden set is empty. Rebuild it: python eval/build_golden_set.py --count 200")
        return False

    logger.info("Langfuse: %s", langfuse_status())
    logger.info("Starting RAG evaluation on %d test queries (RAGAS=%s, Judge=%s / %s)...", len(data), enable_ragas, judge_mode.upper(), judge_model)
    thresholds = load_thresholds().get("rag", {})
    rag_system = FootballRAGSystem()

    latencies = []
    numeric_scores = []
    grounding_scores = []
    ragas_records = []

    for i, item in enumerate(data, 1):
        query = item["query"]
        as_of = item.get("as_of_date")
        expected_nums = item.get("expected_numbers", {})
        ground_truth = item.get("ground_truth", "")

        t0 = time.time()
        try:
            with trace_step("rag_eval_item", input={"query": query, "as_of": as_of},
                            metadata={"id": item.get("id"), "type": item.get("type")}) as span:
                res_bundle = rag_system.query_with_context(query, as_of=as_of)
                response = res_bundle["response"]
                retrieved_contexts = res_bundle.get("retrieved_contexts", [])
                span.update(output=response)
                trace_id = current_trace_id()
            duration = time.time() - t0
        except Exception as e:
            logger.error("Query failed for item %d (%s): %s", i, item.get("id"), e)
            continue

        latencies.append(duration)

        num_score = evaluate_numeric_grounding(response, expected_nums)
        numeric_scores.append(num_score)

        grd_score = evaluate_lexical_grounding(response, ground_truth)
        grounding_scores.append(grd_score)

        log_score("numeric_grounding", num_score, trace_id=trace_id)
        log_score("grounding", grd_score, trace_id=trace_id)
        log_score("latency_s", duration, trace_id=trace_id)

        ragas_records.append({
            "question": query,
            "answer": response,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truth,
            "trace_id": trace_id,
        })

        logger.info(
            "[%d/%d] ID=%s | Latency=%.2fs | NumericScore=%.2f | Grounding=%.2f",
            i, len(data), item.get("id"), duration, num_score, grd_score
        )

        # Stream real-time metrics over HTTP to live API so Grafana reflects each step live
        cur_lat = sum(latencies) / len(latencies)
        cur_num = sum(numeric_scores) / len(numeric_scores)
        cur_grd = sum(grounding_scores) / len(grounding_scores)
        try:
            from monitoring.metrics_registry import send_live_metric
            send_live_metric("rag", {
                "avg_latency": cur_lat,
                "avg_numeric": cur_num,
                "avg_grounding": cur_grd,
            })
        except Exception:
            pass

    # Aggregate heuristic metrics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_numeric = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
    avg_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0

    ragas_scores = {}
    if enable_ragas and ragas_records:
        ragas_scores = run_ragas_evaluation(
            ragas_records,
            judge_provider=judge_mode,
            judge_model=judge_model,
        )

    flush_traces()
    rag_system.close()

    # Record metrics to Prometheus registry
    try:
        from monitoring.metrics_registry import record_rag_evaluation_metrics
        record_rag_evaluation_metrics(
            avg_latency=avg_latency,
            avg_numeric=avg_numeric,
            avg_grounding=avg_grounding,
            ragas_scores=ragas_scores,
        )
        logger.info("Prometheus metrics updated successfully (visible at /metrics).")
    except Exception as e:
        logger.debug("Failed updating Prometheus metrics: %s", e)

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("  Queries Evaluated        : %d", len(latencies))
    logger.info("  Average Latency          : %.2fs", avg_latency)
    logger.info("  Numeric Fact Grounding   : %.2f", avg_numeric)
    logger.info("  Grounding / Relevancy    : %.2f", avg_grounding)
    if ragas_scores:
        logger.info("  --- RAGAS Metrics (Judge: %s / %s) ---", judge_mode.upper(), judge_model)
        for k, v in ragas_scores.items():
            logger.info("  RAGAS %-18s: %.2f", k, v)
    logger.info("=" * 60)

    # Check against thresholds
    num_min = thresholds.get("numeric_fact_grounding_min", 0.85)
    faith_min = thresholds.get("faithfulness_min", 0.75)
    relevancy_min = thresholds.get("answer_relevancy_min", 0.80)
    passed = True

    if avg_numeric < num_min:
        logger.warning("FAILED: Numeric Grounding (%.2f) below threshold (%.2f)", avg_numeric, num_min)
        passed = False
    else:
        logger.info("PASSED: Numeric Grounding meets threshold.")

    if ragas_scores:
        faith = ragas_scores.get("faithfulness", 1.0)
        relevancy = ragas_scores.get("answer_relevancy", 1.0)
        if faith < faith_min:
            logger.warning("FAILED: RAGAS Faithfulness (%.2f) below threshold (%.2f)", faith, faith_min)
            passed = False
        if relevancy < relevancy_min:
            logger.warning("FAILED: RAGAS Answer Relevancy (%.2f) below threshold (%.2f)", relevancy, relevancy_min)
            passed = False

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Suite with LLM Judge (Google Gemini / OpenAI)")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--judge", type=str, default="google", choices=["google", "gemini", "openai", "heuristic"], help="Judge provider (defaults to google)")
    parser.add_argument("--judge-model", type=str, default=None, help="Model name for judge LLM (defaults to provider setting in models/llm_config.yaml)")
    parser.add_argument("--ragas", action="store_true", help="Compute RAGAS metrics (faithfulness, answer_relevancy, etc.)")
    args = parser.parse_args()

    enable_ragas = args.ragas or (args.judge in ("google", "gemini", "openai"))
    success = run_eval(
        samples_count=args.samples,
        judge_mode=args.judge,
        judge_model=args.judge_model,
        enable_ragas=enable_ragas,
    )
    sys.exit(0 if success else 1)
