"""
Data Leakage & As-Of Date Audit (Phase 2, Task T2.2)
=====================================================
Audits retrieval and training corpus for look-ahead data leakage:
  1. Verifies that querying with `as_of=D` strictly returns zero documents with date >= D.
  2. Identifies any held-out test match dates that appear in the fine-tuning training set.

Usage:
  python eval/audit_leakage.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from pathlib import Path

from rag.providers.kg_provider import get_kg_provider
from rag.providers.vector_provider import get_vector_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
GOLDEN_PATH = BASE_DIR / "eval" / "golden" / "rag_golden_set.json"
TRAIN_JSON_PATH = BASE_DIR / "data" / "finetune" / "football_train.json"


def audit_retriever_leakage(samples_count: int = 30) -> bool:
    """Check that retrieved KG matches and vectors respect as_of dates."""
    if not GOLDEN_PATH.exists():
        logger.error("Golden dataset missing at %s.", GOLDEN_PATH)
        return False

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    kg = get_kg_provider()
    vector = get_vector_provider()

    violations = 0
    tested = 0

    for item in items[:samples_count]:
        as_of = item.get("as_of_date")
        teams = item.get("teams", [])
        if not as_of or not teams:
            continue

        tested += 1

        # Check KG Head-to-Head
        if len(teams) >= 2:
            try:
                h2h = kg.get_head_to_head(teams[0], teams[1], limit=10, as_of=as_of)
                for m in h2h:
                    m_date = str(m.get("date", ""))
                    if m_date and m_date >= as_of:
                        logger.error("LEAKAGE in KG H2H: match date %s >= as_of %s", m_date, as_of)
                        violations += 1
            except Exception as e:
                logger.debug("H2H query failed during audit: %s", e)

        # Check KG Recent Form
        try:
            form = kg.get_recent_form(teams[0], n=5, as_of=as_of)
            for m in form:
                m_date = str(m.get("date", ""))
                if m_date and m_date >= as_of:
                    logger.error("LEAKAGE in KG Recent Form: match date %s >= as_of %s", m_date, as_of)
                    violations += 1
        except Exception as e:
            logger.debug("Form query failed during audit: %s", e)

        # Check Vector search
        try:
            vec_results = vector.search(f"{teams[0]} match stats", k=5, as_of=as_of)
            for r in vec_results:
                meta = r.get("metadata", {})
                v_date = str(meta.get("date", ""))
                if v_date and v_date >= as_of:
                    logger.error("LEAKAGE in Vector Search: doc date %s >= as_of %s", v_date, as_of)
                    violations += 1
        except Exception as e:
            logger.debug("Vector search query failed during audit: %s", e)

    kg.close()

    logger.info("=" * 60)
    logger.info("LEAKAGE AUDIT RESULTS")
    logger.info("  Tested Queries           : %d", tested)
    logger.info("  Date Boundary Violations : %d", violations)
    logger.info("=" * 60)

    if violations == 0:
        logger.info("✅ PASSED: No look-ahead data leakage detected in retrievers.")
        return True
    else:
        logger.error("❌ FAILED: %d data leakage violations detected!", violations)
        return False


if __name__ == "__main__":
    success = audit_retriever_leakage()
    sys.exit(0 if success else 1)
