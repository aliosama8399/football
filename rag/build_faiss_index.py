"""
FAISS Vector Index Builder (v2)
================================
Embeds the knowledge-base document collections from rag.knowledge_base.docs:

  match_stats   — every row of processed_matches.csv (13,293 docs)
  team_season   — per team × league × season summaries (~1,300 docs)
  team_profile  — tactical profiles from team_tactics.json (96 docs)
  analysis      — tactical analyses from football_tactical.jsonl (13,911 docs)
                  ONLY when rag.kb_index_analyses=true (or --analyses)

Output (paths from models/llm_config.yaml):
    rag/vector_store/faiss.index
    rag/vector_store/faiss_metadata.json

Usage:
    python rag/build_faiss_index.py                 # default (respects kb_index_analyses)
    python rag/build_faiss_index.py --analyses      # force-include analysis docs
    python rag/build_faiss_index.py --analyses-only # only analysis docs
    python rag/build_faiss_index.py --stats-only    # only match stat chunks (legacy)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import faiss
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag.knowledge_base.docs import build_all_docs, load_tactical_analyses

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
CFG_PATH    = BASE_DIR / "models" / "llm_config.yaml"
OUT_DIR     = BASE_DIR / "rag" / "vector_store"
BATCH_SIZE  = 64


def _load_rag_cfg() -> dict:
    if not CFG_PATH.exists():
        return {}
    with open(CFG_PATH) as f:
        return (yaml.safe_load(f) or {}).get("rag", {})


def _collections(analyses_only: bool, stats_only: bool, index_analyses: bool) -> list:
    """Return (label, docs) pairs to embed, honoring the CLI modes + config."""
    cfg = _load_rag_cfg()
    csv_path = BASE_DIR / cfg.get("kb_csv_path", "data/processed/processed_matches.csv")
    tactics_path = BASE_DIR / cfg.get("kb_tactics_path", "rag/knowledge_base/team_tactics.json")
    analyses_path = BASE_DIR / "data" / "finetune" / "football_tactical.jsonl"

    collections = []
    if analyses_only:
        collections.append(("analysis", load_tactical_analyses(analyses_path)))
    elif stats_only:
        from rag.knowledge_base.docs import build_match_stat_chunks
        import pandas as pd
        collections.append(("match_stats (legacy stats-only)",
                            build_match_stat_chunks(pd.read_csv(csv_path, low_memory=False))))
    else:
        docs = build_all_docs(csv_path, tactics_path, analyses_path, index_analyses)
        label = "match_stats + team_season + team_profile"
        if index_analyses:
            label += " + analysis"
        collections.append((label, docs))
    return collections


# ─────────────────────────────────────────────────────────────────────────────
# Embedding + Indexing
# ─────────────────────────────────────────────────────────────────────────────

def embed_docs(docs: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [d["text"] for d in docs]
    all_vecs = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[start:start + BATCH_SIZE]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs)
    return np.vstack(all_vecs).astype("float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    n = vectors.shape[0]
    if n > 10_000:
        nlist = min(256, n // 40)
        quant = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_L2)
        index.train(vectors)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(analyses_only: bool = False, stats_only: bool = False,
         force_analyses: bool = False):
    cfg = _load_rag_cfg()
    embed_model = cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    index_path = BASE_DIR / cfg.get("faiss_index_path", "rag/vector_store/faiss.index")
    meta_path = BASE_DIR / cfg.get("faiss_metadata_path", "rag/vector_store/faiss_metadata.json")
    index_analyses = bool(cfg.get("kb_index_analyses", False)) or force_analyses

    logger.info("Loading embedding model: %s", embed_model)
    model = SentenceTransformer(embed_model)

    docs = []
    for label, coll in _collections(analyses_only, stats_only, index_analyses):
        logger.info("Collection '%s': %d docs", label, len(coll))
        docs += coll

    logger.info("Total documents to index: %d", len(docs))
    if not docs:
        raise SystemExit("No documents to index — aborting.")

    vectors = embed_docs(docs, model)
    index = build_index(vectors)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved → %s", index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    logger.info("Metadata saved → %s  (%d entries)", meta_path, len(docs))
    logger.info("✅  FAISS index build complete (analyses included: %s)", index_analyses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vector index (v2)")
    parser.add_argument("--analyses-only", action="store_true", help="Only embed tactical analyses")
    parser.add_argument("--stats-only",    action="store_true", help="Only embed match stat chunks")
    parser.add_argument("--analyses",      action="store_true",
                        help="Force-include analysis docs (overrides kb_index_analyses)")
    args = parser.parse_args()
    main(analyses_only=args.analyses_only, stats_only=args.stats_only,
         force_analyses=args.analyses)
