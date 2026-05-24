"""
FAISS Vector Index Builder
===========================
Embeds two document collections and saves a single FAISS flat index:

  Collection 1 — Tactical Analyses  (football_tactical.jsonl, 1,753 docs)
  Collection 2 — Match Stat Chunks   (processed_matches.csv, 5,330 docs)

Output:
    rag/vector_store/faiss.index         (FAISS binary index)
    rag/vector_store/faiss_metadata.json (parallel list of metadata dicts)

Usage:
    python rag/build_faiss_index.py
    python rag/build_faiss_index.py --analyses-only
    python rag/build_faiss_index.py --stats-only
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import logging
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent.parent
JSONL_PATH    = BASE_DIR / "data" / "finetune" / "football_tactical.jsonl"
CSV_PATH      = BASE_DIR / "data" / "processed" / "processed_matches.csv"
OUT_DIR       = BASE_DIR / "rag" / "vector_store"
INDEX_PATH    = OUT_DIR / "faiss.index"
META_PATH     = OUT_DIR / "faiss_metadata.json"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE    = 64

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Document Builders
# ─────────────────────────────────────────────────────────────────────────────

def load_analysis_docs(path: Path) -> list[dict]:
    """
    Load tactical analyses from JSONL.
    Each doc has: text (the analysis), metadata (match info).
    """
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            match_id = row.get("match_id", "")
            messages = row.get("messages", [])
            if len(messages) < 2:
                continue

            analysis = messages[1].get("content", "")
            if not analysis:
                continue

            # Parse team names from match_id
            parts = match_id.split("_vs_")
            home_team = parts[0].rsplit("_", maxsplit=1)[-1].strip() if len(parts) >= 2 else "?"
            away_team = parts[1].strip() if len(parts) >= 2 else "?"
            date_str  = parts[0].split("_")[0] if "_" in parts[0] else "?"

            docs.append({
                "text":          analysis,
                "match_id":      match_id,
                "home_team":     home_team,
                "away_team":     away_team,
                "date":          date_str,
                "actual_result": row.get("actual_result", "?"),
                "gnn_prediction":row.get("gnn_prediction", "?"),
                "doc_type":      "analysis",
            })

    logger.info("Loaded %d analysis documents.", len(docs))
    return docs


def load_match_stat_docs(path: Path) -> list[dict]:
    """
    Build a human-readable text chunk for every match row in the CSV.
    """
    df = pd.read_csv(path, low_memory=False)
    docs = []

    for _, row in df.iterrows():
        home  = row.get("HomeTeam", "?")
        away  = row.get("AwayTeam", "?")
        date  = str(row.get("Date", "?"))
        lg    = row.get("League", "?")
        ssn   = row.get("Season", "?")
        hg    = row.get("FTHG", "?")
        ag    = row.get("FTAG", "?")
        ftr   = row.get("FTR",  "?")
        hxg   = row.get("Home_xG", "?")
        axg   = row.get("Away_xG", "?")
        hs    = row.get("HS",  "?")
        as_   = row.get("AS",  "?")
        hst   = row.get("HST", "?")
        ast   = row.get("AST", "?")
        hf    = row.get("HomeForm_5", "?")
        af    = row.get("AwayForm_5", "?")
        hgf   = row.get("HomeGF_5",  "?")
        agf   = row.get("AwayGF_5",  "?")
        hxg5  = row.get("HomexG_5",  "?")
        axg5  = row.get("AwayxG_5",  "?")
        h2h_m = row.get("H2H_Matches", 0)
        h2h_hw= row.get("H2H_HomeWins", 0)
        h2h_aw= row.get("H2H_AwayWins", 0)

        def _fmt(v):
            try:
                return f"{float(v):.2f}"
            except (TypeError, ValueError):
                return str(v)

        text = (
            f"{home} (Home) vs {away} (Away) | {lg} | Season {ssn} | {date}\n"
            f"Result: {ftr} ({hg}-{ag}) | xG: {_fmt(hxg)}-{_fmt(axg)}\n"
            f"Shots: {hs}-{as_} | SOT: {hst}-{ast}\n"
            f"Home Form (5): PPM={_fmt(hf)}, GF={_fmt(hgf)}, xG={_fmt(hxg5)}\n"
            f"Away Form (5): PPM={_fmt(af)}, GF={_fmt(agf)}, xG={_fmt(axg5)}\n"
            f"H2H: {h2h_m} meetings | Home wins: {h2h_hw} | Away wins: {h2h_aw}"
        )

        docs.append({
            "text":     text,
            "home_team": str(home),
            "away_team": str(away),
            "date":      date,
            "league":    str(lg),
            "season":    str(ssn),
            "result":    str(ftr),
            "actual_result": str(ftr),
            "doc_type":  "match_stats",
        })

    logger.info("Loaded %d match-stat documents.", len(docs))
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Embedding + Indexing
# ─────────────────────────────────────────────────────────────────────────────

def embed_docs(docs: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Embed all docs in batches. Returns float32 array shape (N, dim)."""
    texts  = [d["text"] for d in docs]
    all_vecs = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[start:start + BATCH_SIZE]
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs)

    return np.vstack(all_vecs).astype("float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    n   = vectors.shape[0]

    # Use IVFFlat for large indexes (>10k), else flat
    if n > 10_000:
        nlist  = min(256, n // 40)
        quant  = faiss.IndexFlatL2(dim)
        index  = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_L2)
        index.train(vectors)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(vectors)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(analyses_only: bool = False, stats_only: bool = False):
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    docs = []
    if not stats_only:
        docs += load_analysis_docs(JSONL_PATH)
    if not analyses_only:
        docs += load_match_stat_docs(CSV_PATH)

    logger.info("Total documents to index: %d", len(docs))

    vectors = embed_docs(docs, model)
    index   = build_index(vectors)

    # Save index
    faiss.write_index(index, str(INDEX_PATH))
    logger.info("FAISS index saved → %s", INDEX_PATH)

    # Save metadata (strip 'text' from metadata to avoid double storage)
    meta_list = [{k: v for k, v in d.items()} for d in docs]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False)
    logger.info("Metadata saved → %s  (%d entries)", META_PATH, len(meta_list))
    logger.info("✅  FAISS index build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vector index for football RAG")
    parser.add_argument("--analyses-only", action="store_true", help="Only embed tactical analyses")
    parser.add_argument("--stats-only",    action="store_true", help="Only embed match stat chunks")
    args = parser.parse_args()
    main(analyses_only=args.analyses_only, stats_only=args.stats_only)
