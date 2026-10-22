"""
pgvector Index Builder
======================
Embeds the same knowledge-base documents as build_faiss_index.py and stores
them in a PostgreSQL table using the pgvector extension.

Usage:
    python rag/build_pgvector_index.py              # build only if table is empty
    python rag/build_pgvector_index.py --rebuild    # drop + rebuild
    python rag/build_pgvector_index.py --analyses   # also embed tactical analyses
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

from rag.build_faiss_index import _collections, _load_rag_cfg, embed_docs

logger = logging.getLogger(__name__)

EMBED_DIM = 384
DEFAULT_DSN = "postgresql://postgres:123@localhost:5432/football_rag"


def _settings() -> tuple[str, str, str]:
    cfg = _load_rag_cfg()
    dsn = os.getenv("POSTGRES_DSN") or cfg.get("postgres_dsn", DEFAULT_DSN)
    table = cfg.get("pgvector_table", "kb_vectors")
    model = cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    return dsn, table, model


def vector_count(dsn: str, table: str) -> int:
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        if cur.fetchone()[0] is None:
            return 0
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]


def _create_table(cur, table: str, rebuild: bool) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if rebuild:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id        BIGSERIAL PRIMARY KEY,
            text      TEXT NOT NULL,
            metadata  JSONB NOT NULL,
            embedding vector({EMBED_DIM}) NOT NULL
        )
        """
    )


def main(rebuild: bool = False, force_analyses: bool = False) -> None:
    dsn, table, model_name = _settings()

    if not rebuild and vector_count(dsn, table) > 0:
        logger.info("pgvector table '%s' already populated — skipping.", table)
        return

    cfg = _load_rag_cfg()
    index_analyses = bool(cfg.get("kb_index_analyses", False)) or force_analyses

    docs = []
    for label, coll in _collections(False, False, index_analyses):
        logger.info("Collection '%s': %d docs", label, len(coll))
        docs += coll
    if not docs:
        raise SystemExit("No documents to index — aborting.")

    vectors = embed_docs(docs, SentenceTransformer(model_name))

    rows = [
        (d["text"],
         json.dumps({k: v for k, v in d.items() if k != "text"}, default=str),
         "[" + ",".join(f"{x:.6f}" for x in vec) + "]")
        for d, vec in zip(docs, vectors)
    ]

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        _create_table(cur, table, rebuild)
        cur.execute(f"TRUNCATE {table}")
        execute_values(
            cur,
            f"INSERT INTO {table} (text, metadata, embedding) VALUES %s",
            rows,
            template="(%s, %s::jsonb, %s::vector)",
            page_size=500,
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table} "
            "USING hnsw (embedding vector_cosine_ops)"
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_meta ON {table} USING gin (metadata)")
    logger.info("pgvector build complete: %d vectors in '%s'", len(rows), table)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build pgvector knowledge-base index")
    parser.add_argument("--rebuild", action="store_true", help="Drop and rebuild the table")
    parser.add_argument("--analyses", action="store_true", help="Also embed tactical analyses")
    args = parser.parse_args()
    main(rebuild=args.rebuild, force_analyses=args.analyses)
