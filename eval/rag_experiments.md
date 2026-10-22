# RAG Experiment Matrix Results

Generated: 2026-10-15 18:29:09  
Evaluated on: 15 held-out query samples.

| Configuration | Avg Latency | Numeric Grounding | Grounding / Relevancy | Status |
|---|---|---|---|---|
| **Neo4j + pgvector (Default Dense)** | 28.22s | 0.60 | 0.96 | Completed |
| **Postgres + pgvector (Dense)** | 25.83s | 0.55 | 0.96 | Completed |
| **Neo4j + pgvector (Hybrid BM25+Dense)** | 21.39s | 0.55 | 0.96 | Completed |
| **Neo4j + pgvector (Hybrid + Reranker)** | 27.49s | 0.65 | 0.94 | Completed |
| **Neo4j + pgvector (As-Of Filter OFF / Leaky)** | 20.74s | 0.55 | 0.91 | Completed |

## Key Findings & Decisions
- **Neo4j Graph vs PostgreSQL:** Neo4j is default KG; provides unified graph traversal for team relations.
- **pgvector Vector Store:** Replaces standalone external FAISS index in Docker; vectors reside in PostgreSQL with cosine HNSW index.
- **Hybrid (BM25 + Dense):** Improves exact keyword matching for specific team tactical jargon.
- **As-Of Date Rule:** Essential for preventing look-ahead data leakage on historical match predictions.