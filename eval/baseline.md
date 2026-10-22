# Baseline System State & Evaluation Protocol

Date: 2026-10-15  
Project: Football Match Prediction & Tactical Analysis (TEA-GNN + Qwen SLM)

---

## 1. Core Architecture & Configuration

| Component | Default Configuration | Alternatives Supported |
|---|---|---|
| **KG Provider** | `neo4j` (Bolt: `bolt://localhost:7687` or `bolt://neo4j:7687`) | `postgres` (`teams`, `matches`) |
| **Vector Provider** | `pgvector` (`kb_vectors` table, HNSW cosine index) | `faiss` (IVFFlat L2 index) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | - |
| **SLM (Expert 2)** | Fine-tuned Qwen 0.6B (`aliosama8399/football-analysisN` on Hugging Face / vLLM / ONNX) | Base Qwen, OpenAI, Gemini, Ollama |
| **GNN (Expert 1)** | TEA-GNN ONNX runtime (`models/export/gnn/tea_gnn.onnx`) | Torch checkpoint |
| **Experiment Tracker** | MLflow 3.16.0 (SQLite metadata + artifact store) | - |
| **Observability / Tracing** | Langfuse Cloud SDK v3 (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`) | - |

---

## 2. Dataset Baselines

- **Match Dataset:** `data/processed/processed_matches.csv` (13,293 matches across 5 top European leagues: Premier League, La Liga, Serie A, Bundesliga, Ligue 1; seasons 2015-16 to 2024-25).
- **Tactical Profiles:** `rag/knowledge_base/team_tactics.json` (96 top teams).
- **Fine-tuning Corpus:** `data/finetune/football_tactical.jsonl` (13,911 tactical summaries), with split sets `football_train.json` (11,128 samples) and `football_val.json` (2,783 samples).

---

## 3. Temporal Split & Anti-Leakage Rule

To prevent look-ahead bias and data leakage during evaluation:
- **Knowledge Cutoff / Split Date:** `2024-08-01`
- **Historical / In-Sample Knowledge:** All matches with `Date < 2024-08-01` (seasons 2015-16 through 2023-24).
- **Held-Out Test Window:** All matches in season 2024-25 (`Date >= 2024-08-01`).
- **Strict `as_of` Date Rule:** When evaluating a match played on date $D$, retrieval queries are strictly filtered with `date < D`. The retriever is barred from seeing future matches or circular post-match tactical summaries written after $D$.

---

## 4. Evaluation Targets

- **RAG Faithfulness:** $\ge 0.75$
- **Answer Relevancy:** $\ge 0.80$
- **Numeric Fact Grounding:** $\ge 0.90$
- **SLM JSON Format Validity:** $100\%$
- **Prediction Alignment:** GNN and SLM probability direction alignment checked on held-out matches.
