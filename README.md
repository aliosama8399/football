# Football Match Prediction & Tactical Analysis System (XAI + RAG)

This repository contains a state-of-the-art hybrid AI system for football match predictions, explainable analytics, and interactive tactical queries. It combines:
1. **Graph Neural Networks (GNN)** (EdgeConv) and traditional ML classifiers for outcome probabilities.
2. **Explainable AI (XAI)** to translate model parameters and GNNExplainer inputs into detailed, professional tactical reports.
3. **Retrieval-Augmented Generation (RAG)** built on a Dual-Database Knowledge Graph (Neo4j / PostgreSQL) and a FAISS vector index of tactical profiles.
4. **Interactive Dashboard & Web Portal** (FastAPI backend + responsive web UI) for match sandbox predictions and tactical chats.

---

## 🏗️ System Architecture

```
football/
├── api/                    ← FastAPI Server & Web App
│   ├── routes/             ← Authentication, Predictions, Chat, Feedback
│   ├── static/             ← Dashboard Frontend (HTML, CSS, JS)
│   └── main.py             ← API Entry Point
├── data/
│   ├── collectors/         ← Scrapers for football-data.co.uk & Understat
│   ├── processed/          ← Normalized match datasets
│   ├── features/           ← ML-ready tables
│   ├── finetune/           ← Generated training data for LLMs/SLMs
│   ├── collect_all.py      ← Raw data collection orchestrator
│   ├── preprocess.py       ← Feature engineering & rolling form calculator
│   └── build_finetune.py   ← GNN-to-LLM prompt generator
├── models/
│   ├── LlamaFactory/       ← Fine-tuning framework
│   ├── gnn_models.py       ← EdgeConv GNN implementation
│   ├── hf_provider.py      ← GPU/CPU execution manager for fine-tuned LLMs
│   ├── train_gnn.py        ← GNN training script
│   └── train_traditional.py← Ensemble classifier trainer
└── rag/                    ← Retrieval-Augmented Generation System
    ├── SETUP.md            ← Specific database setup configurations
    ├── extract_tactics.py  ← Entity/Relationship extractor from match logs
    ├── build_neo4j_kg.py   ← Neo4j population script
    ├── build_postgres_db.py← PostgreSQL population script
    ├── build_faiss_index.py← FAISS vector store indexer
    └── rag_orchestrator.py ← Retrieval & generation pipeline
```

---

## 🚀 Setup & Installation

### 1. Environment Setup
Activate your base Conda environment or virtual environment:
```powershell
conda activate base
# Or create one:
# conda create -n football python=3.12
# conda activate football
```

Install the dependencies:
```powershell
pip install -r requirements-api.txt
```

### 2. Databases (RAG)
Start your database instances (via Docker or local services):
* **Neo4j** (default): `docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest`
* **PostgreSQL**: `docker run -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:15`

Create the database in Postgres:
```sql
CREATE DATABASE football_rag;
```

Update your configuration parameters in **`models/llm_config.yaml`**:
```yaml
rag:
  neo4j_password: "password"
  postgres_dsn:   "postgresql://postgres:password@localhost:5432/football_rag"
```

For more details, check [rag/SETUP.md](file:///d:/SASUniversityEdition/Machine/MODEL/football/rag/SETUP.md).

---

## 🔄 Data & Model Pipelines

Everything is **config-driven**: `data/config.yaml` is the single source of truth for seasons, leagues, and window sizes (`data/_config.py` loads it — no hard-coded constants in collectors or preprocess). Team-name aliases live in `data/team_registry.py`.

Run the stages in order. Each stage re-runs cleanly on top of the previous one; the only stage that takes a while is weather enrichment inside Stage 2 (~1h, rate-limited).

### Stage 1 — Collect Raw Data

```powershell
# football-data.co.uk match results + Understat xG (via soccerdata), all seasons & leagues from config
python data/collect_all.py

# Per-match player stats (minutes, goals, xG, xA, shots, key passes) — powers H2H blends & through-date ratings
python -m data.collectors.player_match_stats --leagues E0,SP1,D1,I1,F1 --seasons 2425

# Optional: warm all best-11 squad caches (~98 teams, 50-60 min) so first-use predictions never scrape live
python -m data.collectors.prewarm_squads --season 2425
```

Outputs: `data/raw/football_data_uk_combined.csv`, `data/raw/understat_xg_data.csv`, `data/raw/player_match/player_match_{LEAGUE}_{SEASON}.csv`, `data/raw/squads_cache/all_*.json`.

### Stage 2 — Preprocessing & Feature Engineering

```powershell
python data/preprocess.py
python data/sanity_check.py   # quick validation of the processed table
```

Builds rolling form (5-match window), H2H records, referee strictness, weather (Open-Meteo archive, rate-limited ~1h) and season-cumulative features. Outputs: `data/processed/processed_matches.csv`, `data/features/ml_ready_data.csv`.

### Stage 3 — Train Match Prediction Models

```powershell
# GNN (EdgeConv) — uses processed_matches.csv
python models/train_gnn.py

# Traditional ensemble (CatBoost / RandomForest / Voting) — trains 22/23+23/24, tests on 24/25
python models/train_traditional.py

# Optional: hyperparameter tuning (Optuna) — slow, run after a successful baseline
python models/tune_gnn.py
python models/tune_models.py

# Optional: GNNExplainer → LLM finetune dataset (RAG training data for data/finetune/)
python data/build_finetune_dataset.py
```

### Stage 4 — Best-11 Feature (ratings, H2H, lineups)

No training here — team-share ratings are computed on the fly from `processed_matches.csv` (`data/team_totals.py`) + the squad caches. After Stage 1+2 you can smoke-test:

```powershell
python data/best11.py "Real Madrid" SP1 --season 2425 --formation auto
python data/best11.py "Barcelona" SP1 --season 2425 --formation auto --json
```

The feature is structured with design patterns (SOLID) across three
layers:

- `api/repositories/best11_repo.py` — **Repository**: `Best11Repository`
  collects every piece of best-11 data from its sources (squad
  providers + disk cache, team totals, per-match form/H2H stats), the
  same way `TeamGraphRepository` wraps the KG provider.
- `data/players/` — **Strategy** (rating modes: season / through-date /
  H2H-blend; formation auto-fit; rotation subs) + **Facade**:
  `Best11Service.solve()` orchestrates the repositories and strategies.
- Backend service + route — `api/services/best11_service.py`
  (`Best11ApiService`: league mapping, validation, error mapping) and
  `api/routes/best11.py`. Both the REST endpoint and the GraphQL
  resolver go through this one application service:

```powershell
curl "http://localhost:8000/api/v1/best11?team=Barcelona&league=La_Liga&season=2425&formation=auto&opponent=Real+Madrid"
```

`data/best11.py` remains the CLI facade; GraphQL resolvers call
`solve_best11()` through it unchanged.

### Stage 5 — Build RAG Knowledge Base

1. Extract tactical attributes from generated analysis data:
   ```powershell
   python rag/extract_tactics.py
   ```
2. Populate the databases:
   ```powershell
   python rag/build_neo4j_kg.py
   python rag/build_postgres_db.py
   ```
3. Generate the FAISS embeddings index:
   ```powershell
   python rag/build_faiss_index.py
   ```

### Stage 6 — Run the Web Application

```powershell
python -m uvicorn api.main:app --reload --port 8001
```

Open **`http://localhost:8001`** (Admin Portal default credentials: `admin` / `AdminPass123!`).

---

## 📅 Adding a New Season (e.g. 2025-26)

The code is structured so a new season needs **no collector or preprocess changes**:

1. **Edit `data/config.yaml`** — the only file that defines scope:
   ```yaml
   seasons: ['1516', ..., '2425', '2526']        # football-data.co.uk format
   understat_years: [2015, ..., 2024, 2025]      # calendar year of season start
   soccerdata_seasons: ['2015-2016', ..., '2024-2025', '2025-2026']
   ```
   The three lists must stay aligned (same count, matching entries).
2. **`data/team_registry.py`** — add aliases for promoted/new teams (needed for team-name normalization in H2H and provider fusion).
3. **Update the few hard-coded test-season spots** (training scripts split train/test on the current season `2425`):
   | File | Line | Constant |
   |---|---|---|
   | `models/train_traditional.py` | 248 | `test_mask = df['Season'] == 2425` |
   | `models/tune_models.py` | 173 | `test_mask = df['Season'] == 2425` |
   | `data/graph_builder.py` | 185 | `test_seasons = [2425]` |
   | `data/build_finetune_dataset.py` | ~700 | `process_seasons` list |
   | `data/validate_ratings.py` | 32 | `SEASON = "2425"` |
   | `data/collectors/player_probe.py` | 34 | `SEASON = "2425"` |
   | `api/graphql/resolvers.py` | 106 | `best_11` default `season="2425"` |
   Everything else takes the season as a CLI/API argument (`--season`, `--seasons`, GraphQL `season`) — collectors, preprocess, prewarm, player_match_stats, best11, and the H2H feed all read it per call.
4. **Rerun Stages 1 → 6** above with the new season code. Caches are keyed per team+league+season (`all_{Team}_{League}_{Season}.json`), so old-season caches stay valid and new ones are created on demand or via `prewarm_squads`.

**Retraining after a code change** (feature tweaks, bug fixes): only Stages 2, 3, 4, and 6 need rerunning — collection is unchanged. `processed_matches.csv` and the squad caches are idempotent outputs of their stages, so re-running them is safe.


---

## 💻 Running the Web Application

To run the local web server and dashboard UI, run from the root:

```powershell
python -m uvicorn api.main:app --reload --port 8001
```

Open your browser and navigate to **`http://localhost:8001`**.

* **Admin Portal default credentials:**
  * Username: `admin`
  * Password: `AdminPass123!`

---

## 💡 RAG Queries & API Usage

You can use the RAG system directly through Python or via command line queries:

### Python API
```python
from rag import FootballRAGSystem

# Automatically uses the provider defined in llm_config.yaml
rag = FootballRAGSystem()

# Run a query
report = rag.predict_match("Arsenal", "Chelsea")
print(report)
```

### CLI Client
```powershell
# Start an interactive tactical chat
python rag/rag_orchestrator.py

# Predict a single match using RAG context
python rag/rag_orchestrator.py --predict "Real Madrid" "Barcelona"
```
