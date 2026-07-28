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

### Step 1: Collect Raw Data
Downloads 10 seasons of match and expected goals (xG) data for PL, La Liga, Bundesliga, Serie A, and Ligue 1:
```powershell
python data/collect_all.py
```

### Step 2: Preprocessing & Feature Engineering
Calculates rolling form features (past 5 matches), referee strictness, and H2H records:
```powershell
python data/preprocess.py
```

### Step 3: Train Match Prediction Models
Train GNN (EdgeConv) or traditional machine learning models:
```powershell
# Train GNN
python models/train_gnn.py

# Train CatBoost/RandomForest/Voting Ensembles
python models/train_traditional.py
```

### Step 4: Build RAG Knowledge Base
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
