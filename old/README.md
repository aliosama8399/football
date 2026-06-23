# Explainable Football Analytics & Prediction Engine ⚽🤖

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)
![GNN](https://img.shields.io/badge/Graph_Neural_Networks-PyG-blueviolet)

This repository contains a comprehensive **machine learning and AI pipeline** for predicting football match outcomes across Europe's top leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1, and the UEFA Champions League) and generating professional, natural language **tactical analysis reports**. 

The system bridges the gap between raw statistical data, advanced graph-based predictive modeling, and Generative AI.

---

## 🌟 Key Features & Architecture

### 1. Data Pipeline & Feature Engineering
- **Multi-Source Scraping**: Pulls historical and recent match logs from `football-data.co.uk` and expected goals (xG) statistics from **Understat** via custom API scrapers.
- **Advanced Features**: Generates rolling 5-match forms, head-to-head statistics, referee strictness metrics, and performance deltas (xG vs. actual goals). Uses 39 engineered pre-match features avoiding target leakage.

### 2. Machine Learning Baseline
- Trains and evaluates 10 traditional classification models (XGBoost, CatBoost, Random Forest, MLP, KNN, etc.) bundled into a Voting Ensemble to establish benchmark accuracy (~51% on test sets).

### 3. Graph Neural Networks (GNNs)
- **Topological Modeling**: Treats the leagues as a complex network where nodes are teams and edges represent historical matches.
- **Algorithms Implemented**: Graph Convolutional Networks (GCN), GraphSAGE, Graph Attention Networks (GAT), Graph Isomorphism Networks (GIN), and **EdgeCNN** (EdgeConv).
- **Hyperparameter Tuning**: Optuna sweeps for hidden dimensions, dropout rates, and learning rates. EdgeConv achieved the highest test accuracy (61.59%).

### 4. Explainable AI (XAI)
- **GNNExplainer**: Deployed PyTorch Geometric's `GNNExplainer` on the trained EdgeConv model to identify the most influential node features (e.g., away team's recent xGA) and the historical edges (previous matches) that drove the prediction. 

### 5. Generative AI & SLM Fine-Tuning
- **Teacher-Student Distillation**: A Python data builder (`build_finetune_dataset.py`) extracts the GNN probabilities and Explainer outputs and prompts a "Teacher" LLM (Google Gemini / OpenAI) to write professional, pundit-style tactical analyses.
- **Small Language Models (SLMs)**: Leverages **Unsloth** and **QLoRA** to fine-tune compact 1.5B – 3B parameter models (`Qwen2.5-1.5B`, `SmolLM2-1.7B`) locally or on Google Colab. These fine-tuned models inherit the GNN's statistical awareness and learn to reproduce expert analysis and predictions autonomously.

---

## 📂 Repository Structure

```text
├── data/
│   ├── collectors/          # Scrapers for FBRef, Understat, etc.
│   ├── raw/                 # Unprocessed CSVs
│   ├── processed/           # Feature-engineered datasets
│   └── build_finetune_dataset.py # Pipeline generating JSONL data for LLM training
│
├── models/
│   ├── train_traditional.py # Scikit-Learn/XGBoost baseline and ensembles
│   ├── gnn_models.py        # PyTorch Geometric architecture definitions
│   ├── train_gnn.py         # PyTorch GNN training loop and validation
│   ├── tune_gnn.py          # Optuna hyperparameter optimization script
│   ├── explain_match.py     # Interactive CLI to run GNNExplainer and LLM prompt
│   ├── llm_providers.py     # Unified interface for OpenAI, Gemini, Ollama API
│   └── llm_config.yaml      # Configuration and API keys
│
├── notebooks/
│   ├── finetune_qwen25.ipynb   # Unsloth fine-tuning pipeline for Qwen2.5
│   └── finetune_smollm2.ipynb  # Unsloth fine-tuning pipeline for SmolLM2
│
└── .gitignore               # Excludes model weights, large datasets, & cache
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed. For GNN and LLM inference, an NVIDIA GPU with CUDA support is highly recommended.

```bash
pip install -r requirements.txt
```
*(Dependencies include `torch`, `torch_geometric`, `pandas`, `scikit-learn`, `optuna`, `xgboost`, `catboost`, `openai`, `google-generativeai`, `unsloth`)*

### 2. Data Collection & Preprocessing
To gather data and build the 39-feature dataset:
```bash
python data/collect_all.py
python data/preprocess.py
```

### 3. Training the Graph Neural Network
Train the PyTorch Geometric models on the processed edge data:
```bash
python models/train_gnn.py
```
*(Optionally, use `tune_gnn.py` to run the Optuna optimization)*

### 4. Interactive Match Analysis & Explainability
Run a CLI to predict a specific match, extract its critical statistical features via `GNNExplainer`, and query an LLM for a tactical summary:
```bash
python models/explain_match.py --home "Real Madrid" --away "Barcelona" --provider gemini
```

### 5. SLM Fine-Tuning
To train your own Small Language Model (SLM) to replace paid APIs:
1. Generate the JSONL training set: `python data/build_finetune_dataset.py`
2. Run `notebooks/finetune_smollm2.ipynb` (locally or on Colab) to train a LoRA adapter via Unsloth.

---

## 📝 License
This project was developed for academic/research purposes as part of a Master's Thesis. 

*(Please note: Betting/odds data was explicitly excluded from the pipeline for ethical reasons, relying purely on team performance, expected goals, and referee metrics).*
