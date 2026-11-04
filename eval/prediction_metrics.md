# Football Match Prediction Evaluation Report

**Date Generated:** 2026-10-31 19:57:51  
**Matches Evaluated:** 15  
**Langfuse Logging:** Enabled  

## 1. Executive Summary Metrics

| Metric | Result | Benchmark / Target | Status |
| :--- | :---: | :---: | :---: |
| **Expert 1 (GNN) Accuracy** | **53.3%** | > 45.0% | PASS |
| **Expert 2 (SLM) Accuracy** | **46.7%** | > 45.0% | PASS |
| **Hybrid Ensemble Accuracy** | **53.3%** | > 50.0% | PASS |
| **Multi-Class Brier Score** | **0.5461** | < 0.6000 (Random: 0.667) | PASS |
| **Multi-Class Log Loss** | **0.9280** | < 1.0500 (Random: 1.098) | PASS |
| **GNN-SLM Consensus Rate** | **86.7%** | > 60.0% | PASS |
| **Tactical Fact Grounding** | **0.97** | > 0.80 | PASS |
| **Average Latency / Match** | **1.98s** | < 4.00s | PASS |

## 2. Match-by-Match Breakdown

| Match ID | Matchup | Actual | GNN Pred | SLM Pred | Ensemble | Brier Score | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `pred_003` | Auxerre vs Nice | H (2-1) | H (46%) | H | H | 0.450 | **PASS** |
| `pred_006` | Chelsea vs Manchester City | A (0-2) | H (34%) | H | H | 0.694 | **MISS** |
| `pred_009` | Espanyol vs Real Sociedad | A (0-1) | D (44%) | H | H | 0.913 | **MISS** |
| `pred_012` | Nice vs Toulouse | D (1-1) | A (42%) | A | A | 0.597 | **MISS** |
| `pred_015` | Arsenal vs Brighton | D (1-1) | H (44%) | H | H | 0.717 | **MISS** |
| `pred_018` | Bochum vs Borussia M.Gladbach | A (0-2) | D (55%) | D | D | 0.972 | **MISS** |
| `pred_021` | Real Madrid vs Real Betis | H (2-0) | H (36%) | H | H | 0.609 | **PASS** |
| `pred_024` | Genoa vs Roma | D (1-1) | D (46%) | D | D | 0.436 | **PASS** |
| `pred_027` | Verona vs Torino | A (2-3) | D (45%) | D | D | 0.717 | **MISS** |
| `pred_030` | Real Madrid vs Espanyol | H (4-1) | H (77%) | A | H | 0.094 | **PASS** |
| `pred_033` | Osasuna vs Las Palmas | H (2-1) | H (57%) | H | H | 0.308 | **PASS** |
| `pred_036` | Athletic Club vs Celta Vigo | H (3-1) | A (44%) | A | A | 1.119 | **MISS** |
| `pred_039` | Barcelona vs Getafe | H (1-0) | H (89%) | H | H | 0.022 | **PASS** |
| `pred_042` | Augsburg vs Borussia M.Gladbach | H (2-1) | H (67%) | H | H | 0.193 | **PASS** |
| `pred_045` | Las Palmas vs Celta Vigo | A (0-1) | A (55%) | A | A | 0.352 | **PASS** |

## 3. Metric Definitions & Formulas
- **Outcome Accuracy:** Exact match hit rate between predicted class (H, D, A) and official match result.
- **Multi-Class Brier Score:** $\frac{1}{N} \sum_{i=1}^N \sum_{k \in \{H,D,A\}} (p_{ik} - y_{ik})^2$. Penalizes poorly calibrated probability estimates.
- **Log Loss:** $-\frac{1}{N} \sum_{i=1}^N \ln(p_{i, \text{actual}})$. Measures probabilistic confidence penalization.
- **Consensus Rate:** Proportion of matches where the quantitative GNN and the generative LLM agree.