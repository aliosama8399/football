# Football Match Prediction Evaluation Report

**Date Generated:** 2026-10-28 20:23:35  
**Matches Evaluated:** 15  
**Langfuse Logging:** Enabled  

## 1. Executive Summary Metrics

| Metric | Result | Benchmark / Target | Status |
| :--- | :---: | :---: | :---: |
| **Expert 1 (GNN) Accuracy** | **40.0%** | > 45.0% | WARN |
| **Expert 2 (SLM) Accuracy** | **40.0%** | > 45.0% | WARN |
| **Hybrid Ensemble Accuracy** | **40.0%** | > 50.0% | WARN |
| **Multi-Class Brier Score** | **0.6948** | < 0.6000 (Random: 0.667) | WARN |
| **Multi-Class Log Loss** | **1.1477** | < 1.0500 (Random: 1.098) | WARN |
| **GNN-SLM Consensus Rate** | **100.0%** | > 60.0% | PASS |
| **Tactical Fact Grounding** | **0.83** | > 0.80 | PASS |
| **Average Latency / Match** | **19.66s** | < 4.00s | WARN |

## 2. Match-by-Match Breakdown

| Match ID | Matchup | Actual | GNN Pred | SLM Pred | Ensemble | Brier Score | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `pred_003` | Auxerre vs Nice | H (2-1) | A (49%) | A | A | 1.084 | **MISS** |
| `pred_006` | Chelsea vs Manchester City | A (0-2) | A (38%) | A | A | 0.571 | **PASS** |
| `pred_009` | Espanyol vs Real Sociedad | A (0-1) | D (44%) | D | D | 0.987 | **MISS** |
| `pred_012` | Nice vs Toulouse | D (1-1) | H (71%) | H | H | 1.118 | **MISS** |
| `pred_015` | Arsenal vs Brighton | D (1-1) | H (52%) | H | H | 0.853 | **MISS** |
| `pred_018` | Bochum vs Borussia M.Gladbach | A (0-2) | A (49%) | A | A | 0.404 | **PASS** |
| `pred_021` | Real Madrid vs Real Betis | H (2-0) | H (53%) | H | H | 0.334 | **PASS** |
| `pred_024` | Genoa vs Roma | D (1-1) | A (65%) | A | A | 0.945 | **MISS** |
| `pred_027` | Verona vs Torino | A (2-3) | D (53%) | D | D | 0.950 | **MISS** |
| `pred_030` | Real Madrid vs Espanyol | H (4-1) | H (89%) | H | H | 0.021 | **PASS** |
| `pred_033` | Osasuna vs Las Palmas | H (2-1) | D (43%) | D | D | 0.554 | **MISS** |
| `pred_036` | Athletic Club vs Celta Vigo | H (3-1) | A (47%) | A | A | 1.022 | **MISS** |
| `pred_039` | Barcelona vs Getafe | H (1-0) | H (83%) | H | H | 0.053 | **PASS** |
| `pred_042` | Augsburg vs Borussia M.Gladbach | H (2-1) | A (67%) | A | A | 1.346 | **MISS** |
| `pred_045` | Las Palmas vs Celta Vigo | A (0-1) | A (68%) | A | A | 0.181 | **PASS** |

## 3. Metric Definitions & Formulas
- **Outcome Accuracy:** Exact match hit rate between predicted class (H, D, A) and official match result.
- **Multi-Class Brier Score:** $\frac{1}{N} \sum_{i=1}^N \sum_{k \in \{H,D,A\}} (p_{ik} - y_{ik})^2$. Penalizes poorly calibrated probability estimates.
- **Log Loss:** $-\frac{1}{N} \sum_{i=1}^N \ln(p_{i, \text{actual}})$. Measures probabilistic confidence penalization.
- **Consensus Rate:** Proportion of matches where the quantitative GNN and the generative LLM agree.