# Football Match Prediction Evaluation Report

**Date Generated:** 2026-10-20 06:07:27  
**Matches Evaluated:** 15  
**Langfuse Logging:** Enabled  

## 1. Executive Summary Metrics

| Metric | Result | Benchmark / Target | Status |
| :--- | :---: | :---: | :---: |
| **Expert 1 (GNN) Accuracy** | **73.3%** | > 45.0% | PASS |
| **Expert 2 (SLM) Accuracy** | **73.3%** | > 45.0% | PASS |
| **Hybrid Ensemble Accuracy** | **73.3%** | > 50.0% | PASS |
| **Multi-Class Brier Score** | **0.4651** | < 0.6000 (Random: 0.667) | PASS |
| **Multi-Class Log Loss** | **0.8055** | < 1.0500 (Random: 1.098) | PASS |
| **GNN-SLM Consensus Rate** | **100.0%** | > 60.0% | PASS |
| **Tactical Fact Grounding** | **0.96** | > 0.80 | PASS |
| **Average Latency / Match** | **22.66s** | < 4.00s | WARN |

## 2. Match-by-Match Breakdown

| Match ID | Matchup | Actual | GNN Pred | SLM Pred | Ensemble | Brier Score | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `pred_003` | Auxerre vs Nice | H (2-1) | H (61%) | H | H | 0.243 | **PASS** |
| `pred_006` | Chelsea vs Manchester City | A (0-2) | A (73%) | A | A | 0.119 | **PASS** |
| `pred_009` | Espanyol vs Real Sociedad | A (0-1) | A (46%) | A | A | 0.467 | **PASS** |
| `pred_012` | Nice vs Toulouse | D (1-1) | D (44%) | D | D | 0.487 | **PASS** |
| `pred_015` | Arsenal vs Brighton | D (1-1) | H (57%) | H | H | 0.853 | **MISS** |
| `pred_018` | Bochum vs Borussia M.Gladbach | A (0-2) | A (49%) | A | A | 0.403 | **PASS** |
| `pred_021` | Real Madrid vs Real Betis | H (2-0) | H (72%) | H | H | 0.127 | **PASS** |
| `pred_024` | Genoa vs Roma | D (1-1) | D (38%) | D | D | 0.581 | **PASS** |
| `pred_027` | Verona vs Torino | A (2-3) | A (45%) | A | A | 0.498 | **PASS** |
| `pred_030` | Real Madrid vs Espanyol | H (4-1) | H (85%) | H | H | 0.039 | **PASS** |
| `pred_033` | Osasuna vs Las Palmas | H (2-1) | D (44%) | D | D | 0.864 | **MISS** |
| `pred_036` | Athletic Club vs Celta Vigo | H (3-1) | H (61%) | H | H | 0.246 | **PASS** |
| `pred_039` | Barcelona vs Getafe | H (1-0) | H (72%) | H | H | 0.129 | **PASS** |
| `pred_042` | Augsburg vs Borussia M.Gladbach | H (2-1) | D (40%) | D | D | 0.859 | **MISS** |
| `pred_045` | Las Palmas vs Celta Vigo | A (0-1) | D (49%) | D | D | 1.061 | **MISS** |

## 3. Metric Definitions & Formulas
- **Outcome Accuracy:** Exact match hit rate between predicted class (H, D, A) and official match result.
- **Multi-Class Brier Score:** $\frac{1}{N} \sum_{i=1}^N \sum_{k \in \{H,D,A\}} (p_{ik} - y_{ik})^2$. Penalizes poorly calibrated probability estimates.
- **Log Loss:** $-\frac{1}{N} \sum_{i=1}^N \ln(p_{i, \text{actual}})$. Measures probabilistic confidence penalization.
- **Consensus Rate:** Proportion of matches where the quantitative GNN and the generative LLM agree.