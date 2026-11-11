# Football Match Prediction Evaluation Report

**Date Generated:** 2026-11-05 20:00:44  
**Matches Evaluated:** 15  
**Langfuse Logging:** Enabled  

## 1. Executive Summary Metrics

| Metric | Result | Benchmark / Target | Status |
| :--- | :---: | :---: | :---: |
| **Expert 1 (GNN) Accuracy** | **60.0%** | > 45.0% | PASS |
| **Expert 2 (SLM) Accuracy** | **53.3%** | > 45.0% | PASS |
| **Hybrid Ensemble Accuracy** | **60.0%** | > 50.0% | PASS |
| **Multi-Class Brier Score** | **0.5093** | < 0.6000 (Random: 0.667) | PASS |
| **Multi-Class Log Loss** | **0.8659** | < 1.0500 (Random: 1.098) | PASS |
| **GNN-SLM Consensus Rate** | **80.0%** | > 60.0% | PASS |
| **Tactical Fact Grounding** | **0.97** | > 0.80 | PASS |
| **Average Latency / Match** | **13.64s** | < 4.00s | WARN |

## 2. Match-by-Match Breakdown

| Match ID | Matchup | Actual | GNN Pred | SLM Pred | Ensemble | Brier Score | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `pred_003` | Auxerre vs Nice | H (2-1) | H (48%) | H | H | 0.425 | **PASS** |
| `pred_006` | Chelsea vs Manchester City | A (0-2) | A (51%) | A | A | 0.376 | **PASS** |
| `pred_009` | Espanyol vs Real Sociedad | A (0-1) | D (47%) | H | D | 0.727 | **MISS** |
| `pred_012` | Nice vs Toulouse | D (1-1) | D (43%) | D | D | 0.486 | **PASS** |
| `pred_015` | Arsenal vs Brighton | D (1-1) | A (44%) | H | H | 0.666 | **MISS** |
| `pred_018` | Bochum vs Borussia M.Gladbach | A (0-2) | D (53%) | D | D | 0.810 | **MISS** |
| `pred_021` | Real Madrid vs Real Betis | H (2-0) | H (54%) | H | H | 0.323 | **PASS** |
| `pred_024` | Genoa vs Roma | D (1-1) | D (42%) | D | D | 0.511 | **PASS** |
| `pred_027` | Verona vs Torino | A (2-3) | D (45%) | D | D | 0.719 | **MISS** |
| `pred_030` | Real Madrid vs Espanyol | H (4-1) | H (74%) | A | H | 0.114 | **PASS** |
| `pred_033` | Osasuna vs Las Palmas | H (2-1) | H (51%) | H | H | 0.404 | **PASS** |
| `pred_036` | Athletic Club vs Celta Vigo | H (3-1) | A (48%) | A | A | 1.188 | **MISS** |
| `pred_039` | Barcelona vs Getafe | H (1-0) | H (91%) | H | H | 0.016 | **PASS** |
| `pred_042` | Augsburg vs Borussia M.Gladbach | H (2-1) | D (44%) | D | D | 0.564 | **MISS** |
| `pred_045` | Las Palmas vs Celta Vigo | A (0-1) | A (58%) | A | A | 0.313 | **PASS** |

## 3. Metric Definitions & Formulas
- **Outcome Accuracy:** Exact match hit rate between predicted class (H, D, A) and official match result.
- **Multi-Class Brier Score:** $\frac{1}{N} \sum_{i=1}^N \sum_{k \in \{H,D,A\}} (p_{ik} - y_{ik})^2$. Penalizes poorly calibrated probability estimates.
- **Log Loss:** $-\frac{1}{N} \sum_{i=1}^N \ln(p_{i, \text{actual}})$. Measures probabilistic confidence penalization.
- **Consensus Rate:** Proportion of matches where the quantitative GNN and the generative LLM agree.