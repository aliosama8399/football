# Football Match Prediction Evaluation Report

**Date Generated:** 2026-11-19 15:00:28  
**Matches Evaluated:** 15  
**Langfuse Logging:** Enabled  

## 1. Executive Summary Metrics

| Metric | Result | Benchmark / Target | Status |
| :--- | :---: | :---: | :---: |
| **Expert 1 (GNN) Accuracy** | **40.0%** | > 45.0% | WARN |
| **Expert 2 (SLM) Accuracy** | **40.0%** | > 45.0% | WARN |
| **Hybrid Ensemble Accuracy** | **40.0%** | > 50.0% | WARN |
| **Multi-Class Brier Score** | **0.5237** | < 0.6000 (Random: 0.667) | PASS |
| **Multi-Class Log Loss** | **0.8662** | < 1.0500 (Random: 1.098) | PASS |
| **GNN-SLM Consensus Rate** | **100.0%** | > 60.0% | PASS |
| **Tactical Fact Grounding** | **0.80** | > 0.80 | PASS |
| **Average Latency / Match** | **7.27s** | < 4.00s | WARN |

## 2. Match-by-Match Breakdown

| Match ID | Matchup | Actual | GNN Pred | SLM Pred | Ensemble | Brier Score | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `pred_003` | Auxerre vs Nice | H (2-1) | A (44%) | A | A | 0.981 | **MISS** |
| `pred_006` | Chelsea vs Manchester City | A (0-2) | A (61%) | A | A | 0.242 | **PASS** |
| `pred_009` | Espanyol vs Real Sociedad | A (0-1) | D (46%) | D | D | 0.732 | **MISS** |
| `pred_012` | Nice vs Toulouse | D (1-1) | D (42%) | D | D | 0.500 | **PASS** |
| `pred_015` | Arsenal vs Brighton | D (1-1) | A (41%) | A | A | 0.587 | **MISS** |
| `pred_018` | Bochum vs Borussia M.Gladbach | A (0-2) | D (47%) | D | D | 0.590 | **MISS** |
| `pred_021` | Real Madrid vs Real Betis | H (2-0) | H (75%) | H | H | 0.102 | **PASS** |
| `pred_024` | Genoa vs Roma | D (1-1) | A (51%) | A | A | 0.725 | **MISS** |
| `pred_027` | Verona vs Torino | A (2-3) | D (44%) | D | D | 0.715 | **MISS** |
| `pred_030` | Real Madrid vs Espanyol | H (4-1) | H (78%) | H | H | 0.086 | **PASS** |
| `pred_033` | Osasuna vs Las Palmas | H (2-1) | H (48%) | H | H | 0.452 | **PASS** |
| `pred_036` | Athletic Club vs Celta Vigo | H (3-1) | D (48%) | D | D | 0.915 | **MISS** |
| `pred_039` | Barcelona vs Getafe | H (1-0) | H (94%) | H | H | 0.006 | **PASS** |
| `pred_042` | Augsburg vs Borussia M.Gladbach | H (2-1) | D (44%) | D | D | 0.682 | **MISS** |
| `pred_045` | Las Palmas vs Celta Vigo | A (0-1) | D (46%) | D | D | 0.541 | **MISS** |

## 3. Metric Definitions & Formulas
- **Outcome Accuracy:** Exact match hit rate between predicted class (H, D, A) and official match result.
- **Multi-Class Brier Score:** $\frac{1}{N} \sum_{i=1}^N \sum_{k \in \{H,D,A\}} (p_{ik} - y_{ik})^2$. Penalizes poorly calibrated probability estimates.
- **Log Loss:** $-\frac{1}{N} \sum_{i=1}^N \ln(p_{i, \text{actual}})$. Measures probabilistic confidence penalization.
- **Consensus Rate:** Proportion of matches where the quantitative GNN and the generative LLM agree.