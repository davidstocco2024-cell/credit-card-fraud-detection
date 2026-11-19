# Credit Card Fraud Detection

Sistema end-to-end de detección de fraude transaccional con análisis de impacto financiero.

## Problema
Fintech ficticia NovaPay procesa ~142,000 transacciones/día con 0.172% de fraude.
Pérdida esperada sin modelo: **$13M/año**. Objetivo: reducir ≥40% con Recall@200 ≥70%.

## Datasets (iteración progresiva)
1. **ULB Credit Card Fraud** — metodología completa, desbalance extremo.
2. **PaySim** — split temporal real, feature engineering de comportamiento.
3. **IEEE-CIS** — e-commerce real, alta cardinalidad, SHAP.

## Metodología (14 etapas)
[enlace a docs o lista]

## Resultados
| Dataset | PR-AUC | Recall@200 | Ahorro anual estimado |
|---|---|---|---|
| ULB | TBD | TBD | TBD |
| PaySim | TBD | TBD | TBD |
| IEEE-CIS | TBD | TBD | TBD |

## Cómo correr
[instrucciones]


## Bibliography and References

The following references were consulted during the Business Understanding stage to establish the business requirements, define the project scope, identify operational constraints, and understand the fundamentals of fraud analytics.

### Business Analysis

1. **Project Management Institute (PMI).** (2024). *Business Analysis for Practitioners: A Practice Guide* (Second Edition). Project Management Institute.
   - Focus: Needs Assessment, Requirements Analysis, Stakeholder Engagement, Traceability, and Solution Evaluation.
   - Website: https://www.pmi.org/

2. **International Institute of Business Analysis (IIBA).** *A Guide to the Business Analysis Body of Knowledge (BABOK Guide)*.
   - Focus: Strategy Analysis, Requirements Analysis, Requirements Life Cycle Management, and Solution Evaluation.
   - Website: https://www.iiba.org/

### Fraud Analytics

3. **Baesens, B., Van Vlasselaer, V., & Verbeke, W.** (2015). *Fraud Analytics Using Descriptive, Predictive, and Social Network Techniques*. Wiley.
   - Focus: Fraud Detection and Prevention, Data-Driven Fraud Detection, Fraud Analytics Techniques, and the Fraud Analytics Process Model.
   - Publisher: https://www.wiley.com/

### Dataset

4. **ULB Machine Learning Group & Worldline.** *Credit Card Fraud Detection Dataset*.
   - Source: Kaggle.
   - Description: Historical credit card transactions dataset containing 284,807 transactions, including 492 fraudulent transactions.
   - Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

**Note:** These references provide the theoretical and methodological foundation for the business requirements and analytical decisions made throughout this project. They are complemented by practical experimentation and empirical evaluation using the selected dataset.