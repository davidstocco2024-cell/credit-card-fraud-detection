# Business Requirements Document (BRD)

**Project:** Credit Card Fraud Detection: ULB Dataset
**Deliverable:** 01, Stage 1: Business Understanding
**Repository path:** `reports/01_business_requirements_document.md`
**Version:** 1.0
**Date:** 2026-11-18
**Status:** Approved for Stage 2 (Data Understanding)
**Related config:** `config/ulb.yaml`

| Role | Name |
|---|---|
| Author / Data Scientist | Ignacio David Stocco |
| Business stakeholder (simulated) | Head of Fraud Operations |
| Technical reviewer (simulated) | Risk Analytics Lead |

---

## 1. Executive Summary and System Purpose

### 1.1 Summary

Card-not-present fraud is a low-frequency, high-impact event. In the reference dataset (ULB / Worldline-ULB Machine Learning Group, released on Kaggle), only **492 of 284,807 transactions (0.172%)** are fraudulent. Every fraud that goes undetected becomes a direct financial loss (chargeback plus the original amount). Every legitimate transaction that is blocked creates operating cost and damages the customer relationship.

This project designs, builds and documents an end-to-end **fraud scoring system** that ranks incoming transactions by fraud risk so that a limited team of human analysts reviews the transactions that matter most.

### 1.2 Purpose of the system

The system is **not** an automatic judge that blocks cards on its own. It is a **prioritization engine** (decision support) with three goals:

1. **Rank** every transaction by estimated fraud probability.
2. **Concentrate** the real fraud in the top of the ranking, so the analysts' limited daily capacity is spent where expected loss is highest.
3. **Explain** each alert (feature attribution) so an analyst can act on it quickly and the decision can be audited.

### 1.3 Project scope

| In scope | Out of scope |
|---|---|
| Supervised binary classification (`Class`: 0 = legitimate, 1 = fraud) | Real-time streaming infrastructure |
| Imbalance-aware evaluation (PR-AUC, Precision@K, cost-based metrics) | Chargeback and dispute workflow |
| Threshold optimization driven by business cost | Production deployment on real card traffic |
| Explainability (SHAP) and drift monitoring design | Regulatory certification (PCI-DSS, model risk management sign-off) |
| Reproducible pipeline (`src/`, `config/`, `tests/`, `scripts/`) | Customer-facing communication flows |
| Local demo app (`streamlit_app/`) | |

### 1.4 Dataset snapshot (as provided)

| Attribute | Value |
|---|---|
| Source | Credit Card Fraud Detection (ULB), Kaggle |
| File | `data/raw/creditcard.csv` (gitignored) |
| Transactions | 284,807 |
| Frauds (`Class = 1`) | 492 |
| Fraud prevalence | 0.172% (about 1 fraud per 579 transactions) |
| Features | `Time` (seconds elapsed since the first transaction), `Amount`, `V1` to `V28` (PCA components, anonymized), target `Class` |
| Time span | About 2 days of European cardholder transactions |

> **Note on anonymization.** `V1`–`V28` are the output of a PCA transformation applied by the data provider, so their business meaning is not recoverable. Only `Time` and `Amount` are interpretable. This limits feature engineering and domain explainability, and it is documented as a known limitation (Section 5).

### 1.5 Business Context

Fraud is a large and persistent cost for organizations and economies. Published estimates from different sources and years illustrate the scale:

- A typical organization loses about **5% of its revenue** to occupational fraud each year (ACFE, *Report to the Nations*).[^1]
- Non-health insurance fraud in the United States is estimated at more than **$40 billion per year** (FBI).[^2]
- Fraud was estimated to cost the United Kingdom **£73 billion per year** (National Fraud Authority).
- Card issuers lose approximately **seven cents per every hundred dollars** of transactions to fraud (Schrage, Money Crashers, 2012), about 0.07% of transaction value.

> **Note.** These figures cover different types of fraud, regions and years, so they are not directly comparable. They are included to show that fraud is financially material even when its per-transaction rate is very low. This is the core of the "needle in a haystack" problem addressed in this project: in the ULB dataset, fraud is 0.172% of transactions by count, so detection must be both extremely selective and economically meaningful.

[^1]: ACFE, *Report to the Nations*, https://www.acfe.com/report-to-the-nations
[^2]: FBI, *Insurance Fraud*, https://www.fbi.gov/stats-services/publications/insurance-fraud

---

## 2. Business Problem Definition

### 2.1 The "needle in a haystack" problem

Out of every ~579 transactions, about one is fraudulent, and nothing in the raw record flags it. Three properties make the problem hard:

- **Extreme class imbalance.** A model can look excellent by ignoring the minority class entirely. Naive training with default settings tends to collapse towards predicting "legitimate" for everything.
- **Adversarial behavior.** Fraudsters adapt. Patterns learned today decay over time (concept drift), so a static model degrades silently unless monitored.
- **Rare labels, delayed labels.** Confirmed fraud labels often arrive days or weeks later (customer disputes), so the training signal is both scarce and late.

### 2.2 The operational bottleneck: human analyst capacity

Reviewing an alert is a manual task: open the case, inspect the transaction history, contact the customer, decide. Capacity is therefore **fixed and small relative to volume**.

Illustrative scale based on this dataset (about 142,000 transactions per day):

| Strategy | What happens | Consequence |
|---|---|---|
| Review everything | 142,000 reviews/day | Impossible: no team can do this |
| Review at random | At 0.172% prevalence, about 1 in 579 reviewed transactions is fraud | Almost all analyst effort is wasted |
| Rule-based alerts only | Static thresholds (e.g. "amount > X") | High false positive volume; easy for fraudsters to evade |
| **Model-ranked queue (this project)** | Analysts review the top-K highest-risk transactions | Review effort concentrated where fraud actually is |

The core business question is therefore **not** "is this transaction fraud?" but:

> **Given that analysts can review only K transactions per period, which K transactions should they review to prevent the maximum amount of fraud loss?**

This reframing drives every downstream decision: the evaluation metrics (Section 4), the threshold strategy, and the format of the model's output (a ranked, explained alert queue rather than a binary label).

### 2.3 Business objectives

| ID | Objective | Description |
|---|---|---|
| BO-1 | Reduce fraud losses | Detect as much fraudulent value as possible within the analyst review budget |
| BO-2 | Control operating cost | Keep the alert queue manageable and precise so review hours are not wasted |
| BO-3 | Protect customer experience | Minimize unnecessary blocks and verification friction on legitimate customers |
| BO-4 | Auditability | Every alert must be explainable and every result reproducible from versioned code and config |

---

## 3. Asymmetric Cost Analysis

The two error types are **not** equally costly, so a model tuned for symmetric error (as accuracy implicitly does) optimizes the wrong objective.

### 3.1 False Negative (C_FN): undetected fraud

A fraudulent transaction is approved. The loss scales with the value of the transaction:

```
C_FN(i) = Amount_i + C_chargeback + C_investigation + C_reputation
```

| Component | Nature | Dependence on the transaction |
|---|---|---|
| `Amount_i` | Direct loss (goods/cash not recoverable, or refunded to customer) | **Variable**: directly the `Amount` feature |
| `C_chargeback` | Network and issuer chargeback fees | Mostly fixed per event |
| `C_investigation` | Post-fraud case handling, disputes | Mostly fixed per event |
| `C_reputation` | Customer churn / trust erosion after being defrauded | Hard to measure; treated as a fixed uplift parameter |

Key implication: **C_FN is heterogeneous.** Missing a fraud of 2,000 is far more damaging than missing a fraud of 5. A model (and an evaluation protocol) that treats every fraud as worth the same ignores most of the economics. This is why the project also tracks **value-weighted** metrics (Section 4.4), not only count-based ones.

### 3.2 False Positive (C_FP): a legitimate transaction flagged

A legitimate transaction is sent to review or blocked:

```
C_FP(i) = C_review + C_friction
```

| Component | Nature | Description |
|---|---|---|
| `C_review` | Operational, **fixed per alert** | Analyst time (minutes per case multiplied by loaded hourly cost) |
| `C_friction` | Customer-facing, **mostly fixed** | Declined purchase, SMS/call verification, abandoned cart, support contacts, risk of the customer moving to another card |

Key implication: **C_FP is small per event but multiplied by enormous volume.** With 99.83% of traffic being legitimate, even a tiny false positive rate produces thousands of unnecessary alerts, which consumes the finite analyst capacity described in Section 2.2. False positives are therefore costly twice: in direct cost and in **opportunity cost** (an analyst reviewing a false alert is not reviewing a real one).

### 3.3 Asymmetry and decision rule

The expected-cost decision rule for flagging transaction *i*, with model probability `p_i`, is:

```
Flag if:  p_i · C_FN(i)  >  (1 − p_i) · C_FP
```

Equivalently, the break-even probability is:

```
p*_i = C_FP / (C_FP + C_FN(i))
```

Because `C_FN(i)` depends on `Amount_i`, the optimal threshold is **amount-dependent**: high-value transactions justify flagging at a lower probability. This feeds directly into Stage 11 (`src/models/threshold.py`).

### 3.4 Cost parameters (to be defined in `config/ulb.yaml`)

The ULB dataset does not include operating costs, so these are **explicit, documented assumptions** rather than measured facts. They are placeholders to be calibrated and stress-tested in a sensitivity analysis:

| Parameter | Symbol | Illustrative starting value | Notes |
|---|---|---|---|
| Fixed cost per fraud event | `C_chargeback + C_investigation` | 15 (currency units) | Assumption |
| Reputation uplift | `C_reputation` | 0 in base case | Tested as a sensitivity scenario |
| Analyst review cost per alert | `C_review` | 5 | Assumption: minutes per case × loaded hourly cost |
| Customer friction cost per false alert | `C_friction` | 2 | Assumption |
| Review budget | `K` | 100 alerts per evaluation window | Capacity constraint |

> **Honesty clause.** These values are illustrative. The conclusions of the project are reported **as a function of the cost ratio** `C_FN / C_FP` through a sensitivity table, so they do not depend on one arbitrary number. The final report (`reports/final_report.md`) must show how the optimal threshold and expected savings move across plausible ratios.

### 3.5 Baseline for comparison

Every model is compared against two baselines on the same temporal test set:

1. **Do-nothing baseline:** no alerts; total loss equals the sum of fraud amounts plus fixed fraud costs.
2. **Random review baseline:** K random transactions reviewed; expected precision equals prevalence (about 0.17%).

---

## 4. Project Success Criteria

### 4.1 Why Accuracy is discarded

With a prevalence of 0.172%, a trivial classifier that labels **every** transaction as legitimate achieves:

```
Accuracy = 284,315 / 284,807 = 99.83%
```

while detecting **0 of 492 frauds** (recall = 0, precision undefined). Accuracy is dominated by the majority class and rewards doing nothing. It is also blind to asymmetric costs (Section 3). **Accuracy is not reported as a success metric**; if shown, it appears only to demonstrate this failure mode.

ROC-AUC is also treated with caution: with a very large negative class, the false positive rate stays small even when thousands of false alerts are generated, so ROC-AUC can look near-perfect while the alert queue is operationally useless. It is reported as a secondary, complementary metric only.

### 4.2 Primary metric: PR-AUC (Average Precision)

The Precision-Recall curve evaluates only the quality of the positive predictions, so it is not inflated by the abundant true negatives.

- **Baseline reference:** a random classifier has PR-AUC equal to the prevalence, about **0.0017**. Any value well above it is real signal.
- It summarizes the **precision/recall trade-off across all thresholds**, which makes it suitable for model selection and comparison before the operating threshold is chosen.
- Computed with `average_precision_score` (not trapezoidal interpolation, which can be over-optimistic).

### 4.3 Operational metric: Precision@K (K = 100)

PR-AUC is a global statistic; the business operates under a **hard review budget**. Precision@K measures exactly that scenario:

```
Precision@100 = (# true frauds among the 100 highest-scored transactions) / 100
```

- It answers: *"If my analysts review the top 100 alerts, how many are real fraud?"*
- It translates model quality directly into **analyst productivity**.
- It is insensitive to what the model does outside the top K, which is exactly the part the business cannot act on.

It is reported together with **Recall@K** (what share of all frauds is captured within the budget). Both are evaluated for several budgets (`K ∈ {50, 100, 200}`) to show how the system behaves if capacity changes. `K` is stored in `config/ulb.yaml`, not hardcoded.

### 4.4 Complementary metrics

| Metric | Purpose |
|---|---|
| **Value-weighted Recall@K** | Share of total fraud **amount** captured in the top K; reflects heterogeneous `C_FN` |
| **Expected cost / net savings** | Total cost under the chosen threshold vs. the baselines in Section 3.5 |
| **Precision, Recall, F2 at operating threshold** | Reported once the threshold is set (Stage 11); F2 weights recall higher than precision |
| **ROC-AUC** | Secondary, for comparability with the literature |
| **Calibration (Brier score, reliability curve)** | Required because the cost decision rule in 3.3 uses probabilities, not just rankings |

### 4.5 Quantitative acceptance criteria

Targets are defined **relative to baselines** and finalized after the Stage 2–4 data analysis. Initial targets:

| ID | Criterion | Target |
|---|---|---|
| SC-1 | PR-AUC on the held-out test set | Substantially above the 0.0017 random baseline and above a simple logistic regression baseline (target ≥ 0.75; to be revisited after the first baseline run) |
| SC-2 | Precision@100 | ≥ 0.80 (at least 80 of the top 100 alerts are real fraud, subject to the number of frauds in the test window) |
| SC-3 | Expected cost | Lower than both baselines in Section 3.5 under the base cost scenario and across the sensitivity range |
| SC-4 | Evaluation integrity | Temporal or pseudo-temporal split; **no leakage** (resampling and scaling fitted on training folds only); validated by `tests/test_features.py` |
| SC-5 | Reproducibility | Full pipeline runs end-to-end via `scripts/run_pipeline.py` from a clean environment with fixed seeds and pinned dependencies |
| SC-6 | Explainability | SHAP global and per-alert explanations available for the top-K queue |
| SC-7 | Monitoring design | Drift checks (PSI, KS) and rolling metrics specified in `src/monitoring/drift.py` |

> **Caveat on SC-2.** If the test window contains fewer than 100 frauds, Precision@100 has a mathematical ceiling below 1.0. In that case the report states the ceiling explicitly and uses `min(K, #frauds)` as the reference, so the criterion is never judged against an impossible target.

### 4.6 Evaluation principles

1. **Never evaluate on the training distribution.** Split first, then resample or scale inside the training fold only.
2. **Respect time.** `Time` ordering is used for a temporal (or pseudo-temporal) split to approximate how the model would face future data.
3. **Report uncertainty.** With so few positives, metrics are reported with bootstrap confidence intervals, not as single point estimates.
4. **Compare against baselines**, never in isolation.

---

## 5. Assumptions, Risks and Known Limitations

| # | Item | Impact | Mitigation |
|---|---|---|---|
| R-1 | Costs are assumed, not observed | Optimal threshold and savings estimates are assumption-dependent | Sensitivity analysis over the `C_FN / C_FP` ratio |
| R-2 | Only about 2 days of data | Concept drift cannot be studied in depth; temporal split is short | Document explicitly; design monitoring for production; validate approach on PaySim and IEEE-CIS (see `config/`) |
| R-3 | PCA-anonymized features | Limited domain feature engineering and business-readable explanations | Focus on `Time` / `Amount` engineering and on SHAP at component level, stated as a limitation |
| R-4 | Only 492 positive examples | High variance in every metric | Stratified, time-aware validation; bootstrap confidence intervals |
| R-5 | Public, older dataset (2013, European cardholders) | Patterns may not transfer to current fraud typologies | Position results as a methodology demonstration, not a production claim |
| R-6 | Label delay is not represented in the dataset | Real systems train on stale labels | Discuss in monitoring design and final report |

---

## 6. Traceability to Next Stages

| BRD section | Feeds into |
|---|---|
| 1.4 Dataset snapshot | Stage 2–3: `02_data_understanding.ipynb`, `03_data_quality.ipynb`, `src/data/validate.py` |
| 2.2 Review capacity / K | `config/ulb.yaml`, `src/models/evaluate.py` (Precision@K) |
| 3 Cost model | `src/models/threshold.py` (Stage 11), `reports/tables/` cost tables |
| 4 Success criteria | `src/models/evaluate.py`, `tests/test_models.py`, `reports/final_report.md` |
| 4.6 Evaluation principles | `src/data/split.py`, `src/features/audit.py` (leakage audit) |
| 5 Risks | `src/monitoring/drift.py`, final report limitations section |

---

## 7. Approval

| Role | Decision | Date |
|---|---|---|
| Business stakeholder | Approved: objectives and cost framework | 2026-11-18 |
| Technical reviewer | Approved: success criteria and evaluation protocol | 2026-11-18 |

*End of document: Deliverable 01.*