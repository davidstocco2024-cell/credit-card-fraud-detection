# Business Requirements Document (BRD)

**Project:** Credit Card Fraud Detection: ULB Dataset  
**Deliverable:** 01 — Stage 1: Business Understanding  
**Repository Path:** `reports/01_business_requirements_document.md`  
**Version:** 1.1  
**Date:** 2026-11-19  
**Status:** Pending Stage 1 Sign-off (Pre-Stage 2)  
**Related Config:** `config/ulb.yaml` / `src/business_config.py`

| Role | Name |
| :--- | :--- |
| Author / Data Scientist | Ignacio David Stocco |
| Business Stakeholder (Simulated) | Head of Fraud Operations |
| Technical Reviewer (Simulated) | Risk Analytics Lead |

---

## 1. Executive Summary, Purpose, and Project Scope

### 1.1 Executive Summary
Payment fraud is a low-frequency, high-impact adversarial problem. In the reference ULB dataset, only **492 of 284,807 transactions (0.1727%)** are fraudulent (~1 in 579 transactions). Every undetected fraud causes direct financial loss (unrecovered transaction amount plus chargeback fees), while every legitimate transaction mistakenly blocked or flagged creates operational review costs and customer friction (*false insult*).

### 1.2 System Purpose
The system is designed as a **risk-scoring and decision-support engine** rather than an opaque, fully autonomous card-blocking tool. Its core objectives are to:
1. Assign a calibrated fraud probability $\hat{p}$ to each incoming transaction.
2. Prioritize alerts by **Expected Loss** ($\hat{p} \times C_{FN}(\text{Amount})$) so limited analyst capacity focuses on transactions with the highest financial risk, complemented by automated mitigation rules for extreme-risk scores.
3. Provide technical feature attribution (SHAP) per alert to support analyst triage and model risk auditability.

### 1.3 Project Scope
| In Scope | Out of Scope |
| :--- | :--- |
| Supervised binary classification (`Class`: 0 = legitimate, 1 = fraud) | Real-time Kafka/streaming infrastructure |
| Temporal splitting & strict data leakage auditing | End-to-end chargeback & dispute case management |
| Imbalance-aware evaluation (PR-AUC, Precision@100, Recall@100) | Live production deployment on banking card rails |
| Cost-driven threshold optimization & financial ROI simulation | Formal regulatory certification (PCI-DSS / MRM sign-off) |
| SHAP explainability, drift monitoring design & local demo (`streamlit_app/`) | Customer-facing SMS/push notification workflows |

---

## 2. Stakeholders and Operational Fraud Cycle

Aligned with the *Fraud Cycle* (Detection $\rightarrow$ Investigation $\rightarrow$ Action $\rightarrow$ Feedback):
* **Fraud Operations Team (4 Analysts):** Consumes the daily Top-$K$ alert queue, verifies flagged transactions, and feeds confirmed fraud labels back into future retraining cycles.
* **Head of Fraud Operations:** Owns the trade-off between fraud losses, operational headcount, and customer friction (*False Insult Rate*).
* **Risk Analytics Lead:** Audits model generalization, leakage prevention, calibration, and ongoing drift monitoring.

---

## 3. Data Situation and Known Limitations

| Attribute | Value |
| :--- | :--- |
| Source | Credit Card Fraud Detection (ULB / Worldline), Kaggle |
| Local File | `data/raw/creditcard.csv` (gitignored) |
| Volume & Balance | 284,807 transactions; 492 frauds (**0.1727%** prevalence) |
| Features | `Time` (seconds since first tx), `Amount` (EUR), `V1`–`V28` (PCA components), `Class` |
| Time Span | ~2 days of European cardholder transactions |

**Known Technical Limitations:**
1. **PCA Anonymization (`V1`–`V28`):** The underlying business meaning of `V1`–`V28` cannot be recovered. This restricts domain-driven feature engineering (e.g., merchant velocity, geo-distance) and limits SHAP outputs to *technical attribution* rather than semantic fraud narratives.
2. **Short Observation Window (~48 hours):** Limits long-term seasonality and concept-drift testing on real multi-month shifts; drift will therefore be evaluated via controlled stress simulations in Stage 14.
3. **Missing Card/Customer IDs:** Prevents building entity-level historical aggregations or social network features without leaking future records.

---

## 4. Business Assumptions and Operational Constraints

| Type | Definition |
| :--- | :--- |
| **Operational Constraint** | Maximum manual review capacity is **$K = 100$ alerts/day** (4 analysts $\times$ 25 cases/day), representing **0.0702%** of ~142,404 daily transactions. |
| **Theoretical Ceiling** | With ~246 frauds/day on average, manual review alone caps `Recall@100` at **40.65%**, motivating a two-tier policy (Automated High-Confidence Action + Top-$K$ Manual Queue). |
| **Assumption 1 (Currency & Channel)** | Transaction `Amount` is denominated in EUR (€), and model thresholds apply uniformly across card channels since channel metadata (`POS` vs. `CNP`) is absent. |
| **Assumption 2 (Label Integrity)** | `Class = 1` reflects confirmed fraud after chargeback/investigation settlement; delayed label feedback is outside the static dataset scope. |

---

## 5. Asymmetric Financial Cost Model

Let $m$ denote transaction `Amount` (EUR). Error costs are modeled asymmetrically:

* **True Positive ($C_{TP}$):** Fraud is intercepted; only manual review cost is incurred:
  $$C_{TP} = C_{\text{review}} = \mathbf{€5.00}$$
* **False Positive ($C_{FP}$):** Manual review cost plus expected customer churn (*False Insult Rate*):
  $$C_{FP} = C_{\text{review}} + (p_{\text{churn}} \times \text{LTV}) = €5.00 + (0.02 \times €400.00) = \mathbf{€13.00}$$
* **False Negative ($C_{FN}$):** Unrecovered fraudulent amount (10% recovery assumed) plus fixed chargeback fee:
  $$C_{FN}(m) = 0.90 \cdot m + €15.00$$

**Decision Matrix:**

| Predicted \ Actual | Fraud ($Y = 1$) | Legitimate ($Y = 0$) |
| :--- | :--- | :--- |
| **Alert / Intervene ($\hat{Y} = 1$)** | **TP:** €5.00 cost (loss prevented) | **FP:** €13.00 cost (review + friction) |
| **Approve ($\hat{Y} = 0$)** | **FN:** $0.90m + €15.00$ loss | **TN:** €0.00 cost |

---

## 6. Success Criteria (Technical & Business)

### 6.1 Technical Success Criteria (Evaluated on Chronological Hold-Out Test Set)
| Metric | Purpose | Target |
| :--- | :--- | :--- |
| **PR-AUC / Average Precision** | Threshold-free summary of precision-recall trade-off under 0.172% imbalance | Significantly outperform Dummy Baseline ($\approx 0.0017$) and exceed **0.75** |
| **Precision@100** | Purity of the daily 100-alert analyst queue | **$\ge 75\%$** in the Top-100 risk-ranked transactions |
| **Recall@100 & Fraud Capture Rate** | Share of fraud cases and **EUR fraud volume** captured within budget | Maximize captured EUR volume over raw-count baselines |
| **Zero Data Leakage** | Strict temporal splitting; scalers and resampling (`SMOTE`, undersampling) confined strictly to `train` | **100% pass** on Stage 6 Leakage Audit |

### 6.2 Financial Success Criteria
1. **Positive Net Fraud Savings:** Total net savings (fraud prevented minus $C_{TP}$, $C_{FP}$, and residual $C_{FN}$) strictly greater than both the *Approve-All Baseline* and simple default `0.50` thresholding.
2. **Positive Operational ROI:** Net monetary savings clearly exceed analyst team operating costs.

---

## 7. Project Plan, Deliverables, and Governance

The project executes across **14 sequential, gated stages** (Business Understanding through Drift Monitoring). Progression between stages requires: (a) reproducible Python scripts in `src/`, (b) documented stage report in `reports/`, and (c) explicit verification against leakage and business constraints.