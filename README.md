# Credit Card Fraud Detection

End-to-end transactional fraud detection system with financial impact analysis.

## Business Problem

NovaPay, a fictional fintech, processes ~142,000 transactions per day with a 0.172% fraud rate. Without a model, the expected loss is **$13M per year**.

**Goal:** reduce fraud losses by at least 40% with Recall@200 ≥ 70%.

## Datasets (progressive iteration)

| # | Dataset | Focus |
|---|---------|-------|
| 1 | **ULB Credit Card Fraud** | Full methodology, extreme class imbalance |
| 2 | **PaySim** | Temporal split, behavioral feature engineering |
| 3 | **IEEE-CIS** | Real e-commerce data, high cardinality, SHAP |

## Methodology (14 stages)

| # | Stage | Status |
|---|-------|--------|
| 1 | Business Understanding | Done |
| 2 | Data Understanding | In progress |
| 3 | Data Quality | Pending |
| 4 | Exploratory Data Analysis | Pending |
| 5 | Feature Engineering | Pending |
| 6 | Data Leakage Prevention | Pending |
| 7 | Data Splitting | Pending |
| 8 | Class Imbalance | Pending |
| 9 | Model Development | Pending |
| 10 | Model Evaluation | Pending |
| 11 | Threshold Optimization | Pending |
| 12 | Financial Impact Analysis | Pending |
| 13 | Explainability | Pending |
| 14 | Deployment and Monitoring | Pending |

## Dataset Notes (ULB)

- 284,807 transactions, 492 of them fraudulent (0.172%).
- `Time`: seconds elapsed since the first transaction (not a real timestamp). The data covers roughly two days.
- `V1`–`V28`: PCA-transformed features. Their original meaning is not available, which limits interpretability.
- `Amount`: transaction amount.
- `Class`: target variable (0 = legitimate, 1 = fraud).
- Because of the extreme imbalance, accuracy is not used as the main metric. Evaluation relies on Precision, Recall, F1-score and PR-AUC.

## Results

| Dataset | PR-AUC | Recall@200 | Estimated Annual Savings |
|---------|--------|------------|--------------------------|
| ULB | TBD | TBD | TBD |
| PaySim | TBD | TBD | TBD |
| IEEE-CIS | TBD | TBD | TBD |

## Project Structure

```text
Credit-Card-Fraud-Detection/
├── config/
├── data/
│   └── raw/            # datasets are not versioned
├── notebooks/
├── src/
├── reports/
├── tests/
├── README.md
└── requirements.txt
```

## Getting Started

```bash
git clone <repository-url>
cd Credit-Card-Fraud-Detection

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/raw/`.

Then open `notebooks/02_data_understanding.ipynb`.

## Bibliography and References

The following references were consulted to establish the business requirements, define the project scope, identify operational constraints, and understand the fundamentals of fraud analytics.

### Articles



### Business Analysis

1. **Project Management Institute (PMI).** (2024). *Business Analysis for Practitioners: A Practice Guide* (Second Edition). Project Management Institute.
   - Focus: Needs Assessment, Requirements Analysis, Stakeholder Engagement, Traceability, and Solution Evaluation.
   - Website: https://www.pmi.org/

2. **International Institute of Business Analysis (IIBA).** *A Guide to the Business Analysis Body of Knowledge (BABOK Guide)*.
   - Focus: Strategy Analysis, Requirements Analysis, Requirements Life Cycle Management, and Solution Evaluation.
   - Website: https://www.iiba.org/

### Machine Learning




### Fraud Analytics

3. **Baesens, B., Van Vlasselaer, V., & Verbeke, W.** (2015). *Fraud Analytics Using Descriptive, Predictive, and Social Network Techniques*. Wiley.
   - Focus: Fraud detection and prevention, data-driven fraud detection, and the fraud analytics process model.
   - Publisher: https://www.wiley.com/

### Machine Learning for Finantial Fraud


Fight Fraud with Machine Learning

### Dataset

4. **ULB Machine Learning Group & Worldline.** *Credit Card Fraud Detection Dataset*.
   - Source: Kaggle.
   - Description: Credit card transactions with 284,807 records, including 492 frauds.
   - Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

5. **Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G.** (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE Symposium Series on Computational Intelligence (SSCI).
   - Focus: Original research paper associated with the ULB dataset.

---

**Note:** These references provide the theoretical and methodological foundation for the analytical decisions made throughout this project. They are complemented by practical experimentation and empirical evaluation.


