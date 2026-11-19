"""
Stage 1: Business Understanding
Project directory setup, asymmetric cost model (EUR), and operational capacity config
for the Credit Card Fraud Detection (ULB) portfolio project.
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


def initialize_project_structure(base_dir: str = ".") -> None:
    """Create the standard directory layout for the fraud detection repository."""
    folders = [
        "config",
        "data/raw",
        "data/processed",
        "docs",
        "models",
        "notebooks",
        "reports/figures",
        "scripts",
        "src",
        "tests",
    ]
    root = Path(base_dir)
    for folder in folders:
        (root / folder).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Project directory structure verified at: {root.resolve()}")


@dataclass(frozen=True)
class FraudBusinessConfig:
    """
    Asymmetric financial cost parameters (EUR) and daily operational constraints.
    Currency is EUR as the ULB dataset reflects European cardholder transactions.
    """
    currency: str = "EUR"

    # Operational & Customer Friction Costs
    analyst_review_cost: float = 5.0       # Manual investigation cost per alert (TP or FP)
    false_insult_churn_prob: float = 0.02  # Probability of customer churn after FP (2%)
    customer_ltv: float = 400.0            # Average Customer Lifetime Value (EUR)

    # Undetected Fraud Costs (False Negative)
    chargeback_admin_fee: float = 15.0     # Fixed administrative fee per chargeback (EUR)
    loss_recovery_rate: float = 0.10       # Average recovery rate on fraudulent amount (10%)

    # Fraud Operations Team Capacity
    num_analysts: int = 4
    alerts_per_analyst_per_day: int = 25   # Daily review budget K = 100 alerts/day

    @property
    def max_daily_alerts(self) -> int:
        """Operational alert budget K for Precision@K and Recall@K evaluation."""
        return self.num_analysts * self.alerts_per_analyst_per_day

    @property
    def cost_tp(self) -> float:
        """Cost of a True Positive: fraud prevented, manual review cost incurred."""
        return self.analyst_review_cost

    @property
    def cost_fp(self) -> float:
        """Expected cost of a False Positive: manual review + expected churn loss."""
        return self.analyst_review_cost + (self.false_insult_churn_prob * self.customer_ltv)

    def cost_fn(self, transaction_amount: float) -> float:
        """Expected cost of a False Negative given transaction amount."""
        unrecovered_loss = transaction_amount * (1.0 - self.loss_recovery_rate)
        return unrecovered_loss + self.chargeback_admin_fee

    def theoretical_breakeven_prob(self, transaction_amount: float) -> float:
        """
        Bayes Minimum Risk threshold p*(m): minimum fraud probability required
        for an alert to yield positive expected financial value vs. approving.
        """
        c_tp = self.cost_tp
        c_fp = self.cost_fp
        c_fn = self.cost_fn(transaction_amount)

        net_fraud_savings = c_fn - c_tp
        if net_fraud_savings <= 0:
            return 1.0  # Investigating costs more than the potential fraud loss
        return c_fp / (net_fraud_savings + c_fp)


def simulate_cost_sensitivity(config: FraudBusinessConfig) -> pd.DataFrame:
    """Simulate asymmetric cost ratios and theoretical thresholds across amounts."""
    sample_amounts = [20, 50, 100, 250, 500, 1000, 2500]
    records = []

    for amount in sample_amounts:
        c_fn = config.cost_fn(amount)
        c_fp = config.cost_fp
        p_star = config.theoretical_breakeven_prob(amount)
        records.append({
            "Amount_EUR": amount,
            "Cost_TP": config.cost_tp,
            "Cost_FP": round(c_fp, 2),
            "Cost_FN": round(c_fn, 2),
            "FN_to_FP_Ratio": round(c_fn / c_fp, 2),
            "Breakeven_p*": round(p_star, 4),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    initialize_project_structure()
    biz_cfg = FraudBusinessConfig()

    print("\n--- BUSINESS PARAMETERS SUMMARY ---")
    print(f"Max Daily Manual Alerts (K): {biz_cfg.max_daily_alerts} alerts/day")
    print(f"Cost per True Positive:      €{biz_cfg.cost_tp:.2f}")
    print(f"Cost per False Positive:     €{biz_cfg.cost_fp:.2f}")

    df_sim = simulate_cost_sensitivity(biz_cfg)
    print("\n--- ASYMMETRIC COST & BAYES THRESHOLD SENSITIVITY ---")
    print(df_sim.to_string(index=False))