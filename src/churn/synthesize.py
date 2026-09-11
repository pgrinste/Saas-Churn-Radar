"""Synthetic-but-realistic B2B SaaS customer dataset.

Why synthetic: it keeps the repo self-contained (no license questions about
someone else's data) and fully reproducible from a seed. The feature set and
churn drivers mirror what we actually see in B2B SaaS: engagement decay,
underused seats, payment friction, support load. Swap in your own customers
via the same schema - nothing downstream cares where the rows came from.

Columns (one row per customer account):
  mrr                 monthly recurring revenue ($)
  plan_tier           0=starter, 1=growth, 2=scale
  months_active       tenure in months (1-60)
  usage_index         normalized product usage level (0-1)
  usage_trend_3m      change in usage over the last 3 months (-0.5..+0.5)
  support_tickets_90d tickets opened in the last 90 days
  payment_failures    failed card charges in the last 6 months (0-4)
  seat_utilization    fraction of purchased seats actively used (0-1)
  days_since_login    days since any user on the account logged in
  churned             label: 1 if the account cancelled within 30 days of renewal
"""

import numpy as np
import pandas as pd


def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    plan_tier = rng.choice([0, 1, 2], size=n, p=[0.5, 0.35, 0.15])
    base_mrr = {0: 290.0, 1: 890.0, 2: 2400.0}
    mrr = np.array([rng.lognormal(np.log(base_mrr[t]), 0.35) for t in plan_tier])

    months_active = rng.integers(1, 61, size=n).astype(float)
    usage_index = np.clip(rng.beta(2.2, 2.2, size=n), 0.02, 1.0)
    # engagement decay: a chunk of accounts are trending down hard
    decaying = rng.random(n) < 0.35
    usage_trend_3m = np.where(decaying, -rng.uniform(0.05, 0.45, n),
                              rng.normal(0.02, 0.10, n))
    usage_trend_3m = np.clip(usage_trend_3m, -0.5, 0.5)

    support_tickets_90d = rng.poisson(1.2 + 4.0 * decaying, size=n)
    payment_failures = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.78, 0.13, 0.05, 0.025, 0.015])
    seat_utilization = np.clip(rng.beta(2.5, 2.0, size=n) - 0.25 * decaying, 0.05, 1.0)
    days_since_login = np.where(decaying, rng.integers(9, 60, n),
                                rng.integers(0, 9, n)).astype(float)

    # latent churn risk from the drivers above + noise
    risk = (
        -0.75
        - 2.2 * usage_trend_3m
        -1.4 * seat_utilization
        + 0.55 * (support_tickets_90d / 6.0)
        + 0.8 * (payment_failures / 4.0)
        + 0.9 * np.clip(days_since_login / 30.0, 0, 1)
        - 0.5 * usage_index
        + 0.25 * (months_active < 6)          # early-tenure accounts are fragile
        + rng.normal(0, 0.9, n)
    )
    p_churn = 1.0 / (1.0 + np.exp(-risk))
    churned = (rng.random(n) < p_churn).astype(int)

    df = pd.DataFrame({
        "mrr": mrr.round(2),
        "plan_tier": plan_tier,
        "months_active": months_active.astype(int),
        "usage_index": usage_index.round(4),
        "usage_trend_3m": usage_trend_3m.round(4),
        "support_tickets_90d": support_tickets_90d,
        "payment_failures": payment_failures,
        "seat_utilization": seat_utilization.round(4),
        "days_since_login": days_since_login.astype(int),
        "churned": churned,
    })
    return df


FEATURES = [c for c in generate(10).columns if c != "churned"]
