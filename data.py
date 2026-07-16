"""
Synthetic B2B subscription data generator.

Produces two tables that mirror the shape of a real engagement:

  1. customers  - one row per account, enriched with third-party demographics
                  (mixed categorical + continuous: firm size, business type,
                   household/firm income, account-holder age, region).
  2. usage      - one row per account of in-app behaviour
                  (feature adoption, session frequency, engagement depth).

A `churned` flag is generated but deliberately NOT used for clustering. It is
held out so we can test whether the discovered segments predict something they
never saw - the external-validity check in validate.py.

No client data is required to run this repo; everything here is synthetic.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

BUSINESS_TYPES = ["boutique_studio", "corporate_wellness", "solo_trainer", "franchise"]
REGIONS = ["Northeast", "Midwest", "South", "West"]


def _clip_int(x, lo, hi):
    return int(np.clip(round(x), lo, hi))


def make_customers(n: int = 1200) -> pd.DataFrame:
    """Four latent groups with overlapping, noisy demographics."""
    rows = []
    # latent mix - deliberately uneven so cluster sizes differ
    weights = [0.31, 0.24, 0.27, 0.18]
    groups = RNG.choice(len(BUSINESS_TYPES), size=n, p=weights)

    for g in groups:
        if g == 0:  # boutique studio owner
            income = RNG.normal(85_000, 20_000)
            age = RNG.normal(41, 8)
            seats = _clip_int(RNG.normal(4, 2), 1, 25)
            firm = "small"
        elif g == 1:  # corporate wellness buyer
            income = RNG.normal(160_000, 35_000)
            age = RNG.normal(46, 9)
            seats = _clip_int(RNG.normal(45, 25), 5, 400)
            firm = "large"
        elif g == 2:  # solo trainer / freelancer
            income = RNG.normal(52_000, 15_000)
            age = RNG.normal(34, 7)
            seats = 1
            firm = "individual"
        else:  # multi-location operator
            income = RNG.normal(110_000, 40_000)
            age = RNG.normal(44, 10)
            seats = _clip_int(RNG.normal(18, 10), 3, 120)
            firm = "multi_site"

        rows.append(
            dict(
                business_type=BUSINESS_TYPES[g],
                firm_size=firm,
                region=RNG.choice(REGIONS),
                annual_income=round(max(income, 18_000), -2),
                account_holder_age=_clip_int(age, 21, 70),
                seats=seats,
                _latent_group=g,  # ground truth, used only to sanity-check, never fed to the model
            )
        )
    df = pd.DataFrame(rows)
    df.insert(0, "customer_id", [f"C{100000 + i}" for i in range(len(df))])
    return df


def make_usage(customers: pd.DataFrame) -> pd.DataFrame:
    """Behaviour correlated with, but not identical to, the demographic group."""
    rows = []
    for _, c in customers.iterrows():
        g = c["_latent_group"]
        # base engagement by group, with heavy noise so behaviour != demographics
        base = {0: 0.55, 1: 0.75, 2: 0.30, 3: 0.50}[g]
        engagement = float(np.clip(RNG.normal(base, 0.22), 0.02, 1.0))
        rows.append(
            dict(
                customer_id=c["customer_id"],
                features_adopted=_clip_int(engagement * 12 + RNG.normal(0, 1.5), 0, 12),
                sessions_per_week=round(max(engagement * 9 + RNG.normal(0, 2), 0), 1),
                avg_session_min=round(max(engagement * 25 + RNG.normal(0, 6), 1), 1),
                pct_core_actions=round(float(np.clip(engagement + RNG.normal(0, 0.15), 0, 1)), 3),
                _engagement=engagement,
            )
        )
    return pd.DataFrame(rows)


def make_churn(customers: pd.DataFrame, usage: pd.DataFrame) -> pd.Series:
    """
    Held-out outcome. Lower engagement and solo/individual accounts churn more.
    This is NEVER used to build clusters - only to validate them afterwards.
    """
    merged = customers.merge(usage, on="customer_id")
    logit = (
        -0.4
        - 3.2 * merged["_engagement"]
        + 0.9 * (merged["firm_size"] == "individual").astype(float)
        + RNG.normal(0, 0.5, len(merged))
    )
    prob = 1 / (1 + np.exp(-logit))
    churned = (RNG.random(len(merged)) < prob).astype(int)
    return pd.Series(churned, index=merged.index, name="churned")


if __name__ == "__main__":
    cust = make_customers()
    use = make_usage(cust)
    churn = make_churn(cust, use)
    print(cust.drop(columns="_latent_group").head())
    print(use.drop(columns="_engagement").head())
    print("churn rate:", round(churn.mean(), 3))
