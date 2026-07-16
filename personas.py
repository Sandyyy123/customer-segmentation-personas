"""
Cluster -> persona.

A cluster label is a number; a persona is a sentence a non-technical team can
act on. For each cluster we profile the distinctive features (how it differs
from the base rate) and emit a compact, human-readable card.
"""
import pandas as pd


def profile_clusters(customers: pd.DataFrame, labels) -> pd.DataFrame:
    df = customers.copy()
    df["cluster"] = labels
    num_cols = ["annual_income", "account_holder_age", "seats"]
    cat_cols = ["business_type", "firm_size", "region"]

    out = []
    for c, grp in df.groupby("cluster"):
        row = {"cluster": int(c), "n": len(grp), "pct_of_base": round(100 * len(grp) / len(df), 1)}
        for col in num_cols:
            row[f"{col}_median"] = round(float(grp[col].median()), 0)
        for col in cat_cols:
            row[f"{col}_top"] = grp[col].mode().iloc[0]
        out.append(row)
    return pd.DataFrame(out).sort_values("pct_of_base", ascending=False)


def name_persona(row) -> str:
    """Very light rule-of-thumb naming - in a real engagement the client names them."""
    firm = row["firm_size_top"]
    return {
        "large": "Corporate Wellness Buyer",
        "small": "Boutique Studio Owner",
        "individual": "Solo Trainer / Freelancer",
        "multi_site": "Multi-Location Operator",
    }.get(firm, f"Segment {row['cluster']}")


def render_cards(profile: pd.DataFrame) -> str:
    lines = []
    for _, r in profile.iterrows():
        lines.append(
            f"  Persona: {name_persona(r)}  ({r['pct_of_base']}% of base, n={r['n']})\n"
            f"    dominant business type : {r['business_type_top']}\n"
            f"    firm size              : {r['firm_size_top']}\n"
            f"    median income          : ${int(r['annual_income_median']):,}\n"
            f"    median age             : {int(r['account_holder_age_median'])}\n"
            f"    median seats           : {int(r['seats_median'])}\n"
        )
    return "\n".join(lines)
