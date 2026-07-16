"""
End-to-end demo of the segmentation engagement, on synthetic B2B data.

    python main.py

Runs Phase 1 (demographic personas) plus the full validation battery, then a
compact Phase 3 synthesis against the behavioral engagement signal. Every
number printed is computed at run time - nothing here is hard-coded.
"""
import numpy as np
import pandas as pd

import data
import features
import cluster
import validate
import personas


def main():
    pd.set_option("display.width", 120)

    # ---- data -----------------------------------------------------------
    customers = data.make_customers(n=1200)
    usage = data.make_usage(customers)
    churn = data.make_churn(customers, usage)  # held out, never used to cluster

    print("=" * 68)
    print("PHASE 1  -  DEMOGRAPHIC PERSONAS")
    print("=" * 68)

    # ---- feature engineering + clustering -------------------------------
    X, names, _ = features.transform(customers)
    res = cluster.choose_k_and_fit(X)
    Z = res.pca.transform(X)

    print(f"PCA components kept        : {res.pca.n_components_}  "
          f"({res.pca.explained_variance_ratio_.sum():.0%} variance)")
    print(f"Silhouette by k            : {res.scores_by_k}")
    print(f"Chosen k                   : {res.k}  (silhouette {res.silhouette})")

    # ---- validation: beyond statistical separation ----------------------
    print("\n--- Cluster validation ---")
    stab = validate.bootstrap_stability(Z, res.labels, res.k, n_boot=100)
    print(f"Bootstrap stability (ARI)  : {stab['mean_ari']} +/- {stab['std_ari']}  "
          f"(100 resamples)")

    ext = validate.external_validity(res.labels, churn)
    print(f"External validity vs churn : chi2={ext['chi2']}, p={ext['p_value']:.2e}")
    print(f"Churn rate by cluster      : {ext['churn_rate_by_cluster']}")

    verdict = "PASS" if (stab["mean_ari"] >= 0.6 and ext["p_value"] < 0.05) else "REVIEW"
    print(f"Validity verdict           : {verdict} "
          "(stable under resampling AND predicts held-out churn)")

    # ---- personas -------------------------------------------------------
    print("\n--- Personas ---")
    profile = personas.profile_clusters(customers, res.labels)
    print(personas.render_cards(profile))

    # sanity check against the known latent groups (demo only)
    from sklearn.metrics import adjusted_rand_score
    recovery = adjusted_rand_score(customers["_latent_group"], res.labels)
    print(f"(demo check) ARI vs true latent groups: {recovery:.3f}")

    # ---- Phase 3 synthesis: demographic persona x behavior --------------
    print("\n" + "=" * 68)
    print("PHASE 3  -  SYNTHESIS: does demographics predict behaviour?")
    print("=" * 68)
    beh = usage.copy()
    # simple behavioral segment: engagement tertile (stand-in for Phase 2 model)
    beh["behavior_segment"] = pd.qcut(
        beh["pct_core_actions"], 3, labels=["at_risk", "core_loyalist", "power_adopter"]
    )
    synth = pd.crosstab(
        pd.Series(res.labels, name="demographic_persona"),
        beh["behavior_segment"],
        normalize="index",
    ).round(2)
    print(synth)
    print("\nRead: row = demographic persona, cell = share of that persona in each "
          "behavioral segment.\nThe off-diagonal concentrations are the persona "
          "hypotheses to test with surveys.")


if __name__ == "__main__":
    main()
