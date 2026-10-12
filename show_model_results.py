"""
Prints a clean terminal summary of the v2-robust XGBoost fraud model's
results, for showing a supervisor directly in the terminal.

Reads the CSVs already produced by the training/evaluation scripts
(xgboost_model_v2_robust.py, robustness_test.py, xgboost_v2_per_fraudtype.py,
shap_recheck_v2.py) rather than re-running them - retraining would take
several minutes and would overwrite the model file the live Streamlit app is
currently scoring claims against, which is the last thing you want mid-demo.
These numbers are exactly what's already saved under output/model/.

Usage:
    python3 show_model_results.py
"""

import os
import pandas as pd

BASE = os.path.expanduser("~/synthea_kenya/output/model")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    pd.set_option("display.width", 120)

    section("KENYA HEALTH CLAIM FRAUD DETECTION - XGBOOST v2 (ROBUST) MODEL")
    print("Missingness-trained, regularized, adaptive dual-threshold triage.")

    # --- Headline held-out performance -------------------------------
    section("1. HELD-OUT TEST SET PERFORMANCE (clean data)")
    r = pd.read_csv(f"{BASE}/model_results.csv").iloc[0]
    print(f"  F1 score            : {r.test_f1:.4f}")
    print(f"  Balanced accuracy   : {r.test_balanced_accuracy:.4f}")
    print(f"  PR-AUC              : {r.test_pr_auc:.4f}")
    print(f"  (validation set)    : F1 {r.val_f1:.4f} | Bal.Acc {r.val_balanced_accuracy:.4f} | PR-AUC {r.val_pr_auc:.4f}")

    # --- Robustness under noise ---------------------------------------
    section("2. ROBUSTNESS: v1 (baseline) vs v2 (robust) under noise + missingness")
    cmp = pd.read_csv(f"{BASE}/v1_vs_v2_comparison.csv")
    print(cmp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n  Why this matters: v1 was only ever trained on perfectly clean,")
    print("  exact-price data and collapsed under realistic messiness. v2 was")
    print("  deliberately trained with 8% missing cost values so XGBoost learns")
    print("  a sensible default split direction instead of assuming completeness.")

    # --- Per fraud type -------------------------------------------------
    section("3. RECALL BY FRAUD TYPE (v2 model)")
    ft = pd.read_csv(f"{BASE}/per_fraudtype_recall_v2data.csv")
    for _, row in ft.iterrows():
        print(f"  {row.fraud_type:<18} caught {int(row.caught):>5}/{int(row.n):<5}  "
              f"recall {pct(row.recall):>7}   mean score {row.mean_score:.3f}")
    print("\n  Note: collusion_ring is the hardest class to catch (0.67 recall) -")
    print("  expected, since it's a relational pattern (repeated provider-patient")
    print("  pairing) that a row-by-row model like XGBoost can only partially see")
    print("  via engineered features. This is the motivating gap for the planned")
    print("  GraphSAGE component (Prototype 2).")

    # --- Threshold strategy ---------------------------------------------
    section("4. ADAPTIVE DUAL-THRESHOLD TRIAGE (clean vs. noisy data)")
    th = pd.read_csv(f"{BASE}/threshold_tuning_results.csv")
    clean = th[th.strategy.str.startswith("clean")]
    print(clean[["strategy", "f1", "precision", "recall"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Explainability ---------------------------------------------------
    section("5. TOP FEATURE CONTRIBUTIONS (mean |SHAP value|)")
    shap_df = pd.read_csv(f"{BASE}/shap_importance_v2_robust.csv").sort_values("rank").head(6)
    for _, row in shap_df.iterrows():
        tag = " (contextual/engineered)" if row.is_contextual_feature else ""
        print(f"  {int(row['rank'])}. {row.feature:<22} {pct(row.pct_of_total / 100):>7} of total contribution{tag}")

    section("END OF SUMMARY")
    print("Full detail and per-claim explanations are in the live Streamlit")
    print("dashboard (Claim Detail page) and in output/model/*.csv / *.png.\n")


if __name__ == "__main__":
    main()
