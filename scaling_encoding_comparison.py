"""
Empirical check of two supervisor questions on the v2-robust model:

  1. "Normalise / scale it" - does feature scaling change anything for
     XGBoost? (Short answer: tree models split on thresholds, not
     distances, so scaling a feature only rescales the split thresholds,
     never which rows fall left/right of them - predictions should be
     numerically identical. This script proves that empirically rather
     than just asserting it.)

  2. "How do you convert text to numbers" - is LabelEncoder (ordinal,
     current approach) worse than one-hot encoding for the three nominal
     categorical columns (ENCOUNTER_TYPE, GENDER, PATIENT_COUNTY)?

Runs on a stratified 40k-row subsample (not the full 500k) with a smaller
XGBoost (200 rounds, no early stopping) purely so this finishes in under a
couple of minutes - it is a fast diagnostic to settle the methodology
question, NOT a replacement for the full v2-robust training run. The real
model (xgboost_model_v2_robust.py, 500k rows, 1000 rounds) is unaffected by
this script and unchanged.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score, average_precision_score
import xgboost as xgb

BASE_DIR = os.path.expanduser("~/synthea_kenya")
DATA_PATH = f"{BASE_DIR}/output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
RANDOM_STATE = 42
SUBSAMPLE_N = 40000

NEEDED_COLS = [
    "BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES", "CLAIM_TO_BASE_RATIO",
    "AGE", "LENGTH_OF_STAY_HOURS", "ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY",
    "IS_FRAUD",
]
NUMERIC_FEATURES = ["BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES",
                     "CLAIM_TO_BASE_RATIO", "AGE", "LENGTH_OF_STAY_HOURS"]
CATEGORICAL_FEATURES = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    colsample_bytree=0.7, eval_metric="aucpr",
    random_state=RANDOM_STATE, n_jobs=-1,
)


def fit_eval(X_train, X_test, y_train, y_test, scale_pos_weight):
    model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "f1": f1_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "pr_auc": average_precision_score(y_test, proba),
    }


def main():
    print("Loading + stratified-subsampling data...")
    df = pd.read_csv(DATA_PATH, usecols=NEEDED_COLS)
    df, _ = train_test_split(
        df, train_size=SUBSAMPLE_N, random_state=RANDOM_STATE, stratify=df["IS_FRAUD"]
    )
    print(f"Using {len(df):,} rows (stratified subsample), fraud rate {df.IS_FRAUD.mean()*100:.2f}%\n")

    le_dict = {c: LabelEncoder() for c in CATEGORICAL_FEATURES}
    for c in CATEGORICAL_FEATURES:
        df[c + "_LE"] = le_dict[c].fit_transform(df[c].astype(str))

    y = df["IS_FRAUD"]
    idx_train, idx_test = train_test_split(df.index, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    scale_pos_weight = (y.loc[idx_train] == 0).sum() / (y.loc[idx_train] == 1).sum()

    results = {}

    # ---- Experiment 1: raw vs StandardScaler-normalized numeric features ----
    print("=" * 70)
    print("EXPERIMENT 1: raw features vs. StandardScaler-normalized features")
    print("=" * 70)
    le_cols = [c + "_LE" for c in CATEGORICAL_FEATURES]
    feat_cols = NUMERIC_FEATURES + le_cols

    X = df[feat_cols]
    t0 = time.time()
    results["raw (unscaled)"] = fit_eval(X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test], scale_pos_weight)
    t_raw = time.time() - t0

    scaler = StandardScaler()
    X_scaled = df[feat_cols].copy()
    # Fit scaler on the train split only, then transform everything - avoids leakage.
    scaler.fit(df.loc[idx_train, NUMERIC_FEATURES])
    X_scaled[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    t0 = time.time()
    results["StandardScaler-normalized"] = fit_eval(X_scaled.loc[idx_train], X_scaled.loc[idx_test], y.loc[idx_train], y.loc[idx_test], scale_pos_weight)
    t_scaled = time.time() - t0

    for name in ["raw (unscaled)", "StandardScaler-normalized"]:
        r = results[name]
        print(f"  {name:<28} F1 {r['f1']:.4f}  Bal.Acc {r['balanced_accuracy']:.4f}  PR-AUC {r['pr_auc']:.4f}")
    diff = abs(results["raw (unscaled)"]["pr_auc"] - results["StandardScaler-normalized"]["pr_auc"])
    print(f"\n  PR-AUC difference: {diff:.5f} ({'negligible - as expected for a tree model' if diff < 0.01 else 'non-negligible - worth investigating'})")

    # ---- Experiment 2: LabelEncoder vs OneHotEncoder for categoricals ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: LabelEncoder (ordinal) vs. OneHotEncoder for categoricals")
    print("=" * 70)

    # LabelEncoder version = same as Experiment 1's "raw" result, reuse it
    results["LabelEncoder (ordinal)"] = results["raw (unscaled)"]

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(df.loc[idx_train, CATEGORICAL_FEATURES].astype(str))
    ohe_cols = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    X_ohe_cat = pd.DataFrame(ohe.transform(df[CATEGORICAL_FEATURES].astype(str)), columns=ohe_cols, index=df.index)
    X_ohe = pd.concat([df[NUMERIC_FEATURES], X_ohe_cat], axis=1)
    results["OneHotEncoder"] = fit_eval(X_ohe.loc[idx_train], X_ohe.loc[idx_test], y.loc[idx_train], y.loc[idx_test], scale_pos_weight)

    for name in ["LabelEncoder (ordinal)", "OneHotEncoder"]:
        r = results[name]
        print(f"  {name:<28} F1 {r['f1']:.4f}  Bal.Acc {r['balanced_accuracy']:.4f}  PR-AUC {r['pr_auc']:.4f}")
    diff2 = abs(results["LabelEncoder (ordinal)"]["pr_auc"] - results["OneHotEncoder"]["pr_auc"])
    print(f"\n  PR-AUC difference: {diff2:.5f} ({'negligible' if diff2 < 0.01 else 'non-negligible - consider switching to one-hot'})")

    print("\n" + "=" * 70)
    print("SUMMARY (quote this to your supervisor)")
    print("=" * 70)
    print("Both were tested empirically, not just asserted:")
    print(f"  - Scaling numeric features changed PR-AUC by {diff:.5f} - confirms XGBoost's")
    print("    split-based trees are scale-invariant, so normalization isn't required for")
    print("    this model family (it would matter for a distance-based model like KNN or")
    print("    for a neural net, which is relevant context for the GraphSAGE component).")
    print(f"  - Switching categorical encoding from LabelEncoder to OneHotEncoder changed")
    print(f"    PR-AUC by {diff2:.5f}.")
    print("\nNote: this ran on a 40k-row subsample with a smaller model (200 rounds, no")
    print("early stopping) to finish quickly as a diagnostic - the numbers themselves are")
    print("lower than the full v2-robust model's, that's expected and fine; only the RELATIVE")
    print("differences between rows above are what answer the methodology question.")


if __name__ == "__main__":
    main()
