"""
SEED SENSITIVITY CHECK - v2-robust pipeline

Every result reported so far (PR-AUC 0.9889 clean, F1 0.9744, etc.) comes
from a SINGLE random seed (42): one particular train/val/test split, one
particular SMOTE-ENN resample, one particular missingness mask, one
particular XGBoost fit. A single seed can't tell us whether those numbers
are a stable property of the model/pipeline or partly luck of that one
split. This script re-runs the IDENTICAL v2-robust pipeline
(xgboost_model_v2_robust.py) end-to-end across several different seeds,
varying every seed-dependent step together (split, feature-lookup fitting,
SMOTE-ENN, missingness injection, model training), and reports the
mean +/- std of PR-AUC/F1 across seeds for clean/5%-noise/10%-noise test
conditions - giving a confidence range instead of a single point estimate.

Uses the SAME dataset the deployed model was trained on
(master_claims_sample_500k_13pct.csv - no collusion_ring), so this is a
direct sensitivity check on the model actually backing the dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, average_precision_score
from imblearn.combine import SMOTEENN
import xgboost as xgb
import warnings
import os
import time

warnings.filterwarnings("ignore")

SEEDS = [42, 1, 7, 123, 2024]
MISSING_FRAC = 0.08

DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_sample_500k_13pct.csv"
)
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data (shared across all seeds; only splitting/resampling/training varies)...")
df = pd.read_csv(DATA_PATH)

categorical_features = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]
le = LabelEncoder()
df_model = df.copy()
for col in categorical_features:
    df_model[col + "_ENCODED"] = le.fit_transform(df_model[col].astype(str))
encoded_categoricals = [c + "_ENCODED" for c in categorical_features]
y_full = df_model["IS_FRAUD"]

COST_FEATURES = ["BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES"]
RATIO_FEATURE = ["CLAIM_TO_BASE_RATIO"]
CONTEXT_FEATURES = ["COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME"]
OTHER_NUMERICAL = ["AGE", "PATIENT_INCOME", "LENGTH_OF_STAY_HOURS"]
FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + encoded_categoricals
NOISY_COLUMNS = COST_FEATURES + RATIO_FEATURE + ["COST_RATIO", "PROVIDER_Z_SCORE"]


def corrupt(X, noise_frac, seed):
    r = np.random.default_rng(seed)
    Xc = X.copy()
    numeric_cols = [c for c in FEATURE_COLUMNS if c not in encoded_categoricals]
    for col in numeric_cols:
        std = Xc[col].std()
        noise = r.normal(loc=0, scale=std * noise_frac, size=len(Xc))
        Xc[col] = Xc[col] + noise
    mask = r.random(Xc.shape) < noise_frac
    Xc = Xc.mask(mask)
    return Xc


def run_seed(seed):
    t0 = time.time()
    idx_train, idx_temp = train_test_split(
        df_model.index, test_size=0.30, random_state=seed, stratify=y_full
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=seed, stratify=y_full.loc[idx_temp]
    )

    dfm = df_model.copy()
    train_df = dfm.loc[idx_train]

    group_cols = ["PRIMARY_CONDITION", "PATIENT_COUNTY"]
    group_median = train_df.groupby(group_cols)["CLAIM_AMOUNT_KES"].median()
    global_median = train_df["CLAIM_AMOUNT_KES"].median()
    median_lookup = dfm[group_cols].apply(tuple, axis=1).map(group_median).fillna(global_median)
    dfm["COST_RATIO"] = (dfm["CLAIM_AMOUNT_KES"] / median_lookup.replace(0, np.nan)).fillna(1.0)

    provider_mean = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].mean()
    provider_std = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].std()
    global_mean = train_df["CLAIM_AMOUNT_KES"].mean()
    global_std = train_df["CLAIM_AMOUNT_KES"].std()
    mean_lookup = dfm["PROVIDER_ID"].map(provider_mean).fillna(global_mean)
    std_lookup = dfm["PROVIDER_ID"].map(provider_std).fillna(global_std).replace(0, global_std)
    dfm["PROVIDER_Z_SCORE"] = ((dfm["CLAIM_AMOUNT_KES"] - mean_lookup) / std_lookup).fillna(0.0)

    daily_counts = dfm.groupby(["PROVIDER_ID", "SERVICE_DATE"])["ENCOUNTER_ID"].transform("count")
    dfm["PROVIDER_DAILY_VOLUME"] = daily_counts

    X_train = dfm.loc[idx_train, FEATURE_COLUMNS]
    X_val = dfm.loc[idx_val, FEATURE_COLUMNS]
    X_test = dfm.loc[idx_test, FEATURE_COLUMNS]
    y_train = y_full.loc[idx_train]
    y_val = y_full.loc[idx_val]
    y_test = y_full.loc[idx_test]

    smote_enn = SMOTEENN(random_state=seed)
    X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

    rng = np.random.default_rng(seed)
    X_train_res = X_train_res.reset_index(drop=True)
    for col in NOISY_COLUMNS:
        mask = rng.random(len(X_train_res)) < MISSING_FRAC
        X_train_res.loc[mask, col] = np.nan

    neg = sum(y_train_res == 0)
    pos = sum(y_train_res == 1)
    scale_pos_weight = neg / pos

    model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        max_depth=4, subsample=0.8, colsample_bytree=0.7, learning_rate=0.05,
        n_estimators=1000, early_stopping_rounds=30,
        random_state=seed, n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False)

    row = {"seed": seed, "best_iteration": model.best_iteration}
    for label, X_eval in [
        ("clean", X_test),
        ("noise_5pct", corrupt(X_test, 0.05, seed)),
        ("noise_10pct", corrupt(X_test, 0.10, seed)),
    ]:
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1]
        row[f"{label}_f1"] = f1_score(y_test, y_pred)
        row[f"{label}_pr_auc"] = average_precision_score(y_test, y_proba)

    print(f"seed={seed}  clean PR-AUC={row['clean_pr_auc']:.4f} F1={row['clean_f1']:.4f}  "
          f"5%: PR-AUC={row['noise_5pct_pr_auc']:.4f} F1={row['noise_5pct_f1']:.4f}  "
          f"10%: PR-AUC={row['noise_10pct_pr_auc']:.4f} F1={row['noise_10pct_f1']:.4f}  "
          f"({time.time()-t0:.0f}s)")
    return row


results = [run_seed(s) for s in SEEDS]
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/seed_sensitivity_results.csv", index=False)

print(f"\n{'='*70}\nSUMMARY ACROSS {len(SEEDS)} SEEDS ({SEEDS})\n{'='*70}")
summary = {}
for label in ["clean", "noise_5pct", "noise_10pct"]:
    for metric in ["pr_auc", "f1"]:
        col = f"{label}_{metric}"
        summary[col] = {"mean": results_df[col].mean(), "std": results_df[col].std()}
        print(f"{col:20s}  mean={results_df[col].mean():.4f}  std={results_df[col].std():.4f}  "
              f"min={results_df[col].min():.4f}  max={results_df[col].max():.4f}")

pd.DataFrame(summary).T.to_csv(f"{OUTPUT_DIR}/seed_sensitivity_summary.csv")
print(f"\nSaved per-seed results to {OUTPUT_DIR}/seed_sensitivity_results.csv")
print(f"Saved summary to {OUTPUT_DIR}/seed_sensitivity_summary.csv")
