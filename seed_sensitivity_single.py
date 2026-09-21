"""
SEED SENSITIVITY CHECK - v2-robust pipeline (single-seed, subprocess version)

Rewritten after seed_sensitivity.py died silently in the background with
zero output (strongly suspected OOM kill on a 3.8GB-RAM device sandbox: the
original script looped over 5 seeds in ONE process, doing dfm = df_model.copy()
plus SMOTE-ENN plus a 1000-tree XGBoost fit on every iteration with no memory
cleanup in between).

Fix: run ONE seed per process invocation (called 5x from the shell, once per
seed). Each process starts fresh and exits when done, so memory from one
seed can never accumulate into the next - the OS reclaims everything on
exit regardless of what pandas/sklearn/xgboost leaked internally. Also
trims df_model to only the columns actually needed (instead of copying all
26 original columns) to cut baseline memory further, and flushes prints
immediately (python -u) so if it dies again we at least see how far it got.

Usage: python3 seed_sensitivity_single.py <seed>
Appends one row of results to seed_sensitivity_results.csv (creates it with
a header on the first call).
"""

import sys
import os
import time
import gc
import warnings

warnings.filterwarnings("ignore")

if len(sys.argv) != 2:
    print("Usage: python3 seed_sensitivity_single.py <seed>")
    sys.exit(1)

SEED = int(sys.argv[1])
MISSING_FRAC = 0.08

DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_sample_500k_13pct.csv"
)
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/model")
RESULTS_CSV = f"{OUTPUT_DIR}/seed_sensitivity_results.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[seed={SEED}] starting, loading data...", flush=True)
t_start = time.time()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, average_precision_score
from imblearn.combine import SMOTEENN
import xgboost as xgb

COST_FEATURES = ["BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES"]
RATIO_FEATURE = ["CLAIM_TO_BASE_RATIO"]
CONTEXT_FEATURES = ["COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME"]
OTHER_NUMERICAL = ["AGE", "LENGTH_OF_STAY_HOURS"]  # PATIENT_INCOME removed - not applicable to unemployed/informal-sector patients
categorical_features = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]

# Only load the columns we actually use - cuts baseline memory vs. loading
# all 26 original columns (drops free-text/id fields not used as features).
NEEDED_COLS = list(set(
    COST_FEATURES + RATIO_FEATURE + OTHER_NUMERICAL + categorical_features
    + ["IS_FRAUD", "PRIMARY_CONDITION", "PROVIDER_ID", "SERVICE_DATE", "ENCOUNTER_ID"]
))
df = pd.read_csv(DATA_PATH, usecols=NEEDED_COLS)
print(f"[seed={SEED}] loaded {df.shape}, {time.time()-t_start:.0f}s", flush=True)

le = LabelEncoder()
for col in categorical_features:
    df[col + "_ENCODED"] = le.fit_transform(df[col].astype(str))
encoded_categoricals = [c + "_ENCODED" for c in categorical_features]
y_full = df["IS_FRAUD"]

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


idx_train, idx_temp = train_test_split(
    df.index, test_size=0.30, random_state=SEED, stratify=y_full
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, random_state=SEED, stratify=y_full.loc[idx_temp]
)

train_df = df.loc[idx_train]

group_cols = ["PRIMARY_CONDITION", "PATIENT_COUNTY"]
group_median = train_df.groupby(group_cols)["CLAIM_AMOUNT_KES"].median()
global_median = train_df["CLAIM_AMOUNT_KES"].median()
median_lookup = df[group_cols].apply(tuple, axis=1).map(group_median).fillna(global_median)
df["COST_RATIO"] = (df["CLAIM_AMOUNT_KES"] / median_lookup.replace(0, np.nan)).fillna(1.0)
del median_lookup

provider_mean = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].mean()
provider_std = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].std()
global_mean = train_df["CLAIM_AMOUNT_KES"].mean()
global_std = train_df["CLAIM_AMOUNT_KES"].std()
mean_lookup = df["PROVIDER_ID"].map(provider_mean).fillna(global_mean)
std_lookup = df["PROVIDER_ID"].map(provider_std).fillna(global_std).replace(0, global_std)
df["PROVIDER_Z_SCORE"] = ((df["CLAIM_AMOUNT_KES"] - mean_lookup) / std_lookup).fillna(0.0)
del mean_lookup, std_lookup

daily_counts = df.groupby(["PROVIDER_ID", "SERVICE_DATE"])["ENCOUNTER_ID"].transform("count")
df["PROVIDER_DAILY_VOLUME"] = daily_counts
del daily_counts, train_df
gc.collect()

X_train = df.loc[idx_train, FEATURE_COLUMNS]
X_val = df.loc[idx_val, FEATURE_COLUMNS]
X_test = df.loc[idx_test, FEATURE_COLUMNS]
y_train = y_full.loc[idx_train]
y_val = y_full.loc[idx_val]
y_test = y_full.loc[idx_test]

# Free the full dataframe now - only the sliced feature frames are needed
# from here on. This is the single biggest memory saving vs. the original
# script, which kept the full dfm copy alive throughout SMOTE-ENN + fit.
del df
gc.collect()
print(f"[seed={SEED}] features engineered, {time.time()-t_start:.0f}s, "
      f"train={len(X_train)} val={len(X_val)} test={len(X_test)}", flush=True)

smote_enn = SMOTEENN(random_state=SEED)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)
del smote_enn, X_train
gc.collect()
print(f"[seed={SEED}] SMOTE-ENN done, {time.time()-t_start:.0f}s, "
      f"resampled train={len(X_train_res)}", flush=True)

rng = np.random.default_rng(SEED)
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
    random_state=SEED, n_jobs=2, tree_method="hist",
)
model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False)
del X_train_res, y_train_res
gc.collect()
print(f"[seed={SEED}] model fit done (best_iteration={model.best_iteration}), "
      f"{time.time()-t_start:.0f}s", flush=True)

row = {"seed": SEED, "best_iteration": model.best_iteration}
for label, X_eval in [
    ("clean", X_test),
    ("noise_5pct", corrupt(X_test, 0.05, SEED)),
    ("noise_10pct", corrupt(X_test, 0.10, SEED)),
]:
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    row[f"{label}_f1"] = f1_score(y_test, y_pred)
    row[f"{label}_pr_auc"] = average_precision_score(y_test, y_proba)

print(f"seed={SEED}  clean PR-AUC={row['clean_pr_auc']:.4f} F1={row['clean_f1']:.4f}  "
      f"5%: PR-AUC={row['noise_5pct_pr_auc']:.4f} F1={row['noise_5pct_f1']:.4f}  "
      f"10%: PR-AUC={row['noise_10pct_pr_auc']:.4f} F1={row['noise_10pct_f1']:.4f}  "
      f"(total {time.time()-t_start:.0f}s)", flush=True)

row_df = pd.DataFrame([row])
write_header = not os.path.exists(RESULTS_CSV)
row_df.to_csv(RESULTS_CSV, mode="a", header=write_header, index=False)
print(f"[seed={SEED}] appended to {RESULTS_CSV}", flush=True)
