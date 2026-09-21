"""
PROVIDER-PATIENT PAIR FREQUENCY - ablation test on the v2 (collusion_ring)
dataset.

Idea: collusion_ring fraud is a repeated-relationship pattern (the same
provider and the same patient billing together far more often than chance),
not a cost anomaly - which is exactly why the current model (which sees
each claim independently, no relational features) recalls it at only
~67% while upcoding/phantom_billing sit at 90%+. A same-(PROVIDER_ID,
PATIENT_ID)-pair claim count is the cheapest possible relational signal a
row-wise tabular model can be given - not a real graph feature, but a
weak proxy for "these two show up together suspiciously often" that
doesn't require GraphSAGE.

This script trains the IDENTICAL v2-robust pipeline (no PATIENT_INCOME,
matching the just-retrained deployed model) twice on
master_claims_sample_500k_13pct_v2.csv (the dataset that HAS collusion_ring
- note this is NOT the same file the currently-deployed dashboard model was
trained on, which has no collusion_ring at all):

  baseline  - the current feature set (12 features)
  pairfreq  - the current feature set + PROVIDER_PATIENT_PAIR_FREQUENCY (13)

and reports overall PR-AUC/F1 plus per-fraud-type recall for both, so we
can see directly whether the new feature moves collusion_ring recall.

Usage: python3 pair_frequency_ablation.py <baseline|pairfreq>
Appends one row to pair_frequency_ablation_results.csv.
"""

import sys
import os
import gc
import time
import warnings

warnings.filterwarnings("ignore")

if len(sys.argv) != 2 or sys.argv[1] not in ("baseline", "pairfreq"):
    print("Usage: python3 pair_frequency_ablation.py <baseline|pairfreq>")
    sys.exit(1)

VARIANT = sys.argv[1]
RANDOM_STATE = 42
MISSING_FRAC = 0.08

DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
)
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/model")
RESULTS_CSV = f"{OUTPUT_DIR}/pair_frequency_ablation_results.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[{VARIANT}] starting, loading data...", flush=True)
t0 = time.time()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, average_precision_score
from imblearn.combine import SMOTEENN
import xgboost as xgb

categorical_features = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]
NEEDED_COLS = list(set([
    "BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES", "CLAIM_TO_BASE_RATIO",
    "AGE", "LENGTH_OF_STAY_HOURS", "ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY",
    "IS_FRAUD", "PRIMARY_CONDITION", "PROVIDER_ID", "PATIENT_ID", "SERVICE_DATE",
    "ENCOUNTER_ID", "FRAUD_TYPE",
]))
df = pd.read_csv(DATA_PATH, usecols=NEEDED_COLS)
print(f"[{VARIANT}] loaded {df.shape}, {time.time()-t0:.0f}s", flush=True)

le = LabelEncoder()
for col in categorical_features:
    df[col + "_ENCODED"] = le.fit_transform(df[col].astype(str))
encoded_categoricals = [c + "_ENCODED" for c in categorical_features]
y_full = df["IS_FRAUD"]

idx_train, idx_temp = train_test_split(
    df.index, test_size=0.30, random_state=RANDOM_STATE, stratify=y_full
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_full.loc[idx_temp]
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

COST_FEATURES = ["BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES"]
RATIO_FEATURE = ["CLAIM_TO_BASE_RATIO"]
CONTEXT_FEATURES = ["COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME"]
OTHER_NUMERICAL = ["AGE", "LENGTH_OF_STAY_HOURS"]  # PATIENT_INCOME removed, matches deployed model
FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + encoded_categoricals
NOISY_COLUMNS = COST_FEATURES + RATIO_FEATURE + ["COST_RATIO", "PROVIDER_Z_SCORE"]

if VARIANT == "pairfreq":
    # Same-(PROVIDER_ID, PATIENT_ID)-pair claim count, full dataset (like
    # PROVIDER_DAILY_VOLUME: just a co-occurrence count, never touches the
    # label, so no leakage even though it isn't train-split-only).
    pair_counts = df.groupby(["PROVIDER_ID", "PATIENT_ID"])["ENCOUNTER_ID"].transform("count")
    df["PROVIDER_PATIENT_PAIR_FREQUENCY"] = pair_counts
    del pair_counts
    FEATURE_COLUMNS = FEATURE_COLUMNS + ["PROVIDER_PATIENT_PAIR_FREQUENCY"]
    print(f"[{VARIANT}] pair-frequency stats:\n{df['PROVIDER_PATIENT_PAIR_FREQUENCY'].describe()}", flush=True)

print(f"[{VARIANT}] features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}", flush=True)

X_train = df.loc[idx_train, FEATURE_COLUMNS]
X_val = df.loc[idx_val, FEATURE_COLUMNS]
X_test = df.loc[idx_test, FEATURE_COLUMNS]
y_train = y_full.loc[idx_train]
y_val = y_full.loc[idx_val]
y_test = y_full.loc[idx_test]
fraud_type_test = df.loc[idx_test, "FRAUD_TYPE"]

del df
gc.collect()
print(f"[{VARIANT}] features engineered, {time.time()-t0:.0f}s", flush=True)

smote_enn = SMOTEENN(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)
del smote_enn, X_train
gc.collect()
print(f"[{VARIANT}] SMOTE-ENN done, {time.time()-t0:.0f}s, resampled train={len(X_train_res)}", flush=True)

rng = np.random.default_rng(RANDOM_STATE)
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
    random_state=RANDOM_STATE, n_jobs=2, tree_method="hist",
)
model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False)
del X_train_res, y_train_res
gc.collect()
print(f"[{VARIANT}] model fit done (best_iteration={model.best_iteration}), {time.time()-t0:.0f}s", flush=True)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
overall_f1 = f1_score(y_test, y_pred)
overall_pr_auc = average_precision_score(y_test, y_proba)
print(f"[{VARIANT}] Overall Test F1={overall_f1:.4f}  PR-AUC={overall_pr_auc:.4f}", flush=True)

rows = []
for ftype in ["upcoding", "phantom_billing", "collusion_ring"]:
    mask = (fraud_type_test == ftype).values
    n = int(mask.sum())
    caught = int(y_pred[mask].sum())
    recall = caught / n if n > 0 else float("nan")
    mean_score = float(y_proba[mask].mean()) if n > 0 else float("nan")
    print(f"[{VARIANT}] {ftype:20s} n={n:6d} caught={caught:6d} recall={recall:.4f} mean_score={mean_score:.4f}", flush=True)
    rows.append({
        "variant": VARIANT, "fraud_type": ftype, "n": n, "caught": caught,
        "recall": recall, "mean_score": mean_score,
        "overall_f1": overall_f1, "overall_pr_auc": overall_pr_auc,
    })

out_df = pd.DataFrame(rows)
write_header = not os.path.exists(RESULTS_CSV)
out_df.to_csv(RESULTS_CSV, mode="a", header=write_header, index=False)
print(f"[{VARIANT}] appended to {RESULTS_CSV}, total time {time.time()-t0:.0f}s", flush=True)
