"""
XGBOOST FRAUD MODEL - v2 "ROBUST" VERSION

Built in response to the fragility found by robustness_test.py: the original
model (PR-AUC 0.9896 clean) collapsed to 0.79 / 0.58 PR-AUC under 5% / 10%
noise+missingness, because it had only ever seen perfectly clean, exact
Synthea price-sheet values.

Three changes, applied together:

  STEP 1 - Train with missingness
    After SMOTE-ENN resampling, 8% of values in every cost-related column
    of the TRAINING set are randomly set to NaN before XGBoost ever sees
    them. XGBoost's native missing-value handling (`missing=np.nan`, the
    default) then has to learn a "default direction" for every tree split
    on those columns, instead of assuming they're always present.
    (Missingness is injected AFTER SMOTE-ENN, not before, because SMOTE's
    nearest-neighbour interpolation cannot operate on NaNs - imputing
    before resampling would defeat the purpose.)

  STEP 2 - Contextual/relative cost features (supplementing, not replacing,
  the raw cost columns, so we can compare like-for-like against v1)
    - COST_RATIO: CLAIM_AMOUNT_KES / the median claim amount for the same
      (PRIMARY_CONDITION, PATIENT_COUNTY) pair, computed from the TRAINING
      split only (mapped onto val/test - no test-set leakage) with the
      training-wide median as a fallback for unseen combinations.
    - PROVIDER_Z_SCORE: (CLAIM_AMOUNT_KES - provider_mean) / provider_std,
      with provider_mean/std computed from the TRAINING split only.
    - PROVIDER_DAILY_VOLUME: count of claims by the same PROVIDER_ID on the
      same SERVICE_DATE. NOTE: SERVICE_DATE in this dataset is date-only
      (fix_dataset.py truncated the original timestamp), so this is a
      same-calendar-day claim count, not a true rolling 24-hour window -
      that would need the un-truncated Synthea timestamps, which is a
      follow-up data-pipeline fix, not something this script can recover.
      This one uses the full dataset (train+val+test) since it only counts
      co-occurring rows and never touches the label - no leakage.

  STEP 3 - Regularized hyperparameters
    max_depth 6->4, colsample_bytree 0.8->0.7, learning_rate 0.1->0.05,
    scale_pos_weight kept DYNAMIC (computed from the actual post-SMOTE-ENN
    class balance, same approach as v1) rather than the fixed 6.7 suggested
    - 6.7 assumes exactly the raw 87/13 ratio, but the ratio that matters is
      the resampled *training* ratio, which SMOTE-ENN changes.
    Because learning_rate dropped 2x, n_estimators is raised to 1000 with
    early_stopping_rounds=30 (on the clean validation set) so the slower
    learner still gets enough rounds to converge without hand-picking a
    fixed tree count.

Evaluated on: clean test set, and the SAME 5%/10% noise+missingness
corruption used in robustness_test.py, so the numbers are directly
comparable to the v1 baseline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score, average_precision_score, classification_report
from imblearn.combine import SMOTEENN
import xgboost as xgb
import warnings
import os
import gc

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MISSING_FRAC = 0.08  # 8% - middle of the requested 5-10% range

DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_sample_500k_13pct.csv"
)
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("XGBOOST FRAUD MODEL - v2 ROBUST")
print("=" * 60)

# ============================================================
# STEP 0 - LOAD + BASE ENCODING (same as v1)
# ============================================================
print("\nLoading data...")
NEEDED_COLS = list(set([
    "BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES", "CLAIM_TO_BASE_RATIO",
    "AGE", "LENGTH_OF_STAY_HOURS", "ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY",
    "IS_FRAUD", "PRIMARY_CONDITION", "PROVIDER_ID", "SERVICE_DATE", "ENCOUNTER_ID",
]))
df = pd.read_csv(DATA_PATH, usecols=NEEDED_COLS)
print(f"Dataset shape: {df.shape}, fraud rate: {df['IS_FRAUD'].mean()*100:.2f}%")

categorical_features = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]
le = LabelEncoder()
# No df.copy() here - df is not reused after this, so mutating it in place
# (instead of holding two full 500k-row frames at once) avoids doubling
# peak memory on the 3.8GB device sandbox.
df_model = df
for col in categorical_features:
    df_model[col + "_ENCODED"] = le.fit_transform(df_model[col].astype(str))
encoded_categoricals = [c + "_ENCODED" for c in categorical_features]

# ============================================================
# STEP 0b - SPLIT FIRST (identical random_state/proportions to v1,
# so row membership matches exactly and results are comparable)
# ============================================================
y_full = df_model["IS_FRAUD"]
idx_train, idx_temp = train_test_split(
    df_model.index, test_size=0.30, random_state=RANDOM_STATE, stratify=y_full
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_full.loc[idx_temp]
)
print(f"Train/Val/Test sizes: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

# ============================================================
# STEP 2 - CONTEXTUAL FEATURES (fit on TRAIN only, applied everywhere)
# ============================================================
print("\nEngineering contextual cost features (fit on train split only)...")

train_df = df_model.loc[idx_train]

# --- Regional cost anomaly ratio ---
group_cols = ["PRIMARY_CONDITION", "PATIENT_COUNTY"]
group_median = train_df.groupby(group_cols)["CLAIM_AMOUNT_KES"].median()
global_median = train_df["CLAIM_AMOUNT_KES"].median()

median_lookup = df_model[group_cols].apply(tuple, axis=1).map(group_median)
median_lookup = median_lookup.fillna(global_median)
df_model["COST_RATIO"] = (df_model["CLAIM_AMOUNT_KES"] / median_lookup.replace(0, np.nan)).fillna(1.0)

# --- Provider peer deviation (z-score) ---
provider_mean = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].mean()
provider_std = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].std()
global_mean = train_df["CLAIM_AMOUNT_KES"].mean()
global_std = train_df["CLAIM_AMOUNT_KES"].std()

mean_lookup = df_model["PROVIDER_ID"].map(provider_mean).fillna(global_mean)
std_lookup = df_model["PROVIDER_ID"].map(provider_std).fillna(global_std).replace(0, global_std)
df_model["PROVIDER_Z_SCORE"] = ((df_model["CLAIM_AMOUNT_KES"] - mean_lookup) / std_lookup).fillna(0.0)

# --- Time-based volatility (same-day claim count per provider) ---
# Approximation: SERVICE_DATE is date-only in this dataset (see docstring),
# so this is same-calendar-day volume rather than a true rolling 24h window.
daily_counts = df_model.groupby(["PROVIDER_ID", "SERVICE_DATE"])["ENCOUNTER_ID"].transform("count")
df_model["PROVIDER_DAILY_VOLUME"] = daily_counts

print("New features added: COST_RATIO, PROVIDER_Z_SCORE, PROVIDER_DAILY_VOLUME")
print(df_model[["COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME"]].describe())

# ============================================================
# FEATURE SET
# ============================================================
COST_FEATURES = ["BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES"]
RATIO_FEATURE = ["CLAIM_TO_BASE_RATIO"]
CONTEXT_FEATURES = ["COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME"]
OTHER_NUMERICAL = ["AGE", "LENGTH_OF_STAY_HOURS"]  # PATIENT_INCOME removed - not applicable to unemployed/informal-sector patients

FEATURE_COLUMNS = (
    COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + encoded_categoricals
)
# Columns that get missingness injected during training (cost-related only -
# demographics, county, encounter type and daily volume stay intact)
NOISY_COLUMNS = COST_FEATURES + RATIO_FEATURE + ["COST_RATIO", "PROVIDER_Z_SCORE"]

print(f"\nFeatures used ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

X_train = df_model.loc[idx_train, FEATURE_COLUMNS]
X_val = df_model.loc[idx_val, FEATURE_COLUMNS]
X_test = df_model.loc[idx_test, FEATURE_COLUMNS]
y_train = y_full.loc[idx_train]
y_val = y_full.loc[idx_val]
y_test = y_full.loc[idx_test]

# Free the full dataframe now that the feature slices are carved out - it's
# not needed again, and holding it alive through SMOTE-ENN + training was
# the likely cause of a silent OOM kill on this 3.8GB sandbox.
del df_model, df, train_df, median_lookup, mean_lookup, std_lookup, daily_counts
gc.collect()

# ============================================================
# SMOTE-ENN (on clean values, as in v1)
# ============================================================
print("\nApplying SMOTE-ENN rebalancing...")
smote_enn = SMOTEENN(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)
print(f"Before: {y_train.value_counts().to_dict()}  After: {pd.Series(y_train_res).value_counts().to_dict()}")

# ============================================================
# STEP 1 - INJECT MISSINGNESS INTO THE RESAMPLED TRAINING SET
# (after SMOTE-ENN, since SMOTE can't interpolate across NaNs)
# ============================================================
print(f"\nInjecting {MISSING_FRAC*100:.0f}% missingness into training cost columns: {NOISY_COLUMNS}")
rng = np.random.default_rng(RANDOM_STATE)
X_train_res = X_train_res.reset_index(drop=True)
for col in NOISY_COLUMNS:
    mask = rng.random(len(X_train_res)) < MISSING_FRAC
    X_train_res.loc[mask, col] = np.nan
missing_pct_check = X_train_res[NOISY_COLUMNS].isna().mean() * 100
print("Actual missing % achieved per column:")
print(missing_pct_check)

# ============================================================
# STEP 3 - TRAIN WITH REGULARIZED HYPERPARAMETERS
# ============================================================
print("\nTraining XGBoost (regularized config)...")
neg = sum(y_train_res == 0)
pos = sum(y_train_res == 1)
scale_pos_weight = neg / pos  # dynamic - see docstring for why not the fixed 6.7
print(f"Dynamic scale_pos_weight: {scale_pos_weight:.3f}")

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.7,
    learning_rate=0.05,
    n_estimators=1000,
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=2, tree_method="hist",
)
model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False)
print(f"Best iteration: {model.best_iteration} (of 1000 max)")

# ============================================================
# EVALUATION HELPERS
# ============================================================
def evaluate(X, y, label):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    f1 = f1_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    pr_auc = average_precision_score(y, y_proba)
    print(f"\n{label}")
    print(f"F1: {f1:.4f}  Balanced Accuracy: {bal_acc:.4f}  PR-AUC: {pr_auc:.4f}")
    return {"experiment": label, "f1": f1, "balanced_accuracy": bal_acc, "pr_auc": pr_auc}


def corrupt(X, noise_frac, seed):
    """Identical corruption function to robustness_test.py, extended to the
    new engineered cost-context columns so the comparison against v1 is
    apples-to-apples (a real messy claims feed would corrupt these too,
    since they're derived from the same raw cost fields)."""
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


# ============================================================
# RESULTS
# ============================================================
print(f"\n{'='*60}\nRESULTS - v2 ROBUST MODEL\n{'='*60}")

results = []
results.append(evaluate(X_test, y_test, "v2 - CLEAN test set"))

for noise_frac in [0.05, 0.10]:
    X_noisy = corrupt(X_test, noise_frac, seed=RANDOM_STATE)
    results.append(
        evaluate(X_noisy, y_test, f"v2 - {int(noise_frac*100)}% noise+missingness test set")
    )

print(f"\n{'='*60}\nTest Classification Report (clean test set)\n{'='*60}")
print(classification_report(y_test, model.predict(X_test), target_names=["Legitimate", "Fraud"]))

# ============================================================
# COMPARISON TABLE (v1 numbers hardcoded from the earlier verified run)
# ============================================================
v1_reference = {
    "v1 - CLEAN test set": {"f1": 0.9770, "balanced_accuracy": 0.9819, "pr_auc": 0.9896},
    "v1 - 5% noise+missingness test set": {"f1": 0.5934, "balanced_accuracy": 0.8822, "pr_auc": 0.7924},
    "v1 - 10% noise+missingness test set": {"f1": 0.4569, "balanced_accuracy": 0.8095, "pr_auc": 0.5845},
}

print(f"\n{'='*60}\nv1 (fragile) vs v2 (robust) COMPARISON\n{'='*60}")
comparison_rows = []
for v1_label, v2_row in zip(v1_reference.keys(), results):
    v1_row = v1_reference[v1_label]
    comparison_rows.append({
        "condition": v1_label.replace("v1 - ", ""),
        "v1_pr_auc": v1_row["pr_auc"],
        "v2_pr_auc": v2_row["pr_auc"],
        "v1_f1": v1_row["f1"],
        "v2_f1": v2_row["f1"],
    })
comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))

# ============================================================
# SAVE
# ============================================================
model.save_model(f"{OUTPUT_DIR}/xgboost_fraud_model_v2_robust.json")
comparison_df.to_csv(f"{OUTPUT_DIR}/v1_vs_v2_comparison.csv", index=False)
pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/v2_robust_results.csv", index=False)
print(f"\nModel saved to {OUTPUT_DIR}/xgboost_fraud_model_v2_robust.json")
print(f"Comparison saved to {OUTPUT_DIR}/v1_vs_v2_comparison.csv")
