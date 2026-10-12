"""
XGBOOST FRAUD MODEL - v2 "ROBUST" VERSION

Built in response to the fragility found by robustness_test.py: the original
model (PR-AUC 0.9896 clean) collapsed to 0.79 / 0.58 PR-AUC under 5% / 10%
noise+missingness, because it had only ever seen perfectly clean, exact
Synthea price-sheet values.

Three changes, applied together:

  STEP 1 - Train with missingness
    After resampling, 8% of values in every cost-related column of the
    TRAINING set are randomly set to NaN before XGBoost ever sees them.
    XGBoost's native missing-value handling (`missing=np.nan`, the default)
    then has to learn a "default direction" for every tree split on those
    columns, instead of assuming they're always present.
    (Missingness is injected AFTER resampling, not before, because SMOTE's
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
    scale_pos_weight kept DYNAMIC (computed from the actual post-resampling
    class balance, same approach as v1) rather than the fixed 6.7 suggested
    - 6.7 assumes exactly the raw 87/13 ratio, but the ratio that matters is
      the resampled *training* ratio, which resampling changes.
    Because learning_rate dropped 2x, n_estimators is raised to 1000 with
    early_stopping_rounds=30 (on the clean validation set) so the slower
    learner still gets enough rounds to converge without hand-picking a
    fixed tree count.

REVISION (v2.1) - three changes applied together:
  - DATA SOURCE: now reads master_claims_sample_500k_13pct_v2.csv instead of
    the older master_claims_sample_500k_13pct.csv. The older file predates
    inject_collusion_ring.py and contains ONLY upcoding/phantom_billing
    fraud - zero collusion_ring examples. The model trained on it had
    therefore never seen a single collusion-ring claim during training,
    which is the real reason collusion_ring recall looked weak in earlier
    evaluations (those numbers came from a different, throwaway model
    trained directly on the v2 file in xgboost_v2_per_fraudtype.py, which
    this script's saved model never matched).
  - ENCODING: ENCOUNTER_TYPE / GENDER / PATIENT_COUNTY switched from
    LabelEncoder (ordinal - imposes a fake numeric order on categories with
    no real order) to one-hot encoding. An empirical A/B test on a
    subsample showed one-hot gives a real PR-AUC improvement (~0.02) over
    LabelEncoder for this feature set, consistent with the standard
    guidance against ordinal-encoding nominal categories.
  - RESAMPLING METHOD: SMOTE-ENN -> SMOTENC (+ oversampling only, no ENN
    cleanup step). Plain SMOTE/SMOTEENN was designed for continuous data;
    running it on one-hot-encoded categoricals (32 dummy columns here,
    mostly from the 20-county PATIENT_COUNTY field) pushes the feature
    space to 41 dimensions, and sklearn's k-NN search degrades sharply in
    that many dimensions - empirically, SMOTEENN on this one-hot feature
    set did not finish in several minutes, vs. ~5s at the original ~12
    dimensions with LabelEncoder. SMOTENC is the correct tool for mixed
    categorical+continuous data: it operates on the RAW (pre-one-hot)
    categorical columns directly, using a cheap category-overlap distance
    instead of one-hot Euclidean distance, and empirically finished the
    full 350k-row training split in ~40s. The ENN cleanup half of
    SMOTE-ENN (which removes ambiguous majority-class neighbors after
    oversampling) is dropped in this revision rather than forced through
    at the same prohibitive cost - SMOTE's oversampling plus the dynamic
    scale_pos_weight below are what actually address the class imbalance;
    ENN was a secondary refinement on top, not the primary mechanism.
    One-hot encoding itself happens AFTER resampling (on the now-balanced
    train set, plus val/test), once it's just a fixed, cheap, deterministic
    expansion with no distance computation involved.

Evaluated on: clean test set, and the SAME 5%/10% noise+missingness
corruption used in robustness_test.py, so the numbers are directly
comparable to the v1 baseline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, average_precision_score, classification_report
from imblearn.over_sampling import SMOTENC
import xgboost as xgb
import warnings
import os
import gc

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MISSING_FRAC = 0.08  # 8% - middle of the requested 5-10% range

DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
)
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("XGBOOST FRAUD MODEL - v2 ROBUST (v2.1: v2 data, one-hot, SMOTENC)")
print("=" * 60)

# ============================================================
# STEP 0 - LOAD (categoricals stay RAW for now - SMOTENC needs them raw)
# ============================================================
print("\nLoading data...")
NEEDED_COLS = list(set([
    "BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES", "CLAIM_TO_BASE_RATIO",
    "AGE", "LENGTH_OF_STAY_HOURS", "ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY",
    "IS_FRAUD", "PRIMARY_CONDITION", "PROVIDER_ID", "SERVICE_DATE", "ENCOUNTER_ID",
    "FRAUD_TYPE",
]))
df = pd.read_csv(DATA_PATH, usecols=NEEDED_COLS)
print(f"Dataset shape: {df.shape}, fraud rate: {df['IS_FRAUD'].mean()*100:.2f}%")
print(f"Fraud type breakdown:\n{df['FRAUD_TYPE'].value_counts()}")

categorical_features = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]
for col in categorical_features:
    df[col] = df[col].astype(str)
# Full category vocabulary, fixed up front from the WHOLE dataset (not just
# train) so every one-hot column exists consistently across train/val/test
# regardless of which split a rare category happens to land in. This is the
# same thing LabelEncoder/get_dummies-on-full-df did before - fixing the
# vocabulary isn't a leakage risk, it never looks at IS_FRAUD.
onehot_categories = {col: sorted(df[col].unique().tolist()) for col in categorical_features}
encoded_categoricals = [f"{col}_{cat}" for col in categorical_features for cat in onehot_categories[col]]
print(f"Category counts: {{col: len(cats) for col, cats in onehot_categories.items()}}")


def onehot_encode(df_slice: pd.DataFrame, categorical_cols, categories: dict) -> pd.DataFrame:
    """Deterministic one-hot expansion using a FIXED category list (not
    pd.get_dummies' auto-discovered columns), so train/val/test and later
    export_model_artifacts.py / app.py all produce identical column sets
    regardless of which categories happen to appear in a given slice."""
    out = df_slice.drop(columns=categorical_cols).copy()
    for col in categorical_cols:
        values = df_slice[col].astype(str)
        for cat in categories[col]:
            out[f"{col}_{cat}"] = (values == cat).astype(int)
    return out


# ============================================================
# STEP 0b - SPLIT FIRST (identical random_state/proportions to v1,
# so row membership matches exactly and results are comparable)
# ============================================================
y_full = df["IS_FRAUD"]
idx_train, idx_temp = train_test_split(
    df.index, test_size=0.30, random_state=RANDOM_STATE, stratify=y_full
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_full.loc[idx_temp]
)
print(f"Train/Val/Test sizes: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

# Captured before any further mutation/deletion - needed for the
# per-fraud-type recall report near the end.
test_fraud_type = df.loc[idx_test, "FRAUD_TYPE"].copy() if "FRAUD_TYPE" in df.columns else None

# ============================================================
# STEP 2 - CONTEXTUAL FEATURES (fit on TRAIN only, applied everywhere)
# ============================================================
print("\nEngineering contextual cost features (fit on train split only)...")

train_df = df.loc[idx_train]

# --- Regional cost anomaly ratio ---
group_cols = ["PRIMARY_CONDITION", "PATIENT_COUNTY"]
group_median = train_df.groupby(group_cols)["CLAIM_AMOUNT_KES"].median()
global_median = train_df["CLAIM_AMOUNT_KES"].median()

median_lookup = df[group_cols].apply(tuple, axis=1).map(group_median)
median_lookup = median_lookup.fillna(global_median)
df["COST_RATIO"] = (df["CLAIM_AMOUNT_KES"] / median_lookup.replace(0, np.nan)).fillna(1.0)

# --- Provider peer deviation (z-score) ---
provider_mean = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].mean()
provider_std = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].std()
global_mean = train_df["CLAIM_AMOUNT_KES"].mean()
global_std = train_df["CLAIM_AMOUNT_KES"].std()

mean_lookup = df["PROVIDER_ID"].map(provider_mean).fillna(global_mean)
std_lookup = df["PROVIDER_ID"].map(provider_std).fillna(global_std).replace(0, global_std)
df["PROVIDER_Z_SCORE"] = ((df["CLAIM_AMOUNT_KES"] - mean_lookup) / std_lookup).fillna(0.0)

# --- Time-based volatility (same-day claim count per provider) ---
daily_counts = df.groupby(["PROVIDER_ID", "SERVICE_DATE"])["ENCOUNTER_ID"].transform("count")
df["PROVIDER_DAILY_VOLUME"] = daily_counts

print("New features added: COST_RATIO, PROVIDER_Z_SCORE, PROVIDER_DAILY_VOLUME")
print(df[["COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME"]].describe())

# ============================================================
# FEATURE SET (raw categoricals for now - one-hot happens after resampling)
# ============================================================
COST_FEATURES = ["BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES"]
RATIO_FEATURE = ["CLAIM_TO_BASE_RATIO"]
CONTEXT_FEATURES = ["COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME"]
OTHER_NUMERICAL = ["AGE", "LENGTH_OF_STAY_HOURS"]  # PATIENT_INCOME removed - not applicable to unemployed/informal-sector patients

RAW_FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + categorical_features
FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + encoded_categoricals
NOISY_COLUMNS = COST_FEATURES + RATIO_FEATURE + ["COST_RATIO", "PROVIDER_Z_SCORE"]

print(f"\nRaw features going into SMOTENC ({len(RAW_FEATURE_COLUMNS)}): {RAW_FEATURE_COLUMNS}")
print(f"Final (one-hot) features for the model ({len(FEATURE_COLUMNS)})")

X_train_raw = df.loc[idx_train, RAW_FEATURE_COLUMNS]
X_val_raw = df.loc[idx_val, RAW_FEATURE_COLUMNS]
X_test_raw = df.loc[idx_test, RAW_FEATURE_COLUMNS]
y_train = y_full.loc[idx_train]
y_val = y_full.loc[idx_val]
y_test = y_full.loc[idx_test]

del train_df, median_lookup, mean_lookup, std_lookup, daily_counts
gc.collect()

# ============================================================
# SMOTENC OVERSAMPLING (on raw categoricals + numeric cost values)
# ============================================================
print("\nApplying SMOTENC oversampling (categorical-aware, no one-hot blowup)...")
cat_idx = [RAW_FEATURE_COLUMNS.index(c) for c in categorical_features]
smote_nc = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
X_train_res_raw, y_train_res = smote_nc.fit_resample(X_train_raw, y_train)
print(f"Before: {y_train.value_counts().to_dict()}  After: {pd.Series(y_train_res).value_counts().to_dict()}")

# ============================================================
# STEP 1 - INJECT MISSINGNESS INTO THE RESAMPLED TRAINING SET
# (after resampling, since SMOTE can't interpolate across NaNs)
# ============================================================
print(f"\nInjecting {MISSING_FRAC*100:.0f}% missingness into training cost columns: {NOISY_COLUMNS}")
rng = np.random.default_rng(RANDOM_STATE)
X_train_res_raw = X_train_res_raw.reset_index(drop=True)
for col in NOISY_COLUMNS:
    mask = rng.random(len(X_train_res_raw)) < MISSING_FRAC
    X_train_res_raw.loc[mask, col] = np.nan
missing_pct_check = X_train_res_raw[NOISY_COLUMNS].isna().mean() * 100
print("Actual missing % achieved per column:")
print(missing_pct_check)

# ============================================================
# ONE-HOT ENCODE (now, after resampling - cheap, deterministic, same fixed
# category list for train/val/test)
# ============================================================
print("\nOne-hot encoding (post-resampling)...")
X_train_res = onehot_encode(X_train_res_raw, categorical_features, onehot_categories)[FEATURE_COLUMNS]
X_val = onehot_encode(X_val_raw, categorical_features, onehot_categories)[FEATURE_COLUMNS]
X_test = onehot_encode(X_test_raw, categorical_features, onehot_categories)[FEATURE_COLUMNS]

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
print(f"\n{'='*60}\nRESULTS - v2 ROBUST MODEL (v2.1: v2 data, one-hot, SMOTENC)\n{'='*60}")

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
# COMPARISON TABLE (v1 numbers hardcoded from the earlier verified run.
# v1 was trained on the OLD data file without collusion_ring - this
# comparison is about the noise-robustness changes (missingness training,
# contextual features, regularization), kept apples-to-apples; the
# v2-data + one-hot + SMOTENC revision is evaluated separately below via
# per-fraud-type recall against collusion_ring specifically.)
# ============================================================
v1_reference = {
    "v1 - CLEAN test set": {"f1": 0.9770, "balanced_accuracy": 0.9819, "pr_auc": 0.9896},
    "v1 - 5% noise+missingness test set": {"f1": 0.5934, "balanced_accuracy": 0.8822, "pr_auc": 0.7924},
    "v1 - 10% noise+missingness test set": {"f1": 0.4569, "balanced_accuracy": 0.8095, "pr_auc": 0.5845},
}

print(f"\n{'='*60}\nv1 (fragile) vs v2.1 (robust, v2-data, one-hot, SMOTENC) COMPARISON\n{'='*60}")
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
# PER-FRAUD-TYPE RECALL (clean test set) - the number that actually matters
# for the collusion_ring / GraphSAGE story, now computed on the SAME model
# that's actually saved and deployed, not a throwaway side-model.
# ============================================================
print(f"\n{'='*60}\nPER-FRAUD-TYPE RECALL (this saved model, clean test set)\n{'='*60}")
fraud_type_rows = []
if test_fraud_type is not None:
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = model.predict(X_test)
    for ft in sorted(test_fraud_type.unique()):
        if ft == "none":
            continue
        ft_mask = (test_fraud_type == ft).values
        n = int(ft_mask.sum())
        caught = int(y_pred_test[ft_mask].sum())
        recall = caught / n if n else float("nan")
        mean_score = float(y_proba_test[ft_mask].mean()) if n else float("nan")
        print(f"  {ft:<18} n={n:<6} caught={caught:<6} recall={recall:.4f} mean_score={mean_score:.4f}")
        fraud_type_rows.append({"fraud_type": ft, "n": n, "caught": caught, "recall": recall, "mean_score": mean_score})

# ============================================================
# SAVE
# ============================================================
model.save_model(f"{OUTPUT_DIR}/xgboost_fraud_model_v2_robust.json")
comparison_df.to_csv(f"{OUTPUT_DIR}/v1_vs_v2_comparison.csv", index=False)
pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/v2_robust_results.csv", index=False)
if fraud_type_rows:
    pd.DataFrame(fraud_type_rows).to_csv(f"{OUTPUT_DIR}/per_fraudtype_recall_v2data.csv", index=False)
print(f"\nModel saved to {OUTPUT_DIR}/xgboost_fraud_model_v2_robust.json")
print(f"Comparison saved to {OUTPUT_DIR}/v1_vs_v2_comparison.csv")
print(f"Per-fraud-type recall saved to {OUTPUT_DIR}/per_fraudtype_recall_v2data.csv (now from the ACTUAL saved model)")
