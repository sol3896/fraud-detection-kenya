"""
Runs the v2-robust XGBoost pipeline on the NEW dataset (which now includes
collusion_ring, a deliberately non-cost-based relational fraud type), and
breaks recall down BY FRAUD_TYPE on the test set. The expectation, if the
collusion injection worked as intended, is that XGBoost should still catch
upcoding and phantom_billing well (cost anomalies) but struggle specifically
on collusion_ring (a graph anomaly no row-wise tabular model can see) -
which is exactly the gap GraphSAGE is meant to fill.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score, average_precision_score
from imblearn.combine import SMOTEENN
import xgboost as xgb
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MISSING_FRAC = 0.08

DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
)
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data (v2, with collusion_ring)...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}, fraud rate: {df['IS_FRAUD'].mean()*100:.2f}%")
print(df["FRAUD_TYPE"].value_counts())

categorical_features = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]
le = LabelEncoder()
df_model = df.copy()
for col in categorical_features:
    df_model[col + "_ENCODED"] = le.fit_transform(df_model[col].astype(str))
encoded_categoricals = [c + "_ENCODED" for c in categorical_features]

y_full = df_model["IS_FRAUD"]
idx_train, idx_temp = train_test_split(
    df_model.index, test_size=0.30, random_state=RANDOM_STATE, stratify=y_full
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_full.loc[idx_temp]
)

train_df = df_model.loc[idx_train]
group_cols = ["PRIMARY_CONDITION", "PATIENT_COUNTY"]
group_median = train_df.groupby(group_cols)["CLAIM_AMOUNT_KES"].median()
global_median = train_df["CLAIM_AMOUNT_KES"].median()
median_lookup = df_model[group_cols].apply(tuple, axis=1).map(group_median).fillna(global_median)
df_model["COST_RATIO"] = (df_model["CLAIM_AMOUNT_KES"] / median_lookup.replace(0, np.nan)).fillna(1.0)

provider_mean = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].mean()
provider_std = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].std()
global_mean = train_df["CLAIM_AMOUNT_KES"].mean()
global_std = train_df["CLAIM_AMOUNT_KES"].std()
mean_lookup = df_model["PROVIDER_ID"].map(provider_mean).fillna(global_mean)
std_lookup = df_model["PROVIDER_ID"].map(provider_std).fillna(global_std).replace(0, global_std)
df_model["PROVIDER_Z_SCORE"] = ((df_model["CLAIM_AMOUNT_KES"] - mean_lookup) / std_lookup).fillna(0.0)

daily_counts = df_model.groupby(["PROVIDER_ID", "SERVICE_DATE"])["ENCOUNTER_ID"].transform("count")
df_model["PROVIDER_DAILY_VOLUME"] = daily_counts

COST_FEATURES = ["BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES"]
RATIO_FEATURE = ["CLAIM_TO_BASE_RATIO"]
CONTEXT_FEATURES = ["COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME"]
OTHER_NUMERICAL = ["AGE", "PATIENT_INCOME", "LENGTH_OF_STAY_HOURS"]
FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + encoded_categoricals
NOISY_COLUMNS = COST_FEATURES + RATIO_FEATURE + ["COST_RATIO", "PROVIDER_Z_SCORE"]

X_train = df_model.loc[idx_train, FEATURE_COLUMNS]
X_val = df_model.loc[idx_val, FEATURE_COLUMNS]
X_test = df_model.loc[idx_test, FEATURE_COLUMNS]
y_train = y_full.loc[idx_train]
y_val = y_full.loc[idx_val]
y_test = y_full.loc[idx_test]
fraud_type_test = df_model.loc[idx_test, "FRAUD_TYPE"]

print("\nApplying SMOTE-ENN + training-time missingness injection...")
smote_enn = SMOTEENN(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

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
    random_state=RANDOM_STATE, n_jobs=-1,
)
model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False)
print(f"Model trained. Best iteration: {model.best_iteration}")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

overall_f1 = f1_score(y_test, y_pred)
overall_pr_auc = average_precision_score(y_test, y_proba)
print(f"\nOverall Test F1: {overall_f1:.4f}   Overall Test PR-AUC: {overall_pr_auc:.4f}")

print(f"\n{'='*60}\nRECALL BY FRAUD TYPE (on test set)\n{'='*60}")
results = []
for ftype in ["upcoding", "phantom_billing", "collusion_ring"]:
    mask = fraud_type_test == ftype
    n = mask.sum()
    caught = y_pred[mask.values].sum()
    recall = caught / n if n > 0 else float("nan")
    mean_score = y_proba[mask.values].mean()
    print(f"{ftype:20s}  n={n:6d}  caught={caught:6d}  recall={recall:.4f}  mean_fraud_score={mean_score:.4f}")
    results.append({"fraud_type": ftype, "n": int(n), "caught": int(caught), "recall": recall, "mean_score": mean_score})

# Compare to a random provider-patient pair's mean score, for reference
legit_mask = fraud_type_test == "none"
print(f"{'legitimate (none)':20s}  n={legit_mask.sum():6d}  mean_fraud_score={y_proba[legit_mask.values].mean():.4f}")

pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/per_fraudtype_recall_v2data.csv", index=False)
print(f"\nSaved to {OUTPUT_DIR}/per_fraudtype_recall_v2data.csv")
