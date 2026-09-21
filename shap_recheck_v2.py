"""
SHAP RECHECK - v2-robust model

The original shap_summary.png (still in output/model/) was computed on the
v1 model, before the contextual features (COST_RATIO, PROVIDER_Z_SCORE,
PROVIDER_DAILY_VOLUME) existed and before missingness-robust training. This
script re-runs SHAP specifically on the v2-robust model to answer the open
question: are the three engineered contextual features actually pulling
real predictive weight, or is the model still leaning almost entirely on
the raw cost columns (which the earlier feature-ablation test showed carry
nearly all the signal - A2 ablation collapsed to random-guess PR-AUC when
cost fields were dropped)?

Reproduces the exact v2-robust feature engineering (train-split-only
lookups, same RANDOM_STATE=42 split) so the SHAP explanations are computed
against features that match how the deployed model actually sees data, then
loads the already-trained, already-validated model (no retraining) and
computes SHAP values on a sample of the clean test set.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 3000  # test-set rows to explain (SHAP is expensive; a sample is standard practice)

DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_sample_500k_13pct.csv"
)
MODEL_PATH = os.path.expanduser(
    "~/synthea_kenya/output/model/xgboost_fraud_model_v2_robust.json"
)
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)

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
OTHER_NUMERICAL = ["AGE", "LENGTH_OF_STAY_HOURS"]  # PATIENT_INCOME removed - not applicable to unemployed/informal-sector patients
FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + encoded_categoricals

X_test = df_model.loc[idx_test, FEATURE_COLUMNS]
y_test = y_full.loc[idx_test]

print(f"Test set: {len(X_test)} rows. Sampling {SHAP_SAMPLE_SIZE} for SHAP (stratified by class)...")
sample_idx, _ = train_test_split(
    X_test.index, train_size=min(SHAP_SAMPLE_SIZE, len(X_test)),
    random_state=RANDOM_STATE, stratify=y_test
)
X_sample = X_test.loc[sample_idx]

print("Loading the already-trained v2-robust model (not retraining)...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

print("Computing SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
importance_df["rank"] = importance_df.index + 1
importance_df["is_contextual_feature"] = importance_df["feature"].isin(CONTEXT_FEATURES)

total_shap = importance_df["mean_abs_shap"].sum()
importance_df["pct_of_total"] = importance_df["mean_abs_shap"] / total_shap * 100

print(f"\n{'='*70}\nSHAP FEATURE IMPORTANCE (v2-robust model, {len(X_sample)}-row sample)\n{'='*70}")
print(importance_df.to_string(index=False))

context_total_pct = importance_df.loc[importance_df["is_contextual_feature"], "pct_of_total"].sum()
raw_cost_total_pct = importance_df.loc[importance_df["feature"].isin(COST_FEATURES + RATIO_FEATURE), "pct_of_total"].sum()
print(f"\nEngineered contextual features (COST_RATIO, PROVIDER_Z_SCORE, PROVIDER_DAILY_VOLUME) "
      f"account for {context_total_pct:.1f}% of total |SHAP| weight.")
print(f"Raw cost fields (BASE/CLAIM/PAYER cost + CLAIM_TO_BASE_RATIO) account for {raw_cost_total_pct:.1f}%.")

importance_df.to_csv(f"{OUTPUT_DIR}/shap_importance_v2_robust.csv", index=False)

# Bar chart (matches the style of the earlier v1 shap_summary.png for comparability)
plt.figure(figsize=(9, 6))
colors = ["#d62728" if c else "#1f77b4" for c in importance_df["is_contextual_feature"]]
plt.barh(importance_df["feature"][::-1], importance_df["mean_abs_shap"][::-1],
         color=colors[::-1])
plt.xlabel("Mean |SHAP value|")
plt.title("v2-robust model: feature importance (red = engineered contextual feature)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary_v2_robust.png", dpi=150)
print(f"\nSaved importance table to {OUTPUT_DIR}/shap_importance_v2_robust.csv")
print(f"Saved bar chart to {OUTPUT_DIR}/shap_summary_v2_robust.png")
