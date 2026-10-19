"""
ACID TEST - targeted SHAP ablation on collusion_ring rows ONLY.

Same approach collusion_leakage_check.py already used numerically (filter
test set to FRAUD_TYPE == 'collusion_ring', run SHAP.TreeExplainer on just
that slice), but this script also produces the actual summary_plot figure
as visual proof of which feature XGBoost is really using to catch the ring.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
)
MODEL_PATH = os.path.expanduser(
    "~/synthea_kenya/output/model/xgboost_fraud_model_v2_robust.json"
)
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/model")

print("Loading data...")
df = pd.read_csv(DATA_PATH)

categorical_features = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]
for col in categorical_features:
    df[col] = df[col].astype(str)
onehot_categories = {col: sorted(df[col].unique().tolist()) for col in categorical_features}
encoded_categoricals = [f"{col}_{cat}" for col in categorical_features for cat in onehot_categories[col]]


def onehot_encode(df_slice, categorical_cols, categories):
    out = df_slice.drop(columns=categorical_cols).copy()
    for col in categorical_cols:
        values = df_slice[col].astype(str)
        for cat in categories[col]:
            out[f"{col}_{cat}"] = (values == cat).astype(int)
    return out


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

# PROVIDER_Z_SCORE baseline is fit on this provider's TRAIN-LABELED-
# LEGITIMATE claims only (never known fraud rows) - the acid-test SHAP
# ablation (collusion_shap_plot.py) proved this feature was the dominant
# proxy XGBoost used to catch collusion_ring (26.3% of SHAP weight on
# those rows alone), because ring providers' aggregate mean/std was being
# fit on a population where ring fraud claims vastly outnumbered their
# real legitimate history - so the "baseline" a ring claim was compared
# against was mostly other ring claims, not genuine normal behaviour.
# Fitting only on legit rows matches realistic deployment too: a real
# system builds a provider's cost profile from its established clean
# history, not by blending in claims already known to be fraudulent.
train_legit_df = train_df[train_df["IS_FRAUD"] == 0]
provider_mean = train_legit_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].mean()
provider_std = train_legit_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].std()
global_mean = train_legit_df["CLAIM_AMOUNT_KES"].mean()
global_std = train_legit_df["CLAIM_AMOUNT_KES"].std()
mean_lookup = df["PROVIDER_ID"].map(provider_mean).fillna(global_mean)
std_lookup = df["PROVIDER_ID"].map(provider_std).fillna(global_std).replace(0, global_std)
df["PROVIDER_Z_SCORE"] = ((df["CLAIM_AMOUNT_KES"] - mean_lookup) / std_lookup).fillna(0.0)

daily_counts = df.groupby(["PROVIDER_ID", "SERVICE_DATE"])["ENCOUNTER_ID"].transform("count")
df["PROVIDER_DAILY_VOLUME"] = daily_counts

COST_FEATURES = ["BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES"]
RATIO_FEATURE = ["CLAIM_TO_BASE_RATIO"]
CONTEXT_FEATURES = ["COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME"]
OTHER_NUMERICAL = ["AGE", "LENGTH_OF_STAY_HOURS"]
ANOMALY_FEATURES = ["GEOGRAPHIC_MISMATCH", "CLINICAL_INCONSISTENCY"]
FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + ANOMALY_FEATURES + encoded_categoricals
RAW_FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + ANOMALY_FEATURES + categorical_features

X_test_robust = onehot_encode(df.loc[idx_test, RAW_FEATURE_COLUMNS], categorical_features, onehot_categories)[FEATURE_COLUMNS]
y_test = y_full.loc[idx_test]
df_test = df.loc[idx_test]

# ---- exactly the acid test requested ----
collusion_indices = (y_test == 1) & (df_test["FRAUD_TYPE"] == "collusion_ring")
X_collusion = X_test_robust[collusion_indices]
print(f"collusion_ring rows isolated from test set: {len(X_collusion)}")

print("Loading model...")
model_v2 = xgb.XGBClassifier()
model_v2.load_model(MODEL_PATH)

print("Running SHAP exclusively on collusion_ring rows...")
explainer = shap.TreeExplainer(model_v2)
shap_values_collusion = explainer(X_collusion)

print("Saving summary_plot (bar, mean |SHAP|)...")
plt.figure()
shap.summary_plot(shap_values_collusion, X_collusion, plot_type="bar", max_display=15, show=False)
plt.title("SHAP - collusion_ring test rows ONLY (mean |SHAP|)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_collusion_ring_acidtest_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saving summary_plot (beeswarm, signed, shows direction + spread)...")
plt.figure()
shap.summary_plot(shap_values_collusion, X_collusion, max_display=15, show=False)
plt.title("SHAP - collusion_ring test rows ONLY (signed, beeswarm)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_collusion_ring_acidtest_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

mean_abs = np.abs(shap_values_collusion.values).mean(axis=0)
mean_signed = shap_values_collusion.values.mean(axis=0)
out = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "mean_abs_shap": mean_abs,
    "mean_signed_shap": mean_signed,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
out["pct_of_total"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum() * 100
print("\nTop 10 (acid test, n={}):".format(len(X_collusion)))
print(out.head(10).to_string(index=False))
out.to_csv(f"{OUTPUT_DIR}/shap_collusion_ring_acidtest.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR}/shap_collusion_ring_acidtest_bar.png")
print(f"Saved: {OUTPUT_DIR}/shap_collusion_ring_acidtest_beeswarm.png")
print(f"Saved: {OUTPUT_DIR}/shap_collusion_ring_acidtest.csv")
