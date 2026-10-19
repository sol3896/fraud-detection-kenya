"""
COLLUSION_RING LEAKAGE CHECK

Diagnostic for the question: is the v2-robust XGBoost model's 78.9% recall
on collusion_ring genuine graph-structure leakage into tabular features, or
something else? No retraining - uses the already-trained, already-exported
model and asks two questions directly:

  1. RAW STATS: do the 12 ring providers actually look different on
     PROVIDER_DAILY_VOLUME (or any other tabular feature) compared to
     normal providers, as a side-effect of inject_collusion_ring.py's own
     mechanics (e.g. concentrating many claims on a small clique)?

  2. SHAP ON COLLUSION_RING ROWS ONLY: reuses shap_recheck_v2.py's exact
     feature engineering/split, but instead of SHAP over a general sample,
     computes SHAP specifically for FRAUD_TYPE == 'collusion_ring' test
     rows and reports which features are actually pushing those particular
     predictions toward "fraud" (mean SIGNED shap, not mean |shap| - sign
     tells us direction, which the general top-15 table doesn't).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
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

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# ---------- PART 1: raw stats on ring providers ----------
ring_providers = df.loc[df["FRAUD_TYPE"] == "collusion_ring", "PROVIDER_ID"].unique()
print(f"\n{'='*70}\nPART 1 - RAW FEATURE STATS: ring providers vs normal providers\n{'='*70}")
print(f"Number of distinct providers in collusion_ring rows: {len(ring_providers)}")

daily_counts = df.groupby(["PROVIDER_ID", "SERVICE_DATE"])["ENCOUNTER_ID"].transform("count")
df["PROVIDER_DAILY_VOLUME"] = daily_counts

is_ring_provider = df["PROVIDER_ID"].isin(ring_providers)
print("\nPROVIDER_DAILY_VOLUME - ring providers (ALL their rows, fraud+legit):")
print(df.loc[is_ring_provider, "PROVIDER_DAILY_VOLUME"].describe())
print("\nPROVIDER_DAILY_VOLUME - non-ring providers:")
print(df.loc[~is_ring_provider, "PROVIDER_DAILY_VOLUME"].describe())

print("\nPROVIDER_DAILY_VOLUME - collusion_ring-LABELED rows only:")
print(df.loc[df["FRAUD_TYPE"] == "collusion_ring", "PROVIDER_DAILY_VOLUME"].describe())
print("\nPROVIDER_DAILY_VOLUME - non-fraud rows only (baseline):")
print(df.loc[df["FRAUD_TYPE"] == "none", "PROVIDER_DAILY_VOLUME"].describe())

print("\nGENDER distribution, collusion_ring rows vs overall:")
print("collusion_ring:", df.loc[df["FRAUD_TYPE"] == "collusion_ring", "GENDER"].value_counts(normalize=True).to_dict())
print("overall:       ", df["GENDER"].value_counts(normalize=True).to_dict())

print("\nPATIENT_COUNTY top-5, collusion_ring rows vs overall:")
print("collusion_ring:", df.loc[df["FRAUD_TYPE"] == "collusion_ring", "PATIENT_COUNTY"].value_counts(normalize=True).head(5).to_dict())
print("overall:       ", df["PATIENT_COUNTY"].value_counts(normalize=True).head(5).to_dict())

print("\nCLAIM_AMOUNT_KES / BASE_COST_KES / CLAIM_TO_BASE_RATIO - collusion_ring vs none:")
for col in ["CLAIM_AMOUNT_KES", "BASE_COST_KES", "CLAIM_TO_BASE_RATIO", "PAYER_COVERAGE_KES"]:
    r = df.loc[df["FRAUD_TYPE"] == "collusion_ring", col].mean()
    n = df.loc[df["FRAUD_TYPE"] == "none", col].mean()
    print(f"  {col}: collusion_ring mean={r:.2f}  none mean={n:.2f}  ratio={r/n:.3f}")

# ---------- PART 2: SHAP specifically on collusion_ring test rows ----------
print(f"\n{'='*70}\nPART 2 - SHAP ON COLLUSION_RING TEST ROWS (signed, mean)\n{'='*70}")

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

COST_FEATURES = ["BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES"]
RATIO_FEATURE = ["CLAIM_TO_BASE_RATIO"]
CONTEXT_FEATURES = ["COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME"]
OTHER_NUMERICAL = ["AGE", "LENGTH_OF_STAY_HOURS"]
ANOMALY_FEATURES = ["GEOGRAPHIC_MISMATCH", "CLINICAL_INCONSISTENCY"]
FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + ANOMALY_FEATURES + encoded_categoricals
RAW_FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + ANOMALY_FEATURES + categorical_features

X_test = onehot_encode(df.loc[idx_test, RAW_FEATURE_COLUMNS], categorical_features, onehot_categories)[FEATURE_COLUMNS]
fraud_type_test = df.loc[idx_test, "FRAUD_TYPE"]

ring_mask = fraud_type_test == "collusion_ring"
print(f"collusion_ring rows in test set: {ring_mask.sum()}")
X_ring = X_test.loc[ring_mask]

print("Loading model...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

print("Computing SHAP for collusion_ring test rows...")
explainer = shap.TreeExplainer(model)
shap_values_ring = explainer.shap_values(X_ring)

mean_signed = shap_values_ring.mean(axis=0)
mean_abs = np.abs(shap_values_ring).mean(axis=0)
ring_shap_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "mean_signed_shap": mean_signed,
    "mean_abs_shap": mean_abs,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
total = ring_shap_df["mean_abs_shap"].sum()
ring_shap_df["pct_of_total"] = ring_shap_df["mean_abs_shap"] / total * 100

print("\nTop 15 features driving collusion_ring predictions (signed mean SHAP):")
print(ring_shap_df.head(15).to_string(index=False))

ring_shap_df.to_csv(
    os.path.expanduser("~/synthea_kenya/output/model/shap_collusion_ring_only.csv"),
    index=False,
)
print("\nSaved to ~/synthea_kenya/output/model/shap_collusion_ring_only.csv")
