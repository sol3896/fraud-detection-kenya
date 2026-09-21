"""
EXPORT MODEL ARTIFACTS FOR THE STREAMLIT DASHBOARD

Prototype 1 in the Gantt chart calls for a basic web interface that scores
claims with XGBoost. The trained model (xgboost_fraud_model_v2_robust.json)
by itself isn't enough to score a NEW claim submitted through a form: three
of its features (COST_RATIO, PROVIDER_Z_SCORE, PROVIDER_DAILY_VOLUME) are
computed from lookups fit on the TRAINING split only (see
xgboost_model_v2_robust.py's docstring - this is deliberate, to avoid
leakage). Re-fitting those lookups from scratch on every dashboard restart
would be slow and would silently drift from the validated model's inputs.

This script reproduces the EXACT same load -> encode -> split -> engineer
pipeline as xgboost_model_v2_robust.py / xgboost_threshold_tuning.py (same
RANDOM_STATE=42, same train/val/test split), so the lookups are guaranteed
to match what the saved model was actually trained/tuned against. It does
NOT retrain the model - it loads the already-saved, already-verified
xgboost_fraud_model_v2_robust.json - and instead:

  1. Rebuilds the train-only lookups (group cost median, provider mean/std,
     provider historical daily volume) and saves them as JSON.
  2. Re-derives t_high / t_low exactly as xgboost_threshold_tuning.py does,
     so the dashboard's traffic-light thresholds match the validated
     adaptive-triage numbers, not a hand-typed copy of them.
  3. Saves dropdown reference data (known providers + their org IDs and
     historical claim volume, known counties/conditions/encounter types)
     so the Streamlit form can offer realistic choices instead of free text.

Output: output/model/dashboard_artifacts.json
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from imblearn.combine import SMOTEENN
import xgboost as xgb
import warnings
import os
import json

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MISSING_FRAC = 0.08
TARGET_RECALL = 0.95

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
print(f"Shape: {df.shape}, fraud rate: {df['IS_FRAUD'].mean()*100:.2f}%")

categorical_features = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]
le_map = {}
df_model = df.copy()
for col in categorical_features:
    le = LabelEncoder()
    df_model[col + "_ENCODED"] = le.fit_transform(df_model[col].astype(str))
    le_map[col] = list(le.classes_)  # index = encoded value
encoded_categoricals = [c + "_ENCODED" for c in categorical_features]

y_full = df_model["IS_FRAUD"]
idx_train, idx_temp = train_test_split(
    df_model.index, test_size=0.30, random_state=RANDOM_STATE, stratify=y_full
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_full.loc[idx_temp]
)
train_df = df_model.loc[idx_train]

# --- COST_RATIO lookup: (PRIMARY_CONDITION, PATIENT_COUNTY) -> median ---
group_cols = ["PRIMARY_CONDITION", "PATIENT_COUNTY"]
group_median = train_df.groupby(group_cols)["CLAIM_AMOUNT_KES"].median()
global_median = float(train_df["CLAIM_AMOUNT_KES"].median())
median_lookup = df_model[group_cols].apply(tuple, axis=1).map(group_median).fillna(global_median)
df_model["COST_RATIO"] = (df_model["CLAIM_AMOUNT_KES"] / median_lookup.replace(0, np.nan)).fillna(1.0)
# JSON keys must be strings -> join tuple with a separator unlikely to collide
group_median_lookup = {f"{c}||{co}": float(v) for (c, co), v in group_median.items()}

# --- PROVIDER_Z_SCORE lookup ---
provider_mean = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].mean()
provider_std = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].std()
global_mean = float(train_df["CLAIM_AMOUNT_KES"].mean())
global_std = float(train_df["CLAIM_AMOUNT_KES"].std())
mean_lookup = df_model["PROVIDER_ID"].map(provider_mean).fillna(global_mean)
std_lookup = df_model["PROVIDER_ID"].map(provider_std).fillna(global_std).replace(0, global_std)
df_model["PROVIDER_Z_SCORE"] = ((df_model["CLAIM_AMOUNT_KES"] - mean_lookup) / std_lookup).fillna(0.0)
provider_mean_lookup = {str(k): float(v) for k, v in provider_mean.items()}
provider_std_lookup = {str(k): float(v) for k, v in provider_std.items() if pd.notna(v)}

# --- PROVIDER_DAILY_VOLUME: historical average, used as a baseline the app
# adds today's app-submitted claims on top of (see app.py) ---
daily_counts = df_model.groupby(["PROVIDER_ID", "SERVICE_DATE"])["ENCOUNTER_ID"].transform("count")
df_model["PROVIDER_DAILY_VOLUME"] = daily_counts
provider_avg_daily_volume = (
    df_model.groupby("PROVIDER_ID")["PROVIDER_DAILY_VOLUME"].mean().to_dict()
)
provider_avg_daily_volume = {str(k): float(v) for k, v in provider_avg_daily_volume.items()}
global_avg_daily_volume = float(df_model["PROVIDER_DAILY_VOLUME"].mean())

COST_FEATURES = ["BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES"]
RATIO_FEATURE = ["CLAIM_TO_BASE_RATIO"]
CONTEXT_FEATURES = ["COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME"]
OTHER_NUMERICAL = ["AGE", "LENGTH_OF_STAY_HOURS"]  # PATIENT_INCOME removed - not applicable to unemployed/informal-sector patients
FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + encoded_categoricals
NOISY_COLUMNS = COST_FEATURES + RATIO_FEATURE + ["COST_RATIO", "PROVIDER_Z_SCORE"]

X_train = df_model.loc[idx_train, FEATURE_COLUMNS]
X_val = df_model.loc[idx_val, FEATURE_COLUMNS]
y_train = y_full.loc[idx_train]
y_val = y_full.loc[idx_val]

print("\nApplying SMOTE-ENN + training-time missingness injection (must match training exactly)...")
smote_enn = SMOTEENN(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

rng = np.random.default_rng(RANDOM_STATE)
X_train_res = X_train_res.reset_index(drop=True)
for col in NOISY_COLUMNS:
    mask = rng.random(len(X_train_res)) < MISSING_FRAC
    X_train_res.loc[mask, col] = np.nan

print("\nLoading the already-trained, already-validated model (not retraining)...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)


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


print("\nDeriving t_high (F1-optimal, clean val)...")
y_val_proba_clean = model.predict_proba(X_val)[:, 1]
precisions, recalls, pr_thresholds = precision_recall_curve(y_val, y_val_proba_clean)
f1_scores = np.divide(
    2 * precisions * recalls, precisions + recalls,
    out=np.zeros_like(precisions), where=(precisions + recalls) != 0
)
best_idx = np.argmax(f1_scores[:-1])
t_high = float(pr_thresholds[best_idx])
print(f"t_high = {t_high:.4f}")

print("\nDeriving t_low (recall-oriented, noisy val)...")
X_val_noisy = corrupt(X_val, MISSING_FRAC, seed=RANDOM_STATE)
y_val_proba_noisy = model.predict_proba(X_val_noisy)[:, 1]
precisions_n, recalls_n, pr_thresholds_n = precision_recall_curve(y_val, y_val_proba_noisy)
eligible = np.where(recalls_n[:-1] >= TARGET_RECALL)[0]
if len(eligible) > 0:
    t_low_idx = eligible[np.argmax(pr_thresholds_n[eligible])]
    t_low = float(pr_thresholds_n[t_low_idx])
else:
    t_low = float(pr_thresholds_n.min())
print(f"t_low = {t_low:.4f}")

# --- Reference/dropdown data ---
provider_org_lookup = (
    df.drop_duplicates("PROVIDER_ID").set_index("PROVIDER_ID")["ORGANIZATION_ID"].to_dict()
)
provider_org_lookup = {str(k): str(v) for k, v in provider_org_lookup.items()}
provider_claim_counts = df["PROVIDER_ID"].value_counts()
top_providers = [str(p) for p in provider_claim_counts.head(30).index]  # for a manageable dropdown

# PROVIDER_ID is Synthea's anonymous per-clinician UUID (no readable name exists
# for it), but providers_kenya.csv keys by ORGANIZATION_ID and gives the actual
# facility name (e.g. "Level 6 - National Referral Hospital - Nyandarua Town").
# Exporting this so the dashboard can show a real facility name instead of a
# raw UUID in the provider dropdown.
print("\nLoading facility names (providers_kenya.csv, keyed by ORGANIZATION_ID)...")
providers_ref = pd.read_csv(
    os.path.expanduser("~/synthea_kenya/output/kenyanised/providers_kenya.csv"),
    usecols=["Id", "NAME"],
)
organization_names = dict(zip(providers_ref["Id"].astype(str), providers_ref["NAME"].astype(str)))

conditions = sorted(df["PRIMARY_CONDITION"].dropna().unique().tolist())
counties = sorted(df["PATIENT_COUNTY"].dropna().unique().tolist())

form_defaults = {
    "BASE_COST_KES": {"min": float(df["BASE_COST_KES"].min()), "max": float(df["BASE_COST_KES"].max()), "median": float(df["BASE_COST_KES"].median())},
    "CLAIM_AMOUNT_KES": {"min": float(df["CLAIM_AMOUNT_KES"].min()), "max": float(df["CLAIM_AMOUNT_KES"].max()), "median": float(df["CLAIM_AMOUNT_KES"].median())},
    "PAYER_COVERAGE_KES": {"min": float(df["PAYER_COVERAGE_KES"].min()), "max": float(df["PAYER_COVERAGE_KES"].max()), "median": float(df["PAYER_COVERAGE_KES"].median())},
    "AGE": {"min": int(df["AGE"].min()), "max": int(df["AGE"].max()), "median": float(df["AGE"].median())},
    "LENGTH_OF_STAY_HOURS": {"min": float(df["LENGTH_OF_STAY_HOURS"].min()), "max": float(df["LENGTH_OF_STAY_HOURS"].max()), "median": float(df["LENGTH_OF_STAY_HOURS"].median())},
}

artifacts = {
    "feature_columns": FEATURE_COLUMNS,
    "cost_features": COST_FEATURES,
    "categorical_features": categorical_features,
    "label_encoders": le_map,  # {col: [class_at_index_0, class_at_index_1, ...]}
    "group_cols": group_cols,
    "group_median_lookup": group_median_lookup,  # "condition||county" -> median
    "global_median": global_median,
    "provider_mean_lookup": provider_mean_lookup,
    "provider_std_lookup": provider_std_lookup,
    "global_mean": global_mean,
    "global_std": global_std,
    "provider_avg_daily_volume": provider_avg_daily_volume,
    "global_avg_daily_volume": global_avg_daily_volume,
    "provider_org_lookup": provider_org_lookup,
    "organization_names": organization_names,
    "top_providers": top_providers,
    "conditions": conditions,
    "counties": counties,
    "encounter_types": le_map["ENCOUNTER_TYPE"],
    "genders": le_map["GENDER"],
    "t_high": t_high,
    "t_low": t_low,
    "form_defaults": form_defaults,
}

with open(f"{OUTPUT_DIR}/dashboard_artifacts.json", "w") as f:
    json.dump(artifacts, f, indent=2)

print(f"\nSaved artifacts to {OUTPUT_DIR}/dashboard_artifacts.json")
print(f"t_high={t_high:.4f}  t_low={t_low:.4f}")
print(f"{len(top_providers)} providers, {len(conditions)} conditions, {len(counties)} counties in reference data")
