"""
HYBRID XGBOOST + GRAPHSAGE EVALUATION

The thesis's core premise: XGBoost specialises in cost-anomaly fraud
(upcoding, phantom_billing) and GraphSAGE specialises in relational fraud
(collusion_ring) - a claim neither model's specialty covers should not be
missed just because the OTHER model wasn't built for it. This script
builds the simplest possible hybrid rule - flag a claim as fraud if
EITHER model says so - and evaluates it against each model standalone,
on the exact same test split, to quantify what the hybrid actually buys
over either model alone.

Reuses:
  - xgboost_model_v2_robust.py's exact feature engineering + the already-
    trained, already-saved model (no retraining).
  - graphsage_collusion.py's already-saved test predictions
    (output/model/graphsage_test_predictions.csv), which scores ring-vs-
    not-ring specifically.

Both scripts build their split from the SAME CSV with the SAME
RANDOM_STATE=42 train_test_split calls, so their row order is identical
by construction - verified with an assertion before combining, not just
assumed.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score
import xgboost as xgb
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
MODEL_PATH = "output/model/xgboost_fraud_model_v2_robust.json"
GRAPHSAGE_PRED_PATH = "output/model/graphsage_test_predictions.csv"

print("Loading data + reproducing the exact xgboost_model_v2_robust.py split/features...")
NEEDED_COLS = list(set([
    "BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES", "CLAIM_TO_BASE_RATIO",
    "AGE", "LENGTH_OF_STAY_HOURS", "ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY",
    "IS_FRAUD", "PRIMARY_CONDITION", "PROVIDER_ID", "SERVICE_DATE", "ENCOUNTER_ID",
    "FRAUD_TYPE", "GEOGRAPHIC_MISMATCH", "CLINICAL_INCONSISTENCY",
]))
df = pd.read_csv(DATA_PATH, usecols=NEEDED_COLS)

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
idx_train, idx_temp = train_test_split(df.index, test_size=0.30, random_state=RANDOM_STATE, stratify=y_full)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_full.loc[idx_temp])
train_df = df.loc[idx_train]
train_legit_df = train_df[train_df["IS_FRAUD"] == 0]

group_cols = ["PRIMARY_CONDITION", "PATIENT_COUNTY"]
group_median = train_df.groupby(group_cols)["CLAIM_AMOUNT_KES"].median()
global_median = train_df["CLAIM_AMOUNT_KES"].median()
median_lookup = df[group_cols].apply(tuple, axis=1).map(group_median).fillna(global_median)
df["COST_RATIO"] = (df["CLAIM_AMOUNT_KES"] / median_lookup.replace(0, np.nan)).fillna(1.0)

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

# Build test set in ORIGINAL ASCENDING ROW ORDER (matches graphsage_collusion.py's
# boolean test_mask filtering), NOT idx_test's shuffled order.
test_mask_bool = pd.Series(False, index=df.index)
test_mask_bool.loc[idx_test] = True
X_test = onehot_encode(df.loc[test_mask_bool, RAW_FEATURE_COLUMNS], categorical_features, onehot_categories)[FEATURE_COLUMNS]
y_test = y_full.loc[test_mask_bool]
fraud_type_test = df.loc[test_mask_bool, "FRAUD_TYPE"].to_numpy()

print("Loading XGBoost model and scoring test set...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
xgb_probs = model.predict_proba(X_test)[:, 1]
xgb_pred = (xgb_probs >= 0.5).astype(int)

print("Loading GraphSAGE test predictions...")
gs = pd.read_csv(GRAPHSAGE_PRED_PATH)
assert len(gs) == len(fraud_type_test), f"length mismatch: graphsage={len(gs)} xgboost_test={len(fraud_type_test)}"
match_rate = (gs["FRAUD_TYPE"].to_numpy() == fraud_type_test).mean()
print(f"Row-order alignment check (FRAUD_TYPE match rate between the two test sets): {match_rate*100:.2f}%")
assert match_rate == 1.0, "GraphSAGE and XGBoost test rows are NOT aligned - do not trust the hybrid numbers below"

graphsage_pred_ring = gs["pred_ring"].to_numpy()
graphsage_score_ring = gs["score"].to_numpy()

# ============================================================
# HYBRID RULE: flag fraud if XGBoost says fraud OR GraphSAGE says ring
# ============================================================
hybrid_pred = ((xgb_pred == 1) | (graphsage_pred_ring == 1)).astype(int)
y_test_np = y_test.to_numpy()

print(f"\n{'='*70}\nPER-FRAUD-TYPE RECALL - standalone vs hybrid\n{'='*70}")
print(f"{'fraud_type':<18}{'n':>8}{'xgb_recall':>14}{'graphsage_recall':>18}{'hybrid_recall':>16}")
for ft in ["upcoding", "phantom_billing", "collusion_ring"]:
    m = fraud_type_test == ft
    n = m.sum()
    xgb_r = recall_score(np.ones(n), xgb_pred[m]) if n else float("nan")
    gs_r = recall_score(np.ones(n), graphsage_pred_ring[m]) if n else float("nan")
    hy_r = recall_score(np.ones(n), hybrid_pred[m]) if n else float("nan")
    print(f"{ft:<18}{n:>8}{xgb_r:>14.4f}{gs_r:>18.4f}{hy_r:>16.4f}")

print(f"\n{'='*70}\nFALSE POSITIVE RATE on LEGITIMATE ('none') claims\n{'='*70}")
m = fraud_type_test == "none"
n = m.sum()
xgb_fpr = xgb_pred[m].mean()
gs_fpr = graphsage_pred_ring[m].mean()
hy_fpr = hybrid_pred[m].mean()
print(f"n={n}  xgb_fpr={xgb_fpr*100:.3f}%  graphsage_fpr={gs_fpr*100:.3f}%  hybrid_fpr={hy_fpr*100:.3f}%")

print(f"\n{'='*70}\nOVERALL (all fraud types pooled as one binary task)\n{'='*70}")
for name, pred in [("XGBoost alone", xgb_pred), ("GraphSAGE alone (ring-only head)", graphsage_pred_ring), ("Hybrid (OR rule)", hybrid_pred)]:
    rec = recall_score(y_test_np, pred)
    prec = precision_score(y_test_np, pred)
    f1 = f1_score(y_test_np, pred)
    print(f"{name:<36} recall={rec:.4f}  precision={prec:.4f}  f1={f1:.4f}")

out = pd.DataFrame({
    "FRAUD_TYPE": fraud_type_test,
    "IS_FRAUD": y_test_np,
    "xgb_prob": xgb_probs,
    "xgb_pred": xgb_pred,
    "graphsage_ring_score": graphsage_score_ring,
    "graphsage_ring_pred": graphsage_pred_ring,
    "hybrid_pred": hybrid_pred,
})
out.to_csv("output/model/hybrid_test_predictions.csv", index=False)
print("\nSaved output/model/hybrid_test_predictions.csv")
