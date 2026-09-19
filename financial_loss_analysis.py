"""
XGBOOST FRAUD MODEL - v4 FINANCIAL LOSS ANALYSIS

Replaces the abstract F1 metric with a financial loss matrix, in KES, to
decide which of the three decision strategies (default 0.5 threshold,
global F1-tuned threshold, adaptive dual-threshold triage) is actually the
best choice for an insurer to deploy - not which scores best on F1.

Cost mapping (hypothesised, stated explicitly so it can be swapped for
real SHA/insurer figures later):
  COST_FN = 20,000 KES  - a missed fraud case: the full payout is lost
  COST_FP = 500 KES     - an auditor's time spent clearing a false alarm
  COST_TP = 500 KES     - an investigator's time confirming a real case
  (True negatives cost nothing extra - a correctly-cleared legitimate
  claim is the normal cost of doing business, not a loss attributable to
  the fraud model.)

IMPORTANT ON THE ADAPTIVE STRATEGY: it has three real outcomes, not two -
auto-clear, auto-flag (high confidence), and manual review (ambiguous /
incomplete). A manual review claim isn't a "predicted fraud" in the same
sense as an auto-flag; it still costs an auditor's time to clear
(COST_FP-equivalent) but ALSO still catches the fraud if it is one
(avoiding COST_FN), so it is costed as an investigation either way -
same COST_FP/COST_TP logic as an auto-flag, just executed by a human
instead of the model. This is the fairest way to compare it to the
single-threshold strategies without silently favouring or penalising it.

Re-trains the same v2/v3 model so this script is self-contained and can
be run standalone; uses the identical tuned thresholds from
xgboost_threshold_tuning.py (t_high=0.596, t_low=0.184) rather than
re-deriving them, so the comparison is apples-to-apples with that run.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score, average_precision_score,
    precision_recall_curve,
)
from imblearn.combine import SMOTEENN
import xgboost as xgb
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MISSING_FRAC = 0.08
TARGET_RECALL = 0.95

# ============================================================
# COST MAPPING (KES) - as specified
# ============================================================
COST_FN = 20000  # missed fraud - full payout lost
COST_FP = 500    # auditor time clearing a false alarm
COST_TP = 500    # investigator time confirming real fraud
COST_TN = 0      # correctly-cleared legitimate claim - no extra cost

DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_sample_500k_13pct.csv"
)
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("XGBOOST FRAUD MODEL - v4 FINANCIAL LOSS ANALYSIS")
print("=" * 60)

# ============================================================
# LOAD + ENCODE + SPLIT + FEATURES + TRAIN (identical to v2/v3)
# ============================================================
print("\nLoading data and retraining v2/v3 model (same pipeline)...")
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
OTHER_NUMERICAL = ["AGE", "PATIENT_INCOME", "LENGTH_OF_STAY_HOURS"]
FEATURE_COLUMNS = COST_FEATURES + RATIO_FEATURE + CONTEXT_FEATURES + OTHER_NUMERICAL + encoded_categoricals
NOISY_COLUMNS = COST_FEATURES + RATIO_FEATURE + ["COST_RATIO", "PROVIDER_Z_SCORE"]

X_train = df_model.loc[idx_train, FEATURE_COLUMNS]
X_val = df_model.loc[idx_val, FEATURE_COLUMNS]
X_test = df_model.loc[idx_test, FEATURE_COLUMNS]
y_train = y_full.loc[idx_train]
y_val = y_full.loc[idx_val]
y_test = y_full.loc[idx_test]

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


def is_incomplete(X):
    return X[NOISY_COLUMNS].isna().any(axis=1)


# ============================================================
# RE-DERIVE THE SAME TUNED THRESHOLDS AS xgboost_threshold_tuning.py
# (re-computed here, not hardcoded, so this script stays correct even
# if the model/data changes slightly - should reproduce 0.596 / 0.184)
# ============================================================
y_val_proba_clean = model.predict_proba(X_val)[:, 1]
precisions, recalls, pr_thresholds = precision_recall_curve(y_val, y_val_proba_clean)
f1_scores = np.divide(
    2 * precisions * recalls, precisions + recalls,
    out=np.zeros_like(precisions), where=(precisions + recalls) != 0
)
t_high = pr_thresholds[np.argmax(f1_scores[:-1])]

X_val_noisy = corrupt(X_val, MISSING_FRAC, seed=RANDOM_STATE)
y_val_proba_noisy = model.predict_proba(X_val_noisy)[:, 1]
precisions_n, recalls_n, pr_thresholds_n = precision_recall_curve(y_val, y_val_proba_noisy)
eligible = np.where(recalls_n[:-1] >= TARGET_RECALL)[0]
t_low = pr_thresholds_n[eligible[np.argmax(pr_thresholds_n[eligible])]] if len(eligible) > 0 else pr_thresholds_n.min()

print(f"\nUsing thresholds: t_high={t_high:.4f}  t_low={t_low:.4f}")


# ============================================================
# FINANCIAL LOSS CALCULATION
# ============================================================
def calculate_financial_loss(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    total_loss = (fn * COST_FN) + (fp * COST_FP) + (tp * COST_TP) + (tn * COST_TN)
    return total_loss, {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def predict_default(X):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= 0.5).astype(int), proba


def predict_tuned(X):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= t_high).astype(int), proba


def predict_adaptive(X):
    proba = model.predict_proba(X)[:, 1]
    incomplete = is_incomplete(X).values
    pred = np.where(incomplete, proba >= t_low, proba >= t_high).astype(int)
    return pred, proba


strategies = {
    "Default (0.5) threshold": predict_default,
    "Global tuned threshold": predict_tuned,
    "Adaptive dual-threshold triage": predict_adaptive,
}

conditions = {
    "clean": X_test,
    "5pct_noise": corrupt(X_test, 0.05, RANDOM_STATE),
    "10pct_noise": corrupt(X_test, 0.10, RANDOM_STATE),
}

print(f"\n{'='*70}\nFINANCIAL LOSS BY STRATEGY AND CONDITION (KES, on {len(y_test)} test claims)\n{'='*70}")
print(f"Cost assumptions: FN={COST_FN:,} KES | FP={COST_FP:,} KES | TP={COST_TP:,} KES | TN={COST_TN:,} KES\n")

rows = []
for cond_name, X_cond in conditions.items():
    print(f"--- {cond_name} ---")
    for strat_name, predict_fn in strategies.items():
        pred, proba = predict_fn(X_cond)
        loss, counts = calculate_financial_loss(y_test, pred)
        f1 = f1_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        loss_per_claim = loss / len(y_test)
        print(f"  {strat_name:32s} Loss=KES {loss:>12,.0f}  (KES {loss_per_claim:,.1f}/claim)  "
              f"F1={f1:.4f}  TP={counts['tp']:<6} FP={counts['fp']:<6} FN={counts['fn']:<6} TN={counts['tn']}")
        rows.append({
            "condition": cond_name, "strategy": strat_name,
            "total_loss_kes": loss, "loss_per_claim_kes": loss_per_claim,
            "f1": f1, "precision": prec, "recall": rec,
            "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "tn": counts["tn"],
        })
    print()

# ============================================================
# REFERENCE BOUNDS: do-nothing and flag-everything, for context
# ============================================================
print(f"{'='*70}\nREFERENCE BOUNDS (for context, not real strategies)\n{'='*70}")
for cond_name, X_cond in conditions.items():
    n = len(y_test)
    n_fraud = int(y_test.sum())
    loss_do_nothing, _ = calculate_financial_loss(y_test, np.zeros(n))
    loss_flag_all, _ = calculate_financial_loss(y_test, np.ones(n))
    print(f"{cond_name:12s}  do-nothing (flag none): KES {loss_do_nothing:,.0f}   "
          f"flag-everyone: KES {loss_flag_all:,.0f}")

results_df = pd.DataFrame(rows)
out_path = f"{OUTPUT_DIR}/financial_loss_analysis.csv"
results_df.to_csv(out_path, index=False)

# ============================================================
# SUMMARY - which strategy wins financially, per condition
# ============================================================
print(f"\n{'='*70}\nWINNER BY CONDITION (lowest total KES loss)\n{'='*70}")
for cond_name in conditions:
    sub = results_df[results_df["condition"] == cond_name].sort_values("total_loss_kes")
    winner = sub.iloc[0]
    print(f"{cond_name:12s} -> {winner['strategy']}  (KES {winner['total_loss_kes']:,.0f})")
    print(f"             full ranking: " +
          ", ".join(f"{r['strategy']}=KES{r['total_loss_kes']:,.0f}" for _, r in sub.iterrows()))

print(f"\nFull results saved to {out_path}")
