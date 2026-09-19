"""
XGBOOST FRAUD MODEL - v3 THRESHOLD TUNING (adaptive triage)

Motivation: v2's F1 still dropped under noise (0.72 at 5%, 0.58 at 10%)
because it kept the default 0.5 decision cutoff. F1 is threshold-sensitive
in a way PR-AUC isn't - the model's RANKING of fraud vs legitimate barely
degraded (PR-AUC only fell from 0.99 to 0.81-0.93), but 0.5 stopped being
a good cutoff once claim scores got noisier. This script fixes that with
two changes, both tuned on the VALIDATION split only (never on test):

  1. GLOBAL THRESHOLD TUNING
     Instead of the default 0.5, find the threshold that maximizes F1 on
     the clean validation set via precision_recall_curve. This alone
     recovers most of the lost F1 on clean and lightly-noisy data.

  2. ADAPTIVE DUAL-THRESHOLD TRIAGE (the real-world deployment version)
     A single global threshold can't be both "confident enough to
     auto-decide" and "sensitive enough to catch fraud hiding in noisy
     rows" at the same time. So instead of one cutoff, this uses two:
       - t_high: the F1-optimal threshold from the CLEAN validation set.
         Score >= t_high -> auto-flag as fraud (the model is confident).
       - t_low: a lower, recall-oriented threshold tuned on a NOISY
         (8%-corrupted) copy of the validation set, picked as the highest
         threshold that still achieves >= TARGET_RECALL (95%) recall
         under noise. Score >= t_low -> route to a human manual-review
         queue instead of auto-clearing.
     The adaptive rule only relaxes the threshold for rows the pipeline
     can tell are incomplete (at least one missing cost field) - a
     complete row still needs to clear the stricter t_high bar. This
     mirrors how a real Kenyan SHA/insurer fraud unit would want to
     operate: don't silently drop a claim just because a field is
     missing - route it to a human instead of guessing.

     Caveat: "incomplete" here is measured directly (a NaN in a cost
     field), which is exactly what our synthetic noise experiment can
     inject and detect. True "high variance" scoring (e.g. from model
     ensemble disagreement) would need multiple models/seeds and is a
     natural extension, not implemented here.

Re-trains the same v2-robust model (missingness training + contextual
features + regularized hyperparameters) so this script is self-contained.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, average_precision_score,
    precision_recall_curve,
)
from imblearn.combine import SMOTEENN
import xgboost as xgb
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MISSING_FRAC = 0.08
TARGET_RECALL = 0.95  # for the manual-review threshold t_low

DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_sample_500k_13pct.csv"
)
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("XGBOOST FRAUD MODEL - v3 THRESHOLD TUNING / ADAPTIVE TRIAGE")
print("=" * 60)

# ============================================================
# LOAD + ENCODE + SPLIT (identical to v2)
# ============================================================
print("\nLoading data...")
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

# ============================================================
# CONTEXTUAL FEATURES (fit on train only, identical to v2)
# ============================================================
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

# ============================================================
# SMOTE-ENN + missingness-injected training (identical to v2)
# ============================================================
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


# ============================================================
# CORRUPTION HELPER (identical to robustness_test.py / v2)
# ============================================================
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
    """A row is 'incomplete' if any of its cost-related fields are missing."""
    return X[NOISY_COLUMNS].isna().any(axis=1)


# ============================================================
# STEP 1 - GLOBAL THRESHOLD TUNING (on clean validation set)
# ============================================================
print(f"\n{'='*60}\nSTEP 1 - Tuning global F1-optimal threshold on CLEAN validation set\n{'='*60}")

y_val_proba_clean = model.predict_proba(X_val)[:, 1]
precisions, recalls, pr_thresholds = precision_recall_curve(y_val, y_val_proba_clean)
f1_scores = np.divide(
    2 * precisions * recalls, precisions + recalls,
    out=np.zeros_like(precisions), where=(precisions + recalls) != 0
)
best_idx = np.argmax(f1_scores[:-1])  # last point has no corresponding threshold
t_high = pr_thresholds[best_idx]
print(f"Default threshold: 0.5000")
print(f"t_high (F1-optimal on clean val): {t_high:.4f}  "
      f"(val F1 at this threshold: {f1_scores[best_idx]:.4f})")

# ============================================================
# STEP 2 - RECALL-ORIENTED THRESHOLD FOR MANUAL REVIEW
# (tuned on a NOISY copy of validation - matches the training noise level)
# ============================================================
print(f"\n{'='*60}\nSTEP 2 - Tuning t_low (manual-review threshold) on NOISY (8%) validation set,"
      f"\ntargeting >= {TARGET_RECALL*100:.0f}% recall\n{'='*60}")

X_val_noisy = corrupt(X_val, MISSING_FRAC, seed=RANDOM_STATE)
y_val_proba_noisy = model.predict_proba(X_val_noisy)[:, 1]
precisions_n, recalls_n, pr_thresholds_n = precision_recall_curve(y_val, y_val_proba_noisy)

# thresholds are ascending; recall is non-increasing as threshold rises.
# Find the highest threshold that still achieves >= TARGET_RECALL.
eligible = np.where(recalls_n[:-1] >= TARGET_RECALL)[0]
if len(eligible) > 0:
    t_low_idx = eligible[np.argmax(pr_thresholds_n[eligible])]
    t_low = pr_thresholds_n[t_low_idx]
else:
    t_low = pr_thresholds_n.min()
print(f"t_low (recall-oriented on noisy val): {t_low:.4f}  "
      f"(val recall at this threshold: {recalls_n[t_low_idx] if len(eligible)>0 else 'N/A':.4f})")
print(f"\nDeployment interpretation: score < {t_low:.3f} = clear; "
      f"{t_low:.3f} <= score < {t_high:.3f} = route to manual audit "
      f"(especially for incomplete claims); score >= {t_high:.3f} = auto-flag as fraud.")


# ============================================================
# EVALUATION - three strategies x three noise conditions
# ============================================================
def eval_fixed(X, y, threshold, label):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "strategy": label, "f1": f1_score(y, pred),
        "precision": precision_score(y, pred), "recall": recall_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "pr_auc": average_precision_score(y, proba),
    }


def eval_adaptive(X, y, t_high, t_low, label):
    proba = model.predict_proba(X)[:, 1]
    incomplete = is_incomplete(X).values
    # Complete rows need the stricter t_high bar; incomplete rows are
    # "flagged" (treated as positive/needs-review) at the lower t_low bar.
    pred = np.where(incomplete, proba >= t_low, proba >= t_high).astype(int)
    n_flagged_for_review = int((incomplete & (proba >= t_low) & (proba < t_high)).sum())
    return {
        "strategy": label, "f1": f1_score(y, pred),
        "precision": precision_score(y, pred), "recall": recall_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "pr_auc": average_precision_score(y, proba),
        "n_incomplete_rows": int(incomplete.sum()),
        "n_routed_to_manual_review": n_flagged_for_review,
    }


print(f"\n{'='*60}\nRESULTS ACROSS NOISE CONDITIONS\n{'='*60}")

conditions = {"clean": X_test, "5pct_noise": corrupt(X_test, 0.05, RANDOM_STATE),
              "10pct_noise": corrupt(X_test, 0.10, RANDOM_STATE)}

all_results = []
for cond_name, X_cond in conditions.items():
    r_default = eval_fixed(X_cond, y_test, 0.5, f"{cond_name} - default (0.5) threshold")
    r_tuned = eval_fixed(X_cond, y_test, t_high, f"{cond_name} - global tuned threshold ({t_high:.3f})")
    r_adaptive = eval_adaptive(X_cond, y_test, t_high, t_low, f"{cond_name} - adaptive dual-threshold triage")

    print(f"\n--- {cond_name} ---")
    for r in (r_default, r_tuned, r_adaptive):
        extra = ""
        if "n_routed_to_manual_review" in r:
            extra = f"  [incomplete rows: {r['n_incomplete_rows']}, routed to manual review: {r['n_routed_to_manual_review']}]"
        print(f"{r['strategy']:55s} F1={r['f1']:.4f} Prec={r['precision']:.4f} "
              f"Rec={r['recall']:.4f} PR-AUC={r['pr_auc']:.4f}{extra}")
        all_results.append(r)

results_df = pd.DataFrame(all_results)
out_path = f"{OUTPUT_DIR}/threshold_tuning_results.csv"
results_df.to_csv(out_path, index=False)

print(f"\n{'='*60}\nSUMMARY TABLE\n{'='*60}")
print(results_df[["strategy", "f1", "precision", "recall", "pr_auc"]].to_string(index=False))
print(f"\nFull results saved to {out_path}")
print(f"\nTuned thresholds -> t_high (auto-flag): {t_high:.4f}   t_low (manual review, incomplete rows): {t_low:.4f}")
