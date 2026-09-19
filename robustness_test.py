"""
ROBUSTNESS / OVERFITTING STRESS TEST for the XGBoost fraud model.

Two experiments, run back to back:

  EXPERIMENT A - Drop rule-based features
    The synthetic fraud injector (kenyanise.py -> inject_fraud()) creates
    fraud in exactly two ways, and both of them ONLY touch cost fields:
      - upcoding:        inflates CLAIM_AMOUNT_KES by 1.4x-2.5x
                          (BASE_COST_KES is left untouched, which is exactly
                          what CLAIM_TO_BASE_RATIO is designed to catch)
      - phantom_billing:  overwrites BOTH BASE_COST_KES and CLAIM_AMOUNT_KES
                          with new random high values (15k-80k), well outside
                          normal SHA tariff bands for the encounter type
      - PAYER_COVERAGE_KES is computed BEFORE fraud injection, so for fraud
        rows it silently stops matching the (now inflated) claim amount -
        another side-effect of the generation order, not genuine behaviour.

    So CLAIM_TO_BASE_RATIO, CLAIM_AMOUNT_KES, BASE_COST_KES and
    PAYER_COVERAGE_KES are all, to different degrees, direct mirrors of the
    programmatic rule rather than independent behavioural evidence.
    This experiment retrains with those features removed in two stages:
      A1) drop only CLAIM_TO_BASE_RATIO (the most obviously engineered one)
      A2) drop all four cost fields (the full stress version)

  EXPERIMENT B - Add noise / missingness to the test set
    Takes the ORIGINAL model (trained on the full feature set, the same one
    that scored PR-AUC 0.9896 / F1 0.9770) and evaluates it on copies of the
    test set corrupted with 5% and 10% random Gaussian noise on numeric
    columns plus 5% / 10% randomly-missing feature values (left as NaN,
    which XGBoost handles natively). If PR-AUC collapses aggressively here,
    the model has overfitted to the simulator's noise-free perfection rather
    than learning a pattern that would survive real, messy claims data.

Uses the exact same data path, split, and SMOTE-ENN + XGBoost settings as
xgboost_model.py so results are directly comparable.
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

# ============================================================
# PATHS
# ============================================================
DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_sample_500k_13pct.csv"
)
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

print("=" * 60)
print("ROBUSTNESS / OVERFITTING STRESS TEST")
print("=" * 60)

# ============================================================
# LOAD + ENCODE (identical to xgboost_model.py)
# ============================================================
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}, fraud rate: {df['IS_FRAUD'].mean()*100:.2f}%")

categorical_features = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]

le = LabelEncoder()
df_model = df.copy()
for col in categorical_features:
    df_model[col + "_ENCODED"] = le.fit_transform(df_model[col].astype(str))
encoded_categoricals = [c + "_ENCODED" for c in categorical_features]

COST_FEATURES = ["BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES"]
RATIO_FEATURE = ["CLAIM_TO_BASE_RATIO"]
OTHER_NUMERICAL = ["AGE", "PATIENT_INCOME", "LENGTH_OF_STAY_HOURS"]

FULL_FEATURES = COST_FEATURES + RATIO_FEATURE + OTHER_NUMERICAL + encoded_categoricals
NO_RATIO_FEATURES = COST_FEATURES + OTHER_NUMERICAL + encoded_categoricals
NO_COST_FEATURES = RATIO_FEATURE + OTHER_NUMERICAL + encoded_categoricals  # (kept for reference, not used directly)
BEHAVIOURAL_ONLY_FEATURES = OTHER_NUMERICAL + encoded_categoricals

target_column = "IS_FRAUD"
y = df_model[target_column]

# Identical split every time (same random_state) so all experiments share
# the same train/val/test row membership -> fair comparison.
def make_split(feature_cols):
    X = df_model[feature_cols]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_eval(feature_cols, label):
    print(f"\n{'-'*60}\n{label}\nFeatures ({len(feature_cols)}): {feature_cols}\n{'-'*60}")
    X_train, X_val, X_test, y_train, y_val, y_test = make_split(feature_cols)

    smote_enn = SMOTEENN(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

    neg = sum(y_train_res == 0)
    pos = sum(y_train_res == 1)
    scale_pos_weight = neg / pos

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"Test F1:               {f1:.4f}")
    print(f"Test Balanced Accuracy: {bal_acc:.4f}")
    print(f"Test PR-AUC:            {pr_auc:.4f}")

    return {
        "experiment": label,
        "n_features": len(feature_cols),
        "features": ",".join(feature_cols),
        "test_f1": f1,
        "test_balanced_accuracy": bal_acc,
        "test_pr_auc": pr_auc,
    }, model, X_test, y_test


results = []

# ------------------------------------------------------------
# EXPERIMENT A0 - baseline (reproduces the original xgboost_model.py run)
# ------------------------------------------------------------
r0, baseline_model, X_test_full, y_test_full = train_and_eval(
    FULL_FEATURES, "A0 - BASELINE (all features, incl. CLAIM_TO_BASE_RATIO)"
)
results.append(r0)

# ------------------------------------------------------------
# EXPERIMENT A1 - drop only the engineered ratio feature
# ------------------------------------------------------------
r1, _, _, _ = train_and_eval(
    NO_RATIO_FEATURES, "A1 - DROP CLAIM_TO_BASE_RATIO ONLY"
)
results.append(r1)

# ------------------------------------------------------------
# EXPERIMENT A2 - drop every cost field that the injector directly
# manipulates (the strict "no rule mirrors at all" version)
# ------------------------------------------------------------
r2, _, _, _ = train_and_eval(
    BEHAVIOURAL_ONLY_FEATURES,
    "A2 - DROP ALL COST FIELDS (behavioural/demographic features only)",
)
results.append(r2)

# ============================================================
# EXPERIMENT B - noise / missingness stress test on the BASELINE model
# ============================================================
print(f"\n{'='*60}")
print("EXPERIMENT B - NOISE / MISSINGNESS STRESS TEST (on baseline model)")
print(f"{'='*60}")

rng = np.random.default_rng(RANDOM_STATE)
numeric_cols = [c for c in FULL_FEATURES if c not in encoded_categoricals]


def corrupt(X, noise_frac, seed):
    """Add Gaussian noise (scaled to each column's std) to numeric columns,
    then randomly null out noise_frac of ALL feature values (missingness).
    XGBoost handles NaN natively, so no imputation is applied - this
    mirrors what a real, messier claims feed would look like."""
    r = np.random.default_rng(seed)
    Xc = X.copy()

    for col in numeric_cols:
        std = Xc[col].std()
        noise = r.normal(loc=0, scale=std * noise_frac, size=len(Xc))
        Xc[col] = Xc[col] + noise

    mask = r.random(Xc.shape) < noise_frac
    Xc = Xc.mask(mask)
    return Xc


for noise_frac in [0.05, 0.10]:
    X_noisy = corrupt(X_test_full, noise_frac, seed=RANDOM_STATE)
    y_pred_n = baseline_model.predict(X_noisy)
    y_proba_n = baseline_model.predict_proba(X_noisy)[:, 1]

    f1_n = f1_score(y_test_full, y_pred_n)
    bal_acc_n = balanced_accuracy_score(y_test_full, y_pred_n)
    pr_auc_n = average_precision_score(y_test_full, y_proba_n)

    label = f"B - {int(noise_frac*100)}% noise + missingness (baseline model, clean-trained)"
    print(f"\n{label}")
    print(f"Test F1:               {f1_n:.4f}  (clean was {r0['test_f1']:.4f})")
    print(f"Test Balanced Accuracy: {bal_acc_n:.4f}  (clean was {r0['test_balanced_accuracy']:.4f})")
    print(f"Test PR-AUC:            {pr_auc_n:.4f}  (clean was {r0['test_pr_auc']:.4f})")

    results.append({
        "experiment": label,
        "n_features": len(FULL_FEATURES),
        "features": ",".join(FULL_FEATURES),
        "test_f1": f1_n,
        "test_balanced_accuracy": bal_acc_n,
        "test_pr_auc": pr_auc_n,
    })

# ============================================================
# SAVE SUMMARY
# ============================================================
results_df = pd.DataFrame(results)
out_path = f"{OUTPUT_DIR}/robustness_test_results.csv"
results_df.to_csv(out_path, index=False)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(results_df[["experiment", "n_features", "test_f1", "test_balanced_accuracy", "test_pr_auc"]]
      .to_string(index=False))
print(f"\nFull results saved to {out_path}")
