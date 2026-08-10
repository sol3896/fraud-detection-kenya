import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    precision_recall_curve
)
from imblearn.combine import SMOTEENN
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
DATA_PATH = os.path.expanduser(
    "~/synthea_kenya/output/kenyanised/master_claims_kenya_clean.csv"
)
OUTPUT_DIR = os.path.expanduser(
    "~/synthea_kenya/output/model"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print("XGBOOST FRAUD DETECTION MODEL")
print("=" * 50)

# ============================================================
# STEP 1 - LOAD DATA
# ============================================================
print("\nStep 1: Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['IS_FRAUD'].mean()*100:.1f}%")

# ============================================================
# STEP 2 - FEATURE SELECTION
# ============================================================
print("\nStep 2: Selecting features...")

# Categorical features to encode
categorical_features = [
    "ENCOUNTER_TYPE",
    "GENDER",
    "PATIENT_COUNTY",
    "FRAUD_TYPE"
]

# Numerical features
numerical_features = [
    "BASE_COST_KES",
    "CLAIM_AMOUNT_KES",
    "PAYER_COVERAGE_KES",
    "AGE",
    "PATIENT_INCOME",
    "LENGTH_OF_STAY_HOURS",
    "CLAIM_TO_BASE_RATIO"
]

# Encode categorical features
le = LabelEncoder()
df_model = df.copy()

for col in categorical_features:
    if col in df_model.columns:
        df_model[col + "_ENCODED"] = le.fit_transform(
            df_model[col].astype(str)
        )

encoded_categoricals = [
    col + "_ENCODED" for col in categorical_features
    if col in df_model.columns
]

feature_columns = numerical_features + encoded_categoricals
target_column = "IS_FRAUD"

X = df_model[feature_columns]
y = df_model[target_column]

print(f"Features used: {len(feature_columns)}")
print(f"Feature list: {feature_columns}")
print(f"Class distribution:")
print(y.value_counts())

# ============================================================
# STEP 3 - TRAIN TEST SPLIT
# ============================================================
print("\nStep 3: Splitting data...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f"Training set: {X_train.shape[0]} records")
print(f"Validation set: {X_val.shape[0]} records")
print(f"Test set: {X_test.shape[0]} records")

# ============================================================
# STEP 4 - SMOTE-ENN REBALANCING
# ============================================================
print("\nStep 4: Applying SMOTE-ENN rebalancing...")
print("This may take a few minutes on large datasets...")

smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(
    X_train, y_train
)

print(f"Before resampling: {y_train.value_counts().to_dict()}")
print(f"After resampling: {pd.Series(y_train_resampled).value_counts().to_dict()}")

# ============================================================
# STEP 5 - TRAIN XGBOOST MODEL
# ============================================================
print("\nStep 5: Training XGBoost model...")

# Calculate scale_pos_weight for imbalanced data
neg_count = sum(y_train_resampled == 0)
pos_count = sum(y_train_resampled == 1)
scale_pos_weight = neg_count / pos_count

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_resampled,
    y_train_resampled,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print("Training complete.")

# ============================================================
# STEP 6 - EVALUATE ON VALIDATION SET
# ============================================================
print("\nStep 6: Evaluating on validation set...")

y_val_pred = model.predict(X_val)
y_val_proba = model.predict_proba(X_val)[:, 1]

val_f1 = f1_score(y_val, y_val_pred)
val_bal_acc = balanced_accuracy_score(y_val, y_val_pred)
val_pr_auc = average_precision_score(y_val, y_val_proba)

print(f"Validation F1 Score:          {val_f1:.4f}")
print(f"Validation Balanced Accuracy: {val_bal_acc:.4f}")
print(f"Validation PR-AUC:            {val_pr_auc:.4f}")
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred,
      target_names=["Legitimate", "Fraud"]))

# ============================================================
# STEP 7 - EVALUATE ON TEST SET
# ============================================================
print("\nStep 7: Evaluating on test set...")

y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

test_f1 = f1_score(y_test, y_test_pred)
test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
test_pr_auc = average_precision_score(y_test, y_test_proba)

print(f"Test F1 Score:          {test_f1:.4f}")
print(f"Test Balanced Accuracy: {test_bal_acc:.4f}")
print(f"Test PR-AUC:            {test_pr_auc:.4f}")
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred,
      target_names=["Legitimate", "Fraud"]))

# ============================================================
# STEP 8 - SHAP EXPLANATIONS
# ============================================================
print("\nStep 8: Computing SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:1000])

print("SHAP values computed for 1000 test samples.")

# Feature importance from SHAP
shap_importance = pd.DataFrame({
    "feature": feature_columns,
    "mean_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_shap", ascending=False)

print("\nTop 10 features by SHAP importance:")
print(shap_importance.head(10).to_string(index=False))

# Save SHAP summary plot
print("\nSaving SHAP summary plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test[:1000],
    feature_names=feature_columns,
    show=False
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150)
plt.close()
print(f"SHAP plot saved to {OUTPUT_DIR}/shap_summary.png")

# ============================================================
# STEP 9 - GENERATE PLAIN TEXT EXPLANATION FOR ONE CLAIM
# ============================================================
print("\nStep 9: Generating plain text explanation for sample claim...")

sample_idx = 0
sample_claim = X_test.iloc[sample_idx]
sample_shap = shap_values[sample_idx]
sample_fraud_score = y_test_proba[sample_idx]
sample_true_label = y_test.iloc[sample_idx]

# Get top 3 contributing features
feature_contributions = list(zip(feature_columns, sample_shap))
feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
top_features = feature_contributions[:3]

print(f"\nSample claim fraud score: {sample_fraud_score:.3f}")
print(f"True label: {'FRAUD' if sample_true_label == 1 else 'LEGITIMATE'}")
print(f"Predicted: {'FRAUD' if sample_fraud_score > 0.5 else 'LEGITIMATE'}")

explanation_parts = []
for feature, contribution in top_features:
    value = sample_claim[feature]
    direction = "increased" if contribution > 0 else "decreased"
    explanation_parts.append(
        f"{feature} = {value:.2f} {direction} fraud risk"
    )

plain_text_explanation = "Flagged because: " + "; ".join(explanation_parts)
print(f"\nPlain text explanation:")
print(plain_text_explanation)

# ============================================================
# STEP 10 - SAVE MODEL AND RESULTS
# ============================================================
print("\nStep 10: Saving model and results...")

model.save_model(f"{OUTPUT_DIR}/xgboost_fraud_model.json")

results = {
    "val_f1": val_f1,
    "val_balanced_accuracy": val_bal_acc,
    "val_pr_auc": val_pr_auc,
    "test_f1": test_f1,
    "test_balanced_accuracy": test_bal_acc,
    "test_pr_auc": test_pr_auc
}

results_df = pd.DataFrame([results])
results_df.to_csv(f"{OUTPUT_DIR}/model_results.csv", index=False)

print(f"Model saved to {OUTPUT_DIR}/xgboost_fraud_model.json")
print(f"Results saved to {OUTPUT_DIR}/model_results.csv")

print("\n" + "=" * 50)
print("XGBOOST MODEL COMPLETE")
print(f"Final Test PR-AUC: {test_pr_auc:.4f}")
print(f"Final Test F1:     {test_f1:.4f}")
print("=" * 50)