"""
Sanity check on the GraphSAGE 100% recall result: is it really using
relational/graph structure, or just exploiting raw node degree (how many
times a PATIENT_ID/PROVIDER_ID appears)? The 60 fabricated ring-patient
identities get reused ~2300x each while a real patient averages ~9 claims
- if a trivial degree threshold alone separates collusion_ring perfectly,
GraphSAGE isn't demonstrating anything beyond a one-line rule.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score

RANDOM_STATE = 42
df = pd.read_csv("output/kenyanised/master_claims_sample_500k_13pct_v2.csv",
                  usecols=["PATIENT_ID", "PROVIDER_ID", "IS_FRAUD", "FRAUD_TYPE"])

y_full = df["IS_FRAUD"]
idx_train, idx_temp = train_test_split(df.index, test_size=0.30, random_state=RANDOM_STATE, stratify=y_full)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_full.loc[idx_temp])
split = pd.Series("train", index=df.index)
split.loc[idx_val] = "val"
split.loc[idx_test] = "test"
df["SPLIT"] = split.values

patient_degree = df["PATIENT_ID"].value_counts()
provider_degree = df["PROVIDER_ID"].value_counts()
df["PATIENT_DEGREE"] = df["PATIENT_ID"].map(patient_degree)
df["PROVIDER_DEGREE"] = df["PROVIDER_ID"].map(provider_degree)
df["LOG_PATIENT_DEGREE"] = np.log1p(df["PATIENT_DEGREE"])
df["LOG_PROVIDER_DEGREE"] = np.log1p(df["PROVIDER_DEGREE"])

print("PATIENT_DEGREE describe, collusion_ring rows vs rest:")
print(df.loc[df["FRAUD_TYPE"]=="collusion_ring","PATIENT_DEGREE"].describe())
print(df.loc[df["FRAUD_TYPE"]!="collusion_ring","PATIENT_DEGREE"].describe())

y = (df["FRAUD_TYPE"] == "collusion_ring").astype(int)
X = df[["LOG_PATIENT_DEGREE", "LOG_PROVIDER_DEGREE"]]

train_mask = df["SPLIT"] == "train"
test_mask = df["SPLIT"] == "test"

clf = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X[train_mask], y[train_mask])
probs = clf.predict_proba(X[test_mask])[:, 1]
pred = (probs >= 0.5).astype(int)
y_test = y[test_mask].to_numpy()
ft_test = df.loc[test_mask, "FRAUD_TYPE"].to_numpy()

pr_auc = average_precision_score(y_test, probs)
print(f"\nDegree-only logistic regression - test PR-AUC: {pr_auc:.4f}")

ring_mask = ft_test == "collusion_ring"
recall = recall_score(y_test[ring_mask], pred[ring_mask])
print(f"collusion_ring recall (degree-only, n={ring_mask.sum()}): {recall:.4f}")
for ft in ["none", "upcoding", "phantom_billing"]:
    m = ft_test == ft
    print(f"  '{ft}' flagged as ring: {pred[m].mean()*100:.2f}%")

# simplest possible rule: is patient one of the ~60 highest-frequency patients?
threshold_check = df.loc[df["FRAUD_TYPE"]=="collusion_ring","PATIENT_DEGREE"].min()
print(f"\nMinimum PATIENT_DEGREE among ANY collusion_ring row: {threshold_check}")
print(f"PATIENT_DEGREE 99th percentile among non-ring rows: {df.loc[df['FRAUD_TYPE']!='collusion_ring','PATIENT_DEGREE'].quantile(0.99):.0f}")
