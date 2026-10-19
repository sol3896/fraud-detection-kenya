"""
Second-order sanity check: with degree closed off, does GraphSAGE's 100%
recall now reduce to just memorising the 300 fixed ring-patient
demographic profiles (AGE/GENDER/PATIENT_COUNTY), with no real graph
reasoning at all? Train a classifier on ONLY those static per-row
attributes (plus provider county/level) - no PATIENT_ID/PROVIDER_ID
identity, no degree, no graph edges whatsoever - and see how far that
gets alone.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, recall_score

RANDOM_STATE = 42
df = pd.read_csv("output/kenyanised/master_claims_sample_500k_13pct_v2.csv",
                  usecols=["AGE", "GENDER", "PATIENT_COUNTY", "PROVIDER_COUNTY",
                           "PROVIDER_LEVEL", "IS_FRAUD", "FRAUD_TYPE"])

y_full = df["IS_FRAUD"]
idx_train, idx_temp = train_test_split(df.index, test_size=0.30, random_state=RANDOM_STATE, stratify=y_full)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_full.loc[idx_temp])
split = pd.Series("train", index=df.index)
split.loc[idx_val] = "val"
split.loc[idx_test] = "test"
df["SPLIT"] = split.values

cat_cols = ["GENDER", "PATIENT_COUNTY", "PROVIDER_COUNTY", "PROVIDER_LEVEL"]
X = pd.get_dummies(df[["AGE"] + cat_cols], columns=cat_cols)
y = (df["FRAUD_TYPE"] == "collusion_ring").astype(int)

train_mask = (df["SPLIT"] == "train").to_numpy()
test_mask = (df["SPLIT"] == "test").to_numpy()

clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                              random_state=RANDOM_STATE, n_jobs=4)
clf.fit(X[train_mask], y[train_mask])
probs = clf.predict_proba(X[test_mask])[:, 1]
pred = (probs >= 0.5).astype(int)
y_test = y[test_mask].to_numpy()
ft_test = df.loc[test_mask, "FRAUD_TYPE"].to_numpy()

pr_auc = average_precision_score(y_test, probs)
ring_mask = ft_test == "collusion_ring"
recall = recall_score(y_test[ring_mask], pred[ring_mask])
print(f"Attributes-only (AGE/GENDER/COUNTY/LEVEL, no IDs, no degree, no graph) - "
      f"test PR-AUC: {pr_auc:.4f}")
print(f"collusion_ring recall: n={ring_mask.sum()} recall={recall:.4f}")
for ft in ["none", "upcoding", "phantom_billing"]:
    m = ft_test == ft
    print(f"  '{ft}' flagged as ring: {pred[m].mean()*100:.2f}%")
