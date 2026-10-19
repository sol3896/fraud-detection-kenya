"""
TUNE THE HYBRID DECISION RULE

hybrid_ensemble_eval.py used a naive rule (XGBoost prob>=0.5 OR GraphSAGE
ring-prob>=0.5). The two models have very different score distributions
(XGBoost: balanced two-class problem; GraphSAGE: a sparse single-class
"is this a ring claim" head), so there's no reason 0.5/0.5 is the best
operating point for either. This script tunes BOTH thresholds
independently via a grid search that maximises F1 on the VALIDATION
split (never the test split - thresholds picked by looking at test
performance are not a real tuning result), then reports the test-set
numbers at the tuned operating point for comparison against the naive
0.5/0.5 rule.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import warnings

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
XGB_MODEL_PATH = "output/model/xgboost_fraud_model_v2_robust.json"
GS_CHECKPOINT_PATH = "output/model/graphsage_collusion_checkpoint.pt"

print("Loading data...")
NEEDED_COLS = list(set([
    "BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES", "CLAIM_TO_BASE_RATIO",
    "AGE", "LENGTH_OF_STAY_HOURS", "ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY",
    "IS_FRAUD", "PRIMARY_CONDITION", "PROVIDER_ID", "PATIENT_ID", "SERVICE_DATE",
    "ENCOUNTER_ID", "FRAUD_TYPE", "GEOGRAPHIC_MISMATCH", "CLINICAL_INCONSISTENCY",
    "PROVIDER_COUNTY", "PROVIDER_LEVEL",
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
split = pd.Series("train", index=df.index)
split.loc[idx_val] = "val"
split.loc[idx_test] = "test"
df["SPLIT"] = split.values

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

print("Scoring XGBoost on val + test...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)
X_all = onehot_encode(df[RAW_FEATURE_COLUMNS], categorical_features, onehot_categories)[FEATURE_COLUMNS]
df["XGB_PROB"] = xgb_model.predict_proba(X_all)[:, 1]

# ============================================================
# Rebuild the IDENTICAL graph graphsage_collusion.py built (same node
# feature layout: type one-hot, age, gender, county, provider level - NO
# degree, per the leak fix) and reload its trained weights.
# ============================================================
print("\nRebuilding graph + loading GraphSAGE checkpoint...")
patient_ids = df["PATIENT_ID"].unique()
provider_ids = df["PROVIDER_ID"].unique()
n_patients, n_providers = len(patient_ids), len(provider_ids)
n_nodes = n_patients + n_providers
patient_node_id = {pid: i for i, pid in enumerate(patient_ids)}
provider_node_id = {pid: n_patients + i for i, pid in enumerate(provider_ids)}
df["PATIENT_NODE"] = df["PATIENT_ID"].map(patient_node_id)
df["PROVIDER_NODE"] = df["PROVIDER_ID"].map(provider_node_id)

counties = sorted(set(df["PATIENT_COUNTY"].astype(str).unique()) | set(df["PROVIDER_COUNTY"].astype(str).unique()))
county_idx = {c: i for i, c in enumerate(counties)}
levels = sorted(df["PROVIDER_LEVEL"].astype(str).unique())
level_idx = {l: i for i, l in enumerate(levels)}
n_county, n_level = len(counties), len(levels)
feat_dim = 2 + 1 + 2 + n_county + n_level
X = np.zeros((n_nodes, feat_dim), dtype=np.float32)

patient_first = df.drop_duplicates("PATIENT_ID", keep="first").set_index("PATIENT_ID")
age_mean, age_std = patient_first["AGE"].mean(), patient_first["AGE"].std()
for pid, node in patient_node_id.items():
    row = patient_first.loc[pid]
    X[node, 0] = 1.0
    X[node, 2] = (row["AGE"] - age_mean) / (age_std if age_std else 1.0)
    X[node, 3 if str(row["GENDER"]) == "M" else 4] = 1.0
    X[node, 5 + county_idx[str(row["PATIENT_COUNTY"])]] = 1.0

provider_first = df.drop_duplicates("PROVIDER_ID", keep="first").set_index("PROVIDER_ID")
for pid, node in provider_node_id.items():
    row = provider_first.loc[pid]
    X[node, 1] = 1.0
    X[node, 5 + county_idx[str(row["PROVIDER_COUNTY"])]] = 1.0
    X[node, 5 + n_county + level_idx[str(row["PROVIDER_LEVEL"])]] = 1.0

src = df["PATIENT_NODE"].to_numpy()
dst = df["PROVIDER_NODE"].to_numpy()
edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
x_t = torch.tensor(X, dtype=torch.float32)
edge_index_t = torch.tensor(edge_index, dtype=torch.long)
edge_src_t = torch.tensor(src, dtype=torch.long)
edge_dst_t = torch.tensor(dst, dtype=torch.long)


class GraphSAGEEdgeClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.edge_mlp = nn.Sequential(nn.Linear(out_dim * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h

    def edge_logits(self, h, edge_src, edge_dst):
        return self.edge_mlp(torch.cat([h[edge_src], h[edge_dst]], dim=1)).squeeze(-1)


ckpt = torch.load(GS_CHECKPOINT_PATH, weights_only=False)
assert ckpt["feat_dim"] == feat_dim, f"feature dim mismatch: checkpoint={ckpt['feat_dim']} rebuilt={feat_dim}"
model = GraphSAGEEdgeClassifier(in_dim=feat_dim)
model.load_state_dict(ckpt["model_state"])
model.eval()
with torch.no_grad():
    h = model.encode(x_t, edge_index_t)
    logits = model.edge_logits(h, edge_src_t, edge_dst_t)
    df["GS_PROB"] = torch.sigmoid(logits).numpy()
print("GraphSAGE scoring done.")

y_ring = (df["FRAUD_TYPE"] == "collusion_ring").astype(int)

# ============================================================
# GRID SEARCH on VAL split only
# ============================================================
val_mask = (df["SPLIT"] == "val").to_numpy()
test_mask = (df["SPLIT"] == "test").to_numpy()
y_val = y_full.loc[val_mask].to_numpy() if hasattr(y_full, "loc") else None
y_val = df.loc[val_mask, "IS_FRAUD"].to_numpy()
y_test = df.loc[test_mask, "IS_FRAUD"].to_numpy()
xgb_val, gs_val = df.loc[val_mask, "XGB_PROB"].to_numpy(), df.loc[val_mask, "GS_PROB"].to_numpy()
xgb_test, gs_test = df.loc[test_mask, "XGB_PROB"].to_numpy(), df.loc[test_mask, "GS_PROB"].to_numpy()
ft_val = df.loc[val_mask, "FRAUD_TYPE"].to_numpy()
ft_test = df.loc[test_mask, "FRAUD_TYPE"].to_numpy()

print(f"\n{'='*70}\nGRID SEARCH (val split) - independent thresholds for each model\n{'='*70}")
thresholds = np.arange(0.05, 0.96, 0.05)
best = {"f1": -1}
for t_xgb in thresholds:
    xgb_flag = xgb_val >= t_xgb
    for t_gs in thresholds:
        gs_flag = gs_val >= t_gs
        pred = (xgb_flag | gs_flag).astype(int)
        f1 = f1_score(y_val, pred)
        if f1 > best["f1"]:
            best = {"f1": f1, "t_xgb": t_xgb, "t_gs": t_gs,
                    "precision": precision_score(y_val, pred), "recall": recall_score(y_val, pred)}

print(f"Best VAL operating point: t_xgb={best['t_xgb']:.2f}  t_graphsage={best['t_gs']:.2f}  "
      f"F1={best['f1']:.4f}  precision={best['precision']:.4f}  recall={best['recall']:.4f}")

# ============================================================
# Apply tuned thresholds to TEST, compare against naive 0.5/0.5
# ============================================================
def evaluate(t_xgb, t_gs, label):
    pred = ((xgb_test >= t_xgb) | (gs_test >= t_gs)).astype(int)
    print(f"\n{label} (t_xgb={t_xgb:.2f}, t_graphsage={t_gs:.2f}) - TEST SET:")
    rec, prec, f1 = recall_score(y_test, pred), precision_score(y_test, pred), f1_score(y_test, pred)
    print(f"  overall recall={rec:.4f}  precision={prec:.4f}  f1={f1:.4f}")
    fp_rate = pred[ft_test == "none"].mean()
    print(f"  false-positive rate on legit claims: {fp_rate*100:.3f}%")
    for ft in ["upcoding", "phantom_billing", "collusion_ring"]:
        m = ft_test == ft
        print(f"    {ft:<18} recall={pred[m].mean():.4f}  (n={m.sum()})")
    return pred


print(f"\n{'='*70}\nTEST SET COMPARISON\n{'='*70}")
naive_pred = evaluate(0.5, 0.5, "NAIVE rule")
tuned_pred = evaluate(best["t_xgb"], best["t_gs"], "TUNED rule")

out = df.loc[test_mask, ["FRAUD_TYPE", "IS_FRAUD", "XGB_PROB", "GS_PROB"]].copy()
out["naive_pred"] = naive_pred
out["tuned_pred"] = tuned_pred
out.to_csv("output/model/hybrid_tuned_test_predictions.csv", index=False)

with open("output/model/hybrid_tuned_thresholds.txt", "w") as f:
    f.write(f"t_xgb={best['t_xgb']}\nt_graphsage={best['t_gs']}\n")
print(f"\nSaved tuned thresholds to output/model/hybrid_tuned_thresholds.txt")
print("Saved output/model/hybrid_tuned_test_predictions.csv")
