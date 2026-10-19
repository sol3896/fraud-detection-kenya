"""
PROTOTYPE 2 - GraphSAGE for collusion_ring fraud detection.

Builds the bipartite PATIENT_ID <-> PROVIDER_ID claims graph (nodes =
patients + providers, edges = individual claims) and trains a 2-layer
GraphSAGE encoder + MLP edge-classification head to predict, for each
claim, whether it belongs to the collusion_ring fraud type.

Why this is the right task for a GNN here: the acid-test SHAP ablation
(collusion_shap_plot.py, run against the corrected XGBoost model) showed
collusion_ring claims carry almost no usable row-wise tabular signal
(54.0% recall, no single feature above ~13% of SHAP weight) - exactly
what inject_collusion_ring.py's design intends, since the only real
anomaly it creates is RELATIONAL: 12 providers and 60 patient identities
transacting with each other far more densely than any real
geography-driven relationship would. A model that only ever sees one row
at a time cannot see that density. GraphSAGE's neighbourhood aggregation
can: each node's embedding is built from its neighbours' embeddings, so a
ring provider's embedding ends up looking like "connected to an unusually
tight, repeating cluster of patients" - something a per-row model has no
way to represent.

SPLIT: reuses the EXACT same RANDOM_STATE=42 / 70-15-15 train_test_split
as xgboost_model_v2_robust.py, on the same master_claims_sample_500k_13pct_v2.csv,
so the test-set collusion_ring rows are the IDENTICAL 3,246 claims the
XGBoost 54.0% recall number was computed on - a fair, directly comparable
baseline.

GRAPH: transductive setting - message passing uses ALL edges (train+val+
test claims), since the bipartite structure itself (which patients see
which providers) is observable at inference time in a real deployment;
only the FRAUD LABELS of val/test edges are hidden from the loss. This is
standard practice for link/edge classification GNNs, not a leak: the
label itself is still predicted from unseen edges, only the fact that an
edge (claim) exists is used for structure, which is legitimately known.

NODE FEATURES (deliberately NON-COST, to keep this complementary to
XGBoost rather than redundant with it; raw node DEGREE is deliberately
EXCLUDED - see the note above process_chunk/feature-build below, it was
a trivial univariate leak, not a relational signal):
  - node type one-hot (is_patient, is_provider)
  - patient-only: AGE (normalised), GENDER one-hot
  - shared: COUNTY one-hot (patient's county for patient nodes, provider's
    county for provider nodes - same 20-county vocabulary)
  - provider-only: PROVIDER_LEVEL one-hot
(zero-padded for the node type they don't apply to)
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

t0 = time.time()
torch.manual_seed(42)
np.random.seed(42)

RANDOM_STATE = 42
DATA_PATH = "output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
OUTPUT_DIR = "output/model"

print("Loading data...")
df = pd.read_csv(DATA_PATH, usecols=[
    "PATIENT_ID", "PROVIDER_ID", "GENDER", "AGE", "PATIENT_COUNTY",
    "PROVIDER_COUNTY", "PROVIDER_LEVEL", "IS_FRAUD", "FRAUD_TYPE",
])
print(f"Loaded {len(df)} claims")

# ============================================================
# EXACT SAME SPLIT as xgboost_model_v2_robust.py, so the test-set
# collusion_ring rows match 1:1 for a fair comparison.
# ============================================================
y_full = df["IS_FRAUD"]
idx_train, idx_temp = train_test_split(
    df.index, test_size=0.30, random_state=RANDOM_STATE, stratify=y_full
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_full.loc[idx_temp]
)
split = pd.Series("train", index=df.index)
split.loc[idx_val] = "val"
split.loc[idx_test] = "test"
df["SPLIT"] = split.values
print(df["SPLIT"].value_counts())

n_collusion_test = ((df["SPLIT"] == "test") & (df["FRAUD_TYPE"] == "collusion_ring")).sum()
print(f"collusion_ring rows in test split: {n_collusion_test} "
      f"(should match the 3246 from the XGBoost acid test)")

# ============================================================
# BUILD NODE ID SPACE: patients first, then providers
# ============================================================
patient_ids = df["PATIENT_ID"].unique()
provider_ids = df["PROVIDER_ID"].unique()
n_patients = len(patient_ids)
n_providers = len(provider_ids)
n_nodes = n_patients + n_providers
print(f"\nGraph: {n_patients} patient nodes + {n_providers} provider nodes = {n_nodes} nodes, {len(df)} claim edges")

patient_node_id = {pid: i for i, pid in enumerate(patient_ids)}
provider_node_id = {pid: n_patients + i for i, pid in enumerate(provider_ids)}

df["PATIENT_NODE"] = df["PATIENT_ID"].map(patient_node_id)
df["PROVIDER_NODE"] = df["PROVIDER_ID"].map(provider_node_id)

# ============================================================
# NODE FEATURES
# ============================================================
counties = sorted(set(df["PATIENT_COUNTY"].astype(str).unique()) | set(df["PROVIDER_COUNTY"].astype(str).unique()))
county_idx = {c: i for i, c in enumerate(counties)}
levels = sorted(df["PROVIDER_LEVEL"].astype(str).unique())
level_idx = {l: i for i, l in enumerate(levels)}
n_county = len(counties)
n_level = len(levels)

# feature layout: [is_patient, is_provider, age_norm, gender_M, gender_F,
#                   county(n_county), level(n_level)]
# NOTE: raw/log node degree was DELIBERATELY REMOVED from the feature set.
# degree_only_baseline.py proved a 2-feature logistic regression on just
# log(patient_degree)/log(provider_degree) alone got PERFECT separation
# (PR-AUC 1.0) on collusion_ring - because the injector reused a small
# fixed ring-patient pool, total appearance count was a trivial tabular
# "tell" with no graph reasoning involved at all, and GraphSAGE had direct
# access to that same shortcut through this feature. The injector fix
# (N_RING_PATIENTS 60->300, see inject_collusion_ring.py) closes the worst
# of this at the DATA level, but keeping degree out of the node features
# entirely closes it at the MODEL level too, forcing GraphSAGE to rely on
# actual neighbourhood composition (which specific neighbours a node has,
# aggregated through SAGEConv's mean operator) rather than a magnitude
# count that was never a genuinely relational signal.
feat_dim = 2 + 1 + 2 + n_county + n_level
X = np.zeros((n_nodes, feat_dim), dtype=np.float32)
print(f"\nNode feature dim: {feat_dim} (counties={n_county}, levels={n_level}) - degree EXCLUDED")

# patient-node features: first occurrence per patient
patient_first = df.drop_duplicates("PATIENT_ID", keep="first").set_index("PATIENT_ID")
age_mean, age_std = patient_first["AGE"].mean(), patient_first["AGE"].std()
for pid, node in patient_node_id.items():
    row = patient_first.loc[pid]
    X[node, 0] = 1.0
    X[node, 2] = (row["AGE"] - age_mean) / (age_std if age_std else 1.0)
    if str(row["GENDER"]) == "M":
        X[node, 3] = 1.0
    else:
        X[node, 4] = 1.0
    X[node, 5 + county_idx[str(row["PATIENT_COUNTY"])]] = 1.0

# provider-node features: first occurrence per provider
provider_first = df.drop_duplicates("PROVIDER_ID", keep="first").set_index("PROVIDER_ID")
for pid, node in provider_node_id.items():
    row = provider_first.loc[pid]
    X[node, 1] = 1.0
    X[node, 5 + county_idx[str(row["PROVIDER_COUNTY"])]] = 1.0
    X[node, 5 + n_county + level_idx[str(row["PROVIDER_LEVEL"])]] = 1.0

print(f"Node feature build done in {time.time()-t0:.1f}s")

# ============================================================
# EDGE INDEX (undirected - both directions, for message passing)
# ============================================================
src = df["PATIENT_NODE"].to_numpy()
dst = df["PROVIDER_NODE"].to_numpy()
edge_index = np.stack([
    np.concatenate([src, dst]),
    np.concatenate([dst, src]),
])

x_t = torch.tensor(X, dtype=torch.float32)
edge_index_t = torch.tensor(edge_index, dtype=torch.long)

# edge-level (claim-level) labels + masks, in ORIGINAL row order
y_ring = (df["FRAUD_TYPE"] == "collusion_ring").astype(np.float32).to_numpy()
edge_src_t = torch.tensor(src, dtype=torch.long)
edge_dst_t = torch.tensor(dst, dtype=torch.long)
y_t = torch.tensor(y_ring, dtype=torch.float32)

train_mask = torch.tensor((df["SPLIT"] == "train").to_numpy())
val_mask = torch.tensor((df["SPLIT"] == "val").to_numpy())
test_mask = torch.tensor((df["SPLIT"] == "test").to_numpy())
fraud_type_arr = df["FRAUD_TYPE"].to_numpy()

print(f"Tensors built. edge_index shape {edge_index_t.shape}, x shape {x_t.shape}")


# ============================================================
# MODEL: 2-layer GraphSAGE encoder + MLP edge head
# ============================================================
class GraphSAGEEdgeClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(out_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        return h

    def edge_logits(self, h, edge_src, edge_dst):
        e = torch.cat([h[edge_src], h[edge_dst]], dim=1)
        return self.edge_mlp(e).squeeze(-1)


device = torch.device("cpu")
model = GraphSAGEEdgeClassifier(in_dim=feat_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

n_pos = y_t[train_mask].sum().item()
n_neg = train_mask.sum().item() - n_pos
pos_weight = torch.tensor(n_neg / max(n_pos, 1))
print(f"\nTrain edges: {int(train_mask.sum())}  pos={int(n_pos)} neg={int(n_neg)}  pos_weight={pos_weight:.2f}")
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

CHECKPOINT_PATH = f"{OUTPUT_DIR}/graphsage_collusion_checkpoint.pt"
N_EPOCHS = 60
best_val_prauc = -1.0
best_state = None
history = []

print(f"\nTraining for up to {N_EPOCHS} epochs (full-batch, transductive)...")
t_train0 = time.time()
for epoch in range(1, N_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    h = model.encode(x_t, edge_index_t)
    logits = model.edge_logits(h, edge_src_t, edge_dst_t)
    loss = loss_fn(logits[train_mask], y_t[train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        h = model.encode(x_t, edge_index_t)
        logits = model.edge_logits(h, edge_src_t, edge_dst_t)
        val_probs = torch.sigmoid(logits[val_mask]).numpy()
        val_y = y_t[val_mask].numpy()
        val_prauc = average_precision_score(val_y, val_probs)

    history.append({"epoch": epoch, "train_loss": loss.item(), "val_pr_auc": val_prauc})
    if val_prauc > best_val_prauc:
        best_val_prauc = val_prauc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"  epoch {epoch:3d}  train_loss={loss.item():.4f}  val_pr_auc={val_prauc:.4f}  "
              f"elapsed={time.time()-t_train0:.1f}s")

print(f"\nTraining done in {time.time()-t_train0:.1f}s. Best val PR-AUC: {best_val_prauc:.4f}")
model.load_state_dict(best_state)
torch.save({"model_state": best_state, "feat_dim": feat_dim}, CHECKPOINT_PATH)
pd.DataFrame(history).to_csv(f"{OUTPUT_DIR}/graphsage_training_history.csv", index=False)

# ============================================================
# EVALUATION
# ============================================================
model.eval()
with torch.no_grad():
    h = model.encode(x_t, edge_index_t)
    logits = model.edge_logits(h, edge_src_t, edge_dst_t)
    probs = torch.sigmoid(logits).numpy()

test_mask_np = test_mask.numpy()
test_probs = probs[test_mask_np]
test_y = y_ring[test_mask_np]
test_fraud_type = fraud_type_arr[test_mask_np]

test_pred = (test_probs >= 0.5).astype(int)
pr_auc = average_precision_score(test_y, test_probs)
f1 = f1_score(test_y, test_pred)
print(f"\n{'='*60}\nTEST SET RESULTS (GraphSAGE, collusion_ring vs everything else)\n{'='*60}")
print(f"PR-AUC: {pr_auc:.4f}   F1: {f1:.4f}")

ring_mask = test_fraud_type == "collusion_ring"
ring_recall = recall_score(test_y[ring_mask], test_pred[ring_mask]) if ring_mask.sum() else float("nan")
print(f"\ncollusion_ring recall (GraphSAGE): n={ring_mask.sum()} "
      f"caught={int(test_pred[ring_mask].sum())} recall={ring_recall:.4f} "
      f"mean_score={test_probs[ring_mask].mean():.4f}")
print("(compare to XGBoost's 54.0% recall on the SAME 3246 test rows, with no graph structure)")

for ft in ["none", "upcoding", "phantom_billing"]:
    m = test_fraud_type == ft
    if m.sum():
        print(f"  false-positive-style check on '{ft}': n={m.sum()} "
              f"flagged_as_ring={int(test_pred[m].sum())} ({test_pred[m].mean()*100:.2f}%) "
              f"mean_score={test_probs[m].mean():.4f}")

results_df = pd.DataFrame({
    "FRAUD_TYPE": test_fraud_type,
    "y_true_ring": test_y,
    "pred_ring": test_pred,
    "score": test_probs,
})
results_df.to_csv(f"{OUTPUT_DIR}/graphsage_test_predictions.csv", index=False)
print(f"\nSaved checkpoint to {CHECKPOINT_PATH}")
print(f"Saved test predictions to {OUTPUT_DIR}/graphsage_test_predictions.csv")
print(f"TOTAL elapsed: {time.time()-t0:.1f}s")
