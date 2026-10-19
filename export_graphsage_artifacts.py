"""
Export GraphSAGE as plain-numpy-friendly artifacts for the live dashboard
(app.py), so the dashboard does NOT need a torch/torch_geometric
dependency at all - it only needs a per-node embedding lookup (computed
ONCE here, offline) and the tiny edge-MLP's weights, which are just two
small matrices applied by hand with numpy.

This mirrors export_model_artifacts.py's philosophy exactly: training-
time machinery stays in the training script, the live app gets a small,
self-contained JSON it can score from without re-deriving anything.

LIMITATION this makes explicit (and app.py surfaces to the user): a brand
new patient or provider with no claims in the training graph has no
embedding, so GraphSAGE has nothing to say about them yet - which is
realistic, not a bug. A collusion ring is, by definition, a pattern that
only becomes visible after repeated interactions accumulate; the hybrid
system correctly falls back to XGBoost's cost-based read alone until a
new node's relational history exists.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

DATA_PATH = "output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
GS_CHECKPOINT_PATH = "output/model/graphsage_collusion_checkpoint.pt"
THRESHOLDS_PATH = "output/model/hybrid_tuned_thresholds.txt"
OUT_PATH = "output/model/graphsage_artifacts.json"

print("Loading data...")
df = pd.read_csv(DATA_PATH, usecols=[
    "PATIENT_ID", "PROVIDER_ID", "GENDER", "AGE", "PATIENT_COUNTY",
    "PROVIDER_COUNTY", "PROVIDER_LEVEL",
])

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


print("Loading GraphSAGE checkpoint...")
ckpt = torch.load(GS_CHECKPOINT_PATH, weights_only=False)
assert ckpt["feat_dim"] == feat_dim
model = GraphSAGEEdgeClassifier(in_dim=feat_dim)
model.load_state_dict(ckpt["model_state"])
model.eval()

print("Computing node embeddings (one forward pass over the whole graph)...")
with torch.no_grad():
    h = model.encode(x_t, edge_index_t).numpy()

patient_embeddings = {str(pid): h[node].tolist() for pid, node in patient_node_id.items()}
provider_embeddings = {str(pid): h[node].tolist() for pid, node in provider_node_id.items()}

# edge_mlp weights, as plain lists - app.py applies these by hand with
# numpy (matmul + ReLU + matmul + sigmoid), no torch needed at inference.
lin1 = model.edge_mlp[0]
lin2 = model.edge_mlp[3]
edge_mlp_weights = {
    "W1": lin1.weight.detach().numpy().tolist(),  # (64, 64)
    "b1": lin1.bias.detach().numpy().tolist(),     # (64,)
    "W2": lin2.weight.detach().numpy().tolist(),   # (1, 64)
    "b2": lin2.bias.detach().numpy().tolist(),      # (1,)
}

t_xgb, t_graphsage = 0.5, 0.5
try:
    with open(THRESHOLDS_PATH) as f:
        for line in f:
            k, v = line.strip().split("=")
            if k == "t_xgb":
                t_xgb = float(v)
            elif k == "t_graphsage":
                t_graphsage = float(v)
    print(f"Loaded tuned thresholds: t_xgb={t_xgb} t_graphsage={t_graphsage}")
except FileNotFoundError:
    print("No tuned thresholds file found, defaulting to 0.5/0.5")

artifacts = {
    "patient_embeddings": patient_embeddings,
    "provider_embeddings": provider_embeddings,
    "edge_mlp_weights": edge_mlp_weights,
    "embedding_dim": int(h.shape[1]),
    "t_xgb": t_xgb,
    "t_graphsage": t_graphsage,
    "n_patients_with_history": len(patient_embeddings),
    "n_providers_with_history": len(provider_embeddings),
}
with open(OUT_PATH, "w") as f:
    json.dump(artifacts, f)
print(f"\nSaved {OUT_PATH}")
print(f"  {len(patient_embeddings)} patients with graph history, "
      f"{len(provider_embeddings)} providers with graph history")
import os
print(f"  file size: {os.path.getsize(OUT_PATH)/1e6:.1f} MB")
