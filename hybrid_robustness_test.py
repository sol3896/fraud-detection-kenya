"""
HYBRID ROBUSTNESS / STRESS TEST (v2) - replaces the stale robustness_test.py

WHY THIS REPLACES THE OLD SCRIPT
robustness_test.py (the original, in the repo root) targets a pipeline that
no longer exists: the OLD data file (no collusion_ring), LabelEncoder
instead of one-hot, SMOTEENN instead of SMOTENC, and a feature set that
still includes PATIENT_INCOME. Its noise+missingness methodology is sound
and worth keeping - in fact xgboost_model_v2_robust.py already folded a
version of it directly into training (missingness-aware training is WHY
that model exists). What was never stress-tested under noise is the thing
that actually ships now: the HYBRID XGBoost+GraphSAGE system with its TUNED
decision thresholds (t_xgb=0.85, t_graphsage=0.95, from
tune_hybrid_threshold.py). That is the system `app.py` runs live, so that is
what this script tests - end to end, using the already-trained, already-
saved models (no retraining, which also means every run is fast and exactly
reproducible against the checkpoints already on disk).

THREE STRESS CONDITIONS, evaluated on the identical held-out test split used
throughout this project (same random_state=42 train/val/test partition as
every other script):

  1. CLEAN - baseline, no corruption. Reproduces tune_hybrid_threshold.py's
     TEST numbers exactly, as a sanity check that nothing drifted.

  2. XGBOOST-SIDE NOISE+MISSINGNESS (5% / 10%) - identical `corrupt()`
     function to xgboost_model_v2_robust.py: Gaussian noise scaled to each
     numeric column's std, then that same fraction of cells set to NaN.
     Applied only to XGBoost's tabular features; GraphSAGE's score for each
     claim is untouched (it doesn't see these columns at all - this isolates
     how much the HYBRID system can still catch collusion rings purely via
     GraphSAGE when XGBoost's view of a claim gets degraded).

  3. MISSING GRAPH HISTORY (10% / 25%) - a different, arguably more
     realistic failure mode for a live relational model: a patient or
     provider who is genuinely new (first claim ever, or outside the graph
     snapshot used to train GraphSAGE) has NO precomputed embedding.
     graphsage_ring_score() already returns None for this case (verified
     during the app.py wiring step) rather than fabricating a score. This
     experiment randomly drops 10%/25% of TEST claims' GraphSAGE coverage
     (simulating "new to the system") and checks how much of the hybrid's
     lift the system loses when GraphSAGE can't see a claim at all, vs.
     falling back to XGBoost alone for those claims.

For every condition: overall recall/precision/F1, false-positive rate on
legitimate claims, and per-fraud-type recall - for XGBoost-alone, GraphSAGE-
alone (where available), and the TUNED hybrid rule.
"""
import gc
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.nn import SAGEConv

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE = os.path.expanduser("~/synthea_kenya")
DATA_PATH = f"{BASE}/output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
XGB_MODEL_PATH = f"{BASE}/output/model/xgboost_fraud_model_v2_robust.json"
GS_CHECKPOINT_PATH = f"{BASE}/output/model/graphsage_collusion_checkpoint.pt"
THRESH_PATH = f"{BASE}/output/model/hybrid_tuned_thresholds.txt"
OUTPUT_DIR = f"{BASE}/output/model"

print("=" * 70)
print("HYBRID ROBUSTNESS / STRESS TEST (XGBoost + GraphSAGE, tuned thresholds)")
print("=" * 70)

# ============================================================
# LOAD TUNED THRESHOLDS
# ============================================================
t_xgb, t_gs = 0.5, 0.5
if os.path.exists(THRESH_PATH):
    with open(THRESH_PATH) as f:
        for line in f:
            if line.startswith("t_xgb="):
                t_xgb = float(line.strip().split("=")[1])
            elif line.startswith("t_graphsage="):
                t_gs = float(line.strip().split("=")[1])
print(f"Tuned thresholds loaded: t_xgb={t_xgb}  t_graphsage={t_gs}")

# ============================================================
# LOAD DATA + REPRODUCE XGBOOST FEATURE ENGINEERING (identical to
# xgboost_model_v2_robust.py / tune_hybrid_threshold.py)
# ============================================================
print("\nLoading data...")
NEEDED_COLS = list(set([
    "BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES", "CLAIM_TO_BASE_RATIO",
    "AGE", "LENGTH_OF_STAY_HOURS", "ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY",
    "IS_FRAUD", "PRIMARY_CONDITION", "PROVIDER_ID", "PATIENT_ID", "SERVICE_DATE",
    "ENCOUNTER_ID", "FRAUD_TYPE", "GEOGRAPHIC_MISMATCH", "CLINICAL_INCONSISTENCY",
    "PROVIDER_LEVEL", "PROVIDER_COUNTY",
]))
df = pd.read_csv(DATA_PATH, usecols=lambda c: c in NEEDED_COLS)
print(f"Dataset shape: {df.shape}, fraud rate: {df['IS_FRAUD'].mean()*100:.2f}%")

categorical_features = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]
for col in categorical_features:
    df[col] = df[col].astype(str)
onehot_categories = {col: sorted(df[col].unique().tolist()) for col in categorical_features}
encoded_categoricals = [f"{col}_{cat}" for col in categorical_features for cat in onehot_categories[col]]


def onehot_encode(df_slice, cat_cols, categories):
    out = df_slice.drop(columns=cat_cols).copy()
    for col in cat_cols:
        values = df_slice[col].astype(str)
        for cat in categories[col]:
            out[f"{col}_{cat}"] = (values == cat).astype(int)
    return out


y_full = df["IS_FRAUD"]
idx_train, idx_temp = train_test_split(df.index, test_size=0.30, random_state=RANDOM_STATE, stratify=y_full)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_full.loc[idx_temp])
test_mask = df.index.isin(idx_test)
print(f"Train/Val/Test sizes: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

train_df = df.loc[idx_train]
# PROVIDER_Z_SCORE must be fit on train-LEGITIMATE rows only (this session's
# own leak fix, established earlier: mixing in known-fraud rows when
# computing a provider's "normal" cost baseline let the z-score leak the
# label). group_median (COST_RATIO) uses all train rows, matching
# xgboost_model_v2_robust.py / tune_hybrid_threshold.py exactly.
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
NOISY_COLUMNS = COST_FEATURES + RATIO_FEATURE + ["COST_RATIO", "PROVIDER_Z_SCORE"]

test_df = df.loc[test_mask].reset_index(drop=True)
X_test = onehot_encode(test_df[RAW_FEATURE_COLUMNS], categorical_features, onehot_categories)[FEATURE_COLUMNS]
y_test = test_df["IS_FRAUD"].values
fraud_type_test = test_df["FRAUD_TYPE"].values
patient_ids_test = test_df["PATIENT_ID"].astype(str).values
provider_ids_test = test_df["PROVIDER_ID"].astype(str).values

del train_df, median_lookup, mean_lookup, std_lookup, daily_counts
gc.collect()

print(f"Test set: {len(test_df)} rows, {y_test.sum()} fraud")

# ============================================================
# LOAD XGBOOST MODEL
# ============================================================
import xgboost as xgb
print("\nLoading XGBoost model...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)


def corrupt_tabular(X, noise_frac, seed):
    """Identical corruption function to xgboost_model_v2_robust.py."""
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


# ============================================================
# REBUILD GRAPHSAGE GRAPH (identical to graphsage_collusion.py post-fix:
# no degree feature, 33-dim) + LOAD CHECKPOINT
# ============================================================
print("\nRebuilding GraphSAGE bipartite graph (no-degree, 33-dim features)...")
graph_df = df[["PATIENT_ID", "PROVIDER_ID", "AGE", "GENDER", "PATIENT_COUNTY",
               "PROVIDER_COUNTY", "PROVIDER_LEVEL"]].copy()
patient_ids = sorted(graph_df["PATIENT_ID"].astype(str).unique())
provider_ids = sorted(graph_df["PROVIDER_ID"].astype(str).unique())
patient_idx = {pid: i for i, pid in enumerate(patient_ids)}
provider_idx = {pid: i + len(patient_ids) for i, pid in enumerate(provider_ids)}
n_patients, n_providers = len(patient_ids), len(provider_ids)
n_nodes = n_patients + n_providers

county_cats = sorted(set(graph_df["PATIENT_COUNTY"].astype(str)) | set(graph_df["PROVIDER_COUNTY"].astype(str)))
level_cats = sorted(graph_df["PROVIDER_LEVEL"].astype(str).unique())
n_county, n_level = len(county_cats), len(level_cats)
county_idx = {c: i for i, c in enumerate(county_cats)}
level_idx = {l: i for i, l in enumerate(level_cats)}
feat_dim = 2 + 1 + 2 + n_county + n_level

patient_age = graph_df.groupby(graph_df["PATIENT_ID"].astype(str))["AGE"].mean()
age_min, age_max = patient_age.min(), patient_age.max()
patient_gender = graph_df.groupby(graph_df["PATIENT_ID"].astype(str))["GENDER"].first()
patient_county_map = graph_df.groupby(graph_df["PATIENT_ID"].astype(str))["PATIENT_COUNTY"].first()
provider_county_map = graph_df.groupby(graph_df["PROVIDER_ID"].astype(str))["PROVIDER_COUNTY"].first()
provider_level_map = graph_df.groupby(graph_df["PROVIDER_ID"].astype(str))["PROVIDER_LEVEL"].first()

x = np.zeros((n_nodes, feat_dim), dtype=np.float32)
for pid in patient_ids:
    i = patient_idx[pid]
    x[i, 0] = 1.0
    age = patient_age.get(pid, (age_min + age_max) / 2)
    x[i, 2] = (age - age_min) / (age_max - age_min + 1e-9)
    g = str(patient_gender.get(pid, "M"))
    x[i, 3] = 1.0 if g == "M" else 0.0
    x[i, 4] = 1.0 if g == "F" else 0.0
    c = str(patient_county_map.get(pid, county_cats[0]))
    x[i, 5 + county_idx.get(c, 0)] = 1.0
for pid in provider_ids:
    i = provider_idx[pid]
    x[i, 1] = 1.0
    c = str(provider_county_map.get(pid, county_cats[0]))
    x[i, 5 + county_idx.get(c, 0)] = 1.0
    lvl = str(provider_level_map.get(pid, level_cats[0]))
    x[i, 5 + n_county + level_idx.get(lvl, 0)] = 1.0

edge_src, edge_dst = [], []
for pid, rid in zip(graph_df["PATIENT_ID"].astype(str), graph_df["PROVIDER_ID"].astype(str)):
    a, b = patient_idx[pid], provider_idx[rid]
    edge_src += [a, b]
    edge_dst += [b, a]
edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
x_t = torch.tensor(x, dtype=torch.float32)
print(f"Graph: {n_nodes} nodes ({n_patients} patients, {n_providers} providers), feat_dim={feat_dim}")


class GraphSAGEEdgeClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)
        self.dropout = nn.Dropout(0.2)
        self.edge_mlp = nn.Sequential(
            nn.Linear(out_dim * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def encode(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h

    def edge_logits(self, h, src, dst):
        e = torch.cat([h[src], h[dst]], dim=1)
        return self.edge_mlp(e).squeeze(-1)


print("Loading GraphSAGE checkpoint...")
ckpt = torch.load(GS_CHECKPOINT_PATH, weights_only=False, map_location="cpu")
assert ckpt["feat_dim"] == feat_dim, f"feat_dim mismatch: ckpt={ckpt['feat_dim']} rebuilt={feat_dim}"
gs_model = GraphSAGEEdgeClassifier(feat_dim)
gs_model.load_state_dict(ckpt["model_state"])
gs_model.eval()

test_patient_node = np.array([patient_idx[p] for p in patient_ids_test])
test_provider_node = np.array([provider_idx[p] for p in provider_ids_test])
src_t = torch.tensor(test_patient_node, dtype=torch.long)
dst_t = torch.tensor(test_provider_node, dtype=torch.long)

with torch.no_grad():
    h_clean = gs_model.encode(x_t, edge_index)
    gs_logits_clean = gs_model.edge_logits(h_clean, src_t, dst_t)
    gs_proba_clean = torch.sigmoid(gs_logits_clean).numpy()

print(f"GraphSAGE clean test scores computed. Mean score: {gs_proba_clean.mean():.4f}")


def graphsage_scores_with_corrupted_features(noise_frac, seed):
    """Corrupt GraphSAGE's NODE features (age/gender/county/level one-hots)
    with the same noise+missingness recipe as corrupt_tabular, then redo the
    encode() forward pass once over the whole (corrupted) graph and re-score
    the test edges. This tests whether GraphSAGE's relational signal
    survives messier demographic/attribute data (the kind of thing a real
    claims feed with incomplete patient registration would produce) -
    structural/topology info (edge_index) is untouched, since that's not a
    'noisy value', it's the hospital-visit record itself."""
    r = np.random.default_rng(seed)
    xc = x.copy()
    # numeric col 2 is age_norm; cols 3-4 gender; rest are one-hot county/level
    std = xc[:, 2].std()
    xc[:, 2] = xc[:, 2] + r.normal(0, std * noise_frac, size=n_nodes)
    xc[:, 2] = np.clip(xc[:, 2], 0.0, 1.0)
    missing_mask = r.random(n_nodes) < noise_frac
    xc[missing_mask, :] = 0.0  # simulate a node whose attributes are unknown
    xc_t = torch.tensor(xc, dtype=torch.float32)
    with torch.no_grad():
        h_c = gs_model.encode(xc_t, edge_index)
        logits_c = gs_model.edge_logits(h_c, src_t, dst_t)
        return torch.sigmoid(logits_c).numpy()


def apply_missing_graph_history(gs_scores, drop_frac, seed):
    """Simulate claims whose patient/provider GraphSAGE has no embedding for
    (new to the system). Returns gs_scores with drop_frac fraction set to
    None (np.nan) - mirrors graphsage_ring_score()'s real None-return path,
    letting the hybrid rule fall back to XGBoost-only for those rows."""
    r = np.random.default_rng(seed)
    out = gs_scores.copy().astype(float)
    mask = r.random(len(out)) < drop_frac
    out[mask] = np.nan
    return out


# ============================================================
# EVALUATION HELPER (mirrors tune_hybrid_threshold.py's evaluate())
# ============================================================
def evaluate(xgb_proba, gs_proba, y_true, fraud_types, label):
    xgb_pred = (xgb_proba >= t_xgb).astype(int)
    gs_available = ~np.isnan(gs_proba)
    gs_pred = np.zeros(len(gs_proba), dtype=int)
    gs_pred[gs_available] = (gs_proba[gs_available] >= t_gs).astype(int)
    hybrid_pred = ((xgb_pred == 1) | (gs_pred == 1)).astype(int)

    row = {"condition": label}
    for name, pred in [("xgboost_alone", xgb_pred), ("hybrid_tuned", hybrid_pred)]:
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        legit_mask = y_true == 0
        fp_rate = pred[legit_mask].mean() if legit_mask.sum() else float("nan")
        row[f"{name}_recall"] = recall
        row[f"{name}_precision"] = precision
        row[f"{name}_f1"] = f1
        row[f"{name}_fp_rate"] = fp_rate

    print(f"\n{'-'*70}\n{label}\n{'-'*70}")
    print(f"  XGBoost-alone : recall={row['xgboost_alone_recall']:.4f}  precision={row['xgboost_alone_precision']:.4f}  "
          f"f1={row['xgboost_alone_f1']:.4f}  FP-rate={row['xgboost_alone_fp_rate']*100:.3f}%")
    print(f"  Hybrid(tuned) : recall={row['hybrid_tuned_recall']:.4f}  precision={row['hybrid_tuned_precision']:.4f}  "
          f"f1={row['hybrid_tuned_f1']:.4f}  FP-rate={row['hybrid_tuned_fp_rate']*100:.3f}%")
    print(f"  GraphSAGE coverage this condition: {gs_available.mean()*100:.1f}% of claims had a usable score")

    print("  Per-fraud-type recall (hybrid vs xgboost-alone):")
    for ft in sorted(set(fraud_types)):
        if ft == "none":
            continue
        ft_mask = fraud_types == ft
        n = int(ft_mask.sum())
        if n == 0:
            continue
        h_recall = hybrid_pred[ft_mask].mean()
        x_recall = xgb_pred[ft_mask].mean()
        print(f"    {ft:<18} n={n:<6} xgb_recall={x_recall:.4f}  hybrid_recall={h_recall:.4f}")
        row[f"recall_{ft}_xgb"] = x_recall
        row[f"recall_{ft}_hybrid"] = h_recall
    return row


# ============================================================
# RUN ALL CONDITIONS
# ============================================================
results = []

# 1. CLEAN baseline
xgb_proba_clean = xgb_model.predict_proba(X_test)[:, 1]
results.append(evaluate(xgb_proba_clean, gs_proba_clean, y_test, fraud_type_test, "1. CLEAN (no corruption)"))

# 2. XGBoost-side noise+missingness (GraphSAGE untouched)
for frac in [0.05, 0.10]:
    X_noisy = corrupt_tabular(X_test, frac, seed=RANDOM_STATE)
    xgb_proba_noisy = xgb_model.predict_proba(X_noisy)[:, 1]
    results.append(evaluate(
        xgb_proba_noisy, gs_proba_clean, y_test, fraud_type_test,
        f"2. XGBoost tabular features corrupted: {int(frac*100)}% noise+missingness"
    ))

# 3. GraphSAGE node-feature noise+missingness (XGBoost untouched)
for frac in [0.05, 0.10]:
    gs_proba_noisy = graphsage_scores_with_corrupted_features(frac, seed=RANDOM_STATE)
    results.append(evaluate(
        xgb_proba_clean, gs_proba_noisy, y_test, fraud_type_test,
        f"3. GraphSAGE node features corrupted: {int(frac*100)}% noise+missingness"
    ))

# 4. Missing graph history (simulated "new to the system" claims)
for frac in [0.10, 0.25]:
    gs_proba_dropped = apply_missing_graph_history(gs_proba_clean, frac, seed=RANDOM_STATE)
    results.append(evaluate(
        xgb_proba_clean, gs_proba_dropped, y_test, fraud_type_test,
        f"4. Missing GraphSAGE history (simulated new patients/providers): {int(frac*100)}% of claims"
    ))

# 5. COMBINED worst case: 10% xgb-side noise + 10% graph-feature noise + 25% missing history
X_noisy10 = corrupt_tabular(X_test, 0.10, seed=RANDOM_STATE)
xgb_proba_combined = xgb_model.predict_proba(X_noisy10)[:, 1]
gs_proba_combined = graphsage_scores_with_corrupted_features(0.10, seed=RANDOM_STATE)
gs_proba_combined = apply_missing_graph_history(gs_proba_combined, 0.25, seed=RANDOM_STATE + 1)
results.append(evaluate(
    xgb_proba_combined, gs_proba_combined, y_test, fraud_type_test,
    "5. COMBINED WORST CASE (10% xgb noise + 10% graph-feature noise + 25% missing history)"
))

# ============================================================
# SAVE
# ============================================================
results_df = pd.DataFrame(results)
out_path = f"{OUTPUT_DIR}/hybrid_robustness_results.csv"
results_df.to_csv(out_path, index=False)

print(f"\n{'='*70}\nSUMMARY TABLE\n{'='*70}")
summary_cols = ["condition", "xgboost_alone_recall", "xgboost_alone_fp_rate",
                "hybrid_tuned_recall", "hybrid_tuned_precision", "hybrid_tuned_f1", "hybrid_tuned_fp_rate"]
print(results_df[summary_cols].to_string(index=False))
print(f"\nFull results saved to {out_path}")
