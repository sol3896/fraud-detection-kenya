"""
EXPORT GRAPH NEIGHBOURHOOD for the Claim Detail "relational view".

app.py can already score a (patient, provider) pair with GraphSAGE, but a
bare number doesn't show an auditor WHY the pattern looks like a ring. This
exports a compact adjacency summary from the same claims data the graph was
built from, so the live app can draw the claim's local neighbourhood:

  provider_top_patients[provider_id] = [[patient_id, n_claims], ...]  (top 25)
  patient_top_providers[patient_id]  = [[provider_id, n_claims], ...] (top 8)

Pure counts - no labels (IS_FRAUD / FRAUD_TYPE) are exported, so the
visualisation can't leak ground truth; it only shows structure plus the
model's own scores (computed live in the app).
"""
import os
import json
import pandas as pd

BASE = os.path.expanduser("~/synthea_kenya")
DATA_PATH = f"{BASE}/output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
OUT_PATH = f"{BASE}/output/model/graph_neighborhood.json"
TOP_PATIENTS_PER_PROVIDER = 25
TOP_PROVIDERS_PER_PATIENT = 8

df = pd.read_csv(DATA_PATH, usecols=["PATIENT_ID", "PROVIDER_ID"])
df["PATIENT_ID"] = df["PATIENT_ID"].astype(str)
df["PROVIDER_ID"] = df["PROVIDER_ID"].astype(str)
pairs = df.groupby(["PROVIDER_ID", "PATIENT_ID"]).size().rename("n").reset_index()
print(f"{len(pairs)} unique (provider, patient) pairs from {len(df)} claims")

prov_top = {}
for pid, g in pairs.sort_values("n", ascending=False).groupby("PROVIDER_ID"):
    prov_top[pid] = [[a, int(n)] for a, n in zip(g["PATIENT_ID"].head(TOP_PATIENTS_PER_PROVIDER), g["n"].head(TOP_PATIENTS_PER_PROVIDER))]

pat_top = {}
for pat, g in pairs.sort_values("n", ascending=False).groupby("PATIENT_ID"):
    pat_top[pat] = [[r, int(n)] for r, n in zip(g["PROVIDER_ID"].head(TOP_PROVIDERS_PER_PATIENT), g["n"].head(TOP_PROVIDERS_PER_PATIENT))]

with open(OUT_PATH, "w") as f:
    json.dump({"provider_top_patients": prov_top, "patient_top_providers": pat_top}, f, separators=(",", ":"))
print(f"Saved {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB): "
      f"{len(prov_top)} providers, {len(pat_top)} patients")
