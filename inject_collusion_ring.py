"""
INJECT RELATIONAL ("COLLUSION RING") FRAUD

Motivation: robustness_test.py's A2 ablation showed the existing fraud
(upcoding, phantom_billing) is 100% cost-magnitude - drop the cost fields
and PR-AUC collapses to the random-guess floor (0.127). That means there is
currently NO graph-structured fraud pattern anywhere in this dataset, so a
GraphSAGE GNN built on it would have nothing relational to find - it would
just re-discover the same cost signal XGBoost already found, through a
harder-to-explain architecture. This script fixes that gap by adding a
third fraud type that is deliberately NOT a cost anomaly.

COLLUSION_RING pattern: a small fixed "ring" of N_RING_PROVIDERS providers
and N_RING_PATIENTS patient identities. A subset of currently-legitimate
claims are relabelled so their PROVIDER_ID and patient identity/demographics
are reassigned into this ring (drawn uniformly at random from the ring
pools), while every cost/clinical field on the row (BASE_COST_KES,
CLAIM_AMOUNT_KES, PAYER_COVERAGE_KES, CLAIM_TO_BASE_RATIO,
LENGTH_OF_STAY_HOURS, ENCOUNTER_TYPE, diagnosis/procedure codes, dates) is
left untouched. The result: individually, a ring claim looks financially
unremarkable (normal cost for its encounter type) - but structurally, the
bipartite patient-provider graph now has a small, abnormally dense clique
(12 providers x 60 patients getting orders of magnitude more repeat
interactions than any real geography-driven patient-provider relationship
would produce). That density is invisible to a row-wise tabular model like
XGBoost, which only ever sees one claim at a time - it is exactly the kind
of pattern a graph neural network (GraphSAGE) aggregating neighbourhood
information is designed to catch. This makes the "hybrid XGBoost + GNN"
premise of the thesis genuinely justified: each model should now be good
at catching a DIFFERENT fraud type.

Output: a new file (the original master_claims_kenya_clean.csv is left
untouched so existing results remain reproducible).
"""

import pandas as pd
import numpy as np

RANDOM_STATE = 42
N_RING_PROVIDERS = 12
N_RING_PATIENTS = 60
COLLUSION_RATE = 0.04  # fraction of TOTAL rows converted to collusion_ring fraud
MIN_PROVIDER_VOLUME = 50  # only pick "real" providers with enough existing claims

rng = np.random.default_rng(RANDOM_STATE)

# Same SHA tariff bands as kenyanise.py, used to redraw costs for ring rows so
# ring providers' cost distribution looks like a NORMAL provider's, not an
# artifact of merging claims from many unrelated original providers. Without
# this, PROVIDER_Z_SCORE/PROVIDER_DAILY_VOLUME leak the ring structure back
# into tabular features, which defeats the point of a graph-only anomaly.
SHA_TARIFFS = {
    "wellness":     {"min": 500,   "max": 2000},
    "ambulatory":   {"min": 1000,  "max": 5000},
    "outpatient":   {"min": 1500,  "max": 8000},
    "inpatient":    {"min": 15000, "max": 80000},
    "emergency":    {"min": 5000,  "max": 30000},
    "urgentcare":   {"min": 2000,  "max": 10000},
    "virtual":      {"min": 500,   "max": 1500},
}

def redraw_cost(encounter_type, seed_rng):
    tariff = SHA_TARIFFS.get(str(encounter_type).lower(), {"min": 1000, "max": 10000})
    base = round(seed_rng.uniform(tariff["min"], tariff["max"]), 2)
    claim = round(base * seed_rng.uniform(1.0, 2.5), 2)
    payer = round(claim * seed_rng.uniform(0.5, 0.9), 2)
    ratio = round(claim / base, 3) if base else 1.0
    return base, claim, payer, ratio

IN_PATH = "output/kenyanised/master_claims_kenya_clean.csv"
OUT_PATH = "output/kenyanised/master_claims_kenya_with_collusion.csv"

print("Loading master dataset...")
df = pd.read_csv(IN_PATH)
print(f"Loaded {len(df)} rows. Existing fraud breakdown:")
print(df["FRAUD_TYPE"].value_counts())

demographic_cols = [
    "PATIENT_ID", "PATIENT_FIRST_NAME", "PATIENT_LAST_NAME", "GENDER",
    "AGE", "PATIENT_CITY", "PATIENT_COUNTY", "PATIENT_INCOME",
]

# ============================================================
# STEP 1 - pick ring providers (real, established providers)
# ============================================================
provider_counts = df["PROVIDER_ID"].value_counts()
eligible_providers = provider_counts[provider_counts >= MIN_PROVIDER_VOLUME].index.to_numpy()
ring_providers = rng.choice(eligible_providers, size=N_RING_PROVIDERS, replace=False)

# Provider -> ORGANIZATION_ID lookup (first occurrence), so reassigned rows
# stay internally consistent (a claim at provider X should carry X's org).
provider_org_lookup = df.drop_duplicates("PROVIDER_ID").set_index("PROVIDER_ID")["ORGANIZATION_ID"]

print(f"\nSelected {N_RING_PROVIDERS} ring providers: {list(ring_providers)}")

# ============================================================
# STEP 2 - build a small fixed pool of ring patient identities
# ============================================================
sample_rows = df.sample(n=N_RING_PATIENTS, random_state=RANDOM_STATE)
ring_patients = []
for i, (_, row) in enumerate(sample_rows.iterrows()):
    ring_patients.append({
        "PATIENT_ID": f"RING-PATIENT-{i:03d}",
        "PATIENT_FIRST_NAME": row["PATIENT_FIRST_NAME"],
        "PATIENT_LAST_NAME": row["PATIENT_LAST_NAME"],
        "GENDER": row["GENDER"],
        "AGE": row["AGE"],
        "PATIENT_CITY": row["PATIENT_CITY"],
        "PATIENT_COUNTY": row["PATIENT_COUNTY"],
        "PATIENT_INCOME": row["PATIENT_INCOME"],
    })
ring_patients_df = pd.DataFrame(ring_patients)
print(f"Built {N_RING_PATIENTS} synthetic ring-patient identities.")

# ============================================================
# STEP 3 - select currently-legitimate rows to convert
# ============================================================
legit_idx = df.index[df["IS_FRAUD"] == 0].to_numpy()
n_collusion = int(len(df) * COLLUSION_RATE)
collusion_idx = rng.choice(legit_idx, size=n_collusion, replace=False)
print(f"\nConverting {n_collusion} currently-legitimate rows to collusion_ring fraud "
      f"({n_collusion/len(df)*100:.2f}% of all rows)...")

# Randomly assign each selected row to one ring provider and one ring patient
# (uniform, independent draws -> creates a dense many-to-many bipartite clique,
# since 12 providers x 60 patients = 720 pairs share ~n_collusion/720 claims
# each, versus a handful at most for a real geography-driven pair).
assigned_provider = rng.choice(ring_providers, size=n_collusion)
assigned_patient_slot = rng.integers(0, N_RING_PATIENTS, size=n_collusion)

df.loc[collusion_idx, "PROVIDER_ID"] = assigned_provider
df.loc[collusion_idx, "ORGANIZATION_ID"] = pd.Series(assigned_provider).map(provider_org_lookup).values

ring_patient_records = ring_patients_df.iloc[assigned_patient_slot].reset_index(drop=True)
for col in demographic_cols:
    df.loc[collusion_idx, col] = ring_patient_records[col].values

# Jitter AGE/PATIENT_INCOME per claim so they don't repeat as EXACT values
# across thousands of claims (a real patient's claims wouldn't all carry the
# identical income to the cent, and 60 fixed values repeating ~thousands of
# times each is a trivial tabular fingerprint - it lets a row-wise model spot
# collusion_ring without any graph reasoning at all, defeating the point of
# this fraud type). PATIENT_ID itself stays exactly constant per ring patient
# (that's not a model feature - it's only used to build graph edges later),
# so the underlying same-patient identity signal is preserved for GraphSAGE,
# while the tabular AGE/INCOME columns no longer act as a lookup key.
n = len(collusion_idx)
income_jitter = rng.uniform(0.90, 1.10, size=n)
df.loc[collusion_idx, "PATIENT_INCOME"] = (
    df.loc[collusion_idx, "PATIENT_INCOME"].to_numpy() * income_jitter
).round(2)
age_jitter = rng.integers(-2, 3, size=n)  # -2..+2 years
df.loc[collusion_idx, "AGE"] = (
    df.loc[collusion_idx, "AGE"].to_numpy() + age_jitter
).clip(min=0)

# Redraw costs from the row's OWN encounter type's normal tariff band, so a
# ring provider's cost distribution looks statistically identical to any
# other provider seeing that same mix of encounter types - the only real
# anomaly left is the density of the (provider, patient) connections, not
# anything a per-row or per-provider aggregate feature can pick up.
encounter_types = df.loc[collusion_idx, "ENCOUNTER_TYPE"].values
new_base, new_claim, new_payer, new_ratio = [], [], [], []
for et in encounter_types:
    b, c, p, r = redraw_cost(et, rng)
    new_base.append(b); new_claim.append(c); new_payer.append(p); new_ratio.append(r)

df.loc[collusion_idx, "BASE_COST_KES"] = new_base
df.loc[collusion_idx, "CLAIM_AMOUNT_KES"] = new_claim
df.loc[collusion_idx, "PAYER_COVERAGE_KES"] = new_payer
df.loc[collusion_idx, "CLAIM_TO_BASE_RATIO"] = new_ratio

df.loc[collusion_idx, "IS_FRAUD"] = 1
df.loc[collusion_idx, "FRAUD_TYPE"] = "collusion_ring"

# ============================================================
# VERIFY + SAVE
# ============================================================
print(f"\nNew fraud breakdown:")
print(df["FRAUD_TYPE"].value_counts())
print(f"New overall fraud rate: {df['IS_FRAUD'].mean()*100:.2f}%")

ring_mask = df["FRAUD_TYPE"] == "collusion_ring"
pair_counts = df[ring_mask].groupby(["PROVIDER_ID", "PATIENT_ID"]).size()
print(f"\nRing density check: {ring_mask.sum()} collusion claims across "
      f"{df[ring_mask]['PROVIDER_ID'].nunique()} providers x "
      f"{df[ring_mask]['PATIENT_ID'].nunique()} patients")
print(f"Claims per (provider, patient) pair in the ring - mean {pair_counts.mean():.1f}, "
      f"min {pair_counts.min()}, max {pair_counts.max()}")

non_ring_pairs = df[~ring_mask].groupby(["PROVIDER_ID", "PATIENT_ID"]).size()
print(f"For comparison, claims per (provider, patient) pair OUTSIDE the ring - "
      f"mean {non_ring_pairs.mean():.2f}, max {non_ring_pairs.max()}")

# Sanity check: cost fields for collusion rows should look like normal
# (non-fraud) claims of the same encounter type, not inflated like upcoding.
print("\nCLAIM_TO_BASE_RATIO by fraud type (sanity check - collusion_ring "
      "should look like 'none', not like upcoding):")
print(df.groupby("FRAUD_TYPE")["CLAIM_TO_BASE_RATIO"].describe()[["mean", "50%", "max"]])

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
