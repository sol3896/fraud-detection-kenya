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

# Mirrors the real SHA/SHIF tariff-cap model in kenyanise.py's
# build_master_dataset(), so collusion_ring rows (which keep their real
# ENCOUNTER_TYPE/PRIMARY_CONDITION, only PROVIDER_ID/ORGANIZATION_ID/
# PROVIDER_LEVEL and patient identity get reassigned) end up with a
# coverage/copay that's consistent with every other row in the dataset,
# instead of the old flat random 50-90% coverage.
INPATIENT_BED_DAY_KES = {
    "Level 2 - Dispensary": 2240,
    "Level 3 - Health Centre": 2240,
    "Level 4 - Sub-District Hospital": 3360,
    "Level 5 - District Hospital": 3920,
    "Level 6 - National Referral Hospital": 4480,
    "Faith-Based Hospital": 3920,
    "Private Hospital": 4480,
    "Private Clinic": 3360,
}
INPATIENT_BED_DAY_DEFAULT_KES = 3360
MATERNITY_TARIFF_KES = {"normal_delivery": 10000, "c_section": 30000}
MATERNITY_CONDITIONS = {"Pregnant (finding)", "Pre-eclampsia (disorder)"}
CONDITION_OUTPATIENT_TARIFF_KES = {
    "Diabetes mellitus (disorder)": 4300,
    "Hypertension (disorder)": 2850,
    "Asthma (disorder)": 700,
}
GENERAL_OUTPATIENT_TARIFF_KES = 1500
PHF_COVERAGE_FRACTION_RANGE = (0.90, 1.00)
EMERGENCY_COVERAGE_FRACTION = 0.95
FALLBACK_COVERAGE_FRACTION_RANGE = (0.5, 0.9)

def redraw_cost(encounter_type, condition, provider_level, los_hours, seed_rng):
    tariff = SHA_TARIFFS.get(str(encounter_type).lower(), {"min": 1000, "max": 10000})
    base = round(seed_rng.uniform(tariff["min"], tariff["max"]), 2)
    claim = round(base * seed_rng.uniform(1.0, 1.3), 2)

    etype = str(encounter_type).lower()
    if condition in MATERNITY_CONDITIONS:
        tariff_kes = (
            MATERNITY_TARIFF_KES["c_section"]
            if condition == "Pre-eclampsia (disorder)"
            else MATERNITY_TARIFF_KES["normal_delivery"]
        )
        payer = min(claim, tariff_kes)
    elif etype == "inpatient":
        bed_rate = INPATIENT_BED_DAY_KES.get(provider_level, INPATIENT_BED_DAY_DEFAULT_KES)
        los_days = max(los_hours / 24, 1) if pd.notna(los_hours) else 1
        payer = min(claim, bed_rate * los_days)
    elif etype in ("wellness", "ambulatory", "virtual"):
        payer = claim * seed_rng.uniform(*PHF_COVERAGE_FRACTION_RANGE)
    elif etype == "emergency":
        payer = claim * EMERGENCY_COVERAGE_FRACTION
    elif etype in ("outpatient", "urgentcare"):
        tariff_kes = CONDITION_OUTPATIENT_TARIFF_KES.get(condition, GENERAL_OUTPATIENT_TARIFF_KES)
        payer = min(claim, tariff_kes)
    else:
        payer = claim * seed_rng.uniform(*FALLBACK_COVERAGE_FRACTION_RANGE)

    payer = round(payer, 2)
    copay = round(max(claim - payer, 0), 2)
    ratio = round(claim / base, 3) if base else 1.0
    return base, claim, payer, copay, ratio

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

# Provider -> ORGANIZATION_ID / PROVIDER_LEVEL lookup (first occurrence), so
# reassigned rows stay internally consistent (a claim at provider X should
# carry X's org and X's facility level, which the coverage model needs).
provider_org_lookup = df.drop_duplicates("PROVIDER_ID").set_index("PROVIDER_ID")["ORGANIZATION_ID"]
has_provider_level = "PROVIDER_LEVEL" in df.columns
if has_provider_level:
    provider_level_lookup = df.drop_duplicates("PROVIDER_ID").set_index("PROVIDER_ID")["PROVIDER_LEVEL"]
has_provider_county = "PROVIDER_COUNTY" in df.columns
if has_provider_county:
    provider_county_lookup = df.drop_duplicates("PROVIDER_ID").set_index("PROVIDER_ID")["PROVIDER_COUNTY"]

print(f"\nSelected {N_RING_PROVIDERS} ring providers: {list(ring_providers)}")

# Each ring provider's EXISTING encounter-type case mix, captured BEFORE any
# conversion happens. Root cause of the PROVIDER_Z_SCORE leak found by
# collusion_leakage_check.py: converted rows were assigned to ring providers
# uniformly at random, regardless of encounter type, so a provider who
# originally specialised in (say) inpatient claims suddenly absorbed ~1800
# claims spanning every encounter type in the dataset. PROVIDER_Z_SCORE is
# computed per-provider across ALL their claims (not per encounter type), so
# that case-mix shift - not cost magnitude - is what was leaking the ring
# structure into a tabular feature. Fixing it: provider assignment below is
# now stratified by each row's own ENCOUNTER_TYPE, weighted by how often
# each ring provider already saw that encounter type, so a ring provider's
# post-conversion case mix (and therefore its mean/std/z-score) stays close
# to what it looked like before any claims were reassigned to it.
provider_etype_profile = {}
for p in ring_providers:
    counts = df.loc[df["PROVIDER_ID"] == p, "ENCOUNTER_TYPE"].value_counts()
    total = counts.sum()
    provider_etype_profile[p] = (counts / total).to_dict() if total > 0 else {}
print("Ring provider encounter-type profiles (for case-mix-preserving assignment):")
for p in ring_providers:
    print(f"  {p}: {provider_etype_profile[p]}")

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
# Only convert rows whose ENCOUNTER_TYPE at least one ring provider has
# genuinely handled before. Without this, a type none of the 12 ring
# providers has ever seen (e.g. urgentcare/hospice here) has to fall back
# to a uniform provider choice in STEP 3 below, which forces mismatched,
# differently-costed claims onto providers that otherwise match their
# assigned type almost perfectly - exactly the residual PROVIDER_Z_SCORE
# leak collusion_leakage_check.py still found after the first stratified-
# assignment fix. Simplest closure: don't convert those uncovered-type
# rows into ring fraud at all.
covered_types = {
    et for et in df["ENCOUNTER_TYPE"].astype(str).unique()
    if any(profile.get(et, 0.0) > 0 for profile in provider_etype_profile.values())
}
uncovered_types = set(df["ENCOUNTER_TYPE"].astype(str).unique()) - covered_types
print(f"\nEncounter types with ring-provider coverage: {sorted(covered_types)}")
print(f"Encounter types with NO ring-provider history (excluded from conversion): {sorted(uncovered_types)}")

legit_idx = df.index[
    (df["IS_FRAUD"] == 0) & (df["ENCOUNTER_TYPE"].astype(str).isin(covered_types))
].to_numpy()
n_collusion = int(len(df) * COLLUSION_RATE)
collusion_idx = rng.choice(legit_idx, size=n_collusion, replace=False)
print(f"\nConverting {n_collusion} currently-legitimate rows to collusion_ring fraud "
      f"({n_collusion/len(df)*100:.2f}% of all rows)...")

# Randomly assign each selected row to one ring provider and one ring patient
# (uniform, independent draws -> creates a dense many-to-many bipartite clique,
# since 12 providers x 60 patients = 720 pairs share ~n_collusion/720 claims
# each, versus a handful at most for a real geography-driven pair).
# Stratified-by-encounter-type provider assignment (see profile comment
# above) - a FLOOR keeps every provider reachable even for an encounter
# type it has never personally seen, so the bipartite clique still spans
# all 12 providers x 60 patients densely; it just no longer forces a
# uniformly-random case-mix shift onto each provider.
candidate_etypes = df.loc[collusion_idx, "ENCOUNTER_TYPE"].astype(str).to_numpy()
assigned_provider = np.empty(n_collusion, dtype=object)
for et in np.unique(candidate_etypes):
    mask = candidate_etypes == et
    raw_weights = np.array([provider_etype_profile[p].get(et, 0.0) for p in ring_providers])
    assert raw_weights.sum() > 0, f"encounter type {et} slipped through despite coverage filter"
    if raw_weights.sum() > 0:
        # At least one ring provider has genuinely handled this encounter
        # type before - restrict assignment to exactly those providers
        # (weighted by their historical share), with NO floor added. A
        # small additive floor was tried first but still let rare,
        # wildly-mismatched-cost claims (e.g. an emergency claim landing on
        # a wellness-only specialist) leak into niche providers often
        # enough, across the full 137k-row injection volume, to keep
        # skewing their aggregate mean/std - which is exactly the
        # PROVIDER_Z_SCORE leak this stratification is meant to close.
        weights = raw_weights / raw_weights.sum()
    else:
        # No ring provider has ever handled this encounter type - there is
        # no non-mismatched choice available, so fall back to uniform
        # (unavoidable small leak, limited to these rare types only).
        weights = np.full(N_RING_PROVIDERS, 1.0 / N_RING_PROVIDERS)
    assigned_provider[mask] = rng.choice(ring_providers, size=int(mask.sum()), p=weights)
assigned_patient_slot = rng.integers(0, N_RING_PATIENTS, size=n_collusion)

df.loc[collusion_idx, "PROVIDER_ID"] = assigned_provider
df.loc[collusion_idx, "ORGANIZATION_ID"] = pd.Series(assigned_provider).map(provider_org_lookup).values
if has_provider_level:
    df.loc[collusion_idx, "PROVIDER_LEVEL"] = pd.Series(assigned_provider).map(provider_level_lookup).values
if has_provider_county:
    df.loc[collusion_idx, "PROVIDER_COUNTY"] = pd.Series(assigned_provider).map(provider_county_lookup).values

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

# Recompute GEOGRAPHIC_MISMATCH / CLINICAL_INCONSISTENCY for ring rows -
# both PATIENT_COUNTY and GENDER were just reassigned above (to the ring
# patient's demographics) and PROVIDER_COUNTY to the ring provider's county,
# so the values fix_dataset.py computed earlier (before this script ran)
# are now stale for exactly these rows.
if has_provider_county and "GEOGRAPHIC_MISMATCH" in df.columns:
    df.loc[collusion_idx, "GEOGRAPHIC_MISMATCH"] = (
        df.loc[collusion_idx, "PATIENT_COUNTY"] != df.loc[collusion_idx, "PROVIDER_COUNTY"]
    ).astype(int)
if "CLINICAL_INCONSISTENCY" in df.columns:
    df.loc[collusion_idx, "CLINICAL_INCONSISTENCY"] = (
        (df.loc[collusion_idx, "GENDER"] == "M")
        & df.loc[collusion_idx, "PRIMARY_CONDITION"].isin(MATERNITY_CONDITIONS)
    ).astype(int)

# Redraw costs from the row's OWN encounter type's normal tariff band, so a
# ring provider's cost distribution looks statistically identical to any
# other provider seeing that same mix of encounter types - the only real
# anomaly left is the density of the (provider, patient) connections, not
# anything a per-row or per-provider aggregate feature can pick up. Coverage
# is computed with the same real SHA/SHIF tariff-cap rules as every other
# row (condition-specific outpatient tariff, inpatient bed-day rate by the
# REASSIGNED provider's level, maternity tariff, PHF, ECCIF), so ring rows
# don't carry an inconsistent flat-percentage coverage that would leak the
# collusion signal back into tabular PAYER_COVERAGE_KES/PATIENT_COPAY_KES.
encounter_types = df.loc[collusion_idx, "ENCOUNTER_TYPE"].values
conditions = df.loc[collusion_idx, "PRIMARY_CONDITION"].values if "PRIMARY_CONDITION" in df.columns else [None] * n
provider_levels = (
    df.loc[collusion_idx, "PROVIDER_LEVEL"].values if has_provider_level else [None] * n
)
los_hours_vals = (
    df.loc[collusion_idx, "LENGTH_OF_STAY_HOURS"].values
    if "LENGTH_OF_STAY_HOURS" in df.columns else [None] * n
)
new_base, new_claim, new_payer, new_copay, new_ratio = [], [], [], [], []
for et, cond, lvl, los in zip(encounter_types, conditions, provider_levels, los_hours_vals):
    b, c, p, cp, r = redraw_cost(et, cond, lvl, los, rng)
    new_base.append(b); new_claim.append(c); new_payer.append(p)
    new_copay.append(cp); new_ratio.append(r)

df.loc[collusion_idx, "BASE_COST_KES"] = new_base
df.loc[collusion_idx, "CLAIM_AMOUNT_KES"] = new_claim
df.loc[collusion_idx, "PAYER_COVERAGE_KES"] = new_payer
if "PATIENT_COPAY_KES" in df.columns:
    df.loc[collusion_idx, "PATIENT_COPAY_KES"] = new_copay
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
