import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ============================================================
# PATHS
# ============================================================
INPUT_DIR = os.path.expanduser("~/synthea_kenya/output/csv")
OUTPUT_DIR = os.path.expanduser("~/synthea_kenya/output/kenyanised")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# KENYAN REFERENCE DATA
# ============================================================

KENYAN_COUNTIES = [
    "Nairobi", "Mombasa", "Nakuru", "Kiambu", "Kisumu",
    "Machakos", "Meru", "Kajiado", "Uasin Gishu", "Nyeri",
    "Kakamega", "Bungoma", "Kilifi", "Murang'a", "Nyandarua",
    "Kericho", "Bomet", "Migori", "Homa Bay", "Siaya"
]

KENYAN_TOWNS = {
    "Nairobi": ["Nairobi CBD", "Westlands", "Kibera", "Eastleigh", "Karen"],
    "Mombasa": ["Mombasa Island", "Nyali", "Likoni", "Bamburi", "Changamwe"],
    "Nakuru": ["Nakuru Town", "Naivasha", "Gilgil", "Molo", "Njoro"],
    "Kiambu": ["Thika", "Ruiru", "Kiambu Town", "Limuru", "Kikuyu"],
    "Kisumu": ["Kisumu CBD", "Kondele", "Nyalenda", "Mamboleo", "Ahero"],
    "Machakos": ["Machakos Town", "Athi River", "Kangundo", "Matuu", "Wote"],
    "Meru": ["Meru Town", "Nkubu", "Maua", "Timau", "Chuka"],
    "Kajiado": ["Ngong", "Ongata Rongai", "Kajiado Town", "Kitengela", "Namanga"],
    "Uasin Gishu": ["Eldoret", "Turbo", "Burnt Forest", "Moiben", "Ziwa"],
    "Nyeri": ["Nyeri Town", "Karatina", "Othaya", "Mukurweini", "Tetu"]
}

KENYAN_FIRST_NAMES_M = [
    "Brian", "Kevin", "Dennis", "Michael", "Joseph", "David", "Peter",
    "James", "John", "Paul", "Samuel", "Daniel", "George", "Patrick",
    "Emmanuel", "Victor", "Alex", "Moses", "Simon", "Philip",
    "Kipchoge", "Otieno", "Kamau", "Njoroge", "Mwangi", "Odhiambo"
]

KENYAN_FIRST_NAMES_F = [
    "Faith", "Grace", "Mary", "Anne", "Esther", "Mercy", "Joyce",
    "Catherine", "Susan", "Elizabeth", "Sharon", "Lydia", "Priscilla",
    "Caroline", "Beatrice", "Wanjiru", "Akinyi", "Chebet", "Njeri",
    "Wambui", "Mutua", "Adhiambo", "Wanjiku", "Nduta", "Waweru"
]

KENYAN_LAST_NAMES = [
    "Kamau", "Otieno", "Mwangi", "Odhiambo", "Njoroge", "Kimani",
    "Ochieng", "Waweru", "Mutua", "Kipchoge", "Achieng", "Kariuki",
    "Owino", "Gitau", "Mugo", "Ogola", "Ndungu", "Auma", "Chebet",
    "Rotich", "Koech", "Langat", "Kiptoo", "Yego", "Bett",
    "Wanjiru", "Adhiambo", "Njoki", "Wairimu", "Gathoni"
]

# Kenyan disease conditions with SNOMED codes
KENYAN_CONDITIONS = [
    {"CODE": "61462000", "DESCRIPTION": "Malaria (disorder)"},
    {"CODE": "56717001", "DESCRIPTION": "Tuberculosis (disorder)"},
    {"CODE": "86406008", "DESCRIPTION": "Human immunodeficiency virus infection"},
    {"CODE": "38341003", "DESCRIPTION": "Hypertension (disorder)"},
    {"CODE": "73211009", "DESCRIPTION": "Diabetes mellitus (disorder)"},
    {"CODE": "195967001", "DESCRIPTION": "Asthma (disorder)"},
    {"CODE": "422034002", "DESCRIPTION": "Diabetic retinopathy (disorder)"},
    {"CODE": "267036007", "DESCRIPTION": "Anaemia (disorder)"},
    {"CODE": "77386006",  "DESCRIPTION": "Pregnant (finding)"},
    {"CODE": "398254007", "DESCRIPTION": "Pre-eclampsia (disorder)"},
    {"CODE": "40930008",  "DESCRIPTION": "Hypothyroidism (disorder)"},
    {"CODE": "235595009", "DESCRIPTION": "Typhoid fever (disorder)"},
]

# SHA tariff rates in KES by encounter type (used as the "list price"/billed
# amount band before the real SHA/SHIF benefit tariffs cap what gets covered)
SHA_TARIFFS = {
    "wellness":     {"min": 500,   "max": 2000},
    "ambulatory":   {"min": 1000,  "max": 5000},
    "outpatient":   {"min": 1500,  "max": 8000},
    "inpatient":    {"min": 15000, "max": 80000},
    "emergency":    {"min": 5000,  "max": 30000},
    "urgentcare":   {"min": 2000,  "max": 10000},
    "virtual":      {"min": 500,   "max": 1500},
}

# Kenyan provider types
PROVIDER_LEVELS = [
    "Level 2 - Dispensary",
    "Level 3 - Health Centre",
    "Level 4 - Sub-District Hospital",
    "Level 5 - District Hospital",
    "Level 6 - National Referral Hospital",
    "Faith-Based Hospital",
    "Private Hospital",
    "Private Clinic"
]

# Facility levels that bill above tariff more often (private practice, not
# bound to the public/faith-based direct-billing arrangement with SHA)
PRIVATE_PROVIDER_LEVELS = ["Private Hospital", "Private Clinic"]

# ============================================================
# REAL GAZETTED SHA / SHIF BENEFIT TARIFFS (KES)
# Source: SHA benefit tariff schedule (sha.go.ke/benefit-tariffs) and the
# Oct 2024 MOH gazette notice. These are what SHA actually pays out per
# claim, independent of what a facility bills — the coverage model below
# caps PAYER_COVERAGE_KES at these tariffs rather than at a flat % of the
# claim, which is how Kenyan hospitals are actually reimbursed.
# ============================================================

# SHIF inpatient bed-day rate by facility level (Levels 3-6, confirmed exact).
# Level 2 dispensaries don't normally admit inpatients, so they're floored at
# the Level 3 rate; faith-based/private facilities are mapped to the public
# level they are contractually closest to for tariff purposes.
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
INPATIENT_BED_DAY_DEFAULT_KES = 3360  # fallback if level is somehow missing

# Maternity tariff (Levels 2-6)
MATERNITY_TARIFF_KES = {
    "normal_delivery": 10000,
    "c_section": 30000,
}

# SHIF outpatient diagnostics/management tariffs for conditions that map
# directly onto a real benefit category. Conditions not in this map fall
# back to GENERAL_OUTPATIENT_TARIFF_KES.
CONDITION_OUTPATIENT_TARIFF_KES = {
    "Diabetes mellitus (disorder)": 4300,
    "Hypertension (disorder)": 2850,
    "Asthma (disorder)": 700,
}
GENERAL_OUTPATIENT_TARIFF_KES = 1500

# PHF (Primary Healthcare Fund) covers wellness/ambulatory/virtual visits
# near-fully — this is the "no bill to you" capitation layer.
PHF_COVERAGE_FRACTION_RANGE = (0.90, 1.00)

# ECCIF (Emergency, Chronic & Critical Illness Fund) covers emergencies
# generously regardless of SHIF contribution status.
EMERGENCY_COVERAGE_FRACTION = 0.95

# Conditions that trigger the maternity tariff instead of a generic
# inpatient/outpatient one when they're the primary condition on the claim.
MATERNITY_CONDITIONS = {"Pregnant (finding)", "Pre-eclampsia (disorder)"}

# Fallback coverage fraction for any encounter type/condition combination
# not covered by a specific tariff rule above.
FALLBACK_COVERAGE_FRACTION_RANGE = (0.5, 0.9)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_kenyan_town(county):
    if county in KENYAN_TOWNS:
        return random.choice(KENYAN_TOWNS[county])
    return county + " Town"

def get_kenyan_name(gender):
    if gender == "M":
        first = random.choice(KENYAN_FIRST_NAMES_M)
    else:
        first = random.choice(KENYAN_FIRST_NAMES_F)
    last = random.choice(KENYAN_LAST_NAMES)
    return first, last

def get_sha_cost(encounter_class):
    encounter_class = str(encounter_class).lower()
    if encounter_class in SHA_TARIFFS:
        tariff = SHA_TARIFFS[encounter_class]
    else:
        tariff = {"min": 1000, "max": 10000}
    return round(random.uniform(tariff["min"], tariff["max"]), 2)

# ============================================================
# KENYANISE PATIENTS
# ============================================================

def kenyanise_patients():
    print("Kenyanising patients...")
    df = pd.read_csv(f"{INPUT_DIR}/patients.csv")

    for idx, row in df.iterrows():
        # Assign Kenyan county
        county = random.choice(KENYAN_COUNTIES)
        df.at[idx, "STATE"] = "Kenya"
        df.at[idx, "COUNTY"] = county
        df.at[idx, "CITY"] = get_kenyan_town(county)
        df.at[idx, "ZIP"] = random.randint(100, 99999)
        # Replace US coordinates with Kenya bounding box
        df.at[idx, "LAT"] = round(random.uniform(-4.67, 4.62), 6)
        df.at[idx, "LON"] = round(random.uniform(33.91, 41.90), 6)

        # Replace names
        gender = row["GENDER"]
        first, last = get_kenyan_name(gender)
        df.at[idx, "FIRST"] = first
        df.at[idx, "LAST"] = last
        df.at[idx, "MIDDLE"] = ""

        # Replace SSN with Kenyan National ID format
        df.at[idx, "SSN"] = str(random.randint(10000000, 39999999))

        # Replace race and ethnicity
        df.at[idx, "RACE"] = "black"
        df.at[idx, "ETHNICITY"] = "kenyan"

        # Replace birthplace
        df.at[idx, "BIRTHPLACE"] = get_kenyan_town(county) + " " + county + " Kenya"

        # Replace address
        df.at[idx, "ADDRESS"] = str(random.randint(1, 999)) + " " + random.choice([
            "Mombasa Road", "Thika Road", "Ngong Road", "Waiyaki Way",
            "Jogoo Road", "Langata Road", "Uhuru Highway", "Tom Mboya Street"
        ])

    # Remove US-specific columns
    df = df.drop(columns=["PASSPORT", "DRIVERS", "FIPS"], errors="ignore")

    df.to_csv(f"{OUTPUT_DIR}/patients_kenya.csv", index=False)
    print(f"Done. {len(df)} patients saved.")
    return df

# ============================================================
# KENYANISE ENCOUNTERS
# ============================================================

def kenyanise_encounters():
    print("Kenyanising encounters...")
    df = pd.read_csv(f"{INPUT_DIR}/encounters.csv")

    # Replace costs with SHA tariff amounts in KES. The billed amount
    # (CLAIM_AMOUNT_KES) gets only a small realistic markup over the base
    # cost here — the big markups seen in the data should come from actual
    # fraud (upcoding), injected later, not from this baseline draw.
    # PAYER_COVERAGE_KES is NOT computed here on purpose: it needs to be
    # computed against the FINAL, post-fraud-injection claim amount and
    # needs provider-level + condition data that aren't merged in until
    # build_master_dataset(), so it's deferred to that stage.
    df["BASE_ENCOUNTER_COST_KES"] = df["ENCOUNTERCLASS"].apply(get_sha_cost)
    df["TOTAL_CLAIM_COST_KES"] = df["BASE_ENCOUNTER_COST_KES"].apply(
        lambda x: round(x * random.uniform(1.0, 1.3), 2)
    )

    df.to_csv(f"{OUTPUT_DIR}/encounters_kenya.csv", index=False)
    print(f"Done. {len(df)} encounters saved.")
    return df

# ============================================================
# KENYANISE CONDITIONS
# ============================================================

def kenyanise_conditions():
    print("Kenyanising conditions...")
    df = pd.read_csv(f"{INPUT_DIR}/conditions.csv")

    # Get unique patient IDs
    patient_ids = df["PATIENT"].unique()
    encounter_ids = df["ENCOUNTER"].unique()

    # Precompute each patient's first encounter ONCE (O(n) single pass)
    # instead of re-filtering the full (multi-million-row) dataframe inside
    # the loop below for every patient (that was O(n * num_patients), which
    # made this step effectively never finish on the full dataset). This is
    # purely a performance fix - it preserves the exact same loop order and
    # random-number call sequence, so the resulting conditions are
    # distributed identically to before.
    first_encounter_by_patient = df.groupby("PATIENT")["ENCOUNTER"].first().to_dict()

    # Add Kenyan conditions to a subset of patients
    new_rows = []
    for patient_id in patient_ids:
        # Each patient gets 1 to 3 Kenyan conditions
        num_conditions = random.randint(1, 3)
        conditions = random.sample(KENYAN_CONDITIONS, num_conditions)

        # Get an encounter for this patient
        encounter_id = first_encounter_by_patient.get(patient_id)
        if encounter_id is None:
            continue

        for condition in conditions:
            start_year = random.randint(2015, 2024)
            start_month = random.randint(1, 12)
            start_day = random.randint(1, 28)
            start_date = f"{start_year}-{start_month:02d}-{start_day:02d}"

            new_rows.append({
                "START": start_date,
                "STOP": "",
                "PATIENT": patient_id,
                "ENCOUNTER": encounter_id,
                "SYSTEM": "SNOMED-CT",
                "CODE": condition["CODE"],
                "DESCRIPTION": condition["DESCRIPTION"]
            })

    kenya_conditions = pd.DataFrame(new_rows)
    df_combined = pd.concat([df, kenya_conditions], ignore_index=True)

    df_combined.to_csv(f"{OUTPUT_DIR}/conditions_kenya.csv", index=False)
    print(f"Done. {len(df_combined)} conditions saved including {len(kenya_conditions)} Kenyan conditions.")
    return df_combined

# ============================================================
# KENYANISE PROVIDERS
# ============================================================

def kenyanise_providers():
    print("Kenyanising providers...")
    df = pd.read_csv(f"{INPUT_DIR}/organizations.csv")

    for idx, row in df.iterrows():
        county = random.choice(KENYAN_COUNTIES)
        provider_level = random.choice(PROVIDER_LEVELS)
        df.at[idx, "PROVIDER_LEVEL"] = provider_level
        df.at[idx, "PROVIDER_COUNTY"] = county
        df.at[idx, "NAME"] = provider_level + " - " + get_kenyan_town(county)
        df.at[idx, "STATE"] = "Kenya"
        df.at[idx, "CITY"] = get_kenyan_town(county)
        df.at[idx, "LAT"] = round(random.uniform(-4.67, 4.62), 6)
        df.at[idx, "LON"] = round(random.uniform(33.91, 41.90), 6)

    df.to_csv(f"{OUTPUT_DIR}/providers_kenya.csv", index=False)
    print(f"Done. {len(df)} providers saved.")
    return df

# ============================================================
# FRAUD INJECTION
# ============================================================

def inject_fraud(encounters_df):
    print("Injecting fraud patterns...")
    df = encounters_df.copy()

    # Add fraud label columns
    df["IS_FRAUD"] = 0
    df["FRAUD_TYPE"] = "none"

    total_records = len(df)

    # FRAUD TYPE 1 - UPCODING (5% of records)
    upcoding_rate = 0.05
    upcoding_count = int(total_records * upcoding_rate)
    upcoding_indices = random.sample(list(df.index), upcoding_count)

    for idx in upcoding_indices:
        original_cost = df.at[idx, "TOTAL_CLAIM_COST_KES"]
        # Inflate the cost by 40% to 150%
        inflation_factor = random.uniform(1.4, 2.5)
        df.at[idx, "TOTAL_CLAIM_COST_KES"] = round(original_cost * inflation_factor, 2)
        df.at[idx, "IS_FRAUD"] = 1
        df.at[idx, "FRAUD_TYPE"] = "upcoding"

    # FRAUD TYPE 2 - PHANTOM BILLING (3% of records)
    phantom_rate = 0.03
    phantom_count = int(total_records * phantom_rate)

    # Get remaining non-fraud indices
    non_fraud_indices = list(df[df["IS_FRAUD"] == 0].index)
    phantom_indices = random.sample(non_fraud_indices, phantom_count)

    for idx in phantom_indices:
        # Phantom billing — create a fake high-cost encounter
        df.at[idx, "TOTAL_CLAIM_COST_KES"] = round(random.uniform(20000, 80000), 2)
        df.at[idx, "BASE_ENCOUNTER_COST_KES"] = round(random.uniform(15000, 60000), 2)
        df.at[idx, "IS_FRAUD"] = 1
        df.at[idx, "FRAUD_TYPE"] = "phantom_billing"

    fraud_count = len(df[df["IS_FRAUD"] == 1])
    print(f"Done. {fraud_count} fraudulent records injected out of {total_records} total.")
    print(f"  Upcoding: {upcoding_count} records")
    print(f"  Phantom billing: {phantom_count} records")
    print(f"  Fraud rate: {fraud_count/total_records*100:.1f}%")

    df.to_csv(f"{OUTPUT_DIR}/encounters_kenya_with_fraud.csv", index=False)
    return df

# ============================================================
# BUILD MASTER CLAIMS DATASET
# ============================================================

def build_master_dataset(patients_df, encounters_fraud_df, conditions_df, providers_df):
    print("Building master claims dataset...")

    # Merge encounters with patients
    master = encounters_fraud_df.merge(
        patients_df[["Id", "BIRTHDATE", "GENDER", "CITY", "COUNTY",
                      "STATE", "FIRST", "LAST", "RACE", "INCOME"]],
        left_on="PATIENT",
        right_on="Id",
        how="left"
    )

    # Merge in facility level + county via ORGANIZATION_ID, so the coverage
    # model below can apply the real level-specific SHIF inpatient bed-day
    # rate and the private/public copay distinction, and so a geographic
    # mismatch (patient county vs. facility county) can be flagged later.
    provider_level_lookup = providers_df[["Id", "PROVIDER_LEVEL", "PROVIDER_COUNTY"]].rename(
        columns={"Id": "ORG_LOOKUP_ID"}
    )
    master = master.merge(
        provider_level_lookup,
        left_on="ORGANIZATION",
        right_on="ORG_LOOKUP_ID",
        how="left"
    )
    master["PROVIDER_LEVEL"] = master["PROVIDER_LEVEL"].fillna(
        "Level 4 - Sub-District Hospital"
    )
    master["PROVIDER_COUNTY"] = master["PROVIDER_COUNTY"].fillna("Nairobi")
    master = master.drop(columns=["ORG_LOOKUP_ID"])

    # Calculate age at time of encounter. Vectorized (same result as the
    # row-wise .apply() this replaced, verified on identical inputs) -
    # the row-wise version took far too long on the full multi-million-row
    # dataset to run within a single command's time budget.
    master["ENCOUNTER_DATE"] = pd.to_datetime(master["START"]).dt.date
    master["BIRTHDATE"] = pd.to_datetime(master["BIRTHDATE"]).dt.date
    _age_diff_days = (
        pd.to_datetime(master["ENCOUNTER_DATE"]) - pd.to_datetime(master["BIRTHDATE"])
    ).dt.days
    master["AGE_AT_ENCOUNTER"] = (_age_diff_days // 365).where(master["BIRTHDATE"].notna())

    # Add primary condition for each encounter
    condition_map = conditions_df.groupby("ENCOUNTER")["DESCRIPTION"].first()
    master["PRIMARY_CONDITION"] = master["Id_x"].map(condition_map)

    # ------------------------------------------------------------
    # Clinical-consistency fix + deliberate fraud signature.
    # kenyanise_conditions() assigns each patient's conditions purely at
    # random, with no gender awareness, so a male patient can end up with
    # a maternity condition (Pregnant/Pre-eclampsia) as PRIMARY_CONDITION -
    # that's not fraud, it's a data-generation bug, so fix it for every
    # LEGITIMATE row by reassigning to a random non-maternity condition.
    # Then, separately, deliberately RE-INTRODUCE exactly this kind of
    # gender/condition mismatch into a slice of phantom_billing claims for
    # male patients - a fabricated encounter that got a templated maternity
    # diagnosis slapped onto it is a realistic careless-fraud pattern (this
    # is the "clinical inconsistencies" category real fraud-AI products like
    # Smart Detect AI advertise catching). This is what gives the new
    # CLINICAL_INCONSISTENCY feature (computed later, in fix_dataset.py)
    # actual predictive signal instead of just reflecting random noise.
    # ------------------------------------------------------------
    non_maternity_conditions = [
        c["DESCRIPTION"] for c in KENYAN_CONDITIONS
        if c["DESCRIPTION"] not in MATERNITY_CONDITIONS
    ]
    bad_mask = (master["GENDER"] == "M") & master["PRIMARY_CONDITION"].isin(MATERNITY_CONDITIONS)
    if bad_mask.any():
        master.loc[bad_mask, "PRIMARY_CONDITION"] = np.random.choice(
            non_maternity_conditions, size=int(bad_mask.sum())
        )

    phantom_male_mask = (master["FRAUD_TYPE"] == "phantom_billing") & (master["GENDER"] == "M")
    inject_mask = phantom_male_mask & (np.random.random(len(master)) < 0.3)
    if inject_mask.any():
        master.loc[inject_mask, "PRIMARY_CONDITION"] = np.random.choice(
            sorted(MATERNITY_CONDITIONS), size=int(inject_mask.sum())
        )

    # ------------------------------------------------------------
    # Private facilities bill above tariff more often than public/
    # faith-based ones (which bill SHA directly at the gazetted rate).
    # Apply this markup AFTER fraud injection so it stacks on top of
    # whatever the claim amount already is (including fraud inflation),
    # which is what actually happens with a real over-billing facility.
    # ------------------------------------------------------------
    is_private = master["PROVIDER_LEVEL"].isin(PRIVATE_PROVIDER_LEVELS)
    private_markup = np.where(
        is_private,
        np.random.uniform(1.0, 1.4, size=len(master)),
        1.0,
    )
    master["TOTAL_CLAIM_COST_KES"] = (
        master["TOTAL_CLAIM_COST_KES"] * private_markup
    ).round(2)

    # Length of stay in days, used only for the inpatient bed-day tariff.
    # Computed directly from the raw Synthea START/STOP timestamps here
    # (LENGTH_OF_STAY_HOURS isn't computed until the later fix_dataset.py
    # stage) and clipped to a minimum of 1 bed-day.
    start_dt = pd.to_datetime(master["START"])
    stop_dt = pd.to_datetime(master["STOP"])
    los_days = (stop_dt - start_dt).dt.total_seconds() / 86400
    master["_LOS_DAYS"] = los_days.fillna(1).clip(lower=1)

    # ------------------------------------------------------------
    # PAYER_COVERAGE_KES - real SHA/SHIF tariff-cap model, computed
    # against the FINAL (post-fraud, post-private-markup) claim amount.
    # This fixes the earlier bug where coverage was computed before
    # fraud injection and so never reflected the actual billed amount.
    # Rule precedence: maternity > inpatient > PHF (wellness/ambulatory/
    # virtual) > emergency (ECCIF) > outpatient/urgentcare (condition-
    # specific SHIF tariff) > fallback (unmatched encounter types).
    # ------------------------------------------------------------
    etype = master["ENCOUNTERCLASS"].astype(str).str.lower()
    condition = master["PRIMARY_CONDITION"]
    claim = master["TOTAL_CLAIM_COST_KES"]
    level = master["PROVIDER_LEVEL"]
    n = len(master)

    coverage = pd.Series(np.nan, index=master.index, dtype="float64")

    # Maternity (Pregnant / Pre-eclampsia primary condition)
    maternity_mask = condition.isin(MATERNITY_CONDITIONS)
    maternity_tariff = np.where(
        condition == "Pre-eclampsia (disorder)",
        MATERNITY_TARIFF_KES["c_section"],
        MATERNITY_TARIFF_KES["normal_delivery"],
    )
    coverage = coverage.where(~maternity_mask, np.minimum(claim, maternity_tariff))

    # Inpatient bed-day tariff by facility level (non-maternity)
    inpatient_mask = (etype == "inpatient") & coverage.isna()
    bed_rate = level.map(INPATIENT_BED_DAY_KES).fillna(INPATIENT_BED_DAY_DEFAULT_KES)
    inpatient_tariff = bed_rate * master["_LOS_DAYS"]
    coverage = coverage.where(~inpatient_mask, np.minimum(claim, inpatient_tariff))

    # PHF - wellness/ambulatory/virtual covered near-fully
    phf_mask = etype.isin(["wellness", "ambulatory", "virtual"]) & coverage.isna()
    phf_fraction = np.random.uniform(*PHF_COVERAGE_FRACTION_RANGE, size=n)
    coverage = coverage.where(~phf_mask, claim * phf_fraction)

    # Emergency - ECCIF covers generously regardless of condition
    emergency_mask = (etype == "emergency") & coverage.isna()
    coverage = coverage.where(~emergency_mask, claim * EMERGENCY_COVERAGE_FRACTION)

    # Outpatient / urgentcare - condition-specific SHIF tariff, else general
    outpatient_mask = etype.isin(["outpatient", "urgentcare"]) & coverage.isna()
    condition_tariff = condition.map(CONDITION_OUTPATIENT_TARIFF_KES).fillna(
        GENERAL_OUTPATIENT_TARIFF_KES
    )
    coverage = coverage.where(~outpatient_mask, np.minimum(claim, condition_tariff))

    # Fallback for anything still unmatched
    fallback_mask = coverage.isna()
    fallback_fraction = np.random.uniform(*FALLBACK_COVERAGE_FRACTION_RANGE, size=n)
    coverage = coverage.where(~fallback_mask, claim * fallback_fraction)

    master["PAYER_COVERAGE_KES"] = coverage.round(2)
    master["PATIENT_COPAY_KES"] = (claim - master["PAYER_COVERAGE_KES"]).clip(lower=0).round(2)
    master = master.drop(columns=["_LOS_DAYS"])

    # Select and rename final columns
    final_columns = {
        "Id_x": "ENCOUNTER_ID",
        "PATIENT": "PATIENT_ID",
        "PROVIDER": "PROVIDER_ID",
        "ORGANIZATION": "ORGANIZATION_ID",
        "PROVIDER_LEVEL": "PROVIDER_LEVEL",
        "PROVIDER_COUNTY": "PROVIDER_COUNTY",
        "ENCOUNTERCLASS": "ENCOUNTER_TYPE",
        "START": "SERVICE_DATE",
        "STOP": "DISCHARGE_DATE",
        "BASE_ENCOUNTER_COST_KES": "BASE_COST_KES",
        "TOTAL_CLAIM_COST_KES": "CLAIM_AMOUNT_KES",
        "PAYER_COVERAGE_KES": "PAYER_COVERAGE_KES",
        "PATIENT_COPAY_KES": "PATIENT_COPAY_KES",
        "CODE": "PROCEDURE_CODE",
        "DESCRIPTION": "PROCEDURE_DESCRIPTION",
        "REASONCODE": "DIAGNOSIS_CODE",
        "REASONDESCRIPTION": "DIAGNOSIS_DESCRIPTION",
        "PRIMARY_CONDITION": "PRIMARY_CONDITION",
        "FIRST": "PATIENT_FIRST_NAME",
        "LAST": "PATIENT_LAST_NAME",
        "GENDER": "GENDER",
        "AGE_AT_ENCOUNTER": "AGE",
        "CITY": "PATIENT_CITY",
        "COUNTY": "PATIENT_COUNTY",
        "INCOME": "PATIENT_INCOME",
        "IS_FRAUD": "IS_FRAUD",
        "FRAUD_TYPE": "FRAUD_TYPE"
    }

    master = master.rename(columns=final_columns)
    final_cols = [col for col in final_columns.values() if col in master.columns]
    master = master[final_cols]

    master.to_csv(f"{OUTPUT_DIR}/master_claims_kenya.csv", index=False)
    print(f"Done. Master dataset has {len(master)} records.")
    print(f"Fraud breakdown:")
    print(master["FRAUD_TYPE"].value_counts())
    return master

# ============================================================
# RUN THE FULL PIPELINE
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("KENYA SYNTHEA PIPELINE STARTING")
    print("=" * 50)

    patients = kenyanise_patients()
    encounters = kenyanise_encounters()
    conditions = kenyanise_conditions()
    providers = kenyanise_providers()
    encounters_fraud = inject_fraud(encounters)
    master = build_master_dataset(patients, encounters_fraud, conditions, providers)

    print("=" * 50)
    print("PIPELINE COMPLETE")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("Files created:")
    print("  - patients_kenya.csv")
    print("  - encounters_kenya.csv")
    print("  - conditions_kenya.csv")
    print("  - providers_kenya.csv")
    print("  - encounters_kenya_with_fraud.csv")
    print("  - master_claims_kenya.csv")
    print("=" * 50)