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

# SHA tariff rates in KES by encounter type
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

    # Replace costs with SHA tariff amounts in KES
    df["BASE_ENCOUNTER_COST_KES"] = df["ENCOUNTERCLASS"].apply(get_sha_cost)
    df["TOTAL_CLAIM_COST_KES"] = df["BASE_ENCOUNTER_COST_KES"].apply(
        lambda x: round(x * random.uniform(1.0, 2.5), 2)
    )
    df["PAYER_COVERAGE_KES"] = df["TOTAL_CLAIM_COST_KES"].apply(
        lambda x: round(x * random.uniform(0.5, 0.9), 2)
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

    # Add Kenyan conditions to a subset of patients
    new_rows = []
    for patient_id in patient_ids:
        # Each patient gets 1 to 3 Kenyan conditions
        num_conditions = random.randint(1, 3)
        conditions = random.sample(KENYAN_CONDITIONS, num_conditions)

        # Get an encounter for this patient
        patient_encounters = df[df["PATIENT"] == patient_id]["ENCOUNTER"].values
        if len(patient_encounters) == 0:
            continue
        encounter_id = patient_encounters[0]

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

def build_master_dataset(patients_df, encounters_fraud_df, conditions_df):
    print("Building master claims dataset...")

    # Merge encounters with patients
    master = encounters_fraud_df.merge(
        patients_df[["Id", "BIRTHDATE", "GENDER", "CITY", "COUNTY",
                      "STATE", "FIRST", "LAST", "RACE", "INCOME"]],
        left_on="PATIENT",
        right_on="Id",
        how="left"
    )

    # Calculate age at time of encounter
    master["ENCOUNTER_DATE"] = pd.to_datetime(master["START"]).dt.date
    master["BIRTHDATE"] = pd.to_datetime(master["BIRTHDATE"]).dt.date
    master["AGE_AT_ENCOUNTER"] = master.apply(
        lambda row: (row["ENCOUNTER_DATE"] - row["BIRTHDATE"]).days // 365
        if pd.notna(row["BIRTHDATE"]) else None, axis=1
    )

    # Add primary condition for each encounter
    condition_map = conditions_df.groupby("ENCOUNTER")["DESCRIPTION"].first()
    master["PRIMARY_CONDITION"] = master["Id_x"].map(condition_map)

    # Select and rename final columns
    final_columns = {
        "Id_x": "ENCOUNTER_ID",
        "PATIENT": "PATIENT_ID",
        "PROVIDER": "PROVIDER_ID",
        "ORGANIZATION": "ORGANIZATION_ID",
        "ENCOUNTERCLASS": "ENCOUNTER_TYPE",
        "START": "SERVICE_DATE",
        "STOP": "DISCHARGE_DATE",
        "BASE_ENCOUNTER_COST_KES": "BASE_COST_KES",
        "TOTAL_CLAIM_COST_KES": "CLAIM_AMOUNT_KES",
        "PAYER_COVERAGE_KES": "PAYER_COVERAGE_KES",
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
    master = build_master_dataset(patients, encounters_fraud, conditions)

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