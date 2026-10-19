"""
Memory-bounded driver for kenyanise.py's build_master_dataset() logic.

The bridged sandbox VM this runs in only has ~3.8GB RAM (confirmed via
`free -h`), and merging the full ~3.4M-row encounters_fraud dataframe with
patients/providers in one shot gets OOM-killed there. This script performs
the exact same merge/age/coverage/copay logic (reusing kenyanise.py's
constants directly, so the tariff numbers can't drift out of sync) but
streams encounters_kenya_with_fraud.csv in chunks and appends each
processed chunk to master_claims_kenya.csv, bounding peak memory to one
chunk's worth of rows instead of the whole dataset.

Note: chunking changes how many values are drawn from np.random.uniform()
per call (one call per chunk instead of one call for the whole dataset),
so the exact random draws for PHF/fallback coverage differ run-to-run
in a different *pattern* than the non-chunked function would produce -
but the statistical distribution and every tariff-cap rule are identical,
which is all that matters for synthetic data generation.
"""
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
import kenyanise as k

CHUNK_SIZE = 400_000


def process_chunk(chunk, patients_df, provider_level_lookup, condition_map):
    master = chunk.merge(
        patients_df[["Id", "BIRTHDATE", "GENDER", "CITY", "COUNTY",
                      "STATE", "FIRST", "LAST", "RACE", "INCOME"]],
        left_on="PATIENT", right_on="Id", how="left"
    )
    master = master.merge(provider_level_lookup, left_on="ORGANIZATION",
                           right_on="ORG_LOOKUP_ID", how="left")
    master["PROVIDER_LEVEL"] = master["PROVIDER_LEVEL"].fillna(
        "Level 4 - Sub-District Hospital"
    )
    master["PROVIDER_COUNTY"] = master["PROVIDER_COUNTY"].fillna("Nairobi")
    master = master.drop(columns=["ORG_LOOKUP_ID"])

    master["ENCOUNTER_DATE"] = pd.to_datetime(master["START"]).dt.date
    master["BIRTHDATE"] = pd.to_datetime(master["BIRTHDATE"]).dt.date
    diff_days = (
        pd.to_datetime(master["ENCOUNTER_DATE"]) - pd.to_datetime(master["BIRTHDATE"])
    ).dt.days
    master["AGE_AT_ENCOUNTER"] = (diff_days // 365).where(master["BIRTHDATE"].notna())

    master["PRIMARY_CONDITION"] = master["Id_x"].map(condition_map)

    # Clinical-consistency fix + deliberate fraud signature (mirrors
    # kenyanise.py's build_master_dataset() exactly - see its comments).
    non_maternity_conditions = [
        c["DESCRIPTION"] for c in k.KENYAN_CONDITIONS
        if c["DESCRIPTION"] not in k.MATERNITY_CONDITIONS
    ]
    bad_mask = (master["GENDER"] == "M") & master["PRIMARY_CONDITION"].isin(k.MATERNITY_CONDITIONS)
    if bad_mask.any():
        master.loc[bad_mask, "PRIMARY_CONDITION"] = np.random.choice(
            non_maternity_conditions, size=int(bad_mask.sum())
        )

    phantom_male_mask = (master["FRAUD_TYPE"] == "phantom_billing") & (master["GENDER"] == "M")
    inject_mask = phantom_male_mask & (np.random.random(len(master)) < 0.3)
    if inject_mask.any():
        master.loc[inject_mask, "PRIMARY_CONDITION"] = np.random.choice(
            sorted(k.MATERNITY_CONDITIONS), size=int(inject_mask.sum())
        )

    is_private = master["PROVIDER_LEVEL"].isin(k.PRIVATE_PROVIDER_LEVELS)
    private_markup = np.where(
        is_private, np.random.uniform(1.0, 1.4, size=len(master)), 1.0
    )
    master["TOTAL_CLAIM_COST_KES"] = (
        master["TOTAL_CLAIM_COST_KES"] * private_markup
    ).round(2)

    start_dt = pd.to_datetime(master["START"])
    stop_dt = pd.to_datetime(master["STOP"])
    los_days = (stop_dt - start_dt).dt.total_seconds() / 86400
    master["_LOS_DAYS"] = los_days.fillna(1).clip(lower=1)

    etype = master["ENCOUNTERCLASS"].astype(str).str.lower()
    condition = master["PRIMARY_CONDITION"]
    claim = master["TOTAL_CLAIM_COST_KES"]
    level = master["PROVIDER_LEVEL"]
    n = len(master)

    coverage = pd.Series(np.nan, index=master.index, dtype="float64")

    maternity_mask = condition.isin(k.MATERNITY_CONDITIONS)
    maternity_tariff = np.where(
        condition == "Pre-eclampsia (disorder)",
        k.MATERNITY_TARIFF_KES["c_section"],
        k.MATERNITY_TARIFF_KES["normal_delivery"],
    )
    coverage = coverage.where(~maternity_mask, np.minimum(claim, maternity_tariff))

    inpatient_mask = (etype == "inpatient") & coverage.isna()
    bed_rate = level.map(k.INPATIENT_BED_DAY_KES).fillna(k.INPATIENT_BED_DAY_DEFAULT_KES)
    inpatient_tariff = bed_rate * master["_LOS_DAYS"]
    coverage = coverage.where(~inpatient_mask, np.minimum(claim, inpatient_tariff))

    phf_mask = etype.isin(["wellness", "ambulatory", "virtual"]) & coverage.isna()
    phf_fraction = np.random.uniform(*k.PHF_COVERAGE_FRACTION_RANGE, size=n)
    coverage = coverage.where(~phf_mask, claim * phf_fraction)

    emergency_mask = (etype == "emergency") & coverage.isna()
    coverage = coverage.where(~emergency_mask, claim * k.EMERGENCY_COVERAGE_FRACTION)

    outpatient_mask = etype.isin(["outpatient", "urgentcare"]) & coverage.isna()
    condition_tariff = condition.map(k.CONDITION_OUTPATIENT_TARIFF_KES).fillna(
        k.GENERAL_OUTPATIENT_TARIFF_KES
    )
    coverage = coverage.where(~outpatient_mask, np.minimum(claim, condition_tariff))

    fallback_mask = coverage.isna()
    fallback_fraction = np.random.uniform(*k.FALLBACK_COVERAGE_FRACTION_RANGE, size=n)
    coverage = coverage.where(~fallback_mask, claim * fallback_fraction)

    master["PAYER_COVERAGE_KES"] = coverage.round(2)
    master["PATIENT_COPAY_KES"] = (claim - master["PAYER_COVERAGE_KES"]).clip(lower=0).round(2)
    master = master.drop(columns=["_LOS_DAYS"])

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
    final_cols = [c for c in final_columns.values() if c in master.columns]
    return master[final_cols]


def main():
    t0 = time.time()
    patients_df = pd.read_csv(f"{k.OUTPUT_DIR}/patients_kenya.csv")
    providers_df = pd.read_csv(f"{k.OUTPUT_DIR}/providers_kenya.csv")
    provider_level_lookup = providers_df[["Id", "PROVIDER_LEVEL", "PROVIDER_COUNTY"]].rename(
        columns={"Id": "ORG_LOOKUP_ID"}
    )
    conditions_df = pd.read_csv(f"{k.OUTPUT_DIR}/conditions_kenya.csv")
    condition_map = conditions_df.groupby("ENCOUNTER")["DESCRIPTION"].first()
    del conditions_df
    print(f"Loaded lookups in {time.time()-t0:.1f}s", flush=True)

    out_path = f"{k.OUTPUT_DIR}/master_claims_kenya.csv"
    fraud_counts = {}
    total_rows = 0
    first_chunk = True

    t1 = time.time()
    reader = pd.read_csv(
        f"{k.OUTPUT_DIR}/encounters_kenya_with_fraud.csv", chunksize=CHUNK_SIZE
    )
    for i, chunk in enumerate(reader):
        tc = time.time()
        processed = process_chunk(chunk, patients_df, provider_level_lookup, condition_map)
        processed.to_csv(out_path, mode="w" if first_chunk else "a",
                          header=first_chunk, index=False)
        first_chunk = False
        total_rows += len(processed)
        for ft, cnt in processed["FRAUD_TYPE"].value_counts().items():
            fraud_counts[ft] = fraud_counts.get(ft, 0) + int(cnt)
        print(f"  chunk {i}: {len(processed)} rows in {time.time()-tc:.1f}s "
              f"(total so far {total_rows})", flush=True)

    print(f"\nDone in {time.time()-t1:.1f}s. Master dataset has {total_rows} records.",
          flush=True)
    print("Fraud breakdown:", fraud_counts, flush=True)
    print(f"TOTAL: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
