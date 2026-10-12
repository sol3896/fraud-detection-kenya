"""
Exports a stratified sample of the claims dataset as a WEKA-ready .arff
file, with the SAME feature set your v2-robust XGBoost model actually
uses (raw cost columns + the engineered COST_RATIO / PROVIDER_Z_SCORE /
PROVIDER_DAILY_VOLUME features, fit on a train-only split exactly like
xgboost_model_v2_robust.py does - no leakage).

Why a sample, not the full 500k rows: WEKA's Explorer GUI (and especially
wrapper-based attribute selection methods) loads everything into memory and
can get slow or unresponsive on very large files. 30,000 stratified rows is
plenty for CfsSubsetEval / InfoGainAttributeEval / GainRatioAttributeEval to
give stable, reportable results, while staying responsive in the GUI.

Left as raw nominal text (not pre-encoded) for ENCOUNTER_TYPE, GENDER,
PATIENT_COUNTY, and IS_FRAUD as a nominal {0,1} class - WEKA's attribute
selection algorithms handle nominal attributes natively, which is the point
of doing this exercise in WEKA rather than re-deriving it in Python.

Output: ~/synthea_kenya/output/model/claims_for_weka.arff
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.expanduser("~/synthea_kenya")
DATA_PATH = f"{BASE_DIR}/output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
OUT_PATH = f"{BASE_DIR}/output/model/claims_for_weka.arff"
RANDOM_STATE = 42
SAMPLE_N = 30000

NEEDED_COLS = [
    "BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES", "CLAIM_TO_BASE_RATIO",
    "AGE", "LENGTH_OF_STAY_HOURS", "ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY",
    "IS_FRAUD", "PRIMARY_CONDITION", "PROVIDER_ID", "SERVICE_DATE", "ENCOUNTER_ID",
]


def arff_quote(val: str) -> str:
    """ARFF nominal values need single-quoting if they contain spaces or commas."""
    s = str(val)
    if any(ch in s for ch in [" ", ",", "'"]):
        return "'" + s.replace("'", "\\'") + "'"
    return s


def main():
    print("Loading full dataset...")
    df = pd.read_csv(DATA_PATH, usecols=NEEDED_COLS)

    print("Stratified sample for a train-only split (matches the real model's methodology)...")
    idx_train, idx_rest = train_test_split(
        df.index, test_size=0.5, random_state=RANDOM_STATE, stratify=df["IS_FRAUD"]
    )
    train_df = df.loc[idx_train]

    # --- Same 3 engineered features as xgboost_model_v2_robust.py, fit on train only ---
    group_cols = ["PRIMARY_CONDITION", "PATIENT_COUNTY"]
    group_median = train_df.groupby(group_cols)["CLAIM_AMOUNT_KES"].median()
    global_median = train_df["CLAIM_AMOUNT_KES"].median()
    median_lookup = df[group_cols].apply(tuple, axis=1).map(group_median).fillna(global_median)
    df["COST_RATIO"] = (df["CLAIM_AMOUNT_KES"] / median_lookup.replace(0, np.nan)).fillna(1.0)

    provider_mean = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].mean()
    provider_std = train_df.groupby("PROVIDER_ID")["CLAIM_AMOUNT_KES"].std()
    global_mean, global_std = train_df["CLAIM_AMOUNT_KES"].mean(), train_df["CLAIM_AMOUNT_KES"].std()
    mean_lookup = df["PROVIDER_ID"].map(provider_mean).fillna(global_mean)
    std_lookup = df["PROVIDER_ID"].map(provider_std).fillna(global_std).replace(0, global_std)
    df["PROVIDER_Z_SCORE"] = ((df["CLAIM_AMOUNT_KES"] - mean_lookup) / std_lookup).fillna(0.0)

    daily_counts = df.groupby(["PROVIDER_ID", "SERVICE_DATE"])["ENCOUNTER_ID"].transform("count")
    df["PROVIDER_DAILY_VOLUME"] = daily_counts

    # --- Now take the actual stratified export sample ---
    sample_df, _ = train_test_split(
        df, train_size=SAMPLE_N, random_state=RANDOM_STATE, stratify=df["IS_FRAUD"]
    )
    print(f"Exporting {len(sample_df):,} rows, fraud rate {sample_df.IS_FRAUD.mean()*100:.2f}%")

    numeric_attrs = [
        "BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES", "CLAIM_TO_BASE_RATIO",
        "COST_RATIO", "PROVIDER_Z_SCORE", "PROVIDER_DAILY_VOLUME",
        "AGE", "LENGTH_OF_STAY_HOURS",
    ]
    nominal_attrs = ["ENCOUNTER_TYPE", "GENDER", "PATIENT_COUNTY"]
    class_attr = "IS_FRAUD"

    export_cols = numeric_attrs + nominal_attrs + [class_attr]
    out_df = sample_df[export_cols].copy()
    out_df[class_attr] = out_df[class_attr].astype(int).astype(str)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("@RELATION kenya_health_claims_fraud\n\n")
        for col in numeric_attrs:
            f.write(f"@ATTRIBUTE {col} NUMERIC\n")
        for col in nominal_attrs:
            uniques = sorted(out_df[col].dropna().astype(str).unique())
            values = ",".join(arff_quote(v) for v in uniques)
            f.write(f"@ATTRIBUTE {col} {{{values}}}\n")
        f.write(f"@ATTRIBUTE {class_attr} {{0,1}}\n")
        f.write("\n@DATA\n")
        for _, row in out_df.iterrows():
            fields = []
            for col in numeric_attrs:
                v = row[col]
                fields.append("?" if pd.isna(v) else f"{v}")
            for col in nominal_attrs:
                v = row[col]
                fields.append("?" if pd.isna(v) else arff_quote(v))
            fields.append(row[class_attr])
            f.write(",".join(fields) + "\n")

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\nWrote {OUT_PATH} ({size_kb:.0f} KB)")
    print(f"Attributes: {len(numeric_attrs)} numeric + {len(nominal_attrs)} nominal + 1 class label")
    print("Ready to open directly in WEKA Explorer.")


if __name__ == "__main__":
    main()
