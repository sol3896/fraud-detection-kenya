import pandas as pd
import numpy as np

print("Loading master dataset...")
df = pd.read_csv(
    "output/kenyanised/master_claims_kenya.csv"
)

print(f"Original shape: {df.shape}")

# Fix 1 - Clip negative ages to 0
df["AGE"] = df["AGE"].clip(lower=0)
print(f"Negative ages fixed. Min age now: {df['AGE'].min()}")

# Fix 2 - Fill missing diagnosis codes
df["DIAGNOSIS_CODE"] = df["DIAGNOSIS_CODE"].fillna("Z00.00")
df["DIAGNOSIS_DESCRIPTION"] = df["DIAGNOSIS_DESCRIPTION"].fillna(
    "General examination without complaint"
)
print(f"Missing diagnosis codes filled.")

# Fix 3 - Fill missing primary condition
df["PRIMARY_CONDITION"] = df["PRIMARY_CONDITION"].fillna(
    "No condition recorded"
)
print(f"Missing primary conditions filled.")

# Fix 4 - Convert AGE to integer
df["AGE"] = df["AGE"].astype(int)

# Fix 5 - Clean SERVICE_DATE to date only
df["SERVICE_DATE"] = pd.to_datetime(
    df["SERVICE_DATE"],
    utc=True
).dt.tz_localize(None).dt.date

# Fix 6 - Add LENGTH_OF_STAY column
df["DISCHARGE_DATE"] = pd.to_datetime(
    df["DISCHARGE_DATE"],
    errors="coerce",
    utc=True
).dt.tz_localize(None)

service = pd.to_datetime(df["SERVICE_DATE"])
df["LENGTH_OF_STAY_HOURS"] = (
    df["DISCHARGE_DATE"] - service
).dt.total_seconds() / 3600

# Fix 7 - Add CLAIM_TO_BASE_RATIO feature
# This will help XGBoost detect inflated claims
df["CLAIM_TO_BASE_RATIO"] = (
    df["CLAIM_AMOUNT_KES"] / df["BASE_COST_KES"]
).round(3)

# Verify final state
print(f"\nFinal shape: {df.shape}")
print(f"\nMissing values after fixes:")
print(df.isnull().sum())
print(f"\nAge distribution after fix:")
print(df["AGE"].describe())
print(f"\nFraud breakdown:")
print(df["FRAUD_TYPE"].value_counts())
print(f"\nClaim to base ratio by fraud type:")
print(df.groupby("FRAUD_TYPE")["CLAIM_TO_BASE_RATIO"].describe())

# Save cleaned dataset
df.to_csv(
    "output/kenyanised/master_claims_kenya_clean.csv",
    index=False
)
print(f"\nClean dataset saved.")
print("Columns in final dataset:")
for col in df.columns:
    print(f"  {col}")
