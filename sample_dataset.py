import pandas as pd
import numpy as np

print("Loading full dataset...")
df = pd.read_csv(
    "output/kenyanised/master_claims_kenya_clean.csv"
)
print(f"Full dataset: {df.shape}")

# Separate fraud and legitimate
fraud = df[df["IS_FRAUD"] == 1]
legitimate = df[df["IS_FRAUD"] == 0]

print(f"Fraud records: {len(fraud)}")
print(f"Legitimate records: {len(legitimate)}")

# Keep all fraud records (273,027)
# Sample legitimate to get roughly 8% fraud rate in 500k total
# 273,027 fraud / 500,000 total = 8% fraud rate
# So legitimate = 500,000 - 273,027 = 226,973

target_legitimate = 500000 - len(fraud)

legitimate_sample = legitimate.sample(
    n=target_legitimate,
    random_state=42
)

# Combine
df_sample = pd.concat(
    [fraud, legitimate_sample],
    ignore_index=True
)

# Shuffle
df_sample = df_sample.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print(f"\nSampled dataset: {df_sample.shape}")
print(f"Fraud rate: {df_sample['IS_FRAUD'].mean()*100:.1f}%")
print(f"Fraud breakdown:")
print(df_sample["FRAUD_TYPE"].value_counts())

df_sample.to_csv(
    "output/kenyanised/master_claims_sample_500k.csv",
    index=False
)
print("\nSample dataset saved.")
