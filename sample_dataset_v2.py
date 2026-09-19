import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================
TOTAL_SIZE = 500_000
TARGET_FRAUD_RATE = 0.13   # was accidentally ~54.8% before (kept ALL fraud rows
                           # while shrinking the legitimate pool ~13x, which
                           # concentrated fraud far above its true ~8% base rate)
RANDOM_STATE = 42

print("Loading full dataset...")
df = pd.read_csv(
    "output/kenyanised/master_claims_kenya_with_collusion.csv"
)
print(f"Full dataset: {df.shape}")

# Separate fraud and legitimate
fraud = df[df["IS_FRAUD"] == 1]
legitimate = df[df["IS_FRAUD"] == 0]

print(f"Fraud records: {len(fraud)}")
print(f"Legitimate records: {len(legitimate)}")
print(f"Original fraud rate: {len(fraud) / len(df) * 100:.2f}%")

# ------------------------------------------------------------
# Work out how many fraud / legitimate rows we need for the
# TARGET_FRAUD_RATE, instead of blindly keeping every fraud row.
# ------------------------------------------------------------
target_fraud_n = round(TOTAL_SIZE * TARGET_FRAUD_RATE)
target_legit_n = TOTAL_SIZE - target_fraud_n

if target_fraud_n > len(fraud):
    raise ValueError(
        f"Only {len(fraud)} fraud records available, but "
        f"{target_fraud_n} are needed for a {TARGET_FRAUD_RATE*100:.0f}% "
        f"fraud rate at TOTAL_SIZE={TOTAL_SIZE}. Lower TOTAL_SIZE or "
        f"TARGET_FRAUD_RATE."
    )
if target_legit_n > len(legitimate):
    raise ValueError(
        f"Only {len(legitimate)} legitimate records available, but "
        f"{target_legit_n} are needed. Lower TOTAL_SIZE."
    )

print(
    f"\nTarget: {TOTAL_SIZE} total rows @ {TARGET_FRAUD_RATE*100:.0f}% fraud "
    f"-> {target_fraud_n} fraud rows, {target_legit_n} legitimate rows"
)

# Stratified sample of fraud rows, preserving the relative proportion of
# each FRAUD_TYPE (e.g. upcoding vs phantom_billing) seen in the full data.
fraud_frac = target_fraud_n / len(fraud)
fraud_sample = (
    fraud.groupby("FRAUD_TYPE", group_keys=False)[fraud.columns]
    .apply(lambda g: g.sample(frac=fraud_frac, random_state=RANDOM_STATE))
)
# groupby+frac sampling can land a few rows off the exact target due to
# per-group rounding; correct that by trimming or topping up randomly.
if len(fraud_sample) > target_fraud_n:
    fraud_sample = fraud_sample.sample(n=target_fraud_n, random_state=RANDOM_STATE)
elif len(fraud_sample) < target_fraud_n:
    remaining = fraud.drop(fraud_sample.index)
    top_up = remaining.sample(
        n=target_fraud_n - len(fraud_sample), random_state=RANDOM_STATE
    )
    fraud_sample = pd.concat([fraud_sample, top_up])

legitimate_sample = legitimate.sample(
    n=target_legit_n,
    random_state=RANDOM_STATE
)

# Combine
df_sample = pd.concat(
    [fraud_sample, legitimate_sample],
    ignore_index=True
)

# Shuffle
df_sample = df_sample.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print(f"\nSampled dataset: {df_sample.shape}")
print(f"Fraud rate: {df_sample['IS_FRAUD'].mean()*100:.2f}%")
print(f"Fraud breakdown:")
print(df_sample["FRAUD_TYPE"].value_counts())

OUT_PATH = "output/kenyanised/master_claims_sample_500k_13pct_v2.csv"
df_sample.to_csv(OUT_PATH, index=False)
print(f"\nSample dataset saved to {OUT_PATH}")
