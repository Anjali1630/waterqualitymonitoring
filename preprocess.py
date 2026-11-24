import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

INPUT_FILE = "water_quality_dataset.csv"
OUTPUT_FILE = "water_quality_preprocessed.csv"

FEATURES = [
    "pH",
    "turbidity",
    "tds",
    "temperature"
]

print("=" * 65)
print("WATER QUALITY PREPROCESSING")
print("KNN IMPUTATION + KALMAN FILTERING")
print("=" * 65)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset shape:", df.shape)

for column in FEATURES:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print("\nMissing values before KNN:")
print(df[FEATURES].isnull().sum())

# ============================================================
# SORT BY TIMESTAMP
# Kalman filtering needs an ordered sequence
# ============================================================

if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

# ============================================================
# KNN IMPUTATION
# ============================================================

print("\nApplying KNN imputation...")

imputer = KNNImputer(
    n_neighbors=5,
    weights="distance"
)

df[FEATURES] = imputer.fit_transform(
    df[FEATURES]
)

print("KNN imputation completed.")

print("\nMissing values after KNN:")
print(df[FEATURES].isnull().sum())

# ============================================================
# KALMAN FILTER
# ============================================================

def kalman_filter(
    values,
    process_variance=0.01,
    measurement_variance=0.10
):

    values = np.asarray(
        values,
        dtype=float
    )

    filtered = np.zeros(
        len(values)
    )

    estimate = values[0]

    error_estimate = 1.0

    for i in range(
        len(values)
    ):

        predicted_estimate = estimate

        predicted_error = (
            error_estimate +
            process_variance
        )

        kalman_gain = (
            predicted_error /
            (
                predicted_error +
                measurement_variance
            )
        )

        estimate = (
            predicted_estimate +
            kalman_gain *
            (
                values[i] -
                predicted_estimate
            )
        )

        error_estimate = (
            1 -
            kalman_gain
        ) * predicted_error

        filtered[i] = estimate

    return filtered


# ============================================================
# APPLY KALMAN
# ============================================================

print("\nApplying Kalman filtering...")

for column in FEATURES:

    df[column] = kalman_filter(
        df[column].values
    )

print("Kalman filtering completed.")

# ============================================================
# CHECK DATA
# ============================================================

print("\nMissing values after preprocessing:")
print(df[FEATURES].isnull().sum())

print("\nPreprocessed sensor statistics:")
print(
    df[FEATURES].describe()
)

# ============================================================
# RESTORE SHUFFLE
# ============================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)

# ============================================================
# ROUND VALUES
# ============================================================

df["pH"] = df["pH"].round(4)

df["turbidity"] = df[
    "turbidity"
].round(4)

df["tds"] = df[
    "tds"
].round(4)

df["temperature"] = df[
    "temperature"
].round(4)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nFinal dataset shape:",
    df.shape
)

print(
    "\nFirst 10 records:"
)

print(
    df.head(10).to_string(
        index=False
    )
)

print(
    "\nPreprocessed dataset saved:",
    OUTPUT_FILE
)

print("\n" + "=" * 65)
print("PREPROCESSING COMPLETED")
print("=" * 65)