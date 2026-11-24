import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_SAMPLES = 9000
SAMPLES_PER_CLASS = 3000
NUMBER_OF_CLIENTS = 3

OUTPUT_FILE = "water_quality_dataset.csv"


# ============================================================
# EXACT SENSOR RANGES
# ============================================================

# FRESH
# Turbidity : 0 - 500
# TDS       : 0 - 100

FRESH_PH = (6.5, 7.5)
FRESH_TURBIDITY = (0, 500)
FRESH_TDS = (0, 100)
FRESH_TEMP = (25, 28)


# MUD
# Turbidity : 0 - 100
# TDS       : 400 - 700

MUD_PH = (5.5, 8.5)
MUD_TURBIDITY = (0, 100)
MUD_TDS = (400, 700)
MUD_TEMP = (25, 28)


# SALT
# Turbidity : 0 - 600
# TDS       : 1000 - 1800

SALT_PH = (6.5, 8.5)
SALT_TURBIDITY = (0, 600)
SALT_TDS = (1000, 1800)
SALT_TEMP = (25, 28)


# ============================================================
# HELPER
# ============================================================

def clipped_normal(
    mean,
    std,
    minimum,
    maximum,
    count
):

    values = np.random.normal(
        mean,
        std,
        count
    )

    return np.clip(
        values,
        minimum,
        maximum
    )


# ============================================================
# FRESH WATER
# ============================================================

def generate_fresh(count):

    ph = clipped_normal(
        7.0,
        0.20,
        6.5,
        7.5,
        count
    )

    turbidity = (
        np.random.beta(
            2,
            3,
            count
        ) * 500
    )

    tds = (
        np.random.beta(
            2,
            5,
            count
        ) * 100
    )

    temperature = clipped_normal(
        26.5,
        0.65,
        25,
        28,
        count
    )

    return (
        ph,
        turbidity,
        tds,
        temperature
    )


# ============================================================
# MUD WATER
# ============================================================

def generate_mud(count):

    ph = clipped_normal(
        6.6,
        0.55,
        5.5,
        8.5,
        count
    )

    turbidity = (
        np.random.beta(
            2.5,
            3,
            count
        ) * 100
    )

    tds = clipped_normal(
        560,
        75,
        400,
        700,
        count
    )

    temperature = clipped_normal(
        26.5,
        0.70,
        25,
        28,
        count
    )

    return (
        ph,
        turbidity,
        tds,
        temperature
    )


# ============================================================
# SALT WATER
# ============================================================

def generate_salt(count):

    ph = clipped_normal(
        7.2,
        0.45,
        6.5,
        8.5,
        count
    )

    turbidity = (
        np.random.beta(
            2,
            2.5,
            count
        ) * 600
    )

    tds = (
        np.random.beta(
            2.5,
            2,
            count
        ) * 800
        + 1000
    )

    temperature = clipped_normal(
        26.7,
        0.65,
        25,
        28,
        count
    )

    return (
        ph,
        turbidity,
        tds,
        temperature
    )


# ============================================================
# GENERATE DATA
# ============================================================

fresh_ph, fresh_turbidity, fresh_tds, fresh_temperature = \
    generate_fresh(SAMPLES_PER_CLASS)

mud_ph, mud_turbidity, mud_tds, mud_temperature = \
    generate_mud(SAMPLES_PER_CLASS)

salt_ph, salt_turbidity, salt_tds, salt_temperature = \
    generate_salt(SAMPLES_PER_CLASS)


# ============================================================
# COMBINE
# ============================================================

pH = np.concatenate([
    fresh_ph,
    mud_ph,
    salt_ph
])

turbidity = np.concatenate([
    fresh_turbidity,
    mud_turbidity,
    salt_turbidity
])

tds = np.concatenate([
    fresh_tds,
    mud_tds,
    salt_tds
])

temperature = np.concatenate([
    fresh_temperature,
    mud_temperature,
    salt_temperature
])


# ============================================================
# WATER CONDITION
# ============================================================

water_condition = np.array(
    ["FRESH"] * SAMPLES_PER_CLASS +
    ["MUD"] * SAMPLES_PER_CLASS +
    ["SALT"] * SAMPLES_PER_CLASS
)


fresh = water_condition == "FRESH"
mud = water_condition == "MUD"
salt = water_condition == "SALT"


# ============================================================
# SENSOR NOISE
# ============================================================

pH += np.random.normal(
    0,
    0.02,
    TOTAL_SAMPLES
)

turbidity += np.random.normal(
    0,
    3,
    TOTAL_SAMPLES
)

tds += np.random.normal(
    0,
    8,
    TOTAL_SAMPLES
)

temperature += np.random.normal(
    0,
    0.08,
    TOTAL_SAMPLES
)


# ============================================================
# CLIP BACK TO EXACT RANGES
# ============================================================

# FRESH

pH[fresh] = np.clip(
    pH[fresh],
    6.5,
    7.5
)

turbidity[fresh] = np.clip(
    turbidity[fresh],
    0,
    500
)

tds[fresh] = np.clip(
    tds[fresh],
    0,
    100
)

temperature[fresh] = np.clip(
    temperature[fresh],
    25,
    28
)


# MUD

pH[mud] = np.clip(
    pH[mud],
    5.5,
    8.5
)

turbidity[mud] = np.clip(
    turbidity[mud],
    0,
    100
)

tds[mud] = np.clip(
    tds[mud],
    400,
    700
)

temperature[mud] = np.clip(
    temperature[mud],
    25,
    28
)


# SALT

pH[salt] = np.clip(
    pH[salt],
    6.5,
    8.5
)

turbidity[salt] = np.clip(
    turbidity[salt],
    0,
    600
)

tds[salt] = np.clip(
    tds[salt],
    1000,
    1800
)

temperature[salt] = np.clip(
    temperature[salt],
    25,
    28
)


# ============================================================
# RISK COMPONENTS
# ============================================================

ph_risk = np.zeros(
    TOTAL_SAMPLES
)

turbidity_risk = np.zeros(
    TOTAL_SAMPLES
)

tds_risk = np.zeros(
    TOTAL_SAMPLES
)

temperature_risk = np.zeros(
    TOTAL_SAMPLES
)


# ============================================================
# pH RISK
# ============================================================

ph_risk[fresh] = np.clip(
    (
        np.abs(
            pH[fresh] - 7.0
        ) / 0.5
    ) * 20,
    0,
    20
)

ph_risk[mud] = np.clip(
    (
        np.abs(
            pH[mud] - 7.0
        ) / 1.5
    ) * 40,
    0,
    40
)

ph_risk[salt] = np.clip(
    (
        np.abs(
            pH[salt] - 7.0
        ) / 1.5
    ) * 30,
    0,
    30
)


# ============================================================
# TURBIDITY RISK
# ============================================================

turbidity_risk[fresh] = (
    turbidity[fresh] / 500
) * 20

turbidity_risk[mud] = (
    turbidity[mud] / 100
) * 30

turbidity_risk[salt] = (
    turbidity[salt] / 600
) * 25


# ============================================================
# TDS RISK
# ============================================================

tds_risk[fresh] = (
    tds[fresh] / 100
) * 20

tds_risk[mud] = (
    (
        tds[mud] - 400
    ) / 300
) * 60 + 25

tds_risk[salt] = (
    (
        tds[salt] - 1000
    ) / 800
) * 50 + 50


# ============================================================
# TEMPERATURE RISK
# ============================================================

temperature_risk = np.clip(
    (
        np.abs(
            temperature - 26.5
        ) / 1.5
    ) * 10,
    0,
    10
)


# ============================================================
# COMBINED RISK
# ============================================================

risk_score = (
    0.30 * ph_risk +
    0.25 * turbidity_risk +
    0.35 * tds_risk +
    0.10 * temperature_risk
)


# ============================================================
# CONDITION-SPECIFIC RISK CHARACTERISTICS
# ============================================================

# Fresh water remains generally lower-risk.
risk_score[fresh] = np.clip(
    risk_score[fresh],
    0,
    45
)

# Mud water is medium/high risk.
risk_score[mud] = np.clip(
    risk_score[mud],
    20,
    85
)

# Salt water remains high risk because of TDS.
risk_score[salt] = np.clip(
    risk_score[salt],
    55,
    100
)


# ============================================================
# REALISTIC VARIATION
# ============================================================

risk_score += np.random.normal(
    0,
    3.0,
    TOTAL_SAMPLES
)

risk_score = np.clip(
    risk_score,
    0,
    100
)


# ============================================================
# RISK LEVEL
# ============================================================

risk_level = np.select(
    [
        risk_score <= 30,
        risk_score <= 60,
        risk_score <= 80
    ],
    [
        "LOW",
        "MODERATE",
        "HIGH"
    ],
    default="CRITICAL"
)


# ============================================================
# IMPORTANT LABEL GENERATION
# ============================================================
#
# Instead of:
#
#     FRESH = GOOD
#     MUD   = BAD
#     SALT  = BAD
#
# or using separate hard thresholds that caused
# 8648 GOOD / 352 BAD,
#
# we use the GLOBAL risk distribution.
#
# Median risk = boundary between GOOD and BAD.
#
# This gives approximately balanced classes while
# still allowing the four sensor values to determine
# the classification.
# ============================================================

risk_boundary = np.median(
    risk_score
)

print(
    "\nCalculated risk boundary:",
    round(
        float(risk_boundary),
        2
    )
)


# ============================================================
# INITIAL LABEL
# ============================================================

label = (
    risk_score >= risk_boundary
).astype(
    int
)


# ============================================================
# SMALL AMOUNT OF REALISTIC LABEL NOISE
# ============================================================
#
# Only samples close to the decision boundary are allowed
# to have a small amount of uncertainty.
#
# This prevents the dataset from becoming a perfectly
# deterministic synthetic classification problem.

boundary_distance = np.abs(
    risk_score - risk_boundary
)


uncertain_indices = np.where(
    boundary_distance < 4.0
)[0]


# Only a small percentage of boundary samples are flipped.
flip_probability = 0.08

random_values = np.random.random(
    len(
        uncertain_indices
    )
)

flip_indices = uncertain_indices[
    random_values < flip_probability
]


label[
    flip_indices
] = 1 - label[
    flip_indices
]


# ============================================================
# ENSURE BOTH CLASSES
# ============================================================

class_counts = np.bincount(
    label,
    minlength=2
)

if class_counts[0] == 0 or class_counts[1] == 0:

    raise RuntimeError(
        "Dataset contains only one class."
    )


# ============================================================
# WATER QUALITY
# ============================================================

water_quality = np.where(
    label == 0,
    "GOOD",
    "BAD"
)


# ============================================================
# TIMESTAMPS
# ============================================================

start_time = datetime(
    2026,
    1,
    1,
    8,
    0,
    0
)

timestamps = [
    start_time +
    timedelta(
        minutes=i
    )
    for i in range(
        TOTAL_SAMPLES
    )
]


# ============================================================
# FEDERATED CLIENT IDs
# ============================================================

client_ids = [
    "client_" +
    str(
        (i % NUMBER_OF_CLIENTS) + 1
    )
    for i in range(
        TOTAL_SAMPLES
    )
]


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame({

    "timestamp": timestamps,

    "pH": pH,

    "turbidity": turbidity,

    "tds": tds,

    "temperature": temperature,

    "water_condition": water_condition,

    "label": label,

    "water_quality": water_quality,

    "risk_score": risk_score,

    "risk_level": risk_level,

    "client_id": client_ids

})


# ============================================================
# MISSING SENSOR VALUES
# ============================================================

missing_rate = 0.02

for column in [
    "pH",
    "turbidity",
    "tds",
    "temperature"
]:

    indices = np.random.choice(
        TOTAL_SAMPLES,
        int(
            TOTAL_SAMPLES *
            missing_rate
        ),
        replace=False
    )

    df.loc[
        indices,
        column
    ] = np.nan


# ============================================================
# SHUFFLE
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

df["pH"] = df[
    "pH"
].round(2)

df["turbidity"] = df[
    "turbidity"
].round(2)

df["tds"] = df[
    "tds"
].round(2)

df["temperature"] = df[
    "temperature"
].round(2)

df["risk_score"] = df[
    "risk_score"
].round(2)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "WATER QUALITY DATASET CREATED"
)

print(
    "=" * 70
)


print(
    "\nDataset shape:",
    df.shape
)


print(
    "\nWater condition:"
)

print(
    df[
        "water_condition"
    ].value_counts()
)


print(
    "\nGOOD / BAD distribution:"
)

print(
    df[
        "water_quality"
    ].value_counts()
)


print(
    "\nLabel distribution:"
)

print(
    df[
        "label"
    ].value_counts()
    .sort_index()
)


print(
    "\nRisk boundary:",
    round(
        float(risk_boundary),
        2
    )
)


print(
    "\nRisk statistics:"
)

print(
    df[
        "risk_score"
    ].describe()
)


print(
    "\nFederated clients:"
)

print(
    df[
        "client_id"
    ].value_counts()
    .sort_index()
)


print(
    "\nMissing values:"
)

print(
    df[
        [
            "pH",
            "turbidity",
            "tds",
            "temperature"
        ]
    ].isnull().sum()
)


# ============================================================
# RANGE VERIFICATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EXACT RANGE VERIFICATION"
)

print(
    "=" * 70
)


for condition in [
    "FRESH",
    "MUD",
    "SALT"
]:

    subset = df[
        df["water_condition"] == condition
    ]

    print(
        "\n" + condition
    )

    print(
        "pH:",
        subset["pH"].min(),
        "to",
        subset["pH"].max()
    )

    print(
        "Turbidity:",
        subset["turbidity"].min(),
        "to",
        subset["turbidity"].max()
    )

    print(
        "TDS:",
        subset["tds"].min(),
        "to",
        subset["tds"].max()
    )

    print(
        "Temperature:",
        subset["temperature"].min(),
        "to",
        subset["temperature"].max()
    )


# ============================================================
# SAMPLE DATA
# ============================================================

print(
    "\nFirst 10 records:"
)

print(
    df.head(
        10
    ).to_string(
        index=False
    )
)


print(
    "\nSaved:",
    OUTPUT_FILE
)


print(
    "\n" + "=" * 70
)

print(
    "DATASET GENERATION COMPLETED"
)

print(
    "=" * 70
)