from datetime import datetime
import os
import sqlite3

from flask import Flask, jsonify, render_template, request

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# SHAP
# ============================================================

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# FILE CONFIGURATION
# ============================================================

DATABASE_FILE = "water_quality.db"
SCALER_FILE = "water_quality_scaler.pkl"
TFLITE_MODEL_FILE = "global_water_quality_idcnn.tflite"


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_NAMES = [
    "pH",
    "Turbidity",
    "TDS",
    "Temperature"
]

collection_running = True


# ============================================================
# LATEST SENSOR DATA
# ============================================================
#
# pH IS INTENTIONALLY FIXED AT 7.3
#
# The ESP32 pH value is not used by the backend.
# ============================================================

latest_sensor_data = {

    "sensor_id": "ESP32_LOCAL_WIFI",

    # FIXED pH VALUE
    "ph": 7.3,

    "turbidity": 0.0,

    "tds": 0.0,

    "temperature": 25.0,

    "api_timestamp":
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}


# ============================================================
# LOAD TFLITE MODEL & SCALER
# ============================================================

if not os.path.exists(TFLITE_MODEL_FILE):

    raise FileNotFoundError(
        "TFLite model file not found. Put "
        + TFLITE_MODEL_FILE
        + " in the project folder."
    )


interpreter = tf.lite.Interpreter(
    model_path=TFLITE_MODEL_FILE
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

print(
    "TFLite model loaded:",
    TFLITE_MODEL_FILE
)


# ============================================================
# LOAD SCALER
# ============================================================

if not os.path.exists(SCALER_FILE):

    raise FileNotFoundError(
        "water_quality_scaler.pkl not found."
    )


scaler = joblib.load(SCALER_FILE)


SCALER_FEATURE_NAMES = (
    list(scaler.feature_names_in_)
    if hasattr(scaler, "feature_names_in_")
    else [
        "pH",
        "turbidity",
        "tds",
        "temperature"
    ]
)


# ============================================================
# DATABASE SETUP
# ============================================================

def get_db():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    connection = get_db()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sensor_records (

            db_id INTEGER PRIMARY KEY AUTOINCREMENT,

            sensor_id TEXT,

            ph REAL NOT NULL,

            turbidity REAL NOT NULL,

            tds REAL NOT NULL,

            temperature REAL NOT NULL,

            prediction TEXT NOT NULL,

            confidence REAL NOT NULL,

            risk_score REAL NOT NULL,

            risk_level TEXT NOT NULL,

            api_timestamp TEXT,

            stored_timestamp TEXT NOT NULL

        )
        """
    )

    connection.commit()

    connection.close()


initialize_database()


# ============================================================
# MODEL INPUT PREPARATION
# ============================================================

def prepare_model_input(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if values.ndim == 1:

        values = values.reshape(
            1,
            -1
        )

    scaled_values = scaler.transform(
        values
    )

    return scaled_values.reshape(
        scaled_values.shape[0],
        4,
        1
    )


# ============================================================
# MODEL PROBABILITIES
# ============================================================

def model_probabilities(values):

    model_input = prepare_model_input(
        values
    )

    predictions = []

    for sample in model_input:

        input_data = np.expand_dims(
            sample,
            axis=0
        ).astype(
            input_details[0]["dtype"]
        )

        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )

        predictions.append(
            output[0]
        )

    return np.asarray(
        predictions,
        dtype=np.float32
    )


def model_probability(values):

    return model_probabilities(values)[:, 1]


# ============================================================
# PIECEWISE RISK INTERPOLATION
# ============================================================

def interpolate_risk(
    value,
    points
):

    """
    Piecewise-linear risk interpolation.

    points:
        [(measurement, risk), ...]

    Risk is returned from 0 to 100.
    """

    value = float(value)

    # Below minimum point
    if value <= points[0][0]:

        return float(
            points[0][1]
        )

    # Find interval
    for i in range(
        1,
        len(points)
    ):

        x0, y0 = points[i - 1]

        x1, y1 = points[i]

        if value <= x1:

            fraction = (
                (value - x0)
                /
                (x1 - x0)
            )

            return float(
                y0
                +
                fraction * (y1 - y0)
            )

    # Above maximum point
    return float(
        points[-1][1]
    )


# ============================================================
# INDIVIDUAL PARAMETER RISK CALCULATIONS
# ============================================================

def calculate_parameter_risks(
    sensor_data
):

    ph = float(
        sensor_data["ph"]
    )

    turbidity = max(
        0.0,
        float(sensor_data["turbidity"])
    )

    tds = max(
        0.0,
        float(sensor_data["tds"])
    )

    temperature = float(
        sensor_data["temperature"]
    )


    # ========================================================
    # pH RISK
    # ========================================================
    #
    # Preferred operational range:
    # 6.5 - 8.5
    #
    # NOTE:
    # pH is fixed to 7.3 in this prototype.
    # Therefore this component normally contributes
    # essentially zero risk.
    # ========================================================

    if 6.5 <= ph <= 8.5:

        ph_risk = 0.0

    elif ph < 6.5:

        ph_risk = interpolate_risk(
            ph,
            [
                (0.0, 100),
                (4.0, 100),
                (5.0, 80),
                (6.0, 45),
                (6.5, 0),
            ]
        )

    else:

        ph_risk = interpolate_risk(
            ph,
            [
                (8.5, 0),
                (9.0, 35),
                (10.0, 70),
                (11.0, 90),
                (14.0, 100),
            ]
        )


    # ========================================================
    # TURBIDITY RISK
    # ========================================================
    #
    # <1 NTU  -> preferred
    # 1-5     -> increasing concern
    # 5-10    -> high
    # >10     -> severe
    #
    # ========================================================

    turbidity_risk = interpolate_risk(
        turbidity,
        [
            (0.0, 0),
            (1.0, 5),
            (2.0, 15),
            (5.0, 40),
            (10.0, 75),
            (20.0, 95),
            (50.0, 100),
        ]
    )


    # ========================================================
    # TDS RISK
    # ========================================================
    #
    # This is primarily a dissolved-solids /
    # acceptability indicator.
    #
    # 0-300       -> very low
    # 300-600     -> generally acceptable
    # 600-900     -> elevated
    # 900-1200    -> high
    # >1200       -> very high
    #
    # 500 ppm therefore receives only a modest risk
    # contribution rather than being treated as BAD.
    #
    # ========================================================

    tds_risk = interpolate_risk(
        tds,
        [
            (0.0, 5),
            (300.0, 5),
            (500.0, 15),
            (600.0, 20),
            (900.0, 50),
            (1200.0, 80),
            (1500.0, 95),
            (2000.0, 100),
        ]
    )


    # ========================================================
    # TEMPERATURE RISK
    # ========================================================
    #
    # Temperature is treated as an operational /
    # acceptability factor rather than a stand-alone
    # drinking-water safety limit.
    #
    # ========================================================

    if temperature <= 25.0:

        temperature_risk = 5.0

    else:

        temperature_risk = interpolate_risk(
            temperature,
            [
                (25.0, 5),
                (30.0, 15),
                (35.0, 35),
                (40.0, 60),
                (50.0, 85),
                (60.0, 100),
            ]
        )


    return {

        "ph": round(
            ph_risk,
            2
        ),

        "turbidity": round(
            turbidity_risk,
            2
        ),

        "tds": round(
            tds_risk,
            2
        ),

        "temperature": round(
            temperature_risk,
            2
        ),

    }

# ============================================================
# HUMAN-READABLE WATER SOLUTION / RECOMMENDATION ENGINE
# ============================================================

def generate_water_recommendations(
    sensor_data,
    prediction_result,
    shap_features
):

    ph = float(sensor_data["ph"])
    turbidity = float(sensor_data["turbidity"])
    tds = float(sensor_data["tds"])
    temperature = float(sensor_data["temperature"])

    quality = prediction_result.get(
        "quality_classification",
        "UNKNOWN"
    )

    risk_score = float(
        prediction_result.get(
            "risk_score",
            0.0
        )
    )

    recommendations = []

    attention_items = []

    # ========================================================
    # pH
    # ========================================================

    if ph < 6.0:

        attention_items.append(
            "pH is strongly acidic."
        )

        recommendations.append({
            "parameter": "pH",
            "status": "HIGH ATTENTION",
            "problem": (
                f"pH is {ph:.2f}, which is below "
                "the preferred range."
            ),
            "solution": (
                "Investigate the water source and consider "
                "appropriate neutralisation or alkalinity "
                "treatment before drinking use."
            )
        })

    elif ph < 6.5:

        attention_items.append(
            "pH is slightly acidic."
        )

        recommendations.append({
            "parameter": "pH",
            "status": "MONITOR",
            "problem": (
                f"pH is {ph:.2f}, slightly below "
                "the preferred operational range."
            ),
            "solution": (
                "Check the source and treatment system. "
                "If the value remains low, appropriate "
                "neutralisation treatment may be required."
            )
        })

    elif ph > 9.0:

        attention_items.append(
            "pH is strongly alkaline."
        )

        recommendations.append({
            "parameter": "pH",
            "status": "HIGH ATTENTION",
            "problem": (
                f"pH is {ph:.2f}, which is above "
                "the preferred range."
            ),
            "solution": (
                "Investigate the source and treatment "
                "system. Appropriate pH correction should "
                "be performed before drinking use."
            )
        })

    elif ph > 8.5:

        attention_items.append(
            "pH is slightly alkaline."
        )

        recommendations.append({
            "parameter": "pH",
            "status": "MONITOR",
            "problem": (
                f"pH is {ph:.2f}, slightly above "
                "the preferred operational range."
            ),
            "solution": (
                "Continue monitoring pH and inspect the "
                "treatment/source conditions if the value "
                "continues increasing."
            )
        })


    # ========================================================
    # TURBIDITY
    # ========================================================

    if turbidity > 10:

        attention_items.append(
            "Turbidity is very high."
        )

        recommendations.append({
            "parameter": "Turbidity",
            "status": "HIGH ATTENTION",
            "problem": (
                f"Turbidity is {turbidity:.2f} NTU, "
                "indicating a substantial amount of "
                "light-scattering suspended material."
            ),
            "solution": (
                "Do not rely on the water for drinking "
                "without further treatment. Inspect the "
                "source and filtration system. Sediment "
                "filtration and appropriate treatment should "
                "be considered, followed by another turbidity "
                "measurement."
            )
        })

    elif turbidity > 5:

        attention_items.append(
            "Turbidity is high."
        )

        recommendations.append({
            "parameter": "Turbidity",
            "status": "ATTENTION",
            "problem": (
                f"Turbidity is {turbidity:.2f} NTU, "
                "which is considerably above the preferred "
                "low-turbidity condition."
            ),
            "solution": (
                "Inspect or replace sediment/cartridge "
                "filters and check the water source for "
                "suspended particles. Re-measure turbidity "
                "after filtration."
            )
        })

    elif turbidity > 1:

        attention_items.append(
            "Turbidity is above the preferred low level."
        )

        recommendations.append({
            "parameter": "Turbidity",
            "status": "MONITOR",
            "problem": (
                f"Turbidity is {turbidity:.2f} NTU. "
                "The water is still below the 5 NTU "
                "attention boundary used by this prototype, "
                "but it is above the preferred low-turbidity "
                "condition."
            ),
            "solution": (
                "Continue monitoring. If turbidity increases, "
                "inspect the sediment filter or other "
                "filtration stages and re-measure after "
                "treatment."
            )
        })


    # ========================================================
    # TDS
    # ========================================================

    if tds > 1200:

        attention_items.append(
            "TDS is very high."
        )

        recommendations.append({
            "parameter": "TDS",
            "status": "HIGH ATTENTION",
            "problem": (
                f"TDS is {tds:.2f} ppm, indicating a "
                "very high concentration of dissolved "
                "material."
            ),
            "solution": (
                "Investigate the water source and determine "
                "what dissolved substances are present. "
                "Depending on the source composition, "
                "membrane treatment such as reverse osmosis "
                "may be considered. Re-test TDS after treatment."
            )
        })

    elif tds > 900:

        attention_items.append(
            "TDS is high."
        )

        recommendations.append({
            "parameter": "TDS",
            "status": "ATTENTION",
            "problem": (
                f"TDS is {tds:.2f} ppm, indicating "
                "elevated dissolved-solids concentration."
            ),
            "solution": (
                "Check the source water and treatment system. "
                "If TDS continues increasing, investigate the "
                "dissolved constituents and consider suitable "
                "treatment such as membrane filtration."
            )
        })

    elif tds > 600:

        attention_items.append(
            "TDS is elevated."
        )

        recommendations.append({
            "parameter": "TDS",
            "status": "MONITOR",
            "problem": (
                f"TDS is {tds:.2f} ppm, which is elevated "
                "compared with low-TDS water."
            ),
            "solution": (
                "Continue monitoring TDS. If it continues "
                "rising, investigate the source and determine "
                "which dissolved substances are responsible "
                "before selecting a treatment method."
            )
        })


    # ========================================================
    # TEMPERATURE
    # ========================================================

    if temperature > 50:

        attention_items.append(
            "Water temperature is extremely high."
        )

        recommendations.append({
            "parameter": "Temperature",
            "status": "HIGH ATTENTION",
            "problem": (
                f"Water temperature is {temperature:.1f}°C."
            ),
            "solution": (
                "Allow the water to return to an appropriate "
                "storage/serving temperature and investigate "
                "whether the sensor, storage environment or "
                "water source is causing the unusually high "
                "temperature."
            )
        })

    elif temperature > 40:

        attention_items.append(
            "Water temperature is unusually high."
        )

        recommendations.append({
            "parameter": "Temperature",
            "status": "ATTENTION",
            "problem": (
                f"Water temperature is {temperature:.1f}°C, "
                "which is unusually warm."
            ),
            "solution": (
                "Check the storage environment and source "
                "conditions. Keep drinking water appropriately "
                "stored and continue monitoring temperature."
            )
        })

    elif temperature > 35:

        attention_items.append(
            "Water temperature is warm."
        )

        recommendations.append({
            "parameter": "Temperature",
            "status": "MONITOR",
            "problem": (
                f"Water temperature is {temperature:.1f}°C."
            ),
            "solution": (
                "Monitor storage conditions and keep the "
                "water protected from excessive heat."
            )
        })


    # ========================================================
    # NO MAJOR PROBLEMS
    # ========================================================

    if len(recommendations) == 0:

        recommendations.append({

            "parameter": "Overall",

            "status": "NO MAJOR ACTION",

            "problem": (
                "No measured parameter is currently "
                "showing a significant condition requiring "
                "corrective treatment."
            ),

            "solution": (
                "Continue normal monitoring and periodically "
                "check the sensor readings for changes."
            )

        })


    # ========================================================
    # SHAP-BASED PRIORITY
    # ========================================================

    ranked_shap = sorted(
        shap_features,
        key=lambda item:
            abs(
                float(
                    item.get(
                        "shap_value",
                        0
                    )
                )
            ),
        reverse=True
    )


    top_shap_parameter = (
        ranked_shap[0]["feature"]
        if ranked_shap
        else None
    )


    # ========================================================
    # HUMAN SUMMARY
    # ========================================================

    if quality == "GOOD":

        if attention_items:

            summary = (
                "The water is currently classified as GOOD, "
                "but some parameters should continue to be "
                "monitored. No major corrective treatment is "
                "indicated by the current readings."
            )

        else:

            summary = (
                "The current measurements indicate good "
                "water quality with no major corrective "
                "action required. Continue routine monitoring."
            )

    elif quality == "AVERAGE":

        summary = (
            "The water is classified as AVERAGE. One or more "
            "parameters are showing elevated values. The "
            "identified parameters should be monitored and "
            "appropriate filtration or treatment considered "
            "if the condition persists."
        )

    else:

        summary = (
            "The water is classified as BAD based on the "
            "combined AI and sensor assessment. Corrective "
            "treatment and re-testing are recommended before "
            "considering the water suitable for drinking."
        )


    # ========================================================
    # SHAP PRIORITY MESSAGE
    # ========================================================

    shap_priority = ""

    if top_shap_parameter:

        shap_priority = (
            f"The AI model was influenced most strongly by "
            f"{top_shap_parameter}. This does not by itself "
            f"mean that this parameter is the most dangerous "
            f"physically; it indicates that it had the largest "
            f"influence on the model's prediction for this "
            f"particular sample."
        )


    return {

        "summary":
            summary,

        "shap_priority":
            shap_priority,

        "attention_items":
            attention_items,

        "recommendations":
            recommendations,

        "quality":
            quality,

        "risk_score":
            round(
                risk_score,
                2
            ),

    }
# ============================================================
# COMPOSITE RISK SCORE
# ============================================================

def calculate_risk_score(
    sensor_data,
    bad_probability
):

    risks = calculate_parameter_risks(
        sensor_data
    )


    # ========================================================
    # PHYSICAL SENSOR RISK
    # ========================================================
    #
    # Turbidity receives the highest weight.
    #
    # TDS is significant but is not treated as a direct
    # toxicity measurement.
    #
    # pH is currently fixed at 7.3.
    # ========================================================

    sensor_risk = (

        0.20 * risks["ph"]

        + 0.35 * risks["turbidity"]

        + 0.30 * risks["tds"]

        + 0.15 * risks["temperature"]

    )


    # ========================================================
    # AI MODEL RISK
    # ========================================================

    model_risk = (
        float(bad_probability)
        * 100.0
    )


    # ========================================================
    # HYBRID RISK
    # ========================================================
    #
    # 55% AI model
    # 45% physical sensor risk
    #
    # ========================================================

    final_risk = (

        0.55 * model_risk

        + 0.45 * sensor_risk

    )


    return round(
        float(
            np.clip(
                final_risk,
                0,
                100
            )
        ),
        2
    )


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(
    score
):

    if score <= 30:

        return "LOW"

    elif score <= 60:

        return "MODERATE"

    elif score <= 80:

        return "HIGH"

    else:

        return "CRITICAL"


# ============================================================
# QUALITY CLASSIFICATION
# ============================================================

def get_quality_classification(
    score
):

    if score <= 30:

        return "GOOD"

    elif score <= 60:

        return "AVERAGE"

    else:

        return "BAD"


# ============================================================
# WATER QUALITY PREDICTION
# ============================================================

def predict_water_quality(
    sensor_data
):

    # ========================================================
    # MODEL INPUT
    # ========================================================

    values = np.array(
        [
            sensor_data["ph"],
            sensor_data["turbidity"],
            sensor_data["tds"],
            sensor_data["temperature"],
        ],
        dtype=np.float32
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    probabilities = model_probabilities(
        values
    )[0]


    good_probability = float(
        probabilities[0]
    )

    bad_probability = float(
        probabilities[1]
    )


    predicted_class = int(
        np.argmax(
            probabilities
        )
    )


    if predicted_class == 1:

        prediction = "BAD"

        confidence = float(
            np.clip(
                bad_probability * 100,
                51.0,
                98.75
            )
        )

    else:

        prediction = "GOOD"

        confidence = float(
            np.clip(
                good_probability * 100,
                51.0,
                98.75
            )
        )


    # ========================================================
    # PARAMETER RISKS
    # ========================================================

    parameter_risks = (
        calculate_parameter_risks(
            sensor_data
        )
    )


    # ========================================================
    # COMPOSITE RISK
    # ========================================================

    risk_score = calculate_risk_score(
        sensor_data,
        bad_probability
    )


    # ========================================================
    # HARD PROTOTYPE SAFETY OVERRIDES
    # ========================================================

    violations = []


    ph = float(
        sensor_data["ph"]
    )

    turbidity = float(
        sensor_data["turbidity"]
    )

    tds = float(
        sensor_data["tds"]
    )

    temperature = float(
        sensor_data["temperature"]
    )


    # --------------------------------------------------------
    # pH
    # --------------------------------------------------------

    if ph < 6.0 or ph > 9.0:

        violations.append(
            f"pH {ph:.2f} is outside the "
            f"prototype extreme operating boundary "
            f"of 6.0-9.0"
        )


    # --------------------------------------------------------
    # TURBIDITY
    # --------------------------------------------------------

    if turbidity > 10.0:

        violations.append(
            f"Turbidity {turbidity:.2f} NTU "
            f"exceeds the prototype severe-turbidity "
            f"boundary of 10 NTU"
        )


    # --------------------------------------------------------
    # TDS
    # --------------------------------------------------------
    #
    # 1200 ppm is treated as the upper prototype
    # acceptability boundary.
    #
    # It is NOT claimed to be a universal health limit.
    #
    # --------------------------------------------------------

    if tds > 1200.0:

        violations.append(
            f"TDS {tds:.2f} ppm is above the "
            f"prototype very-poor-acceptability "
            f"boundary of 1200 ppm"
        )


    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------
    #
    # Very high temperature triggers an operational
    # warning/override.
    #
    # --------------------------------------------------------

    if temperature > 50.0:

        violations.append(
            f"Water temperature "
            f"{temperature:.2f} C is extremely "
            f"high for normal drinking-water use"
        )


    # ========================================================
    # DETERMINE WHETHER OVERRIDE OCCURRED
    # ========================================================

    is_violation = (
        len(violations) > 0
    )


    violation_reason = (
        "; ".join(violations)
        if is_violation
        else ""
    )


    # ========================================================
    # APPLY OVERRIDE
    # ========================================================

    if is_violation:

        prediction = "BAD"

        confidence = max(
            confidence,
            95.0
        )

        risk_score = max(
            risk_score,
            85.0
        )


    # ========================================================
    # IMPORTANT:
    # CLASSIFICATION IS CALCULATED AFTER OVERRIDE
    # ========================================================

    risk_level = get_risk_level(
        risk_score
    )

    quality_classification = (
        get_quality_classification(
            risk_score
        )
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "prediction":
            prediction,

        "confidence":
            round(
                confidence,
                2
            ),

        "bad_probability":
            round(
                bad_probability * 100,
                2
            ),

        "good_probability":
            round(
                good_probability * 100,
                2
            ),

        "risk_score":
            round(
                risk_score,
                2
            ),

        "risk_level":
            risk_level,

        "quality_classification":
            quality_classification,

        "parameter_risks":
            parameter_risks,

        "override_triggered":
            is_violation,

        "override_reason":
            violation_reason,
    }


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def calculate_shap_explanation(
    sensor_data,
    prediction_result=None
):

    if not SHAP_AVAILABLE:

        return {

            "available":
                False,

            "message":
                "SHAP is not installed. "
                "Run: pip install shap",

            "explanation":
                "SHAP is not installed.",
        }


    # ========================================================
    # CURRENT SAMPLE
    # ========================================================

    current_values = np.array(
        [[
            sensor_data["ph"],
            sensor_data["turbidity"],
            sensor_data["tds"],
            sensor_data["temperature"],
        ]],
        dtype=np.float32
    )


    try:

        # ====================================================
        # SHAP BACKGROUND DATA
        # ====================================================
        #
        # Covers low-risk, moderate and degraded conditions.
        # ====================================================

        background = np.array(
            [

                [7.2, 0.3, 50, 22],

                [7.3, 0.5, 100, 24],

                [7.0, 0.8, 200, 25],

                [7.5, 1.5, 300, 26],

                [7.4, 3.0, 500, 28],

                [7.2, 5.0, 600, 30],

                [6.8, 7.0, 800, 32],

                [6.5, 10.0, 1000, 35],

                [6.0, 15.0, 1200, 40],

            ],
            dtype=np.float32
        )


        # ====================================================
        # KERNEL SHAP
        # ====================================================

        explainer = shap.KernelExplainer(
            model_probability,
            background
        )


        shap_values = explainer.shap_values(
            current_values,
            nsamples=150
        )


        # ====================================================
        # NORMALIZE SHAP OUTPUT SHAPE
        # ====================================================

        values = (

            np.asarray(
                shap_values[0]
            )

            if isinstance(
                shap_values,
                list
            )

            else np.asarray(
                shap_values
            )

        )


        if values.ndim == 3:

            values = values[
                0,
                :,
                0
            ]

        elif values.ndim == 2:

            values = values[0]

        else:

            values = values.reshape(
                -1
            )


        values = np.asarray(
            values,
            dtype=np.float64
        ).reshape(
            -1
        )


        # ====================================================
        # PARAMETER RISKS
        # ====================================================

        parameter_risks = (
            calculate_parameter_risks(
                sensor_data
            )
        )


        # ====================================================
        # CURRENT SENSOR VALUES
        # ====================================================

        ph = float(
            sensor_data["ph"]
        )

        turbidity = float(
            sensor_data["turbidity"]
        )

        tds = float(
            sensor_data["tds"]
        )

        temperature = float(
            sensor_data["temperature"]
        )


        # ====================================================
        # PARAMETER INTERPRETATION FUNCTIONS
        # ====================================================

        def describe_ph(value):

            if 6.5 <= value <= 8.5:

                return (
                    "within the preferred "
                    "operational pH range"
                )

            elif 6.0 <= value < 6.5:

                return (
                    "slightly acidic and outside "
                    "the preferred range"
                )

            elif 8.5 < value <= 9.0:

                return (
                    "slightly alkaline and outside "
                    "the preferred range"
                )

            else:

                return (
                    "well outside the preferred "
                    "operational pH range"
                )


        def describe_turbidity(value):

            if value < 1:

                return (
                    "very low turbidity and within "
                    "the preferred <1 NTU target"
                )

            elif value <= 5:

                return (
                    "elevated above the preferred "
                    "1 NTU target but below 5 NTU"
                )

            elif value <= 10:

                return (
                    "high turbidity requiring attention"
                )

            else:

                return (
                    "very high turbidity for "
                    "drinking-water use"
                )


        def describe_tds(value):

            if value <= 300:

                return (
                    "low dissolved-solids "
                    "concentration"
                )

            elif value <= 600:

                return (
                    "moderate dissolved-solids "
                    "concentration with generally "
                    "acceptable palatability"
                )

            elif value <= 900:

                return (
                    "elevated dissolved-solids "
                    "concentration"
                )

            elif value <= 1200:

                return (
                    "high dissolved-solids "
                    "concentration with increasingly "
                    "poor palatability"
                )

            else:

                return (
                    "very high dissolved-solids "
                    "concentration"
                )


        def describe_temperature(value):

            if value <= 25:

                return (
                    "cool/normal temperature with "
                    "low temperature-related concern"
                )

            elif value <= 30:

                return (
                    "slightly warm but not by itself "
                    "evidence of unsafe water"
                )

            elif value <= 35:

                return (
                    "warm water with increased "
                    "operational and acceptability concern"
                )

            elif value <= 40:

                return (
                    "unusually warm water with "
                    "increased operational concern"
                )

            else:

                return (
                    "very high water temperature "
                    "requiring investigation"
                )


        descriptions = {

            "pH":
                describe_ph(ph),

            "Turbidity":
                describe_turbidity(
                    turbidity
                ),

            "TDS":
                describe_tds(tds),

            "Temperature":
                describe_temperature(
                    temperature
                ),

        }


        risk_lookup = {

            "pH":
                parameter_risks["ph"],

            "Turbidity":
                parameter_risks["turbidity"],

            "TDS":
                parameter_risks["tds"],

            "Temperature":
                parameter_risks["temperature"],

        }


        # ====================================================
        # BUILD FEATURE EXPLANATIONS
        # ====================================================

        features = []


        for name, value, shap_value in zip(
            FEATURE_NAMES,
            current_values[0],
            values
        ):

            shap_value = float(
                shap_value
            )


            if shap_value > 0:

                direction = (
                    "Pushes the AI model "
                    "toward BAD"
                )

            elif shap_value < 0:

                direction = (
                    "Pushes the AI model "
                    "toward GOOD"
                )

            else:

                direction = (
                    "Neutral AI contribution"
                )


            features.append({

                "feature":
                    name,

                "value":
                    round(
                        float(value),
                        3
                    ),

                "shap_value":
                    round(
                        shap_value,
                        6
                    ),

                "impact":
                    round(
                        abs(shap_value),
                        6
                    ),

                "direction":
                    direction,

                "parameter_risk":
                    risk_lookup[name],

                "interpretation":
                    descriptions[name],

            })


        # ====================================================
        # RANK FEATURES BY SHAP IMPACT
        # ====================================================

        ranked = sorted(
            features,
            key=lambda item:
                item["impact"],
            reverse=True
        )


        top_f = ranked[0]

        second_f = ranked[1]
        # ========================================================
        # HUMAN-READABLE SHAP INSIGHT
        # ========================================================

        parameter_risks = calculate_parameter_risks(
            sensor_data
        )

        if prediction_result:

            pred_label = prediction_result.get(
                "prediction",
                "UNKNOWN"
            )

            quality = prediction_result.get(
                "quality_classification",
                "UNKNOWN"
            )

            conf_val = prediction_result.get(
                "confidence",
                0.0
            )

            risk_val = prediction_result.get(
                "risk_score",
                0.0
            )

            risk_level = prediction_result.get(
                "risk_level",
                "UNKNOWN"
            )

        else:

            pred_label = "UNKNOWN"
            quality = "UNKNOWN"
            conf_val = 0.0
            risk_val = 0.0
            risk_level = "UNKNOWN"


        # ========================================================
        # FIND AI-INFLUENCING FEATURES
        # ========================================================

        positive_features = [
            f for f in features
            if f["shap_value"] > 0
        ]

        negative_features = [
            f for f in features
            if f["shap_value"] < 0
        ]


        positive_features = sorted(
            positive_features,
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )

        negative_features = sorted(
            negative_features,
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )


        # ========================================================
        # HUMAN SHAP SENTENCE
        # ========================================================

        if quality == "GOOD":

            opening = (
                f"The water is currently classified as GOOD "
                f"with a LOW overall risk score of "
                f"{risk_val:.2f}/100. "
            )

        elif quality == "AVERAGE":

            opening = (
                f"The water is currently classified as AVERAGE "
                f"with a {risk_level} risk level and a composite "
                f"risk score of {risk_val:.2f}/100. "
            )

        else:

            opening = (
                f"The water is currently classified as BAD "
                f"with a {risk_level} risk level and a composite "
                f"risk score of {risk_val:.2f}/100. "
            )


        # ========================================================
        # MODEL EXPLANATION
        # ========================================================

        model_sentence = (
            f"The AI model predicted {pred_label} with "
            f"{conf_val:.2f}% confidence. "
        )


        if positive_features:

            strongest_bad = positive_features[0]

            model_sentence += (
                f"The parameter influencing the model most toward "
                f"a poorer-quality prediction was "
                f"{strongest_bad['feature']} "
                f"(SHAP impact {abs(strongest_bad['shap_value']):.4f}). "
            )


        if negative_features:

            strongest_good = negative_features[0]

            model_sentence += (
                f"The strongest parameter supporting the GOOD "
                f"prediction was {strongest_good['feature']} "
                f"(SHAP impact {abs(strongest_good['shap_value']):.4f}). "
            )


        # ========================================================
        # HUMAN INTERPRETATION OF EACH SENSOR
        # ========================================================

        human_sensor_explanations = []

        for feature in ranked:

            name = feature["feature"]

            value = feature["value"]

            shap_value = feature["shap_value"]

            parameter_risk = feature["parameter_risk"]


            if name == "pH":

                if 6.5 <= value <= 8.5:

                    text = (
                        f"pH is {value:.2f}, which is within the "
                        f"preferred operating range. It is therefore "
                        f"not currently a major source of concern."
                    )

                else:

                    text = (
                        f"pH is {value:.2f}, which is outside the "
                        f"preferred operating range and should be "
                        f"investigated."
                    )


            elif name == "Turbidity":

                if value < 1:

                    text = (
                        f"Turbidity is {value:.2f} NTU, indicating "
                        f"very clear water."
                    )

                elif value <= 5:

                    text = (
                        f"Turbidity is {value:.2f} NTU. The water "
                        f"is not highly turbid, but there is more "
                        f"suspended material than in very clear water. "
                        f"This parameter should be monitored."
                    )

                elif value <= 10:

                    text = (
                        f"Turbidity is {value:.2f} NTU, which is "
                        f"high enough to require attention to the "
                        f"filtration system."
                    )

                else:

                    text = (
                        f"Turbidity is {value:.2f} NTU, which is "
                        f"very high and requires corrective treatment "
                        f"and re-testing."
                    )


            elif name == "TDS":

                if value <= 300:

                    text = (
                        f"TDS is {value:.2f} ppm, indicating a "
                        f"low concentration of dissolved solids."
                    )

                elif value <= 600:

                    text = (
                        f"TDS is {value:.2f} ppm. This represents "
                        f"a moderate dissolved-solids concentration "
                        f"and is not, by itself, a reason to classify "
                        f"the water as unsafe."
                    )

                elif value <= 900:

                    text = (
                        f"TDS is {value:.2f} ppm, indicating "
                        f"elevated dissolved solids. The trend "
                        f"should be monitored."
                    )

                elif value <= 1200:

                    text = (
                        f"TDS is {value:.2f} ppm, indicating a "
                        f"high dissolved-solids concentration and "
                        f"potentially poorer taste/acceptability."
                    )

                else:

                    text = (
                        f"TDS is {value:.2f} ppm, which is very "
                        f"high and requires investigation of the "
                        f"source and treatment process."
                    )


            elif name == "Temperature":

                if value <= 25:

                    text = (
                        f"Temperature is {value:.1f}°C, which is "
                        f"relatively cool and contributes little "
                        f"temperature-related concern."
                    )

                elif value <= 30:

                    text = (
                        f"Temperature is {value:.1f}°C. The water "
                        f"is slightly warm, but this temperature "
                        f"alone does not indicate unsafe water."
                    )

                elif value <= 35:

                    text = (
                        f"Temperature is {value:.1f}°C, which is "
                        f"warm and should be monitored, especially "
                        f"during storage."
                    )

                elif value <= 40:

                    text = (
                        f"Temperature is {value:.1f}°C, which is "
                        f"unusually warm and deserves attention."
                    )

                else:

                    text = (
                        f"Temperature is {value:.1f}°C, which is "
                        f"extremely high for normal drinking-water "
                        f"conditions."
                    )


            if shap_value > 0:

                shap_text = (
                    f"For the AI model, this parameter slightly "
                    f"pushed the prediction toward BAD."
                )

            elif shap_value < 0:

                shap_text = (
                    f"For the AI model, this parameter supported "
                    f"the GOOD prediction."
                )

            else:

                shap_text = (
                    f"For the AI model, this parameter had almost "
                    f"no directional influence."
                )


            human_sensor_explanations.append(
                f"{text} {shap_text}"
            )


        # ========================================================
        # COMPLETE HUMAN-READABLE SHAP INSIGHT
        # ========================================================

        human_insight = (
            opening
            + model_sentence
            + " ".join(
                human_sensor_explanations
            )
            + " "
            + "SHAP shows which measurements influenced the "
            "AI model for this particular reading; it does "
            "not mean that the parameter with the largest "
            "SHAP value is automatically the most dangerous "
            "water-quality parameter. The final assessment "
            "also considers the engineered sensor-risk rules."
        )

        # ====================================================
        # PREDICTION INFORMATION
        # ====================================================

        if prediction_result:

            pred_label = (
                prediction_result.get(
                    "prediction",
                    "UNKNOWN"
                )
            )

            quality = (
                prediction_result.get(
                    "quality_classification",
                    "UNKNOWN"
                )
            )

            conf_val = (
                prediction_result.get(
                    "confidence",
                    0.0
                )
            )

            risk_val = (
                prediction_result.get(
                    "risk_score",
                    0.0
                )
            )

            risk_level = (
                prediction_result.get(
                    "risk_level",
                    "UNKNOWN"
                )
            )

            is_override = (
                prediction_result.get(
                    "override_triggered",
                    False
                )
            )

            override_msg = (
                prediction_result.get(
                    "override_reason",
                    ""
                )
            )

        else:

            pred_label = "UNKNOWN"

            quality = "UNKNOWN"

            conf_val = 0.0

            risk_val = 0.0

            risk_level = "UNKNOWN"

            is_override = False

            override_msg = ""


        # ====================================================
        # INDIVIDUAL FEATURE SENTENCES
        # ====================================================

        feature_sentences = []


        for feature in ranked:

            feature_sentences.append(

                f"{feature['feature']} = "
                f"{feature['value']}: "
                f"{feature['interpretation']}. "
                f"Rule-based parameter risk = "
                f"{feature['parameter_risk']:.1f}/100. "
                f"SHAP value = "
                f"{feature['shap_value']:+.5f}, "
                f"which "
                f"{feature['direction'].lower()}."

            )


        detailed_features = " ".join(
            feature_sentences
        )


        # ====================================================
        # MAIN SHAP NARRATIVE
        # ====================================================

        if is_override:

            opening = (

                "A deterministic prototype "
                "safety override was triggered. "

                f"Reason: {override_msg}. "

                f"The final classification is "
                f"{quality} with a "
                f"{risk_level} risk level and "
                f"composite risk score of "
                f"{risk_val:.2f}/100. "

            )

        else:

            opening = (

                f"The AI model prediction is "
                f"{pred_label} with "
                f"{conf_val:.2f}% confidence. "

                f"The combined prototype assessment "
                f"is {quality}, with a "
                f"{risk_level} risk level and "
                f"composite risk score of "
                f"{risk_val:.2f}/100. "

            )


        shap_summary = (

            f"Within the AI model itself, "
            f"{top_f['feature']} has the largest "
            f"SHAP influence for this sample "
            f"(|SHAP|={top_f['impact']:.5f}), "
            f"followed by "
            f"{second_f['feature']} "
            f"(|SHAP|={second_f['impact']:.5f}). "

        )


        methodology = (

            "SHAP values explain how each measured "
            "feature changes the neural-network "
            "prediction relative to the SHAP "
            "background dataset. A positive SHAP "
            "value pushes the model toward the BAD "
            "class, while a negative value pushes "
            "the model toward the GOOD class. "
            "The absolute SHAP value indicates the "
            "magnitude of that feature's influence "
            "on the model output. "

            "The rule-based parameter risk is kept "
            "separate from SHAP because SHAP explains "
            "the trained AI model, whereas parameter "
            "risk represents the engineered physical "
            "quality assessment. "

        )


        limitation = (

            "This prototype evaluates pH, turbidity, "
            "TDS and temperature only. These "
            "measurements cannot establish complete "
            "drinking-water potability because "
            "microbiological pathogens and many "
            "chemical contaminants are not directly "
            "measured. Therefore, the classification "
            "should be interpreted as a prototype "
            "water-quality assessment rather than "
            "a laboratory certification of potable water."

        )


        narrative = (

            opening
            + shap_summary
            + detailed_features
            + " "
            + methodology
            + limitation

        )


        # ====================================================
        # RETURN SHAP RESULT
        # ====================================================

        return {

            "available":
                True,

            "features":
                features,

            "ranked_features":
                ranked,

            "top_feature":
                top_f["feature"],

            "top_impact":
                top_f["impact"],

            "second_feature":
                second_f["feature"],

            "parameter_risks":
                parameter_risks,

            "message":
                narrative,

            "explanation":
                narrative,
            "human_insight": human_insight
            

        }


    except Exception as error:

        err_msg = (
            "SHAP calculation failed: "
            + str(error)
        )

        return {

            "available":
                False,

            "message":
                err_msg,

            "explanation":
                err_msg,

        }


# ============================================================
# DATABASE UTILITIES
# ============================================================

def store_record(
    sensor_data,
    prediction
):

    connection = get_db()

    stored_timestamp = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    cursor = connection.execute(

        """
        INSERT INTO sensor_records (

            sensor_id,
            ph,
            turbidity,
            tds,
            temperature,
            prediction,
            confidence,
            risk_score,
            risk_level,
            api_timestamp,
            stored_timestamp

        )

        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,

        (

            sensor_data["sensor_id"],

            sensor_data["ph"],

            sensor_data["turbidity"],

            sensor_data["tds"],

            sensor_data["temperature"],

            prediction["prediction"],

            prediction["confidence"],

            prediction["risk_score"],

            prediction["risk_level"],

            sensor_data["api_timestamp"],

            stored_timestamp,

        ),
    )


    connection.commit()


    record_id = (
        cursor.lastrowid
    )


    row = connection.execute(

        """
        SELECT *
        FROM sensor_records
        WHERE db_id = ?
        """,

        (
            record_id,
        )

    ).fetchone()


    connection.close()


    return dict(row)


# ============================================================
# GET LATEST FIVE RECORDS
# ============================================================

def get_latest_five():

    connection = get_db()

    rows = connection.execute(

        """
        SELECT *
        FROM sensor_records
        ORDER BY db_id DESC
        LIMIT 5
        """

    ).fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# GET ALL RECORDS
# ============================================================

def get_all_records():

    connection = get_db()

    rows = connection.execute(

        """
        SELECT *
        FROM sensor_records
        ORDER BY db_id DESC
        """

    ).fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# DATABASE COUNT
# ============================================================

def get_database_count():

    connection = get_db()

    count = connection.execute(

        """
        SELECT COUNT(*)
        FROM sensor_records
        """

    ).fetchone()[0]

    connection.close()

    return count


# ============================================================
# ESP32 SENSOR UPLOAD
# ============================================================

@app.route(
    "/api/sensor_upload",
    methods=["POST"]
)
def sensor_upload():

    global latest_sensor_data

    try:

        content = request.get_json(
            force=True,
            silent=True
        )


        if content:

            latest_sensor_data = {

                "sensor_id":
                    str(
                        content.get(
                            "id",
                            "ESP32_LOCAL_WIFI"
                        )
                    ),

                # =========================================
                # pH IS ALWAYS FIXED AT 7.3
                # =========================================

                "ph":
                    7.3,

                "turbidity":
                    float(
                        content.get(
                            "turbidity",
                            0.0
                        )
                    ),

                "tds":
                    float(
                        content.get(
                            "tds",
                            0.0
                        )
                    ),

                "temperature":
                    float(
                        content.get(
                            "temperature",
                            content.get(
                                "temp",
                                content.get(
                                    "Temp",
                                    25.0
                                )
                            )
                        )
                    ),

                "api_timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
            }


            return jsonify({

                "status":
                    "success",

                "message":
                    "Telemetry accepted"

            }), 200


    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 400


    return jsonify({

        "status":
            "error",

        "message":
            "Missing JSON payload"

    }), 400


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# DATABASE PAGE
# ============================================================

@app.route("/database")
def database_page():

    records = get_all_records()

    return render_template(

        "database.html",

        records=records,

        database_count=len(
            records
        )

    )


# ============================================================
# DATABASE API
# ============================================================

@app.route("/api/database")
def database_api():

    records = get_all_records()

    return jsonify({

        "success":
            True,

        "count":
            len(records),

        "records":
            records

    })


# ============================================================
# LATEST RECORDS API
# ============================================================

@app.route("/api/latest")
def latest_api():

    records = get_latest_five()

    return jsonify({

        "success":
            True,

        "count":
            get_database_count(),

        "records":
            records

    })


# ============================================================
# START COLLECTION
# ============================================================

@app.route(
    "/start",
    methods=["POST"]
)
def start_collection():

    global collection_running

    collection_running = True

    return jsonify({

        "success":
            True,

        "running":
            True,

        "message":
            "Monitoring started."

    })


# ============================================================
# STOP COLLECTION
# ============================================================

@app.route(
    "/stop",
    methods=["POST"]
)
def stop_collection():

    global collection_running

    collection_running = False

    return jsonify({

        "success":
            True,

        "running":
            False,

        "message":
            "Monitoring stopped."

    })


# ============================================================
# COLLECT / PREDICT / EXPLAIN
# ============================================================

@app.route("/collect")
def collect():

    if not collection_running:

        return jsonify({

            "success":
                False,

            "running":
                False,

            "message":
                "Monitoring is stopped.",

        })


    sensor_data = latest_sensor_data


    # ========================================================
    # AI PREDICTION
    # ========================================================

    prediction = predict_water_quality(
        sensor_data
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    explanation = calculate_shap_explanation(

        sensor_data,

        prediction

    )

    solution = generate_water_recommendations(
    sensor_data,
    prediction,
    explanation.get(
        "features",
        []
    )
    )
    # ========================================================
    # STORE DATABASE RECORD
    # ========================================================

    stored_record = store_record(

        sensor_data,

        prediction

    )


    latest_five = get_latest_five()


    # ========================================================
    # RETURN COMPLETE RESPONSE
    # ========================================================

    return jsonify({

        "success":
            True,

        "running":
            True,

        "record":
            stored_record,

        "prediction":
            prediction,

        "explanation":
            explanation,

        "latest_five":
            latest_five,

        "database_count":
            get_database_count(),

    })


# ============================================================
# EXPLAIN API
# ============================================================
@app.route(
    "/explain",
    methods=["POST"]
)
def explain():

    try:

        sensor_data = latest_sensor_data


        prediction = predict_water_quality(
            sensor_data
        )


        explanation = calculate_shap_explanation(
            sensor_data,
            prediction
        )


        solution = generate_water_recommendations(
            sensor_data,
            prediction,
            explanation.get(
                "features",
                []
            )
        )


        return jsonify({

            "success": True,

            "sensor_data":
                sensor_data,

            "prediction":
                prediction,

            "explanation":
                explanation,

            "solution":
                solution,

        })


    except Exception as error:

        return jsonify({

            "success": False,

            "error":
                str(error)

        })


# ============================================================
# CLEAR DATABASE
# ============================================================

@app.route(
    "/clear_database",
    methods=["POST"]
)
def clear_database():

    connection = get_db()

    connection.execute(
        """
        DELETE FROM sensor_records
        """
    )

    connection.commit()

    connection.close()


    return jsonify({

        "success":
            True,

        "database_count":
            0,

        "message":
            "All database records deleted.",

    })


# ============================================================
# DATABASE COUNT
# ============================================================

@app.route("/database_count")
def database_count():

    return jsonify({

        "success":
            True,

        "count":
            get_database_count()

    })


# ============================================================
# LAUNCH SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "AQUA AI - LOCAL WI-FI SERVER ACTIVE"
    )

    print(
        "Listening on: "
        "http://192.168.1.4:5000"
    )

    print(
        "ESP32 Target: "
        "http://192.168.1.4:5000/api/sensor_upload"
    )

    print(
        "=" * 70
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )