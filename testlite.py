import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


# ============================================================
# FILES
# ============================================================

DATASET = "water_quality_preprocessed.csv"
KERAS_MODEL = "global_water_quality_idcnn.keras"
TFLITE_MODEL = "global_water_quality_idcnn.tflite"
SCALER_FILE = "water_quality_scaler.pkl"


# ============================================================
# LOAD DATA
# ============================================================

FEATURES = [
    "pH",
    "turbidity",
    "tds",
    "temperature"
]

df = pd.read_csv(DATASET)

X = df[FEATURES].values.astype(np.float32)
y = df["label"].values.astype(np.int32)


# ============================================================
# CREATE THE SAME TEST SET
# ============================================================

RANDOM_STATE = 42

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


# ============================================================
# LOAD THE SAME SCALER
# ============================================================

scaler = joblib.load(
    SCALER_FILE
)

X_test = scaler.transform(
    X_test
).astype(np.float32)

X_test = X_test.reshape(
    X_test.shape[0],
    4,
    1
)


print("=" * 70)
print("KERAS vs TFLITE IDCNN TEST")
print("=" * 70)

print("\nTest samples:", len(X_test))
print("Test input shape:", X_test.shape)


# ============================================================
# 1. KERAS MODEL
# ============================================================

print("\nLoading Keras IDCNN...")

keras_model = load_model(
    KERAS_MODEL
)

keras_predictions = keras_model.predict(
    X_test,
    verbose=0
)

keras_classes = np.argmax(
    keras_predictions,
    axis=1
)


# ============================================================
# 2. TFLITE MODEL
# ============================================================

print("Loading TFLite IDCNN...")

interpreter = tf.lite.Interpreter(
    model_path=TFLITE_MODEL
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("\nTFLite input:")
print("Shape:", input_details[0]["shape"])
print("Type:", input_details[0]["dtype"])

print("\nTFLite output:")
print("Shape:", output_details[0]["shape"])
print("Type:", output_details[0]["dtype"])


# ============================================================
# RUN TFLITE PREDICTION
# ============================================================

tflite_predictions = []

for sample in X_test:

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

    tflite_predictions.append(
        output[0]
    )


tflite_predictions = np.array(
    tflite_predictions
)

tflite_classes = np.argmax(
    tflite_predictions,
    axis=1
)


# ============================================================
# COMPARE
# ============================================================

class_match = np.mean(
    keras_classes == tflite_classes
)

probability_difference = np.max(
    np.abs(
        keras_predictions -
        tflite_predictions
    )
)


print("\n" + "=" * 70)
print("COMPARISON RESULTS")
print("=" * 70)

print(
    "\nKeras predictions:",
    len(keras_classes)
)

print(
    "TFLite predictions:",
    len(tflite_classes)
)

print(
    "\nPrediction agreement:",
    round(class_match * 100, 4),
    "%"
)

print(
    "Maximum probability difference:",
    probability_difference
)


# ============================================================
# SHOW SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 PREDICTIONS")
print("=" * 70)

for i in range(
    min(10, len(X_test))
):

    print(
        "\nSample",
        i + 1
    )

    print(
        "Actual:",
        "GOOD" if y_test[i] == 0 else "BAD"
    )

    print(
        "Keras:",
        "GOOD" if keras_classes[i] == 0 else "BAD",
        keras_predictions[i]
    )

    print(
        "TFLite:",
        "GOOD" if tflite_classes[i] == 0 else "BAD",
        tflite_predictions[i]
    )


print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)