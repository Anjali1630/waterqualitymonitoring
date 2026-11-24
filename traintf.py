import os
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "water_quality_preprocessed.csv"

MODEL_FILE = "global_water_quality_idcnn.keras"

TFLITE_MODEL_FILE = "global_water_quality_idcnn.tflite"

SCALER_FILE = "water_quality_scaler.pkl"

RESULTS_FOLDER = "results"

FEATURES = [
    "pH",
    "turbidity",
    "tds",
    "temperature"
]

RANDOM_STATE = 42

NUMBER_OF_CLIENTS = 3

# Controlled federated training
FEDERATED_ROUNDS = 15

LOCAL_EPOCHS = 5

FINE_TUNE_EPOCHS = 20

BATCH_SIZE = 32

LEARNING_RATE = 0.0003

FINE_TUNE_LR = 0.00005


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "WATER QUALITY IDCNN + FEDERATED LEARNING"
)

print(
    "KNN + KALMAN + MINMAX + SMOTE + IDCNN + FEDAVG"
)

print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET
)

print(
    "\nDataset shape:",
    df.shape
)


# ============================================================
# SENSOR FEATURES
# ============================================================

for feature in FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


df[FEATURES] = df[
    FEATURES
].fillna(
    df[FEATURES].median()
)


# ============================================================
# LABEL
# ============================================================

if "label" not in df.columns:

    raise ValueError(
        "The dataset must contain a 'label' column."
    )


y = df[
    "label"
].astype(
    int
).values


X = df[
    FEATURES
].values.astype(
    np.float32
)


print(
    "\nOriginal class distribution:"
)

print(
    pd.Series(
        y
    ).value_counts().sort_index()
)

print(
    "\n0 = GOOD"
)

print(
    "1 = BAD"
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

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


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# MINMAX SCALER
# FIT ONLY ON TRAINING DATA
# ============================================================

print(
    "\nApplying MinMax normalization..."
)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(
    X_train
).astype(
    np.float32
)

X_val = scaler.transform(
    X_val
).astype(
    np.float32
)

X_test = scaler.transform(
    X_test
).astype(
    np.float32
)


joblib.dump(
    scaler,
    SCALER_FILE
)

print(
    "Scaler saved:",
    SCALER_FILE
)

print(
    "TFLite model saved:",
    TFLITE_MODEL_FILE
)


# ============================================================
# SMOTE
# TRAINING DATA ONLY
# ============================================================

print(
    "\nApplying SMOTE..."
)

print(
    "\nBefore SMOTE:"
)

print(
    pd.Series(
        y_train
    ).value_counts().sort_index()
)


smote = SMOTE(
    random_state=RANDOM_STATE,
    k_neighbors=5
)


X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)


print(
    "\nAfter SMOTE:"
)

print(
    pd.Series(
        y_train
    ).value_counts().sort_index()
)


# ============================================================
# IDCNN INPUT
# 4 SENSOR FEATURES × 1 CHANNEL
# ============================================================

X_train = X_train.reshape(
    X_train.shape[0],
    4,
    1
)

X_val = X_val.reshape(
    X_val.shape[0],
    4,
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    4,
    1
)


print(
    "\nIDCNN input shape:",
    X_train.shape
)


# ============================================================
# GRAPH IMAGE FUNCTIONS
# ============================================================

def save_federated_graph():
    if not round_numbers:
        return

    plt.figure(figsize=(9, 5))
    plt.plot(
        round_numbers,
        np.asarray(global_accuracy) * 100,
        marker="o",
        linewidth=2
    )
    plt.xlabel("Federated Round")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Federated Learning Accuracy")
    plt.grid(True, alpha=0.3)
    plt.xticks(round_numbers)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_FOLDER, "federated_accuracy.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(
        round_numbers,
        global_loss,
        marker="o",
        linewidth=2
    )
    plt.xlabel("Federated Round")
    plt.ylabel("Validation Loss")
    plt.title("Federated Learning Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.xticks(round_numbers)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_FOLDER, "federated_loss.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


def save_confusion_matrix_image(cm):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0, 1], ["GOOD", "BAD"])
    plt.yticks([0, 1], ["GOOD", "BAD"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_FOLDER, "confusion_matrix.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


def save_final_metrics_image(accuracy, precision, recall, f1, roc_auc):
    names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    values = [
        accuracy * 100,
        precision * 100,
        recall * 100,
        f1 * 100,
        roc_auc * 100
    ]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, values)
    plt.ylabel("Score (%)")
    plt.title("Final Model Performance")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 1, 99),
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_FOLDER, "final_metrics.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# FEDERATED CLIENT CREATION
# ============================================================

client_X = [
    []
    for _ in range(
        NUMBER_OF_CLIENTS
    )
]

client_y = [
    []
    for _ in range(
        NUMBER_OF_CLIENTS
    )
]


# ============================================================
# STRATIFIED CLIENT DISTRIBUTION
# ============================================================

for class_id in [0, 1]:

    class_indices = np.where(
        y_train == class_id
    )[0]

    np.random.shuffle(
        class_indices
    )

    parts = np.array_split(
        class_indices,
        NUMBER_OF_CLIENTS
    )

    for client_id, indices in enumerate(
        parts
    ):

        client_X[
            client_id
        ].extend(
            X_train[
                indices
            ]
        )

        client_y[
            client_id
        ].extend(
            y_train[
                indices
            ]
        )


client_X = [

    np.asarray(
        data,
        dtype=np.float32
    )

    for data in client_X

]


client_y = [

    np.asarray(
        data,
        dtype=np.int32
    )

    for data in client_y

]


print(
    "\nFederated clients:"
)


for client_id in range(
    NUMBER_OF_CLIENTS
):

    print(
        "Client",
        client_id + 1,
        "samples:",
        len(
            client_X[
                client_id
            ]
        ),
        "distribution:",
        np.bincount(
            client_y[
                client_id
            ],
            minlength=2
        )
    )


# ============================================================
# IDCNN MODEL
# ============================================================

def create_idcnn(
    learning_rate=LEARNING_RATE
):

    inputs = tf.keras.Input(
        shape=(4, 1),
        name="sensor_input"
    )

    # --------------------------------------------------------
    # IDCNN BLOCK 1
    # Standard convolution followed by dilated convolution
    # --------------------------------------------------------

    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        activation="relu"
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        dilation_rate=2,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    # --------------------------------------------------------
    # IDCNN BLOCK 2
    # Expanded dilated feature extraction
    # --------------------------------------------------------

    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        dilation_rate=2,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    # --------------------------------------------------------
    # GLOBAL FEATURE AGGREGATION
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = tf.keras.layers.Dense(
        32,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.20
    )(x)

    outputs = tf.keras.layers.Dense(
        2,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="IDCNN"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# FEDERATED AVERAGING
# ============================================================

def federated_average(
    weights_list,
    client_sizes
):

    total_size = float(
        sum(
            client_sizes
        )
    )

    averaged_weights = []


    for layer_values in zip(
        *weights_list
    ):

        layer = np.zeros_like(
            layer_values[0]
        )


        for weights, size in zip(
            layer_values,
            client_sizes
        ):

            layer += (
                weights *
                (
                    size /
                    total_size
                )
            )


        averaged_weights.append(
            layer
        )


    return averaged_weights


# ============================================================
# INITIAL GLOBAL MODEL
# ============================================================

global_model = create_idcnn()


round_numbers = []

global_accuracy = []

global_loss = []


best_val_accuracy = 0.0

best_weights = (
    global_model.get_weights()
)


# ============================================================
# FEDERATED TRAINING
# ============================================================

for round_id in range(

    1,

    FEDERATED_ROUNDS + 1

):

    print(
        "\n" + "=" * 70
    )

    print(
        "FEDERATED IDCNN ROUND",
        round_id,
        "/",
        FEDERATED_ROUNDS
    )

    print(
        "=" * 70
    )


    client_weights = []

    client_sizes = []


    # ========================================================
    # LOCAL CLIENT TRAINING
    # ========================================================

    for client_id in range(
        NUMBER_OF_CLIENTS
    ):

        print(
            "\nTraining IDCNN Client",
            client_id + 1
        )


        local_model = create_idcnn()


        # Start every client from
        # the SAME global model

        local_model.set_weights(
            global_model.get_weights()
        )


        callbacks = [

            tf.keras.callbacks.EarlyStopping(

                monitor="val_loss",

                patience=2,

                restore_best_weights=True

            ),

            tf.keras.callbacks.ReduceLROnPlateau(

                monitor="val_loss",

                factor=0.5,

                patience=1,

                min_lr=0.00001

            )

        ]


        history = local_model.fit(

            client_X[
                client_id
            ],

            client_y[
                client_id
            ],

            epochs=LOCAL_EPOCHS,

            batch_size=BATCH_SIZE,

            validation_split=0.15,

            callbacks=callbacks,

            verbose=0

        )


        best_epoch = int(

            np.argmin(

                history.history[
                    "val_loss"
                ]

            )

        )


        train_acc = (

            history.history[
                "accuracy"
            ][
                best_epoch
            ]

        )


        val_acc = (

            history.history[
                "val_accuracy"
            ][
                best_epoch
            ]

        )


        print(
            "Training Accuracy:",
            round(
                train_acc * 100,
                2
            ),
            "%"
        )


        print(
            "Validation Accuracy:",
            round(
                val_acc * 100,
                2
            ),
            "%"
        )


        # Save client weights

        client_weights.append(
            local_model.get_weights()
        )


        client_sizes.append(
            len(
                client_X[
                    client_id
                ]
            )
        )


        del local_model


    # ========================================================
    # FEDAVG
    # ========================================================

    global_weights = federated_average(

        client_weights,

        client_sizes

    )


    global_model.set_weights(
        global_weights
    )


    # ========================================================
    # GLOBAL VALIDATION
    # ========================================================

    loss, accuracy = global_model.evaluate(

        X_val,

        y_val,

        verbose=0

    )


    print(
        "\nGlobal IDCNN Validation Accuracy:",
        round(
            accuracy * 100,
            2
        ),
        "%"
    )


    print(
        "Global IDCNN Validation Loss:",
        round(
            loss,
            4
        )
    )


    round_numbers.append(
        round_id
    )

    global_accuracy.append(
        accuracy
    )

    global_loss.append(
        loss
    )

    # SAVE FEDERATED RESULTS AFTER EVERY ROUND
    # This keeps the dashboard graph source current while
    # federated training is still running.
    federated_results_live = pd.DataFrame({
        "Round": round_numbers,
        "Global_Validation_Accuracy": global_accuracy,
        "Global_Validation_Loss": global_loss
    })

    federated_results_live.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "federated_round_results.csv"
        ),
        index=False
    )

    # Update graph images after every federated round.
    save_federated_graph()


    # ========================================================
    # SAVE BEST GLOBAL MODEL
    # ========================================================

    if accuracy > best_val_accuracy:

        best_val_accuracy = accuracy

        best_weights = (
            global_model.get_weights()
        )


# ============================================================
# RESTORE BEST FEDERATED MODEL
# ============================================================

global_model.set_weights(
    best_weights
)


print(
    "\nBest federated validation accuracy:",
    round(
        best_val_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# GLOBAL IDCNN FINE-TUNING
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "GLOBAL IDCNN FINE-TUNING"
)

print(
    "=" * 70
)


global_model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=FINE_TUNE_LR

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


fine_callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=0.000005

    )

]


fine_history = global_model.fit(

    X_train,

    y_train,

    epochs=FINE_TUNE_EPOCHS,

    batch_size=BATCH_SIZE,

    validation_data=(
        X_val,
        y_val
    ),

    callbacks=fine_callbacks,

    verbose=1

)


# ============================================================
# FINAL TEST
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL GLOBAL IDCNN MODEL EVALUATION"
)

print(
    "=" * 70
)


probabilities = global_model.predict(

    X_test,

    verbose=0

)


# ============================================================
# DIRECT SOFTMAX DECISION
# ============================================================

predictions = np.argmax(

    probabilities,

    axis=1

)


good_probability = probabilities[
    :, 0
]

bad_probability = probabilities[
    :, 1
]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(

    y_test,

    predictions

)


precision = precision_score(

    y_test,

    predictions,

    zero_division=0

)


recall = recall_score(

    y_test,

    predictions,

    zero_division=0

)


f1 = f1_score(

    y_test,

    predictions,

    zero_division=0

)


roc_auc = roc_auc_score(

    y_test,

    bad_probability

)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\nFinal Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print(
    "Final Precision:",
    round(
        precision * 100,
        2
    ),
    "%"
)


print(
    "Final Recall:",
    round(
        recall * 100,
        2
    ),
    "%"
)


print(
    "Final F1 Score:",
    round(
        f1 * 100,
        2
    ),
    "%"
)


print(
    "Final ROC-AUC:",
    round(
        roc_auc,
        4
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    predictions,

    target_names=[
        "GOOD",
        "BAD"
    ],

    zero_division=0

)


print(
    "\nClassification Report:"
)

print(
    report
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    predictions

)


print(
    "\nConfusion Matrix:"
)

print(
    cm
)

# SAVE CONFUSION MATRIX FOR THE DASHBOARD
confusion_matrix_df = pd.DataFrame(
    cm,
    index=["Actual_GOOD", "Actual_BAD"],
    columns=["Predicted_GOOD", "Predicted_BAD"]
)

confusion_matrix_df.to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "confusion_matrix.csv"
    )
)


# ============================================================
# TARGET INFORMATION
# ============================================================

print(
    "\nModel decision:"
)

print(
    "The model uses 2-class Softmax + argmax."
)

print(
    "No custom 0.34 decision threshold is used."
)


if accuracy >= 0.85:

    print(
        "\n85%+ ACCURACY ACHIEVED"
    )

else:

    print(
        "\n85% TARGET NOT REACHED IN THIS RUN."
    )

    print(
        "Do not force the result; "
        "improve the data/model based on evaluation."
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC-AUC"

    ],

    "Value": [

        accuracy,

        precision,

        recall,

        f1,

        roc_auc

    ],

    "Percentage": [

        accuracy * 100,

        precision * 100,

        recall * 100,

        f1 * 100,

        roc_auc * 100

    ]

})


metrics_df.to_csv(

    os.path.join(

        RESULTS_FOLDER,

        "metrics.csv"

    ),

    index=False

)

# DEDICATED ACCURACY GRAPH DATA
accuracy_history_df = pd.DataFrame({
    "Round": round_numbers,
    "Accuracy": global_accuracy,
    "Accuracy_Percentage": np.asarray(global_accuracy) * 100,
    "Validation_Loss": global_loss
})

accuracy_history_df.to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "accuracy_history.csv"
    ),
    index=False
)

# Final refresh of federated graph images.
save_federated_graph()

# FINAL KPI VALUES
final_metrics_df = pd.DataFrame([{
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc
}])

final_metrics_df.to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "final_metrics.csv"
    ),
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({

    "Actual_Label": y_test,

    "Predicted_Label": predictions,

    "GOOD_Probability": good_probability,

    "BAD_Probability": bad_probability,

    "Actual_Quality": np.where(

        y_test == 0,

        "GOOD",

        "BAD"

    ),

    "Predicted_Quality": np.where(

        predictions == 0,

        "GOOD",

        "BAD"

    )

})


predictions_df.to_csv(

    os.path.join(

        RESULTS_FOLDER,

        "final_predictions.csv"

    ),

    index=False

)


# ============================================================
# SAVE FEDERATED RESULTS
# ============================================================

federated_results = pd.DataFrame({

    "Round": round_numbers,

    "Global_Validation_Accuracy":
        global_accuracy,

    "Global_Validation_Loss":
        global_loss

})


federated_results.to_csv(

    os.path.join(

        RESULTS_FOLDER,

        "federated_round_results.csv"

    ),

    index=False

)

# FINAL ACCURACY HISTORY
accuracy_history_df = pd.DataFrame({
    "Round": round_numbers,
    "Accuracy": global_accuracy,
    "Accuracy_Percentage": np.asarray(global_accuracy) * 100,
    "Validation_Loss": global_loss
})

accuracy_history_df.to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "accuracy_history.csv"
    ),
    index=False
)

# FINAL CONFUSION MATRIX
confusion_matrix_df.to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "confusion_matrix.csv"
    )
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(

    os.path.join(

        RESULTS_FOLDER,

        "classification_report.txt"

    ),

    "w"

) as file:

    file.write(
        report
    )


# ============================================================
# SAVE MODEL
# ============================================================

global_model.save(

    MODEL_FILE

)


# ============================================================
# TENSORFLOW LITE CONVERSION
# ============================================================

print(
    "\nConverting final global IDCNN to TensorFlow Lite..."
)

converter = tf.lite.TFLiteConverter.from_keras_model(
    global_model
)

# Float32 TFLite keeps the same numerical input/output behavior
# as the trained Keras model and is directly testable in Python.
tflite_model = converter.convert()

with open(
    TFLITE_MODEL_FILE,
    "wb"
) as file:

    file.write(
        tflite_model
    )

print(
    "TFLite model saved:",
    TFLITE_MODEL_FILE
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TRAINING COMPLETED"
)

print(
    "=" * 70
)


print(
    "\nFinal Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print(
    "Final Precision:",
    round(
        precision * 100,
        2
    ),
    "%"
)


print(
    "Final Recall:",
    round(
        recall * 100,
        2
    ),
    "%"
)


print(
    "Final F1 Score:",
    round(
        f1 * 100,
        2
    ),
    "%"
)


print(
    "Final ROC-AUC:",
    round(
        roc_auc,
        4
    )
)


print(
    "\nArchitecture:"
)

print(
    "Conv1D(dilation=1) → BatchNorm → "
    "Conv1D(dilation=2) → BatchNorm → "
    "Conv1D(dilation=1) → BatchNorm → "
    "Conv1D(dilation=2) → BatchNorm → "
    "GlobalAveragePooling → Dense → Softmax"
)


print(
    "\nFederated Learning:"
)

print(
    "3 local IDCNN clients → FedAvg → Global IDCNN"
)


print(
    "\nDecision:"
)

print(
    "2-class Softmax + argmax"
)

print(
    "No custom 0.34 threshold"
)


print(
    "\nModel saved:",
    MODEL_FILE
)

print(
    "Scaler saved:",
    SCALER_FILE
)

print(
    "Results folder:",
    RESULTS_FOLDER
)

print(
    "\nGraph source CSVs:"
)

print("  confusion_matrix.csv")
print("  federated_round_results.csv")
print("  accuracy_history.csv")
print("  metrics.csv")
print("  final_metrics.csv")



print(
    "\n" + "=" * 70
)