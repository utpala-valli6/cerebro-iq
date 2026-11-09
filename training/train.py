import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import train_test_split


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_DIR = os.path.join("dataset", "Brain_Data_Organised")
MODEL_DIR = "model"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 20

SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 2. FIND IMAGES
# ============================================================

normal_dir = os.path.join(DATASET_DIR, "Normal")
stroke_dir = os.path.join(DATASET_DIR, "Stroke")

extensions = (".jpg", ".jpeg", ".png", ".bmp")

normal_images = [
    os.path.join(normal_dir, f)
    for f in os.listdir(normal_dir)
    if f.lower().endswith(extensions)
]

stroke_images = [
    os.path.join(stroke_dir, f)
    for f in os.listdir(stroke_dir)
    if f.lower().endswith(extensions)
]

print("Normal images :", len(normal_images))
print("Stroke images :", len(stroke_images))
print("Total images  :", len(normal_images) + len(stroke_images))


# ============================================================
# 3. IMAGE + LABEL LIST
# ============================================================

image_paths = normal_images + stroke_images

labels = (
    [0] * len(normal_images)
    + [1] * len(stroke_images)
)

image_paths = np.array(image_paths)
labels = np.array(labels)


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    stratify=labels,
    random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=SEED
)

print("\nDataset split")
print("----------------------")
print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Testing    :", len(X_test))


# ============================================================
# 5. LOAD IMAGES
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
)


train_ds = (
    train_ds
    .shuffle(
        len(X_train),
        seed=SEED
    )
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


val_ds = (
    val_ds
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


test_ds = (
    test_ds
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential([

    layers.RandomRotation(0.02),

    layers.RandomZoom(
        height_factor=(-0.05, 0.08),
        width_factor=(-0.05, 0.08)
    ),

    layers.RandomTranslation(
        0.02,
        0.02
    ),

    layers.RandomContrast(0.08)

])


# ============================================================
# 7. EFFICIENTNETB0
# ============================================================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False


inputs = layers.Input(
    shape=(224, 224, 3)
)

x = augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# 8. FIRST TRAINING STAGE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


model_path = os.path.join(
    MODEL_DIR,
    "stroke_model_v4.keras"
)


callbacks_head = [

    ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


print("\n======================================")
print("STAGE 1 - TRAINING CLASSIFICATION HEAD")
print("======================================\n")


history_head = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=HEAD_EPOCHS,

    callbacks=callbacks_head
)


# ============================================================
# 9. FINE-TUNING
# ============================================================

print("\n======================================")
print("STAGE 2 - FINE TUNING EFFICIENTNET")
print("======================================\n")


model = tf.keras.models.load_model(
    model_path
)


base_model = model.layers[2]

base_model.trainable = True


# Freeze early layers
for layer in base_model.layers[:-30]:
    layer.trainable = False


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


callbacks_fine = [

    ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


history_fine = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=callbacks_fine
)


# ============================================================
# 10. LOAD BEST MODEL
# ============================================================

model = tf.keras.models.load_model(
    model_path
)


# ============================================================
# 11. TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

probabilities = model.predict(
    test_ds,
    verbose=1
).ravel()


# ============================================================
# 12. VALIDATION PREDICTIONS
# ============================================================

val_probabilities = model.predict(
    val_ds,
    verbose=1
).ravel()


# ============================================================
# 13. FIND BEST THRESHOLD
# ============================================================

thresholds = np.arange(
    0.20,
    0.81,
    0.01
)

best_threshold = 0.50
best_f1 = 0.0


for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    if score > best_f1:

        best_f1 = score

        best_threshold = threshold


print(
    "\nSelected threshold:",
    round(best_threshold, 2)
)


# ============================================================
# 14. SAVE THRESHOLD
# ============================================================

threshold_path = os.path.join(
    MODEL_DIR,
    "threshold.txt"
)

with open(
    threshold_path,
    "w"
) as file:

    file.write(
        str(best_threshold)
    )


# ============================================================
# 15. FINAL TEST PREDICTIONS
# ============================================================

predictions = (
    probabilities >= best_threshold
).astype(int)


# ============================================================
# 16. PERFORMANCE
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

auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n======================================")
print("CEREBROIQ - VERSION 4 TEST RESULTS")
print("======================================")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {auc:.4f}"
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")
print("-----------------------------")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Normal",
            "Stroke"
        ],
        zero_division=0
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix")
print("-----------------------------")

print(cm)


plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "CerebroIQ Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Normal", "Stroke"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Stroke"]
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "model/confusion_matrix_v4.png"
)

plt.show()


# ============================================================
# 19. COMPLETION
# ============================================================

print("\n======================================")
print("TRAINING COMPLETED")
print("======================================")

print(
    "Saved model:",
    model_path
)

print(
    "Saved threshold:",
    threshold_path
)

print(
    "Threshold:",
    round(best_threshold, 2)
)