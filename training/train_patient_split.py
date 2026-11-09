import os
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NORMAL_DIR = os.path.join(BASE_DIR, "dataset", "Brain_Data_Organised", "Normal")
STROKE_DIR = os.path.join(BASE_DIR, "dataset", "Brain_Data_Organised", "Stroke")

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

def get_patient_id(filename):
    return filename.split(" ")[0]

def collect_images(folder, label):
    data = []

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)
            patient = get_patient_id(filename)
            data.append((path, label, patient))

    return data

normal = collect_images(NORMAL_DIR, 0)
stroke = collect_images(STROKE_DIR, 1)

normal_patients = sorted(set(x[2] for x in normal))
stroke_patients = sorted(set(x[2] for x in stroke))

normal_train, normal_temp = train_test_split(
    normal_patients,
    test_size=0.20,
    random_state=SEED
)

normal_val, normal_test = train_test_split(
    normal_temp,
    test_size=0.50,
    random_state=SEED
)

stroke_train, stroke_temp = train_test_split(
    stroke_patients,
    test_size=0.20,
    random_state=SEED
)

stroke_val, stroke_test = train_test_split(
    stroke_temp,
    test_size=0.50,
    random_state=SEED
)

train_patients = set(normal_train + stroke_train)
val_patients = set(normal_val + stroke_val)
test_patients = set(normal_test + stroke_test)

train_data = [
    x for x in normal + stroke
    if x[2] in train_patients
]

val_data = [
    x for x in normal + stroke
    if x[2] in val_patients
]

test_data = [
    x for x in normal + stroke
    if x[2] in test_patients
]

print("\nPATIENT SPLIT")
print("Train images:", len(train_data))
print("Validation images:", len(val_data))
print("Test images:", len(test_data))

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    return image, tf.cast(label, tf.float32)

def make_dataset(data, shuffle=False):
    paths = [x[0] for x in data]
    labels = [x[1] for x in data]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            len(data),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

train_ds = make_dataset(train_data, True)
val_ds = make_dataset(val_data)
test_ds = make_dataset(test_data)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomContrast(0.15)
])

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(
    256,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.0005)
)(x)
x = tf.keras.layers.Dropout(0.40)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

checkpoint_path = os.path.join(
    MODEL_DIR,
    "stroke_model_final.keras"
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=7,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

print("\nSTAGE 1: TRAINING")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=callbacks
)

print("\nSTAGE 2: FINE-TUNING")

base_model.trainable = True

for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

model = tf.keras.models.load_model(checkpoint_path)

print("\nFINDING BEST THRESHOLD")

val_images = []
val_labels = []

for images, labels in val_ds:
    val_images.append(images.numpy())
    val_labels.append(labels.numpy())

val_images = np.concatenate(val_images)
val_labels = np.concatenate(val_labels)

val_probabilities = model.predict(
    val_images,
    verbose=0
).ravel()

best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.20, 0.81, 0.01):

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        val_labels,
        predictions,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print("Best threshold:", round(best_threshold, 2))
print("Validation F1:", round(best_f1, 4))

threshold_path = os.path.join(
    MODEL_DIR,
    "threshold_final.txt"
)

with open(threshold_path, "w") as f:
    f.write(str(best_threshold))

print("\nFINAL TEST EVALUATION")

test_images = []
test_labels = []

for images, labels in test_ds:
    test_images.append(images.numpy())
    test_labels.append(labels.numpy())

test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

test_probabilities = model.predict(
    test_images,
    verbose=0
).ravel()

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)

auc = roc_auc_score(
    test_labels,
    test_probabilities
)

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print("\n===================================")
print("CEREBROIQ FINAL MODEL RESULTS")
print("===================================")

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")
print("ROC-AUC   :", round(auc * 100, 2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=["Normal", "Stroke"],
        zero_division=0
    )
)

print("\nModel saved:")
print(checkpoint_path)

print("\nThreshold saved:")
print(threshold_path)