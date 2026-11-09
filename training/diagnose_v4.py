import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

DATASET_DIR = os.path.join("dataset", "Brain_Data_Organised")
MODEL_PATH = os.path.join("model", "stroke_model_v4.keras")
THRESHOLD_PATH = os.path.join("model", "threshold.txt")

IMAGE_SIZE = (224, 224)
SEED = 42

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

image_paths = np.array(normal_images + stroke_images)

labels = np.array(
    [0] * len(normal_images) +
    [1] * len(stroke_images)
)

_, X_temp, _, y_temp = train_test_split(
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

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image

model = tf.keras.models.load_model(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    threshold = float(file.read().strip())

print("\nLoaded model:", MODEL_PATH)
print("Threshold:", threshold)
print("Test images:", len(X_test))

probabilities = []

for path in X_test:
    image = load_image(path)
    image = tf.expand_dims(image, axis=0)

    probability = float(
        model.predict(image, verbose=0)[0][0]
    )

    probabilities.append(probability)

probabilities = np.array(probabilities)

predictions = (
    probabilities >= threshold
).astype(int)

cm = confusion_matrix(y_test, predictions)

accuracy = accuracy_score(y_test, predictions)
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

print("\n==============================")
print("V4 DIAGNOSTIC RESULTS")
print("==============================")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\n==============================")
print("MISCLASSIFIED STROKE IMAGES")
print("==============================")

count = 0

for path, actual, probability, prediction in zip(
    X_test,
    y_test,
    probabilities,
    predictions
):
    if actual == 1 and prediction == 0:
        count += 1

        print("\nImage:", path)
        print("Actual: Stroke")
        print("Predicted: Normal")
        print("Stroke probability:", round(probability, 4))
        print("Normal confidence:", round((1 - probability) * 100, 2), "%")

print("\nTotal Stroke -> Normal errors:", count)