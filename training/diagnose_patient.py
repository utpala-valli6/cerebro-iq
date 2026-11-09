import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

DATASET_DIR = "dataset/Brain_Data_Organised"
MODEL_PATH = "model/stroke_model_patient.keras"
IMAGE_SIZE = (224, 224)

TEST_PATIENTS = {
    "Normal": {"113", "130", "49", "59", "62", "65"},
    "Stroke": {"74", "80", "88", "92"}
}

model = tf.keras.models.load_model(MODEL_PATH)

X = []
y = []
names = []

for label_name, label in [("Normal", 0), ("Stroke", 1)]:

    folder = os.path.join(DATASET_DIR, label_name)

    for filename in os.listdir(folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        patient_id = filename.split(" ")[0]

        if patient_id not in TEST_PATIENTS[label_name]:
            continue

        path = os.path.join(folder, filename)

        image = tf.keras.utils.load_img(
            path,
            target_size=IMAGE_SIZE,
            color_mode="rgb"
        )

        image = tf.keras.utils.img_to_array(image)
        image = image.astype("float32")

        X.append(image)
        y.append(label)
        names.append(filename)

X = np.array(X)
y = np.array(y)

probabilities = model.predict(X, verbose=0).ravel()

predictions = (probabilities >= 0.5).astype(int)

print("\n" + "=" * 60)
print("CEREBROIQ - CORRECT PATIENT MODEL DIAGNOSTIC")
print("=" * 60)

print("\nTest images:", len(y))
print("Actual Normal:", np.sum(y == 0))
print("Actual Stroke:", np.sum(y == 1))

print("\nPredicted Normal:", np.sum(predictions == 0))
print("Predicted Stroke:", np.sum(predictions == 1))

print("\nConfusion Matrix")
print("-----------------------------")
print(confusion_matrix(y, predictions))

print("\nClassification Report")
print("-----------------------------")
print(classification_report(
    y,
    predictions,
    target_names=["Normal", "Stroke"],
    zero_division=0
))

print("ROC-AUC:", round(roc_auc_score(y, probabilities), 4))

print("\nPrediction probability range")
print("-----------------------------")
print("Minimum:", round(float(probabilities.min()), 4))
print("Maximum:", round(float(probabilities.max()), 4))
print("Mean:", round(float(probabilities.mean()), 4))

print("\nDone.")