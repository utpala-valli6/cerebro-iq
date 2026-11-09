import os
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_DIR = os.path.join("dataset", "Brain_Data_Organised")
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

def get_patient_id(path):
    filename = os.path.basename(path)
    return filename.split(" ")[0]

train_patients = set(get_patient_id(p) for p in X_train)
val_patients = set(get_patient_id(p) for p in X_val)
test_patients = set(get_patient_id(p) for p in X_test)

print("\n==============================")
print("PATIENT SPLIT CHECK")
print("==============================")

print("Training patients  :", len(train_patients))
print("Validation patients:", len(val_patients))
print("Test patients      :", len(test_patients))

print("\nTrain ∩ Validation:",
      len(train_patients & val_patients))

print("Train ∩ Test:",
      len(train_patients & test_patients))

print("Validation ∩ Test:",
      len(val_patients & test_patients))

print("\nPatients appearing in BOTH training and test:")
print(sorted(train_patients & test_patients))

print("\nPatients appearing in BOTH validation and test:")
print(sorted(val_patients & test_patients))