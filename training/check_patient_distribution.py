import os
from collections import defaultdict

DATASET_DIR = "dataset/Brain_Data_Organised"

patients = {
    "Normal": defaultdict(list),
    "Stroke": defaultdict(list)
}

for label in ["Normal", "Stroke"]:
    folder = os.path.join(DATASET_DIR, label)

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            patient_id = filename.split(" ")[0]
            patients[label][patient_id].append(filename)

print("=" * 50)
print("CEREBROIQ - PATIENT DISTRIBUTION")
print("=" * 50)

for label in ["Normal", "Stroke"]:
    print(f"\n{label}")
    print("-" * 30)
    print("Patients :", len(patients[label]))
    print("Images   :", sum(len(x) for x in patients[label].values()))

    counts = [len(x) for x in patients[label].values()]

    print("Min images/patient :", min(counts))
    print("Max images/patient :", max(counts))
    print("Average images/patient :", round(sum(counts) / len(counts), 2))

print("\n" + "=" * 50)
print("PATIENT IDS")
print("=" * 50)

normal_ids = set(patients["Normal"].keys())
stroke_ids = set(patients["Stroke"].keys())

print("Normal patient IDs :", sorted(normal_ids))
print("Stroke patient IDs :", sorted(stroke_ids))

print("\nSame ID in both classes:")
print(sorted(normal_ids & stroke_ids))