import os
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset/Brain_Data_Organised"
SEED = 42

patient_data = {}

for label in ["Normal", "Stroke"]:
    folder = os.path.join(DATASET_DIR, label)
    patient_ids = sorted({
        filename.split(" ")[0]
        for filename in os.listdir(folder)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    })
    patient_data[label] = patient_ids

for label in ["Normal", "Stroke"]:
    patients = patient_data[label]

    train_ids, temp_ids = train_test_split(
        patients,
        test_size=0.20,
        random_state=SEED
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=SEED
    )

    print("\n" + "=" * 50)
    print(label)
    print("=" * 50)

    print("Total patients :", len(patients))
    print("Train patients :", len(train_ids))
    print("Validation     :", len(val_ids))
    print("Test patients  :", len(test_ids))

    print("\nTrain:", sorted(train_ids))
    print("\nValidation:", sorted(val_ids))
    print("\nTest:", sorted(test_ids))

print("\nDone.")