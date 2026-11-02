
import numpy as np
from data_loader import load_ecg_record
from feature_extraction import extract_features
from model import train_model, evaluate_model

# Separate ECG records for training and testing
TRAIN_RECORDS = [
    "100", "101", "102", "103", "104", "105",
    "106", "107", "108", "109", "111", "112",
    "113", "114", "115", "116", "117", "118",
    "119", "121", "122", "123", "124",
    "208", "209", "210", "212", "213", "214",
    "215", "217", "219", "220", "221", "222",
    "223", "228", "230", "231", "232", "233",
    "234"
]
TEST_RECORDS = [
    "200", "201", "202", "203", "205", "207"
]

# Map MIT-BIH heartbeat annotations to five AAMI classes
AAMI_GROUPS = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q"
}


def prepare_dataset(record_ids):
    all_features = []
    all_labels = []

    for record_id in record_ids:
        print(f"Loading ECG record {record_id}...")

        record, annotation = load_ecg_record(record_id)
        features, labels = extract_features(record, annotation)

        # Keep only supported heartbeat annotations
        selected = np.isin(labels, list(AAMI_GROUPS.keys()))

        if np.any(selected):
            selected_features = features[selected]
            selected_labels = [
                AAMI_GROUPS[label] for label in labels[selected]
            ]

            all_features.append(selected_features)
            all_labels.append(np.array(selected_labels))

    if not all_features:
        raise ValueError("No matching ECG beats found.")

    return (
        np.vstack(all_features),
        np.concatenate(all_labels)
    )


def main():
    print("Preparing training data...")
    X_train, y_train = prepare_dataset(TRAIN_RECORDS)

    print("\nPreparing testing data...")
    X_test, y_test = prepare_dataset(TEST_RECORDS)

    print("\nTraining samples:", len(y_train))
    print("Testing samples:", len(y_test))

    print("\nTraining class distribution:")
    classes, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(classes, counts):
        print(f"{label}: {count}")

    print("\nTesting class distribution:")
    classes, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(classes, counts):
        print(f"{label}: {count}")

    print("\nTraining Random Forest...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
