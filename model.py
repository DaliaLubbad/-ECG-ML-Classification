
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


def train_model(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the classifier on unseen test data.
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Evaluation")
    print("------------------------")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        xticks_rotation=45
    )

    
    plt.title("ECG Classification - Confusion Matrix")

    import os
    os.makedirs("results", exist_ok=True)

    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=300)
    plt.show()


    return predictions
