import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_reviews.csv"
MODEL_DIR = PROJECT_ROOT / "models"


# Load test dataset
test_df = pd.read_csv(TEST_PATH)

X_test = test_df["cleaned_review"]
y_test = test_df["sentiment"]


# Model paths
model_paths = {
    "Multinomial Naive Bayes": MODEL_DIR / "naive_bayes_model.pkl",
    "Logistic Regression": MODEL_DIR / "logistic_regression_model.pkl"
}


results = []


print("\n========== MODEL EVALUATION ==========")

print("\nTesting Dataset Shape:", test_df.shape)


# Evaluate each model
for model_name, model_path in model_paths.items():

    print(f"\nEvaluating {model_name}...")

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, pos_label="positive"
    )

    recall = recall_score(
        y_test, y_pred, pos_label="positive"
    )

    f1 = f1_score(
        y_test, y_pred, pos_label="positive"
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        y_pred,
        labels=["negative", "positive"]
    ))


# Final comparison
results_df = pd.DataFrame(results)

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# Identify best model based on F1 score
best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nBest Performing Model:", best_model["Model"])

print("Best Accuracy:", round(best_model["Accuracy"] * 100, 2), "%")

print("Best F1 Score:", round(best_model["F1 Score"] * 100, 2), "%")


print("\n========== EVALUATION COMPLETE ==========")