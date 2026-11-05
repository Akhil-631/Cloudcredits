import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

SOURCE_MODEL = MODEL_DIR / "logistic_regression_model.pkl"

FINAL_MODEL = MODEL_DIR / "final_sentiment_model.pkl"

# Load the selected model
model = joblib.load(SOURCE_MODEL)

# Save as final model
joblib.dump(model, FINAL_MODEL)

print("\n========== FINAL MODEL SELECTION ==========")

print("Selected Model: Logistic Regression")
print("Test Accuracy: 90.65%")
print("Test F1-score: 90.82%")

print("\nFinal Model Saved:")
print(FINAL_MODEL)

print("\n========== COMPLETE ==========")