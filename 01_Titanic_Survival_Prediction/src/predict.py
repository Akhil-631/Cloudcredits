import joblib
import pandas as pd

from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "titanic_survival_model_retained.pkl"


def load_model():
    """Load the trained Titanic survival prediction model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


def predict_survival(passenger_data, model=None):
    """
    Predict whether a passenger would survive.

    Parameters:
        passenger_data: Dictionary containing passenger features.
        model: Optional preloaded machine learning model.

    Returns:
        Dictionary containing prediction and probabilities.
    """

    if model is None:
        model = load_model()

    passenger_df = pd.DataFrame([passenger_data])

    prediction = model.predict(passenger_df)[0]

    probabilities = model.predict_proba(passenger_df)[0]

    result = {
        "prediction": int(prediction),
        "prediction_label": (
            "Survived" if prediction == 1 else "Did Not Survive"
        ),
        "survival_probability": float(probabilities[1]),
        "non_survival_probability": float(probabilities[0])
    }

    return result