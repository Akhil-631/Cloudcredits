import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_reviews.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load training dataset
train_df = pd.read_csv(TRAIN_PATH)

X_train = train_df["cleaned_review"]
y_train = train_df["sentiment"]


print("\n========== FINAL MODEL TRAINING ==========")

print("\nTraining Dataset Shape:", train_df.shape)


# Common TF-IDF configuration
def create_tfidf():

    return TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )


# Define both models
models = {

    "Multinomial Naive Bayes": {

        "pipeline": Pipeline([
            ("tfidf", create_tfidf()),
            ("classifier", MultinomialNB())
        ]),

        "filename": "naive_bayes_model.pkl"
    },

    "Logistic Regression": {

        "pipeline": Pipeline([
            ("tfidf", create_tfidf()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),

        "filename": "logistic_regression_model.pkl"
    }
}


# Train and save both models
for model_name, details in models.items():

    print(f"\nTraining {model_name}...")

    model = details["pipeline"]

    model.fit(X_train, y_train)

    model_path = MODEL_DIR / details["filename"]

    joblib.dump(model, model_path)

    vocabulary_size = len(
        model.named_steps["tfidf"].vocabulary_
    )

    print("Training completed successfully!")

    print("Vocabulary Size:", vocabulary_size)

    print("Model Saved:", model_path)


print("\n========== ALL MODELS TRAINED ==========")