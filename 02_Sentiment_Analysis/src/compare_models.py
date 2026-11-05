import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_validate


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_reviews.csv"


# Load training dataset
train_df = pd.read_csv(TRAIN_PATH)

X_train = train_df["cleaned_review"]
y_train = train_df["sentiment"]


print("\n========== MODEL COMPARISON ==========")

print("\nTraining Dataset Shape:", train_df.shape)


# Define TF-IDF configuration
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


# Define machine learning pipelines
models = {

    "Multinomial Naive Bayes": Pipeline([
        ("tfidf", tfidf),
        ("classifier", MultinomialNB())
    ]),

    "Logistic Regression": Pipeline([
        ("tfidf", tfidf),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])
}


# Define cross-validation strategy
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Evaluation metrics
scoring = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]


# Compare models
results = []

for model_name, pipeline in models.items():

    print(f"\nEvaluating: {model_name}")

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({

        "Model": model_name,

        "Accuracy": scores["test_accuracy"].mean(),

        "Precision": scores["test_precision_macro"].mean(),

        "Recall": scores["test_recall_macro"].mean(),

        "F1 Score": scores["test_f1_macro"].mean()

    })


# Display comparison
results_df = pd.DataFrame(results)

print("\n========== CROSS-VALIDATION RESULTS ==========")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n========== COMPARISON COMPLETE ==========")