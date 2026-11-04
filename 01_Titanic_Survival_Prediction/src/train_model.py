import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocessing import create_preprocessor


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "titanic.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "titanic_survival_model_retained.pkl"


# Selected features
FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

TARGET = "survived"

def load_and_prepare_data():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

    # Select features and target
    X = df[FEATURES]
    y = df[TARGET]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nData splitting completed!")
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    return X_train, X_test, y_train, y_test



def create_baseline_models():

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    return models

def build_model_pipelines(models):

    pipelines = {}

    for name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("classifier", classifier)
        ])

        pipelines[name] = pipeline

    return pipelines

def evaluate_baseline_models(pipelines, X_train, y_train, X_test, y_test):

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    results = []

    for name, pipeline in pipelines.items():

        # Train model
        pipeline.fit(X_train, y_train)

        # Generate predictions
        predictions = pipeline.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print(f"\n{name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    results_df = pd.DataFrame(results)

    print("\nBaseline Model Comparison:")
    print(results_df.to_string(index=False))

    return results_df

def perform_cross_validation(pipelines, X_train, y_train):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    print("\n5-Fold Cross-Validation Results")

    for name, pipeline in pipelines.items():

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        print(f"\n{name}")

        for metric in scoring:
            mean_score = scores[f"test_{metric}"].mean()
            print(f"Mean {metric}: {mean_score:.4f}")

    return cv

def tune_models(X_train, y_train, cv):

    param_grids = {

        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"]
        },

        "Decision Tree": {
            "classifier__max_depth": [None, 5, 10, 15, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__class_weight": [None, "balanced"]
        },

        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 15, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__class_weight": [None, "balanced"]
        }
    }

    base_models = create_baseline_models()
    pipelines = build_model_pipelines(base_models)

    tuned_models = {}

    for name, pipeline in pipelines.items():

        print(f"\nTuning {name}...")

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=20,
            scoring="f1",
            cv=cv,
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        tuned_models[name] = search

        print("Best Parameters:", search.best_params_)
        print("Best CV F1 Score:", round(search.best_score_, 4))

    return tuned_models

def evaluate_tuned_models(tuned_models, X_test, y_test):

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix
    )

    results = []

    for name, search in tuned_models.items():

        best_model = search.best_estimator_

        predictions = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print(f"\n{'=' * 45}")
        print(name)
        print(f"{'=' * 45}")

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

    results_df = pd.DataFrame(results)

    print("\nTuned Model Comparison:")
    print(results_df.to_string(index=False))

    return results_df

def save_final_model(tuned_models):

    # Select tuned Logistic Regression
    final_model = tuned_models[
        "Logistic Regression"
    ].best_estimator_

    # Create models directory if necessary
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save complete preprocessing + model pipeline
    joblib.dump(final_model, MODEL_PATH)

    print("\nFinal model saved successfully!")
    print("Model location:", MODEL_PATH)

    return final_model

def main():

    # Step 1: Load and split data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Step 2: Create baseline models
    models = create_baseline_models()
    pipelines = build_model_pipelines(models)

    # Step 3: Evaluate baseline models
    evaluate_baseline_models(
        pipelines,
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Step 4: Cross-validation
    cv = perform_cross_validation(
        pipelines,
        X_train,
        y_train
    )

    # Step 5: Hyperparameter tuning
    tuned_models = tune_models(
        X_train,
        y_train,
        cv
    )

    # Step 6: Evaluate tuned models
    evaluate_tuned_models(
        tuned_models,
        X_test,
        y_test
    )

    # Step 7: Save final model
    save_final_model(tuned_models)


if __name__ == "__main__":
    main()