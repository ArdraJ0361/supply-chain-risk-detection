from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ---------------------------------------------------------
# 1. Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"

TARGET = "is_suspected_fraud"
RANDOM_STATE = 42


# ---------------------------------------------------------
# 2. Load labels and embeddings
# ---------------------------------------------------------

def load_data():
    print("Loading labels...")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    validation_df = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = np.load(
        EMBEDDINGS_DIR / "train_embeddings.npy"
    )

    X_validation = np.load(
        EMBEDDINGS_DIR / "validation_embeddings.npy"
    )

    y_train = train_df[TARGET].astype(int).to_numpy()
    y_validation = validation_df[TARGET].astype(int).to_numpy()

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training embedding count does not match training labels."
        )

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "Validation embedding count does not match validation labels."
        )

    print(f"Training embeddings: {X_train.shape}")
    print(f"Validation embeddings: {X_validation.shape}")
    print(f"Training fraud rate: {y_train.mean():.2%}")
    print(f"Validation fraud rate: {y_validation.mean():.2%}")

    return X_train, y_train, X_validation, y_validation


# ---------------------------------------------------------
# 3. Train embeddings-only XGBoost
# ---------------------------------------------------------

def train_model(X_train, y_train):
    print("\nTraining embeddings-only XGBoost model...")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError("Training data contains no positive examples.")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=negative_count / positive_count,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
    )

    model.fit(X_train, y_train)

    print("Training completed.")

    return model


# ---------------------------------------------------------
# 4. Evaluate on validation data
# ---------------------------------------------------------

def evaluate_model(model, X_validation, y_validation):
    print("\nEvaluating on validation data...")

    probabilities = model.predict_proba(X_validation)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "XGBoost text embeddings only",
        "average_precision_pr_auc": float(
            average_precision_score(y_validation, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_validation, probabilities)
        ),
        "precision": float(
            precision_score(
                y_validation, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_validation, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_validation, predictions, zero_division=0
            )
        ),
        "classification_threshold": 0.5,
    }

    print("\nValidation metrics:")

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            target_names=["Not suspected fraud", "Suspected fraud"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_validation, predictions))

    return metrics


# ---------------------------------------------------------
# 5. Save the model and metrics
# ---------------------------------------------------------

def save_results(model, metrics):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RESULTS_DIR / "embedding_model.joblib"
    metrics_path = RESULTS_DIR / "embedding_metrics.json"

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


# ---------------------------------------------------------
# 6. Main program
# ---------------------------------------------------------

def main():
    X_train, y_train, X_validation, y_validation = load_data()

    model = train_model(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    save_results(model, metrics)


if __name__ == "__main__":
    main()