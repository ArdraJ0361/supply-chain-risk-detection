from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

TARGET = "is_suspected_fraud"

NUMERIC_FEATURES = [
    "Product Price",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Days for shipment (scheduled)",
    "Latitude",
    "Longitude",
]

RANDOM_STATE = 42


def main():
    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    X_train = train_df[NUMERIC_FEATURES].copy()
    y_train = train_df[TARGET].astype(int)

    X_validation = validation_df[NUMERIC_FEATURES].copy()
    y_validation = validation_df[TARGET].astype(int)

    # Learn imputation values using training data only.
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_validation = imputer.transform(X_validation)

    # Train the anomaly detector on records labelled as non-fraud.
    normal_records = X_train[y_train.to_numpy() == 0]

    print(f"Training records: {len(X_train)}")
    print(f"Non-fraud-labelled records used for fitting: {len(normal_records)}")
    print(f"Validation records: {len(X_validation)}")

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )

    print("\nTraining Isolation Forest...")

    model.fit(normal_records)

    print("Training completed.")

    # Isolation Forest's score_samples gives lower scores to
    # more anomalous records, so reverse the sign.
    anomaly_scores = -model.score_samples(X_validation)

    # Evaluate ranking performance using the true validation labels.
    pr_auc = average_precision_score(
        y_validation,
        anomaly_scores,
    )

    roc_auc = roc_auc_score(
        y_validation,
        anomaly_scores,
    )

    # Flag the highest-scoring 2.26% of validation records.
    # This uses the training-set positive-label prevalence as a
    # provisional operational budget, not validation labels.
    training_prevalence = float(y_train.mean())

    threshold = float(
        np.quantile(
            anomaly_scores,
            1.0 - training_prevalence,
        )
    )

    predictions = (anomaly_scores >= threshold).astype(int)

    metrics = {
        "model": "Isolation Forest numeric-only baseline",
        "average_precision_pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(
            precision_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "flagged_records": int(predictions.sum()),
        "anomaly_threshold": threshold,
        "flagged_fraction": float(predictions.mean()),
        "threshold_rule": (
            "Flag the highest-scoring records up to approximately "
            "the training-set suspected-fraud prevalence."
        ),
    }

    print("\nValidation metrics:")

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RESULTS_DIR / "isolation_forest_baseline.joblib"
    metrics_path = RESULTS_DIR / "isolation_forest_metrics.json"

    joblib.dump(
        {
            "imputer": imputer,
            "model": model,
            "numeric_features": NUMERIC_FEATURES,
        },
        model_path,
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()