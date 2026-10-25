from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
MODEL_PATH = PROJECT_ROOT / "results" / "xgboost_baseline.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    print("Loading validation dataset and trained model...")

    validation_df = pd.read_csv(VALIDATION_PATH)
    model = joblib.load(MODEL_PATH)

    target = "is_suspected_fraud"
    y_validation = validation_df[target].astype(int)

    numeric_features = [
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

    categorical_features = [
        "Type",
        "Category Name",
        "Department Name",
        "Product Name",
        "Customer Country",
        "Customer Segment",
        "Market",
        "Order Country",
        "Order Region",
        "Shipping Mode",
    ]

    feature_columns = numeric_features + categorical_features

    X_validation = validation_df[feature_columns].copy()

    for column in categorical_features:
        X_validation[column] = X_validation[column].astype("string")

    probabilities = model.predict_proba(X_validation)[:, 1]

    pr_auc = average_precision_score(y_validation, probabilities)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_validation, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_validation, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_validation, predictions, zero_division=0
                ),
                "flagged_records": int(predictions.sum()),
            }
        )

    results = pd.DataFrame(rows)

    print(f"\nPR-AUC (independent of classification threshold): {pr_auc:.4f}")
    print("\nThreshold comparison:")
    print(results.round(4).to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "threshold_analysis.csv"
    results.to_csv(output_path, index=False)

    metrics_path = RESULTS_DIR / "baseline_metrics.json"

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        metrics["validation_pr_auc"] = float(pr_auc)
        metrics["threshold_analysis_file"] = "threshold_analysis.csv"

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

    print(f"\nSaved threshold comparison to: {output_path}")


if __name__ == "__main__":
    main()