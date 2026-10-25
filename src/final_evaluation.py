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

from baseline_models import (
    load_datasets,
    prepare_features,
    build_model,
    train_model,
)


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

TARGET = "is_suspected_fraud"

# Candidate thresholds selected using validation data only.
THRESHOLDS = np.arange(0.05, 0.96, 0.05)


# ---------------------------------------------------------
# 2. Load test data
# ---------------------------------------------------------

def load_test_data():
    """Load the test split without using it for model selection."""

    test_path = DATA_DIR / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}. "
            "Run src/data_preprocessing.py first."
        )

    test_df = pd.read_csv(test_path)

    if TARGET not in test_df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    return test_df


# ---------------------------------------------------------
# 3. Select threshold using validation data
# ---------------------------------------------------------

def select_threshold(y_validation, validation_probabilities):
    """Select the threshold with the highest validation F1-score."""

    threshold_results = []

    for threshold in THRESHOLDS:
        predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        threshold_results.append(
            {
                "threshold": float(threshold),
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
            }
        )

    threshold_df = pd.DataFrame(threshold_results)

    # Select the best F1. If tied, prefer higher precision.
    threshold_df = threshold_df.sort_values(
        by=["f1_score", "precision"],
        ascending=[False, False],
    )

    best_result = threshold_df.iloc[0]
    best_threshold = float(best_result["threshold"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    threshold_path = RESULTS_DIR / "final_threshold_analysis.csv"
    threshold_df.sort_values("threshold").to_csv(
        threshold_path,
        index=False,
    )

    print("\nThreshold selection (validation set only)")
    print(threshold_df.sort_values(
        "f1_score", ascending=False
    ).head(10).round(4).to_string(index=False))

    print(f"\nSelected threshold: {best_threshold:.2f}")
    print(f"Validation F1-score: {best_result['f1_score']:.4f}")
    print(f"Saved threshold analysis: {threshold_path}")

    return best_threshold


# ---------------------------------------------------------
# 4. Evaluate on the test set
# ---------------------------------------------------------

def evaluate_test_model(
    pipeline,
    X_test,
    y_test,
    test_probabilities,
    threshold,
):
    """Evaluate the final model on the test set."""

    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    metrics = {
        "model": "Structured XGBoost",
        "threshold": float(threshold),
        "test_average_precision_pr_auc": float(
            average_precision_score(y_test, test_probabilities)
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, test_probabilities)
        ),
        "test_precision": float(
            precision_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
        "test_recall": float(
            recall_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
        "test_f1_score": float(
            f1_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
        "test_rows": int(len(y_test)),
        "test_positive_count": int(np.sum(y_test == 1)),
    }

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Not suspected fraud", "Suspected fraud"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)

    metrics["confusion_matrix"] = matrix.tolist()

    metrics_path = RESULTS_DIR / "final_test_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nSaved test metrics: {metrics_path}")

    return metrics


# ---------------------------------------------------------
# 5. Main workflow
# ---------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load training and validation data.
    train_df, validation_df = load_datasets()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = prepare_features(train_df, validation_df)

    # Train the selected model on training data only.
    pipeline = build_model()
    pipeline = train_model(pipeline, X_train, y_train)

    # Select the threshold using validation data only.
    validation_probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    best_threshold = select_threshold(
        y_validation.to_numpy(),
        validation_probabilities,
    )

    # The test set is accessed only after the threshold is selected.
    print("\nLoading the test set for final evaluation...")
    test_df = load_test_data()

    feature_columns = (
        pipeline.named_steps["preprocessor"].feature_names_in_.tolist()
    )

    missing_features = set(feature_columns) - set(test_df.columns)

    if missing_features:
        raise ValueError(
            f"Missing test features: {sorted(missing_features)}"
        )

    X_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET].astype(int).to_numpy()

    categorical_features = (
        pipeline.named_steps["preprocessor"]
        .transformers_[1][2]
    )

    for column in categorical_features:
        X_test[column] = X_test[column].astype("string")

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Evaluate only after model training and threshold selection.
    evaluate_test_model(
        pipeline,
        X_test,
        y_test,
        test_probabilities,
        best_threshold,
    )

    # Save the final trained pipeline for reproducibility.
    model_path = RESULTS_DIR / "final_xgboost_model.joblib"
    joblib.dump(pipeline, model_path)

    print(f"Saved final model: {model_path}")


if __name__ == "__main__":
    main()