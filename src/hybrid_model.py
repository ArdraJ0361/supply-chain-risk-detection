from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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


# Use the same structured features as the initial baseline.
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

CATEGORICAL_FEATURES = [
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


# ---------------------------------------------------------
# 2. Load the datasets and embeddings
# ---------------------------------------------------------

def load_data():
    print("Loading datasets and embeddings...")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    validation_df = pd.read_csv(DATA_DIR / "validation.csv")

    X_train_embeddings = np.load(
        EMBEDDINGS_DIR / "train_embeddings.npy"
    )

    X_validation_embeddings = np.load(
        EMBEDDINGS_DIR / "validation_embeddings.npy"
    )

    y_train = train_df[TARGET].astype(int).to_numpy()
    y_validation = validation_df[TARGET].astype(int).to_numpy()

    if len(train_df) != len(X_train_embeddings):
        raise ValueError(
            "Training rows do not match the number of embeddings."
        )

    if len(validation_df) != len(X_validation_embeddings):
        raise ValueError(
            "Validation rows do not match the number of embeddings."
        )

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    missing_train = set(feature_columns) - set(train_df.columns)
    missing_validation = set(feature_columns) - set(validation_df.columns)

    if missing_train or missing_validation:
        raise ValueError(
            f"Missing training columns: {sorted(missing_train)}; "
            f"missing validation columns: {sorted(missing_validation)}"
        )

    X_train_structured = train_df[feature_columns].copy()
    X_validation_structured = validation_df[feature_columns].copy()

    for column in CATEGORICAL_FEATURES:
        X_train_structured[column] = (
            X_train_structured[column].astype("string")
        )
        X_validation_structured[column] = (
            X_validation_structured[column].astype("string")
        )

    print(f"Training structured data: {X_train_structured.shape}")
    print(f"Validation structured data: {X_validation_structured.shape}")
    print(f"Training embeddings: {X_train_embeddings.shape}")
    print(f"Validation embeddings: {X_validation_embeddings.shape}")

    return (
        X_train_structured,
        X_validation_structured,
        X_train_embeddings,
        X_validation_embeddings,
        y_train,
        y_validation,
    )


# ---------------------------------------------------------
# 3. Preprocess structured features
# ---------------------------------------------------------

def prepare_structured_features(
    X_train_structured,
    X_validation_structured,
):
    print("\nPreprocessing structured features...")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
    )

    X_train_processed = preprocessor.fit_transform(
        X_train_structured
    )

    X_validation_processed = preprocessor.transform(
        X_validation_structured
    )

    print(
        "Training structured features after encoding:",
        X_train_processed.shape,
    )

    print(
        "Validation structured features after encoding:",
        X_validation_processed.shape,
    )

    return preprocessor, X_train_processed, X_validation_processed


# ---------------------------------------------------------
# 4. Combine structured features and embeddings
# ---------------------------------------------------------

def combine_features(
    X_train_structured,
    X_validation_structured,
    X_train_embeddings,
    X_validation_embeddings,
):
    X_train_structured = csr_matrix(X_train_structured)
    X_validation_structured = csr_matrix(X_validation_structured)

    # Scale embedding dimensions using training data only.
    scaler = StandardScaler()

    X_train_embeddings_scaled = scaler.fit_transform(
        X_train_embeddings
    )

    X_validation_embeddings_scaled = scaler.transform(
        X_validation_embeddings
    )

    X_train_embeddings_sparse = csr_matrix(
        X_train_embeddings_scaled
    )

    X_validation_embeddings_sparse = csr_matrix(
        X_validation_embeddings_scaled
    )

    X_train_hybrid = hstack(
        [
            X_train_structured,
            X_train_embeddings_sparse,
        ],
        format="csr",
    )

    X_validation_hybrid = hstack(
        [
            X_validation_structured,
            X_validation_embeddings_sparse,
        ],
        format="csr",
    )

    print("\nCombined feature shapes:")
    print("Training:", X_train_hybrid.shape)
    print("Validation:", X_validation_hybrid.shape)

    return scaler, X_train_hybrid, X_validation_hybrid


# ---------------------------------------------------------
# 5. Train the hybrid XGBoost model
# ---------------------------------------------------------

def train_model(X_train, y_train):
    print("\nTraining hybrid XGBoost model...")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

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
# 6. Evaluate the hybrid model
# ---------------------------------------------------------

def evaluate_model(model, X_validation, y_validation):
    print("\nEvaluating on validation data...")

    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "Hybrid XGBoost: structured features + MiniLM embeddings",
        "average_precision_pr_auc": float(
            average_precision_score(
                y_validation,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_validation,
                probabilities,
            )
        ),
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
            target_names=[
                "Not suspected fraud",
                "Suspected fraud",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_validation, predictions))

    return metrics


# ---------------------------------------------------------
# 7. Save model, preprocessing, and metrics
# ---------------------------------------------------------

def save_results(
    model,
    preprocessor,
    embedding_scaler,
    metrics,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RESULTS_DIR / "hybrid_model.joblib"
    metrics_path = RESULTS_DIR / "hybrid_metrics.json"

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "embedding_scaler": embedding_scaler,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        model_path,
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


# ---------------------------------------------------------
# 8. Main program
# ---------------------------------------------------------

def main():
    (
        X_train_structured,
        X_validation_structured,
        X_train_embeddings,
        X_validation_embeddings,
        y_train,
        y_validation,
    ) = load_data()

    (
        preprocessor,
        X_train_structured_processed,
        X_validation_structured_processed,
    ) = prepare_structured_features(
        X_train_structured,
        X_validation_structured,
    )

    (
        embedding_scaler,
        X_train_hybrid,
        X_validation_hybrid,
    ) = combine_features(
        X_train_structured_processed,
        X_validation_structured_processed,
        X_train_embeddings,
        X_validation_embeddings,
    )

    model = train_model(
        X_train_hybrid,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_validation_hybrid,
        y_validation,
    )

    save_results(
        model,
        preprocessor,
        embedding_scaler,
        metrics,
    )


if __name__ == "__main__":
    main()