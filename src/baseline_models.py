from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# ---------------------------------------------------------
# 1. Project paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

RANDOM_STATE = 42


# ---------------------------------------------------------
# 2. Define the features
# ---------------------------------------------------------

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

TARGET = "is_suspected_fraud"


# ---------------------------------------------------------
# 3. Load the prepared datasets
# ---------------------------------------------------------

def load_datasets():
    """Load the training and validation datasets."""

    for path in [TRAIN_PATH, VALIDATION_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Run src/data_preprocessing.py first."
            )

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    print("Datasets loaded successfully.")
    print(f"Training shape: {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")

    return train_df, validation_df


# ---------------------------------------------------------
# 4. Prepare features and target
# ---------------------------------------------------------

def prepare_features(train_df, validation_df):
    """Select features and separate the target labels."""

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    missing_train = set(feature_columns + [TARGET]) - set(train_df.columns)
    missing_validation = (
        set(feature_columns + [TARGET]) - set(validation_df.columns)
    )

    if missing_train:
        raise ValueError(
            f"Missing columns in training data: {sorted(missing_train)}"
        )

    if missing_validation:
        raise ValueError(
            f"Missing columns in validation data: "
            f"{sorted(missing_validation)}"
        )

    X_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET].astype(int)

    X_validation = validation_df[feature_columns].copy()
    y_validation = validation_df[TARGET].astype(int)

    # Treat categorical values as strings so that missing values
    # and mixed data types do not cause encoding errors.
    for column in CATEGORICAL_FEATURES:
        X_train[column] = X_train[column].astype("string")
        X_validation[column] = X_validation[column].astype("string")

    print("\nFeatures prepared.")
    print(f"Number of numeric features: {len(NUMERIC_FEATURES)}")
    print(f"Number of categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Training fraud rate: {y_train.mean():.2%}")
    print(f"Validation fraud rate: {y_validation.mean():.2%}")

    return X_train, y_train, X_validation, y_validation


# ---------------------------------------------------------
# 5. Build the preprocessing pipeline
# ---------------------------------------------------------

def build_model():
    """Create preprocessing steps and the XGBoost classifier."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    # Give more weight to suspected-fraud records because they
    # represent only about 2.25% of the dataset.
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


# ---------------------------------------------------------
# 6. Train the model
# ---------------------------------------------------------

def train_model(pipeline, X_train, y_train):
    """Train XGBoost using only the training dataset."""

    print("\nTraining XGBoost model...")

    classifier = pipeline.named_steps["classifier"]

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError("The training dataset has no positive examples.")

    classifier.set_params(
        scale_pos_weight=negative_count / positive_count
    )

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    return pipeline


# ---------------------------------------------------------
# 7. Evaluate on the validation dataset
# ---------------------------------------------------------

def evaluate_model(pipeline, X_validation, y_validation):
    """Evaluate the model using validation data only."""

    print("\nEvaluating model on validation data...")

    probabilities = pipeline.predict_proba(X_validation)[:, 1]

    # Initial classification threshold.
    # We can tune this using validation data in a later step.
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "XGBoost structured baseline",
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
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

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
# 8. Save results and trained model
# ---------------------------------------------------------

def save_results(pipeline, metrics):
    """Save the model and validation metrics."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RESULTS_DIR / "xgboost_baseline.joblib"
    metrics_path = RESULTS_DIR / "baseline_metrics.json"

    joblib.dump(pipeline, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved validation metrics to: {metrics_path}")


# ---------------------------------------------------------
# 9. Main program
# ---------------------------------------------------------

def main():
    train_df, validation_df = load_datasets()

    X_train, y_train, X_validation, y_validation = prepare_features(
        train_df,
        validation_df,
    )

    pipeline = build_model()
    pipeline = train_model(pipeline, X_train, y_train)

    metrics = evaluate_model(
        pipeline,
        X_validation,
        y_validation,
    )

    save_results(pipeline, metrics)


if __name__ == "__main__":
    main()