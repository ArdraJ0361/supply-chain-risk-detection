from pathlib import Path

import json
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------
# 1. Project configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
RESULTS_DIR = ROOT / "results"

TARGET = "is_suspected_fraud"
RANDOM_STATE = 42
THRESHOLD = 0.5

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

# Same encoder settings for all structured and hybrid experiments.
MIN_FREQUENCY = 10


# ---------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------

def load_data(split):
    """Load a processed dataset split and its target labels."""

    path = DATA_DIR / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run src/data_preprocessing.py first."
        )

    df = pd.read_csv(path)

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = set(feature_columns + [TARGET]) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {split} dataset: {sorted(missing)}"
        )

    X = df[feature_columns].copy()
    y = df[TARGET].astype(int).to_numpy()

    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype("string")

    return X, y


def load_embeddings(split, expected_rows):
    """Load precomputed MiniLM embeddings and validate row counts."""

    path = EMBEDDINGS_DIR / f"{split}_embeddings.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {path}. "
            "Run src/text_embeddings.py first."
        )

    embeddings = np.load(path)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding array, got {embeddings.shape}."
        )

    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Row mismatch for {split}: {expected_rows} dataset rows "
            f"but {embeddings.shape[0]} embeddings."
        )

    return embeddings


# ---------------------------------------------------------
# 3. Consistent structured preprocessing
# ---------------------------------------------------------

def create_preprocessor(include_type=True):
    """Build the same preprocessing pipeline for each experiment."""

    numeric_features = NUMERIC_FEATURES.copy()
    categorical_features = CATEGORICAL_FEATURES.copy()

    if not include_type:
        categorical_features.remove("Type")

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
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=MIN_FREQUENCY,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------
# 4. Consistent XGBoost configuration
# ---------------------------------------------------------

def create_model(y_train):
    """Create an XGBoost classifier with training-set class weighting."""

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    if positive_count == 0:
        raise ValueError("Training data contains no positive examples.")

    return XGBClassifier(
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


# ---------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------

def evaluate_model(name, model, X_validation, y_validation, features):
    """Calculate validation metrics at a fixed threshold."""

    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "model": name,
        "PR-AUC": float(
            average_precision_score(y_validation, probabilities)
        ),
        "ROC-AUC": float(
            roc_auc_score(y_validation, probabilities)
        ),
        "Precision": float(
            precision_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "F1-score": float(
            f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "Threshold": THRESHOLD,
        "Features": features,
    }


# ---------------------------------------------------------
# 6. Run one structured or hybrid experiment
# ---------------------------------------------------------

def run_experiment(
    name,
    include_type,
    use_embeddings,
    X_train,
    y_train,
    X_validation,
    y_validation,
    train_embeddings,
    validation_embeddings,
):
    """Train and evaluate one experiment."""

    print(f"\n{'=' * 65}")
    print(f"Experiment: {name}")
    print(f"{'=' * 65}")

    preprocessor = create_preprocessor(include_type=include_type)

    X_train_structured = preprocessor.fit_transform(X_train)
    X_validation_structured = preprocessor.transform(X_validation)

    print("Structured training shape:", X_train_structured.shape)
    print("Structured validation shape:", X_validation_structured.shape)

    if use_embeddings:
        # Fit the scaler on training embeddings only.
        scaler = StandardScaler()

        train_scaled = scaler.fit_transform(train_embeddings)
        validation_scaled = scaler.transform(validation_embeddings)

        X_train_final = hstack(
            [
                csr_matrix(X_train_structured),
                csr_matrix(train_scaled),
            ],
            format="csr",
        )

        X_validation_final = hstack(
            [
                csr_matrix(X_validation_structured),
                csr_matrix(validation_scaled),
            ],
            format="csr",
        )
    else:
        X_train_final = X_train_structured
        X_validation_final = X_validation_structured

    print("Final training shape:", X_train_final.shape)
    print("Final validation shape:", X_validation_final.shape)

    model = create_model(y_train)

    print("Training XGBoost...")
    model.fit(X_train_final, y_train)

    feature_description = (
        "Structured features + MiniLM embeddings"
        if use_embeddings
        else "Structured features"
    )

    if not include_type:
        feature_description += "; Type excluded"

    if use_embeddings and not include_type:
        feature_description = (
            "Structured features without Type + MiniLM embeddings"
        )

    if not use_embeddings and not include_type:
        feature_description = "Structured features without Type"

    result = evaluate_model(
        name,
        model,
        X_validation_final,
        y_validation,
        feature_description,
    )

    print("\nValidation metrics:")
    for metric, value in result.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return result


# ---------------------------------------------------------
# 7. Main
# ---------------------------------------------------------

def main():
    print("Loading training and validation datasets...")

    X_train, y_train = load_data("train")
    X_validation, y_validation = load_data("validation")

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_validation))
    print(f"Training positive rate: {y_train.mean():.2%}")
    print(f"Validation positive rate: {y_validation.mean():.2%}")

    train_embeddings = load_embeddings("train", len(X_train))
    validation_embeddings = load_embeddings(
        "validation",
        len(X_validation),
    )

    print("Training embeddings:", train_embeddings.shape)
    print("Validation embeddings:", validation_embeddings.shape)

    results = []

    # Experiment 1: Structured features, including Type.
    results.append(
        run_experiment(
            name="Structured XGBoost",
            include_type=True,
            use_embeddings=False,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            train_embeddings=train_embeddings,
            validation_embeddings=validation_embeddings,
        )
    )

    # Experiment 2: MiniLM embeddings only.
    # Use the same training rows and class weighting.
    print(f"\n{'=' * 65}")
    print("Experiment: Embeddings-only XGBoost")
    print(f"{'=' * 65}")

    embedding_scaler = StandardScaler()
    train_embeddings_scaled = embedding_scaler.fit_transform(
        train_embeddings
    )
    validation_embeddings_scaled = embedding_scaler.transform(
        validation_embeddings
    )

    embedding_model = create_model(y_train)
    embedding_model.fit(train_embeddings_scaled, y_train)

    results.append(
        evaluate_model(
            "Embeddings-only XGBoost",
            embedding_model,
            validation_embeddings_scaled,
            y_validation,
            "MiniLM text embeddings",
        )
    )

    # Experiment 3: Structured features + embeddings.
    results.append(
        run_experiment(
            name="Hybrid XGBoost",
            include_type=True,
            use_embeddings=True,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            train_embeddings=train_embeddings,
            validation_embeddings=validation_embeddings,
        )
    )

    # Experiment 4: Structured features without Type.
    results.append(
        run_experiment(
            name="Structured without Type",
            include_type=False,
            use_embeddings=False,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            train_embeddings=train_embeddings,
            validation_embeddings=validation_embeddings,
        )
    )

    # Experiment 5: Hybrid features without Type.
    results.append(
        run_experiment(
            name="Hybrid without Type",
            include_type=False,
            use_embeddings=True,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            train_embeddings=train_embeddings,
            validation_embeddings=validation_embeddings,
        )
    )

    results_df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "fair_model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 85}")
    print("FINAL VALIDATION COMPARISON")
    print(f"{'=' * 85}")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved results to: {output_path}")

    # Save experiment settings for reproducibility.
    config = {
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD,
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weighting": "negative_count / positive_count",
        },
        "onehot_min_frequency": MIN_FREQUENCY,
        "evaluation_split": "validation",
    }

    config_path = RESULTS_DIR / "fair_model_comparison_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print(f"Saved experiment configuration to: {config_path}")


if __name__ == "__main__":
    main()