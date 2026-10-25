from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
RESULTS_DIR = ROOT / "results"

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

# Intentionally exclude "Type" for this sensitivity experiment.
CATEGORICAL_FEATURES = [
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


def load_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    y = df[TARGET].astype(int).to_numpy()

    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[columns].copy()

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string")

    return X, y


def create_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def create_model(y_train):
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=negatives / positives,
        random_state=42,
        n_jobs=4,
        tree_method="hist",
    )


def evaluate(name, model, X_validation, y_validation):
    probabilities = model.predict_proba(X_validation)[:, 1]

    return {
        "model": name,
        "pr_auc": float(
            average_precision_score(y_validation, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_validation, probabilities)
        ),
    }


def main():
    print("Loading training and validation data...")

    X_train, y_train = load_split("train")
    X_validation, y_validation = load_split("validation")

    print(f"Training shape without Type: {X_train.shape}")
    print(f"Validation shape without Type: {X_validation.shape}")

    print("\nPreparing structured features...")

    preprocessor = create_preprocessor()

    X_train_structured = preprocessor.fit_transform(X_train)
    X_validation_structured = preprocessor.transform(X_validation)

    print("Structured features:", X_train_structured.shape)

    results = []

    # Experiment A: structured features without Type.
    print("\nTraining structured model without Type...")

    structured_model = create_model(y_train)
    structured_model.fit(X_train_structured, y_train)

    results.append(
        evaluate(
            "Structured without Type",
            structured_model,
            X_validation_structured,
            y_validation,
        )
    )

    # Experiment B: structured features plus MiniLM embeddings,
    # also without Type.
    print("\nLoading saved embeddings...")

    X_train_embeddings = np.load(
        EMBEDDINGS_DIR / "train_embeddings.npy"
    )
    X_validation_embeddings = np.load(
        EMBEDDINGS_DIR / "validation_embeddings.npy"
    )

    embedding_scaler = StandardScaler()

    X_train_embeddings = embedding_scaler.fit_transform(
        X_train_embeddings
    )
    X_validation_embeddings = embedding_scaler.transform(
        X_validation_embeddings
    )

    X_train_hybrid = hstack([
        csr_matrix(X_train_structured),
        csr_matrix(X_train_embeddings),
    ], format="csr")

    X_validation_hybrid = hstack([
        csr_matrix(X_validation_structured),
        csr_matrix(X_validation_embeddings),
    ], format="csr")

    print("Hybrid features:", X_train_hybrid.shape)

    print("\nTraining hybrid model without Type...")

    hybrid_model = create_model(y_train)
    hybrid_model.fit(X_train_hybrid, y_train)

    results.append(
        evaluate(
            "Hybrid without Type",
            hybrid_model,
            X_validation_hybrid,
            y_validation,
        )
    )

    results_df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "type_ablation_metrics.csv"
    results_df.to_csv(output_path, index=False)

    print("\nValidation results without Type:")
    print(results_df.round(4).to_string(index=False))

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()