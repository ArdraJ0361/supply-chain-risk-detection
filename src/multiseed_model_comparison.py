from pathlib import Path

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
# 1. Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MULTISEED_DIR = ROOT / "data" / "multiseed"
RESULTS_DIR = ROOT / "results"

SEEDS = [42, 21, 7]
SPLITS = ["train", "validation", "test"]

TARGET = "is_suspected_fraud"
MODEL_RANDOM_STATE = 42
THRESHOLD = 0.5
MIN_FREQUENCY = 10

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
# 2. Load and validate datasets
# ---------------------------------------------------------

def load_split(seed, split_name):
    """Load a cleaned split and its aligned embeddings."""

    seed_dir = MULTISEED_DIR / f"seed_{seed}"

    csv_path = seed_dir / f"{split_name}.csv"
    embedding_path = (
        seed_dir
        / "embeddings"
        / f"{split_name}_embeddings.npy"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {embedding_path}"
        )

    df = pd.read_csv(csv_path)

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    required_columns = feature_columns + [TARGET]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in seed {seed}, "
            f"split {split_name}: {sorted(missing)}"
        )

    X = df[feature_columns].copy()
    y = df[TARGET].astype(int).to_numpy()

    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype("string")

    embeddings = np.load(
        embedding_path,
        mmap_mode="r",
    )

    if embeddings.shape != (len(df), 384):
        raise ValueError(
            f"Embedding mismatch for seed {seed}, "
            f"split {split_name}: "
            f"dataset rows={len(df)}, "
            f"embedding shape={embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"Non-finite embedding values found for "
            f"seed {seed}, split {split_name}"
        )

    return X, y, embeddings


# ---------------------------------------------------------
# 3. Structured preprocessing
# ---------------------------------------------------------

def create_preprocessor(include_type=True):
    """Create the structured feature preprocessing pipeline."""

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
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------
# 4. XGBoost model
# ---------------------------------------------------------

def create_model(y_train):
    """Create XGBoost using training-set class weighting."""

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    if positive_count == 0:
        raise ValueError(
            "Training split contains no suspected-fraud examples."
        )

    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=negative_count / positive_count,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
    )


# ---------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test,
    seed,
    test_rows,
    test_embeddings,
    include_type,
    use_embeddings,
):
    """Evaluate model performance on the test split."""

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    result = {
        "seed": seed,
        "model": model_name,
        "test_rows": test_rows,
        "positive_count": int(y_test.sum()),
        "positive_rate": float(y_test.mean()),
        "PR-AUC": float(
            average_precision_score(y_test, probabilities)
        ),
        "ROC-AUC": float(
            roc_auc_score(y_test, probabilities)
        ),
        "Precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "F1-score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "Threshold": THRESHOLD,
        "Type_included": include_type if model_name != "Embeddings-only XGBoost" else None,
        "Embeddings_used": use_embeddings,
        "true_positives": int(
            ((predictions == 1) & (y_test == 1)).sum()
        ),
        "false_positives": int(
            ((predictions == 1) & (y_test == 0)).sum()
        ),
        "false_negatives": int(
            ((predictions == 0) & (y_test == 1)).sum()
        ),
        "true_negatives": int(
            ((predictions == 0) & (y_test == 0)).sum()
        ),
    }

    return result


# ---------------------------------------------------------
# 6. Run one experiment
# ---------------------------------------------------------

def run_experiment(
    seed,
    model_name,
    include_type,
    use_embeddings,
    X_train,
    y_train,
    X_test,
    y_test,
    train_embeddings,
    test_embeddings,
):
    """Train one model and evaluate it on the held-out test split."""

    print(f"\n{'=' * 70}")
    print(f"Seed: {seed} | Experiment: {model_name}")
    print(f"{'=' * 70}")

    if use_embeddings:
        # Fit the scaler on training embeddings only.
        embedding_scaler = StandardScaler()

        train_scaled = embedding_scaler.fit_transform(
            train_embeddings
        )
        test_scaled = embedding_scaler.transform(
            test_embeddings
        )

    if model_name == "Embeddings-only XGBoost":
        X_train_final = csr_matrix(train_scaled)
        X_test_final = csr_matrix(test_scaled)

    else:
        # Fit preprocessing only on training structured features.
        preprocessor = create_preprocessor(
            include_type=include_type
        )

        X_train_structured = preprocessor.fit_transform(
            X_train
        )
        X_test_structured = preprocessor.transform(
            X_test
        )

        if use_embeddings:
            X_train_final = hstack(
                [
                    csr_matrix(X_train_structured),
                    csr_matrix(train_scaled),
                ],
                format="csr",
            )

            X_test_final = hstack(
                [
                    csr_matrix(X_test_structured),
                    csr_matrix(test_scaled),
                ],
                format="csr",
            )
        else:
            X_train_final = X_train_structured
            X_test_final = X_test_structured

    print("Training matrix shape:", X_train_final.shape)
    print("Test matrix shape:", X_test_final.shape)

    model = create_model(y_train)

    print("Training XGBoost...")
    model.fit(X_train_final, y_train)

    result = evaluate_model(
        model_name=model_name,
        model=model,
        X_test=X_test_final,
        y_test=y_test,
        seed=seed,
        test_rows=len(y_test),
        test_embeddings=test_embeddings,
        include_type=include_type,
        use_embeddings=use_embeddings,
    )

    print("\nTest metrics:")
    for metric, value in result.items():
        if isinstance(value, (float, np.floating)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return result


# ---------------------------------------------------------
# 7. Main experiment loop
# ---------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in SEEDS:
        print(f"\n\n{'#' * 80}")
        print(f"LOADING DATA FOR SEED {seed}")
        print(f"{'#' * 80}")

        X_train, y_train, train_embeddings = load_split(
            seed,
            "train",
        )

        X_validation, y_validation, validation_embeddings = (
            load_split(seed, "validation")
        )

        X_test, y_test, test_embeddings = load_split(
            seed,
            "test",
        )

        print("Training rows:", len(y_train))
        print("Validation rows:", len(y_validation))
        print("Test rows:", len(y_test))
        print(f"Training suspected-fraud rate: {y_train.mean():.2%}")
        print(f"Validation suspected-fraud rate: {y_validation.mean():.2%}")
        print(f"Test suspected-fraud rate: {y_test.mean():.2%}")

        print(
            "Training embeddings:",
            train_embeddings.shape,
        )
        print(
            "Validation embeddings:",
            validation_embeddings.shape,
        )
        print(
            "Test embeddings:",
            test_embeddings.shape,
        )

        # Validation data is loaded and checked for completeness,
        # but is not used to train or select the test threshold.
        del X_validation, y_validation, validation_embeddings

        experiments = [
            {
                "name": "Structured XGBoost",
                "include_type": True,
                "use_embeddings": False,
            },
            {
                "name": "Embeddings-only XGBoost",
                "include_type": False,
                "use_embeddings": True,
            },
            {
                "name": "Hybrid XGBoost",
                "include_type": True,
                "use_embeddings": True,
            },
            {
                "name": "Structured without Type",
                "include_type": False,
                "use_embeddings": False,
            },
            {
                "name": "Hybrid without Type",
                "include_type": False,
                "use_embeddings": True,
            },
        ]

        seed_results = []

        for experiment in experiments:
            result = run_experiment(
                seed=seed,
                model_name=experiment["name"],
                include_type=experiment["include_type"],
                use_embeddings=experiment["use_embeddings"],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_embeddings=train_embeddings,
                test_embeddings=test_embeddings,
            )

            seed_results.append(result)
            all_results.append(result)

        seed_results_df = pd.DataFrame(seed_results)

        seed_output_path = (
            RESULTS_DIR
            / f"multiseed_results_seed_{seed}.csv"
        )
        seed_results_df.to_csv(seed_output_path, index=False)

        print(f"\nSaved seed {seed} results to: {seed_output_path}")

        # Release arrays before loading the next seed.
        del (
            X_train,
            y_train,
            X_test,
            y_test,
            train_embeddings,
            test_embeddings,
        )

    results_df = pd.DataFrame(all_results)

    results_path = RESULTS_DIR / "multiseed_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    metric_columns = [
        "PR-AUC",
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1-score",
    ]

    summary_df = (
        results_df.groupby("model")[metric_columns]
        .agg(["mean", "std"])
        .round(4)
    )

    summary_path = (
        RESULTS_DIR / "multiseed_model_comparison_summary.csv"
    )
    summary_df.to_csv(summary_path)

    print(f"\n{'=' * 100}")
    print("MULTI-SEED TEST RESULTS")
    print(f"{'=' * 100}")
    print(
        results_df[
            [
                "seed",
                "model",
                "PR-AUC",
                "ROC-AUC",
                "Precision",
                "Recall",
                "F1-score",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\n{'=' * 100}")
    print("MEAN AND STANDARD DEVIATION ACROSS SEEDS")
    print(f"{'=' * 100}")
    print(summary_df)

    print("\nSaved detailed results to:", results_path)
    print("Saved summary to:", summary_path)


if __name__ == "__main__":
    main()