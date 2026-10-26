from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from multiseed_model_comparison import (
    MULTISEED_DIR,
    RESULTS_DIR,
    SEEDS,
    TARGET,
    THRESHOLD,
    create_model,
    create_preprocessor,
    load_split,
)


def calculate_metrics(y_true, scores, threshold=THRESHOLD):
    """Calculate classification metrics using a fixed threshold."""

    predictions = (scores >= threshold).astype(int)

    return {
        "PR-AUC": average_precision_score(y_true, scores),
        "ROC-AUC": roc_auc_score(y_true, scores),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "Threshold": threshold,
        "true_positives": int(
            ((predictions == 1) & (y_true == 1)).sum()
        ),
        "false_positives": int(
            ((predictions == 1) & (y_true == 0)).sum()
        ),
        "false_negatives": int(
            ((predictions == 0) & (y_true == 1)).sum()
        ),
        "true_negatives": int(
            ((predictions == 0) & (y_true == 0)).sum()
        ),
    }


def run_hybrid_experiment(
    seed,
    X_train,
    y_train,
    X_test,
    y_test,
    train_embeddings,
    test_embeddings,
    shuffle_embeddings,
):
    """Train a hybrid model with aligned or shuffled embeddings."""

    rng = np.random.default_rng(seed)

    if shuffle_embeddings:
        # Break the correspondence between embeddings and records.
        train_indices = rng.permutation(len(train_embeddings))
        test_indices = rng.permutation(len(test_embeddings))

        train_embeddings = train_embeddings[train_indices]
        test_embeddings = test_embeddings[test_indices]

    # Fit structured preprocessing on training data only.
    preprocessor = create_preprocessor(include_type=True)

    X_train_structured = preprocessor.fit_transform(X_train)
    X_test_structured = preprocessor.transform(X_test)

    # Fit embedding scaling on training embeddings only.
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    X_train_hybrid = hstack(
        [
            csr_matrix(X_train_structured),
            csr_matrix(train_scaled),
        ],
        format="csr",
    )

    X_test_hybrid = hstack(
        [
            csr_matrix(X_test_structured),
            csr_matrix(test_scaled),
        ],
        format="csr",
    )

    model = create_model(y_train)

    print(
        f"Training {'shuffled' if shuffle_embeddings else 'aligned'} "
        f"hybrid model for seed {seed}..."
    )

    model.fit(X_train_hybrid, y_train)

    scores = model.predict_proba(X_test_hybrid)[:, 1]

    result = {
        "seed": seed,
        "model": (
            "Hybrid XGBoost - shuffled embeddings"
            if shuffle_embeddings
            else "Hybrid XGBoost - aligned embeddings"
        ),
        "test_rows": len(y_test),
        "positive_count": int(y_test.sum()),
        "positive_rate": float(y_test.mean()),
    }

    result.update(calculate_metrics(y_test, scores))
    return result


def run_type_only_baseline(seed, X_test, y_test):
    """Flag every TRANSFER row as suspected fraud."""

    # The rule is based only on the payment Type field.
    predictions = (
        X_test["Type"].astype(str).str.upper() == "TRANSFER"
    ).astype(int).to_numpy()

    result = {
        "seed": seed,
        "model": "Type-only rule: TRANSFER",
        "test_rows": len(y_test),
        "positive_count": int(y_test.sum()),
        "positive_rate": float(y_test.mean()),
    }

    result.update(calculate_metrics(y_test, predictions.astype(float)))
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in SEEDS:
        print(f"\n{'=' * 70}")
        print(f"SEED {seed}")
        print(f"{'=' * 70}")

        X_train, y_train, train_embeddings = load_split(seed, "train")
        X_test, y_test, test_embeddings = load_split(seed, "test")

        print("Training rows:", len(y_train))
        print("Test rows:", len(y_test))
        print("Training embeddings:", train_embeddings.shape)
        print("Test embeddings:", test_embeddings.shape)

        # Control 1: normal hybrid model with correctly aligned embeddings.
        aligned_result = run_hybrid_experiment(
            seed=seed,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            shuffle_embeddings=False,
        )
        all_results.append(aligned_result)

        # Control 2: independently shuffle train and test embeddings.
        shuffled_result = run_hybrid_experiment(
            seed=seed,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            shuffle_embeddings=True,
        )
        all_results.append(shuffled_result)

        # Control 3: simple rule without model training.
        type_result = run_type_only_baseline(
            seed=seed,
            X_test=X_test,
            y_test=y_test,
        )
        all_results.append(type_result)

        print("\nResults for this seed:")
        for result in (aligned_result, shuffled_result, type_result):
            print(
                f"{result['model']}: "
                f"PR-AUC={result['PR-AUC']:.4f}, "
                f"ROC-AUC={result['ROC-AUC']:.4f}, "
                f"Precision={result['Precision']:.4f}, "
                f"Recall={result['Recall']:.4f}, "
                f"F1={result['F1-score']:.4f}"
            )

    results_df = pd.DataFrame(all_results)

    results_path = (
        RESULTS_DIR / "shuffled_embedding_and_type_control.csv"
    )
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
        RESULTS_DIR / "shuffled_embedding_and_type_control_summary.csv"
    )
    summary_df.to_csv(summary_path)

    print(f"\n{'=' * 90}")
    print("CONTROL EXPERIMENT RESULTS")
    print(f"{'=' * 90}")

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

    print("\nMean and standard deviation across seeds:")
    print(summary_df)

    print("\nSaved detailed results to:", results_path)
    print("Saved summary to:", summary_path)


if __name__ == "__main__":
    main()