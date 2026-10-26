from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataco_supply_chain"
    / "DataCoSupplyChainDataset.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH, encoding="latin-1")

    df["is_suspected_fraud"] = (
        df["Order Status"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("SUSPECTED_FRAUD")
        .astype(int)
    )

    # Reproduce the existing order-level 60/20/20 split.
    # This ensures the baseline uses the same test orders.
    from data_preprocessing import (
        prepare_dataset,
        split_by_order,
    )

    original_order_ids = df["Order Id"].copy()
    prepared_df = prepare_dataset(df)

    _, _, test_df = split_by_order(
        prepared_df,
        original_order_ids,
    )

    # Recover payment type and target for the exact test rows.
    test_df = test_df.copy()
    test_df["Type"] = df.loc[test_df.index, "Type"]
    y_test = test_df["is_suspected_fraud"].to_numpy()

    # Simple rule: all TRANSFER rows are predicted positive.
    y_pred = (
        test_df["Type"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("TRANSFER")
        .astype(int)
        .to_numpy()
    )

    # Use the binary rule as a score for ranking metrics.
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    pr_auc = average_precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print("TRANSFER-only baseline")
    print("----------------------")
    print("Test rows:", len(test_df))
    print("Test suspected-fraud rows:", int(y_test.sum()))
    print("Rows flagged:", int(y_pred.sum()))
    print(f"PR-AUC (average precision): {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Non-fraud", "Suspected fraud"],
            zero_division=0,
        )
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        [
            {
                "model": "TRANSFER-only rule",
                "PR-AUC": pr_auc,
                "ROC-AUC": roc_auc,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "threshold": "binary rule",
                "test_rows": len(test_df),
                "flagged_rows": int(y_pred.sum()),
            }
        ]
    )

    output_path = RESULTS_DIR / "type_only_baseline_metrics.csv"
    results.to_csv(output_path, index=False)

    print("\nSaved results to:", output_path)


if __name__ == "__main__":
    main()