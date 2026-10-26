from pathlib import Path

import pandas as pd


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

    # Load the original dataset before preprocessing.
    df = pd.read_csv(DATASET_PATH, encoding="latin-1")

    # Define the target consistently with the existing pipeline.
    df["is_suspected_fraud"] = (
        df["Order Status"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("SUSPECTED_FRAUD")
        .astype(int)
    )

    # Count fraud and non-fraud records for each payment type.
    counts = pd.crosstab(
        df["Type"],
        df["is_suspected_fraud"],
    )

    counts = counts.reindex(columns=[0, 1], fill_value=0)
    counts.columns = ["non_fraud_count", "suspected_fraud_count"]

    # Calculate fraud rates and class percentages.
    counts["total_count"] = (
        counts["non_fraud_count"]
        + counts["suspected_fraud_count"]
    )

    counts["suspected_fraud_rate_pct"] = (
        counts["suspected_fraud_count"]
        / counts["total_count"]
        * 100
    ).round(4)

    total_fraud = counts["suspected_fraud_count"].sum()

    counts["percentage_of_all_suspected_fraud_pct"] = (
        counts["suspected_fraud_count"]
        / total_fraud
        * 100
    ).round(4)

    counts = counts.sort_values(
        "suspected_fraud_count",
        ascending=False,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "type_fraud_crosstab.csv"
    counts.to_csv(output_path, index=True, index_label="payment_type")

    print("Dataset rows:", len(df))
    print("Unique orders:", df["Order Id"].nunique())
    print("Total suspected-fraud rows:", total_fraud)
    print(
        "Overall suspected-fraud rate:",
        f"{df['is_suspected_fraud'].mean() * 100:.4f}%",
    )

    print("\nPayment type versus suspected-fraud results:")
    print(counts.to_string())

    print("\nSaved results to:", output_path)


if __name__ == "__main__":
    main()