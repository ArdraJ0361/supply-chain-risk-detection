from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessing import (
    load_dataset,
    prepare_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "multiseed"

SEEDS = [42, 21, 7]


def create_order_level_splits(
    df: pd.DataFrame,
    order_ids: pd.Series,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Create stratified 60/20/20 splits without separating order rows."""

    order_labels = (
        df.assign(_order_id=order_ids.to_numpy())
        .groupby("_order_id")["is_suspected_fraud"]
        .first()
    )

    train_orders, temporary_orders = train_test_split(
        order_labels.index.to_numpy(),
        test_size=0.40,
        random_state=seed,
        stratify=order_labels.to_numpy(),
    )

    temporary_labels = order_labels.loc[temporary_orders]

    validation_orders, test_orders = train_test_split(
        temporary_orders,
        test_size=0.50,
        random_state=seed,
        stratify=temporary_labels.to_numpy(),
    )

    order_values = order_ids.to_numpy()

    splits = {
        "train": df.loc[
            pd.Series(order_values, index=df.index).isin(train_orders)
        ].copy(),
        "validation": df.loc[
            pd.Series(order_values, index=df.index).isin(validation_orders)
        ].copy(),
        "test": df.loc[
            pd.Series(order_values, index=df.index).isin(test_orders)
        ].copy(),
    }

    order_sets = {
        name: set(order_ids.loc[split_df.index])
        for name, split_df in splits.items()
    }

    assert order_sets["train"].isdisjoint(order_sets["validation"])
    assert order_sets["train"].isdisjoint(order_sets["test"])
    assert order_sets["validation"].isdisjoint(order_sets["test"])

    assert sum(len(split_df) for split_df in splits.values()) == len(df)

    return splits


def main() -> None:
    original_df = load_dataset()

    # Keep original row indices and order IDs for alignment and validation.
    original_df.index.name = "_source_row"
    order_ids = original_df["Order Id"].copy()

    # Apply the same target creation and column exclusions as the baseline.
    prepared_df = prepare_dataset(original_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print("Dataset rows:", len(prepared_df))
    print("Unique orders:", order_ids.nunique())
    print(
        "Overall suspected-fraud rate:",
        f"{prepared_df['is_suspected_fraud'].mean():.2%}",
    )

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"Seed: {seed}")
        print(f"{'=' * 60}")

        splits = create_order_level_splits(
            prepared_df,
            order_ids,
            seed,
        )

        seed_dir = OUTPUT_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in splits.items():
            # Raw split retains the source order ID for auditing.
            raw_path = seed_dir / f"{split_name}_raw.csv"
            raw_df = split_df.copy()
            raw_df.insert(0, "_source_row", split_df.index)
            raw_df.to_csv(raw_path, index=False)

            # Model-ready split excludes order IDs, post-outcome
            # fields, and other columns removed by prepare_dataset().
            cleaned_path = seed_dir / f"{split_name}.csv"
            cleaned_df = split_df.drop(columns=["Order Id"], errors="ignore")
            cleaned_df.to_csv(cleaned_path, index=False)

            positive_count = int(
                split_df["is_suspected_fraud"].sum()
            )
            fraud_rate = split_df["is_suspected_fraud"].mean()

            print(f"\n{split_name.upper()}")
            print("Rows:", len(split_df))
            print(
                "Unique orders:",
                order_ids.loc[split_df.index].nunique(),
            )
            print("Suspected-fraud rows:", positive_count)
            print("Suspected-fraud rate:", f"{fraud_rate:.2%}")
            print("Saved cleaned split:", cleaned_path)

            summary_rows.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "rows": len(split_df),
                    "unique_orders": order_ids.loc[
                        split_df.index
                    ].nunique(),
                    "suspected_fraud_rows": positive_count,
                    "suspected_fraud_rate": fraud_rate,
                }
            )

        print("\nOrder-level split validation passed.")

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "split_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nAll splits generated successfully.")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()