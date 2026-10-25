from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataco_supply_chain"
    / "DataCoSupplyChainDataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """Load the original DataCo dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH, encoding="latin-1")


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create the target and remove clearly unsuitable columns."""
    df = df.copy()

    # Binary target: 1 = suspected fraud, 0 = other statuses.
    df["is_suspected_fraud"] = (
        df["Order Status"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("SUSPECTED_FRAUD")
        .astype(int)
    )

    # Remove columns that directly reveal the target,
    # contain post-outcome information, or are unsuitable
    # for our initial prediction scenario.
    columns_to_drop = [
        "Order Status",
        "Delivery Status",
        "Days for shipping (real)",
        "Late_delivery_risk",
        "shipping date (DateOrders)",
        "Order Id",
        "Order Item Id",
        "Customer Email",
        "Customer Password",
        "Customer Fname",
        "Customer Lname",
        "Customer Street",
        "Product Description",
        "Product Image",
    ]

    existing_columns = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    df = df.drop(columns=existing_columns)

    return df


def split_by_order(
    df: pd.DataFrame,
    order_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split item rows by unique order IDs.

    All rows from one order remain in the same split.
    """
    order_labels = (
        df.assign(_order_id=order_ids.to_numpy())
        .groupby("_order_id")["is_suspected_fraud"]
        .first()
    )

    train_orders, temp_orders = train_test_split(
        order_labels.index.to_numpy(),
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=order_labels.to_numpy(),
    )

    temp_labels = order_labels.loc[temp_orders]

    val_orders, test_orders = train_test_split(
        temp_orders,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_labels.to_numpy(),
    )

    # Keep original item rows, assigning entire orders to splits.
    order_values = order_ids.to_numpy()

    train_df = df.loc[
        pd.Series(order_values, index=df.index).isin(train_orders)
    ].copy()

    val_df = df.loc[
        pd.Series(order_values, index=df.index).isin(val_orders)
    ].copy()

    test_df = df.loc[
        pd.Series(order_values, index=df.index).isin(test_orders)
    ].copy()

    return train_df, val_df, test_df


def report_split(
    name: str,
    split_df: pd.DataFrame,
    original_order_ids: pd.Series,
) -> None:
    """Print split size, class distribution, and order count."""
    order_ids = original_order_ids.loc[split_df.index]

    print(f"\n--- {name} ---")
    print("Rows:", len(split_df))
    print("Unique orders:", order_ids.nunique())

    print("Target counts:")
    print(split_df["is_suspected_fraud"].value_counts().sort_index())

    print("Target percentages:")
    print(
        split_df["is_suspected_fraud"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )


def main() -> None:
    original_df = load_dataset()

    # Retain order IDs separately for grouping and leakage checks.
    original_order_ids = original_df["Order Id"].copy()

    df = prepare_dataset(original_df)

    train_df, val_df, test_df = split_by_order(
        df,
        original_order_ids,
    )

    # Verify that no order appears in multiple splits.
    train_orders = set(
        original_order_ids.loc[train_df.index]
    )
    val_orders = set(
        original_order_ids.loc[val_df.index]
    )
    test_orders = set(
        original_order_ids.loc[test_df.index]
    )

    assert train_orders.isdisjoint(val_orders)
    assert train_orders.isdisjoint(test_orders)
    assert val_orders.isdisjoint(test_orders)

    print("No order leakage detected across splits.")

    report_split("TRAIN", train_df, original_order_ids)
    report_split("VALIDATION", val_df, original_order_ids)
    report_split("TEST", test_df, original_order_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nPrepared datasets saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()