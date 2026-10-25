import pandas as pd
from pathlib import Path

# Find the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Locate the primary dataset
DATASET_PATH = (
    PROJECT_DIR
    / "data"
    / "dataco_supply_chain"
    / "DataCoSupplyChainDataset.csv"
)

print("Dataset path:", DATASET_PATH)
print("Dataset exists:", DATASET_PATH.exists())

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}"
    )

# Load the dataset
df = pd.read_csv(DATASET_PATH, encoding="latin-1")

# Display basic information
print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- First Five Rows ---")
print(df.head())

print("\n--- Dataset Information ---")
df.info()

print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# Check the potential target variable
if "Order Status" in df.columns:
    print("\n--- Order Status Distribution ---")
    print(df["Order Status"].value_counts(dropna=False))

    print("\n--- Order Status Percentages ---")
    print(
        df["Order Status"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )
else:
    print("\nWARNING: 'Order Status' column was not found.")


# --------------------------------------------------
# Check order-level structure and target consistency
# --------------------------------------------------

print("\n--- Order-Level Analysis ---")

# Number of unique orders
print("Unique Order Ids:", df["Order Id"].nunique())

# Number of rows per order
items_per_order = df.groupby("Order Id").size()

print("\nRows per order distribution:")
print(items_per_order.describe())

print("\nOrders with multiple rows:", (items_per_order > 1).sum())
print("Orders with only one row:", (items_per_order == 1).sum())

# Check whether an order has more than one status
statuses_per_order = df.groupby("Order Id")["Order Status"].nunique()

print(
    "\nOrders with inconsistent statuses:",
    (statuses_per_order > 1).sum()
)

# Check whether suspected-fraud labels are consistent per order
df["is_suspected_fraud"] = (
    df["Order Status"]
    .astype("string")
    .str.strip()
    .str.upper()
    .eq("SUSPECTED_FRAUD")
    .astype(int)
)

labels_per_order = df.groupby("Order Id")["is_suspected_fraud"].nunique()

print(
    "Orders with inconsistent fraud labels:",
    (labels_per_order > 1).sum()
)

# --------------------------------------------------
# Check target association with categorical features
# --------------------------------------------------

print("\n--- Target Association Checks ---")

categorical_columns = [
    "Type",
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Category Name",
    "Department Name",
    "Order Region",
]

for column in categorical_columns:
    if column not in df.columns:
        continue

    print(f"\n{column}:")

    association = pd.crosstab(
        df[column],
        df["is_suspected_fraud"],
        normalize="index"
    )

    print(association.round(4))


# --------------------------------------------------
# Check suspected-fraud rate by payment type
# --------------------------------------------------

print("\n--- Suspected Fraud Rate by Payment Type ---")

payment_fraud_summary = (
    df.groupby("Type")["is_suspected_fraud"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

payment_fraud_summary["fraud_percentage"] = (
    payment_fraud_summary["mean"] * 100
)

print(
    payment_fraud_summary[
        ["count", "sum", "fraud_percentage"]
    ].round(2)
)