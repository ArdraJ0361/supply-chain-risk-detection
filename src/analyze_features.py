from pathlib import Path

import pandas as pd

# Locate the project directory and dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataco_supply_chain"
    / "DataCoSupplyChainDataset.csv"
)

df = pd.read_csv(DATASET_PATH, encoding="latin-1")

# Create a binary target:
# 1 = suspected fraud
# 0 = other order statuses
df["is_suspected_fraud"] = (
    df["Order Status"]
    .astype("string")
    .str.strip()
    .str.upper()
    .eq("SUSPECTED_FRAUD")
    .astype(int)
)

print("\n" + "=" * 70)
print("1. TARGET DISTRIBUTION")
print("=" * 70)

print(df["is_suspected_fraud"].value_counts())
print("\nPercentage:")
print(
    df["is_suspected_fraud"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n" + "=" * 70)
print("2. PAYMENT TYPE VS SUSPECTED FRAUD")
print("=" * 70)

print(
    pd.crosstab(
        df["Type"],
        df["is_suspected_fraud"],
        normalize="index",
    ).round(4)
)

print("\nFraud-label rate by payment type:")
print(
    df.groupby("Type")["is_suspected_fraud"]
    .agg(["count", "mean"])
    .sort_values("mean", ascending=False)
    .round(4)
)

print("\n" + "=" * 70)
print("3. ORDER STATUS VS TARGET")
print("=" * 70)

print(
    pd.crosstab(
        df["Order Status"],
        df["is_suspected_fraud"],
    )
)

print(
    "\nWARNING: Order Status defines the target and must not "
    "be used as a model feature."
)

print("\n" + "=" * 70)
print("4. AVAILABLE TEXT COLUMNS")
print("=" * 70)

text_columns = [
    "Category Name",
    "Department Name",
    "Product Name",
    "Customer Segment",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
    "Shipping Mode",
    "Type",
]

for column in text_columns:
    if column in df.columns:
        print(f"\n{column}")
        print("Unique values:", df[column].nunique())
        print("Examples:")
        print(df[column].dropna().unique()[:10])

print("\n" + "=" * 70)
print("5. POTENTIAL LEAKAGE AND IDENTIFIER COLUMNS")
print("=" * 70)

columns_to_review = [
    "Order Status",
    "Delivery Status",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Profit Ratio",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Customer Email",
    "Customer Password",
    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",
    "Product Description",
]

for column in columns_to_review:
    if column in df.columns:
        print(
            f"{column}: "
            f"{df[column].nunique(dropna=True)} unique values; "
            f"{df[column].isna().sum()} missing"
        )

print("\nFeature investigation complete.")