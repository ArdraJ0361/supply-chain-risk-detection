from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TEXT_COLUMNS = [
    "Category Name",
    "Department Name",
    "Product Name",
    "Customer Segment",
    "Market",
    "Order Region",
    "Order Country",
    "Shipping Mode",
]

BATCH_SIZE = 64


# ---------------------------------------------------------
# 2. Convert selected columns into combined text
# ---------------------------------------------------------

def create_texts(df):
    """Combine selected columns into one text string per row."""

    texts = []

    for _, row in df[TEXT_COLUMNS].iterrows():
        parts = []

        for column in TEXT_COLUMNS:
            value = row[column]

            if pd.notna(value):
                parts.append(f"{column}: {str(value).strip()}")

        texts.append(" | ".join(parts))

    return texts


# ---------------------------------------------------------
# 3. Generate and save embeddings
# ---------------------------------------------------------

def generate_embeddings(split_name, model):
    """Generate embeddings for one dataset split."""

    csv_path = DATA_DIR / f"{split_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    print(f"\nLoading {split_name} dataset...")

    df = pd.read_csv(csv_path)

    missing_columns = set(TEXT_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing text columns in {split_name}: "
            f"{sorted(missing_columns)}"
        )

    texts = create_texts(df)

    print(f"Number of records: {len(texts)}")
    print(f"Generating embeddings for {split_name}...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    output_path = EMBEDDINGS_DIR / f"{split_name}_embeddings.npy"

    np.save(output_path, embeddings)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved embeddings to: {output_path}")

    return embeddings.shape


# ---------------------------------------------------------
# 4. Main program
# ---------------------------------------------------------

def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Pretrained model loaded successfully.")

    for split_name in ["train", "validation", "test"]:
        generate_embeddings(split_name, model)

    print("\nAll embeddings generated successfully.")


if __name__ == "__main__":
    main()