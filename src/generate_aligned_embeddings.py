from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from text_embeddings import MODEL_NAME, TEXT_COLUMNS, BATCH_SIZE, create_texts
from data_preprocessing import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "multiseed" / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = OUTPUT_DIR / "all_rows_embeddings.npy"
SOURCE_ROWS_PATH = OUTPUT_DIR / "source_rows.npy"


def main() -> None:
    print("Loading original dataset...")
    df = load_dataset()

    print(f"Dataset rows: {len(df)}")

    missing_columns = set(TEXT_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing text columns: {sorted(missing_columns)}"
        )

    # Preserve the original row positions explicitly.
    source_rows = np.arange(len(df), dtype=np.int64)

    if EMBEDDINGS_PATH.exists() and SOURCE_ROWS_PATH.exists():
        existing_rows = np.load(SOURCE_ROWS_PATH)

        if (
            len(existing_rows) == len(df)
            and np.array_equal(existing_rows, source_rows)
        ):
            existing_embeddings = np.load(
                EMBEDDINGS_PATH,
                mmap_mode="r",
            )

            if (
                existing_embeddings.ndim == 2
                and existing_embeddings.shape[0] == len(df)
                and existing_embeddings.shape[1] == 384
            ):
                print("Aligned embeddings already exist.")
                print("Shape:", existing_embeddings.shape)
                print("No regeneration needed.")
                return

    print("Creating text from the eight selected fields...")
    texts = create_texts(df)

    print(f"Loading pretrained model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings for the full dataset...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (len(df), 384):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(SOURCE_ROWS_PATH, source_rows)

    print("\nEmbedding generation completed.")
    print("Embedding shape:", embeddings.shape)
    print("Source-row mapping shape:", source_rows.shape)
    print("Embeddings saved to:", EMBEDDINGS_PATH)
    print("Row mapping saved to:", SOURCE_ROWS_PATH)


if __name__ == "__main__":
    main()