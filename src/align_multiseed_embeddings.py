from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MULTISEED_DIR = PROJECT_ROOT / "data" / "multiseed"
EMBEDDINGS_DIR = MULTISEED_DIR / "embeddings"

ALL_EMBEDDINGS_PATH = (
    EMBEDDINGS_DIR / "all_rows_embeddings.npy"
)
SOURCE_ROWS_PATH = EMBEDDINGS_DIR / "source_rows.npy"

SEEDS = [42, 21, 7]
SPLITS = ["train", "validation", "test"]


def main() -> None:
    print("Loading full-dataset embeddings...")

    all_embeddings = np.load(
        ALL_EMBEDDINGS_PATH,
        mmap_mode="r",
    )
    source_rows = np.load(SOURCE_ROWS_PATH)

    if all_embeddings.shape != (len(source_rows), 384):
        raise ValueError(
            "Embedding and source-row dimensions do not match: "
            f"{all_embeddings.shape}, {source_rows.shape}"
        )

    if len(np.unique(source_rows)) != len(source_rows):
        raise ValueError("Source-row mapping contains duplicates.")

    # Source-row values identify positions in the original dataset.
    embedding_by_source_row = {
        int(source_row): position
        for position, source_row in enumerate(source_rows)
    }

    print("Full embedding shape:", all_embeddings.shape)
    print("Source-row mapping validated.")

    for seed in SEEDS:
        seed_dir = MULTISEED_DIR / f"seed_{seed}"
        output_dir = seed_dir / "embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Seed {seed}")
        print(f"{'=' * 60}")

        for split_name in SPLITS:
            raw_path = seed_dir / f"{split_name}_raw.csv"
            csv_path = seed_dir / f"{split_name}.csv"

            raw_df = pd.read_csv(raw_path)
            clean_df = pd.read_csv(csv_path)

            source_indices = raw_df["_source_row"].to_numpy(
                dtype=np.int64
            )

            # Confirm that the raw and cleaned files have identical
            # row counts and preserve their row ordering.
            if len(raw_df) != len(clean_df):
                raise ValueError(
                    f"Row count mismatch for seed {seed}, "
                    f"split {split_name}"
                )

            positions = np.array(
                [
                    embedding_by_source_row[int(index)]
                    for index in source_indices
                ],
                dtype=np.int64,
            )

            aligned_embeddings = np.asarray(
                all_embeddings[positions],
                dtype=np.float32,
            )

            if aligned_embeddings.shape != (len(clean_df), 384):
                raise ValueError(
                    f"Unexpected shape for seed {seed}, "
                    f"split {split_name}: "
                    f"{aligned_embeddings.shape}"
                )

            output_path = (
                output_dir / f"{split_name}_embeddings.npy"
            )
            np.save(output_path, aligned_embeddings)

            print(
                f"{split_name}: "
                f"rows={len(clean_df)}, "
                f"embedding_shape={aligned_embeddings.shape}"
            )
            print("Saved:", output_path)

    print("\nAll multi-seed embeddings aligned successfully.")


if __name__ == "__main__":
    main()