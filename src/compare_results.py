from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def load_json(filename):
    path = RESULTS_DIR / filename

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    results = []

    # Structured baseline
    baseline = load_json("baseline_metrics.json")

    results.append({
        "model": "Structured XGBoost",
        "PR-AUC": baseline.get("average_precision_pr_auc"),
        "ROC-AUC": baseline.get("roc_auc"),
        "Precision": baseline.get("precision"),
        "Recall": baseline.get("recall"),
        "F1-score": baseline.get("f1_score"),
        "Threshold": baseline.get("classification_threshold"),
        "Features": "Structured features, including Type",
    })

    # Embeddings-only model
    embedding = load_json("embedding_metrics.json")

    results.append({
        "model": "Embeddings-only XGBoost",
        "PR-AUC": embedding.get("average_precision_pr_auc"),
        "ROC-AUC": embedding.get("roc_auc"),
        "Precision": embedding.get("precision"),
        "Recall": embedding.get("recall"),
        "F1-score": embedding.get("f1_score"),
        "Threshold": embedding.get("classification_threshold"),
        "Features": "MiniLM text embeddings",
    })

    # Hybrid model
    hybrid = load_json("hybrid_metrics.json")

    results.append({
        "model": "Hybrid XGBoost",
        "PR-AUC": hybrid.get("average_precision_pr_auc"),
        "ROC-AUC": hybrid.get("roc_auc"),
        "Precision": hybrid.get("precision"),
        "Recall": hybrid.get("recall"),
        "F1-score": hybrid.get("f1_score"),
        "Threshold": hybrid.get("classification_threshold"),
        "Features": "Structured features + MiniLM embeddings",
    })

    # Isolation Forest
    isolation = load_json("isolation_forest_metrics.json")

    results.append({
        "model": "Isolation Forest",
        "PR-AUC": isolation.get("average_precision_pr_auc"),
        "ROC-AUC": isolation.get("roc_auc"),
        "Precision": isolation.get("precision"),
        "Recall": isolation.get("recall"),
        "F1-score": isolation.get("f1_score"),
        "Threshold": isolation.get("anomaly_threshold"),
        "Features": "Numerical features; anomaly scores",
    })

    # Models excluding Type
    ablation_path = RESULTS_DIR / "type_ablation_metrics.csv"
    ablation = pd.read_csv(ablation_path)

    for _, row in ablation.iterrows():
        results.append({
            "model": row["model"],
            "PR-AUC": row["pr_auc"],
            "ROC-AUC": row["roc_auc"],
            "Precision": None,
            "Recall": None,
            "F1-score": None,
            "Threshold": None,
            "Features": "Structured features with Type excluded"
            if row["model"].startswith("Structured")
            else "Structured features + embeddings, with Type excluded",
        })

    results_df = pd.DataFrame(results)

    output_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))

    print(f"\nSaved comparison to: {output_path}")


if __name__ == "__main__":
    main()