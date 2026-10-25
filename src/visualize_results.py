from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = ROOT / "data" / "processed"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Model comparison chart
# --------------------------------------------------

def plot_model_comparison():
    results_path = RESULTS_DIR / "fair_model_comparison.csv"

    if not results_path.exists():
        print(f"Skipping model comparison: {results_path} not found.")
        return

    df = pd.read_csv(results_path)

    print("\nModel comparison CSV columns:")
    print(df.columns.tolist())

    # Identify the expected columns, allowing for minor naming differences.
    model_col = next(
        (
            col for col in df.columns
            if col.lower() in {"experiment", "model", "model_name"}
        ),
        None,
    )

    pr_auc_col = next(
        (
            col for col in df.columns
            if col.lower() in {
                "pr_auc",
                "pr-auc",
                "average_precision",
                "test_average_precision_pr_auc",
                "validation_pr_auc",
            }
        ),
        None,
    )

    if model_col is None or pr_auc_col is None:
        print("Could not identify model and PR-AUC columns.")
        print("Please share the CSV column names so we can adjust the script.")
        return

    df = df.dropna(subset=[model_col, pr_auc_col]).copy()
    df = df.sort_values(pr_auc_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(df[model_col], df[pr_auc_col])

    ax.set_title("Model Comparison: Precision–Recall AUC")
    ax.set_xlabel("PR-AUC (Average Precision)")
    ax.set_ylabel("Experiment")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, df[pr_auc_col]):
        ax.text(
            value + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max(df[pr_auc_col].max() * 1.2, 0.01))
    fig.tight_layout()

    output_path = FIGURES_DIR / "model_comparison_pr_auc.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# 3. Final test confusion matrix
# --------------------------------------------------

def plot_confusion_matrix():
    metrics_path = RESULTS_DIR / "final_test_metrics.json"

    if not metrics_path.exists():
        print(f"Skipping confusion matrix: {metrics_path} not found.")
        return

    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)

    matrix = np.array(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not suspected fraud", "Suspected fraud"],
    )

    display.plot(
        ax=ax,
        values_format=",d",
        colorbar=False,
    )

    ax.set_title(
        f"Final XGBoost Test Confusion Matrix\n"
        f"Threshold = {metrics['threshold']:.2f}"
    )

    fig.tight_layout()

    output_path = FIGURES_DIR / "final_test_confusion_matrix.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# 4. Final model precision-recall curve
# --------------------------------------------------

def plot_precision_recall_curve():
    model_path = RESULTS_DIR / "final_xgboost_model.joblib"
    test_path = DATA_DIR / "test.csv"
    metrics_path = RESULTS_DIR / "final_test_metrics.json"

    required_paths = [model_path, test_path, metrics_path]

    if not all(path.exists() for path in required_paths):
        print("Skipping precision-recall curve: required files are missing.")
        return

    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    target = "is_suspected_fraud"
    y_test = test_df[target].astype(int)

    # Use the same feature columns recorded in the fitted preprocessor.
    feature_columns = (
        model.named_steps["preprocessor"]
        .feature_names_in_.tolist()
    )

    X_test = test_df[feature_columns].copy()

    categorical_features = (
        model.named_steps["preprocessor"]
        .transformers_[1][2]
    )

    for column in categorical_features:
        X_test[column] = X_test[column].astype("string")

    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(y_test, probabilities)
    prevalence = y_test.mean()
    threshold = metrics["threshold"]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        recall,
        precision,
        label=f"XGBoost (PR-AUC = {pr_auc:.4f})",
    )

    ax.axhline(
        prevalence,
        linestyle="--",
        label=f"Positive-class prevalence = {prevalence:.3f}",
    )

    selected_predictions = (probabilities >= threshold).astype(int)

    selected_precision = (
        ((selected_predictions == 1) & (y_test.to_numpy() == 1)).sum()
        / max(selected_predictions.sum(), 1)
    )

    selected_recall = (
        ((selected_predictions == 1) & (y_test.to_numpy() == 1)).sum()
        / max((y_test.to_numpy() == 1).sum(), 1)
    )

    ax.scatter(
        [selected_recall],
        [selected_precision],
        label=f"Selected threshold = {threshold:.2f}",
        zorder=3,
    )

    ax.set_title("Precision–Recall Curve on the Test Set")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    fig.tight_layout()

    output_path = FIGURES_DIR / "final_test_precision_recall_curve.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# 5. XGBoost feature importance
# --------------------------------------------------

def plot_feature_importance():
    model_path = RESULTS_DIR / "final_xgboost_model.joblib"

    if not model_path.exists():
        print(f"Skipping feature importance: {model_path} not found.")
        return

    model = joblib.load(model_path)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).head(15)

    importance_df = importance_df.sort_values(
        "importance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    ax.set_title("Top 15 XGBoost Feature Importances")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_path = FIGURES_DIR / "xgboost_feature_importance.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save the values for interpretation and reproducibility.
    csv_path = FIGURES_DIR / "xgboost_feature_importance.csv"
    importance_df.sort_values(
        "importance",
        ascending=False,
    ).to_csv(csv_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Saved: {csv_path}")


# --------------------------------------------------
# 6. Run all visualizations
# --------------------------------------------------

def main():
    print("Generating project visualizations...\n")

    plot_model_comparison()
    plot_confusion_matrix()
    plot_precision_recall_curve()
    plot_feature_importance()

    print("\nVisualization script completed.")


if __name__ == "__main__":
    main()