LLM-Powered Supply Chain Risk & Fraud Detection
This project evaluates whether pretrained MiniLM text embeddings add value to suspected-fraud classification on the DataCo SMART Supply Chain Dataset. It compares structured XGBoost, an Isolation Forest anomaly detector, embedding-only XGBoost, and hybrid models, with feature-ablation and control experiments.

Important interpretation: the target is the dataset's SUSPECTED_FRAUD label, not independently verified fraud. All 4,062 positive records have Type = TRANSFER; every model without Type performs close to chance. Results therefore reveal a strong dataset-specific label association and should not be interpreted as proof of generalizable fraud detection.

Repository: https://github.com/ArdraJ0361/supply-chain-risk-detection
Dataset
Download the DataCo SMART Supply Chain Dataset from Kaggle and place the main file at:

data/dataco_supply_chain/DataCoSupplyChainDataset.csv

The dataset contains 180,519 rows and 53 columns, including 4,062 suspected-fraud records (2.25%). The raw dataset is not committed to Git. See data/README.md for details.
Setup
Python 3.11 is recommended.
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Run all commands from the repository root. The dataset must be downloaded before running the pipeline.
Core experiment pipeline
1. Inspect and preprocess
python src/inspect_dataset.py
python src/analyze_features.py
python src/data_preprocessing.py
The data are split by Order Id into training, validation, and test sets (60:20:20), preventing records from the same order appearing in multiple splits.
2. Generate text embeddings
python src/text_embeddings.py
This encodes eight selected fields using sentence-transformers/all-MiniLM-L6-v2. The combined text format is Column: value | Column: value. MiniLM is a sentence encoder, not a generative LLM. Encoding the full dataset involves about 180,000 texts and may take a while on a CPU.
3. Run the primary and supporting models
python src/baseline_models.py
python src/isolation_forest.py
python src/fair_model_comparison.py
python src/final_evaluation.py
python src/tune_threshold.py
python src/visualize_results.py
visualize_results.py uses results/fair_model_comparison.csv and the results/final_xgboost_model.joblib artifact produced by the preceding steps to generate plots under results/figures/. These scripts depend on the processed splits and, where relevant, generated embeddings. Use validation data for threshold selection; the test set is reserved for final evaluation.
4. Multi-seed and control experiments
Run these in order:
python src/multiseed_experiments.py
python src/generate_aligned_embeddings.py
python src/align_multiseed_embeddings.py
python src/multiseed_model_comparison.py
python src/shuffled_embedding_control.py
python src/analyze_type_fraud.py
python src/type_only_baseline.py
These experiments compare split sensitivity across seeds 42, 21, and 7, assess aligned versus shuffled embeddings, and quantify the relationship between Type and the target label. They use the same underlying dataset and are not independent external evaluations.
Legacy scripts
The following scripts are retained for historical/earlier experiments but are superseded by fair_model_comparison.py and are not required for the recommended pipeline:
- src/hybrid_model.py
- src/feature_ablation.py
- src/embedding_model.py
- src/compare_results.py
src/compare_results.py writes the legacy results/model_comparison.csv; this file is not part of the recommended result set.
Results
Multi-seed comparison
Source file: results/multiseed_model_comparison_summary.csv.

Metrics below are mean ± standard deviation over split seeds 42, 21, and 7. Only the data-split seed varies; all runs use the same underlying dataset. F1 is evaluated at threshold 0.5. The TRANSFER-only rule's PR-AUC is calculated from binary predictions and is not directly comparable to score-based PR-AUC.
Model	PR-AUC	ROC-AUC	F1
Structured XGBoost	0.0802 ± 0.0033	0.8681 ± 0.0056	0.151
Hybrid (aligned embeddings)	0.0826 ± 0.0002	0.8696 ± 0.0018	0.148
Hybrid (shuffled embeddings)	0.0806 ± 0.0033	0.8688 ± 0.0048	0.148
TRANSFER-only rule	0.0810 ± 0.0031	0.8684 ± 0.0031	0.150
Embeddings-only	0.0226 ± 0.0010	0.4936 ± 0.0134	0.041
Structured without Type	0.0227 ± 0.0009	0.5006 ± 0.0149	0.040
Hybrid without Type	0.0225 ± 0.0006	0.4965 ± 0.0152	0.041


Takeaway: No model reliably outperforms the TRANSFER-only rule in these experiments. The small numerical differences should not be interpreted as evidence of a robust improvement.

The shuffled-embedding control results are in results/shuffled_embedding_and_type_control_summary.csv.
Payment Type crosstab
Source file: results/type_fraud_crosstab.csv.
Payment type	Rows	Suspected-fraud rows	Suspected-fraud rate
TRANSFER	49,883	4,062	8.14%
CASH	19,616	0	0.00%
DEBIT	69,295	0	0.00%
PAYMENT	41,725	0	0.00%


All 4,062 suspected-fraud labels occur in the TRANSFER category. On the same held-out test split, flagging every TRANSFER order gives precision 0.0805, recall 1.0000, and F1 0.1489, compared with F1 0.1404 for structured XGBoost at threshold 0.70.

The TRANSFER-only rule metrics are also saved in results/type_only_baseline_metrics.csv.
Headline model test results
The Isolation Forest (numeric features, single validation split) achieved PR-AUC 0.0210 and ROC-AUC 0.4800, which is at chance level. Its metrics are recorded in results/isolation_forest_metrics.json.

The structured XGBoost model on the held-out test set at the validation-selected threshold of 0.70 achieved:
Metric	Result
PR-AUC (Average Precision)	0.0765
ROC-AUC	0.8620
Precision	0.0785
Recall	0.6647
F1-score	0.1404


The hybrid’s small PR-AUC advantage (0.0826 vs 0.0802) is within seed variation, and the shuffled-embedding control does not support a reliable embedding benefit. All models without Type perform near chance (PR-AUC approximately 0.022). These results suggest that text embeddings added no measurable value for the short categorical and product text used here.
Limitations
- SUSPECTED_FRAUD is a dataset label, not independently verified fraud.
- All positive labels occur in Type = TRANSFER; feature availability at prediction time must be verified.
- The dataset is class-imbalanced and model precision is low.
- The text fields are limited; results may not generalize to richer text or other supply chains.
- Multi-seed experiments assess split sensitivity on one dataset, not independent external validity.
Repository structure
supply-chain-risk-detection/
├── data/README.md
├── report/assignment_report.md
├── report/assignment_report.pdf
├── results/
│   └── figures/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
The raw dataset, processed data, generated embeddings, virtual environments, and serialized model artifacts (*.joblib) are excluded from Git by .gitignore. The summary and evaluation CSV/JSON result files currently included under results/ are committed snapshots; rerun the documented scripts to regenerate them. Generated plots are written to results/figures/. See report/assignment_report.pdf for the report and report/assignment_report.md for its editable source.