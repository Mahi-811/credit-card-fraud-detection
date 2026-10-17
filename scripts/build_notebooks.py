"""
Notebook Builder Script
Creates clean, pre-populated Jupyter Notebooks in the notebooks/ folder.
"""

import os
import nbformat as nbf


def build_all_notebooks():
    os.makedirs('notebooks', exist_ok=True)

    # Notebook 1: EDA
    nb1 = nbf.v4.new_notebook()
    nb1.cells = [
        nbf.v4.new_markdown_cell('# Exploratory Data Analysis: Credit Card Fraud Detection\nThis notebook analyzes transaction patterns, class imbalance, amount distributions, and diurnal fraud trends.'),
        nbf.v4.new_code_cell('''import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_dataset, engineer_features, get_dataset_summary'''),
        nbf.v4.new_markdown_cell('## 1. Load Dataset & Summary Metrics'),
        nbf.v4.new_code_cell('''df = load_dataset('../data/raw/creditcard.csv')
summary = get_dataset_summary(df)
for k, v in summary.items():
    print(f"{k}: {v}")'''),
        nbf.v4.new_markdown_cell('## 2. Feature Engineering (Temporal & Amount Buckets)'),
        nbf.v4.new_code_cell('''df_feat = engineer_features(df)
df_feat[['Time', 'HourOfDay', 'Day', 'TimeWindow', 'Amount', 'AmountBucket', 'Class']].head()'''),
        nbf.v4.new_markdown_cell('## 3. Class Imbalance Analysis'),
        nbf.v4.new_code_cell('''from src.evaluation import plot_class_imbalance
plot_class_imbalance(df_feat, '../reports/figures/01_class_imbalance.png')
print('Class imbalance visualized.')'''),
        nbf.v4.new_markdown_cell('## 4. Hourly Fraud Rate Spike (Diurnal Vulnerability)'),
        nbf.v4.new_code_cell('''from src.evaluation import plot_hourly_fraud_trends
plot_hourly_fraud_trends(df_feat, '../reports/figures/03_hourly_fraud_trends.png')''')
    ]
    with open('notebooks/01_exploratory_data_analysis.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb1, f)
    print("[OK] Created notebooks/01_exploratory_data_analysis.ipynb")

    # Notebook 2: Modeling
    nb2 = nbf.v4.new_notebook()
    nb2.cells = [
        nbf.v4.new_markdown_cell('# Model Training & Evaluation: Handling Severe Imbalance\nComparing Logistic Regression and Random Forest using Precision, Recall, F1, ROC-AUC, and PR-AUC.'),
        nbf.v4.new_code_cell('''import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import numpy as np
from src.data_loader import load_dataset, engineer_features
from src.preprocessing import FraudDataPreprocessor
from src.models import ModelTrainer
from src.evaluation import plot_roc_curves, plot_precision_recall_curves, plot_confusion_matrices'''),
        nbf.v4.new_markdown_cell('## 1. Preprocessing & Robust Scaling'),
        nbf.v4.new_code_cell('''df = load_dataset('../data/raw/creditcard.csv')
df_feat = engineer_features(df)
preprocessor = FraudDataPreprocessor()
X, y = preprocessor.prepare_features_and_target(df_feat)
X_train_df, X_test_df, y_train, y_test = preprocessor.split_data(X, y)
X_train_s, X_test_s = preprocessor.fit_transform(X_train_df, X_test_df)
print(f"Train shape: {X_train_s.shape}, Test shape: {X_test_s.shape}")'''),
        nbf.v4.new_markdown_cell('## 2. Train Classifiers with Balanced Weights'),
        nbf.v4.new_code_cell('''trainer = ModelTrainer()
results_df = trainer.train_and_evaluate(X_train_s, y_train.values, X_test_s, y_test.values)
results_df'''),
        nbf.v4.new_markdown_cell('## 3. Benchmark Visualizations'),
        nbf.v4.new_code_cell('''plot_roc_curves(trainer.models, X_test_s, y_test.values, '../reports/figures/04_roc_curves.png')
plot_precision_recall_curves(trainer.models, X_test_s, y_test.values, '../reports/figures/05_precision_recall_curves.png')
plot_confusion_matrices(trainer.models, X_test_s, y_test.values, '../reports/figures/06_confusion_matrices.png')''')
    ]
    with open('notebooks/02_model_training_evaluation.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb2, f)
    print("[OK] Created notebooks/02_model_training_evaluation.ipynb")

    # Notebook 3: Risk Scoring
    nb3 = nbf.v4.new_notebook()
    nb3.cells = [
        nbf.v4.new_markdown_cell('# Transaction Risk Scoring & Policy Decision Engine\nCalibrating model outputs into 0-1000 Risk Scores, Decile Lift analysis, and automated interventions.'),
        nbf.v4.new_code_cell('''import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
from src.data_loader import load_dataset
from src.risk_scorer import TransactionRiskScorer'''),
        nbf.v4.new_markdown_cell('## 1. Load Scorer & Compute Risk Tiers'),
        nbf.v4.new_code_cell('''df = load_dataset('../data/raw/creditcard.csv')
scorer = TransactionRiskScorer(
    model_path='../models/saved_models/best_fraud_model.joblib',
    preprocessor_path='../models/saved_models/preprocessor.joblib'
)
df_scored = scorer.score_transactions(df)
df_scored[['Time', 'Amount', 'Class', 'Fraud_Probability', 'Risk_Score', 'Risk_Tier', 'Policy_Action']].head(10)'''),
        nbf.v4.new_markdown_cell('## 2. Decile Lift Table'),
        nbf.v4.new_code_cell('''decile_df = scorer.calculate_decile_lift(df_scored)
decile_df''')
    ]
    with open('notebooks/03_risk_scoring_and_insights.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb3, f)
    print("[OK] Created notebooks/03_risk_scoring_and_insights.ipynb")


if __name__ == "__main__":
    build_all_notebooks()
