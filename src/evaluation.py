"""
Evaluation & Visualization Suite
Generates publication-quality charts for EDA, Model Benchmarks, and Financial Loss Analysis.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Headless backend for automation & CI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix
)


# Consistent styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_class_imbalance(df: pd.DataFrame, output_path: str = f"{FIGURES_DIR}/01_class_imbalance.png"):
    """Visualize severe class imbalance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    counts = df['Class'].value_counts()
    labels = ['Legitimate (0)', 'Fraudulent (1)']
    colors = ['#2563EB', '#DC2626']

    # Donut chart
    ax1.pie(counts, labels=labels, autopct='%1.2f%%', startangle=90, colors=colors,
            wedgeprops=dict(width=0.4, edgecolor='w'))
    ax1.set_title("Transaction Distribution (Severe Imbalance)", fontsize=13, fontweight='bold', pad=15)

    # Log-scale bar chart
    bars = ax2.bar(labels, counts, color=colors, alpha=0.85, edgecolor='black', width=0.5)
    ax2.set_yscale('log')
    ax2.set_ylabel("Transaction Count (Log Scale)", fontsize=11)
    ax2.set_title("Log-Scale Volume Comparison", fontsize=13, fontweight='bold', pad=15)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval * 1.15, f"{int(yval):,}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_amount_distribution(df: pd.DataFrame, output_path: str = f"{FIGURES_DIR}/02_amount_distribution.png"):
    """Compare transaction amount distribution for fraud vs legit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    legit = df[df['Class'] == 0]['Amount']
    fraud = df[df['Class'] == 1]['Amount']

    # Boxplot with log-scale
    data = [legit, fraud]
    bp = ax1.boxplot(data, tick_labels=['Legitimate', 'Fraudulent'], patch_artist=True, showmeans=True)
    colors = ['#93C5FD', '#FCA5A5']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_yscale('log')
    ax1.set_ylabel("Transaction Amount ($ - Log Scale)", fontsize=11)
    ax1.set_title("Transaction Amount Outliers & Dispersion", fontsize=13, fontweight='bold', pad=15)

    # KDE distribution for amounts < $500
    sns.kdeplot(legit[legit < 500], ax=ax2, label=f'Legit (Mean: ${legit.mean():.2f})', color='#2563EB', fill=True, alpha=0.3)
    sns.kdeplot(fraud[fraud < 500], ax=ax2, label=f'Fraud (Mean: ${fraud.mean():.2f})', color='#DC2626', fill=True, alpha=0.4)
    ax2.set_xlabel("Transaction Amount ($)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Density Distribution (Amounts < $500)", fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_hourly_fraud_trends(df: pd.DataFrame, output_path: str = f"{FIGURES_DIR}/03_hourly_fraud_trends.png"):
    """Visualize 24-hour diurnal cycle and hourly fraud rate."""
    if 'HourOfDay' not in df.columns:
        df['HourOfDay'] = ((df['Time'] // 3600) % 24).astype(int)

    hourly = df.groupby('HourOfDay').agg(
        Total_Tx=('Class', 'count'),
        Fraud_Tx=('Class', 'sum'),
        Fraud_Loss=('Amount', lambda x: x[df.loc[x.index, 'Class'] == 1].sum())
    ).reset_index()

    hourly['Fraud_Rate_Pct'] = (hourly['Fraud_Tx'] / hourly['Total_Tx']) * 100.0

    fig, ax1 = plt.subplots(figsize=(13, 6))

    color_tx = '#64748B'
    color_rate = '#DC2626'

    ax1.set_xlabel('Hour of Day (00:00 - 23:00)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Transaction Volume', color=color_tx, fontsize=12, fontweight='bold')
    bars = ax1.bar(hourly['HourOfDay'], hourly['Total_Tx'], color=color_tx, alpha=0.35, label='Total Volume')
    ax1.tick_params(axis='y', labelcolor=color_tx)
    ax1.set_xticks(range(0, 24))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Fraud Rate (%)', color=color_rate, fontsize=12, fontweight='bold')
    line = ax2.plot(hourly['HourOfDay'], hourly['Fraud_Rate_Pct'], color=color_rate, linewidth=2.8,
                    marker='o', markersize=6, label='Fraud Rate %')
    ax2.tick_params(axis='y', labelcolor=color_rate)

    # Highlight nighttime fraud spike
    ax1.axvspan(1, 5, color='#FEE2E2', alpha=0.5, label='High-Risk Night Window (01:00 - 05:00)')

    plt.title("Diurnal Transaction Volume vs. Fraud Rate % Spike", fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str = f"{FIGURES_DIR}/04_roc_curves.png"
):
    """Plot Receiver Operating Characteristic curves."""
    plt.figure(figsize=(9, 7))
    palette = ['#2563EB', '#16A34A', '#9333EA']

    for idx, (name, clf) in enumerate(models_dict.items()):
        y_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=palette[idx % len(palette)], lw=2.2,
                 label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='#94A3B8', lw=1.5, linestyle='--', label='Random Chance')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_precision_recall_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str = f"{FIGURES_DIR}/05_precision_recall_curves.png"
):
    """Plot Precision-Recall curves (critical for imbalanced classes)."""
    plt.figure(figsize=(9, 7))
    palette = ['#2563EB', '#16A34A', '#9333EA']

    baseline_ap = np.mean(y_test)
    plt.axhline(y=baseline_ap, color='#94A3B8', linestyle='--', label=f'Baseline Chance ({baseline_ap*100:.2f}%)')

    for idx, (name, clf) in enumerate(models_dict.items()):
        y_proba = clf.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, color=palette[idx % len(palette)], lw=2.2,
                 label=f'{name} (PR-AUC = {pr_auc:.4f})')

    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Fraud Detection Rate)', fontsize=11, fontweight='bold')
    plt.ylabel('Precision (True Fraud Accuracy)', fontsize=11, fontweight='bold')
    plt.title('Precision-Recall Curves (Class Imbalance Metric)', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_confusion_matrices(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str = f"{FIGURES_DIR}/06_confusion_matrices.png"
):
    """Plot side-by-side Confusion Matrices."""
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (name, clf) in enumerate(models_dict.items()):
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
                    xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
        axes[idx].set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    output_path: str = f"{FIGURES_DIR}/07_feature_importance.png"
):
    """Plot feature importance ranking."""
    plt.figure(figsize=(10, 6))

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        metric_name = "MDI Feature Importance"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        metric_name = "Absolute Coefficient Magnitude"
    else:
        print("[!] Model does not support direct feature importances.")
        return

    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(12)

    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='crest', edgecolor='black')
    plt.title(f"Top 12 Predictive Features ({metric_name})", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel(metric_name, fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def run_all_evaluations(data_path: str = "data/raw/creditcard.csv"):
    """Execute all exploratory and model visualizations."""
    from src.data_loader import load_dataset, engineer_features
    from src.preprocessing import FraudDataPreprocessor
    from src.models import ModelTrainer

    print("[*] Generating EDA figures...")
    df = load_dataset(data_path)
    df_feat = engineer_features(df)

    plot_class_imbalance(df_feat)
    plot_amount_distribution(df_feat)
    plot_hourly_fraud_trends(df_feat)

    print("[*] Training models to generate benchmark evaluation curves...")
    preprocessor = FraudDataPreprocessor()
    X, y = preprocessor.prepare_features_and_target(df_feat)
    X_train_df, X_test_df, y_train, y_test = preprocessor.split_data(X, y)
    X_train_s, X_test_s = preprocessor.fit_transform(X_train_df, X_test_df)

    trainer = ModelTrainer()
    trainer.train_and_evaluate(X_train_s, y_train.values, X_test_s, y_test.values, quick_run=True)

    plot_roc_curves(trainer.models, X_test_s, y_test.values)
    plot_precision_recall_curves(trainer.models, X_test_s, y_test.values)
    plot_confusion_matrices(trainer.models, X_test_s, y_test.values)

    best_name = max(trainer.results, key=lambda k: trainer.results[k]["PR-AUC"])
    plot_feature_importance(trainer.models[best_name], preprocessor.feature_names)
    print("[OK] All figures generated successfully in reports/figures/")


if __name__ == "__main__":
    run_all_evaluations()
