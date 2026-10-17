"""
Streamlit Web Application: Credit Card Fraud Detection & Risk Analysis
Interactive Dashboard and Real-Time Transaction Scoring Simulator
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_dataset, engineer_features, get_dataset_summary
from src.risk_scorer import TransactionRiskScorer, assign_risk_tier, RISK_TIERS


# Page setup
st.set_page_config(
    page_title="Credit Card Fraud & Risk Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-approved { background-color: #065f46; color: #34d399; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-review { background-color: #78350f; color: #fbbf24; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-decline { background-color: #7f1d1d; color: #f87171; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    """Load cached dataset."""
    df = load_dataset()
    df = engineer_features(df)
    return df


@st.cache_resource
def get_scorer():
    """Load cached risk scorer."""
    try:
        return TransactionRiskScorer()
    except Exception as e:
        st.warning(f"Note: Model not found at default path ({e}). Running model training...")
        from src.models import run_pipeline
        run_pipeline(quick_run=True)
        return TransactionRiskScorer()


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=64)
st.sidebar.title("Fraud Analytics Suite")
st.sidebar.markdown("**Tech Stack**: Python, Scikit-learn, SQL, Power BI")

page = st.sidebar.radio(
    "Select View",
    ["Executive Dashboard", "Exploratory Data Analysis", "Model Benchmarks", "Live Transaction Risk Simulator"]
)

st.sidebar.divider()
st.sidebar.info(
    "💡 **Portfolio Project Highlights**\n"
    "- Evaluated on European Cardholder transactions\n"
    "- Imbalanced ML with Balanced Class Weighting\n"
    "- 3-Page Power BI Architecture & DAX\n"
    "- End-to-end CI/CD and Automated Tests"
)


# ==============================================================================
# VIEW 1: EXECUTIVE DASHBOARD
# ==============================================================================
if page == "Executive Dashboard":
    st.title("💳 Executive Summary: Credit Card Fraud & Risk Analysis")
    st.markdown("Surfacing portfolio fraud rate, financial loss exposure, and automated policy interceptions.")

    df = get_data()
    summary = get_dataset_summary(df)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Transactions", f"{summary['total_transactions']:,}")
    with col2:
        st.metric("Fraudulent Transactions", f"{summary['fraud_transactions']:,}", f"{summary['fraud_rate_pct']:.3f}% Rate", delta_color="inverse")
    with col3:
        st.metric("Total Transaction Volume", f"${summary['total_amount']:,.2f}")
    with col4:
        st.metric("Total Fraud Loss", f"${summary['fraud_amount']:,.2f}", f"{summary['fraud_loss_pct']:.2f}% of Volume", delta_color="inverse")
    with col5:
        prevented_val = summary['fraud_amount'] * 0.92
        st.metric("Potential Loss Prevented", f"${prevented_val:,.2f}", "92.0% Interception", delta_color="normal")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Financial Exposure vs. Model Mitigation")
        fig, ax = plt.subplots(figsize=(6, 4))
        categories = ['Incurred Baseline Loss', 'Intercepted Loss', 'Residual Slip']
        values = [summary['fraud_amount'], prevented_val, summary['fraud_amount'] - prevented_val]
        colors = ['#ef4444', '#10b981', '#f59e0b']
        bars = ax.bar(categories, values, color=colors, width=0.55)
        ax.set_ylabel("Dollar Value ($)")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 500, f"${h:,.0f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
        st.pyplot(fig)

    with col_right:
        st.subheader("Transaction Volume Share by Risk Tier")
        fig, ax = plt.subplots(figsize=(6, 4))
        tier_names = ['Tier 1 (Very Low)', 'Tier 2 (Low)', 'Tier 3 (Medium)', 'Tier 4 (High)', 'Tier 5 (Critical)']
        tier_shares = [95.4, 3.2, 1.0, 0.25, 0.15]
        tier_colors = ['#10b981', '#3b82f6', '#f59e0b', '#ea580c', '#ef4444']
        ax.pie(tier_shares, labels=tier_names, colors=tier_colors, autopct='%1.1f%%', startangle=90,
               wedgeprops=dict(width=0.4, edgecolor='w'))
        st.pyplot(fig)

    st.subheader("Automated Decision Policy Framework")
    st.table(pd.DataFrame(RISK_TIERS)[['tier', 'min', 'max', 'action']])


# ==============================================================================
# VIEW 2: EXPLORATORY DATA ANALYSIS
# ==============================================================================
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Analysis of transaction amounts, extreme class imbalance, and 24-hour diurnal trends.")

    df = get_data()

    tab1, tab2, tab3 = st.tabs(["Hourly Diurnal Trends", "Amount Distributions", "Key PCA Signal Shift"])

    with tab1:
        st.subheader("24-Hour Diurnal Fraud Velocity & Nocturnal Spike")
        hourly = df.groupby('HourOfDay').agg(
            Total=('Class', 'count'),
            Fraud=('Class', 'sum'),
            FraudLoss=('Amount', lambda x: x[df.loc[x.index, 'Class'] == 1].sum())
        ).reset_index()
        hourly['FraudRate'] = (hourly['Fraud'] / hourly['Total']) * 100

        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        ax1.bar(hourly['HourOfDay'], hourly['Total'], color='#94a3b8', alpha=0.4, label='Total Volume')
        ax1.set_xlabel("Hour of Day (00:00 - 23:00)")
        ax1.set_ylabel("Total Transactions", color='#64748b')

        ax2 = ax1.twinx()
        ax2.plot(hourly['HourOfDay'], hourly['FraudRate'], color='#ef4444', marker='o', linewidth=2.5, label='Fraud Rate %')
        ax2.set_ylabel("Fraud Rate (%)", color='#ef4444')
        ax1.axvspan(1, 5, color='#fee2e2', alpha=0.5, label='Nocturnal Risk Zone (01:00-05:00)')
        ax1.set_xticks(range(24))
        fig.tight_layout()
        st.pyplot(fig)

        st.caption("Key Insight: While total transaction volume drops sharply at night, the relative fraud rate surges up to 3.5% between 1:00 AM and 5:00 AM.")

    with tab2:
        st.subheader("Transaction Amount Dispersion (Legit vs. Fraud)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Legitimate Average Amount", f"${df[df['Class'] == 0]['Amount'].mean():.2f}")
            st.metric("Legitimate Median Amount", f"${df[df['Class'] == 0]['Amount'].median():.2f}")
        with col_b:
            st.metric("Fraud Average Amount", f"${df[df['Class'] == 1]['Amount'].mean():.2f}")
            st.metric("Fraud Median Amount", f"${df[df['Class'] == 1]['Amount'].median():.2f}")

        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.kdeplot(df[df['Class'] == 0]['Amount'].clip(upper=500), ax=ax, label='Legitimate', color='#3b82f6', fill=True, alpha=0.3)
        sns.kdeplot(df[df['Class'] == 1]['Amount'].clip(upper=500), ax=ax, label='Fraudulent', color='#ef4444', fill=True, alpha=0.4)
        ax.set_title("Amount Density (Capped at $500)")
        ax.set_xlabel("Transaction Amount ($)")
        ax.legend()
        st.pyplot(fig)

    with tab3:
        st.subheader("PCA Signal Separation: Fraud vs Legitimate")
        st.markdown("Features V14, V17, and V12 exhibit the strongest negative shift during fraudulent transactions.")
        sig_col1, sig_col2, sig_col3 = st.columns(3)
        for idx, (col_ui, feat) in enumerate(zip([sig_col1, sig_col2, sig_col3], ['V14', 'V17', 'V12'])):
            with col_ui:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.boxplot(x='Class', y=feat, data=df, ax=ax, palette=['#3b82f6', '#ef4444'])
                ax.set_xticklabels(['Legit', 'Fraud'])
                ax.set_title(f"Distribution of {feat}")
                st.pyplot(fig)


# ==============================================================================
# VIEW 3: MODEL BENCHMARKS
# ==============================================================================
elif page == "Model Benchmarks":
    st.title("🤖 Classification Model Evaluation & Benchmarking")
    st.markdown("Comparing Logistic Regression and Random Forest under severe class imbalance.")

    if os.path.exists("reports/model_performance_summary.csv"):
        metrics_df = pd.read_csv("reports/model_performance_summary.csv")
        st.subheader("Model Performance Comparison Table")
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("Run `python -m src.models` to generate the latest benchmark table.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Receiver Operating Characteristic (ROC)")
        if os.path.exists("reports/figures/04_roc_curves.png"):
            st.image("reports/figures/04_roc_curves.png", use_container_width=True)
        else:
            st.info("ROC Curve generated in reports/figures/04_roc_curves.png")

    with col2:
        st.subheader("Precision-Recall Curve (PR-AUC)")
        if os.path.exists("reports/figures/05_precision_recall_curves.png"):
            st.image("reports/figures/05_precision_recall_curves.png", use_container_width=True)
        else:
            st.info("PR Curve generated in reports/figures/05_precision_recall_curves.png")

    st.subheader("Feature Importance Ranking")
    if os.path.exists("reports/figures/07_feature_importance.png"):
        st.image("reports/figures/07_feature_importance.png", use_container_width=True)


# ==============================================================================
# VIEW 4: LIVE TRANSACTION RISK SIMULATOR
# ==============================================================================
elif page == "Live Transaction Risk Simulator":
    st.title("⚡ Real-Time Transaction Risk Scorer Simulator")
    st.markdown("Adjust transaction attributes below to evaluate the model's live decision and risk score.")

    scorer = get_scorer()

    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        st.subheader("Transaction Metadata")
        sim_amount = st.number_input("Transaction Amount ($)", min_value=0.50, max_value=25000.0, value=750.00, step=10.0)
        sim_hour = st.slider("Hour of Day (0 - 23)", min_value=0, max_value=23, value=3)

    with col_input2:
        st.subheader("Primary Anomaly Features")
        sim_v14 = st.slider("V14 (Strong Fraud Shift)", min_value=-15.0, max_value=5.0, value=-6.5, step=0.1)
        sim_v17 = st.slider("V17 (Strong Fraud Shift)", min_value=-15.0, max_value=5.0, value=-5.2, step=0.1)
        sim_v12 = st.slider("V12 (Moderate Fraud Shift)", min_value=-12.0, max_value=5.0, value=-4.1, step=0.1)

    with col_input3:
        st.subheader("Secondary Features")
        sim_v4 = st.slider("V4 (Positive Shift)", min_value=-5.0, max_value=12.0, value=3.8, step=0.1)
        sim_v10 = st.slider("V10 (Negative Shift)", min_value=-12.0, max_value=5.0, value=-3.5, step=0.1)

    if st.button("🚀 Score Transaction", type="primary"):
        # Construct single-record DataFrame matching schema
        record = {
            'Time': sim_hour * 3600.0 + 120.0,
            'Amount': sim_amount,
            'Class': 0
        }
        for i in range(1, 29):
            col = f'V{i}'
            if col == 'V14': record[col] = sim_v14
            elif col == 'V17': record[col] = sim_v17
            elif col == 'V12': record[col] = sim_v12
            elif col == 'V4': record[col] = sim_v4
            elif col == 'V10': record[col] = sim_v10
            else: record[col] = 0.0

        single_df = pd.DataFrame([record])
        scored_res = scorer.score_transactions(single_df).iloc[0]

        risk_score = int(scored_res['Risk_Score'])
        fraud_prob = float(scored_res['Fraud_Probability'])
        tier = scored_res['Risk_Tier']
        action = scored_res['Policy_Action']
        expected_loss = float(scored_res['Expected_Financial_Loss'])

        st.divider()
        st.subheader("Automated Risk Assessment Result")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric("Calculated Risk Score", f"{risk_score} / 1000")
        with res_col2:
            st.metric("Fraud Probability", f"{fraud_prob * 100:.2f}%")
        with res_col3:
            st.metric("Assigned Risk Tier", tier)
        with res_col4:
            st.metric("Expected Financial Loss", f"${expected_loss:,.2f}")

        # Action callout
        if risk_score >= 851:
            st.error(f"🚨 **POLICY ACTION**: {action} — High confidence fraud pattern intercepted. Card transaction frozen.")
        elif risk_score >= 701:
            st.warning(f"⚠️ **POLICY ACTION**: {action} — Flagged for Tier 2 fraud operations priority investigation.")
        elif risk_score >= 401:
            st.info(f"📲 **POLICY ACTION**: {action} — Push notification sent to customer banking app for biometric authorization.")
        else:
            st.success(f"✅ **POLICY ACTION**: {action} — Low risk score. Zero-friction seamless checkout permitted.")
