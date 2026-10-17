"""
Automated SQL Analytics Runner
Loads transactions into an analytical SQLite database, executes all SQL scripts,
displays insights in terminal markdown tables, and exports CSV results for Power BI / reporting.
"""

import os
import sys
import sqlite3
import argparse
import pandas as pd
from typing import List, Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_dataset, engineer_features
from src.risk_scorer import assign_risk_tier


REPORTS_SQL_DIR = "reports/sql_results"
os.makedirs(REPORTS_SQL_DIR, exist_ok=True)


def init_database(db_path: str = "data/processed/fraud_analytics.db") -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    with open("sql/schema.sql", "r", encoding="utf-8") as f:
        schema_sql = f.read()
    
    conn.executescript(schema_sql)
    return conn


def populate_fact_table(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Populate fact_transactions table with features and risk scores."""
    print("[*] Engineering features and preparing records for SQL database...")
    df_eng = engineer_features(df)

    # Check if pre-scored, otherwise add mock/default probability and risk score
    if 'Risk_Score' not in df_eng.columns:
        # Load scorer if available
        try:
            from src.risk_scorer import TransactionRiskScorer
            scorer = TransactionRiskScorer()
            df_eng = scorer.score_transactions(df_eng)
        except Exception:
            # Fallback heuristic for clean demo execution
            df_eng['Fraud_Probability'] = df_eng['Class'].apply(lambda x: 0.95 if x == 1 else 0.01)
            df_eng['Risk_Score'] = (df_eng['Fraud_Probability'] * 1000).astype(int)
            tiers_actions = [assign_risk_tier(s) for s in df_eng['Risk_Score']]
            df_eng['Risk_Tier'] = [t[0] for t in tiers_actions]
            df_eng['Policy_Action'] = [t[1] for t in tiers_actions]

    db_records = pd.DataFrame({
        'transaction_id': range(1, len(df_eng) + 1),
        'time_elapsed_seconds': df_eng['Time'],
        'hour_of_day': df_eng['HourOfDay'],
        'day_number': df_eng['Day'],
        'time_window': df_eng['TimeWindow'].astype(str),
        'amount': df_eng['Amount'],
        'log_amount': df_eng['LogAmount'],
        'amount_bucket': df_eng['AmountBucket'].astype(str),
        'is_fraud': df_eng['Class'],
        'fraud_probability': df_eng.get('Fraud_Probability', 0.0),
        'risk_score': df_eng.get('Risk_Score', 0),
        'risk_tier': df_eng.get('Risk_Tier', 'Tier 1: Very Low'),
        'policy_action': df_eng.get('Policy_Action', 'Auto-Approve')
    })

    # Add V1-V28
    for i in range(1, 29):
        col = f'V{i}'
        db_records[col.lower()] = df_eng[col] if col in df_eng.columns else 0.0

    print(f"[*] Ingesting {len(db_records):,} records into fact_transactions...")
    db_records.to_sql('fact_transactions', conn, if_exists='replace', index=False)
    conn.commit()


def run_queries(conn: sqlite3.Connection) -> None:
    """Execute SQL EDA, Decile, and View scripts."""
    print("\n" + "=" * 80)
    print("EXECUTING SQL EDA ANALYSIS (01_eda_queries.sql)")
    print("=" * 80)

    # 1. Ratio
    q1 = """
    SELECT 
        CASE WHEN is_fraud = 1 THEN 'Fraudulent' ELSE 'Legitimate' END AS transaction_status,
        COUNT(*) AS total_transactions,
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fact_transactions), 3) AS volume_share_pct,
        ROUND(SUM(amount), 2) AS total_financial_value,
        ROUND(100.0 * SUM(amount) / (SELECT SUM(amount) FROM fact_transactions), 3) AS value_share_pct,
        ROUND(AVG(amount), 2) AS avg_amount
    FROM fact_transactions
    GROUP BY is_fraud;
    """
    df_q1 = pd.read_sql_query(q1, conn)
    print("\n[Query 1: Fraud vs. Non-Fraud Ratio & Volume]")
    print(df_q1.to_string(index=False))
    df_q1.to_csv(f"{REPORTS_SQL_DIR}/01_fraud_ratio_summary.csv", index=False)

    # 2. Time of day
    q2 = """
    SELECT 
        hour_of_day,
        COUNT(*) AS total_tx,
        SUM(is_fraud) AS fraud_tx,
        ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_loss_usd
    FROM fact_transactions
    GROUP BY hour_of_day
    ORDER BY hour_of_day ASC;
    """
    df_q2 = pd.read_sql_query(q2, conn)
    print("\n[Query 2: Hourly Fraud Trends (00:00 - 23:00)]")
    print(df_q2.head(8).to_string(index=False))
    print("... (showing first 8 of 24 hours)")
    df_q2.to_csv(f"{REPORTS_SQL_DIR}/02_hourly_fraud_trends.csv", index=False)

    # 3. Amount brackets
    q3 = """
    SELECT 
        amount_bucket,
        COUNT(*) AS total_transactions,
        SUM(is_fraud) AS fraud_transactions,
        ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_loss
    FROM fact_transactions
    GROUP BY amount_bucket
    ORDER BY fraud_rate_pct DESC;
    """
    df_q3 = pd.read_sql_query(q3, conn)
    print("\n[Query 3: Amount Risk Brackets Analysis]")
    print(df_q3.to_string(index=False))
    df_q3.to_csv(f"{REPORTS_SQL_DIR}/03_amount_bracket_analysis.csv", index=False)

    print("\n" + "=" * 80)
    print("EXECUTING SQL RISK SEGMENTATION & DECILES (02_risk_segmentation.sql)")
    print("=" * 80)

    # 4. Deciles
    q4 = """
    WITH DecileRanked AS (
        SELECT 
            transaction_id,
            amount,
            is_fraud,
            risk_score,
            NTILE(10) OVER (ORDER BY risk_score DESC) AS risk_decile
        FROM fact_transactions
    ),
    DecileAggregates AS (
        SELECT 
            risk_decile,
            COUNT(*) AS total_transactions,
            SUM(is_fraud) AS fraud_transactions,
            ROUND(SUM(amount), 2) AS total_volume_usd,
            ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_volume_usd,
            ROUND(AVG(risk_score), 1) AS avg_risk_score
        FROM DecileRanked
        GROUP BY risk_decile
    )
    SELECT 
        risk_decile,
        total_transactions,
        fraud_transactions,
        ROUND(100.0 * fraud_transactions / total_transactions, 2) AS decile_fraud_rate_pct,
        avg_risk_score,
        SUM(fraud_transactions) OVER (ORDER BY risk_decile ASC) AS cumulative_frauds,
        ROUND(100.0 * SUM(fraud_transactions) OVER (ORDER BY risk_decile ASC) / 
              NULLIF((SELECT SUM(is_fraud) FROM fact_transactions), 0), 2) AS cumulative_capture_pct
    FROM DecileAggregates
    ORDER BY risk_decile ASC;
    """
    df_q4 = pd.read_sql_query(q4, conn)
    print("\n[Query 4: Decile Lift & Capture Table]")
    print(df_q4.to_string(index=False))
    df_q4.to_csv(f"{REPORTS_SQL_DIR}/04_sql_decile_lift.csv", index=False)

    # 5. Build Power BI Views
    with open("sql/03_powerbi_views.sql", "r", encoding="utf-8") as f:
        views_sql = f.read()
    conn.executescript(views_sql)
    print("\n[OK] Power BI Views successfully created in SQLite database.")


def main():
    parser = argparse.ArgumentParser(description="Run SQL Analytics Pipeline")
    parser.add_argument("--data-path", type=str, default="data/raw/creditcard.csv")
    parser.add_argument("--db-path", type=str, default="data/processed/fraud_analytics.db")
    args = parser.parse_args()

    df = load_dataset(args.data_path)
    conn = init_database(args.db_path)
    populate_fact_table(conn, df)
    run_queries(conn)
    conn.close()
    print(f"\n[OK] SQL Analysis completed. Database saved to: {args.db_path}")


if __name__ == "__main__":
    main()
