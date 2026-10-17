-- ============================================================================
-- Credit Card Transactions & Fraud Analytics Schema
-- Compatible with SQLite, PostgreSQL, DuckDB, and SQL Server
-- ============================================================================

DROP TABLE IF EXISTS fact_transactions;

CREATE TABLE fact_transactions (
    transaction_id INTEGER PRIMARY KEY,
    time_elapsed_seconds REAL NOT NULL,
    hour_of_day INTEGER NOT NULL,
    day_number INTEGER NOT NULL,
    time_window VARCHAR(30) NOT NULL,
    amount REAL NOT NULL,
    log_amount REAL NOT NULL,
    amount_bucket VARCHAR(30) NOT NULL,
    v1 REAL, v2 REAL, v3 REAL, v4 REAL, v5 REAL,
    v6 REAL, v7 REAL, v8 REAL, v9 REAL, v10 REAL,
    v11 REAL, v12 REAL, v13 REAL, v14 REAL, v15 REAL,
    v16 REAL, v17 REAL, v18 REAL, v19 REAL, v20 REAL,
    v21 REAL, v22 REAL, v23 REAL, v24 REAL, v25 REAL,
    v26 REAL, v27 REAL, v28 REAL,
    is_fraud INTEGER NOT NULL DEFAULT 0,
    fraud_probability REAL,
    risk_score INTEGER,
    risk_tier VARCHAR(40),
    policy_action VARCHAR(40)
);

-- Analytical indexes to accelerate EDA and BI queries
CREATE INDEX idx_tx_class ON fact_transactions(is_fraud);
CREATE INDEX idx_tx_hour ON fact_transactions(hour_of_day);
CREATE INDEX idx_tx_amount ON fact_transactions(amount);
CREATE INDEX idx_tx_risk_tier ON fact_transactions(risk_tier);
CREATE INDEX idx_tx_time_window ON fact_transactions(time_window);
