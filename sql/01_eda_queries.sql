-- ============================================================================
-- SQL Script 01: Exploratory Data Analysis (EDA) Queries
-- Performs deep-dive analysis on class imbalance, amount stats, & diurnal patterns
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 1: Overall Fraud vs. Non-Fraud Ratio & Financial Impact
-- Surfaces global fraud rate by count vs. volume
-- ----------------------------------------------------------------------------
SELECT 
    CASE WHEN is_fraud = 1 THEN 'Fraudulent' ELSE 'Legitimate' END AS transaction_status,
    COUNT(*) AS total_transactions,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fact_transactions), 4) AS volume_share_pct,
    ROUND(SUM(amount), 2) AS total_financial_value,
    ROUND(100.0 * SUM(amount) / (SELECT SUM(amount) FROM fact_transactions), 4) AS value_share_pct,
    ROUND(AVG(amount), 2) AS avg_transaction_amount,
    ROUND(MIN(amount), 2) AS min_amount,
    ROUND(MAX(amount), 2) AS max_amount
FROM fact_transactions
GROUP BY is_fraud;


-- ----------------------------------------------------------------------------
-- Query 2: Transaction Amount Distribution & Financial Risk Profile
-- Evaluates skew and variance across legitimate vs fraudulent operations
-- ----------------------------------------------------------------------------
SELECT 
    is_fraud,
    COUNT(*) AS tx_count,
    ROUND(AVG(amount), 2) AS mean_amount,
    ROUND(MIN(amount), 2) AS min_amount,
    ROUND(MAX(amount), 2) AS max_amount,
    ROUND(SUM(amount), 2) AS total_loss,
    -- Amount variance proxy
    ROUND(AVG(amount * amount) - (AVG(amount) * AVG(amount)), 2) AS amount_variance
FROM fact_transactions
GROUP BY is_fraud;


-- ----------------------------------------------------------------------------
-- Query 3: Time-of-Day (Hourly) Fraud Velocity & Anomaly Trends
-- Detects diurnal spikes and nocturnal fraud vulnerability (00:00 - 23:00)
-- ----------------------------------------------------------------------------
SELECT 
    hour_of_day,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_loss_usd,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount ELSE NULL END), 2) AS avg_fraud_ticket_size
FROM fact_transactions
GROUP BY hour_of_day
ORDER BY hour_of_day ASC;


-- ----------------------------------------------------------------------------
-- Query 4: Time Window Segment Analysis (Dayparting)
-- Groups transactions into business segments (Late Night, Morning, Afternoon, Evening)
-- ----------------------------------------------------------------------------
SELECT 
    time_window,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_loss,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS avg_fraud_amount
FROM fact_transactions
GROUP BY time_window
ORDER BY fraud_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- Query 5: Transaction Amount Risk Brackets (Micro-testing vs High-ticket Drains)
-- Analyzes fraud concentration across value tiers
-- ----------------------------------------------------------------------------
SELECT 
    amount_bucket,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_loss
FROM fact_transactions
GROUP BY amount_bucket
ORDER BY fraud_rate_pct DESC;
