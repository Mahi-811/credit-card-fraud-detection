-- ============================================================================
-- SQL Script 03: Analytical Views for Power BI Semantic Model
-- Clean, aggregated tabular views designed for Power BI Import / DirectQuery
-- ============================================================================

-- ----------------------------------------------------------------------------
-- View 1: Executive KPI Summary View (Page 1)
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_executive_kpis;

CREATE VIEW vw_executive_kpis AS
SELECT 
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS total_fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS global_fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_transaction_volume,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_financial_loss,
    ROUND(AVG(amount), 2) AS avg_ticket_size,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount ELSE NULL END), 2) AS avg_fraud_ticket_size,
    -- Estimated savings assuming Tier 4 & Tier 5 transactions are intercepted
    ROUND(SUM(CASE WHEN risk_score >= 700 AND is_fraud = 1 THEN amount ELSE 0 END), 2) AS potential_loss_prevented,
    ROUND(100.0 * SUM(CASE WHEN risk_score >= 700 AND is_fraud = 1 THEN amount ELSE 0 END) / 
          NULLIF(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 0), 2) AS fraud_prevention_rate_pct
FROM fact_transactions;


-- ----------------------------------------------------------------------------
-- View 2: Hourly Fraud Trends & Diurnal Vulnerability (Page 2)
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_hourly_fraud_trends;

CREATE VIEW vw_hourly_fraud_trends AS
SELECT 
    hour_of_day,
    day_number,
    time_window,
    COUNT(*) AS transaction_count,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount
FROM fact_transactions
GROUP BY hour_of_day, day_number, time_window;


-- ----------------------------------------------------------------------------
-- View 3: Risk Bracket & Policy Intervention Matrix (Page 3)
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_risk_bracket_analysis;

CREATE VIEW vw_risk_bracket_analysis AS
SELECT 
    risk_tier,
    policy_action,
    amount_bucket,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_volume_usd,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_loss_usd,
    ROUND(AVG(risk_score), 1) AS mean_risk_score
FROM fact_transactions
GROUP BY risk_tier, policy_action, amount_bucket;
