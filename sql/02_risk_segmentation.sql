-- ============================================================================
-- SQL Script 02: Transaction Risk Segmentation & Decile Analysis
-- Advanced Analytical SQL using Window Functions (NTILE, SUM OVER)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 1: Risk Decile Analysis (Gain & Lift Table)
-- Divides transactions into 10 risk buckets based on Risk Score
-- ----------------------------------------------------------------------------
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
    total_volume_usd,
    fraud_volume_usd,
    avg_risk_score,
    SUM(fraud_transactions) OVER (ORDER BY risk_decile ASC) AS cumulative_frauds_captured,
    ROUND(100.0 * SUM(fraud_transactions) OVER (ORDER BY risk_decile ASC) / 
          NULLIF((SELECT SUM(is_fraud) FROM fact_transactions), 0), 2) AS cumulative_fraud_capture_pct
FROM DecileAggregates
ORDER BY risk_decile ASC;


-- ----------------------------------------------------------------------------
-- Query 2: Risk Tier Policy Distribution & Intervention Matrix
-- Evaluates automated decision outcomes (Auto-Approve, 2FA, Manual Review, Decline)
-- ----------------------------------------------------------------------------
SELECT 
    risk_tier,
    policy_action,
    COUNT(*) AS transaction_volume,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fact_transactions), 2) AS volume_share_pct,
    SUM(is_fraud) AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / NULLIF(COUNT(*), 0), 2) AS tier_fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_ticket_value,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_loss_exposed,
    ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) / 
          NULLIF((SELECT SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) FROM fact_transactions), 0), 2) AS pct_of_total_fraud_loss
FROM fact_transactions
GROUP BY risk_tier, policy_action
ORDER BY MIN(risk_score) DESC;


-- ----------------------------------------------------------------------------
-- Query 3: Investigator Work Queue (Top 20 Critical Anomaly Flags)
-- Prioritized queue for fraud operations teams
-- ----------------------------------------------------------------------------
SELECT 
    transaction_id,
    hour_of_day,
    amount,
    amount_bucket,
    risk_score,
    risk_tier,
    policy_action,
    ROUND(fraud_probability, 4) AS fraud_probability,
    ROUND(amount * fraud_probability, 2) AS expected_loss_usd,
    ROUND(v14, 2) AS v14_signal,
    ROUND(v17, 2) AS v17_signal,
    ROUND(v12, 2) AS v12_signal
FROM fact_transactions
WHERE risk_score >= 700
ORDER BY risk_score DESC, amount DESC
LIMIT 20;
