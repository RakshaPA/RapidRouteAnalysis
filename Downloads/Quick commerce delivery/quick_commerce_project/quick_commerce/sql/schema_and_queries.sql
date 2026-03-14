-- ============================================================
-- Quick Commerce Delivery Analytics — SQL Schema & Queries
-- Database: SQLite (used via Python sqlite3 / SQLAlchemy)
-- ============================================================

-- ─────────────────────────────────────────────
-- TABLE CREATION
-- ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS orders (
    id                  INTEGER PRIMARY KEY,
    warehouse_block     TEXT    NOT NULL,
    mode_of_shipment    TEXT    NOT NULL,
    cost_of_product     INTEGER NOT NULL,
    prior_purchases     INTEGER NOT NULL,
    product_importance  TEXT    NOT NULL,
    gender              TEXT    NOT NULL,
    discount_offered    INTEGER NOT NULL,
    weight_gms          INTEGER NOT NULL,
    customer_care_calls INTEGER NOT NULL,
    customer_rating     INTEGER NOT NULL,
    reached_on_time     INTEGER NOT NULL   -- 1 = delayed, 0 = on time
);


-- ─────────────────────────────────────────────
-- ANALYTICAL QUERIES
-- ─────────────────────────────────────────────

-- Q1: Overall delay rate
SELECT
    ROUND(100.0 * SUM(reached_on_time) / COUNT(*), 2) AS delay_rate_pct
FROM orders;


-- Q2: Delay rate by warehouse block
SELECT
    warehouse_block,
    COUNT(*)                                              AS total_orders,
    SUM(reached_on_time)                                  AS delayed_orders,
    ROUND(100.0 * SUM(reached_on_time) / COUNT(*), 2)    AS delay_pct
FROM orders
GROUP BY warehouse_block
ORDER BY delay_pct DESC;


-- Q3: Delay rate by shipment mode
SELECT
    mode_of_shipment,
    COUNT(*)                                              AS total_orders,
    SUM(reached_on_time)                                  AS delayed,
    ROUND(100.0 * SUM(reached_on_time) / COUNT(*), 2)    AS delay_pct
FROM orders
GROUP BY mode_of_shipment
ORDER BY delay_pct DESC;


-- Q4: Average discount offered for delayed vs on-time orders
SELECT
    CASE reached_on_time WHEN 1 THEN 'Delayed' ELSE 'On Time' END AS status,
    ROUND(AVG(discount_offered), 2)                                AS avg_discount,
    ROUND(AVG(cost_of_product), 2)                                 AS avg_cost,
    ROUND(AVG(weight_gms), 2)                                      AS avg_weight_gms
FROM orders
GROUP BY reached_on_time;


-- Q5: Customer care calls vs delay
SELECT
    customer_care_calls,
    COUNT(*)                                              AS total,
    ROUND(100.0 * SUM(reached_on_time) / COUNT(*), 2)    AS delay_pct
FROM orders
GROUP BY customer_care_calls
ORDER BY customer_care_calls;


-- Q6: Product importance vs delay rate
SELECT
    product_importance,
    COUNT(*)                                              AS total,
    ROUND(100.0 * SUM(reached_on_time) / COUNT(*), 2)    AS delay_pct
FROM orders
GROUP BY product_importance
ORDER BY delay_pct DESC;


-- Q7: Prior purchases vs delay
SELECT
    prior_purchases,
    COUNT(*)                                              AS total,
    ROUND(100.0 * SUM(reached_on_time) / COUNT(*), 2)    AS delay_pct
FROM orders
GROUP BY prior_purchases
ORDER BY prior_purchases;


-- Q8: Gender-wise delay breakdown
SELECT
    CASE gender WHEN 'M' THEN 'Male' ELSE 'Female' END AS gender,
    COUNT(*)                                             AS total,
    ROUND(100.0 * SUM(reached_on_time) / COUNT(*), 2)   AS delay_pct
FROM orders
GROUP BY gender;


-- Q9: High-value orders (cost > 250) delay rate
SELECT
    CASE WHEN cost_of_product > 250 THEN 'High Value' ELSE 'Standard' END AS order_type,
    COUNT(*)                                                                 AS total,
    ROUND(100.0 * SUM(reached_on_time) / COUNT(*), 2)                       AS delay_pct
FROM orders
GROUP BY order_type;


-- Q10: Weight bucket vs delay
SELECT
    CASE
        WHEN weight_gms < 2000 THEN 'Light (<2kg)'
        WHEN weight_gms < 4000 THEN 'Medium (2-4kg)'
        WHEN weight_gms < 6000 THEN 'Heavy (4-6kg)'
        ELSE 'Very Heavy (>6kg)'
    END                                                  AS weight_bucket,
    COUNT(*)                                             AS total,
    ROUND(100.0 * SUM(reached_on_time) / COUNT(*), 2)   AS delay_pct
FROM orders
GROUP BY weight_bucket
ORDER BY delay_pct DESC;
