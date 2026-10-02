CREATE TABLE IF NOT EXISTS raw_leads (
    id BIGSERIAL PRIMARY KEY,
    lead_id TEXT,
    created_at TIMESTAMP,
    product_type TEXT,
    channel TEXT,
    device TEXT,
    partner TEXT,
    city TEXT,
    insurance_company TEXT,
    payment_type TEXT,
    minutes_since_abandonment INTEGER,
    days_to_policy_expiry INTEGER,
    price NUMERIC,
    discount_percent NUMERIC,
    has_previous_purchase INTEGER,
    visited_offer_page INTEGER,
    incoming_call_last_24h INTEGER,
    sessions_last_7d INTEGER,
    offer_views_last_7d INTEGER,
    price_comparisons_last_7d INTEGER,
    days_since_last_visit NUMERIC,
    expected_margin NUMERIC,
    completed_purchase INTEGER,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lead_scores (
    lead_id TEXT NOT NULL,
    source_row_id BIGINT NOT NULL,
    purchase_probability DOUBLE PRECISION NOT NULL
        CHECK (purchase_probability BETWEEN 0 AND 1),
    priority BIGINT NOT NULL CHECK (priority > 0),
    score_rank BIGINT NOT NULL CHECK (score_rank > 0),
    prediction_timestamp TIMESTAMPTZ NOT NULL,
    model_version TEXT NOT NULL,
    PRIMARY KEY (model_version, lead_id),
    UNIQUE (model_version, priority)
);
