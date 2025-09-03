CREATE EXTENSION IF NOT EXISTS vector;

CREATE INDEX IF NOT EXISTS idx_review_embeddings_hnsw ON review_embeddings USING hnsw (content_vec vector_cosine_ops) WITH (m=16, ef_construction=64);

-- CREATE INDEX IF NOT EXISTS idx_review_embeddings_app_id ON review_embeddings(app_id);
-- CREATE INDEX IF NOT EXISTS idx_review_embeddings_country ON review_embeddings(country);
-- CREATE INDEX IF NOT EXISTS idx_review_embeddings_review_id ON review_embeddings(review_id);
-- CREATE INDEX IF NOT EXISTS idx_clean_reviews_app_id ON clean_reviews(app_id);
-- CREATE INDEX IF NOT EXISTS idx_clean_reviews_country ON clean_reviews(country);
-- CREATE INDEX IF NOT EXISTS idx_clean_reviews_reviewed_at ON clean_reviews(reviewed_at);

CREATE TABLE IF NOT EXISTS rag_query_logs (
    id SERIAL PRIMARY KEY,
    query_hash VARCHAR(64) NOT NULL,
    app_id VARCHAR(255) NOT NULL,
    query_text TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_time_ms INTEGER,
    result_count INTEGER,
    confidence_score DECIMAL(3,2)
);

CREATE INDEX IF NOT EXISTS idx_rag_query_logs_query_hash ON rag_query_logs(query_hash);
CREATE INDEX IF NOT EXISTS idx_rag_query_logs_app_id ON rag_query_logs(app_id);
CREATE INDEX IF NOT EXISTS idx_rag_query_logs_created_at ON rag_query_logs(created_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_query_logs_app_id_query_hash_unique ON rag_query_logs(app_id, query_hash);

CREATE TABLE IF NOT EXISTS embedding_cache (
    id SERIAL PRIMARY KEY,
    text_hash VARCHAR(64) UNIQUE NOT NULL,
    text_content TEXT NOT NULL,
    embedding_vector vector(1536) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_text_hash ON embedding_cache(text_hash);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_expires_at ON embedding_cache(expires_at);

CREATE TABLE IF NOT EXISTS rag_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    app_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_metrics_name_timestamp ON rag_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_rag_metrics_app_id ON rag_metrics(app_id);

INSERT INTO rag_metrics (metric_name, metric_value, app_id) VALUES
('avg_query_time_ms', 0.0, NULL),
('total_queries', 0.0, NULL),
('avg_confidence_score', 0.0, NULL)
ON CONFLICT DO NOTHING;

CREATE OR REPLACE FUNCTION cleanup_expired_embeddings()
RETURNS void AS $$
BEGIN
    DELETE FROM embedding_cache WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_rag_metrics(
    p_metric_name VARCHAR(100),
    p_metric_value DECIMAL(10,4),
    p_app_id VARCHAR(255) DEFAULT NULL
)
RETURNS void AS $$
BEGIN
    INSERT INTO rag_metrics (metric_name, metric_value, app_id)
    VALUES (p_metric_name, p_metric_value, p_app_id);
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT USAGE ON SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;
