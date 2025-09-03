package storage

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	"github.com/quiby-ai/review-rag/internal/types"
)

type Repository interface {
	SearchSimilarReviews(ctx context.Context, queryEmbedding []float32, appID string, topN int, annProbes int) ([]types.RetrievedReview, error)
	GetReviewDetails(ctx context.Context, reviewIDs []string) (map[string]ReviewDetails, error)
	RAGRetrieval(ctx context.Context, queryEmbedding []float32, topK int, appID string) ([]types.RetrievedReview, error)
	InitRAGTables(ctx context.Context) error
	HealthCheck(ctx context.Context) error
	Close() error
}

type ReviewDetails struct {
	ID           string
	Content      string
	Rating       int16
	Country      string
	ReviewedAt   time.Time
	HelpfulCount *int
}

type postgresRepository struct {
	db *pgxpool.Pool
}

func NewPostgresRepository(dsn string) (Repository, error) {
	pool, err := pgxpool.New(context.Background(), dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := pool.Ping(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	repo := &postgresRepository{db: pool}

	if err := repo.initTables(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to initialize tables: %w", err)
	}

	return repo, nil
}

func (r *postgresRepository) initTables(ctx context.Context) error {
	queries := []string{
		`CREATE EXTENSION IF NOT EXISTS vector;`,
		`SET hnsw.ef_search = 96;`,
	}

	indexQueries := []string{
		`CREATE INDEX IF NOT EXISTS idx_review_embeddings_hnsw ON review_embeddings USING hnsw (content_vec vector_cosine_ops) WITH (m=16, ef_construction=64);`,
		// `CREATE INDEX IF NOT EXISTS idx_review_embeddings_app_id ON review_embeddings(app_id);`,
		// `CREATE INDEX IF NOT EXISTS idx_review_embeddings_country ON review_embeddings(country);`,
		// `CREATE INDEX IF NOT EXISTS idx_review_embeddings_review_id ON review_embeddings(review_id);`,
		// `CREATE INDEX IF NOT EXISTS idx_clean_reviews_app_id ON clean_reviews(app_id);`,
		// `CREATE INDEX IF NOT EXISTS idx_clean_reviews_country ON clean_reviews(country);`,
		// `CREATE INDEX IF NOT EXISTS idx_clean_reviews_reviewed_at ON clean_reviews(reviewed_at);`,
	}

	for i, query := range queries {
		if _, err := r.db.Exec(ctx, query); err != nil {
			return fmt.Errorf("failed to execute query %d: %w", i+1, err)
		}
	}

	for _, query := range indexQueries {
		if _, err := r.db.Exec(ctx, query); err != nil {
			continue
		}
	}

	return nil
}

func (r *postgresRepository) InitRAGTables(ctx context.Context) error {
	queries := []string{
		`CREATE TABLE IF NOT EXISTS rag_query_logs (
			id SERIAL PRIMARY KEY,
			query_hash VARCHAR(64) NOT NULL,
			app_id VARCHAR(255) NOT NULL,
			query_text TEXT NOT NULL,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			processing_time_ms INTEGER,
			result_count INTEGER,
			confidence_score DECIMAL(3,2)
		);`,

		`CREATE INDEX IF NOT EXISTS idx_rag_query_logs_query_hash ON rag_query_logs(query_hash);`,
		`CREATE INDEX IF NOT EXISTS idx_rag_query_logs_app_id ON rag_query_logs(app_id);`,
		`CREATE INDEX IF NOT EXISTS idx_rag_query_logs_created_at ON rag_query_logs(created_at);`,

		`CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_query_logs_app_id_query_hash_unique ON rag_query_logs(app_id, query_hash);`,

		`CREATE TABLE IF NOT EXISTS embedding_cache (
			id SERIAL PRIMARY KEY,
			text_hash VARCHAR(64) UNIQUE NOT NULL,
			text_content TEXT NOT NULL,
			embedding_vector vector(1536) NOT NULL,
			model_name VARCHAR(100) NOT NULL,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			expires_at TIMESTAMP WITH TIME ZONE NOT NULL
		);`,

		`CREATE INDEX IF NOT EXISTS idx_embedding_cache_text_hash ON embedding_cache(text_hash);`,
		`CREATE INDEX IF NOT EXISTS idx_embedding_cache_expires_at ON embedding_cache(expires_at);`,

		`CREATE TABLE IF NOT EXISTS rag_metrics (
			id SERIAL PRIMARY KEY,
			metric_name VARCHAR(100) NOT NULL,
			metric_value DECIMAL(10,4) NOT NULL,
			app_id VARCHAR(255),
			timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		);`,

		`CREATE INDEX IF NOT EXISTS idx_rag_metrics_name_timestamp ON rag_metrics(metric_name, timestamp);`,
		`CREATE INDEX IF NOT EXISTS idx_rag_metrics_app_id ON rag_metrics(app_id);`,

		`INSERT INTO rag_metrics (metric_name, metric_value, app_id) VALUES
		('avg_query_time_ms', 0.0, NULL),
		('total_queries', 0.0, NULL),
		('avg_confidence_score', 0.0, NULL)
		ON CONFLICT DO NOTHING;`,

		`CREATE OR REPLACE FUNCTION cleanup_expired_embeddings()
		RETURNS void AS $$
		BEGIN
			DELETE FROM embedding_cache WHERE expires_at < NOW();
		END;
		$$ LANGUAGE plpgsql;`,

		`CREATE OR REPLACE FUNCTION update_rag_metrics(
			p_metric_name VARCHAR(100),
			p_metric_value DECIMAL(10,4),
			p_app_id VARCHAR(255) DEFAULT NULL
		)
		RETURNS void AS $$
		BEGIN
			INSERT INTO rag_metrics (metric_name, metric_value, app_id)
			VALUES (p_metric_name, p_metric_value, p_app_id);
		END;
		$$ LANGUAGE plpgsql;`,
	}

	for i, query := range queries {
		if _, err := r.db.Exec(ctx, query); err != nil {
			return fmt.Errorf("failed to execute RAG table query %d: %w", i+1, err)
		}
	}

	return nil
}

func (r *postgresRepository) SearchSimilarReviews(ctx context.Context, queryEmbedding []float32, appID string, topN int, annProbes int) ([]types.RetrievedReview, error) {
	queryEmbeddingVec := pgvector.NewVector(queryEmbedding)

	// Build the WHERE clause for filtering
	whereClause := "WHERE re.app_id = $1"
	args := []any{queryEmbeddingVec, appID}
	argIndex := 3

	// Add ANN probes and limit
	args = append(args, annProbes, topN)

	query := fmt.Sprintf(`
		SET ivfflat.probes = $%d;

		SELECT
			re.review_id,
			re.rating,
			re.content_vec <=> $1 as distance,
			1 - (re.content_vec <=> $1) as similarity
		FROM review_embeddings re
		%s
		ORDER BY re.content_vec <=> $1
		LIMIT $%d;
	`, argIndex, whereClause, argIndex+1)

	rows, err := r.db.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query similar reviews: %w", err)
	}
	defer rows.Close()

	var reviews []types.RetrievedReview
	for rows.Next() {
		var review types.RetrievedReview
		var distance, similarity float64
		if err := rows.Scan(
			&review.ID,
			&review.Rating,
			&distance,
			&similarity,
		); err != nil {
			return nil, fmt.Errorf("failed to scan review: %w", err)
		}
		review.Similarity = similarity
		reviews = append(reviews, review)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating rows: %w", err)
	}

	return reviews, nil
}

func (r *postgresRepository) GetReviewDetails(ctx context.Context, reviewIDs []string) (map[string]ReviewDetails, error) {
	if len(reviewIDs) == 0 {
		return make(map[string]ReviewDetails), nil
	}

	placeholders := make([]string, len(reviewIDs))
	args := make([]any, len(reviewIDs))
	for i, id := range reviewIDs {
		placeholders[i] = fmt.Sprintf("$%d", i+1)
		args[i] = id
	}

	query := fmt.Sprintf(`
		SELECT
			cr.id,
			COALESCE(cr.content_en, cr.content_clean) as content,
			cr.rating,
			cr.country,
			cr.reviewed_at
		FROM clean_reviews cr
		WHERE cr.id = ANY(%s)
	`, fmt.Sprintf("(%s)", placeholders[0]))

	rows, err := r.db.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query review details: %w", err)
	}
	defer rows.Close()

	details := make(map[string]ReviewDetails)
	for rows.Next() {
		var detail ReviewDetails
		if err := rows.Scan(
			&detail.ID,
			&detail.Content,
			&detail.Rating,
			&detail.Country,
			&detail.ReviewedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan review detail: %w", err)
		}
		details[detail.ID] = detail
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating rows: %w", err)
	}

	return details, nil
}

func (r *postgresRepository) RAGRetrieval(ctx context.Context, queryEmbedding []float32, topK int, appID string) ([]types.RetrievedReview, error) {
	// Normalize the query embedding (L2 normalization)
	queryVec := pgvector.NewVector(queryEmbedding)

	// Set default topK if not specified
	if topK <= 0 {
		topK = 20
	}

	// Prepare the SQL query with optional filters
	query := `
		SELECT
			cr.id,
			cr.app_id,
			cr.title,
			cr.content_clean AS content,
			cr.response_content_clean AS response_content,
			cr.rating,
			cr.country,
			cr.language,
			cr.reviewed_at AS date,
			(re.content_vec <=> $1) AS distance
		FROM review_embeddings re
		JOIN clean_reviews cr ON cr.id = re.review_id
		WHERE
			($3 IS NULL OR cr.app_id = $3)
		ORDER BY re.content_vec <=> $1
		LIMIT $2;
	`

	// Execute the query with parameters
	rows, err := r.db.Query(ctx, query, queryVec, topK, appID)
	if err != nil {
		return nil, fmt.Errorf("failed to execute RAG retrieval query: %w", err)
	}
	defer rows.Close()

	var reviews []types.RetrievedReview
	for rows.Next() {
		var review types.RetrievedReview
		var distance float64
		var responseContent *string

		if err := rows.Scan(
			&review.ID,
			&review.AppID,
			&review.Title,
			&review.Content,
			&responseContent,
			&review.Rating,
			&review.Country,
			&review.Language,
			&review.Date,
			&distance,
		); err != nil {
			return nil, fmt.Errorf("failed to scan review: %w", err)
		}

		review.ResponseContent = responseContent
		review.Distance = distance
		review.Similarity = 1.0 - distance // Convert distance to similarity
		reviews = append(reviews, review)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating rows: %w", err)
	}

	return reviews, nil
}

func (r *postgresRepository) HealthCheck(ctx context.Context) error {
	return r.db.Ping(ctx)
}

func (r *postgresRepository) Close() error {
	r.db.Close()
	return nil
}
