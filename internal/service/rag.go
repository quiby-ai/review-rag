package service

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/quiby-ai/review-rag/internal/embedding"
	"github.com/quiby-ai/review-rag/internal/storage"
	"github.com/quiby-ai/review-rag/internal/types"
)

type RAGService struct {
	embedClient embedding.Client
	repo        storage.Repository
	config      RAGConfig
}

type RAGConfig struct {
	TopN          int
	TopK          int
	ANNProbes     int
	MinConfidence float64
}

func NewRAGService(embedClient embedding.Client, repo storage.Repository, config RAGConfig) *RAGService {
	return &RAGService{
		embedClient: embedClient,
		repo:        repo,
		config:      config,
	}
}

func (s *RAGService) Query(ctx context.Context, query types.RAGQuery) (*types.RAGResponse, error) {
	startTime := time.Now()

	queryEmbedding, err := s.embedClient.GenerateEmbedding(ctx, query.Query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	retrievedReviews, err := s.repo.RAGRetrieval(
		ctx,
		queryEmbedding,
		s.config.TopK,
		query.AppID,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve reviews: %w", err)
	}

	if len(retrievedReviews) == 0 {
		return s.buildEmptyResponse(query, startTime), nil
	}

	answer := s.generateAnswer(query.Query, retrievedReviews)

	confidence := s.calculateConfidence(retrievedReviews)

	processingTime := time.Since(startTime).Milliseconds()

	return &types.RAGResponse{
		Answer:           answer,
		RetrievedReviews: retrievedReviews,
		Confidence:       confidence,
		ProcessingTime:   float64(processingTime) / 1000.0,
		QueryHash:        s.embedClient.GetQueryHash(query.Query),
	}, nil
}

func (s *RAGService) buildEmptyResponse(query types.RAGQuery, startTime time.Time) *types.RAGResponse {
	return &types.RAGResponse{
		Answer:           "No relevant reviews found for your query.",
		RetrievedReviews: []types.RetrievedReview{},
		Confidence:       0.0,
		ProcessingTime:   time.Since(startTime).Seconds(),
		QueryHash:        s.embedClient.GetQueryHash(query.Query),
	}
}

func (s *RAGService) generateAnswer(query string, reviews []types.RetrievedReview) string {
	if len(reviews) == 0 {
		return "No relevant reviews found to answer your query."
	}

	var positiveCount, negativeCount int
	var avgRating float64

	for _, review := range reviews {
		if review.Rating >= 4 {
			positiveCount++
		} else if review.Rating <= 2 {
			negativeCount++
		}
		avgRating += float64(review.Rating)
	}

	avgRating = avgRating / float64(len(reviews))

	// Build answer based on sentiment and patterns
	var answer strings.Builder
	answer.WriteString(fmt.Sprintf("Based on %d relevant reviews, ", len(reviews)))

	if positiveCount > negativeCount {
		answer.WriteString("the overall sentiment is positive. ")
	} else if negativeCount > positiveCount {
		answer.WriteString("the overall sentiment is negative. ")
	} else {
		answer.WriteString("the sentiment is mixed. ")
	}

	answer.WriteString(fmt.Sprintf("The average rating is %.1f/5. ", avgRating))

	answer.WriteString("Here are the most relevant reviews that address your query.")

	return answer.String()
}

func (s *RAGService) calculateConfidence(reviews []types.RetrievedReview) float64 {
	if len(reviews) == 0 {
		return 0.0
	}

	var totalSimilarity float64
	for _, review := range reviews {
		totalSimilarity += review.Similarity
	}

	confidence := totalSimilarity / float64(len(reviews))
	if confidence < s.config.MinConfidence {
		confidence = s.config.MinConfidence
	}

	return confidence
}
