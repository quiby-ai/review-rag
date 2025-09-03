package service

import (
	"context"
	"testing"

	"github.com/quiby-ai/review-rag/internal/storage"
	"github.com/quiby-ai/review-rag/internal/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

type MockEmbeddingClient struct {
	mock.Mock
}

func (m *MockEmbeddingClient) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	args := m.Called(ctx, text)
	return args.Get(0).([]float32), args.Error(1)
}

func (m *MockEmbeddingClient) GetQueryHash(text string) string {
	args := m.Called(text)
	return args.String(0)
}

type MockRepository struct {
	mock.Mock
}

func (m *MockRepository) SearchSimilarReviews(ctx context.Context, queryEmbedding []float32, appID string, topN int, annProbes int) ([]types.RetrievedReview, error) {
	args := m.Called(ctx, queryEmbedding, appID, topN, annProbes)
	return args.Get(0).([]types.RetrievedReview), args.Error(1)
}

func (m *MockRepository) GetReviewDetails(ctx context.Context, reviewIDs []string) (map[string]storage.ReviewDetails, error) {
	args := m.Called(ctx, reviewIDs)
	return args.Get(0).(map[string]storage.ReviewDetails), args.Error(1)
}

func (m *MockRepository) RAGRetrieval(ctx context.Context, queryEmbedding []float32, topK int, appID string) ([]types.RetrievedReview, error) {
	args := m.Called(ctx, queryEmbedding, topK, appID)
	return args.Get(0).([]types.RetrievedReview), args.Error(1)
}

func (m *MockRepository) InitRAGTables(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

func (m *MockRepository) HealthCheck(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

func (m *MockRepository) Close() error {
	args := m.Called()
	return args.Error(0)
}

func TestRAGService_Query_Success(t *testing.T) {

	mockEmbed := &MockEmbeddingClient{}
	mockRepo := &MockRepository{}

	service := NewRAGService(mockEmbed, mockRepo, RAGConfig{
		TopN:          20,
		TopK:          5,
		ANNProbes:     10,
		MinConfidence: 0.7,
	})

	query := types.RAGQuery{
		Query: "What do users think about the app?",
		AppID: "com.test.app",
	}

	expectedEmbedding := []float32{0.1, 0.2, 0.3}
	mockEmbed.On("GenerateEmbedding", mock.Anything, query.Query).Return(expectedEmbedding, nil)
	mockEmbed.On("GetQueryHash", query.Query).Return("test-hash-123")

	expectedReviews := []types.RetrievedReview{
		{ID: "review-1", Similarity: 0.9, Country: "US", Rating: 5},
		{ID: "review-2", Similarity: 0.8, Country: "US", Rating: 4},
	}
	mockRepo.On("RAGRetrieval", mock.Anything, expectedEmbedding, 5, query.AppID).Return(expectedReviews, nil)

	ctx := context.Background()
	response, err := service.Query(ctx, query)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.Equal(t, "test-hash-123", response.QueryHash)
	assert.Len(t, response.RetrievedReviews, 2)
	assert.True(t, response.ProcessingTime >= 0)

	mockEmbed.AssertExpectations(t)
	mockRepo.AssertExpectations(t)
}

func TestRAGService_Query_NoResults(t *testing.T) {

	mockEmbed := &MockEmbeddingClient{}
	mockRepo := &MockRepository{}

	service := NewRAGService(mockEmbed, mockRepo, RAGConfig{
		TopN:          20,
		TopK:          5,
		ANNProbes:     10,
		MinConfidence: 0.7,
	})

	query := types.RAGQuery{
		Query: "What do users think about the app?",
		AppID: "com.test.app",
	}

	expectedEmbedding := []float32{0.1, 0.2, 0.3}
	mockEmbed.On("GenerateEmbedding", mock.Anything, query.Query).Return(expectedEmbedding, nil)
	mockEmbed.On("GetQueryHash", query.Query).Return("test-hash-123")

	mockRepo.On("RAGRetrieval", mock.Anything, expectedEmbedding, 5, query.AppID).Return([]types.RetrievedReview{}, nil)

	ctx := context.Background()
	response, err := service.Query(ctx, query)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.Equal(t, "No relevant reviews found for your query.", response.Answer)
	assert.Len(t, response.RetrievedReviews, 0)
	assert.Equal(t, 0.0, response.Confidence)
	assert.Equal(t, "test-hash-123", response.QueryHash)

	mockEmbed.AssertExpectations(t)
	mockRepo.AssertExpectations(t)
}

func TestRAGService_Query_EmbeddingError(t *testing.T) {

	mockEmbed := &MockEmbeddingClient{}
	mockRepo := &MockRepository{}

	service := NewRAGService(mockEmbed, mockRepo, RAGConfig{
		TopN:          20,
		TopK:          5,
		ANNProbes:     10,
		MinConfidence: 0.7,
	})

	query := types.RAGQuery{
		Query: "What do users think about the app?",
		AppID: "com.test.app",
	}

	mockEmbed.On("GenerateEmbedding", mock.Anything, query.Query).Return([]float32{}, assert.AnError)

	ctx := context.Background()
	response, err := service.Query(ctx, query)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "failed to generate embedding")

	mockEmbed.AssertExpectations(t)
}

func TestRAGService_Query_SearchError(t *testing.T) {

	mockEmbed := &MockEmbeddingClient{}
	mockRepo := &MockRepository{}

	service := NewRAGService(mockEmbed, mockRepo, RAGConfig{
		TopN:          20,
		TopK:          5,
		ANNProbes:     10,
		MinConfidence: 0.7,
	})

	query := types.RAGQuery{
		Query: "What do users think about the app?",
		AppID: "com.test.app",
	}

	expectedEmbedding := []float32{0.1, 0.2, 0.3}
	mockEmbed.On("GenerateEmbedding", mock.Anything, query.Query).Return(expectedEmbedding, nil)

	mockRepo.On("RAGRetrieval", mock.Anything, expectedEmbedding, 5, query.AppID).Return([]types.RetrievedReview(nil), assert.AnError)

	ctx := context.Background()
	response, err := service.Query(ctx, query)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "failed to retrieve reviews")

	mockEmbed.AssertExpectations(t)
	mockRepo.AssertExpectations(t)
}
