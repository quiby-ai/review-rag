package types

import "time"

type RAGQuery struct {
	Query string `json:"query" validate:"required,max=1000"`
	AppID string `json:"appId" validate:"required"`
}

type RetrievedReview struct {
	ID              string    `json:"id"`
	AppID           string    `json:"app_id"`
	Title           string    `json:"title"`
	Content         string    `json:"content"`
	ResponseContent *string   `json:"response_content,omitempty"`
	Rating          int16     `json:"rating"`
	Country         string    `json:"country"`
	Language        string    `json:"language"`
	Date            time.Time `json:"date"`
	Distance        float64   `json:"distance"`
	Similarity      float64   `json:"similarity"`
}

type RAGResponse struct {
	Answer           string            `json:"answer"`
	RetrievedReviews []RetrievedReview `json:"retrievedReviews"`
	Confidence       float64           `json:"confidence"`
	ProcessingTime   float64           `json:"processingTime"`
	QueryHash        string            `json:"queryHash"`
}

type EmbeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type EmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model  string `json:"model"`
	Object string `json:"object"`
	Usage  struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

type HealthResponse struct {
	Status    string            `json:"status"`
	Timestamp time.Time         `json:"timestamp"`
	Version   string            `json:"version"`
	Database  DatabaseHealth    `json:"database"`
	Embedding EmbeddingHealth   `json:"embedding"`
	Metrics   map[string]string `json:"metrics"`
}

type DatabaseHealth struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type EmbeddingHealth struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}
