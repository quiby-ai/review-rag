package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/quiby-ai/review-rag/config"
	"github.com/quiby-ai/review-rag/internal/embedding"
	"github.com/quiby-ai/review-rag/internal/handler"
	"github.com/quiby-ai/review-rag/internal/service"
	"github.com/quiby-ai/review-rag/internal/storage"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	repo, err := storage.NewPostgresRepository(cfg.Database.DSN)
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	defer repo.Close()

	if err := repo.InitRAGTables(context.Background()); err != nil {
		log.Fatalf("Failed to initialize RAG tables: %v", err)
	}

	embedClient := embedding.NewClient(
		cfg.Embed.Endpoint,
		cfg.Embed.APIKey,
		cfg.Embed.Model,
		cfg.Embed.Timeout,
	)

	ragService := service.NewRAGService(embedClient, repo, service.RAGConfig{
		TopN:          cfg.RAG.TopN,
		TopK:          cfg.RAG.TopK,
		ANNProbes:     cfg.RAG.ANNProbes,
		MinConfidence: cfg.RAG.MinConfidence,
	})

	ragHandler := handler.NewRAGHandler(ragService)

	mux := http.NewServeMux()
	mux.HandleFunc("/", ragHandler.HandleRAGQuery)
	mux.HandleFunc("/healthz", ragHandler.HandleHealthCheck)

	server := &http.Server{
		Addr:         ":" + cfg.Server.Port,
		Handler:      mux,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
	}

	go func() {
		log.Printf("Starting RAG service on port %s", cfg.Server.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited")
}
