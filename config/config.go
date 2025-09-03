package config

import (
	"fmt"
	"time"

	"github.com/spf13/viper"
)

type Config struct {
	Server   ServerConfig
	Database DatabaseConfig
	Embed    EmbedConfig
	RAG      RAGConfig
}

type ServerConfig struct {
	Port         string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	IdleTimeout  time.Duration
}

type DatabaseConfig struct {
	DSN string
}

type EmbedConfig struct {
	Model    string
	Endpoint string
	APIKey   string
	Timeout  time.Duration
	CacheTTL time.Duration
}

type RAGConfig struct {
	TopN           int
	TopK           int
	ANNProbes      int
	MinConfidence  float64
	MaxQueryLength int
}

func Load() (*Config, error) {
	viper.SetConfigName("config")
	viper.SetConfigType("toml")
	viper.AddConfigPath("/")

	viper.AutomaticEnv()

	viper.BindEnv("PG_DSN")
	viper.BindEnv("OPENAI_API_KEY")

	if err := viper.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	config := &Config{
		Server: ServerConfig{
			Port:         viper.GetString("server.port"),
			ReadTimeout:  viper.GetDuration("server.read_timeout_seconds"),
			WriteTimeout: viper.GetDuration("server.write_timeout_seconds"),
			IdleTimeout:  viper.GetDuration("server.idle_timeout_seconds"),
		},
		Database: DatabaseConfig{
			DSN: viper.GetString("PG_DSN"),
		},
		Embed: EmbedConfig{
			Model:    viper.GetString("embed.model"),
			Endpoint: viper.GetString("embed.endpoint"),
			APIKey:   viper.GetString("OPENAI_API_KEY"),
			Timeout:  viper.GetDuration("embed.timeout_seconds"),
			CacheTTL: viper.GetDuration("embed.cache_ttl_seconds"),
		},
		RAG: RAGConfig{
			TopN:           viper.GetInt("rag.top_n"),
			TopK:           viper.GetInt("rag.top_k"),
			ANNProbes:      viper.GetInt("rag.ann_probes"),
			MinConfidence:  viper.GetFloat64("rag.min_confidence"),
			MaxQueryLength: viper.GetInt("rag.max_query_length"),
		},
	}

	if config.Database.DSN == "" {
		return nil, fmt.Errorf("PG_DSN environment variable is required")
	}

	if config.Embed.APIKey == "" {
		return nil, fmt.Errorf("EMBED_API_KEY environment variable is required")
	}

	return config, nil
}
