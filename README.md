# Review RAG Service

A microservice that enables intelligent querying of app store reviews using AI-powered search and answer generation.

## What it does

This service allows you to ask natural language questions about app store reviews and get intelligent answers based on the most relevant reviews. For example, you can ask "What do users think about the app performance?" and get a summary based on actual user reviews.

## How it works

1. **Takes your question** - You send a natural language query like "What are the main complaints?"
2. **Finds relevant reviews** - Uses vector similarity search to find the most relevant reviews from the database
3. **Generates an answer** - Analyzes the retrieved reviews and provides a comprehensive answer with supporting evidence

## API

**POST /** - Ask questions about reviews

```json
{
  "query": "What do users think about the app performance?",
  "appId": "1234567890"
}
```

**GET /healthz** - Check service status
