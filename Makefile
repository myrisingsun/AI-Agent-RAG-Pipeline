.PHONY: up-infra down-infra init-collections seed-test-data \
        api-dev ui-dev test test-unit test-integration eval \
        lint format help

# =============================================================================
# Infrastructure
# =============================================================================

up-infra:  ## Start all Docker services (Qdrant, Redis, PG, MinIO)
	docker compose up -d --wait
	@echo "All services healthy."
	@echo "  Qdrant:     http://localhost:6333"
	@echo "  MinIO UI:   http://localhost:9001"

down-infra:  ## Stop and remove all Docker services
	docker compose down

down-infra-volumes:  ## Stop services AND delete all data volumes
	docker compose down -v

# =============================================================================
# Qdrant collections
# =============================================================================

init-collections:  ## Create 4 Qdrant collections (idempotent)
	python scripts/init_collections.py

# =============================================================================
# Test data
# =============================================================================

seed-test-data:  ## Load sample vectors into all 4 collections
	python scripts/seed_test_data.py

# =============================================================================
# Development servers
# =============================================================================

api-dev:  ## Run FastAPI backend with hot-reload (port 8000)
	uvicorn src.api.main:app --reload --port 8000

ui-dev:  ## Run React frontend with HMR (port 5173)
	cd frontend && npm run dev

# =============================================================================
# Testing
# =============================================================================

test:  ## Run all tests
	pytest tests/ -x -v --tb=short

test-unit:  ## Run unit tests only (no infrastructure required)
	pytest tests/unit/ -x -v --tb=short

test-integration:  ## Run integration tests (requires Docker or testcontainers)
	pytest tests/integration/ -x -v --tb=short -m integration

test-one:  ## Run a single test file: make test-one FILE=tests/unit/test_embeddings.py
	pytest $(FILE) -x -v --tb=long -s

# =============================================================================
# Evaluation
# =============================================================================

eval:  ## Run RAGAS + DeepEval on eval-dataset
	python -m src.rag.evaluation.ragas_eval

# =============================================================================
# Code quality
# =============================================================================

lint:  ## Run ruff + mypy
	ruff check src/ tests/ scripts/
	mypy src/ --strict --ignore-missing-imports

format:  ## Auto-format with ruff
	ruff format src/ tests/ scripts/
	ruff check src/ tests/ scripts/ --fix

# =============================================================================
# API types (frontend)
# =============================================================================

gen-types:  ## Regenerate TypeScript types from OpenAPI spec
	cd frontend && npx openapi-typescript http://localhost:8000/openapi.json -o src/types/api.ts

# =============================================================================

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
