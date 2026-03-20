# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Что это
RAG pipeline для анализа кредитной обеспечительной документации банка.
Пользователи — кредитные аналитики, юристы. On-premise, данные не покидают периметр.

Два сценария: Q&A с citations по загруженным документам + автоматическая проверка соответствия нормативной базе ЦБ.

## Стек
- Python 3.11+, FastAPI, async/await
- Qdrant (векторная БД), LangChain (оркестрация MVP)
- Embedding: multilingual-e5-large (MVP, CPU, 512 tok, 560MB), BGE-M3/Qwen3-Embedding-0.6B (production, GPU)
- Reranker: mmarco-mMiniLMv2-L12-H384 (cross-encoder, top-50 → top-5)
- LLM: vLLM (self-hosted, `VLLM_API_URL` в .env)
- PostgreSQL 15, Redis, MinIO, Celery
- Frontend: React 18 + TypeScript + Vite + Tailwind + Zustand
- Тесты: pytest + pytest-asyncio, Vitest (frontend)

## Архитектура
```
[React :5173] ← WS/REST → [FastAPI :8000 /api/v1/] ← async → [RAG Pipeline] ← gRPC → [Qdrant :6333]
                                      ↕                              ↕
                                 [PostgreSQL :5432]           [vLLM (self-hosted)]
```

Retrieval pipeline: dense + BM25 → hybrid search с RRF fusion → cross-encoder reranker → context builder → LLM.
Query router определяет, в какую из 4 коллекций направить запрос.

## Запуск
```bash
make up-infra              # Qdrant:6333/6334, Redis:6379, PG:5432, MinIO:9000/9001
make init-collections      # Создать 4 коллекции Qdrant
make seed-test-data        # Загрузить тестовые данные
make api-dev               # uvicorn src.api.main:app --reload --port 8000
make ui-dev                # cd frontend && npm run dev (порт 5173, proxy → 8000)
make test                  # pytest tests/ -x
make eval                  # RAGAS + DeepEval на eval-dataset
```

Запуск одного теста:
```bash
pytest tests/unit/test_embeddings.py -x -k "test_name"
pytest tests/integration/test_api_endpoints.py -x -s
cd frontend && npx vitest run src/components/chat/ChatPanel.test.tsx
```

Регенерация типов из OpenAPI:
```bash
npx openapi-typescript http://localhost:8000/openapi.json -o frontend/src/types/api.ts
```

## API Endpoints
| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/v1/documents/upload` | Загрузка пакета → индексация |
| GET | `/api/v1/documents/{id}` | Документ + метаданные |
| POST | `/api/v1/search` | Q&A → retrieval → ответ с citations |
| POST | `/api/v1/validate` | Автопроверка по нормативной базе |
| GET | `/api/v1/collections/stats` | Статистика коллекций Qdrant |
| WS | `/ws/chat` | Стриминг токенов LLM + события |

## 4 коллекции Qdrant
| Коллекция | Что внутри | Chunking | Ключевые payload-поля |
|---|---|---|---|
| `normative_base` | Законы, нормативы ЦБ | Иерархический по статьям | `law_article`, `effective_date`, `doc_type` |
| `deal_precedents` | Закрытые кредитные дела | По документам сделки | `deal_id`, `outcome`, `risk_flags` |
| `reference_templates` | Эталоны, few-shot примеры | По разделам | `doc_type`, `template_version` |
| `current_package` | Текущий пакет (временная, TTL) | Fixed-size MVP / semantic prod | `session_id`, `doc_type`, `page` |

## Правила кода

### Python
- Type hints обязательны на всех публичных функциях и методах
- Pydantic v2 для моделей данных, Pydantic Settings для конфигов
- async def для любого I/O (Qdrant, PostgreSQL, HTTP, файлы)
- Никогда не хардкодить параметры — всё через .env / RAGConfig
- Логирование: `structlog.get_logger()`, JSON-формат, без print()
- Импорты: stdlib → third-party → local, разделены пустой строкой

### React / TypeScript
- TypeScript strict mode, no `any`
- Типы API — генерация из OpenAPI (команда выше)
- Стили: только Tailwind utility classes, без CSS-файлов
- State: Zustand для UI state, @tanstack/react-query для server state
- Компонент = один файл, именование PascalCase.tsx

### Общее
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`
- Ветки: `feat/rag-*`, `feat/api-*`, `feat/ui-*`

## Паттерны (следуй при создании новых компонентов)

### Новый embedding-провайдер
1. Создать `src/rag/embeddings/{name}.py`
2. Наследовать `EmbeddingProvider` из `src/rag/embeddings/base.py`
3. Реализовать `async embed_texts(texts: list[str]) -> list[list[float]]`
4. Реализовать `async embed_query(text: str) -> list[float]`
5. Добавить в `RAGConfig.embedding_provider` (enum в config.py)
6. Тест: `tests/unit/test_embeddings.py`

### Новая стратегия chunking
1. Создать `src/rag/chunking/{name}.py`
2. Наследовать `ChunkingStrategy` из `src/rag/chunking/base.py`
3. Реализовать `chunk_document(doc: ParsedDocument) -> list[Chunk]`
4. Каждый Chunk обязан содержать ChunkMetadata (см. src/schemas/chunk.py)
5. Зарегистрировать маппинг doc_type → strategy в config.py
6. Тест: `tests/unit/test_chunking.py`

Рекомендуемый маппинг: финансовая отчётность → `table_aware`, законы → `hierarchical`, договоры → `semantic`, всё остальное → `fixed_size`.

### Новый API endpoint
1. Создать/дополнить файл в `src/api/routes/`
2. Pydantic-схемы request/response в `src/schemas/`
3. Бизнес-логика — в `src/rag/`, НЕ в route handler
4. Route handler: только валидация → вызов сервиса → return
5. Dependency injection через `Depends()` (см. `src/api/deps.py`)
6. Тест: `tests/integration/test_api_endpoints.py` через `httpx.AsyncClient`

### Новый React компонент
1. Создать `frontend/src/components/{section}/{Name}.tsx`
2. Props — отдельный interface в том же файле
3. API вызовы — через хук в `frontend/src/hooks/`
4. Не использовать useEffect для fetch — использовать react-query

## Целевые метрики качества
| Метрика | MVP | Production |
|---|---|---|
| MRR@10 | ≥ 0.65 | ≥ 0.75 |
| Recall@5 | ≥ 0.75 | ≥ 0.85 |
| Faithfulness (RAGAS) | ≥ 0.85 | ≥ 0.90 |
| Context Precision | ≥ 0.70 | ≥ 0.80 |
| Search latency p95 | < 500ms | < 200ms |
| Upload → result | < 30s | < 15s |
| Q&A first token | < 2s | < 1s |

## Важные ограничения
- Никаких внешних API (OpenAI, Cohere и т.д.) — всё self-hosted
- Промпты лежат в `prompts/` и версионируются в git. Изменение промпта = PR с `make eval`
- Eval-dataset: `data/ground_truth/` — не генерировать, только размеченные вручную данные
- MinIO bucket `documents` — оригиналы файлов. Никогда не удалять оригинал.
- `current_package` коллекция — временная, удалять по завершении сессии (TTL)
- Аутентификация: JWT → банковский SSO (`src/api/middleware/auth.py`)
- PII маскирование: Presidio (Блок 6 Security)

## Чего НЕ делать
- Не использовать `print()` — только structlog
- Не писать бизнес-логику в API route handlers
- Не создавать синхронные функции для I/O
- Не использовать `WidthType.PERCENTAGE` в docx (ломает Google Docs)
- Не коммитить .env, модели, данные в git
- Не использовать localStorage в React (нет поддержки в банковском окружении)
