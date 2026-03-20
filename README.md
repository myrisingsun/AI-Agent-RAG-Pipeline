# AI-агент КОД — RAG Pipeline

> Блок 2: RAG Pipeline для AI-агента анализа кредитной обеспечительной документации

## Обзор

RAG (Retrieval-Augmented Generation) pipeline обеспечивает два сценария для кредитных аналитиков и юристов банка:

1. **Q&A** — аналитик загружает пакет документов, задаёт вопросы, получает ответы со ссылками на конкретные разделы
2. **Автоматическая проверка** — агент проверяет документы на соответствие нормативной базе ЦБ и выдаёт отчёт с замечаниями

### Архитектура

```
[React Frontend] ← WS/REST → [FastAPI /api/v1/] ← async → [RAG Pipeline] ← gRPC → [Qdrant]
                                      ↕                          ↕
                                 [PostgreSQL]              [LLM (vLLM)]
```

Аналитик работает через React UI. Qdrant Dashboard — инструмент разработчика, не пользователя.

## Технологический стек

### Backend (Python 3.11+)

| Компонент | Технология | Назначение |
|---|---|---|
| API | FastAPI + uvicorn | REST + WebSocket стриминг LLM |
| RAG оркестрация | LangChain (MVP) | Интеграции Qdrant, Docling, vLLM |
| Embedding MVP | multilingual-e5-large | CPU, 512 токенов, 560MB |
| Embedding Prod | BGE-M3 / Qwen3-Embedding-0.6B | GPU, hybrid native |
| Векторная БД | Qdrant | Hybrid search, payload filters |
| Reranker | mmarco-mMiniLMv2-L12-H384 | Cross-encoder, top-50 → top-5 |
| Chunking MVP | Fixed-size 400 tok + 15% overlap | tokenizers для подсчёта |
| Chunking Prod | Docling semantic/table-aware | По типу документа |
| БД метаданных | PostgreSQL 15 + SQLAlchemy async | JSONB для extracted_fields |
| Очередь | Celery + Redis | Фоновая индексация |
| Хранилище | MinIO | S3-совместимый, on-premise |
| Eval | RAGAS + DeepEval | Faithfulness, MRR, regression tests |

### Frontend (React 18+)

| Компонент | Технология | Назначение |
|---|---|---|
| Framework | React 18 + TypeScript strict | UI |
| Сборка | Vite | HMR, быстрый dev |
| Стили | Tailwind CSS | Utility-first |
| State | Zustand + @tanstack/react-query | UI state + server state |
| Типы API | openapi-typescript | Генерация из OpenAPI spec |
| Иконки | lucide-react | MIT |

### Инфраструктура (Docker)

| Сервис | Образ | Порт |
|---|---|---|
| Qdrant | qdrant/qdrant:1.12 | 6333 (REST), 6334 (gRPC) |
| Redis | redis:7-alpine | 6379 |
| PostgreSQL | postgres:15-alpine | 5432 |
| MinIO | minio/minio:latest | 9000 (API), 9001 (Console) |

## Быстрый старт

```bash
# 1. Клонировать и настроить
git clone <repo>
cd ai-agent-kod
cp .env.example .env  # отредактировать под своё окружение

# 2. Поднять инфраструктуру
make up-infra

# 3. Создать коллекции Qdrant
make init-collections

# 4. Загрузить тестовые данные
make seed-test-data

# 5. Запустить backend
make api-dev

# 6. Запустить frontend (в отдельном терминале)
make ui-dev

# Открыть http://localhost:5173
```

## Структура проекта

```
ai-agent-kod/
├── CLAUDE.md                          # Инструкции для Claude Code
├── README.md                          # Этот файл
├── docker-compose.yml
├── .env.example
├── pyproject.toml
├── Makefile
│
├── src/
│   ├── api/                           # FastAPI backend
│   │   ├── main.py                    # App, CORS, lifespan
│   │   ├── deps.py                    # Dependency injection
│   │   ├── routes/                    # Endpoints
│   │   │   ├── documents.py           # POST /documents/upload
│   │   │   ├── search.py             # POST /search (Q&A)
│   │   │   ├── validation.py         # POST /validate (автопроверка)
│   │   │   ├── collections.py        # GET /collections/stats
│   │   │   └── ws.py                 # WebSocket /ws/chat
│   │   └── middleware/
│   │       ├── auth.py               # JWT → банковский SSO
│   │       └── audit.py              # Audit log в PostgreSQL
│   │
│   ├── rag/                           # RAG Pipeline
│   │   ├── config.py                  # RAGConfig (Pydantic Settings)
│   │   ├── embeddings/               # Провайдеры embedding-моделей
│   │   │   ├── base.py               # ABC: EmbeddingProvider
│   │   │   ├── e5_large.py           # MVP
│   │   │   ├── bge_m3.py            # Production
│   │   │   └── qwen3.py             # Production
│   │   ├── chunking/                 # Стратегии нарезки
│   │   │   ├── base.py               # ABC: ChunkingStrategy
│   │   │   ├── fixed_size.py         # MVP: 400 tok + overlap
│   │   │   ├── semantic.py           # Docling: по разделам договора
│   │   │   ├── table_aware.py        # Финотчётность: таблица = атомарный чанк
│   │   │   └── hierarchical.py       # Законы: по статьям
│   │   ├── vectorstore/              # Qdrant-обёртка
│   │   │   ├── client.py            # QdrantManager: init, health check
│   │   │   ├── collections.py       # Конфигурации 4 коллекций
│   │   │   └── operations.py        # upsert, search, delete, scroll
│   │   ├── indexing/                 # Фоновая индексация
│   │   │   ├── pipeline.py          # IndexingPipeline
│   │   │   ├── normative_loader.py  # Нормативная база
│   │   │   ├── precedent_loader.py  # Прецеденты из АБС
│   │   │   ├── template_loader.py   # Эталоны
│   │   │   └── package_indexer.py   # Текущий пакет (on-the-fly)
│   │   ├── retrieval/               # Онлайн-поиск
│   │   │   ├── pipeline.py          # RetrievalPipeline
│   │   │   ├── hybrid_search.py     # Dense + BM25 + RRF
│   │   │   ├── reranker.py          # Cross-encoder
│   │   │   ├── context_builder.py   # Сборка контекста для LLM
│   │   │   └── query_router.py      # Роутинг по коллекциям
│   │   ├── validation/              # Автопроверка по нормативам
│   │   │   ├── engine.py            # ValidationEngine
│   │   │   ├── rules.py             # Бизнес-правила
│   │   │   └── report.py            # Генерация отчёта
│   │   └── evaluation/              # RAG Evaluation
│   │       ├── ragas_eval.py        # RAGAS metrics
│   │       ├── deepeval_tests.py    # Pytest regression
│   │       └── benchmark_embeddings.py
│   │
│   ├── schemas/                      # Pydantic-модели (= контракт API)
│   └── common/                       # Утилиты, logging, settings
│
├── frontend/                          # React Frontend
│   ├── src/
│   │   ├── api/                      # REST client + WebSocket hook
│   │   ├── components/
│   │   │   ├── layout/              # Sidebar, Header
│   │   │   ├── documents/           # UploadZone, DocumentList, DocumentViewer
│   │   │   ├── chat/                # ChatPanel, MessageBubble, SourceCard
│   │   │   ├── validation/          # ValidationPanel, FindingCard, ReportExport
│   │   │   └── admin/               # CollectionStats
│   │   ├── hooks/                   # useChat, useDocuments, useValidation
│   │   ├── stores/                  # Zustand stores
│   │   └── types/                   # Генерация из OpenAPI
│   └── public/
│
├── tests/
│   ├── unit/                         # Моки, быстрые тесты
│   ├── integration/                  # Testcontainers, API тесты
│   └── eval/                         # RAGAS + DeepEval
│
├── data/
│   ├── raw/                          # Тестовые документы
│   ├── ground_truth/                 # Eval-dataset (ручная разметка)
│   ├── normative/                    # Нормативная база
│   └── templates/                    # Эталонные документы
│
├── prompts/                           # Git-версионированные промпты
│   └── rag/
│       ├── qa_system.txt
│       ├── normative_search.txt
│       └── validation_check.txt
│
└── scripts/                           # CLI-утилиты
    ├── init_collections.py
    ├── load_normative_base.py
    ├── benchmark_embeddings.py
    └── seed_test_data.py
```

## 4 коллекции Qdrant

| Коллекция | Источник | Обновление | Стратегия chunking | Ключевые метаданные |
|---|---|---|---|---|
| `normative_base` | Законы РФ, нормативы ЦБ, инструкции банка | По событию / раз в квартал | Иерархический по статьям | `law_article`, `effective_date` |
| `deal_precedents` | Закрытые кредитные дела из АБС | При закрытии сделки | По документам сделки | `deal_id`, `outcome`, `risk_flags` |
| `reference_templates` | Эталоны, few-shot примеры для extraction | При добавлении нового типа | По разделам | `doc_type`, `template_version` |
| `current_package` | Документы текущего загруженного пакета | Создаётся при загрузке, удаляется по TTL | Fixed-size (MVP) / semantic | `session_id`, `doc_type`, `page` |

## API Endpoints

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/v1/documents/upload` | Загрузка пакета документов → индексация |
| GET | `/api/v1/documents/{id}` | Получение документа и его метаданных |
| POST | `/api/v1/search` | Q&A запрос → retrieval → ответ с citations |
| POST | `/api/v1/validate` | Запуск автопроверки по нормативной базе |
| GET | `/api/v1/collections/stats` | Статистика коллекций Qdrant |
| WS | `/ws/chat` | Стриминг токенов LLM + промежуточные события |

## Метрики качества

| Метрика | Target MVP | Target Production |
|---|---|---|
| MRR@10 | ≥ 0.65 | ≥ 0.75 |
| Recall@5 | ≥ 0.75 | ≥ 0.85 |
| Faithfulness (RAGAS) | ≥ 0.85 | ≥ 0.90 |
| Context Precision | ≥ 0.70 | ≥ 0.80 |
| Search latency p95 | < 500ms | < 200ms |
| UI: upload → result | < 30s | < 15s |
| UI: Q&A first token | < 2s | < 1s |

## Дорожная карта (8 спринтов × 2 недели)

| # | Спринт | Результат |
|---|---|---|
| S1 | Foundation | Qdrant + 4 коллекции + e5-large |
| S2 | Indexing Pipeline | Документ → Qdrant E2E |
| S3 | Retrieval Pipeline | Поиск + reranker + context builder |
| S4 | Hybrid + Evaluation | Hybrid search + embedding benchmark |
| S5 | Semantic Chunking + Q&A + Автопроверка | Q&A с citations + ValidationEngine |
| S6 | Hardening | Incremental indexing, monitoring, integration |
| S7 | FastAPI + WebSocket | API + стриминг + auth + audit |
| S8 | React Frontend | UI: загрузка, Q&A, проверка, экспорт |

Спринты 7–8 параллельны с 5–6 при наличии фронтенд-разработчика (12–14 нед. вместо 16).

## Зависимости от других блоков

- **Блок 1 (Parsing):** Docling парсит документы → RAG получает текст + структуру
- **Блок 3 (LLM):** RAG передаёт контекст в LLM (vLLM) для генерации ответов
- **Блок 4 (Extraction):** RAG обогащает extraction few-shot контекстом из reference_templates
- **Блок 5 (Infra):** PostgreSQL, MinIO, Redis, Docker Compose
- **Блок 6 (Security):** JWT → SSO, Presidio (маскирование ПДн), audit log
- **Блок 7 (Quality):** RAGAS + DeepEval для evaluation

## Лицензии

Весь стек — open-source (Apache 2.0 / MIT). Никакие данные не покидают периметр банка.
