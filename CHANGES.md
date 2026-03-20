# API Layer — изменения

## Новые файлы

### `src/api/`
| Файл | Описание |
|---|---|
| `main.py` | FastAPI приложение, lifespan (инициализация Qdrant, embedding, LLM, сервисов), CORS, роутер |
| `deps.py` | FastAPI `Depends()` для всех сервисов и инфраструктуры |
| `middleware/auth.py` | JWT аутентификация; при `jwt_secret=dev` — пропускает проверку (dev режим) |
| `middleware/logging.py` | Middleware для логирования каждого HTTP запроса (structlog) |
| `routes/documents.py` | `POST /api/v1/documents/upload`, `GET /api/v1/documents/{id}` |
| `routes/search.py` | `POST /api/v1/search` — Q&A с citations |
| `routes/validate.py` | `POST /api/v1/validate` — проверка нормативного соответствия |
| `routes/collections.py` | `GET /api/v1/collections/stats` — статистика 4 коллекций Qdrant |
| `routes/websocket.py` | `WS /ws/chat` — стриминг токенов LLM |

### `src/rag/llm/`
| Файл | Описание |
|---|---|
| `client.py` | httpx клиент для vLLM (OpenAI-compatible): `complete()` и `stream()` |

### `src/rag/pipeline/`
| Файл | Описание |
|---|---|
| `ingestion.py` | `IngestionService`: parse → chunk → embed → upsert в Qdrant |
| `retrieval.py` | `RetrievalService`: embed query → vector search → build context → LLM answer |
| `validation.py` | `ValidationService`: поиск по normative_base → LLM compliance check |

### `src/schemas/`
| Файл | Описание |
|---|---|
| `api.py` | Pydantic модели: `SearchRequest/Response`, `ValidateRequest/Response`, `DocumentUploadResponse`, `WsChatMessage`, `WsToken`, `WsCitation`, `WsDone`, `WsError` |

### `prompts/rag/`
| Файл | Описание |
|---|---|
| `qa.txt` | Промпт для Q&A с citations |
| `validate.txt` | Промпт для нормативной проверки документов |

### `tests/integration/`
| Файл | Описание |
|---|---|
| `test_api_endpoints.py` | 9 интеграционных тестов: upload, get, search, validate, collections stats |

---

## Изменённые файлы

### `pyproject.toml`
```
+ python-multipart>=0.0.18     # обязателен для FastAPI UploadFile
+ python-jose[cryptography]>=3.3.0  # JWT аутентификация
```

### `src/rag/config.py`
```
+ jwt_secret: str = "dev"      # "dev" → пропуск JWT проверки
+ jwt_algorithm: str = "HS256"
```

### `src/rag/embeddings/base.py`
```
+ async initialize() -> None   # хук инициализации (загрузка весов модели)
```

### `src/rag/vectorstore/operations.py`
```
# upsert_chunks: добавлен chunk.text в Qdrant payload
- payload=chunk.metadata.model_dump(exclude_none=True)
+ payload = chunk.metadata.model_dump(exclude_none=True)
+ payload["text"] = chunk.text
```

---

## Ограничения MVP

- Загрузка файлов: только `.txt`. PDF/DOCX — заглушка (production: Docling)
- Реестр документов: in-memory `dict` в `app.state`. Production: PostgreSQL
- Парсинг ответа LLM в `validation.py`: keyword heuristic. Production: structured JSON output
- Hybrid search (BM25 + RRF): не реализован (Sprint 4)
