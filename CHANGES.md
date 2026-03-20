# Changelog

## Sprint 3 — React Frontend

### Новые файлы

#### `frontend/` — React 18 + TypeScript + Vite + Tailwind + Zustand

| Файл | Описание |
|---|---|
| `src/types/api.ts` | TypeScript-интерфейсы, точно соответствующие `src/schemas/api.py` |
| `src/store/sessionStore.ts` | Zustand-стор: `sessionId`, список загруженных документов |
| `src/hooks/useChat.ts` | WebSocket-хук: стриминг токенов, citations, состояние соединения |
| `src/hooks/useDocuments.ts` | React Query: загрузка файлов через `multipart/form-data` |
| `src/hooks/useValidation.ts` | React Query mutation: нормативная проверка документа |
| `src/components/chat/ChatPanel.tsx` | Чат-интерфейс: textarea + кнопка отправки, статус WS |
| `src/components/chat/MessageBubble.tsx` | Пузырь сообщения (user / assistant + стриминг-курсор) |
| `src/components/chat/CitationCard.tsx` | Карточки источников под ответом ассистента |
| `src/components/documents/UploadPanel.tsx` | Drag-and-drop загрузка файлов (PDF, DOCX, TXT, до 50 МБ) |
| `src/components/validation/ValidationReport.tsx` | Отчёт о нарушениях (critical / warning / info) |
| `src/components/layout/Header.tsx` | Шапка: session ID, кнопка новой сессии |
| `src/components/layout/Sidebar.tsx` | Боковая панель: список документов + запуск валидации |

#### Конфигурация

| Файл | Описание |
|---|---|
| `vite.config.ts` | Tailwind plugin + proxy `/api/*` → `localhost:8000`, `/ws/*` → `ws://localhost:8000` |
| `src/index.css` | `@import "tailwindcss"` |

---

### Исправления backend

#### `src/api/routes/documents.py`
- Добавлена проверка пустого файла → `400 Bad Request` вместо `500`
- Добавлен `except ChunkingError` → `400 Bad Request` с текстом ошибки
- Ранее `ChunkingError` пробрасывался как необработанное исключение → `500 Internal Server Error`

---

### Исправленные баги

| Баг | Причина | Исправление |
|---|---|---|
| WebSocket не подключается | Хардкод `ws://localhost:8000` — обходил Vite proxy | URL из `window.location.host` |
| Чат блокируется после первого вопроса | `assistantIdRef.current` обнулялся до выполнения колбэка `setMessages` | Захват `id` до очистки ref |
| Upload 500 на пустом файле | `ChunkingError` не перехватывался в route handler | `try/except ChunkingError` → 400 |
| Несоответствие типов API | Frontend типы писались вручную и расходились с backend схемами | Перегенерированы по `src/schemas/api.py` |
| WsChatMessage неверный формат | Frontend слал `{type: 'chat', session_id: null}`, backend ждал `{query, session_id, collection}` | Исправлен формат, `session_id` обязателен |

---

## Sprint 2 — Document Parsing, Reranker, Query Router

### Новые файлы

#### `src/rag/parsing/`
| Файл | Описание |
|---|---|
| `base.py` | `DocumentParser` ABC: `parse()`, `supported_extensions()` |
| `txt.py` | `TxtParser` — UTF-8 с заменой невалидных символов |
| `pdf.py` | `PdfParser` — pypdf, постраничный парсинг, ошибка на пустом/отсканированном |
| `docx.py` | `DocxParser` — python-docx, параграфы + таблицы |
| `factory.py` | `get_parser(filename)` — реестр по расширению |

#### `src/common/storage.py`
- `MinIOStorage` — async обёртка: `initialize()`, `ensure_bucket()`, `upload()` → `"{bucket}/{doc_id}/{filename}"`

#### `src/rag/reranker/`
| Файл | Описание |
|---|---|
| `base.py` | `Reranker` ABC: `initialize()`, `rerank(query, hits, top_k)` |
| `cross_encoder.py` | `CrossEncoderReranker` — sentence-transformers CrossEncoder, `run_in_executor`, перезаписывает `hit["score"]` |

#### `src/rag/router.py`
- `QueryRouter.route(query, session_id)`: session_id → `current_package`, keyword-наборы для normative/precedents/templates, default → `current_package`

#### `tests/unit/`
| Файл | Тестов |
|---|---|
| `test_parsing.py` | 7 — factory routing, TxtParser, ошибки PDF/DOCX |
| `test_reranker.py` | 7 — мок модели: сортировка, top_k, пустой ввод, перезапись score, uninit |
| `test_router.py` | 8 — приоритет session, ключевые слова, default, пустой запрос |

---

## Sprint 1 — API Layer

### Новые файлы

#### `src/api/`
| Файл | Описание |
|---|---|
| `main.py` | FastAPI приложение, lifespan, CORS, роутинг |
| `deps.py` | FastAPI `Depends()` для всех сервисов |
| `middleware/auth.py` | JWT; `jwt_secret=dev` → dev bypass |
| `middleware/logging.py` | HTTP request logging (structlog) |
| `routes/documents.py` | `POST /upload`, `GET /{id}` |
| `routes/search.py` | `POST /search` |
| `routes/validate.py` | `POST /validate` |
| `routes/collections.py` | `GET /collections/stats` |
| `routes/websocket.py` | `WS /ws/chat` — стриминг токенов |

#### `src/rag/llm/client.py`
- httpx клиент vLLM: `complete()` и `stream()` (SSE)

#### `src/rag/pipeline/`
| Файл | Описание |
|---|---|
| `ingestion.py` | parse → MinIO → chunk → embed → upsert |
| `retrieval.py` | embed query → search → rerank → context → LLM |
| `validation.py` | normative_base search → LLM compliance check |

#### `prompts/rag/`
- `qa.txt`, `validate.txt`

#### `tests/integration/test_api_endpoints.py`
- 9 тестов: health, upload, get, search, validate, stats

### Исправления Sprint 1

| Проблема | Исправление |
|---|---|
| `.env` inline comments ломали значения | Убраны все inline комментарии |
| Qdrant 401 Unauthorized | `https=self._config.qdrant_prefer_grpc` вместо `https=bool(api_key)` |
| `qdrant-client 1.17` API change | `.search()` → `.query_points()`, результат `.points` |
| Reranker model not found | Добавлен суффикс `-v1` |
