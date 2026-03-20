# Changelog

## [Unreleased]

---

## Sprint 5 — Парсинг документов + Compliance Validator + Reranker

### Парсинг документов
- **PdfParser**: страницы теперь разделяются символом `\x0c` (form-feed) — чанкеры могут определять номер страницы по смещению в тексте
- **DocxParser**: исправлен порядок извлечения — параграфы и таблицы теперь идут в порядке документа (раньше таблицы шли после всех параграфов); таблицы рендерятся в формате Markdown с заголовком и разделительной строкой

### Стратегии чанкинга
- **TableAwareChunkingStrategy** — новая стратегия для финансовой отчётности:
  - Markdown-таблицы (строки с `|`) остаются единым чанком
  - Текстовые параграфы объединяются до достижения `max_chars`
  - Номера страниц трекаются через `\x0c`-маркеры из PdfParser
- **ChunkingStrategyFactory**: `FINANCIAL_REPORT` теперь использует `TableAwareChunkingStrategy` вместо `FixedSizeChunkingStrategy`

### Compliance Validator
- **`prompts/rag/validate.txt`**: промпт переписан — LLM возвращает структурированный JSON `{"issues": [...], "summary": "..."}`
- **`ValidationService._parse_issues`**: основной путь — парсинг JSON (severity/article/violation/recommendation); fallback — keyword-эвристика при ошибке парсинга
- **`_extract_summary`**: извлекает поле `summary` из JSON-ответа
- **`_find_citation`**: матчит цитату по артикулу статьи, а не просто берёт первый хит
- **Фикс**: заменён `str.format()` на `str.replace()` — JSON-скобки в промпте больше не ломают форматирование

### Frontend — ValidationReport
- **Статус-баннер**: цветовая индикация трёх состояний (compliant / non_compliant / review_required)
- **SeveritySummary**: счётчики нарушений по типу (critical / warning / info)
- **Чипы статей**: список проверенных нормативных статей
- **IssueCard**: severity badge + артикул в `font-mono` + раскрываемый `CitationBlock` с текстом источника и % релевантности
- **Sidebar**: per-document индикатор загрузки, панель ошибок с кнопкой закрытия, отчёт занимает нижнюю часть сайдбара

### Тесты
- `test_parsing.py` — тесты DOCX (порядок документа, markdown-таблицы, header separator), page separator константа
- `test_table_aware_chunking.py` — 8 тестов: детекция таблиц, пустой документ, смешанный контент, page tracking, metadata
- `test_validation_parser.py` — 16 тестов: JSON-парсинг, множественные severity, fallback-эвристика, `_find_citation`, `_extract_summary`
- `test_retrieval_service.py` — 9 тестов: reranker вызывается с `candidate_limit`, `top_k` соответствует запросу, disabled-путь не вызывает reranker, session filter

**Итого: 101 unit-тест, все проходят.**

---

## Sprint 4 — Hybrid Search + Chunking + PostgreSQL + Eval

### Hybrid Search
- BM25 sparse vectors через tokenizer (TF-нормализация, token ID как индекс)
- Qdrant `Prefetch` (dense + sparse) → `FusionQuery(RRF)` для слияния результатов
- `hybrid_search_enabled: bool = True` в конфиге

### Стратегии чанкинга
- **HierarchicalChunkingStrategy**: regex-сплит по артикулам (`Статья`, `Пункт`, `Раздел`, `Глава`, `п.`, `ст.`, `Art.`); fallback на параграфы при отсутствии маркеров
- **SemanticChunkingStrategy**: параграф-ориентированный сплит с merge коротких параграфов
- **ChunkingStrategyFactory**: `normative → hierarchical`, `contract → semantic`, остальные → `fixed_size`

### PostgreSQL
- `DocumentRecord` ORM (SQLAlchemy 2.0 async + asyncpg)
- `DocumentRepository`: `save`, `get_by_id`, `list_by_session`
- Non-fatal инициализация в lifespan — API работает без БД

### RAGAS Eval
- `compute_mrr`, `compute_recall`, `compute_precision` — чистые функции
- `ragas_eval.py` — harness для запуска оценки: читает `data/ground_truth/eval_dataset.json`, пишет результаты в `data/eval_results/`
- Seed dataset: 5 запросов по кредитной документации

---

## Sprint 3 — Frontend

- React 18 + TypeScript + Vite + Tailwind + Zustand
- **ChatPanel**: WebSocket стриминг токенов, citations, StrictMode-совместимый hook
- **UploadPanel**: drag-and-drop, мультизагрузка, error handling
- **Sidebar**: список документов, кнопка валидации
- **Header**: session ID, кнопка новой сессии
- Vite proxy: `/api/*` → `:8000`, `/ws/*` → `ws://localhost:8000`

---

## Sprint 1–2 — Backend Core

- FastAPI + async/await, структурированные логи (structlog)
- Qdrant: 4 коллекции (`normative_base`, `deal_precedents`, `reference_templates`, `current_package`)
- E5Large embedding provider (multilingual, CPU, 1024 dim)
- Cross-encoder reranker (mmarco-mMiniLMv2-L12-H384, top-50 → top-5)
- Query router: авто-определение коллекции по типу запроса
- MinIO: хранение оригиналов документов
- JWT middleware, CORS
- WebSocket endpoint для стриминга LLM-ответов
