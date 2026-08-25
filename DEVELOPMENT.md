# SHAKARAG — Development Guide & Architecture Report

A retrieval-augmented system that answers natural-language questions against **live SQL databases**
(PostgreSQL/Odoo, MySQL, SQL Server, warehouses) by retrieving schema knowledge, generating
read-only queries, executing them safely, and explaining the results — exposed via CLI, FastAPI,
and a multi-user Django web app ("liquid glass" UI).

---

## 1. Repository map

```
rag/
├── rag/                      # Core library (framework-agnostic)
│   ├── config.py             #   Env-driven Settings dataclass (single source of config)
│   ├── db.py                 #   Postgres connection + read-only executor (psycopg)
│   ├── serializers.py        #   DB-value → JSON-safe conversion; row_to_list(dict|tuple)
│   ├── introspect.py         #   Postgres catalog → TableRecord/Column/ForeignKey models
│   ├── enrich.py             #   Odoo ir_model / ir_model_fields metadata (labels, help)
│   ├── warehouse.py          #   Fact/dim/bridge/relation classifier + value-column picker
│   ├── chunking.py           #   TableRecord list → embeddable "schema card" chunks
│   ├── embeddings.py         #   Embedding providers: OpenAI-compatible or local ONNX MiniLM
│   ├── store.py              #   ChromaDB wrapper (upsert/query/reset, embedder-tag guard)
│   ├── rerank.py             #   Heuristic re-scoring of retrieved chunks (entity/intent/role)
│   ├── examples.py           #   ExampleStore: per-database verified Q→SQL few-shot memory
│   ├── llm.py                #   OpenAI-compatible chat client + robust JSON extraction
│   ├── sqlguard.py           #   Static read-only validation (keywords, functions, literals)
│   ├── pipeline.py           #   RagPipeline: retrieve → clarify → SQL gen → run → repair → answer
│   └── dialects/             #   One adapter per database family
│       ├── base.py           #     BaseDialect ABC (introspect/execute_readonly/sample_values)
│       ├── postgres.py       #     pg_catalog introspection, BEGIN READ ONLY
│       ├── mysql.py          #     information_schema, START TRANSACTION READ ONLY
│       ├── sqlserver.py      #     sys catalogs, explicit txn + ROLLBACK
│       └── generic.py        #     Any SQLAlchemy URL (Snowflake/BQ/Oracle/Trino/ClickHouse…)
├── scripts/
│   ├── ingest.py             #   Index builder (introspect → classify → sample → chunk → upsert)
│   └── ask.py                #   CLI question interface (--dry-run/--no-exec/--json)
├── api/main.py               #   Thin FastAPI service (/health /ask /reindex /debug/context)
├── tests/                    #   pytest suite (44 tests)
│   ├── test_sqlguard.py      #   Read-only enforcement
│   ├── test_chunking.py      #   Card building/splitting
│   ├── test_dialects.py      #   Registry aliases, MySQL/MSSQL catalog builders
│   ├── test_warehouse.py     #   Role classifier, example store, value candidates
│   ├── test_pipeline_learning.py  # Few-shot injection, execution-error self-repair
│   ├── test_rerank.py        #   Entity/intent/staging ranking rules
│   └── test_serializers.py   #   dict-row regression (the "TABLE_SCHEMA bug")
├── webapp/                   # Django 5/6 project
│   ├── manage.py
│   ├── webapp/settings.py    #   SQLite dev DB, LOGGING (logs/webapp.log + errors.log)
│   └── chat/
│       ├── models.py         #   DatabaseProfile LLMProfile ChatSession ChatMessage
│       │                     #   QueryExample UserAccess
│       ├── crypto.py         #   Fernet-at-rest for passwords/API keys (key from SECRET_KEY)
│       ├── forms.py          #   Register/Database/LLM/Session forms
│       ├── views.py          #   Auth, sessions, chat send/feedback/language,
│       │                     #   profile CRUD, user admin; visibility/edit helpers
│       ├── rag_service.py    #   Django↔core bridge: settings build, pipeline cache,
│       │                     #   background indexing thread, feedback sync
│       ├── admin.py  urls.py  tests.py (28 tests)
│       ├── management/commands/seed_defaults.py
│       ├── templates/chat/*.html        # base + pages (glass UI)
│       ├── templates/registration/login.html
│       └── static/chat/{style.css,chat.js,poll.js}
├── .env                      # Secrets (gitignored) — DB creds, LLM endpoint
├── .zed/{settings,debug,tasks}.json     # Editor: pylsp, debugpy launch configs
├── docker-compose.rag.yml    # Optional containerized API service
├── requirements.txt
└── DEVELOPMENT.md            # This file
```

---

## 2. Data flow (the whole system in one picture)

```
                    INGEST (scripts/ingest.py, ~60s per large DB)
Postgres/MySQL/MSSQL/…
  │ dialect.introspect()            tables, columns, PK/FK, indexes, comments, view defs
  │ warehouse.classify_tables()     FACT / DIMENSION / RELATION / BRIDGE roles + grain hints
  │ dialect.sample_values()         real distinct values for low-cardinality text columns
  │ enrich.fetch_odoo_metadata()    Odoo labels/help (postgres only, auto-skip elsewhere)
  ▼
chunking.build_chunks()             one "schema card" per table; big tables split
  ▼ embeddings → ChromaDB           collection <name>  (+ <name>-examples created lazily)

                    ASK (pipeline.ask)
question ──embed──▶ top-3k candidates ──rerank.rerank_hits──▶ top-k context
                          │  (entity match · count/agg intent · role boost · staging penalty)
                          ▼
              [clarify gate]  LLM decides: ambiguous? → {"action":"clarify",
                              question,options} → ASK USER (session.pending_question)
                          ▼
              [few-shot] ExampleStore.search(question) → inject VERIFIED EXAMPLES
                          ▼
              LLM → {"sql","explanation"} ── sqlguard.validate_sql ──✗ retry ≤2 w/ error
                          ▼
              dialect.execute_readonly() ──✗ feed DB error back to LLM (one repair)
                          ▼
              rows ── LLM synthesize_answer(language=…) ──▶ answer + Query used + Notes

                    FEEDBACK (👍 in UI)
assistant msg + question ──▶ ExampleStore.add()  ← permanent few-shot for this database
```

---

## 3. Core concepts you must know before touching code

### 3.1 TableRecord / Column (`rag/introspect.py`)
The universal schema model every dialect fills:
`TableRecord{schema,name,kind('r'|'v'),row_estimate,comment,columns[],primary_key[],
foreign_keys[],referenced_by[],unique_constraints[],indexes[],view_def,warehouse_role,role_reason}`.
`kind_label`, `qualified` are derived. Chunking consumes these objects only — **adding a dialect
never touches chunking**.

### 3.2 Dialects (`rag/dialects/`)
Registry `get_dialect(settings)` maps `DB_DIALECT` (+aliases like mssql/bq/mariadb) to a class.
Each must implement:
- `introspect() -> dict[(schema,table), TableRecord]`
- `execute_readonly(sql,max_rows) -> (columns,rows,truncated)` — **defense layer 3**
- optional `sample_values(schema,table,column,k)`
- `prompt_hints()` — engine-specific syntax guidance injected into the SQL prompt

Rules learned the hard way (keep them):
- Always alias information_schema columns explicitly (MySQL returns `NON_UNIQUE` otherwise).
- Rows may be dicts (`as_dict=True`) — use `serializers.row_to_list`, never `tuple(row)` blindly.
- Anchor nothing to CWD: paths come from `config._REPO_ROOT`.

### 3.3 Safety model (three layers, all must pass)
1. `sqlguard.validate_sql` — strips comments/literals, requires SELECT/WITH start, blocks DML/DDL
   keywords & dangerous functions, single statement.
2. Guard-retry: guard failures are fed back to the LLM (≤2 attempts).
3. Execution sandbox per dialect (READ ONLY txn / txn+ROLLBACK) + `statement_timeout` +
   `MAX_ROWS` fetch cap. For warehouses without read-only txns: use a read-only role.

### 3.4 Learning loop
- 👍 on an assistant message → `QueryExample` row + `ExampleStore.add()` into Chroma collection
  `<collection>-examples` (separate from schema index → re-indexing never wipes memory).
- Similar future questions get the example injected as "VERIFIED QUERY EXAMPLES" (few-shot).
- 👎 removes it. IDs are deterministic (sha256 of normalized question) → re-rating = upsert.

### 3.5 Warehouse intelligence
- Roles are heuristic (names, FK counts, measure/date columns, row ratios) and rendered on cards:
  `Warehouse role: FACT … Grain: one row per combination of (...)`.
- Value samples appear as `— Values seen: draft, done, cancel`.
- `rerank.py` fixes pure-vector mistakes: CamelCase entity matching (`companies→DimCompany`),
  count-intent boosts dimensions, aggregation boosts facts, ETL/vw/stage demotion.

---

## 4. Configuration reference (`.env`)

| Var | Default | Notes |
|---|---|---|
| DB_DIALECT | postgres | postgres/mysql/mssql/snowflake/bigquery/oracle/redshift/trino/clickhouse/duckdb/databricks |
| DB_HOST/PORT/USER/PASSWORD/NAME | — | ignored when DB_URL set |
| DB_URL | — | full SQLAlchemy URL for warehouses |
| CHROMA_DIR | .chroma | resolved against repo root, never CWD |
| COLLECTION | odoo_schema | one index per database |
| EMBED_PROVIDER | auto | auto→OpenAI if key else local ONNX MiniLM |
| OPENAI_API_KEY/BASE_URL, EMBED_MODEL | — | embeddings endpoint |
| LLM_BASE_URL/LLM_MODEL/API_KEY | — | any OpenAI-compatible endpoint (9Router/Ollama/OpenAI…) |
| TOP_K / CONTEXT_CHAR_BUDGET | 8 / 14000 | retrieval size |
| MAX_ROWS / STATEMENT_TIMEOUT_MS | 100 / 15000 | execution caps |
| SAMPLE_VALUES / VALUE_SAMPLE_MAX_ROWS | 1 / 200000 | value sampling switch+limit |
| EXAMPLES_TOP_K | 2 | few-shots injected |
| DJANGO_SECRET_KEY | dev value | derives Fernet keys — SET IN PROD |

> The web app reads DB connection details from **DatabaseProfile rows**, not from `.env`;
> `.env` is used for embeddings, LLM defaults and CLI runs. Collection names live on profiles.

---

## 5. Web app internals (`webapp/`)

### Models
- `DatabaseProfile` — dialect, conn fields (`password_enc` Fernet), `collection_name` (unique),
  owner (nullable ⇒ public), index_status/index_error/indexed_at/indexed_vectors.
- `LLMProfile` — base_url/model/api_key_enc/temperature.
- `ChatSession` — user+database+llm+language+auto_clarify+pending_question (clarify state).
- `ChatMessage` — role/content/meta(JSON: type,sql,columns,rows,tables_used,error,options…).
- `QueryExample` — audit copy of 👍 examples (rating ±1, active).
- `UserAccess` — per-user `can_edit_databases` / `can_edit_llms` (management flags).

### Access model (current policy)
- **Visibility**: every authenticated user sees ALL profiles (chatting is open).
- **Management**: add/edit/delete/index gated by superuser OR owner OR UserAccess flag;
  buttons hidden server-side too (CreateView.dispatch guards).
- Helpers in `views.py`: `_access(user)`, `visible_profiles(user,model)` (=all today),
  `can_edit_profiles(user,model)`, `_editable(user,model)`.

### Request flows worth knowing
- `chat_send`: pending-question merge → `rag_service.run_ask(allow_clarify=session.auto_clarify
  and not pending)` → if clarify: save `pending_question`, meta.type="clarify".
- `chat_feedback`: finds preceding user message, upserts QueryExample, mirrors into ExampleStore.
- `_reindex_worker` (thread): sets INDEXING → `run_ingest(drop=True, settings=built-from-profile)`
  → READY + vectors/timezone-aware timestamp; failures persist traceback into index_error.
- Pipeline cache key: `(dbp.pk, collection, dbp.updated_at_ms, llmp.pk, llmp.updated_at_ms)`.

### Frontend
Single CSS file (`static/chat/style.css`) implementing the liquid-glass design system:
aurora blobs + grain backdrop, glass panels (`backdrop-filter`), caustic conic rim on `.chat-main`,
teal reserved for answers/focus, Sora/Instrument Sans/Vazirmatn/JetBrains Mono.
JS: `chat.js` (send/clarify chips/feedback/rendering), `poll.js` (index-status polling).
All buttons are `.btn` flex pills — do not reintroduce bare inline-block styling.

---

## 6. Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # fill DB_PASSWORD, LLM_* etc.

# core CLI
python scripts/ingest.py --drop --dump-catalog catalog.json
python scripts/ask.py "how many orders?" --show-context

# Django web app
cd webapp && ../.venv/bin/python manage.py migrate
../.venv/bin/python manage.py seed_defaults        # shared profiles from .env
../.venv/bin/python manage.py createsuperuser
../.venv/bin/python manage.py runserver 8000

# FastAPI alternative
uvicorn api.main:app --port 8100
```

### Testing
```bash
pytest tests/ -q                                  # core (44)
cd webapp && ../.venv/bin/python manage.py test chat   # django (28)
```
When adding features, extend the closest existing file; catalog builders and rerank rules are
pure functions — test without servers.

### Debugging (Zed)
`.zed/debug.json` has debugpy configs: Django runserver (8000/8111, `--noreload`), Django tests,
ingest script, attach-on-5678. `.zed/tasks.json` wires migrate/tests/ingest commands.
Project LSP is pylsp+jedi pointed at `.venv` (per user global preference), ruff formats.

### Logs
`webapp/logs/webapp.log` (INFO: asks, indexing, feedback) and `errors.log` (ERROR only),
plus console. Index failures persist full tracebacks onto the profile's `index_error`.

---

## 7. HTTP APIs

FastAPI (`api/main.py`, port 8100): `GET /health` · `POST /ask {question,execute,dry_run,top_k}` ·
`POST /reindex {drop}` · `GET /debug/context?q=`.

Django (port 8000, prefix `/`):
`register/ login/ logout/` · `chat/` `chat/<pk>/` `chat/<pk>/send/` `chat/<pk>/language/`
`chat/<pk>/feedback/` · `db/` CRUD + `<pk>/(edit|delete|reindex|status)` ·
`llm/` CRUD · `users/` admin (+access/create/toggle-active).

---

## 8. Extension recipes

**Add a database dialect**: subclass `BaseDialect` in `dialects/`, implement `introspect()` +
`execute_readonly()` (+`sample_values` if cheap), register canonical name in `dialects/__init__.py`,
add label/hints, install driver, unit-test the catalog builder with canned rows.

**Add a profile field** (e.g. SSH tunnel): models → migration → form fields → `build_rag_settings`
in `rag_service.py` → dialect connect. Remember cache-key invalidation comes free via updated_at.

**Change prompting**: SYSTEM_SQL/SYSTEM_CLARIFY/SYSTEM_ANSWER in `rag/pipeline.py` use
`__TOKEN__` replacement via `render_sql_system` — never `.format()` (literal JSON braces!).

**Tune accuracy**: raise TOP_K/CONTEXT_CHAR_BUDGET → upgrade embedder (tag change forces
re-index automatically) → thumbs-up good answers → add COMMENT ON TABLE/COLUMN in source DBs.

## 9. Known limitations & roadmap candidates
- Generic dialect relies on SQLAlchemy reflection quality per backend (some lack comments/rows).
- Clarify gate costs one extra LLM call per message (skipped when a verified example matches).
- No streaming responses; answers arrive as one payload.
- Single SQLite tenant DB — swap `DATABASES` for Postgres before real multi-user deployment.
- MCP server / Playwright E2E identified as next integrations (skills available in repo notes).
