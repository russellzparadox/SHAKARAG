# Schema RAG — multi-database

A retrieval-augmented generation (RAG) system over **your live SQL database**. It:

1. Introspects the full schema: tables, views, columns, types, defaults, PK/FK relations, indexes, comments.
2. Enriches it with Odoo's own metadata when the source is an Odoo Postgres DB (`ir_model` / `ir_model_fields`: model labels, field descriptions, help tooltips).
3. Chunks everything into per-table "schema cards" and embeds them into a local ChromaDB vector store.
4. Answers natural-language questions by retrieving relevant schema, generating **read-only SQL in your database's dialect**, executing it safely (dialect-appropriate read-only transaction + timeout + keyword/function guard), and synthesizing an answer with results + caveats.

## Supported databases

| Database | `DB_DIALECT` | Driver | Read-only enforcement |
|---|---|---|---|
| PostgreSQL (+ Odoo) | `postgres` | `psycopg[binary]` | `BEGIN TRANSACTION READ ONLY` ✓ |
| MySQL / MariaDB 8+ | `mysql` | `pymysql` | `START TRANSACTION READ ONLY` ✓ |
| SQL Server / Azure SQL | `mssql` | `pymssql` | explicit txn + always ROLLBACK ✓ |
| Snowflake | `snowflake` | SQLAlchemy URL | guard + read-only role recommended |
| BigQuery | `bigquery` | SQLAlchemy URL | inherently SELECT-only |
| Oracle | `oracle` | SQLAlchemy URL | guard + read-only user recommended |
| Redshift | `redshift` | SQLAlchemy URL | guard + read-only user recommended |
| Trino/Presto, ClickHouse, DuckDB, Databricks… | name it | SQLAlchemy URL | guard |

Install drivers as needed:

```bash
pip install pymysql cryptography          # MySQL/MariaDB
pip install pymssql                       # SQL Server
pip install sqlalchemy                    # warehouses
pip install snowflake-sqlalchemy          # + Snowflake
pip install sqlalchemy-bigquery           # + BigQuery
```

Configure via `.env` (per-database settings; keep one `COLLECTION` per database):

```bash
DB_DIALECT=mysql
DB_HOST=localhost
DB_PORT=3306
DB_USER=ro_user
DB_PASSWORD=...
DB_NAME=mydb
COLLECTION=mydb_schema
```

For warehouses, prefer a full SQLAlchemy `DB_URL` (host/port/user/password are ignored then):

```bash
DB_DIALECT=snowflake
DB_URL=snowflake://user:pass@account-xy/dbname/schema?warehouse=COMPUTE_WH
DB_URL=bigquery://my-project/my_dataset
DB_URL=clickhouse+native://user:pass@host:9000/db
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in DB_PASSWORD etc.
```

`.env` keys:

| Key | Default | Purpose |
|---|---|---|
| `DB_DIALECT` | `postgres` | `postgres`, `mysql`, `mssql`, `snowflake`, `bigquery`, `oracle`, `redshift`, `trino`, `clickhouse`, `duckdb`, `databricks` |
| `DB_URL` | — | Full SQLAlchemy URL — overrides host/port/user/pass for warehouse dialects |
| `DB_HOST` / `DB_PORT` | `localhost` / `5433` | Where Postgres is reachable (host port mapping is `5433->5432`) |
| `DB_USER` / `DB_PASSWORD` / `DB_NAME` | — | Credentials |
| `EMBED_PROVIDER` | `auto` | `auto`: OpenAI if `OPENAI_API_KEY` set, else local ONNX MiniLM (no API needed) |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `EMBED_MODEL` | — | Any OpenAI-compatible embeddings endpoint |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | — | Any OpenAI-compatible chat endpoint: 9Router (`http://localhost:20128/v1`), Ollama (`http://localhost:11434/v1`), LM Studio, vLLM, Groq, OpenAI… |
| `TOP_K`, `CONTEXT_CHAR_BUDGET` | `8`, `14000` | Retrieval tuning |
| `MAX_ROWS`, `STATEMENT_TIMEOUT_MS` | `100`, `15000` | Execution safety |

## Usage

### 1. Build the index

```bash
python scripts/ingest.py --drop --dump-catalog catalog.json
```

### 2. Ask questions (CLI)

```bash
# end-to-end (requires LLM_BASE_URL/LLM_MODEL configured)
python scripts/ask.py "which product categories have the most sold units?"

python scripts/ask.py "top 10 customers by total invoiced amount" --show-context

# just see what the retriever finds (works without any LLM)
python scripts/ask.py "where are sales orders stored" --dry-run --show-context

# generate SQL only, don't run it
python scripts/ask.py "count active partners per country" --no-exec

# machine readable
python scripts/ask.py "..." --json
```

### 3. HTTP API

```bash
uvicorn api.main:app --port 8100
curl -s localhost:8100/health
curl -s localhost:8100/ask -H 'content-type: application/json' \
     -d '{"question": "how many sale orders this month?"}'
curl -s localhost:8100/reindex -X POST -H 'content-type: application/json' -d '{"drop": true}'
```

### 4. Run as a container inside your compose network

The DB service in your `docker-compose.yaml` is reachable as `db` on `odoo_net`:

```bash
ODOO_NET_NAME=<your actual network name> docker compose -f docker-compose.rag.yml up -d --build
```

Find the real network name with `docker network ls` (compose usually prefixes it with the project folder name). Inside that network use `DB_HOST=db`, `DB_PORT=5432`.

## Safety model

- Generated SQL must pass a static guard: single statement, starts with `SELECT`/`WITH`, no DML/DDL keywords, no dangerous functions (`pg_sleep`, `lo_import`, `dblink`, …). String literals are stripped before scanning so values like `'delete'` don't false-positive.
- Execution runs in an explicit `BEGIN TRANSACTION READ ONLY` with `statement_timeout`.
- Results are capped at `MAX_ROWS`; large answers note truncation.

## Notes

- Row counts shown to the model are planner estimates (`reltuples`), good for ranking but approximate; exact counts come from executed `COUNT(*)`.
- Archived Odoo records have `active = false`; the SQL prompt tells the model to filter them when appropriate.
- The default local embedder downloads a small ONNX MiniLM model on first run (~90 MB).
- **Rotate the DB password if it was ever shared or committed**; keep `.env` out of git (already ignored).

## Tests

```bash
pytest tests/ -q
```
