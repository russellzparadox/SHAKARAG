from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    db_dialect: str
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str
    db_url: str | None

    chroma_dir: Path
    collection: str
    embed_provider: str
    openai_api_key: str | None
    openai_base_url: str
    embed_model: str | None

    llm_base_url: str | None
    llm_model: str | None
    llm_api_key: str | None
    llm_temperature: float

    top_k: int
    context_char_budget: int

    max_rows: int
    statement_timeout_ms: int

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)


def load_settings() -> Settings:
    return Settings(
        db_dialect=_env("DB_DIALECT", "postgres"),
        db_host=_env("DB_HOST", "localhost"),
        db_port=_env_int("DB_PORT", 5433),
        db_user=_env("DB_USER", "shaka"),
        db_password=_env("DB_PASSWORD", ""),
        db_name=_env("DB_NAME", "shaka"),
        db_url=_env("DB_URL"),
        chroma_dir=Path(_env("CHROMA_DIR", ".chroma")),
        collection=_env("COLLECTION", "odoo_schema"),
        embed_provider=_env("EMBED_PROVIDER", "auto"),
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        embed_model=_env("EMBED_MODEL", "text-embedding-3-small"),
        llm_base_url=_env("LLM_BASE_URL"),
        llm_model=_env("LLM_MODEL"),
        llm_api_key=_env("LLM_API_KEY"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.1),
        top_k=_env_int("TOP_K", 8),
        context_char_budget=_env_int("CONTEXT_CHAR_BUDGET", 14000),
        max_rows=_env_int("MAX_ROWS", 100),
        statement_timeout_ms=_env_int("STATEMENT_TIMEOUT_MS", 15000),
    )
