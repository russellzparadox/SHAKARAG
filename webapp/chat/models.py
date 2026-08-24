from __future__ import annotations

from django.contrib.auth.models import User
from django.db import models


class DatabaseProfile(models.Model):
    class Dialect(models.TextChoices):
        POSTGRES = "postgres", "PostgreSQL"
        MYSQL = "mysql", "MySQL / MariaDB"
        MSSQL = "mssql", "SQL Server"
        SNOWFLAKE = "snowflake", "Snowflake"
        BIGQUERY = "bigquery", "BigQuery"
        ORACLE = "oracle", "Oracle"
        REDSHIFT = "redshift", "Amazon Redshift"
        TRINO = "trino", "Trino / Presto"
        CLICKHOUSE = "clickhouse", "ClickHouse"
        DUCKDB = "duckdb", "DuckDB"
        DATABRICKS = "databricks", "Databricks"

    class IndexStatus(models.TextChoices):
        NONE = "none", "Not indexed"
        INDEXING = "indexing", "Indexing…"
        READY = "ready", "Ready"
        ERROR = "error", "Error"

    name = models.CharField(max_length=120, unique=True)
    dialect = models.CharField(max_length=20, choices=Dialect.choices, default=Dialect.POSTGRES)
    host = models.CharField(max_length=255, blank=True)
    port = models.PositiveIntegerField(null=True, blank=True)
    db_user = models.CharField(max_length=120, blank=True)
    password_enc = models.TextField(blank=True)
    db_name = models.CharField(max_length=120, blank=True)
    db_url = models.CharField(
        max_length=500, blank=True,
        help_text="Full SQLAlchemy URL for warehouses. Overrides host/port/user/password.",
    )
    collection_name = models.SlugField(
        max_length=80, unique=True,
        help_text="Vector store collection. One per database.",
    )
    owner = models.ForeignKey(User, null=True, blank=True, on_delete=models.CASCADE, related_name="db_profiles")
    index_status = models.CharField(max_length=12, choices=IndexStatus.choices, default=IndexStatus.NONE)
    index_error = models.TextField(blank=True)
    indexed_at = models.DateTimeField(null=True, blank=True)
    indexed_vectors = models.PositiveIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return f"{self.name} ({self.get_dialect_display()})"


class LLMProfile(models.Model):
    name = models.CharField(max_length=120, unique=True)
    base_url = models.CharField(max_length=300)
    model = models.CharField(max_length=160)
    api_key_enc = models.TextField(blank=True)
    temperature = models.FloatField(default=0.1)
    owner = models.ForeignKey(User, null=True, blank=True, on_delete=models.CASCADE, related_name="llm_profiles")
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class ChatSession(models.Model):
    class Language(models.TextChoices):
        AUTO = "auto", "Auto"
        EN = "en", "English"
        FA = "fa", "فارسی (Persian)"
        AR = "ar", "العربية (Arabic)"
        DE = "de", "Deutsch"
        FR = "fr", "Français"
        ES = "es", "Español"
        TR = "tr", "Türkçe"
        ZH = "zh", "中文"
        RU = "ru", "Русский"

    title = models.CharField(max_length=200, default="New chat")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="chat_sessions")
    database = models.ForeignKey(DatabaseProfile, on_delete=models.PROTECT, related_name="sessions")
    llm = models.ForeignKey(LLMProfile, on_delete=models.PROTECT, related_name="sessions")
    language = models.CharField(max_length=8, choices=Language.choices, default=Language.AUTO)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title


class ChatMessage(models.Model):
    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=Role.choices)
    content = models.TextField()
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]


class QueryExample(models.Model):
    database = models.ForeignKey(DatabaseProfile, on_delete=models.CASCADE, related_name="examples")
    message = models.ForeignKey(ChatMessage, null=True, blank=True, on_delete=models.SET_NULL)
    question = models.TextField()
    sql = models.TextField()
    notes = models.TextField(blank=True, default="")
    rating = models.SmallIntegerField(choices=[(1, "helpful"), (-1, "not helpful")])
    active = models.BooleanField(default=True)
    created_by = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"[{self.rating:+d}] {self.question[:60]}"
