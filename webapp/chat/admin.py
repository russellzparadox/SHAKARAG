from django.contrib import admin

from .models import ChatMessage, ChatSession, DatabaseProfile, LLMProfile


@admin.register(DatabaseProfile)
class DatabaseProfileAdmin(admin.ModelAdmin):
    list_display = ("name", "dialect", "host", "db_name", "collection_name", "index_status", "indexed_vectors")
    list_filter = ("dialect", "index_status")
    search_fields = ("name", "host", "db_name")


@admin.register(LLMProfile)
class LLMProfileAdmin(admin.ModelAdmin):
    list_display = ("name", "base_url", "model", "temperature")


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ("role", "content", "created_at")


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "database", "llm", "created_at")
    inlines = [ChatMessageInline]
