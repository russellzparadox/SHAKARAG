from __future__ import annotations

import json

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LogoutView
from django.db.models import Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse, reverse_lazy
from django.views import View
from django.views.generic import CreateView, DeleteView, DetailView, ListView, UpdateView

from .crypto import decrypt
from .forms import DatabaseProfileForm, LLMProfileForm, RegisterForm, SessionCreateForm
from .models import ChatMessage, ChatSession, DatabaseProfile, LLMProfile
from . import rag_service


def _editable(user, model):
    if user.is_superuser:
        return model.objects.all()
    return model.objects.filter(Q(owner=user) | Q(owner__isnull=True))


def home(request):
    if request.user.is_authenticated:
        return redirect("chat:sessions")
    return redirect("login")


class RegisterView(CreateView):
    form_class = RegisterForm
    template_name = "chat/register.html"
    success_url = reverse_lazy("chat:sessions")

    def form_valid(self, form):
        response = super().form_valid(form)
        login(self.request, self.object)
        messages.success(self.request, f"Welcome, {self.object.username}!")
        return response


class SessionsView(LoginRequiredMixin, View):
    template_name = "chat/sessions.html"

    def get(self, request):
        sessions = ChatSession.objects.filter(user=request.user).select_related("database", "llm")
        form = SessionCreateForm(user=request.user)
        return render(request, self.template_name, {"sessions": sessions, "form": form})

    def post(self, request):
        form = SessionCreateForm(request.POST, user=request.user)
        if not form.is_valid():
            messages.error(request, "Pick a database and an LLM to start a chat.")
            return redirect("chat:sessions")
        session = form.save(commit=False)
        session.user = request.user
        if not session.title:
            session.title = "New chat"
        session.save()
        return redirect("chat:detail", pk=session.pk)


class ChatDetailView(LoginRequiredMixin, DetailView):
    model = ChatSession
    template_name = "chat/chat_detail.html"

    def get_queryset(self):
        return ChatSession.objects.filter(user=self.user_ok())

    def user_ok(self):
        return self.request.user


@login_required
def chat_send(request, pk):
    session = get_object_or_404(ChatSession.objects.filter(user=request.user), pk=pk)
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        payload = json.loads(request.body or "{}")
        question = (payload.get("question") or "").strip()
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    if not question:
        return JsonResponse({"error": "Empty question"}, status=400)

    user_msg = ChatMessage.objects.create(session=session, role=ChatMessage.Role.USER, content=question)

    try:
        result = rag_service.run_ask(session.database, session.llm, question, answer_language=session.language)
    except Exception as exc:
        result = {"answer": f"Request failed: {exc}", "error": str(exc)}

    assistant_msg = ChatMessage.objects.create(
        session=session,
        role=ChatMessage.Role.ASSISTANT,
        content=result.get("answer") or "(no answer)",
        meta={
            k: result.get(k)
            for k in ("sql", "explanation", "columns", "rows", "row_count", "truncated", "tables_used", "error")
        },
    )
    return JsonResponse({"user": _msg_json(user_msg), "assistant": _msg_json(assistant_msg)})


@login_required
def chat_language(request, pk):
    session = get_object_or_404(ChatSession.objects.filter(user=request.user), pk=pk)
    if request.method == "POST":
        lang = request.POST.get("language", "auto")
        valid = [c[0] for c in ChatSession.Language.choices]
        if lang in valid:
            session.language = lang
            session.save(update_fields=["language"])
            messages.success(request, "Answer language updated.")
    return redirect("chat:detail", pk=session.pk)


@login_required
def chat_feedback(request, pk):
    session = get_object_or_404(ChatSession.objects.filter(user=request.user), pk=pk)
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        payload = json.loads(request.body or "{}")
        message_id = int(payload.get("message_id") or 0)
        value = payload.get("value")
    except (json.JSONDecodeError, TypeError, ValueError):
        return JsonResponse({"error": "Invalid payload"}, status=400)
    if value not in ("up", "down"):
        return JsonResponse({"error": "value must be up/down"}, status=400)

    msg = get_object_or_404(
        ChatMessage.objects.filter(session=session, role=ChatMessage.Role.ASSISTANT),
        pk=message_id,
    )
    meta = msg.meta or {}
    if not meta.get("sql"):
        return JsonResponse({"error": "No SQL on this message to learn from"}, status=400)
    question = (
        ChatMessage.objects.filter(session=session, role=ChatMessage.Role.USER, pk__lt=msg.pk)
        .order_by("-pk")
        .values_list("content", flat=True)
        .first()
    )
    if not question:
        return JsonResponse({"error": "No matching question found"}, status=400)

    from .models import QueryExample

    example, created = QueryExample.objects.get_or_create(
        database=session.database,
        message=msg,
        defaults={
            "question": question,
            "sql": meta["sql"],
            "notes": meta.get("explanation") or "",
            "rating": 1 if value == "up" else -1,
            "active": value == "up",
            "created_by": request.user,
        },
    )
    if not created:
        example.rating = 1 if value == "up" else -1
        example.active = value == "up"
        example.save(update_fields=["rating", "active"])

    outcome = rag_service.record_feedback(
        session.database,
        question=question,
        sql=meta["sql"],
        notes=meta.get("explanation") or "",
        helpful=(value == "up"),
    )
    return JsonResponse({"status": outcome, "value": value})


def _msg_json(msg: ChatMessage) -> dict:
    return {
        "id": msg.pk,
        "role": msg.role,
        "content": msg.content,
        "at": msg.created_at.strftime("%H:%M"),
        "meta": msg.meta or {},
    }


class DbProfileListView(LoginRequiredMixin, ListView):
    model = DatabaseProfile
    template_name = "chat/db_profiles.html"

    def get_queryset(self):
        return DatabaseProfile.objects.all()


class DbProfileCreateView(LoginRequiredMixin, CreateView):
    model = DatabaseProfile
    form_class = DatabaseProfileForm
    template_name = "chat/db_profile_form.html"
    success_url = reverse_lazy("chat:db_list")

    def form_valid(self, form):
        form.instance.owner = self.request.user
        messages.success(self.request, "Database profile created. Index it to enable chat.")
        return super().form_valid(form)


class DbProfileUpdateView(LoginRequiredMixin, UpdateView):
    model = DatabaseProfile
    form_class = DatabaseProfileForm
    template_name = "chat/db_profile_form.html"
    success_url = reverse_lazy("chat:db_list")

    def get_queryset(self):
        return _editable(self.request.user, DatabaseProfile)

    def form_valid(self, form):
        messages.success(self.request, "Saved. Re-index if connection details changed.")
        return super().form_valid(form)


class DbProfileDeleteView(LoginRequiredMixin, DeleteView):
    model = DatabaseProfile
    template_name = "chat/db_confirm_delete.html"
    success_url = reverse_lazy("chat:db_list")

    def get_queryset(self):
        return DatabaseProfile.objects.filter(owner=self.request.user)


@login_required
def db_reindex(request, pk):
    dbp = get_object_or_404(DatabaseProfile.objects.all(), pk=pk)
    if request.method == "POST":
        if dbp.index_status == DatabaseProfile.IndexStatus.INDEXING:
            messages.warning(request, "Indexing already running.")
        else:
            rag_service.start_reindex(dbp)
            messages.info(request, f"Indexing '{dbp.name}' started.")
    return redirect("chat:db_list")


@login_required
def db_status(request, pk):
    dbp = get_object_or_404(DatabaseProfile.objects.all(), pk=pk)
    return JsonResponse(
        {
            "status": dbp.index_status,
            "error": dbp.index_error,
            "vectors": dbp.indexed_vectors,
            "indexed_at": dbp.indexed_at.strftime("%Y-%m-%d %H:%M") if dbp.indexed_at else None,
        }
    )


class LlmProfileListView(LoginRequiredMixin, ListView):
    model = LLMProfile
    template_name = "chat/llm_profiles.html"

    def get_queryset(self):
        return LLMProfile.objects.all()


class LlmProfileCreateView(LoginRequiredMixin, CreateView):
    model = LLMProfile
    form_class = LLMProfileForm
    template_name = "chat/llm_profile_form.html"
    success_url = reverse_lazy("chat:llm_list")

    def form_valid(self, form):
        form.instance.owner = self.request.user
        return super().form_valid(form)


class LlmProfileUpdateView(LoginRequiredMixin, UpdateView):
    model = LLMProfile
    form_class = LLMProfileForm
    template_name = "chat/llm_profile_form.html"
    success_url = reverse_lazy("chat:llm_list")

    def get_queryset(self):
        return _editable(self.request.user, LLMProfile)


class LlmProfileDeleteView(LoginRequiredMixin, DeleteView):
    model = LLMProfile
    template_name = "chat/llm_confirm_delete.html"
    success_url = reverse_lazy("chat:llm_list")

    def get_queryset(self):
        return LLMProfile.objects.filter(owner=self.request.user)


class AppLogoutView(LogoutView):
    next_page = reverse_lazy("login")
