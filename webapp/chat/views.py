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
from django.views.decorators.http import require_POST
from django.views.generic import CreateView, DeleteView, DetailView, ListView, UpdateView

from .crypto import decrypt
from .forms import DatabaseProfileForm, LLMProfileForm, RegisterForm, SessionCreateForm
from .models import ChatMessage, ChatSession, DatabaseProfile, LLMProfile
from . import rag_service


def _access(user):
    from .models import UserAccess

    if not user.is_authenticated:
        return None
    access, _ = UserAccess.objects.get_or_create(user=user)
    return access


def visible_profiles(user, model):
    """All authenticated users may see and chat with every profile.
    Permissions only gate management (add/edit/delete/index)."""
    return model.objects.all()


def can_edit_profiles(user, model) -> bool:
    if user.is_superuser:
        return True
    access = _access(user)
    if model is DatabaseProfile:
        return bool(access and access.can_edit_databases)
    return bool(access and access.can_edit_llms)


def _editable(user, model):
    qs = visible_profiles(user, model)
    if can_edit_profiles(user, model):
        return qs
    return qs.filter(owner=user)


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

    # Conversation memory: last N turns (excluding the message just created) so
    # follow-up questions like "now show their codes" resolve against prior context.
    prior = (
        ChatMessage.objects.filter(session=session, id__lt=user_msg.id)
        .order_by("-id")
        .values("id", "role", "content", "meta")[:10]
    )
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in reversed(list(prior))
        if (m.get("meta") or {}).get("type") != "clarify"
    ]

    pending = session.pending_question
    if pending:
        question_for_rag = f"{pending}\n(User's clarification: {question})"
        # The pending question is the real entity anchor for a clarified turn —
        # make sure it leads the history so contextual retrieval resolves against it.
        session.pending_question = ""
        session.save(update_fields=["pending_question"])
        history = [{"role": "user", "content": pending}] + [
            h for h in history if h["content"] != pending
        ]
    else:
        question_for_rag = question

    try:
        result = rag_service.run_ask(
            session.database,
            session.llm,
            question_for_rag,
            answer_language=session.language,
            allow_clarify=session.auto_clarify and not bool(pending),
            history=history,
        )
    except Exception as exc:
        result = {"answer": f"Request failed: {exc}", "error": str(exc)}

    is_clarify = bool(result.get("clarify"))
    if is_clarify:
        session.pending_question = question_for_rag
        session.save(update_fields=["pending_question"])

    assistant_meta: dict = {
        "type": "clarify" if is_clarify else "answer",
        "options": result.get("options") or [],
    }
    for key in ("sql", "explanation", "columns", "rows", "row_count", "truncated", "tables_used", "error", "route", "doc_sources"):
        assistant_meta[key] = result.get(key)

    assistant_msg = ChatMessage.objects.create(
        session=session,
        role=ChatMessage.Role.ASSISTANT,
        content=(result.get("clarify_question") or result.get("answer") or "(no answer)")
        if is_clarify
        else (result.get("answer") or "(no answer)"),
        meta=assistant_meta,
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
        session.auto_clarify = request.POST.get("auto_clarify") == "on"
        session.save(update_fields=["language", "auto_clarify"])
        messages.success(request, "Chat settings updated.")
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
        return visible_profiles(self.request.user, DatabaseProfile)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["can_edit"] = can_edit_profiles(self.request.user, DatabaseProfile)
        ctx["editable_ids"] = set(_editable(self.request.user, DatabaseProfile).values_list("pk", flat=True))
        return ctx


class DbProfileCreateView(LoginRequiredMixin, CreateView):
    model = DatabaseProfile
    form_class = DatabaseProfileForm
    template_name = "chat/db_profile_form.html"
    success_url = reverse_lazy("chat:db_list")

    def dispatch(self, request, *args, **kwargs):
        if not can_edit_profiles(request.user, DatabaseProfile):
            messages.error(request, "You don't have permission to manage databases.")
            return redirect("chat:db_list")
        return super().dispatch(request, *args, **kwargs)

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
        return _editable(self.request.user, DatabaseProfile)


@login_required
def db_reindex(request, pk):
    dbp = get_object_or_404(visible_profiles(request.user, DatabaseProfile), pk=pk)
    if request.method == "POST":
        if dbp.index_status == DatabaseProfile.IndexStatus.INDEXING:
            messages.warning(request, "Indexing already running.")
        else:
            rag_service.start_reindex(dbp)
            messages.info(request, f"Indexing '{dbp.name}' started.")
    return redirect("chat:db_list")


@login_required
def db_status(request, pk):
    dbp = get_object_or_404(visible_profiles(request.user, DatabaseProfile), pk=pk)
    return JsonResponse(
        {
            "status": dbp.index_status,
            "error": dbp.index_error,
            "vectors": dbp.indexed_vectors,
            "indexed_at": dbp.indexed_at.strftime("%Y-%m-%d %H:%M") if dbp.indexed_at else None,
        }
    )


@login_required
@require_POST
def db_upload_doc(request, pk):
    """Upload a PDF/Word/Excel file; it gets extracted and indexed into this DB's collection."""
    dbp = get_object_or_404(visible_profiles(request.user, DatabaseProfile), pk=pk)
    upload = request.FILES.get("file")
    if not upload:
        return JsonResponse({"error": "No file provided."}, status=400)
    try:
        stats = rag_service.ingest_document(dbp, upload.name, upload.read())
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        return JsonResponse({"error": f"Indexing failed: {exc}"}, status=500)

    return JsonResponse(
        {
            "ok": True,
            "filename": stats["filename"],
            "kind": stats["kind"],
            "chunks": stats["chunks"],
            "total_vectors": stats["store_count"],
        }
    )


class LlmProfileListView(LoginRequiredMixin, ListView):
    model = LLMProfile
    template_name = "chat/llm_profiles.html"

    def get_queryset(self):
        return visible_profiles(self.request.user, LLMProfile)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["can_edit"] = can_edit_profiles(self.request.user, LLMProfile)
        ctx["editable_ids"] = set(_editable(self.request.user, LLMProfile).values_list("pk", flat=True))
        return ctx


class LlmProfileCreateView(LoginRequiredMixin, CreateView):
    model = LLMProfile
    form_class = LLMProfileForm
    template_name = "chat/llm_profile_form.html"
    success_url = reverse_lazy("chat:llm_list")

    def dispatch(self, request, *args, **kwargs):
        if not can_edit_profiles(request.user, LLMProfile):
            messages.error(request, "You don't have permission to manage LLMs.")
            return redirect("chat:llm_list")
        return super().dispatch(request, *args, **kwargs)

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
        return _editable(self.request.user, LLMProfile)


class AppLogoutView(LogoutView):
    next_page = reverse_lazy("login")


def _require_admin(view_func):
    def wrapper(request, *args, **kwargs):
        if not (request.user.is_authenticated and request.user.is_superuser):
            messages.error(request, "Admin access required.")
            return redirect("chat:sessions")
        return view_func(request, *args, **kwargs)

    return wrapper


@_require_admin
def user_admin(request):
    from django.contrib.auth.models import User

    from .models import DatabaseProfile, LLMProfile, UserAccess

    users = (
        User.objects.select_related("access")
        .order_by("username")
    )
    context = {"users": users}
    return render(request, "chat/users.html", context)


@_require_admin
def user_access_save(request, pk):
    from django.contrib.auth.models import User

    from .models import UserAccess

    user = get_object_or_404(User, pk=pk)
    if request.method == "POST":
        access, _ = UserAccess.objects.get_or_create(user=user)
        access.can_edit_databases = request.POST.get("can_edit_databases") == "on"
        access.can_edit_llms = request.POST.get("can_edit_llms") == "on"
        access.save()
        messages.success(request, f"Access updated for {user.username}.")
    return redirect("chat:user_admin")


@_require_admin
def user_create(request):
    from django.contrib.auth.models import User

    if request.method == "POST":
        username = (request.POST.get("username") or "").strip()
        password = request.POST.get("password") or ""
        email = (request.POST.get("email") or "").strip()
        if len(username) < 3 or len(password) < 8:
            messages.error(request, "Username needs 3+ chars, password 8+.")
            return redirect("chat:user_admin")
        if User.objects.filter(username=username).exists():
            messages.error(request, f"User '{username}' already exists.")
            return redirect("chat:user_admin")
        User.objects.create_user(username=username, email=email, password=password)
        messages.success(request, f"User '{username}' created.")
    return redirect("chat:user_admin")


@_require_admin
def user_toggle_active(request, pk):
    from django.contrib.auth.models import User

    user = get_object_or_404(User, pk=pk)
    if request.method == "POST" and user != request.user:
        user.is_active = not user.is_active
        user.save(update_fields=["is_active"])
        messages.success(
            request,
            f"{user.username} {'enabled' if user.is_active else 'disabled'}.",
        )
    return redirect("chat:user_admin")
