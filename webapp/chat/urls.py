from django.urls import path

from . import views

app_name = "chat"

urlpatterns = [
    path("", views.home, name="home"),
    path("register/", views.RegisterView.as_view(), name="register"),
    path("chat/", views.SessionsView.as_view(), name="sessions"),
    path("chat/<int:pk>/", views.ChatDetailView.as_view(), name="detail"),
    path("chat/<int:pk>/send/", views.chat_send, name="send"),
    path("chat/<int:pk>/language/", views.chat_language, name="language"),
    path("chat/<int:pk>/feedback/", views.chat_feedback, name="feedback"),
    path("db/", views.DbProfileListView.as_view(), name="db_list"),
    path("db/new/", views.DbProfileCreateView.as_view(), name="db_new"),
    path("db/<int:pk>/edit/", views.DbProfileUpdateView.as_view(), name="db_edit"),
    path("db/<int:pk>/delete/", views.DbProfileDeleteView.as_view(), name="db_delete"),
    path("db/<int:pk>/reindex/", views.db_reindex, name="db_reindex"),
    path("db/<int:pk>/status/", views.db_status, name="db_status"),
    path("db/<int:pk>/icrl/rebuild/", views.db_icrl_rebuild, name="db_icrl_rebuild"),
    path("db/<int:pk>/icrl/status/", views.db_icrl_status, name="db_icrl_status"),
    path("db/<int:pk>/icrl/sample/", views.db_icrl_sample, name="db_icrl_sample"),
    path("db/<int:pk>/docs/", views.db_upload_doc, name="db_upload_doc"),
    path("llm/", views.LlmProfileListView.as_view(), name="llm_list"),
    path("llm/new/", views.LlmProfileCreateView.as_view(), name="llm_new"),
    path("llm/<int:pk>/edit/", views.LlmProfileUpdateView.as_view(), name="llm_edit"),
    path("llm/<int:pk>/delete/", views.LlmProfileDeleteView.as_view(), name="llm_delete"),
    path("users/", views.user_admin, name="user_admin"),
    path("users/<int:pk>/access/", views.user_access_save, name="user_access_save"),
    path("users/create/", views.user_create, name="user_create"),
    path("users/<int:pk>/toggle-active/", views.user_toggle_active, name="user_toggle_active"),
]
