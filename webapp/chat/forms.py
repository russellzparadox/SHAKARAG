from __future__ import annotations

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .crypto import decrypt, encrypt
from .models import ChatSession, DatabaseProfile, LLMProfile


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=False)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")


class DatabaseProfileForm(forms.ModelForm):
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(render_value=False),
        required=False,
        help_text="Leave blank on edit to keep the stored one.",
    )

    class Meta:
        model = DatabaseProfile
        fields = [
            "name", "dialect", "host", "port", "db_user", "db_name", "db_url",
            "collection_name",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            self.fields["password"].help_text = "Stored. Leave blank to keep."

    def save(self, commit=True):
        obj = super().save(commit=False)
        raw = self.cleaned_data.get("password")
        if raw:
            obj.password_enc = encrypt(raw)
        elif not obj.password_enc:
            obj.password_enc = ""
        if commit:
            obj.save()
        return obj

    def decrypted_password(self) -> str:
        return decrypt(self.instance.password_enc)


class LLMProfileForm(forms.ModelForm):
    api_key = forms.CharField(
        label="API key",
        widget=forms.PasswordInput(render_value=False),
        required=False,
        help_text="Leave blank on edit to keep the stored one.",
    )

    class Meta:
        model = LLMProfile
        fields = ["name", "base_url", "model", "temperature"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            self.fields["api_key"].help_text = "Stored. Leave blank to keep."

    def save(self, commit=True):
        obj = super().save(commit=False)
        raw = self.cleaned_data.get("api_key")
        if raw:
            obj.api_key_enc = encrypt(raw)
        if commit:
            obj.save()
        return obj


class SessionCreateForm(forms.ModelForm):
    class Meta:
        model = ChatSession
        fields = ["title", "database", "llm", "language"]
        labels = {"database": "Database", "llm": "LLM", "language": "Answer language"}

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["database"].queryset = DatabaseProfile.objects.all()
        self.fields["llm"].queryset = LLMProfile.objects.all()
        self.fields["title"].required = False
