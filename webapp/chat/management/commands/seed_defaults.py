from __future__ import annotations

import os

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from chat.crypto import encrypt
from chat.models import DatabaseProfile, LLMProfile


class Command(BaseCommand):
    help = "Create shared default profiles from the repo .env (Odoo Postgres + configured LLM)."

    def add_arguments(self, parser):
        parser.add_argument("--username", type=str, default=None, help="Owner username (default: first superuser, else shared)")

    def handle(self, *args, **options):
        from rag.config import load_settings

        s = load_settings()
        owner = None
        if options["username"]:
            owner = User.objects.filter(username=options["username"]).first()
            if owner is None:
                self.stdout.write(self.style.ERROR(f"No user '{options['username']}'."))
                return
        if owner is None:
            owner = User.objects.filter(is_superuser=True).first()

        llm = None
        if s.llm_ready:
            llm, created = LLMProfile.objects.update_or_create(
                name="9router default",
                defaults={
                    "base_url": s.llm_base_url,
                    "model": s.llm_model,
                    "api_key_enc": encrypt(s.llm_api_key or ""),
                    "temperature": s.llm_temperature,
                    "owner": owner,
                },
            )
            self.stdout.write(self.style.SUCCESS(f"LLM profile '9router default' ({s.llm_model}) {'created' if created else 'updated'}"))

        dbp, created = DatabaseProfile.objects.update_or_create(
            name="Odoo shaka (Postgres)",
            defaults={
                "dialect": "postgres",
                "host": s.db_host,
                "port": s.db_port,
                "db_user": s.db_user,
                "password_enc": encrypt(s.db_password),
                "db_name": s.db_name,
                "collection_name": s.collection,
                "owner": owner,
            },
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Database profile 'Odoo shaka (Postgres)' {'created' if created else 'updated'} "
                f"(collection={s.collection})"
            )
        )
        if not os.getenv("CI"):
            self.stdout.write("Tip: open Databases → Index now to build its vector index.")
