from __future__ import annotations

import base64
import hashlib

from django.conf import settings as dj_settings

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:
    Fernet = None
    InvalidToken = None

PREFIX = "enc1:"


def _fernet():
    if Fernet is None:
        raise RuntimeError("Install 'cryptography' to store secrets encrypted.")
    key = hashlib.sha256(str(dj_settings.SECRET_KEY).encode()).digest()
    return Fernet(base64.urlsafe_b64encode(key))


def encrypt(value: str | None) -> str:
    if not value:
        return ""
    return PREFIX + _fernet().encrypt(value.encode()).decode()


def decrypt(value: str | None) -> str:
    if not value:
        return ""
    if not value.startswith(PREFIX):
        return value
    try:
        return _fernet().decrypt(value[len(PREFIX) :].encode()).decode()
    except InvalidToken:
        return ""
