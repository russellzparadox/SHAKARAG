from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..introspect import TableRecord


class BaseDialect(ABC):
    name: str = "base"
    label: str = "SQL"
    supports_odoo_metadata: bool = False

    def __init__(self, settings) -> None:
        self.settings = settings

    @abstractmethod
    def introspect(self) -> dict[tuple[str, str], TableRecord]:
        ...

    @abstractmethod
    def execute_readonly(self, sql: str, max_rows: int) -> tuple[list[str], list[list[Any]], bool]:
        ...

    def prompt_hints(self) -> str:
        return ""

    @staticmethod
    def missing_driver(package: str, extra: str = "") -> RuntimeError:
        return RuntimeError(
            f"Driver package '{package}' is required for this database. "
            f"Install it with: pip install {package}{extra}"
        )
