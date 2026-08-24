from __future__ import annotations

import datetime as dt
import decimal
import uuid
from typing import Any


def cell(value: Any) -> Any:
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return repr(bytes(value))[:120]
    if isinstance(value, uuid.UUID):
        return str(value)
    return value
