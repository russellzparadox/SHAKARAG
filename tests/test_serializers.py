from datetime import datetime

from rag.serializers import cell, row_to_list


def test_tuple_row_values_preserved():
    row = ("shop", "customers", 12)
    assert row_to_list(row) == ["shop", "customers", 12]


def test_dict_row_uses_values_not_keys():
    row = {"TABLE_SCHEMA": "dw", "TABLE_NAME": "f_sales", "n": 3}
    assert row_to_list(row) == ["dw", "f_sales", 3]


def test_dict_row_with_mixed_types():
    row = {"id": 1, "created": datetime(2026, 8, 24, 10, 0), "price": None}
    out = row_to_list(row)
    assert out[0] == 1
    assert out[1] == "2026-08-24T10:00:00"
    assert out[2] is None


def test_cell_serializes_decimals_and_bytes():
    import decimal

    assert cell(decimal.Decimal("1.25")) == 1.25
    assert cell(b"abc").startswith("b'abc'")
