from rag.chunking import build_chunks
from rag.introspect import Column, ForeignKey, TableRecord


def _table(name: str, n_cols: int = 5, kind: str = "r", fks: list | None = None) -> TableRecord:
    rec = TableRecord(
        schema="public",
        name=name,
        kind=kind,
        row_estimate=1000,
        comment=None,
        columns=[
            Column(
                name=f"col_{i}",
                type="character varying" if i else "integer",
                nullable=True,
                default=None,
                comment=None,
                identity=False,
                generated=False,
            )
            for i in range(n_cols)
        ],
        primary_key=[],
        foreign_keys=fks or [],
    )
    return rec


def test_basic_chunk_structure():
    rec = _table("res_partner", 5)
    chunks = build_chunks([rec], {}, {})
    assert len(chunks) == 1
    c = chunks[0]
    assert c.id == "res_partner::full"
    assert c.metadata["table"] == "res_partner"
    assert c.metadata["kind"] == "TABLE"
    assert f"{c.metadata['kind']} {c.metadata['table']}" in c.text
    assert "SQL TABLE res_partner" in c.text
    assert "- col_0 integer NOT NULL" in c.text or "- col_0 integer" in c.text


def test_large_table_splits_into_parts():
    rec = _table("big_table", 140)
    rec.primary_key = ["col_0"]
    fk = ForeignKey("fk1", ["col_2"], "public.res_partner", ["id"])
    rec.foreign_keys = [fk]
    chunks = build_chunks([rec], {}, {})
    types = [c.metadata["chunk_type"] for c in chunks]
    assert "summary" in types
    assert types.count("columns") >= 3
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    summary = next(c for c in chunks if c.metadata["chunk_type"] == "summary")
    assert "res_partner(id)" in summary.text


def test_view_kind_and_metadata():
    rec = _table("my_view", 3, kind="v")
    chunks = build_chunks([rec], {"my_view": type("M", (), {"model": "my.view", "label": "A View", "info": None})}, {})
    assert chunks[0].metadata["kind"] == "VIEW"
    assert chunks[0].metadata["model"] == "my.view"


def test_field_enrichment_in_text():
    rec = _table("sale_order", 2)
    fields = {
        ("sale_order", "col_1"): type(
            "F", (), {"description": "Total Untaxed", "help": "Sum before tax", "ttype": "monetary", "relation": None}
        )
    }
    chunks = build_chunks([rec], {}, fields)
    assert "Total Untaxed" in chunks[0].text
