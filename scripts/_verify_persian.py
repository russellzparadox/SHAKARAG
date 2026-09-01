"""Quick end-to-end check: Persian query against a temp index built with the
multilingual-e5 embedder, exercising the real VectorStore query path.
(Independent of the slow full admin1 reindex.)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")

from rag.chunking import build_chunks  # noqa: E402
from rag.embeddings import E5MultilingualEmbedder  # noqa: E402
from rag.store import VectorStore  # noqa: E402


def _table(name, schema, cols, fks=None):
    from rag.introspect import Column, ForeignKey, TableRecord, ForeignKey as FK

    rec = TableRecord(
        schema=schema, name=name, kind="r", row_estimate=1000, comment=None,
        columns=cols, primary_key=["id"], foreign_keys=fks or [],
        warehouse_role="dim", role_reason="test",
    )
    return rec


def _col(name, sample=None, pk=False):
    from rag.introspect import Column
    return Column(
        name=name, type="nvarchar", nullable=True, default=None,
        comment=None, identity=False, generated=False,
        pk=pk, sample_values=sample or [],
    )


def build():
    tmp = tempfile.mkdtemp(prefix="icrlpersian_")
    embedder = E5MultilingualEmbedder()
    tables = [
        _table("DimCompany", "BI", [
            _col("CompanyID", pk=True), _col("CompanyTitle", ["هلدینگ کالوپ", "کوهپایه"]),
            _col("EnTitle", ["HoldingKalup"]),
        ]),
        _table("PurchasingDepartment", "BI", [
            _col("CompanyID", pk=True),
            _col("PurchasingDepartmentTitle", ["خرید"]),
            _col("PurchasingDepartmentEnglishTitle"),
        ]),
        _table("DimSupplier", "BI", [
            _col("SupplierID", pk=True), _col("SupplierTitle", ["تامین"]),
        ]),
    ]
    chunks = build_chunks(tables, models_by_table={}, fields_by_key={})
    store = VectorStore(tmp, "persian_test", embedder)
    store.upsert(chunks)
    print(f"indexed {store.count} chunks into {tmp}")
    return store


def run(store):
    hits = store.query_text("همه شرکت ها رو لیست کن", top_k=5)
    rank = []
    for h in hits:
        m = h.get("metadata") or {}
        rank.append((round(h.get("distance", 1.0), 4), m.get("table", "?")))
    print("\nPersian query top-5:")
    for d, t in rank:
        print(f"  {d:.4f}  {t}")
    return rank


if __name__ == "__main__":
    store = build()
    run(store)