from rag.examples import ExampleStore, example_id
from rag.introspect import Column, ForeignKey, TableRecord


class HashEmbedder:
    tag = "hash-test"

    def __call__(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            for i, ch in enumerate(t.lower()):
                v[(ord(ch) + i) % 64] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out


def _tmp_store(tmp_path, name="tcol"):
    return ExampleStore(str(tmp_path), name, HashEmbedder())


def test_example_id_stable_and_normalized():
    a = example_id("c", "How  MANY   orders?")
    b = example_id("c", "how many orders?")
    assert a == b
    assert example_id("other", "how many orders?") != a


def test_add_search_roundtrip(tmp_path):
    store = _tmp_store(tmp_path)
    store.add(
        question="revenue per month",
        sql="SELECT month, SUM(amount) FROM f_sales GROUP BY month",
        notes="uses f_sales",
    )
    assert store.count() == 1
    hits = store.search("what is the revenue each month", k=1)
    assert len(hits) == 1
    assert "SUM(amount)" in hits[0]["sql"]
    assert hits[0]["notes"] == "uses f_sales"


def test_remove(tmp_path):
    store = _tmp_store(tmp_path)
    store.add(question="q1", sql="SELECT 1")
    assert store.count() == 1
    assert store.remove("Q1 ") is True
    assert store.count() == 0
    assert store.search("q1") == []


def test_upsert_updates_same_question(tmp_path):
    store = _tmp_store(tmp_path)
    store.add(question="q", sql="SELECT 1")
    store.add(question="q", sql="SELECT 2")
    assert store.count() == 1
    assert store.search("q")[0]["sql"] == "SELECT 2"


def _fact_table(name="f_sales"):
    rec = TableRecord(schema="dw", name=name, kind="r", row_estimate=900000, comment=None)
    rec.columns = [
        Column(name="id", type="bigint", nullable=False, default=None, comment=None, identity=True, generated=False),
        Column(name="customer_id", type="int", nullable=True, default=None, comment=None, identity=False, generated=False),
        Column(name="product_id", type="int", nullable=True, default=None, comment=None, identity=False, generated=False),
        Column(name="order_date", type="date", nullable=True, default=None, comment=None, identity=False, generated=False),
        Column(name="amount_total", type="numeric(18,2)", nullable=True, default=None, comment=None, identity=False, generated=False),
        Column(name="qty", type="integer", nullable=True, default=None, comment=None, identity=False, generated=False),
        Column(name="status_code", type="varchar(20)", nullable=True, default=None, comment=None, identity=False, generated=False),
    ]
    for ref in ("customers", "products"):
        rec.foreign_keys.append(ForeignKey(f"fk_{ref}", [f"{ref[:-1]}_id"], ref, ["id"]))
    return rec


def test_classifies_fact_by_structure():
    from rag.warehouse import classify_tables

    fact = _fact_table()
    dim = TableRecord(schema="dw", name="customers", kind="r", row_estimate=5000, comment=None)
    dim.columns = [
        Column(name="id", type="int", nullable=False, default=None, comment=None, identity=True, generated=False),
        Column(name="name", type="varchar(100)", nullable=True, default=None, comment=None, identity=False, generated=False),
        Column(name="city", type="varchar(60)", nullable=True, default=None, comment=None, identity=False, generated=False),
        Column(name="segment", type="varchar(30)", nullable=True, default=None, comment=None, identity=False, generated=False),
    ]
    dim.referenced_by = fact.foreign_keys

    classify_tables([fact, dim])
    assert fact.warehouse_role == "fact"
    assert dim.warehouse_role == "dimension"


def test_classifies_by_name():
    from rag.warehouse import classify_tables

    f = TableRecord(schema="dw", name="fact_orders", kind="r", row_estimate=10, comment=None)
    d = TableRecord(schema="dw", name="dim_date", kind="r", row_estimate=10, comment=None)
    classify_tables([f, d])
    assert f.warehouse_role == "fact"
    assert d.warehouse_role == "dimension"


def test_thin_join_table_is_relation():
    from rag.warehouse import classify_tables

    rel = TableRecord(schema="public", name="a_b_rel", kind="r", row_estimate=40, comment=None)
    rel.columns = [
        Column(name="id", type="int", nullable=False, default=None, comment=None, identity=True, generated=False),
        Column(name="a_id", type="int", nullable=False, default=None, comment=None, identity=False, generated=False),
        Column(name="b_id", type="int", nullable=False, default=None, comment=None, identity=False, generated=False),
    ]
    rel.foreign_keys = [ForeignKey("fa", ["a_id"], "a", ["id"]), ForeignKey("fb", ["b_id"], "b", ["id"])]
    classify_tables([rel])
    assert rel.warehouse_role == "relation"


def test_value_column_candidates_skip_secrets_and_keys():
    from rag.warehouse import candidate_value_columns

    rec = TableRecord(schema="s", name="users", kind="r", row_estimate=10, comment=None)
    rec.columns = [
        Column(name="id", type="int", nullable=False, default=None, comment=None, identity=True, generated=False),
        Column(name="email_token", type="varchar(40)", nullable=True, default=None, comment=None, identity=False, generated=False),
        Column(name="password_hash", type="varchar(80)", nullable=True, default=None, comment=None, identity=False, generated=False),
        Column(name="status", type="varchar(20)", nullable=True, default=None, comment=None, identity=False, generated=False),
        Column(name="country_code", type="varchar(3)", nullable=True, default=None, comment=None, identity=False, generated=False),
        Column(name="balance", type="numeric", nullable=True, default=None, comment=None, identity=False, generated=False),
    ]
    picked = candidate_value_columns(rec)
    assert "status" in picked and "country_code" in picked
    assert "password_hash" not in picked and "email_token" not in picked
