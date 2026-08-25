from rag.rerank import rerank_hits


def _hit(table, distance, role="", schema="", label=""):
    return {
        "id": f"{table}::1",
        "distance": distance,
        "metadata": {"table": table, "schema": schema, "label": label, "wh_role": role},
        "text": "",
    }


def test_dimcompany_beats_etl_view_for_company_count():
    hits = [
        _hit("vwDataSourceConnectionStatus", 0.55, role="", schema="ETL"),
        _hit("DimCompany", 0.62, role="dimension", schema="BI"),
    ]
    out = rerank_hits(hits, "how many companies are there")
    assert out[0]["metadata"]["table"] == "DimCompany"


def test_staging_penalty_pushes_etl_down():
    hits = [
        _hit("ETL.vwStageSales", 0.40),
        _hit("FactSales", 0.58, role="fact"),
    ]
    out = rerank_hits(hits, "total sales amount by region")
    assert out[0]["metadata"]["table"] == "FactSales"


def test_agg_intent_boosts_facts():
    hits = [
        _hit("DimProduct", 0.50, role="dimension"),
        _hit("FactInternetSales", 0.55, role="fact"),
    ]
    out = rerank_hits(hits, "what is the revenue trend per month")
    assert out[0]["metadata"]["table"] == "FactInternetSales"


def test_plural_and_singular_matching():
    hits = [
        _hit("SomeUnrelatedTable", 0.45),
        _hit("DimCustomer", 0.60, role="dimension"),
    ]
    out = rerank_hits(hits, "list all customers")
    assert out[0]["metadata"]["table"] == "DimCustomer"


def test_expand_question_corrects_schema_typos():
    from rag.rerank import expand_question

    vocab = {"dim", "supplier", "company", "sales", "logistic"}
    out = expand_question("give me suplier info of xperial", vocab)
    assert "supplier" in out
    assert out.startswith("give me suplier info of xperial")


def test_fuzzy_misspelling_still_matches_entity():
    hits = [
        _hit("DimPurchasingAgent", 0.52, role="dimension"),
        _hit("DimSupplier", 0.58, role="dimension"),
    ]
    out = rerank_hits(hits, "give me suplier info of Xperial")
    assert out[0]["metadata"]["table"] == "DimSupplier"


def test_no_boost_without_signal():
    hits = [_hit("Aaa", 0.5), _hit("Bbb", 0.7)]
    out = rerank_hits(hits, "zzz qqq")
    assert [h["metadata"]["table"] for h in out] == ["Aaa", "Bbb"]
