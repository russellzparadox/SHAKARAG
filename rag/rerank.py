from __future__ import annotations

import re
from typing import Any

STOPWORDS = {
    "how", "many", "much", "count", "number", "total", "list", "show", "all",
    "the", "a", "an", "are", "there", "is", "was", "were", "of", "in", "on",
    "for", "to", "and", "or", "per", "by", "each", "from", "what", "which",
    "give", "me", "please", "get", "find", "display", "select", "with",
    "does", "do", "has", "have", "had", "at", "be", "been", "it", "its",
    "their", "this", "that", "these", "those", "we", "i", "you", "us",
}

COUNT_INTENT = re.compile(
    r"\b(how many|how much|number of|count( of| all)?|list (all|the)?|show (all|the|me)|"
    r"total number|unique|distinct)\b",
    re.IGNORECASE,
)
AGG_INTENT = re.compile(
    r"\b(total|sum|average|avg|revenue|sales|amount(s)?|trend|per (month|day|week|year|quarter)|"
    r"grouped by|group by|top \d+|ranking|most|best|worst|growth)\b",
    re.IGNORECASE,
)
STAGING_PATTERN = re.compile(
    r"(?:^|[^a-z0-9])(etl|stage|staging|tmp|temp|raw|landing|audit|backup|vw|v_?view)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)

ROLE_BOOSTS_COUNT = {"dimension": 0.09, "fact": 0.01, "relation": -0.08, "bridge": -0.06}
ROLE_BOOSTS_AGG = {"fact": 0.10, "dimension": 0.0, "relation": -0.04}
MAX_NAME_BOOST = 0.22
MAX_TOTAL_BOOST = 0.32


def _camel_split(text: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)


def _tokens(text: str) -> set[str]:
    parts = re.split(r"[^a-zA-Z0-9]+", _camel_split(text))
    return {t.lower() for t in parts if len(t) >= 3}


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _question_words(question: str) -> list[str]:
    raw = [w for w in re.split(r"[^a-z0-9]+", question.lower()) if len(w) >= 3]
    return [w for w in raw if w not in STOPWORDS]


def _name_tokens_for(hit: dict[str, Any]) -> set[str]:
    meta = hit.get("metadata") or {}
    parts = [
        str(meta.get("table") or ""),
        str(meta.get("label") or ""),
        str(meta.get("model") or "").replace(".", "_"),
    ]
    out: set[str] = set()
    for p in parts:
        out |= _tokens(p)
    return out


def rerank_hits(hits: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    qwords = _question_words(question)
    is_count = bool(COUNT_INTENT.search(question or ""))
    is_agg = bool(AGG_INTENT.search(question or ""))

    scored: list[tuple[float, dict[str, Any]]] = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        distance = float(hit.get("distance", 1.0))
        boost = 0.0
        role = str(meta.get("wh_role") or "")

        name_tokens = _name_tokens_for(hit)
        matches = 0
        for w in qwords:
            variants = {w, _singular(w)}
            if variants & name_tokens:
                matches += 1
        if matches:
            boost += min(MAX_NAME_BOOST, 0.11 * matches)

        if is_count and not is_agg:
            boost += ROLE_BOOSTS_COUNT.get(role, 0.0)
        elif is_agg:
            boost += ROLE_BOOSTS_AGG.get(role, 0.0)

        table_schema = f"{meta.get('schema') or ''} {meta.get('table') or ''}"
        if STAGING_PATTERN.search(table_schema) and not STAGING_PATTERN.search(question.lower()):
            boost -= 0.18

        boost = max(-0.25, min(boost, MAX_TOTAL_BOOST))
        scored.append((distance - boost, hit))

    scored.sort(key=lambda pair: pair[0])
    return [hit for _, hit in scored]
