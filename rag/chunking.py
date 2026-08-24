from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .introspect import TableRecord

FULL_DOC_MAX_CHARS = 3800
COLS_PER_CHUNK = 55
REL_CAP = 14
IDX_CAP = 15
HELP_CAP = 160
INFO_CAP = 400


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


def _qual(rec: TableRecord) -> str:
    return rec.qualified


def _fk_out_lines(rec: TableRecord) -> list[str]:
    lines = []
    for fk in rec.foreign_keys[:REL_CAP]:
        cols = ", ".join(fk.columns)
        refs = ", ".join(fk.ref_columns) or "?"
        lines.append(f"- {cols} -> {fk.ref_table}({refs})")
    if len(rec.foreign_keys) > REL_CAP:
        lines.append(f"- (+{len(rec.foreign_keys) - REL_CAP} more foreign keys)")
    return lines


def _fk_in_lines(rec: TableRecord) -> list[str]:
    lines = []
    for fk in rec.referenced_by[:REL_CAP]:
        src = fk.source or "?"
        cols = ", ".join(fk.columns)
        refs = ", ".join(fk.ref_columns)
        lines.append(f"- {src}({cols}) references this table")
    if len(rec.referenced_by) > REL_CAP:
        lines.append(f"- (+{len(rec.referenced_by) - REL_CAP} more incoming references)")
    return lines


def _column_line(rec: TableRecord, col, field_info=None) -> str:
    parts = [f"- {col.name} {col.type}"]
    if not col.nullable:
        parts.append("NOT NULL")
    if col.pk:
        parts.append("[PK]")
    if col.default:
        parts.append(f"DEFAULT {col.default}")
    if col.identity:
        parts.append("IDENTITY")
    line = " ".join(parts)

    desc_bits: list[str] = []
    if field_info is not None:
        if field_info.description:
            desc_bits.append(field_info.description)
        if field_info.help:
            help_text = field_info.help[:HELP_CAP]
            desc_bits.append(help_text)
    if not desc_bits and col.comment:
        desc_bits.append(col.comment)
    if desc_bits:
        line += f" — {' | '.join(desc_bits)}"
    return line


def _header_block(rec: TableRecord, model_info, n_chunks_hint: int = 0) -> list[str]:
    label = model_info.label if model_info else None
    info = model_info.info if model_info else None

    head = f"PostgreSQL {rec.kind_label} {_qual(rec)}"
    if model_info:
        head += f" (Odoo model: {model_info.model}, Label: \"{label}\")"
    if rec.comment:
        head += f"\nComment: {rec.comment}"
    if info:
        head += f"\nAbout: {info[:INFO_CAP]}"

    meta_bits = [f"Rows≈{max(rec.row_estimate, 0)}", f"Columns:{len(rec.columns)}"]
    if rec.primary_key:
        meta_bits.append("PRIMARY KEY (" + ", ".join(rec.primary_key) + ")")
    block = [
        head,
        "Stats: " + " · ".join(meta_bits),
    ]

    relations = _fk_out_lines(rec)
    if relations:
        block.append("Foreign keys (outgoing):")
        block.extend(relations)

    incoming = _fk_in_lines(rec)
    if incoming:
        block.append("Referenced by:")
        block.extend(incoming)

    if rec.unique_constraints:
        uniqs = ["(" + ", ".join(u) + ")" for u in rec.unique_constraints[:8]]
        block.append("Unique constraints: " + "; ".join(uniqs))

    idx_defs = [(n, d) for n, d in rec.indexes if not n.endswith("_pkey")]
    if idx_defs:
        shown = [f"- {d}" for _, d in idx_defs[:IDX_CAP]]
        extra = len(idx_defs) - IDX_CAP
        if extra > 0:
            shown.append(f"- (+{extra} more indexes)")
        block.append("Indexes:")
        block.extend(shown)

    return block


def build_chunks(
    tables: Iterable[TableRecord],
    models_by_table: dict,
    fields_by_key: dict,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for rec in tables:
        qual = _qual(rec)
        model_info = models_by_table.get(rec.name)
        columns = rec.columns
        column_lines = [
            _column_line(rec, col, fields_by_key.get((rec.name, col.name))) for col in columns
        ]

        header = "\n".join(_header_block(rec, model_info))
        columns_text = "\n".join(column_lines)
        full_text = header + "\n\nColumns:\n" + columns_text

        base_meta = {
            "schema": rec.schema,
            "table": rec.name,
            "kind": rec.kind_label,
            "ncols": len(columns),
            "rows_est": max(int(rec.row_estimate), 0),
            "model": model_info.model if model_info else "",
            "label": model_info.label if model_info else "",
        }

        if len(full_text) <= FULL_DOC_MAX_CHARS or not column_lines:
            chunks.append(
                Chunk(
                    id=f"{qual}::full",
                    text=full_text,
                    metadata={**base_meta, "chunk_type": "full", "part": 0},
                )
            )
            continue

        summary_note = (
            f"\n\n(Columns listed separately in part chunks; total {len(columns)} columns.)"
        )
        chunks.append(
            Chunk(
                id=f"{qual}::summary",
                text=header + summary_note + "\n\nKey columns:\n"
                + "\n".join(line for line, col in zip(column_lines, columns) if col.pk or col.fk_ref)[:1500],
                metadata={**base_meta, "chunk_type": "summary", "part": 0},
            )
        )

        for part_start in range(0, len(columns), COLS_PER_CHUNK):
            group = range(part_start, min(part_start + COLS_PER_CHUNK, len(columns)))
            part_no = part_start // COLS_PER_CHUNK + 1
            total_parts = (len(columns) + COLS_PER_CHUNK - 1) // COLS_PER_CHUNK
            text = (
                f"{_qual(rec)} ({model_info.model if model_info else 'table'}) "
                f"columns part {part_no}/{total_parts}:\n"
                + "\n".join(column_lines[i] for i in group)
            )
            chunks.append(
                Chunk(
                    id=f"{qual}::cols{part_no}",
                    text=text,
                    metadata={**base_meta, "chunk_type": "columns", "part": part_no},
                )
            )

    return chunks


def dump_catalog(tables: Iterable[TableRecord], path: Path) -> int:
    payload = []
    for rec in tables:
        payload.append(
            {
                "qualified": rec.qualified,
                "kind": rec.kind_label,
                "row_estimate": rec.row_estimate,
                "comment": rec.comment,
                "primary_key": rec.primary_key,
                "foreign_keys": [
                    {"name": fk.name, "columns": fk.columns, "ref_table": fk.ref_table, "ref_columns": fk.ref_columns}
                    for fk in rec.foreign_keys
                ],
                "referenced_by_count": len(rec.referenced_by),
                "unique_constraints": rec.unique_constraints,
                "indexes": [name for name, _ in rec.indexes],
                "view_def": rec.view_def,
                "columns": [
                    {
                        "name": c.name,
                        "type": c.type,
                        "nullable": c.nullable,
                        "default": c.default,
                        "pk": c.pk,
                        "fk_ref": c.fk_ref,
                        "comment": c.comment,
                    }
                    for c in rec.columns
                ],
            }
        )
    path.write_text(json.dumps(payload, indent=2))
    return len(payload)
