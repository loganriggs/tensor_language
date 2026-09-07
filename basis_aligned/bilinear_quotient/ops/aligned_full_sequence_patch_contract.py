"""Fail-closed row contract for absolute-index full-sequence activation patches."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


class FullSequenceAlignmentError(ValueError):
    """Raised when a row cannot support an absolute-index sequence patch."""


def derive_full_sequence_alignment_contract(
    rows: Iterable[Mapping[str, object]], *, required_panels: Iterable[str]
) -> dict[str, object]:
    materialized = list(rows)
    panels = tuple(required_panels)
    if not panels or len(set(panels)) != len(panels):
        raise FullSequenceAlignmentError("required panels must be nonempty and unique")
    selected = [row for row in materialized if str(row.get("transform_id")) in panels]
    counts = Counter(str(row.get("transform_id")) for row in selected)
    if set(counts) != set(panels) or any(counts[panel] == 0 for panel in panels):
        raise FullSequenceAlignmentError("one or more required panels are empty")
    row_ids, lengths = [], {}
    for row in selected:
        row_id = str(row.get("row_id", ""))
        base_ids, donor_ids = row.get("base_ids"), row.get("donor_ids")
        if not row_id or not isinstance(base_ids, list) or not isinstance(donor_ids, list):
            raise FullSequenceAlignmentError("row id or token arrays are missing")
        if len(base_ids) != len(donor_ids):
            raise FullSequenceAlignmentError(
                f"row {row_id} is not token-position aligned: {len(base_ids)} != {len(donor_ids)}"
            )
        base_position = row.get("base_semantic_position")
        donor_position = row.get("donor_semantic_position")
        if base_position != donor_position or base_position != len(base_ids) - 1:
            raise FullSequenceAlignmentError(
                f"row {row_id} has unequal or nonterminal semantic positions"
            )
        row_ids.append(row_id)
        lengths[row_id] = len(base_ids)
    if len(row_ids) != len(set(row_ids)):
        raise FullSequenceAlignmentError("duplicate row ids")
    return {
        "patch_semantics": "absolute_token_index_through_equal_terminal_semantic_position",
        "panel_counts": {panel: counts[panel] for panel in panels},
        "row_ids": row_ids,
        "token_lengths": lengths,
    }

