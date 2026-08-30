from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import torch

from ordered_successor_masks_v1 import OrderedLexicon, SuccessorMasks
import ordered_successor_tensor_select_v3_budget as budget


def _records(count: int) -> list[dict[str, object]]:
    return [{"document_id": f"doc-{index}"} for index in range(count)]


def _known_335_masks(rows: torch.Tensor, _lexicon: OrderedLexicon) -> SuccessorMasks:
    shape = (len(rows), 256)
    positive = torch.zeros(shape, dtype=torch.bool)
    wrong = torch.zeros(shape, dtype=torch.bool)
    none = torch.zeros(shape, dtype=torch.bool)
    positive[:199, 0] = True
    none[:199, 1] = True
    wrong[199:334, 0] = True
    positive[334, 0] = True
    wrong[334, :65] = True
    # Keep cells disjoint in the final row.
    positive[334, 0] = False
    positive[334, 65] = True
    none[334, 66] = True
    eligible = positive | wrong | none
    pair_index = torch.full(shape, -1, dtype=torch.int16)
    pair_index[eligible] = 0
    zero = torch.zeros(shape, dtype=torch.bool)
    return SuccessorMasks(
        eligible, positive, zero.clone(), zero.clone(), wrong, none,
        zero.clone(), pair_index,
    )


def test_prospective_contract_is_fixed_larger_and_non_authorizing() -> None:
    status = budget.prospective_status()
    assert status == {
        "status": "PROSPECTIVE_NO_GO_REQUIRES_V3_FREEZER_AND_INDEPENDENT_AUDIT",
        "row_materialization_authorized": False,
        "model_forward_authorized": False,
        "v2_select_documents": 192,
        "v3_select_documents": 384,
        "registered_greedy_minimum": 335,
        "margin_documents": 49,
        "scored_positions": 73_728,
    }


def test_exact_v2_audit_and_terminal_failure_lineage_replays() -> None:
    lineage = budget.validate_v2_lineage()
    assert lineage["v2_source_commit"] == budget.V2_SOURCE_COMMIT
    assert lineage["v2_failure_commit"] == budget.V2_FAILURE_COMMIT
    assert lineage["v2_independent_audit_sha256"] == budget.V2_AUDIT_SHA256
    assert lineage["v2_terminal_failure_sha256"] == budget.V2_FAILURE_SHA256


@pytest.mark.parametrize("kind", ("audit", "failure"))
def test_any_parent_byte_mutation_fails_closed(tmp_path: Path, kind: str) -> None:
    audit = tmp_path / "audit.json"
    failure = tmp_path / "failure.json"
    shutil.copyfile(budget.V2_AUDIT, audit)
    shutil.copyfile(budget.V2_FAILURE, failure)
    target = audit if kind == "audit" else failure
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="lineage byte hash changed"):
        budget.validate_v2_lineage(audit_path=audit, failure_path=failure)


def test_registered_greedy_known_answer_stops_at_335_and_fills_384() -> None:
    rows = torch.zeros(400, 257, dtype=torch.long)
    result = budget.allocate_v3_budget(
        rows, _records(400), OrderedLexicon("toy", ((1,), (2,))),
        mask_builder=_known_335_masks,
    )
    assert result.support_first_count == 335
    assert result.support_first_last_candidate == 334
    assert tuple(result.selected_rows.shape) == (384, 257)
    assert len(result.selected_records) == 384
    assert [item["candidate_scan_ordinal"] for item in result.selected_records] == list(range(384))
    assert result.census["positive_clean"] == {
        "positions": 200, "documents": 200, "passed": True,
    }
    assert result.census["wrong_source_clean"] == {
        "positions": 200, "documents": 136, "passed": True,
    }
    assert result.census["no_source_clean"] == {
        "positions": 200, "documents": 200, "passed": True,
    }
    assert result.pair_occupancy["0->1"] == {"positions": 600, "documents": 335}


def test_budget_below_registered_stop_fails_without_reranking() -> None:
    rows = torch.zeros(400, 257, dtype=torch.long)
    with pytest.raises(RuntimeError, match="requires 335 documents, budget is 334"):
        budget.allocate_v3_budget(
            rows, _records(400), OrderedLexicon("toy", ((1,), (2,))), budget=334,
            mask_builder=_known_335_masks,
        )


def test_registered_count_replay_gate_rejects_changed_support() -> None:
    rows = torch.zeros(400, 257, dtype=torch.long)

    def powered_immediately(values: torch.Tensor, _lexicon: OrderedLexicon) -> SuccessorMasks:
        shape = (len(values), 256)
        positive = torch.zeros(shape, dtype=torch.bool)
        wrong = torch.zeros(shape, dtype=torch.bool)
        none = torch.zeros(shape, dtype=torch.bool)
        positive[:30, :7] = True
        wrong[30:60, :7] = True
        none[60:90, :7] = True
        eligible = positive | wrong | none
        pair_index = torch.full(shape, -1, dtype=torch.int16)
        pair_index[eligible] = 0
        zero = torch.zeros(shape, dtype=torch.bool)
        return SuccessorMasks(
            eligible, positive, zero.clone(), zero.clone(), wrong, none,
            zero.clone(), pair_index,
        )

    with pytest.raises(RuntimeError, match="stopping count changed"):
        budget.allocate_v3_budget(
            rows, _records(400), OrderedLexicon("toy", ((1,), (2,))),
            mask_builder=powered_immediately, require_registered_stopping_count=True,
        )
