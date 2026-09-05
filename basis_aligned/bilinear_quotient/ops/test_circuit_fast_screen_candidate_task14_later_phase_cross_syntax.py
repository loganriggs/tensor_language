#!/usr/bin/env python3
# BQLANE: cpu

import hashlib
import json
from pathlib import Path

import pytest

import circuit_battery_task14 as task14
import circuit_fast_screen_candidate_task14_ood_cross_syntax as ood_candidate
import circuit_fast_screen_candidate_task14_phase_cross_syntax as shared
import circuit_fast_screen_candidate_task14_test_cross_syntax as test_candidate


EXPECTED = {
    "TEST": {
        "authority": "cb90557800d619fdce0587910ce7e09726759dbcb3df614eb614f8f99bec3e21",
        "plan": "028974ace2024e4127acab46234eae8f4ce15cacdc95f58ad35db076e9ba133f",
    },
    "OOD": {
        "authority": "95a49814bca8a72c03bb0392282e2149d87ac43cddd5b389559cc6d55a7dec4f",
        "plan": "ec576b2e75cc0831a44520454f7e03ea62d19d8ea3d2bcc834844e253f80a004",
    },
}


@pytest.mark.parametrize("candidate", (test_candidate, ood_candidate))
def test_later_phase_authority_and_plan_are_frozen(candidate) -> None:
    rows = candidate.build_rows()
    plan = candidate.compile_plan(rows)
    assert candidate.validate_rows(rows) == EXPECTED[candidate.PHASE]["authority"]
    assert plan["compiled_sha256"] == EXPECTED[candidate.PHASE]["plan"]
    assert len(rows) == 64
    assert {row["split"] for row in rows} == {candidate.PHASE}
    cells = {
        cell: sum(row["cell_id"] == cell for row in rows)
        for cell in {row["cell_id"] for row in rows}
    }
    assert len(cells) == 4 and set(cells.values()) == {16}
    assert plan["site_ids"] == ["attn:11:head:03"]
    assert plan["price"] == {
        "forward_calls": 10,
        "example_evaluations": 320,
        "backward_calls": 0,
        "model_updates": 0,
        "raw_numeric_evidence_bytes": 2560,
    }
    assert [call["kind"] for call in plan["calls"]] == [
        "native", "native", "native", "native",
        "exact_single_position_interchange", "exact_single_position_interchange",
        "zero_removal", "zero_removal",
        "native_head_replay", "native_head_replay",
    ]
    assert plan["score"] == {
        "minimum_native_cell_accuracy": 0.85,
        "minimum_cell_direction_fraction": 0.75,
        "minimum_cell_mean_recovery": 0.40,
        "minimum_cell_median_normalized_removal_damage": 0.25,
        "minimum_cell_positive_removal_fraction": 0.65,
        "maximum_native_head_replay_absolute_logit_error": 1.0e-4,
    }


@pytest.mark.parametrize("candidate", (test_candidate, ood_candidate))
def test_relations_use_frozen_cyclic_cross_noun_semantics(candidate) -> None:
    authority, _ = task14.build_authority()
    phase_rows, _ = task14.split_rows(authority, candidate.PHASE)
    source = {f"{row['row_id']}:{side}": row
              for row in phase_rows for side in ("base", "donor")}
    panels = {
        str(row["group_id"]): row for row in phase_rows
        if row["transform_id"] == "A1"
    }
    strata: dict[tuple[object, ...], list[str]] = {}
    for group_id, row in panels.items():
        key: tuple[object, ...] = (
            row["base_subject_number"], row["base_attractor_plural"],
        )
        if candidate.PHASE == "OOD":
            key += (row["base_second_attractor_plural"],)
        strata.setdefault(key, []).append(group_id)
    expected_donor = {}
    for group_ids in strata.values():
        ordered = sorted(group_ids, key=lambda item: panels[item]["group_number"])
        expected_donor.update({
            group_id: ordered[(index + 1) % len(ordered)]
            for index, group_id in enumerate(ordered)
        })
    for relation in candidate.build_rows():
        target = source[relation["target_endpoint_id"]]
        donor = source[relation["donor_endpoint_id"]]
        assert target["group_id"] == relation["group_id"]
        assert donor["group_id"] == relation["donor_group_id"]
        assert relation["donor_group_id"] == expected_donor[relation["group_id"]]
        assert target["group_id"] != donor["group_id"]
        assert {target["transform_id"], donor["transform_id"]} == {"A1", "A2"}
        assert target["base_subject_number"] == donor["base_subject_number"]
        assert target["base_attractor_plural"] == donor["base_attractor_plural"]
        if candidate.PHASE == "OOD":
            assert target["base_second_attractor_plural"] == \
                donor["base_second_attractor_plural"]
        assert relation["target_endpoint_id"].endswith(":base")
        assert relation["donor_endpoint_id"].endswith(":donor")
        assert relation["base_answer_id"] == relation["donor_foil_id"]
        assert relation["base_foil_id"] == relation["donor_answer_id"]


def test_ood_keeps_unequal_sequence_positions_explicit() -> None:
    rows = ood_candidate.build_rows()
    assert all(len(row["base_ids"]) != len(row["donor_ids"]) for row in rows)
    assert all(row["base_semantic_position"] == len(row["base_ids"]) - 1
               and row["donor_semantic_position"] == len(row["donor_ids"]) - 1
               for row in rows)


def test_factory_reads_generator_but_no_result_artifact(monkeypatch) -> None:
    observed: list[Path] = []
    original = Path.read_bytes

    def recording_read_bytes(path: Path) -> bytes:
        observed.append(path.resolve())
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", recording_read_bytes)
    test_candidate.build_rows()
    assert observed and set(observed) == {Path(task14.__file__).resolve()}
    assert set(test_candidate.EXPECTED_SOURCE_SHA256) == {
        "task14_generator_file", "task14_full_authority", "task14_phase_records",
    }


def test_factory_refuses_select_and_mutated_rows() -> None:
    with pytest.raises(shared.PhaseCrossSyntaxAuthorityError, match="TEST or OOD"):
        shared.PhaseCrossSyntaxConfig(
            phase="SELECT", schema="x", validation_scope="x",
            expected_phase_records_sha256="0" * 64, correction="x",
            donor_rule="x",
        )
    rows = test_candidate.build_rows()
    rows[0] = dict(rows[0], base_text=rows[0]["base_text"] + " altered")
    with pytest.raises(
        shared.PhaseCrossSyntaxAuthorityError,
        match="exact regenerated TEST authority",
    ):
        test_candidate.validate_rows(rows)


@pytest.mark.parametrize("candidate", (test_candidate, ood_candidate))
def test_frozen_dryrun_receipt_matches_compiler(candidate) -> None:
    path = Path(__file__).parent / (
        f"circuit_fast_screen_task14_{candidate.PHASE.lower()}_cross_syntax_dryrun.json"
    )
    receipt = json.loads(path.read_text())
    plan = candidate.compile_plan()
    calls_payload = json.dumps(
        plan["calls"], sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    assert receipt["authority_sha256"] == plan["authority_sha256"]
    assert receipt["compiled_sha256"] == plan["compiled_sha256"]
    assert receipt["call_manifest_sha256"] == hashlib.sha256(calls_payload).hexdigest()
    assert receipt["price"] == plan["price"]
    assert receipt["score"] == plan["score"]
    assert receipt["outcome_files_read"] is False
    assert receipt["gpu_accessed"] is False
    assert receipt["queue_touched"] is False
