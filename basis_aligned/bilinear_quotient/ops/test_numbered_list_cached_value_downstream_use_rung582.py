from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).with_name("numbered_list_cached_value_downstream_use_rung582.py")
SPEC = importlib.util.spec_from_file_location("r582", PATH)
r582 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(r582)


@pytest.fixture(scope="module")
def rows():
    return r582.build_rows()


def test_rows_are_complete_group_disjoint_and_source_matched(rows):
    result = r582.validate_rows(rows)
    assert result == {
        "rows": 1440,
        "groups": 40,
        "split_rows": {"FIT": 576, "SELECT": 288, "FINAL_TEST": 288, "OOD": 288},
        "group_disjoint": True,
        "source_matched_action_cells": True,
    }
    split_by_group = {}
    for row in rows:
        split_by_group.setdefault(row["group_id"], row["split"])
        assert split_by_group[row["group_id"]] == row["split"]
        assert row["ids"][row["source_position"]] == row["source_id"]
        assert row["query_position"] == len(row["ids"]) - 1


def test_prerequisite_authorities_are_pinned_and_hold_the_required_boundary():
    observed = r582.validate_authorities()
    assert observed == {str(path): digest for path, digest in r582.AUTHORITIES.items()}


def test_action_cells_hold_final_source_exactly_fixed(rows):
    lookup = {(row["group_id"], row["representation"], row["source_level"], row["condition"]): row
              for row in rows}
    for group_id in {row["group_id"] for row in rows}:
        for representation in ("list", "digit", "word"):
            for source in (0, 1):
                copy = lookup[(group_id, representation, source, "factorial_copy")]
                successor = lookup[(group_id, representation, source, "factorial_successor")]
                assert copy["source_value"] == successor["source_value"]
                assert copy["source_id"] == successor["source_id"]
                assert copy["action"] == "copy" and successor["action"] == "successor"
                assert copy["answer_id"] == copy["source_id"]
                assert successor["answer_id"] != successor["source_id"]


def test_bilinear_response_is_exact_and_swap_gauge_invariant():
    rng = np.random.default_rng(582)
    left = rng.normal(size=(13, 7))
    right = rng.normal(size=(13, 7))
    down = rng.normal(size=(5, 13))
    x0 = rng.normal(size=(11, 7))
    x1 = rng.normal(size=(11, 7))
    observed = r582.bilinear_response(left, right, down, x0, x1)
    swapped = r582.bilinear_response(right, left, down, x0, x1)
    assert float(observed["relative_squared_error"]) < 1e-28
    assert np.allclose(observed["joint_response"], observed["direct_response"], atol=1e-12)
    assert np.allclose(observed["background_cross"], swapped["background_cross"], atol=1e-12)
    assert np.allclose(observed["contrast_self"], swapped["contrast_self"], atol=1e-12)


def test_bilinear_response_is_invariant_to_native_product_rescaling():
    rng = np.random.default_rng(1582)
    left = rng.normal(size=(9, 4)); right = rng.normal(size=(9, 4)); down = rng.normal(size=(3, 9))
    x0 = rng.normal(size=(6, 4)); x1 = rng.normal(size=(6, 4))
    scale_left = rng.uniform(.3, 2.0, size=9)
    scale_right = rng.uniform(.3, 2.0, size=9)
    transformed_left = left * scale_left[:, None]
    transformed_right = right * scale_right[:, None]
    transformed_down = down / (scale_left * scale_right)[None, :]
    a = r582.bilinear_response(left, right, down, x0, x1)
    b = r582.bilinear_response(transformed_left, transformed_right, transformed_down, x0, x1)
    for key in ("background_cross", "contrast_self", "joint_response"):
        assert np.allclose(a[key], b[key], atol=1e-11)


def test_action_gap_uses_matched_group_source_representation_and_surface():
    records = []
    for group in ("g0", "g1"):
        for source in (0, 1):
            for representation in ("list", "digit", "word"):
                for surface in ("factorial", "surface"):
                    records.append({"group_id": group, "source_level": source,
                                    "representation": representation,
                                    "condition": f"{surface}_copy", "damage": 0.25})
                    records.append({"group_id": group, "source_level": source,
                                    "representation": representation,
                                    "condition": f"{surface}_successor", "damage": 1.0 + source})
    gaps = r582.action_gap_records(records, "damage")
    assert len(gaps) == 24
    assert gaps[("g0", 0, "list:factorial")] == .75
    assert gaps[("g1", 1, "word:surface")] == 1.75


def test_action_gap_fails_closed_on_missing_match():
    with pytest.raises(ValueError, match="unmatched"):
        r582.action_gap_records([{"group_id": "g", "source_level": 0,
                                  "representation": "list",
                                  "condition": "factorial_successor", "damage": 1.0}], "damage")


@pytest.mark.parametrize("split,expected", [("FIT", 16 * 3 * 2 * 4), ("SELECT", 8 * 3 * 2 * 4)])
def test_null_maps_are_complete_reproducible_and_semantically_matched(rows, split, expected):
    maps = r582.deterministic_null_maps(rows, split)
    assert maps == r582.deterministic_null_maps(rows, split)
    assert all(len(mapping) == expected for mapping in maps.values())
    by_id = {row["row_id"]: row for row in rows}
    for recipient_id, donor_id in maps["different_group_same_cell"].items():
        recipient, donor = by_id[recipient_id], by_id[donor_id]
        assert recipient["group_id"] != donor["group_id"]
        assert (recipient["representation"], recipient["source_level"], recipient["condition"]) == (
            donor["representation"], donor["source_level"], donor["condition"])
    for recipient_id, donor_id in maps["same_source_other_action"].items():
        recipient, donor = by_id[recipient_id], by_id[donor_id]
        assert recipient["group_id"] == donor["group_id"]
        assert recipient["source_id"] == donor["source_id"]
        assert recipient["action"] != donor["action"]


def test_mobius_is_exact_for_two_removals():
    result = r582.two_factor_mobius(native=10.0, remove_cross=7.0,
                                    remove_self=8.0, remove_joint=4.0)
    assert result == {"cross": -3.0, "self": -2.0, "cross_x_self": -1.0}


def test_content_addressed_group_bootstrap_is_reproducible_and_cell_specific():
    values = {f"g{i}": float(i) for i in range(8)}
    a = r582.deterministic_group_bootstrap(values, cell_id="fit:list:source0", replicates=64)
    b = r582.deterministic_group_bootstrap(values, cell_id="fit:list:source0", replicates=64)
    c = r582.deterministic_group_bootstrap(values, cell_id="fit:list:source1", replicates=64)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    with pytest.raises(ValueError):
        r582.deterministic_group_bootstrap({}, cell_id="empty")


def test_dryrun_prices_every_frozen_arm_and_opens_no_split(rows):
    document = r582.dryrun_document(rows)
    assert document["fit_arm_forwards_per_batch"] == 16
    assert document["select_arm_forwards_per_batch"] == 7
    assert document["maximum_model_forwards_if_eventually_executed"] == (
        16 * document["fit_batches"] + 7 * document["select_batches"])
    assert document["model_forwards"] == document["model_backwards"] == 0
    assert document["model_loaded"] is False
    assert document["opened_splits"] == []
    assert document["FINAL_TEST_or_OOD_opened"] is False


def test_module_has_no_torch_or_model_dependency():
    source = PATH.read_text()
    assert "import torch" not in source
    assert "bilin18_observed_model" not in source
    assert "load_bilin18" not in source


def test_generated_rows_and_receipt_are_recomputable(rows):
    stored = json.loads(r582.ROWS.read_text())
    receipt = json.loads(r582.RECEIPT.read_text())
    assert stored["rows"] == rows
    assert stored["model_loaded"] is False and stored["outcomes_opened"] == []
    assert receipt["rows_sha256"] == r582.file_sha256(r582.ROWS)
    assert receipt["source_matched_action_cells"] is True
