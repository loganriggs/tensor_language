from __future__ import annotations

import hashlib

import pytest

import mlp1_empirical_moment_tensor_v1_row_manifest as manifest


def small_census() -> dict[str, object]:
    return {
        "registry_file_count": 2,
        "registry_files": {"a": "0" * 64, "b": "1" * 64},
        "excluded_document_index_count": 2,
        "excluded_document_indices_sha256": "2" * 64,
    }


def test_ordering_digest_is_literal_registered_byte_formula() -> None:
    for index in (0, 17, 1_091_395):
        expected = hashlib.sha256(
            b"bilin18_mlp1_empirical_moment_v1\0" + str(index).encode("ascii")
        ).digest()
        assert manifest.ordering_digest(index) == expected
    with pytest.raises(RuntimeError, match="literal nonnegative"):
        manifest.ordering_digest(True)


def test_survivor_order_subtracts_exclusions_before_sha_order() -> None:
    expected = sorted(
        set(range(20)) - {2, 5, 11},
        key=lambda index: (manifest.ordering_digest(index), index),
    )[:7]
    assert manifest.order_surviving_indices(20, {2, 5, 11}, limit=7) == tuple(expected)
    with pytest.raises(RuntimeError, match="outside pinned parquet"):
        manifest.order_surviving_indices(20, {20}, limit=1)
    with pytest.raises(RuntimeError, match="only 1 unexcluded"):
        manifest.order_surviving_indices(3, {0, 1}, limit=2)


def test_exact_roles_masks_and_nested_prefix_arithmetic() -> None:
    value = manifest.build_role_manifest(
        parquet_rows=7_000, excluded_indices={1, 3, 5}, registry_census=small_census(),
    )
    assert tuple(value["roles"]) == manifest.ROLES
    assert value["cross_role_document_disjoint"] is True
    all_indices = []
    for role in manifest.ROLES:
        record = value["roles"][role]
        all_indices.extend(record["ordered_document_indices"])
        assert record["document_count"] == 2_084
        assert record["eligible_row_count"] == 400_000
        assert record["complete_window_count"] == 2_083
        assert record["complete_window_position_mask"] == {
            "position_start_inclusive": 64,
            "position_stop_exclusive": 256,
            "position_count": 192,
        }
        assert record["partial_window"] == {
            "document_ordinal_zero_indexed": 2_083,
            "position_start_inclusive": 64,
            "position_stop_exclusive": 128,
            "position_count": 64,
        }
    assert len(all_indices) == len(set(all_indices)) == 6_252
    assert not {1, 3, 5}.intersection(all_indices)

    fit100 = value["fit_nested_prefixes"]["FIT100"]
    fit200 = value["fit_nested_prefixes"]["FIT200"]
    fit400 = value["fit_nested_prefixes"]["FIT400"]
    assert (fit100["row_count"], fit100["complete_window_count"],
            fit100["partial_window"]["position_stop_exclusive"]) == (100_000, 520, 224)
    assert (fit200["row_count"], fit200["complete_window_count"],
            fit200["partial_window"]["position_stop_exclusive"]) == (200_000, 1_041, 192)
    assert (fit400["row_count"], fit400["complete_window_count"],
            fit400["partial_window"]["position_stop_exclusive"]) == (400_000, 2_083, 128)


def test_manifest_is_strictly_nonauthorizing_and_semantically_replayed() -> None:
    value = manifest.build_role_manifest(
        parquet_rows=6_260, excluded_indices=(), registry_census=small_census(),
    )
    assert value["authority"] == "none"
    assert value["token_identity_state"] == "not_loaded_not_hashed_not_authorized"
    assert all(value[key] is False for key in (
        "authorized_for_tokenization", "authorized_for_activation_capture",
        "authorized_for_model_forward", "authorized_for_scientific_outcomes",
    ))
    value["roles"]["FIT"]["partial_window"]["position_count"] = 65
    with pytest.raises(RuntimeError, match="mask or hash is not canonical"):
        manifest.validate_role_manifest(value)


def test_streamed_ledger_hash_is_order_and_mask_sensitive() -> None:
    indices = tuple(range(2_084))
    original = manifest.document_position_ledger_sha256(
        indices, full_windows=2_083, final_stop=128,
    )
    permuted = manifest.document_position_ledger_sha256(
        (*indices[:-2], indices[-1], indices[-2]), full_windows=2_083, final_stop=128,
    )
    longer = manifest.document_position_ledger_sha256(
        indices, full_windows=2_083, final_stop=129,
    )
    assert len(original) == 64
    assert original != permuted != longer

