from __future__ import annotations

import inspect

import pytest

import validate_mlp2_cmr_v1 as validation


def test_source_closure_is_exact_once_and_contains_runtime_contracts() -> None:
    assert len(validation.SOURCE_CLOSURE) == len(set(validation.SOURCE_CLOSURE))
    names = {path.name for path in validation.SOURCE_CLOSURE}
    assert names == {
        "MLP2_CMR_V1_PREREGISTRATION.md",
        "MLP2_CMR_V1_VALIDATION_ADDENDUM.md",
        "validate_mlp2_cmr_v1.py",
        "test_validate_mlp2_cmr_v1.py",
        "mlp2_cmr_v1_validation_runtime.py",
        "test_mlp2_cmr_v1_validation_runtime.py",
        "mlp2_cmr_v1_validation_statistics.py",
        "test_mlp2_cmr_v1_validation_statistics.py",
        "mlp2_cmr_v1_physical_program.py",
        "test_mlp2_cmr_v1_physical_program.py",
        "project_mlp2_cmr_v1_validation_rows.py",
        "test_project_mlp2_cmr_v1_validation_rows.py",
        "MLP2_CMR_V1_MARGIN_FREQUENCY_ADDENDUM.md",
        "materialize_mlp2_cmr_v1_token_rows.py",
        "test_materialize_mlp2_cmr_v1_token_rows.py",
        "project_mlp2_cmr_v1_fit_selector_rows.py",
        "test_project_mlp2_cmr_v1_fit_selector_rows.py",
        "bilin18_observed_model_facade.py",
        "tt_model.py",
    }


def test_selector_gauge_replay_checks_zero_error_as_a_number_not_false() -> None:
    valid = {
        "gauge_and_permutation_audit": {
            "channel_permutation": {
                "derangement_equivariant": True,
                "hash_random_equivariant": True,
                "suffix_support_equivariant": True,
            },
            "dyadic_reciprocal": {
                "canonical_down_max_abs_error": 0.0,
                "derangement_exact": True,
                "hash_random_exact": True,
            },
            "general_reciprocal_functional": {
                "canonical_down_max_relative_error": 4.8e-16,
                "hash_byte_replay_required": False,
            },
        },
    }
    assert validation._selector_gauge_passes(valid)
    invalid = {
        "gauge_and_permutation_audit": {
            **valid["gauge_and_permutation_audit"],
            "dyadic_reciprocal": {
                **valid["gauge_and_permutation_audit"]["dyadic_reciprocal"],
                "derangement_exact": False,
            },
        },
    }
    assert not validation._selector_gauge_passes(invalid)


def test_runner_is_role_only_and_never_names_combined_or_replication_rows() -> None:
    source = inspect.getsource(validation)
    assert "mlp2_cmr_v1_token_rows.pt" not in source
    assert "mlp2_cmr_v1_replication" not in source
    assert '"authorized_role": "VALIDATION"' in source
    assert '"authorized_for_replication_execution": False' in source
    assert '"raw_logits_published": False' in source
    assert '"per_token_losses_published": False' in source
    assert '"validation_targets_published": False' in source


def test_expected_parent_hashes_are_complete_exact_and_files_exist() -> None:
    assert set(validation.PARENT_PATHS) == set(validation.EXPECTED_PARENTS)
    assert all(path.is_file() for path in validation.PARENT_PATHS.values())
    assert {
        name: validation.file_sha256(path)
        for name, path in validation.PARENT_PATHS.items()
    } == validation.EXPECTED_PARENTS


def test_capability_is_closed_one_use_and_noncopyable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(validation, "LOCK", tmp_path / "lock")
    monkeypatch.setattr(validation, "AUTHORITY", tmp_path / "authority")
    validation.projection.base.write_create_only(
        validation.LOCK, validation.canonical_json_bytes({"nonce": "n"}),
    )
    stat = validation.LOCK.stat(follow_symlinks=False)
    inode = (stat.st_dev, stat.st_ino)
    authority = {
        "status": "authority_frozen_before_validation_model_access",
        "authorized_role": "VALIDATION",
        "authorized_forward_calls": {
            arm: validation.runtime.CALLS for arm in validation.runtime.ALL_ARMS
        },
    }
    validation.projection.base.write_create_only(
        validation.AUTHORITY, validation.canonical_json_bytes(authority),
    )
    authority_hash = validation.file_sha256(validation.AUTHORITY)
    capability = validation._mint_capability("n", inode, authority_hash)
    with pytest.raises(TypeError, match="constructible"):
        validation._ValidationCapability(object(), "n", inode, authority_hash)
    validation._consume_capability(capability)
    with pytest.raises(RuntimeError, match="fresh"):
        validation._consume_capability(capability)
