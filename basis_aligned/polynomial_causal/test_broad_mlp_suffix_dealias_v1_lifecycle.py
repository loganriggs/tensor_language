import json

import pytest

import broad_mlp_suffix_dealias_v1_lifecycle as lifecycle


def test_parent_artifacts_and_semantics_are_exact():
    value = lifecycle.parent_authority()
    assert value["shared_program_sha256"] == lifecycle.EXPECTED_PARENT["shared_program_sha256"]
    assert value["model_binding"]["component_tree_sha256"] == lifecycle.EXPECTED_PARENT[
        "component_tree_sha256"
    ]


def test_noncanonical_namespace_is_safe_and_pristine(tmp_path):
    paths = lifecycle.output_paths(tmp_path, "test_broad_mlp")
    paths.require_pristine()
    paths.receipt.write_text("{}")
    with pytest.raises(RuntimeError, match="already spent"):
        paths.require_pristine()


def test_parent_mutation_is_rejected(tmp_path, monkeypatch):
    fake = tmp_path / "authority.json"
    fake.write_text(json.dumps({"status": "wrong"}))
    protected = dict(lifecycle.PROTECTED_SHA256)
    protected[fake] = lifecycle.file_sha256(fake)
    monkeypatch.setattr(lifecycle, "PROTECTED_SHA256", protected)
    monkeypatch.setattr(
        lifecycle, "PARENT_PATHS",
        lifecycle.OutputPaths(
            authority=fake, payload=lifecycle.PARENT_PATHS.payload,
            manifest=lifecycle.PARENT_PATHS.manifest,
            receipt=lifecycle.PARENT_PATHS.receipt,
            failure=lifecycle.PARENT_PATHS.failure, lock=lifecycle.PARENT_PATHS.lock,
        ),
    )
    with pytest.raises(RuntimeError, match="semantics changed"):
        lifecycle.parent_authority()
