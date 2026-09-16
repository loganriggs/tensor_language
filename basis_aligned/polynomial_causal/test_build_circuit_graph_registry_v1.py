import json

from basis_aligned.polynomial_causal import build_circuit_graph_registry_v1 as registry


def _package(root, name, manifest=None, verifier=False):
    package = root / name
    package.mkdir()
    if manifest is not None:
        (package / "manifest.json").write_text(json.dumps(manifest))
    if verifier:
        (package / "verify_export.py").write_text("print('not run in unit fixture')\n")
    return package


def test_registry_preserves_unknowns_and_explicit_dependencies(tmp_path):
    _package(tmp_path, "raw_package")
    _package(tmp_path, "partial", {"schema": "partial_v1", "ports": ["x"],
                                    "dependency": "raw_package", "nodes": ["n"]})
    traits = {trait: True for trait in registry.TRAITS}
    _package(tmp_path, "declared", {"schema": "declared_v1", "public_data_inputs": ["tokens"],
                                     "external_activation_inputs": 0, "four_traits": traits}, verifier=True)
    result = registry.build_registry(tmp_path, run_verifiers=False)
    records = {record["package"]: record for record in result["packages"]}
    assert result["counts"] == {"packages": 3, "with_manifest": 2, "with_declared_inputs": 2,
                                 "four_trait_verified": 0, "four_trait_declared_unverified": 1,
                                 "partial_or_unassessed": 2, "explicit_dependency_edges": 1,
                                 "dependency_cycles": 0}
    assert records["raw_package"]["four_traits"] == {trait: None for trait in registry.TRAITS}
    assert records["partial"]["dependencies"] == ["raw_package"]
    assert records["declared"]["maturity"] == "four_trait_declared_unverified"
    assert records["declared"]["external_activation_inputs"] == 0


def test_dependency_cycles_are_reported(tmp_path):
    _package(tmp_path, "a", {"dependency": "b"})
    _package(tmp_path, "b", {"dependency": "a"})
    result = registry.build_registry(tmp_path, run_verifiers=False)
    assert result["dependency_cycles"] == [["a", "b", "a"]]
    assert result["counts"]["dependency_cycles"] == 1


def test_markdown_exposes_verified_and_missing_boundaries(tmp_path):
    _package(tmp_path, "unknown")
    result = registry.build_registry(tmp_path, run_verifiers=False)
    rendered = registry.render_markdown(result)
    assert "unknown | partial_or_unassessed" in rendered
    assert "Missing evidence is recorded as unknown" in rendered
