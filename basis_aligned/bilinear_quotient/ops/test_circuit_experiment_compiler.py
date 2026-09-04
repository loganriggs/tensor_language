#!/usr/bin/env python3
# BQLANE: cpu
"""Shared model-free tests and exact shadow parity for the circuit compiler."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import circuit_artifact_package as package
import circuit_experiment_spec as compiler
import circuit_managed_entry as managed
import induction_selector_payload_frozen_factor_rung585_manifest as r585
import induction_selector_payload_three_source_rows_rung578 as r578
import numbered_list_cached_value_downstream_use_rung590 as r590


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
R578_ROWS = ROOT / "induction_selector_payload_three_source_rows_rung578.json"
R590_DRYRUN = ROOT / "numbered_list_cached_value_downstream_use_rung590_dryrun.json"
ZERO_SHA = "0" * 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimal_spec(**updates) -> compiler.CircuitExperimentSpec:
    base = compiler.CircuitExperimentSpec(
        experiment_id="synthetic-circuit",
        rung=1,
        artifacts=(),
        phases=(compiler.PhaseSpec("FIT"),),
        authority_tables=(),
        calls=(),
    )
    return replace(base, **updates)


def table_spec(name, identity_fields, records, **kwargs):
    return compiler.AuthorityTableSpec(
        name, identity_fields, expected_records_sha256=compiler.canonical_sha256(records),
        **kwargs,
    )


def arm_specs(names, *, native=(), counterfactual=(), null=()):
    return tuple(compiler.ArmSpec(
        name, "native" if name in native else "counterfactual" if name in counterfactual
        else "null" if name in null else "control", "undirected"
    ) for name in names)


def r590_call_families() -> tuple[compiler.CallFamilySpec, ...]:
    output = []
    for split in ("FIT", "SELECT"):
        selected_guard = "fit_always" if split == "FIT" else "selected_only"
        capture_arms = ("source_present", "source_deleted", "native_smoke")
        output.append(compiler.CallFamilySpec(
            name=f"{split.lower()}_capture", split=split,
            arms=capture_arms,
            arm_call_kinds=(("native_smoke", "native_logits_smoke"),),
            arm_batch_limits=(("native_smoke", 1),),
            batch_size=24, call_kind="trajectory", guard=selected_guard,
            call_id_template="{split}:capture:{batch}:{arm}",
            axis_order="batch_arm", sort_policy="legacy_python_repr",
            arm_specs=arm_specs(capture_arms, native=("source_present",),
                                counterfactual=("source_deleted",)),
        ))
        real_arms = r590.SELECTION_NAMES if split == "FIT" else tuple(
            f"selected_site_{component}" for component in r590.COMPONENTS
        )
        output.append(compiler.CallFamilySpec(
            name=f"{split.lower()}_real", split=split, arms=tuple(real_arms),
            batch_size=24, call_kind="component_suffix", guard=selected_guard,
            call_id_template="{split}:real:{arm}:{batch}",
            axis_order="arm_batch", sort_policy="legacy_python_repr",
            arm_specs=arm_specs(real_arms),
        ))
        output.append(compiler.CallFamilySpec(
            name=f"{split.lower()}_null", split=split, arms=tuple(r590.NULLS),
            batch_size=24, call_kind="null_component_suffix",
            guard="provisional_only" if split == "FIT" else "selected_only",
            call_id_template="{split}:null:{arm}:{batch}",
            filters=(("condition", tuple(sorted(r590.r588.ELIGIBLE_CONDITIONS))),),
            axis_order="arm_batch", sort_policy="legacy_python_repr",
            arm_specs=arm_specs(tuple(r590.NULLS), null=tuple(r590.NULLS)),
        ))
    return tuple(output)


def test_production_vertical_slice_stays_below_kill_criterion() -> None:
    paths = (
        OPS / "circuit_experiment_spec.py",
        OPS / "circuit_artifact_package.py",
        OPS / "circuit_managed_entry.py",
    )
    lines = sum(len(path.read_text().splitlines()) for path in paths)
    assert lines <= 1_200


def test_r578_rows_regenerate_and_compile_exact_identities() -> None:
    saved = json.loads(R578_ROWS.read_text())
    regenerated = r578.build_dataset()
    assert regenerated == saved
    rows = saved["rows"]
    counts = {split: sum(row["split"] == split for row in rows) for split in r578.SPLITS}
    compiled = compiler.compile_authority_tables(
        (table_spec(
            "rows", ("row_id",), group_fields=("group_id",),
            expected_counts=counts, expected_total=len(rows), records=rows,
        ),),
        {"rows": rows},
    )["rows"]
    assert compiled["count"] == saved["row_count"] == 5_400
    assert compiled["ordered_identities"] == [row["row_id"] for row in rows]
    assert compiled["records_sha256"] == compiler.canonical_sha256(rows)
    assert sha256(R578_ROWS) == r585.ROWS_SHA256


def test_r585_authority_and_manifest_shadow_parity() -> None:
    authority = r585.build_authority_manifest()
    manifests = r585.build_cell_manifests(authority)
    scale_lookup = r585.build_control_scale_lookup(manifests)
    bootstrap = r585.expected_bootstrap_cells(manifests)
    saved = r585.build_dryrun()
    tables = {
        "rows": authority["rows"],
        "endpoints": authority["endpoints"],
        "directions": authority["directions"],
        "target_cells": manifests["target_cells"],
        "control_cells": manifests["control_cells"],
        "coverage_keys": manifests["coverage_keys"],
        "eligible_control_arm_cells": manifests["eligible_control_arm_cells"],
        "structural_identities": manifests["structural_identities"],
        "control_scale_lookup": scale_lookup,
        "bootstrap": bootstrap,
    }
    table_specs = (
        table_spec("rows", ("row_id",), tables["rows"], group_fields=("group_id",), expected_counts={"FIT": 1872, "SELECT": 936}),
        table_spec("endpoints", ("split", "endpoint_id"), tables["endpoints"], expected_counts={"FIT": 1728, "SELECT": 864}),
        table_spec("directions", ("directed_id",), tables["directions"], group_fields=("group_id",), expected_counts={"FIT": 3744, "SELECT": 1872}),
        table_spec("target_cells", ("cell_id",), tables["target_cells"], expected_counts={"FIT": 20, "SELECT": 20}),
        table_spec("control_cells", ("cell_id",), tables["control_cells"], expected_counts={"FIT": 32, "SELECT": 32}),
        table_spec("coverage_keys", ("split", "arm", "direction", "recipient_condition"), tables["coverage_keys"], expected_counts={"FIT": 24, "SELECT": 24}),
        table_spec("eligible_control_arm_cells", ("cell_id", "arm"), tables["eligible_control_arm_cells"], expected_counts={"FIT": 88, "SELECT": 88}),
        table_spec("structural_identities", ("cell_id", "left_arm", "right_arm"), tables["structural_identities"], split_field=None, expected_total=64),
        table_spec("control_scale_lookup", ("split", "control_cell_id", "arm"), tables["control_scale_lookup"], expected_counts={"FIT": 96, "SELECT": 96}),
        table_spec("bootstrap", ("cell_id",), tables["bootstrap"], split_field=None, expected_total=248),
    )
    compiled = compiler.compile_authority_tables(table_specs, tables)
    assert compiled["directions"]["records_sha256"] == saved["direction_manifest_sha256"]
    assert compiled["target_cells"]["records_sha256"] == saved["target_cell_manifest_sha256"]
    assert compiled["control_cells"]["records_sha256"] == saved["control_cell_manifest_sha256"]
    assert compiled["structural_identities"]["records_sha256"] == saved["structural_identity_manifest_sha256"]
    assert compiled["control_scale_lookup"]["records_sha256"] == saved["control_scale_lookup_sha256"]
    assert compiled["bootstrap"]["ordered_identities_sha256"] == saved["bootstrap_cell_ids_sha256"]
    assert saved["authority_counts"] == {
        split: {
            "rows": compiled["rows"]["counts_by_split"][split],
            "endpoints": compiled["endpoints"]["counts_by_split"][split],
            "directions": compiled["directions"]["counts_by_split"][split],
        }
        for split in ("FIT", "SELECT")
    }


def test_r590_call_manifest_guards_shapes_and_price_shadow_exactly() -> None:
    rows = r590.load_outcome_blind_authority()
    expected_calls = r590.build_forward_call_manifest(rows)
    observed_calls = compiler.compile_call_manifest(rows, r590_call_families())
    assert observed_calls == expected_calls
    summary = compiler.summarize_call_manifest(observed_calls)
    dryrun = json.loads(R590_DRYRUN.read_text())
    expected = dryrun["forward_call_shape_contract"]
    for key in (
        "call_count", "call_kind_counts", "guard_counts", "shape_counts",
        "maximum_batch_size", "manifest_sha256",
    ):
        assert summary[key] == expected[key]
    assert summary["call_count"] == 510
    assert summary["guard_counts"] == {
        "fit_always": 379, "provisional_only": 40, "selected_only": 91
    }
    assert dryrun["fit_maximum_forwards"] == (
        summary["guard_counts"]["fit_always"]
        + summary["guard_counts"]["provisional_only"]
    ) == 419
    assert dryrun["conditional_select_maximum_forwards"] == (
        summary["guard_counts"]["selected_only"]
    ) == 91
    assert dryrun["literal_executable_maximum_forwards"] == summary["call_count"] == 510
    assert dryrun["model_forwards"] == dryrun["model_backwards"] == 0
    assert dryrun["model_weights_updated"] is False


def test_compile_rejects_diagnostic_predicate_with_unretained_input() -> None:
    spec = minimal_spec(
        arrays=(compiler.ArraySpec("full_logits", ("joint",), "float32", ("batch", 50257), False),),
        predicates=(compiler.PredicateSpec(
            "structural_output_identity_failed", "FIT", 0, "science.check",
            ("full_logits",), "diagnostic", "evidence",
        ),),
    )
    with pytest.raises(compiler.SpecError, match="unretained"):
        compiler.validate_spec(spec)
    compiler.validate_spec(replace(
        spec, predicates=(replace(spec.predicates[0], disposition="hard_abort"),)
    ))


def r592_manifest() -> list[dict[str, object]]:
    endpoint_rows = [
        {"row_id": f"endpoint-{index}", "split": "SELECT", "ids": [index % 10] * 30}
        for index in range(48)
    ]
    directed_rows = [
        {"row_id": f"direction-{index}", "split": "SELECT", "ids": [index % 10] * 30}
        for index in range(48)
    ]
    endpoint = compiler.CallFamilySpec(
        "endpoint", "SELECT", ("capture",), 32, "endpoint", "selected_only",
        "{split}:endpoint:{batch}:{arm}", axis_order="batch_arm",
        arm_specs=arm_specs(("capture",)),
    )
    directed = compiler.CallFamilySpec(
        "directed", "SELECT", ("native", "replay", "score", "payload", "joint"),
        32, "directed", "selected_only", "{split}:directed:{batch}:{arm}",
        axis_order="batch_arm",
        arm_specs=arm_specs(("native", "replay", "score", "payload", "joint"),
                            native=("native",),
                            counterfactual=("replay", "score", "payload", "joint")),
    )
    # Families use different source tables in a real compiled experiment.  The
    # synthetic combines unique IDs and filters to exercise only prefix logic.
    endpoint_rows = [dict(row, kind="endpoint") for row in endpoint_rows]
    directed_rows = [dict(row, kind="directed") for row in directed_rows]
    endpoint = replace(endpoint, filters=(("kind", ("endpoint",)),))
    directed = replace(directed, filters=(("kind", ("directed",)),))
    return compiler.compile_call_manifest(endpoint_rows + directed_rows, (endpoint, directed))


@pytest.mark.parametrize("arm", ("native", "replay", "score", "payload", "joint"))
def test_r592_failure_after_every_arm_has_one_unpadded_prefix(arm: str) -> None:
    manifest = r592_manifest()
    index = next(
        i for i, call in enumerate(manifest)
        if call["call_id"] == f"SELECT:directed:0:{arm}"
    )
    prefix = manifest[: index + 1]
    directories = [
        package.call_directory_name(i, str(call["call_id"]))
        for i, call in enumerate(prefix)
    ]
    package.validate_call_prefix(manifest, prefix, directories)
    assert prefix[-1]["arm"] == arm
    assert not any(
        call["call_id"].startswith("SELECT:directed:1:") for call in prefix
    )


def test_r592_literal_batch16_tail_and_bad_prefix_rejected() -> None:
    manifest = r592_manifest()
    tail = [call for call in manifest if call["call_id"].startswith("SELECT:directed:1:")]
    assert len(tail) == 5 and {call["logical_batch_size"] for call in tail} == {16}
    prefix = manifest[:-1]
    directories = [package.call_directory_name(i, str(call["call_id"])) for i, call in enumerate(prefix)]
    package.validate_call_prefix(manifest, prefix, directories)
    with pytest.raises(package.PackageError, match="prefix"):
        package.validate_call_prefix(manifest, prefix[:-1] + [manifest[-1]], directories)


def write_three_nonfinite_arrays(path: Path) -> None:
    np.save(path / "logits.npy", np.array([[0.0, np.nan]], dtype=np.float32))
    np.save(path / "hook_deltas.npy", np.array([[[np.inf, 1.0]]], dtype=np.float32))
    np.save(path / "planned_hook_deltas.npy", np.array([[[-np.inf, 1.0]]], dtype=np.float32))
    np.save(path / "tokens.npy", np.zeros((1, 30), dtype=np.int64))


def test_r592_three_nonfinite_arrays_roundtrip_exactly(tmp_path: Path) -> None:
    write_three_nonfinite_arrays(tmp_path)
    entries = package.write_nonfinite_masks(tmp_path)
    assert [entry["raw_filename"] for entry in entries] == [
        "hook_deltas.npy", "logits.npy", "planned_hook_deltas.npy"
    ]
    assert len({entry["mask_filename"] for entry in entries}) == 3
    package.validate_nonfinite_masks(tmp_path, "nonfinite_observation")


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "path", "shape", "bytes", "hash", "count", "first", "content"),
)
def test_r592_nonfinite_mask_attacks_fail(tmp_path: Path, mutation: str) -> None:
    write_three_nonfinite_arrays(tmp_path)
    entries = package.write_nonfinite_masks(tmp_path)
    index_path = tmp_path / "nonfinite_mask_index.json"
    if mutation == "missing":
        (tmp_path / entries[0]["mask_filename"]).unlink()
    elif mutation == "extra":
        np.save(tmp_path / "nonfinite_masks/extra.mask.npy", np.ones((1,), dtype=np.bool_))
    elif mutation == "path":
        entries[0]["mask_filename"] = "nonfinite_masks/../escape.mask.npy"
    elif mutation == "shape":
        entries[0]["shape"] = [999]
    elif mutation == "bytes":
        entries[0]["mask_byte_length"] += 1
    elif mutation == "hash":
        entries[0]["mask_sha256"] = ZERO_SHA
    elif mutation == "count":
        entries[0]["nonfinite_count"] += 1
    elif mutation == "first":
        entries[0]["first_lexicographic_coordinate"] = [9, 9, 9]
    else:
        mask_path = tmp_path / entries[0]["mask_filename"]
        mask = np.load(mask_path, allow_pickle=False)
        mask.flat[1] = ~mask.flat[1]
        np.save(mask_path, mask, allow_pickle=False)
        entries[0]["mask_sha256"] = sha256(mask_path)
        entries[0]["nonfinite_count"] = int(mask.sum())
        entries[0]["first_lexicographic_coordinate"] = package.first_true_coordinate(mask)
    if mutation not in {"missing", "extra"}:
        index_path.write_bytes(compiler.canonical_json_bytes(entries) + b"\n")
    with pytest.raises(package.PackageError):
        package.validate_nonfinite_masks(tmp_path, "nonfinite_observation")


def test_masks_absent_for_other_predicates(tmp_path: Path) -> None:
    np.save(tmp_path / "logits.npy", np.zeros((1, 2), dtype=np.float32))
    package.validate_nonfinite_masks(tmp_path, "centered_hook_delta_failed")
    write_three_nonfinite_arrays(tmp_path)
    package.write_nonfinite_masks(tmp_path)
    with pytest.raises(package.PackageError, match="finite predicate"):
        package.validate_nonfinite_masks(tmp_path, "centered_hook_delta_failed")


def test_projection_is_recomputed_from_primitive_evidence() -> None:
    evidence = {"values": [1.0, 2.0, 3.0]}
    projector = lambda item: {"mean": sum(item["values"]) / len(item["values"])}
    assert package.validate_science_projection(evidence, {"mean": 2.0}, projector) == {"mean": 2.0}
    with pytest.raises(package.PackageError, match="projection"):
        package.validate_science_projection(evidence, {"mean": 99.0}, projector)


def package_paths(root: Path) -> package.PackagePaths:
    return package.PackagePaths(
        root=root, result=root / "result.json", receipt=root / "receipt.json",
        evidence=root / "evidence", namespace="fixture",
    )


def test_atomic_package_receipt_last_and_rollback(tmp_path: Path) -> None:
    paths = package_paths(tmp_path)
    stage = package.stage_package(
        paths, evidence_files={"rows/a.bin": b"a", "rows/b.bin": b"b"},
        result={"schema": "fixture"},
    )
    with pytest.raises(RuntimeError, match="crash"):
        package.publish_staged_package(
            stage, paths,
            crash=lambda label: (_ for _ in ()).throw(RuntimeError("crash"))
            if label == "published:result" else None,
        )
    assert not paths.result.exists() and not paths.receipt.exists() and not paths.evidence.exists()
    package.publish_staged_package(stage, paths)
    assert package.validate_complete_package(paths)["schema"] == "fixture"


def test_complete_package_rejects_extra_evidence(tmp_path: Path) -> None:
    paths = package_paths(tmp_path)
    stage = package.stage_package(
        paths, evidence_files={"rows/a.bin": b"a"}, result={"schema": "fixture"},
    )
    package.publish_staged_package(stage, paths)
    (paths.evidence / "unlisted.bin").write_bytes(b"not bound")
    with pytest.raises(package.PackageError, match="extra"):
        package.validate_complete_package(paths)


def test_recovery_reclaims_only_recognized_incomplete_publication(tmp_path: Path) -> None:
    paths = package_paths(tmp_path)
    stage = package.stage_package(
        paths, evidence_files={"rows/a.bin": b"a"}, result={"schema": "fixture"},
    )
    # Model a process death after the first rename rather than an exception,
    # because in-process exceptions are rolled back by publish_staged_package.
    paths.evidence.parent.mkdir(parents=True, exist_ok=True)
    (stage / "evidence").replace(paths.evidence)
    package.recover_stale_publication(stage, paths)
    assert not stage.exists() and not paths.evidence.exists()

    stage = package.stage_package(
        paths, evidence_files={"rows/a.bin": b"a"}, result={"schema": "fixture"},
    )
    package.publish_staged_package(stage, paths)
    arbitrary = tmp_path / ".fixture-stage-arbitrary"
    arbitrary.mkdir()
    (arbitrary / "marker.json").write_bytes(b"arbitrary")
    with pytest.raises(package.PackageError, match="unrecognized"):
        package.recover_stale_publication(arbitrary, paths)
    assert paths.receipt.exists() and arbitrary.exists()


def test_recovery_restores_arbitrary_partial_bytes_instead_of_deleting(tmp_path: Path) -> None:
    paths = package_paths(tmp_path)
    stage = package.stage_package(
        paths, evidence_files={"rows/a.bin": b"a"}, result={"schema": "fixture"},
    )
    shutil_target = stage / "evidence"
    # Mimic an incomplete rename, then replace the final evidence with unrelated bytes.
    shutil_target.rename(paths.evidence)
    (paths.evidence / "rows/a.bin").write_bytes(b"arbitrary")
    with pytest.raises(package.PackageError):
        package.recover_stale_publication(stage, paths)
    assert (paths.evidence / "rows/a.bin").read_bytes() == b"arbitrary"
    assert stage.exists()


def test_managed_entry_executes_captured_not_reopened_bytes(tmp_path: Path) -> None:
    source_path = tmp_path / "producer.py"
    old = b"def run_dryrun(): return 'old'\ndef run_science(): return 'science-old'\n"
    source_path.write_bytes(old)
    artifact = compiler.ArtifactRef(
        "producer", "producer.py", hashlib.sha256(old).hexdigest(), "source",
        executable=True, dryrun_access=True,
    )
    spec = minimal_spec(artifacts=(artifact,))
    captured = managed.capture_frozen_artifacts(spec, base_dir=tmp_path)
    source_path.write_text("def run_dryrun(): return 'new'\ndef run_science(): return 'science-new'\n")
    modules = managed.load_verified_modules(
        spec, captured, (managed.ModuleBinding("producer", "fixture_producer"),)
    )
    assert modules["producer"].run_dryrun() == "old"
    source_path.write_bytes(old)
    assert managed.dispatch(
        spec, base_dir=tmp_path,
        bindings=(managed.ModuleBinding("producer", "fixture_producer_dispatch"),),
        producer_role="producer", environment={"BQLIB_DRYRUN": "1"},
    ) == "old"


def test_managed_dryrun_does_not_open_declared_outcome(tmp_path: Path) -> None:
    source = b"def run_dryrun(): return 'outcome-blind'\ndef run_science(): return 'science'\n"
    (tmp_path / "producer.py").write_bytes(source)
    spec = minimal_spec(artifacts=(
        compiler.ArtifactRef(
            "producer", "producer.py", hashlib.sha256(source).hexdigest(), "source",
            executable=True, dryrun_access=True,
        ),
        # Deliberately absent: success proves dry run neither opens nor hashes it.
        compiler.ArtifactRef("historical_result", "absent-result.json", ZERO_SHA, "outcome"),
    ))
    assert managed.dispatch(
        spec, base_dir=tmp_path,
        bindings=(managed.ModuleBinding("producer", "fixture_outcome_blind"),),
        producer_role="producer", environment={"BQLIB_DRYRUN": "1"},
    ) == "outcome-blind"


def test_outcome_bearing_dryrun_dependency_is_rejected() -> None:
    spec = minimal_spec(artifacts=(compiler.ArtifactRef(
        "prior_result", "prior.json", ZERO_SHA, "outcome", dryrun_access=True,
    ),))
    with pytest.raises(compiler.SpecError, match="outcome"):
        compiler.validate_spec(spec)


def test_native_typed_contracts_cannot_bypass_required_semantics() -> None:
    rows = [{"row_id": "r", "split": "FIT", "ids": [1]}]
    with pytest.raises(TypeError):
        compiler.ArmSpec("native")
    with pytest.raises(TypeError):
        compiler.AuthorityTableSpec("rows", ("row_id",))
    with pytest.raises(TypeError):
        compiler.PredicateSpec("p", "FIT", 0, "eval", (), "hard_abort")
    with pytest.raises(TypeError):
        compiler.CallFamilySpec(
            "f", "FIT", ("native",), 1, "margin", "fit_always", "{split}:{arm}:{batch}"
        )
    family = compiler.CallFamilySpec(
        "f", "FIT", ("native",), 1, "margin", "fit_always", "{split}:{arm}:{batch}", ()
    )
    with pytest.raises(compiler.SpecError, match="typed arm roles"):
        compiler.compile_call_manifest(rows, (family,))
    with pytest.raises(compiler.SpecError, match="digest"):
        compiler.compile_authority_tables(
            (compiler.AuthorityTableSpec("rows", ("row_id",), None, expected_counts={"FIT": 1}),),
            {"rows": rows},
        )
    predicate = compiler.PredicateSpec(
        "instrument", "FIT", 0, "instrument_eval", ("margin",), "hard_abort", None
    )
    with pytest.raises(compiler.SpecError, match="kind"):
        compiler.validate_spec(minimal_spec(
            arrays=(compiler.ArraySpec("margin", ("margin",), "float64", ("batch",), True),),
            predicates=(predicate,),
        ))


def _write_call(root: Path, index: int, *, arm: str, role: str, rows: list[str],
                values: np.ndarray, extra: bool = False) -> Path:
    directory = root / package.call_directory_name(index, f"FIT:{arm}:{index % 2}")
    directory.mkdir(parents=True)
    record = {
        "call_id": f"FIT:{arm}:{index % 2}", "split": "FIT", "arm": arm,
        "call_kind": "margin", "call_family": "fit_margin", "arm_role": role,
        "arm_direction": "undirected", "row_ids": rows,
        "logical_batch_size": len(rows), "padded_sequence_length": 4,
        "array_contracts": [{"name": "margin", "dtype": "float64", "shape": ["batch"],
                             "finite_policy": "always"}],
    }
    (directory / "call.json").write_bytes(compiler.canonical_json_bytes(record) + b"\n")
    np.save(directory / "margin.npy", values.astype(np.float64))
    if extra:
        np.save(directory / "extra.npy", np.zeros(len(rows)))
    return directory


def test_dead_arm_is_family_scoped_and_extra_array_cannot_hide_it(tmp_path: Path) -> None:
    calls = tmp_path / "calls"; calls.mkdir()
    native0 = _write_call(calls, 0, arm="native", role="native", rows=["a"], values=np.array([1.]))
    native1 = _write_call(calls, 1, arm="native", role="native", rows=["b"], values=np.array([2.]))
    cf0 = _write_call(calls, 2, arm="counterfactual", role="counterfactual", rows=["a"], values=np.array([1.]))
    cf1 = _write_call(calls, 3, arm="counterfactual", role="counterfactual", rows=["b"],
                      values=np.array([3.]), extra=True)
    package.validate_nonfinite_masks(cf0, "ok")  # one coincident batch is not a dead family
    np.save(cf1 / "margin.npy", np.load(native1 / "margin.npy"))
    with pytest.raises(package.PackageError, match="dead"):
        package.validate_nonfinite_masks(cf0, "ok")
    assert native0.exists()


def _decision_spec() -> compiler.CircuitExperimentSpec:
    return minimal_spec(science=compiler.ScienceProjectionSpec(
        "projector", "decision", ("ok", "hard_abort"), {"score": "number"}
    ))


def test_projector_middle_order_and_captured_environment_are_rejected(monkeypatch) -> None:
    calls = [{"call_id": f"FIT:{index}", "split": "FIT"} for index in range(3)]
    evidence = [{"call_id": f"FIT:{index}", "margin": float(index)} for index in range(3)]
    middle = lambda rows: {"score": rows[len(rows) // 2]["margin"]}
    with pytest.raises(package.PackageError, match="order"):
        package.decide_experiment(
            spec=_decision_spec(), compiled={"predicate_order": [], "call_manifest": calls},
            primitives=evidence, evaluators={}, projector=middle,
        )
    monkeypatch.setenv("CIRCUIT_SNAPSHOT", "secret")
    snapshot = os.environ["CIRCUIT_SNAPSHOT"]
    captured = lambda rows: {"score": float(len(snapshot))}
    with pytest.raises(package.PackageError, match="pure"):
        package.decide_experiment(
            spec=_decision_spec(), compiled={"predicate_order": [], "call_manifest": calls},
            primitives=evidence, evaluators={}, projector=captured,
        )


def test_unknown_primitive_and_vacuous_phase_predicate_fail_closed() -> None:
    calls = [{"call_id": "FIT:0", "split": "FIT"}]
    with pytest.raises(package.PackageError, match="unknown call"):
        package.decide_experiment(
            spec=_decision_spec(), compiled={"predicate_order": [], "call_manifest": calls},
            primitives=[{"call_id": "SELECT:unknown"}], evaluators={}, projector=lambda rows: {"score": 0.},
        )
    predicate = compiler.PredicateSpec(
        "fit_live", "FIT", 0, "live", (), "hard_abort", "instrument"
    )
    spec = replace(_decision_spec(), predicates=(predicate,))
    with pytest.raises(package.PackageError, match="vacuous"):
        package.decide_experiment(
            spec=spec, compiled={"predicate_order": ["fit_live"],
                                 "call_manifest": [{"call_id": "SELECT:0", "split": "SELECT"}]},
            primitives=[{"call_id": "SELECT:0"}], evaluators={"live": lambda rows: True},
            projector=lambda rows: {"score": 0.},
        )


def test_array_policy_and_full_physical_shape_are_enforced(tmp_path: Path) -> None:
    call = _write_call(tmp_path, 0, arm="native", role="native", rows=["a"], values=np.array([1.]))
    record = json.loads((call / "call.json").read_text())
    record["array_contracts"][0]["shape"] = ["batch", "sequence"]
    (call / "call.json").write_bytes(compiler.canonical_json_bytes(record) + b"\n")
    np.save(call / "margin.npy", np.zeros((1, 3)))
    with pytest.raises(package.PackageError, match="physical shape"):
        package.validate_nonfinite_masks(call, "ok")
    record["array_contracts"][0].update(shape=["batch"], finite_policy="always")
    (call / "call.json").write_bytes(compiler.canonical_json_bytes(record) + b"\n")
    np.save(call / "margin.npy", np.array([np.nan]))
    with pytest.raises(package.PackageError, match="finite policy"):
        package.validate_nonfinite_masks(call, "nonfinite_observation")
