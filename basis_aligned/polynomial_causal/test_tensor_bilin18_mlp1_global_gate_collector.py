from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import mlp1_global_gate_analysis as analysis
import tensor_bilin18_mlp1_global_gate_collector as collector


class FakeLock:
    def __init__(self):
        self.assertions = 0

    def assert_owned(self):
        self.assertions += 1


def test_plan_rows_and_direct_parents_validate_without_gpu() -> None:
    plan, fit, validation = collector._plan_and_rows()
    assert plan["plan_fingerprint"] == collector.EXPECTED_PLAN_FINGERPRINT
    assert fit.shape == validation.shape == (16, 256)
    assert not set(plan["cohorts"]["fit"]["document_ids"]) & set(
        plan["cohorts"]["validation"]["document_ids"]
    )
    assert collector.frozen_plan.tensor_raw_sha256(fit) == plan["cohorts"]["fit"][
        "model_input_256_raw_sha256"
    ]


def test_source_closure_contains_executable_math_graph_analysis_and_transitive_program() -> None:
    names = {path.name for path in collector.SOURCES}
    assert {
        Path(collector.__file__).name,
        "test_tensor_bilin18_mlp1_global_gate_collector.py",
        "mlp1_global_gate_plan.json",
        "MLP1_GLOBAL_GATE_RESPONSE_PREREGISTRATION.md",
        "freeze_mlp1_global_gate_row_use.py",
        "mlp_global_gate_response.py",
        "mlp1_global_gate_analysis.py",
        "tensor_bilin18_global_gate_intervention.py",
        "tensor_bilin18_tangent_pilot.py",
        "tensor_bilin18_program.py",
        "tensor_preserving_attention.py",
        "tensor_preserving_mlp.py",
    } <= names
    assert "prepare_mlp1_global_gate_rows.py" not in names
    assert len(collector.SOURCES) == len(set(collector.SOURCES))

    receipt = json.loads(collector.ROWS_RECEIPT.read_text())
    assert receipt["implementation_hashes"] == collector.EXPECTED_ROWS_IMPLEMENTATION_HASHES


def test_namespaces_are_separate_create_only_paths() -> None:
    assert len(set(collector._namespace_paths())) == 4
    assert collector.AUTHORITY_RECEIPT.name.endswith("authority_receipt.json")
    assert collector.BUNDLE.name.endswith("bundle.pt")
    assert collector.OUTPUT.name.endswith("results.json")
    assert collector.FAILURE.name.endswith("failure.json")


def test_authority_refuses_if_any_outcome_namespace_exists(tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "bundle.pt"
    bundle.write_bytes(b"spent")
    monkeypatch.setattr(collector, "AUTHORITY_RECEIPT", tmp_path / "authority.json")
    monkeypatch.setattr(collector, "BUNDLE", bundle)
    monkeypatch.setattr(collector, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(collector, "FAILURE", tmp_path / "failure.json")
    with pytest.raises(RuntimeError, match="every dedicated namespace absent"):
        collector.freeze_authority(FakeLock(), {})


def test_run_refuses_before_authority_or_if_bundle_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(collector, "AUTHORITY_RECEIPT", tmp_path / "authority.json")
    monkeypatch.setattr(collector, "BUNDLE", tmp_path / "bundle.pt")
    monkeypatch.setattr(collector, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(collector, "FAILURE", tmp_path / "failure.json")
    with pytest.raises(RuntimeError, match="freeze global-gate authority"):
        collector.run(FakeLock(), {})
    collector.AUTHORITY_RECEIPT.write_text("{}")
    collector.BUNDLE.write_bytes(b"spent")
    with pytest.raises(RuntimeError, match="already spent"):
        collector.run(FakeLock(), {})


def test_torch_bundle_publication_is_create_only_and_exact(tmp_path) -> None:
    path = tmp_path / "bundle.pt"
    lock = FakeLock()
    value = {"tensor": torch.arange(5), "nested": {"status": "fit"}}
    collector._publish_torch_create_only(path, value, ownership_check=lock.assert_owned)
    assert lock.assertions == 1
    replay = torch.load(path, map_location="cpu", weights_only=True)
    assert analysis.tensor_tree_equal(value, replay)
    with pytest.raises(FileExistsError):
        collector._publish_torch_create_only(
            path, value, ownership_check=lock.assert_owned,
        )


def test_authority_validation_rejects_parent_buffer_mismatch() -> None:
    parent = json.loads(collector.PARENT_PROGRAM_AUTHORITY.read_text())
    runtime = parent["runtime_environment"]
    snapshot = {"fingerprint": "fixed"}
    value = {
        "status": "mlp1_global_gate_authority_frozen_no_outcomes",
        "protected_snapshot": snapshot,
        "plan_fingerprint": collector.EXPECTED_PLAN_FINGERPRINT,
        "plan_sha256": collector.EXPECTED_PLAN_SHA256,
        "program_receipt": parent["program_receipt"],
        "program_buffers": {"wrong": True},
        "runtime_environment": runtime,
        "namespace": collector._namespace_contract(),
        "namespace_absent_before_authority": {
            name: True for name in ("authority_receipt", "bundle", "result", "failure")
        },
        "product_activations_computed": False,
        "score_targets_sampled": False,
        "score_gradients_computed": False,
        "bundle_computed": False,
        "validation_opened": False,
        "result_computed": False,
    }
    with pytest.raises(RuntimeError, match="program buffers differ"):
        collector.validate_authority(value, snapshot=snapshot, runtime_environment=runtime)


def _native_response_batch_and_receipt():
    first = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    second = first + 1
    tokens = torch.arange(8, dtype=torch.int64).reshape(2, 4)
    batch = SimpleNamespace(first=first, second=second)
    receipt = {
        "status": "complete", "row_ids": ["a", "b"],
        "first_probe_seeds": [1, 2, 3], "second_probe_seeds": [4, 5, 6],
        "probe_halves_disjoint": True,
        "tokens_sha256": collector.intervention.tensor_sha256(tokens),
        "first_target_ids_sha256": "a" * 64,
        "second_target_ids_sha256": "b" * 64,
        "first_response_sha256": collector.intervention.tensor_sha256(first),
        "second_response_sha256": collector.intervention.tensor_sha256(second),
        "response_shape_per_half": [2, 3, 4], "source_site": collector.SOURCE_SITE,
        "score_support": [collector.SCORE_START, collector.SCORE_STOP],
        "forward": {
            "attention_calls": tuple(range(18)), "mlp_calls": tuple(range(18)),
            "source_site": collector.SOURCE_SITE,
            "scale_shared_across_positions": True,
            "context_scales_independent": True,
        },
        "all_token_positions_share_each_gate_scale": True,
        "contexts_have_independent_gate_scale_leaves": True,
        "raw_logits_returned": False, "raw_targets_returned": False,
        "raw_residual_vjps_returned": False, "graph_aliases_revoked": True,
    }
    batch.receipt = receipt
    return tokens, batch, receipt


def test_response_receipt_recomputes_returned_hashes_and_fails_closed() -> None:
    tokens, batch, receipt = _native_response_batch_and_receipt()
    assert collector._validate_response_receipt(
        receipt, batch, tokens=tokens, row_ids=("a", "b"),
        first_seeds=(1, 2, 3), second_seeds=(4, 5, 6), dual=False,
        rms=None, orientation=None, permutation=None,
    ) == 6
    corrupted = dict(receipt)
    corrupted["first_response_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="tensor receipt changed"):
        collector._validate_response_receipt(
            corrupted, batch, tokens=tokens, row_ids=("a", "b"),
            first_seeds=(1, 2, 3), second_seeds=(4, 5, 6), dual=False,
            rms=None, orientation=None, permutation=None,
        )


def test_bundle_and_result_guards_recheck_lock_snapshot_and_hashes(
    tmp_path, monkeypatch,
) -> None:
    authority = tmp_path / "authority.json"
    authority.write_text("authority")
    bundle = tmp_path / "bundle.pt"
    output = tmp_path / "result.json"
    failure = tmp_path / "failure.json"
    monkeypatch.setattr(collector, "AUTHORITY_RECEIPT", authority)
    monkeypatch.setattr(collector, "BUNDLE", bundle)
    monkeypatch.setattr(collector, "OUTPUT", output)
    monkeypatch.setattr(collector, "FAILURE", failure)
    expected = {"fingerprint": "snapshot"}
    monkeypatch.setattr(collector, "protected_snapshot", lambda: dict(expected))
    authority_hash = collector.file_sha256(authority)
    lock = FakeLock()
    collector._bundle_publication_guard(lock, expected, authority_hash)
    bundle.write_bytes(b"bundle")
    bundle_hash = collector.file_sha256(bundle)
    collector._result_publication_guard(lock, expected, authority_hash, bundle_hash)
    failure.write_text("late")
    with pytest.raises(RuntimeError, match="namespace changed"):
        collector._result_publication_guard(lock, expected, authority_hash, bundle_hash)


def test_failure_publication_is_bound_to_authority_snapshot_bundle_and_no_result(
    tmp_path, monkeypatch,
) -> None:
    expected = {"fingerprint": "snapshot"}
    authority = tmp_path / "authority.json"
    authority.write_text(json.dumps({"protected_snapshot": expected}))
    bundle = tmp_path / "bundle.pt"
    bundle.write_bytes(b"frozen bundle")
    monkeypatch.setattr(collector, "AUTHORITY_RECEIPT", authority)
    monkeypatch.setattr(collector, "BUNDLE", bundle)
    monkeypatch.setattr(collector, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(collector, "FAILURE", tmp_path / "failure.json")
    monkeypatch.setattr(collector, "protected_snapshot", lambda: dict(expected))
    authority_hash = collector.file_sha256(authority)
    bundle_hash = collector.file_sha256(bundle)
    lock = FakeLock()
    collector._failure_publication_guard(lock, expected, authority_hash, bundle_hash)
    collector._publish_failure(lock, RuntimeError("late scientific failure"))
    failure = json.loads(collector.FAILURE.read_text())
    assert failure["authority_sha256"] == authority_hash
    assert failure["bundle_sha256"] == bundle_hash
    assert failure["protected_snapshot_fingerprint"] == "snapshot"
    assert not collector.OUTPUT.exists()
    collector.FAILURE.unlink()
    collector.OUTPUT.write_text("collision")
    with pytest.raises(RuntimeError, match="failure namespace changed"):
        collector._failure_publication_guard(lock, expected, authority_hash, bundle_hash)


def _install_mock_run(tmp_path, monkeypatch, *, fail_on_validation=False):
    plan, fit_rows, validation_rows = collector._plan_and_rows()
    expected = {"fingerprint": "source-snapshot"}
    authority = tmp_path / "authority.json"
    authority.write_text(json.dumps({
        "program_buffers": {"manifest_sha256": "manifest"},
        "protected_snapshot": expected,
    }))
    monkeypatch.setattr(collector, "AUTHORITY_RECEIPT", authority)
    monkeypatch.setattr(collector, "BUNDLE", tmp_path / "bundle.pt")
    monkeypatch.setattr(collector, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(collector, "FAILURE", tmp_path / "failure.json")
    monkeypatch.setattr(collector, "protected_snapshot", lambda: dict(expected))
    monkeypatch.setattr(
        collector, "validate_authority", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        collector, "_plan_and_rows", lambda: (plan, fit_rows, validation_rows),
    )
    fake_down = SimpleNamespace(weight=torch.ones(2, 4, dtype=torch.float64))
    fake_program = SimpleNamespace(
        mlp_bank=SimpleNamespace(programs=[SimpleNamespace(), SimpleNamespace(down=fake_down)]),
    )
    receipt = {"program": "frozen"}
    manifest = {"manifest_sha256": "manifest"}
    monkeypatch.setattr(
        collector.parent_pilot, "build_rank640_program",
        lambda device: (fake_program, receipt),
    )
    monkeypatch.setattr(
        collector, "_validate_program_against_parent", lambda *args: manifest,
    )
    monkeypatch.setattr(
        collector.authority_helpers, "program_buffer_manifest", lambda program: manifest,
    )
    monkeypatch.setattr(
        collector, "_collect_products",
        lambda program, rows: (torch.ones(2, 3, 4), [{"batch": i} for i in range(4)]),
    )
    monkeypatch.setattr(
        collector, "_control_gauge",
        lambda products, down, seed: (
            torch.ones(4), torch.ones(4), (1, 0, 3, 2), {"status": "complete"},
        ),
    )
    events = []

    def responses(program, rows, row_ids, **kwargs):
        dual = kwargs.get("rms") is not None
        if dual:
            events.append("fit_responses")
            hashes = {f"fit-{index}" for index in range(8)}
        else:
            assert collector.BUNDLE.exists()
            events.append("validation_responses_after_bundle")
            if fail_on_validation:
                raise RuntimeError("injected validation failure")
            hashes = {f"validation-{index}" for index in range(8)}
        value = torch.ones(2, 3, 4, dtype=torch.float64)
        receipts = [{
            "first_probe_seeds": list(kwargs["first_seeds"]),
            "second_probe_seeds": list(kwargs["second_seeds"]),
        } for _ in range(4)]
        return value, value + 1, value + 2 if dual else None, receipts, hashes

    monkeypatch.setattr(collector, "_collect_cohort_responses", responses)
    fit_summary = {"status": "fit_bundle_complete_no_validation_opened"}
    core_bundle = {"frozen": torch.arange(3)}
    monkeypatch.setattr(
        collector.analysis, "build_fit_gate_bundle",
        lambda *args, **kwargs: (fit_summary, core_bundle),
    )
    monkeypatch.setattr(
        collector.analysis, "validate_fit_gate_bundle", lambda *args, **kwargs: None,
    )
    scientific = {"status": "no_admitted_support", "scope": "mock scientific scope"}
    monkeypatch.setattr(
        collector.analysis, "analyze_global_gate_responses",
        lambda *args, **kwargs: (scientific, core_bundle),
    )
    monkeypatch.setattr(
        collector.analysis, "validate_gate_analysis_result", lambda *args, **kwargs: None,
    )
    return events


def test_mocked_full_run_freezes_bundle_before_validation_and_derives_ledger(
    tmp_path, monkeypatch,
) -> None:
    events = _install_mock_run(tmp_path, monkeypatch)
    result = collector.run(FakeLock(), {})
    assert events == ["fit_responses", "validation_responses_after_bundle"]
    assert result["execution"]["product_batches"] == 4
    assert result["execution"]["fit_response_batches"] == 4
    assert result["execution"]["validation_response_batches"] == 4
    assert result["execution"]["backward_passes"] == 512
    assert result["execution"]["unique_target_hashes"] == 16
    assert result["execution"]["bundle_frozen_before_validation"] is True
    assert result["execution"]["validation_did_not_alter_bundle"] is True


def test_post_bundle_failure_preserves_bundle_and_publishes_only_bound_failure(
    tmp_path, monkeypatch,
) -> None:
    events = _install_mock_run(tmp_path, monkeypatch, fail_on_validation=True)
    lock = FakeLock()
    with pytest.raises(RuntimeError, match="injected validation failure") as caught:
        collector.run(lock, {})
    assert events == ["fit_responses", "validation_responses_after_bundle"]
    assert collector.BUNDLE.exists()
    frozen_hash = collector.file_sha256(collector.BUNDLE)
    collector._publish_failure(lock, caught.value)
    assert collector.file_sha256(collector.BUNDLE) == frozen_hash
    assert collector.FAILURE.exists()
    assert not collector.OUTPUT.exists()
    failure = json.loads(collector.FAILURE.read_text())
    assert failure["bundle_sha256"] == frozen_hash
