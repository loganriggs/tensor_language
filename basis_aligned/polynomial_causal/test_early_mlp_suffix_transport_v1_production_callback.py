from __future__ import annotations

import inspect

import pytest
import torch

import bilin18_observed_adapter as observed
import early_mlp_suffix_transport_v1_diagnostic_integration as integration
import early_mlp_suffix_transport_v1_final as final_owner
import early_mlp_suffix_transport_v1_final_capability as final_capability
import early_mlp_suffix_transport_v1_final_execution as final_execution
import early_mlp_suffix_transport_v1_observational_authority as authority
import early_mlp_suffix_transport_v1_observational_execution as observational_execution
import early_mlp_suffix_transport_v1_production_callback as production
import early_mlp_suffix_transport_v1_response_execution as response_execution
import early_mlp_suffix_transport_v1_statistics as statistics


CONTEXT = "1" * 64
PROGRAM = "2" * 64
SUPPORT = "3" * 64
BUNDLE = "4" * 64
RESPONSE = "5" * 64
INTEGRATED = "6" * 64
COMPONENT = "7" * 64
PRODUCER = "8" * 64


def _binding() -> production.FinalDecisionReplayAuthority:
    return production.FinalDecisionReplayAuthority(
        final_context_sha256=CONTEXT, program_payload_sha256=PROGRAM,
        common_support_sha256=SUPPORT, producer_source_sha256=PRODUCER,
    )


def _evidence(**changes) -> production.FinalDecisionReplayEvidence:
    values = {
        "final_context_sha256": CONTEXT,
        "program_payload_sha256": PROGRAM,
        "common_support_sha256": SUPPORT,
        "observation_bundle_sha256": BUNDLE,
        "response_run_receipt_sha256": RESPONSE,
        "integrated_receipt_sha256": INTEGRATED,
        "objective_gates": {name: False for name in final_owner.OBJECTIVE_GATES},
        "transport_observational_gates": {
            name: False for name in statistics.TRANSPORT_OBSERVATIONAL_GATES
        },
        "numerical_payload": {"registered_statistics": {"point": 0.25}},
        "outer_model_returned": True,
        "hooks_restored": True,
        "hooks_inert": True,
        "component_tree_before_sha256": COMPONENT,
        "component_tree_after_sha256": COMPONENT,
        "student_poison_closed": True,
        "gauge_replay_differences": tuple(
            torch.tensor([index / 1e9], dtype=torch.float64) for index in range(8)
        ),
        "svd_replay_difference": torch.tensor([1e-9], dtype=torch.float64),
        "difference_in_differences_replay_difference": torch.tensor(
            [2e-9], dtype=torch.float64,
        ),
    }
    values.update(changes)
    return production.FinalDecisionReplayEvidence(**values)


class _Receipt:
    def __init__(self, sha256: str) -> None:
        self.sha256 = sha256


class _ResponseReceipt(_Receipt):
    final_context_sha256 = CONTEXT
    program_payload_sha256 = PROGRAM
    common_support_sha256 = SUPPORT


class _ResponseRun:
    def __init__(self) -> None:
        self.receipt = _ResponseReceipt(RESPONSE)

    def to_final_statistics_payload(self):
        return {
            "response_run_receipt_sha256": RESPONSE,
            "ordered_unit_identity_sha256": "9" * 64,
            "code_baseline": "code-base", "code_candidate": "code-candidate",
            "logit_baseline": "logit-base", "logit_candidate": "logit-candidate",
            "logit_nulls": tuple(f"null-{index}" for index in range(20)),
            "output_kl_baseline": "kl-base", "output_kl_candidate": "kl-candidate",
            "output_kl_nulls": tuple(f"kl-null-{index}" for index in range(20)),
        }


class _Bundle:
    bundle_sha256 = BUNDLE
    common_support_sha256 = SUPPORT


class _IntegratedReceipt(_Receipt):
    response_run_receipt_sha256 = RESPONSE
    observation_bundle_sha256 = BUNDLE


class _IntegratedResult:
    def __init__(self) -> None:
        self.observations = _Bundle()
        self.evidence_join = "joined-evidence"
        self.receipt = _IntegratedReceipt(INTEGRATED)


class _IntegratedOwner:
    def __init__(self, events) -> None:
        self.events = events

    def execute_all(self):
        self.events.append("integrated")
        return _IntegratedResult()


class _ObservationalExecutor:
    def __init__(self, events) -> None:
        self.events = events
        self.receipt = _Receipt("a" * 64)

    def make_integrated_diagnostic_owner(self, response):
        assert isinstance(response, _ResponseRun)
        self.events.append("make-integrated")
        return _IntegratedOwner(self.events)


class _ReductionEnvelope:
    def __init__(self, **values) -> None:
        self.values = values


def _patch_runtime_types(monkeypatch) -> None:
    monkeypatch.setattr(production.response_execution, "ObservedResponseRunResult", _ResponseRun)
    monkeypatch.setattr(
        production.response_execution, "ObservedResponseRunReceipt", _ResponseReceipt,
    )
    monkeypatch.setattr(
        production.observational_execution,
        "FinalObservationalBatchExecutor", _ObservationalExecutor,
    )
    monkeypatch.setattr(
        production.integration, "IntegratedObservationResult", _IntegratedResult,
    )
    monkeypatch.setattr(
        production.integration, "IntegratedObservationReceipt", _IntegratedReceipt,
    )
    monkeypatch.setattr(production.final_capability, "FinalObservationBundle", _Bundle)
    monkeypatch.setattr(
        production.final_execution, "FinalObservedReductions", _ReductionEnvelope,
    )


def test_one_shot_callback_assembles_response_observation_and_evidence(monkeypatch) -> None:
    _patch_runtime_types(monkeypatch)
    events = []
    factory = object.__new__(authority.FinalObservationalExecutorFactory)

    def build_with_response(self, **kwargs):
        events.append("response")
        assert kwargs["validated_program_bank"]["payload_sha256"] == PROGRAM
        return _ResponseRun(), _ObservationalExecutor(events)

    monkeypatch.setattr(
        authority.FinalObservationalExecutorFactory,
        "build_with_response", build_with_response,
    )
    reducer_calls = []

    def reducer(**kwargs):
        reducer_calls.append(tuple(sorted(kwargs)))
        events.append("decision-replay")
        assert "final_rows" not in kwargs and "adapter" not in kwargs
        return _evidence()

    capability = production.mint_final_decision_replay_capability(
        binding=_binding(), executor=reducer,
    )
    callback = production.FinalProductionCallback(
        executor_factory=factory, decision_replay=capability,
    )
    adapter = object.__new__(observed.ObservedBilin18Adapter)
    result = callback(
        adapter=adapter, final_rows=torch.zeros(192, 513, dtype=torch.long),
        final_records=tuple({"document_id": str(index)} for index in range(192)),
        program_bank={"payload_sha256": PROGRAM},
    )
    assert events == ["response", "make-integrated", "integrated", "decision-replay"]
    assert reducer_calls == [(
        "final_records", "integrated_receipt", "observations", "response_receipt",
    )]
    assert result.values["objective_gates"] == {
        name: False for name in final_owner.OBJECTIVE_GATES
    }
    assert result.values["closure_evidence"]["gauge_replay_differences"][1].item() == 1e-9
    assert result.values["numerical_payload"]["execution_receipts"] == {
        "decision_replay_authority_sha256": _binding().sha256,
        "observational_execution_receipt_sha256": "a" * 64,
        "integrated_observation_receipt_sha256": INTEGRATED,
        "observation_bundle_sha256": BUNDLE,
    }
    with pytest.raises(RuntimeError, match="already closed"):
        callback(
            adapter=adapter, final_rows=torch.zeros(192, 513, dtype=torch.long),
            final_records=(), program_bank={"payload_sha256": PROGRAM},
        )


def test_decision_replay_has_no_gate_or_replay_defaults() -> None:
    objective = {name: False for name in final_owner.OBJECTIVE_GATES}
    objective.pop("rr_beats_ll_ce")
    with pytest.raises(ValueError, match="objective gate evidence is incomplete"):
        _evidence(objective_gates=objective)
    with pytest.raises(ValueError, match="exactly eight gauges"):
        _evidence(gauge_replay_differences=())
    with pytest.raises(ValueError, match="replay evidence is malformed"):
        _evidence(svd_replay_difference=None)
    source = inspect.getsource(production.FinalProductionCallback.__call__)
    assert "torch.zeros" not in source
    assert ".get(" not in source
    assert "setdefault" not in source


def test_stale_decision_replay_poison_closes_capability(monkeypatch) -> None:
    _patch_runtime_types(monkeypatch)
    capability = production.mint_final_decision_replay_capability(
        binding=_binding(),
        executor=lambda **_kwargs: _evidence(observation_bundle_sha256="b" * 64),
    )
    with pytest.raises(RuntimeError, match="differs from its bound run"):
        capability.reduce(
            observations=_Bundle(),
            final_records=tuple({"document_id": str(index)} for index in range(192)),
            response_receipt=_ResponseReceipt(RESPONSE),
            integrated_receipt=_IntegratedReceipt(INTEGRATED),
        )
    with pytest.raises(RuntimeError, match="already closed"):
        capability.reduce(
            observations=_Bundle(), final_records=(),
            response_receipt=_ResponseReceipt(RESPONSE),
            integrated_receipt=_IntegratedReceipt(INTEGRATED),
        )


def test_factory_response_join_is_atomic_and_poisoned_on_failure(monkeypatch) -> None:
    events = []
    factory = object.__new__(authority.FinalObservationalExecutorFactory)
    factory._spent = False
    factory._failed = False
    factory._context = type("Context", (), {"sha256": CONTEXT})()
    factory._inherited = object()
    factory._denominator = object()
    factory._frequency = object()
    adapter = object.__new__(observed.ObservedBilin18Adapter)
    monkeypatch.setattr(authority.lifecycle, "_FINAL_ROLE_LOADS", 1)

    class RealResponse:
        receipt = _ResponseReceipt(RESPONSE)

    monkeypatch.setattr(response_execution, "ObservedResponseRunResult", RealResponse)

    def response(self, **_kwargs):
        events.append("response")
        return RealResponse()

    def build(self, **_kwargs):
        events.append("build")
        raise RuntimeError("synthetic build failure")

    monkeypatch.setattr(observed.ObservedBilin18Adapter, "run_final_response_role", response)
    monkeypatch.setattr(authority.FinalObservationalExecutorFactory, "build", build)
    with pytest.raises(RuntimeError, match="synthetic build failure"):
        factory.build_with_response(
            adapter=adapter, final_rows=torch.zeros(192, 513, dtype=torch.long),
            validated_program_bank={"payload_sha256": PROGRAM},
        )
    assert events == ["response", "build"]
    assert factory._failed is True
    assert factory._inherited is factory._denominator is factory._frequency is None


def test_factory_never_runs_response_before_final_role_authority(monkeypatch) -> None:
    factory = object.__new__(authority.FinalObservationalExecutorFactory)
    factory._spent = False
    factory._failed = False
    factory._context = object()
    factory._inherited = object()
    factory._denominator = object()
    factory._frequency = object()
    calls = []
    monkeypatch.setattr(authority.lifecycle, "_FINAL_ROLE_LOADS", 0)
    monkeypatch.setattr(
        observed.ObservedBilin18Adapter, "run_final_response_role",
        lambda self, **kwargs: calls.append(kwargs),
    )
    with pytest.raises(RuntimeError, match="licensed final-role load"):
        factory.build_with_response(
            adapter=object.__new__(observed.ObservedBilin18Adapter),
            final_rows=torch.zeros(192, 513, dtype=torch.long),
            validated_program_bank={"payload_sha256": PROGRAM},
        )
    assert calls == []
    assert factory._failed is True


def test_callback_source_has_no_io_or_final_role_loader() -> None:
    source = inspect.getsource(production)
    assert "torch.load" not in source
    assert "load_roles" not in source
    assert "load_model" not in source
    assert "from_pretrained" not in source
