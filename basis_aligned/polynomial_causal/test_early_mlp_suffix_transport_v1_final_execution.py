from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import bilin18_observed_adapter as observed
import early_mlp_suffix_transport_v1_final as final_owner
import early_mlp_suffix_transport_v1_final_execution as execution
import early_mlp_suffix_transport_v1_lifecycle as lifecycle


SHA = "a" * 64


def _adapter() -> observed.ObservedBilin18Adapter:
    # The execution boundary uses only nominal capability identity.  No model,
    # checkpoint, facade, or ship is constructed by these tests.
    return object.__new__(observed.ObservedBilin18Adapter)


def _records() -> list[dict[str, str]]:
    return [{"document_id": f"document-{index}"} for index in range(192)]


def _response(*, error: float, student: float, dot: float) -> dict:
    return {
        "error_sum": torch.full((192,), error, dtype=torch.float64),
        "teacher_sum": torch.full((192,), 4.0, dtype=torch.float64),
        "student_sum": torch.full((192,), student, dtype=torch.float64),
        "dot_sum": torch.full((192,), dot, dtype=torch.float64),
        "unit_identity": "b" * 64,
    }


def _output_kl(*, numerator: float, denominator: float) -> dict:
    return {
        "numerator_sum": torch.full((192,), numerator, dtype=torch.float64),
        "denominator_sum": torch.full((192,), denominator, dtype=torch.float64),
        "unit_identity": "b" * 64,
    }


def _closure() -> dict:
    return {
        "outer_model_returned": True,
        "hooks_restored": True,
        "hooks_inert": True,
        "component_tree_before_sha256": "c" * 64,
        "component_tree_after_sha256": "c" * 64,
        "student_poison_closed": True,
        "program_payload_sha256": SHA,
        "common_support_sha256": "d" * 64,
        "arm_support_sha256s": {
            arm: "d" * 64 for arm in execution.REQUIRED_FINAL_ARMS
        },
        "observational_action_call_ledgers": (
            execution.final_actions.expected_observational_action_call_ledgers()
        ),
        "gauge_replay_differences": tuple(torch.zeros(3) for _ in range(8)),
        "svd_replay_difference": torch.zeros(3),
        "difference_in_differences_replay_difference": torch.zeros(3),
        "row_count": 192,
        "scored_tokens_per_row": 192,
        "scored_token_count": 192 * 192,
    }


def _reductions(**changes) -> execution.FinalObservedReductions:
    baseline = _response(error=2.25, student=0.25, dot=1.0)
    candidate = _response(error=1.0, student=1.0, dot=2.0)
    values = {
        "objective_gates": {name: True for name in final_owner.OBJECTIVE_GATES},
        "transport_observational_gates": {
            name: True for name in execution.statistics.TRANSPORT_OBSERVATIONAL_GATES
        },
        "code_baseline": baseline,
        "code_candidate": candidate,
        "logit_baseline": baseline,
        "logit_candidate": candidate,
        "logit_nulls": tuple(deepcopy(baseline) for _ in range(20)),
        "output_kl_baseline": _output_kl(numerator=0.8, denominator=1.0),
        "output_kl_candidate": _output_kl(numerator=0.4, denominator=1.0),
        "output_kl_nulls": tuple(
            _output_kl(numerator=0.9 + index / 100, denominator=1.0)
            for index in range(20)
        ),
        "numerical_payload": {"safe_scalar": 1.25, "counts": [192, 36864]},
        "closure_evidence": _closure(),
    }
    values.update(changes)
    return execution.FinalObservedReductions(**values)


def _bindings() -> dict:
    artifact = {"path": "/frozen", "sha256": "e" * 64, "bytes": 1}
    return {
        "final_attempt": dict(artifact),
        "rows_receipt": dict(artifact),
        "programs": dict(artifact),
        "programs_receipt": dict(artifact),
        "program_payload_sha256": SHA,
        "source_commit": "f" * 40,
        "source_hashes_sha256": "1" * 64,
        "protected_before_sha256": "2" * 64,
    }


def _bank() -> dict:
    return {
        "payload_sha256": SHA,
        "teacher_calibration": {"calibration_passed": True},
    }


def _evaluate(callback):
    return execution.evaluate_loaded_final(
        adapter=_adapter(),
        final_rows=torch.arange(192 * 513, dtype=torch.long).view(192, 513),
        final_records=_records(), validated_program_bank=_bank(),
        bindings=_bindings(), callback=callback,
    )


def test_observed_final_callback_builds_only_the_semantic_envelope() -> None:
    calls = 0

    def callback(**kwargs):
        nonlocal calls
        calls += 1
        assert isinstance(kwargs["adapter"], observed.ObservedBilin18Adapter)
        assert tuple(kwargs["final_rows"].shape) == (192, 513)
        assert len(kwargs["final_records"]) == 192
        return _reductions()

    result = _evaluate(callback)
    assert calls == 1
    assert result["execution_closure"] == {
        "final_role_loads": 1,
        "final_evaluation_callbacks": 1,
        "outer_model_returned": True,
        "hooks_restored": True,
        "hooks_inert": True,
        "component_tree_unchanged": True,
        "student_poison_closed": True,
        "programs_reloaded_semantically": True,
        "common_support_complete": True,
        "observational_action_call_ledger_sha256": execution.runtime.logical_identity_sha256(
            execution.final_actions.expected_observational_action_call_ledgers()
        ),
        "observational_student_outer_forwards": 68 * 48,
        "gauge_replays": 8,
        "gauge_max_abs_drift": 0.0,
        "svd_max_abs_drift": 0.0,
        "difference_in_differences_max_abs_drift": 0.0,
        "row_count": 192,
        "scored_tokens_per_row": 192,
        "scored_token_count": 36864,
    }
    assert result["objective_route"]["passes"] is True
    assert result["transport_route"]["passes"] is True
    assert result["ledger_credit"] == {
        currency: False for currency in final_owner.LEDGER_CURRENCIES
    }
    assert not any(
        torch.is_tensor(value) and value.ndim > 1
        for value in execution._iter_tensors(result)
    )


def test_callback_cannot_mutate_or_return_the_final_role() -> None:
    def mutate(**kwargs):
        kwargs["final_rows"][0, 0] = -1
        return _reductions()

    with pytest.raises(RuntimeError, match="mutated the licensed role tensor"):
        _evaluate(mutate)

    rows = torch.arange(192 * 513, dtype=torch.long).view(192, 513)
    forged = object.__new__(execution.FinalObservedReductions)
    clean = _reductions()
    for field in clean.__dataclass_fields__:
        object.__setattr__(forged, field, getattr(clean, field))
    object.__setattr__(forged, "numerical_payload", {"escaped": rows})
    with pytest.raises(RuntimeError, match="alias of the licensed role tensor"):
        execution.evaluate_loaded_final(
            adapter=_adapter(), final_rows=rows, final_records=_records(),
            validated_program_bank=_bank(), bindings=_bindings(),
            callback=lambda **_kwargs: forged,
        )


def test_reductions_reject_raw_tensors_and_inconsistent_response_geometry() -> None:
    with pytest.raises(RuntimeError, match="cannot contain raw tensors"):
        _reductions(numerical_payload={"raw_logits": torch.zeros(4, 256, 11)})
    inconsistent = _response(error=1.0, student=1.0, dot=2.0)
    inconsistent["error_sum"][0] = 2.0
    with pytest.raises(ValueError, match="inconsistent"):
        _reductions(code_candidate=inconsistent)


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda value: value["arm_support_sha256s"].pop(next(iter(value["arm_support_sha256s"]))), "support"),
        (lambda value: value.__setitem__("hooks_inert", False), "hook/poison"),
        (lambda value: value.__setitem__("student_poison_closed", False), "hook/poison"),
        (lambda value: value.__setitem__("component_tree_after_sha256", "9" * 64), "component tree"),
        (lambda value: value.__setitem__("program_payload_sha256", "9" * 64), "program bank"),
        (
            lambda value: value["observational_action_call_ledgers"]["rr/E"][
                "literal_early_mlp_calls"
            ].__setitem__("2", 47),
            "call ledger",
        ),
        (lambda value: value["gauge_replay_differences"][0].fill_(3e-6), "replay tolerance"),
        (lambda value: value["svd_replay_difference"].fill_(3e-6), "replay tolerance"),
        (lambda value: value["difference_in_differences_replay_difference"].fill_(3e-6), "replay tolerance"),
    ),
)
def test_adversarial_closure_failures(mutator, message) -> None:
    closure = _closure()
    mutator(closure)
    try:
        reductions = _reductions(closure_evidence=closure)
    except RuntimeError as error:
        assert message in str(error)
        return
    with pytest.raises(RuntimeError, match=message):
        _evaluate(lambda **_kwargs: reductions)


def test_callback_type_and_support_are_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="sealed reduction type"):
        _evaluate(lambda **_kwargs: {"objective_gates": {}})
    with pytest.raises(RuntimeError, match="provenance is incomplete"):
        execution.evaluate_loaded_final(
            adapter=_adapter(),
            final_rows=torch.zeros(192, 513, dtype=torch.long),
            final_records=_records()[:-1], validated_program_bank=_bank(),
            bindings=_bindings(), callback=lambda **_kwargs: _reductions(),
        )


def test_execute_final_owns_one_role_load_and_one_callback(monkeypatch) -> None:
    rows = torch.zeros(192, 513, dtype=torch.long)
    receipt = {
        "document_provenance": {"sets": {execution.FINAL_ROLE: _records()}},
    }
    loads = 0
    callbacks = 0
    monkeypatch.setattr(lifecycle, "_FINAL_ROLE_LOADS", 0)
    monkeypatch.setattr(
        final_owner, "terminal_bindings", lambda **_kwargs: (_bindings(), _bank(), {}),
    )

    def load_roles(requested, **kwargs):
        nonlocal loads
        loads += 1
        assert tuple(requested) == (execution.FINAL_ROLE,)
        assert kwargs["operation"] == "final"
        lifecycle._FINAL_ROLE_LOADS += 1
        return receipt, {execution.FINAL_ROLE: rows}

    def callback(**_kwargs):
        nonlocal callbacks
        callbacks += 1
        return _reductions()

    monkeypatch.setattr(lifecycle, "load_roles", load_roles)
    result = execution.execute_final(
        adapter=_adapter(), callback=callback, lock_nonce="owned",
    )
    assert loads == callbacks == lifecycle._FINAL_ROLE_LOADS == 1
    assert result["execution_closure"]["final_evaluation_callbacks"] == 1


def test_execute_final_rejects_a_nonpristine_load_counter(monkeypatch) -> None:
    monkeypatch.setattr(lifecycle, "_FINAL_ROLE_LOADS", 1)
    with pytest.raises(RuntimeError, match="not pristine"):
        execution.execute_final(
            adapter=_adapter(), callback=lambda **_kwargs: _reductions(),
            lock_nonce="owned",
        )
