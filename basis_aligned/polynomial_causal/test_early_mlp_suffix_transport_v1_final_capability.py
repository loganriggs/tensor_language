import pytest
import torch

import early_mlp_suffix_transport_v1_final_capability as capability


SHA = "a" * 64


def _row(value: float = 1.0) -> capability.RowReduction:
    return capability.RowReduction(
        row_sum=torch.full((192,), value, dtype=torch.float64),
        row_count=torch.ones(192, dtype=torch.long),
    )


def _response() -> capability.ResponseReduction:
    return capability.ResponseReduction(
        error_sum=torch.ones(192, dtype=torch.float64),
        teacher_sum=torch.full((192,), 4.0, dtype=torch.float64),
        student_sum=torch.ones(192, dtype=torch.float64),
        dot_sum=torch.full((192,), 2.0, dtype=torch.float64),
        unit_identity="b" * 64,
    )


def _observation(action: capability.FinalAction, **changes):
    response = action.background == "N" and (
        action.arm == "ll" or action.arm == "lt" or action.arm.startswith("a_null_")
    )
    code = action.background == "N" and action.arm in {"ll", "lt"}
    values = {
        "action": action,
        "common_support_sha256": SHA,
        "ce": _row(),
        "teacher_kl": _row() if action.background == "N" else None,
        "copy_ce": _row(),
        "frequency_ce": tuple(_row() for _ in range(9)),
        "code_response": _response() if code else None,
        "logit_response": _response() if response else None,
        "consumer_norm_ratio": tuple(_row() for _ in range(18)),
        "execution_closure_sha256": "c" * 64,
    }
    values.update(changes)
    return capability.FinalArmObservation(**values)


def test_capability_executes_exact_canonical_lattice_once():
    calls = []

    def execute(action):
        calls.append(action.key)
        return _observation(action)

    owned = capability.mint_final_action_capability(
        issuer_id="d" * 64, common_support_sha256=SHA, executor=execute,
    )
    bundle = owned.execute_all()
    assert tuple(calls) == capability.CANONICAL_ACTION_KEYS
    assert len(bundle.observations) == len(capability.BASE_ARMS) * 2 == 68
    assert bundle.common_support_sha256 == SHA
    assert owned.spent is True and owned.failed is False
    with pytest.raises(RuntimeError, match="already closed"):
        owned.execute_all()


def test_wrong_action_or_support_poison_closes_capability():
    def wrong_action(action):
        if action.key == capability.CANONICAL_ACTION_KEYS[1]:
            return _observation(capability.CANONICAL_ACTIONS[0])
        return _observation(action)

    owned = capability.mint_final_action_capability(
        issuer_id="d" * 64, common_support_sha256=SHA, executor=wrong_action,
    )
    with pytest.raises(RuntimeError, match="wrong typed action"):
        owned.execute_all()
    assert owned.spent is True and owned.failed is True
    with pytest.raises(RuntimeError, match="already closed"):
        owned.execute_all()

    mixed = capability.mint_final_action_capability(
        issuer_id="d" * 64, common_support_sha256=SHA,
        executor=lambda action: _observation(action, common_support_sha256="e" * 64),
    )
    with pytest.raises(RuntimeError, match="mixed scored support"):
        mixed.execute_all()


def test_response_and_background_types_follow_registered_semantics():
    with pytest.raises(ValueError, match="requires teacher-KL"):
        _observation(capability.FinalAction("qq", "N"), teacher_kl=None)
    with pytest.raises(ValueError, match="CE-only"):
        _observation(capability.FinalAction("qq", "E"), teacher_kl=_row())
    with pytest.raises(ValueError, match="response reductions"):
        _observation(capability.FinalAction("lt", "N"), logit_response=None)
    with pytest.raises(ValueError, match="response reductions"):
        _observation(capability.FinalAction("qq", "N"), logit_response=_response())


def test_raw_or_graph_bearing_reductions_fail_closed():
    with pytest.raises(ValueError, match="allowed final row reduction"):
        capability.RowReduction(
            row_sum=torch.ones(192, 1, dtype=torch.float64),
            row_count=torch.ones(192, dtype=torch.long),
        )
    graph = torch.ones(192, dtype=torch.float64, requires_grad=True) * 2
    with pytest.raises(ValueError, match="allowed final row reduction"):
        capability.RowReduction(
            row_sum=graph, row_count=torch.ones(192, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="all nine"):
        _observation(capability.FinalAction("qq", "N"), frequency_ce=tuple())
    with pytest.raises(ValueError, match="all live-consumer"):
        _observation(capability.FinalAction("qq", "N"), consumer_norm_ratio=tuple())


def test_response_inner_products_are_consistency_checked_and_cloned():
    source = torch.ones(192, dtype=torch.float64)
    response = capability.ResponseReduction(
        error_sum=source, teacher_sum=4 * source, student_sum=source,
        dot_sum=2 * source, unit_identity="b" * 64,
    )
    source.fill_(9)
    assert torch.equal(response.error_sum, torch.ones(192, dtype=torch.float64))
    with pytest.raises(ValueError, match="inconsistent"):
        capability.ResponseReduction(
            error_sum=2 * torch.ones(192, dtype=torch.float64),
            teacher_sum=4 * torch.ones(192, dtype=torch.float64),
            student_sum=torch.ones(192, dtype=torch.float64),
            dot_sum=2 * torch.ones(192, dtype=torch.float64),
            unit_identity="b" * 64,
        )


def test_direct_construction_without_mint_token_is_forbidden():
    with pytest.raises(TypeError, match="must be minted"):
        capability.FinalActionCapability(
            _token=object(), issuer_id="d" * 64,
            common_support_sha256=SHA, executor=lambda action: _observation(action),
        )
