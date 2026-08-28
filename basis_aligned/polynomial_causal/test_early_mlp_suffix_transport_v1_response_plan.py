from __future__ import annotations

from dataclasses import replace

import pytest

import early_mlp_suffix_transport_v1_response_plan as response


ROWS = "a" * 64
UNIT = "b" * 64


def _plan(batch: int = 0) -> response.ResponseBatchPlan:
    return response.build_response_batch_plan(
        batch_ordinal=batch, ordered_role_rows_sha256=ROWS,
        intervention_unit_sha256=UNIT,
    )


def _receipt(item: response.ResponseForwardPlan) -> response.ResponseForwardReceipt:
    return response.ResponseForwardReceipt(
        forward_plan_sha256=item.sha256, subject_key=item.subject_key,
        perturbation=item.perturbation, batch_ordinal=item.batch_ordinal,
        observed_call_pattern_sha256=item.expected_call_pattern_sha256,
        physical_edit_sha256=response.expected_physical_edit_sha256(item),
        observed_closure_sha256="d" * 64,
    )


def _reductions(plan: response.ResponseBatchPlan) -> dict[
    str, response.ResponseArmReductionReceipt
]:
    teacher = tuple(item.sha256 for item in plan.forwards[:3])
    result = {}
    for index, key in enumerate(response.RESPONSE_ACTION_KEYS):
        start = 3 + 3 * index
        result[key] = response.ResponseArmReductionReceipt(
            action_key=key, batch_plan_sha256=plan.sha256,
            teacher_forward_plan_sha256s=teacher,
            student_forward_plan_sha256s=tuple(
                item.sha256 for item in plan.forwards[start:start + 3]
            ),
            reduction_payload_sha256=f"{index + 1:064x}",
        )
    return result


def test_registered_response_plan_has_one_shared_teacher_and_22_students() -> None:
    plan = _plan()
    assert len(response.RESPONSE_ACTION_KEYS) == 22
    assert response.RESPONSE_ACTION_KEYS[:2] == ("ll/N", "lt/N")
    assert response.RESPONSE_ACTION_KEYS[2:] == tuple(
        f"a_null_{index:02d}/N" for index in range(20)
    )
    assert len(plan.forwards) == 69
    assert tuple(item.subject_key for item in plan.forwards[:3]) == (
        response.TEACHER_KEY,
    ) * 3
    assert tuple(item.perturbation for item in plan.forwards[:3]) == (
        "baseline", "positive", "negative",
    )
    assert sum(item.shared_teacher for item in plan.forwards) == 3
    assert response.expected_full_call_ledger() == {
        "batches": 48, "response_actions": 22,
        "teacher_forwards": 144, "student_forwards": 3168,
        "teacher_triplets": 48, "student_triplets": 1056,
        "forwards_per_batch": 69,
    }


def test_sealed_batch_accepts_only_exact_ordered_physical_receipts() -> None:
    plan = _plan(7)
    receipts = tuple(_receipt(item) for item in plan.forwards)
    sealed = response.seal_response_batch(
        plan=plan, forward_receipts=receipts, reductions=_reductions(plan),
    )
    assert sealed.batch_plan_sha256 == plan.sha256
    assert len(sealed.sha256) == 64

    swapped = list(receipts)
    swapped[3], swapped[6] = swapped[6], swapped[3]
    with pytest.raises(RuntimeError, match="planned forward"):
        response.seal_response_batch(
            plan=plan, forward_receipts=swapped,
            reductions=_reductions(plan),
        )


def test_sign_action_rows_and_call_pattern_cannot_be_relabelled() -> None:
    plan = _plan()
    first_student = plan.forwards[3]
    with pytest.raises(ValueError, match="perturbation/sign"):
        replace(first_student, edit_sign=1)
    with pytest.raises(ValueError, match="physical action"):
        replace(first_student, expected_call_pattern_sha256="e" * 64)
    with pytest.raises(ValueError, match="incomplete or reordered"):
        replace(plan, forwards=plan.forwards[3:] + plan.forwards[:3])
    with pytest.raises(ValueError, match="incomplete or reordered"):
        replace(plan, ordered_role_rows_sha256="f" * 64)


def test_null_and_reduction_labels_are_not_interchangeable() -> None:
    plan = _plan()
    receipts = tuple(_receipt(item) for item in plan.forwards)
    changed = list(receipts)
    null = changed[-3]
    changed[-3] = replace(null, subject_key="a_null_18/N")
    with pytest.raises(RuntimeError, match="planned forward"):
        response.seal_response_batch(
            plan=plan, forward_receipts=changed,
            reductions=_reductions(plan),
        )

    reductions = _reductions(plan)
    reductions["a_null_18/N"], reductions["a_null_19/N"] = (
        reductions["a_null_19/N"], reductions["a_null_18/N"],
    )
    with pytest.raises(RuntimeError, match="planned arm"):
        response.seal_response_batch(
            plan=plan, forward_receipts=receipts, reductions=reductions,
        )

    reordered = {key: _reductions(plan)[key] for key in reversed(reductions)}
    with pytest.raises(RuntimeError, match="incomplete or reordered"):
        response.seal_response_batch(
            plan=plan, forward_receipts=receipts,
            reductions=reordered,
        )


def test_receipts_must_close_every_forward_and_every_reduction() -> None:
    plan = _plan()
    receipts = tuple(_receipt(item) for item in plan.forwards)
    with pytest.raises(RuntimeError, match="count"):
        response.seal_response_batch(
            plan=plan, forward_receipts=receipts[:-1],
            reductions=_reductions(plan),
        )
    missing = _reductions(plan)
    missing.pop("lt/N")
    with pytest.raises(RuntimeError, match="incomplete or reordered"):
        response.seal_response_batch(
            plan=plan, forward_receipts=receipts,
            reductions=missing,
        )


def test_physical_edit_hash_is_derived_from_sign_rows_and_unit() -> None:
    plan = _plan()
    receipts = list(_receipt(item) for item in plan.forwards)
    receipts[1] = replace(receipts[1], physical_edit_sha256="e" * 64)
    with pytest.raises(RuntimeError, match="planned forward"):
        response.seal_response_batch(
            plan=plan, forward_receipts=receipts,
            reductions=_reductions(plan),
        )
