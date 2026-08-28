"""Observed final-execution boundary for early-MLP suffix transport v1.

The semantic final owner intentionally cannot see rows or a model.  This module is
the narrow bridge: it loads the final role once, invokes one source-closed observed
callback, reduces the callback's row-level response sufficient statistics on CPU,
and returns only the semantic result accepted by :mod:`..._final`.

Importing this module performs no artifact I/O and loads no model/checkpoint.  The
callback is deliberately passed an :class:`ObservedBilin18Adapter`; raw model and
checkpoint objects are never part of this API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

import bilin18_observed_adapter as observed
import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_final_capability as final_capability
import early_mlp_suffix_transport_v1_final as final_owner
import early_mlp_suffix_transport_v1_lifecycle as lifecycle
import early_mlp_suffix_transport_v1_runtime as runtime
import early_mlp_suffix_transport_v1_response_execution as response_execution
import early_mlp_suffix_transport_v1_statistics as statistics


FINAL_ROLE = "early_mlp_suffix_transport_v1_final"
FINAL_ROW_COUNT = 192
FINAL_ROW_WIDTH = 513
SCORED_TOKENS_PER_ROW = 192
REPLAY_TOLERANCE = 2e-6

REQUIRED_FINAL_ARMS = final_capability.CANONICAL_ACTION_KEYS

_CLOSURE_FIELDS = {
    "outer_model_returned", "hooks_restored", "hooks_inert",
    "component_tree_before_sha256", "component_tree_after_sha256",
    "student_poison_closed", "program_payload_sha256",
    "common_support_sha256", "arm_support_sha256s",
    "observational_action_call_ledgers", "gauge_replay_differences",
    "svd_replay_difference", "difference_in_differences_replay_difference",
    "row_count", "scored_tokens_per_row", "scored_token_count",
}


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _exact_mapping(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError(f"{name} schema changed")
    return value


def _scalar_tree(value: Any, name: str = "numerical payload") -> Any:
    """Clone a JSON-like finite tree and forbid every tensor/object escape."""

    if torch.is_tensor(value):
        raise RuntimeError(f"{name} cannot contain raw tensors")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise RuntimeError(f"{name} has a malformed mapping key")
        return {key: _scalar_tree(value[key], name) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_scalar_tree(item, name) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{name} contains a nonfinite scalar")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise RuntimeError(f"{name} contains unsupported {type(value).__name__}")


def _response(value: Any, name: str) -> dict[str, Any]:
    checked = statistics.validate_response_sufficient_statistics(
        value, length=FINAL_ROW_COUNT,
    )
    identity = value.get("unit_identity")
    if not _sha256(identity):
        raise RuntimeError(f"{name} response unit identity is malformed")
    return {
        **{key: tensor.detach().cpu().double().contiguous().clone()
           for key, tensor in checked.items()},
        "unit_identity": identity,
    }


def _output_kl(value: Any, name: str) -> dict[str, Any]:
    checked = statistics.validate_output_kl_sufficient_statistics(
        value, length=FINAL_ROW_COUNT,
    )
    identity = value.get("unit_identity")
    if not _sha256(identity):
        raise RuntimeError(f"{name} output-KL unit identity is malformed")
    return {
        **{key: tensor.detach().cpu().double().contiguous().clone()
           for key, tensor in checked.items()},
        "unit_identity": identity,
    }


def _difference_vector(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 1 or value.numel() == 0 or (
        value.numel() > 4096
    ) or not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} replay difference is malformed")
    return value.detach().cpu().double().contiguous().clone()


def _observational_action_call_ledgers(value: Any) -> dict[str, dict[str, Any]]:
    """Validate all 48 observational student forwards for every final action."""

    expected = final_actions.expected_observational_action_call_ledgers()
    supplied = _exact_mapping(
        value, set(REQUIRED_FINAL_ARMS), "observational action call ledgers",
    )
    cleaned: dict[str, dict[str, Any]] = {}
    fields = (
        "deployed_n_calls", "correction_calls", "literal_early_mlp_calls",
    )
    for action in REQUIRED_FINAL_ARMS:
        ledger = _exact_mapping(
            supplied[action], {"outer_forward_count", *fields},
            f"{action} observational call ledger",
        )
        clean: dict[str, Any] = {
            "outer_forward_count": ledger["outer_forward_count"],
        }
        for name in fields:
            counts = _exact_mapping(
                ledger[name], {"0", "1", "2"}, f"{action} {name}",
            )
            clean[name] = dict(counts)
        integers = (
            clean["outer_forward_count"],
            *(count for name in fields for count in clean[name].values()),
        )
        if any(type(item) is not int or item < 0 for item in integers) or clean != (
            expected[action]
        ):
            raise RuntimeError(f"{action} observational student call ledger changed")
        cleaned[action] = clean
    return cleaned


@dataclass(frozen=True, slots=True)
class FinalObservedReductions:
    """Only data permitted to cross the observed callback boundary.

    Response tensors are row-level scalar sufficient statistics, never logits,
    states, codes, or role rows.  Replay differences are small diagnostic vectors;
    the executor, rather than the callback, computes their maxima.
    """

    objective_gates: Mapping[str, bool]
    transport_observational_gates: Mapping[str, bool]
    code_baseline: Mapping[str, Any]
    code_candidate: Mapping[str, Any]
    logit_baseline: Mapping[str, Any]
    logit_candidate: Mapping[str, Any]
    logit_nulls: Sequence[Mapping[str, Any]]
    output_kl_baseline: Mapping[str, Any]
    output_kl_candidate: Mapping[str, Any]
    output_kl_nulls: Sequence[Mapping[str, Any]]
    numerical_payload: Mapping[str, Any]
    closure_evidence: Mapping[str, Any]
    response_run_receipt: response_execution.ObservedResponseRunReceipt
    evidence_join_receipt: final_capability.FinalEvidenceJoinReceipt

    def __post_init__(self) -> None:
        objective = _exact_mapping(
            self.objective_gates, set(final_owner.OBJECTIVE_GATES), "objective gates",
        )
        if any(type(value) is not bool for value in objective.values()):
            raise RuntimeError("objective gate is not a literal boolean")
        observational = _exact_mapping(
            self.transport_observational_gates,
            set(statistics.TRANSPORT_OBSERVATIONAL_GATES),
            "transport observational gates",
        )
        if any(type(value) is not bool for value in observational.values()):
            raise RuntimeError("transport observational gate is not a literal boolean")
        if not isinstance(self.logit_nulls, (tuple, list)) or len(self.logit_nulls) != 20:
            raise RuntimeError("transport requires exactly twenty final null responses")
        if not isinstance(self.output_kl_nulls, (tuple, list)) or len(
            self.output_kl_nulls
        ) != 20:
            raise RuntimeError("transport requires exactly twenty final output-KL nulls")
        responses = {
            "code_baseline": _response(self.code_baseline, "code baseline"),
            "code_candidate": _response(self.code_candidate, "code candidate"),
            "logit_baseline": _response(self.logit_baseline, "logit baseline"),
            "logit_candidate": _response(self.logit_candidate, "logit candidate"),
        }
        nulls = tuple(
            _response(value, f"logit null {index}")
            for index, value in enumerate(self.logit_nulls)
        )
        output_kl = {
            "output_kl_baseline": _output_kl(
                self.output_kl_baseline, "output-KL baseline",
            ),
            "output_kl_candidate": _output_kl(
                self.output_kl_candidate, "output-KL candidate",
            ),
        }
        output_kl_nulls = tuple(
            _output_kl(value, f"output-KL null {index}")
            for index, value in enumerate(self.output_kl_nulls)
        )
        identities = {
            value["unit_identity"] for value in (
                *responses.values(), *nulls, *output_kl.values(), *output_kl_nulls,
            )
        }
        if len(identities) != 1:
            raise RuntimeError("final transport responses do not share ordered units")
        if not isinstance(
            self.response_run_receipt,
            response_execution.ObservedResponseRunReceipt,
        ) or identities != {
            self.response_run_receipt.ordered_unit_identity_sha256
        }:
            raise RuntimeError("final transport responses lack their typed run receipt")
        if type(self.evidence_join_receipt) is not (
            final_capability.FinalEvidenceJoinReceipt
        ) or self.evidence_join_receipt.response_run_receipt_sha256 != (
            self.response_run_receipt.sha256
        ) or self.evidence_join_receipt.ordered_unit_identity_sha256 != (
            self.response_run_receipt.ordered_unit_identity_sha256
        ):
            raise RuntimeError("final transport response run lacks its observational join")
        response_payload = {
            "response_run_receipt_sha256": self.response_run_receipt.sha256,
            "ordered_unit_identity_sha256": (
                self.response_run_receipt.ordered_unit_identity_sha256
            ),
            **responses, "logit_nulls": nulls, **output_kl,
            "output_kl_nulls": output_kl_nulls,
        }
        if final_capability._response_statistics_identity(response_payload) != (
            self.evidence_join_receipt.response_statistics_sha256
        ):
            raise RuntimeError("final transport statistics differ from evidence join")
        closure = _exact_mapping(
            self.closure_evidence, _CLOSURE_FIELDS, "final observed closure evidence",
        )
        gauges = closure["gauge_replay_differences"]
        if not isinstance(gauges, (tuple, list)) or len(gauges) != 8:
            raise RuntimeError("final execution requires exactly eight gauge replays")
        clean_closure = dict(closure)
        clean_closure["gauge_replay_differences"] = tuple(
            _difference_vector(value, f"gauge {index}")
            for index, value in enumerate(gauges)
        )
        clean_closure["svd_replay_difference"] = _difference_vector(
            closure["svd_replay_difference"], "SVD",
        )
        clean_closure["difference_in_differences_replay_difference"] = (
            _difference_vector(
                closure["difference_in_differences_replay_difference"],
                "difference-in-differences",
            )
        )
        clean_closure["observational_action_call_ledgers"] = (
            _observational_action_call_ledgers(
                closure["observational_action_call_ledgers"],
            )
        )
        object.__setattr__(self, "objective_gates", dict(objective))
        object.__setattr__(self, "transport_observational_gates", dict(observational))
        for name, value in responses.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "logit_nulls", nulls)
        for name, value in output_kl.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "output_kl_nulls", output_kl_nulls)
        object.__setattr__(self, "numerical_payload", _scalar_tree(self.numerical_payload))
        object.__setattr__(self, "closure_evidence", clean_closure)


def _require_role_rows(rows: Any) -> None:
    if not torch.is_tensor(rows) or rows.dtype != torch.long or tuple(rows.shape) != (
        FINAL_ROW_COUNT, FINAL_ROW_WIDTH
    ) or rows.device.type != "cpu" or not rows.is_contiguous():
        raise RuntimeError("final role tensor schema changed")


def _shares_storage(value: torch.Tensor, rows: torch.Tensor) -> bool:
    return value.untyped_storage().data_ptr() == rows.untyped_storage().data_ptr()


def _iter_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, FinalObservedReductions):
        for field in value.__dataclass_fields__:
            yield from _iter_tensors(getattr(value, field))


def _execution_closure(
    reductions: FinalObservedReductions, *, program_payload_sha256: str,
) -> dict[str, Any]:
    evidence = reductions.closure_evidence
    literal_true = (
        "outer_model_returned", "hooks_restored", "hooks_inert",
        "student_poison_closed",
    )
    if any(evidence[name] is not True for name in literal_true):
        raise RuntimeError("observed final model/hook/poison closure is incomplete")
    before = evidence["component_tree_before_sha256"]
    after = evidence["component_tree_after_sha256"]
    if not _sha256(before) or after != before:
        raise RuntimeError("observed final component tree changed")
    if evidence["program_payload_sha256"] != program_payload_sha256:
        raise RuntimeError("observed callback did not use the reloaded program bank")
    if reductions.evidence_join_receipt.program_payload_sha256 != (
        program_payload_sha256
    ):
        raise RuntimeError("joined evidence used a different program bank")
    support = evidence["common_support_sha256"]
    arm_support = _exact_mapping(
        evidence["arm_support_sha256s"], set(REQUIRED_FINAL_ARMS),
        "final arm support",
    )
    if not _sha256(support) or any(value != support for value in arm_support.values()):
        raise RuntimeError("final arm support is incomplete or mixed")
    if reductions.evidence_join_receipt.common_support_sha256 != support:
        raise RuntimeError("joined evidence used different scored support")
    action_ledgers = _observational_action_call_ledgers(
        evidence["observational_action_call_ledgers"],
    )
    action_ledger_sha256 = runtime.logical_identity_sha256(action_ledgers)
    observational_forwards = sum(
        ledger["outer_forward_count"] for ledger in action_ledgers.values()
    )
    if evidence["row_count"] != FINAL_ROW_COUNT or evidence[
        "scored_tokens_per_row"
    ] != SCORED_TOKENS_PER_ROW or evidence["scored_token_count"] != (
        FINAL_ROW_COUNT * SCORED_TOKENS_PER_ROW
    ):
        raise RuntimeError("final callback did not close the frozen scored support")
    gauge = max(float(value.abs().max()) for value in evidence[
        "gauge_replay_differences"
    ])
    svd = float(evidence["svd_replay_difference"].abs().max())
    did = float(evidence[
        "difference_in_differences_replay_difference"
    ].abs().max())
    if max(gauge, svd, did) > REPLAY_TOLERANCE:
        raise RuntimeError("final gauge/SVD/DiD replay tolerance failed")
    return {
        "final_role_loads": 1,
        "final_evaluation_callbacks": 1,
        "outer_model_returned": True,
        "hooks_restored": True,
        "hooks_inert": True,
        "component_tree_unchanged": True,
        "student_poison_closed": True,
        "programs_reloaded_semantically": True,
        "common_support_complete": True,
        "observational_action_call_ledger_sha256": action_ledger_sha256,
        "response_run_receipt_sha256": reductions.response_run_receipt.sha256,
        "final_evidence_join_receipt_sha256": reductions.evidence_join_receipt.sha256,
        "observational_student_outer_forwards": observational_forwards,
        "gauge_replays": 8,
        "gauge_max_abs_drift": gauge,
        "svd_max_abs_drift": svd,
        "difference_in_differences_max_abs_drift": did,
        "row_count": FINAL_ROW_COUNT,
        "scored_tokens_per_row": SCORED_TOKENS_PER_ROW,
        "scored_token_count": FINAL_ROW_COUNT * SCORED_TOKENS_PER_ROW,
    }


def evaluate_loaded_final(
    *, adapter: observed.ObservedBilin18Adapter, final_rows: torch.Tensor,
    final_records: Sequence[Mapping[str, Any]], validated_program_bank: Mapping[str, Any],
    bindings: Mapping[str, Any], callback: Callable[..., FinalObservedReductions],
) -> dict[str, Any]:
    """Invoke one observed callback and build the tensor-free semantic envelope.

    This pure boundary is exposed for adversarial tests.  Production callers must
    use :func:`execute_final`, which owns the one legal final-role deserialization.
    """

    if not isinstance(adapter, observed.ObservedBilin18Adapter):
        raise TypeError("final callback requires the sealed observed adapter")
    _require_role_rows(final_rows)
    if not isinstance(final_records, (tuple, list)) or len(final_records) != FINAL_ROW_COUNT:
        raise RuntimeError("final document provenance is incomplete")
    if any(not isinstance(record, Mapping) for record in final_records):
        raise RuntimeError("final document provenance record is malformed")
    program_sha256 = validated_program_bank.get("payload_sha256")
    calibration = validated_program_bank.get("teacher_calibration", {}).get(
        "calibration_passed"
    )
    if not _sha256(program_sha256) or type(calibration) is not bool or bindings.get(
        "program_payload_sha256"
    ) != program_sha256:
        raise RuntimeError("final program/calibration binding changed")
    before_hash = runtime.tensor_identity_sha256(final_rows)
    before_version = final_rows._version
    callback_count = 0

    def invoke() -> FinalObservedReductions:
        nonlocal callback_count
        callback_count += 1
        if callback_count != 1:
            raise RuntimeError("final observed callback may run exactly once")
        return callback(
            adapter=adapter, final_rows=final_rows,
            final_records=tuple(dict(record) for record in final_records),
            program_bank=validated_program_bank,
        )

    reductions = invoke()
    if callback_count != 1 or type(reductions) is not FinalObservedReductions:
        raise RuntimeError("final callback did not return the sealed reduction type")
    if final_rows._version != before_version or runtime.tensor_identity_sha256(
        final_rows
    ) != before_hash:
        raise RuntimeError("final callback mutated the licensed role tensor")
    if any(_shares_storage(value, final_rows) for value in _iter_tensors(reductions)):
        raise RuntimeError("final callback leaked an alias of the licensed role tensor")

    execution = _execution_closure(
        reductions, program_payload_sha256=program_sha256,
    )
    weights = statistics.document_bootstrap_weights(final_records)
    transport = statistics.transport_route_decision(
        code_baseline=reductions.code_baseline,
        code_candidate=reductions.code_candidate,
        logit_baseline=reductions.logit_baseline,
        logit_candidate=reductions.logit_candidate,
        logit_nulls=reductions.logit_nulls,
        output_kl_baseline=reductions.output_kl_baseline,
        output_kl_candidate=reductions.output_kl_candidate,
        output_kl_nulls=reductions.output_kl_nulls,
        weights=weights,
        calibration_passed=calibration,
        observational_gates=reductions.transport_observational_gates,
    )
    return final_owner.build_final_result(
        bindings=bindings,
        execution_closure=execution,
        objective_gates=reductions.objective_gates,
        transport_route=transport,
        numerical_payload=reductions.numerical_payload,
        expected_calibration=calibration,
    )


def execute_final(
    *, adapter: observed.ObservedBilin18Adapter,
    callback: Callable[..., FinalObservedReductions], lock_nonce: str,
    paths: lifecycle.ArtifactPaths = lifecycle.PATHS,
    lock_path: Path = lifecycle.RUN_LOCK,
) -> dict[str, Any]:
    """Load the licensed final role once and return one semantic final result."""

    if lifecycle._FINAL_ROLE_LOADS != 0:
        raise RuntimeError("final role load counter is not pristine")
    bindings, validated_bank, _attempt = final_owner.terminal_bindings(paths=paths)
    receipt, loaded = lifecycle.load_roles(
        (FINAL_ROLE,), operation="final", lock_nonce=lock_nonce,
        paths=paths, lock_path=lock_path,
    )
    if lifecycle._FINAL_ROLE_LOADS != 1 or set(loaded) != {FINAL_ROLE}:
        raise RuntimeError("final role did not deserialize exactly once")
    records = receipt.get("document_provenance", {}).get("sets", {}).get(FINAL_ROLE)
    try:
        return evaluate_loaded_final(
            adapter=adapter, final_rows=loaded[FINAL_ROLE], final_records=records,
            validated_program_bank=validated_bank, bindings=bindings,
            callback=callback,
        )
    finally:
        loaded.clear()
