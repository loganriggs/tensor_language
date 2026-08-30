"""Outcome-blind execution/scoring primitives for the L13H8 bracket canary.

Launch deliberately remains NO-GO; see ``BRACKET_CLOSURE_EXECUTION_V1_NO_GO.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

import bracket_closure_canary_v1 as canary
import bracket_closure_tensor_v1 as tensor_program
import circuit_campaign_runtime as campaign
from tensor_preserving_attention import TensorPreservingSquaredAttention


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_026_083_013
BOOTSTRAP_ORDER_INDEX = 18_999
ROLE_ORDER = ("select", "ood")
CELL_ORDER = (
    "compatible_closer", "incompatible_closer", "no_opener",
    "quote_control", "punctuation_control", "all",
)
NO_GO_BLOCKERS = (
    "old separate-domain plus synthetic roles conflict with mixed-domain FIT/SELECT/OOD",
    "document bootstrap cluster strata after role merge are undefined",
    "SELECT/OOD simultaneous-family relationship is undefined",
    "exact OOD conjunction and non-rescue rule are undefined",
)


def require_launch_ready(authority: "ExecutionAuthority") -> None:
    if type(authority) is not ExecutionAuthority:
        raise ValueError("launch requires an exact ExecutionAuthority")
    if authority.authorized_for_forward:
        raise RuntimeError("authority illegally enables unresolved bracket inference")
    raise RuntimeError("bracket execution is prospectively NO-GO: " + "; ".join(NO_GO_BLOCKERS))


def _sha(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class ProgramAuthority:
    arm: str
    state_sha256: str
    stored_values: int
    native_calls_per_forward: int
    token_table_values: int
    total_input_support: bool

    def __post_init__(self) -> None:
        if self.arm not in canary.ARM_NAMES[1:] or not _sha(self.state_sha256):
            raise ValueError("bracket program authority identity is malformed")
        if type(self.stored_values) is not int or (
            self.stored_values != tensor_program.PRODUCTION_STORED_VALUES
        ) or type(self.native_calls_per_forward) is not int or (
            self.native_calls_per_forward != 0
        ) or type(self.token_table_values) is not int or self.token_table_values != 0 or (
            type(self.total_input_support) is not bool or not self.total_input_support
        ):
            raise ValueError("bracket program authority price/call contract changed")


@dataclass(frozen=True)
class ExecutionAuthority:
    source_commit: str
    source_hashes: tuple[tuple[str, str], ...]
    row_receipt_sha256: str
    row_role_file_sha256: tuple[tuple[str, str], ...]
    model_config_sha256: str
    model_weights_sha256: str
    derangement_sha256: str
    programs: tuple[ProgramAuthority, ...]
    authorized_for_forward: bool
    inference_ruling_sha256: str | None
    independent_audit_sha256: str | None

    def __post_init__(self) -> None:
        if not _sha(self.source_commit, 40) or type(self.source_hashes) is not tuple or any(
            type(item) is not tuple or len(item) != 2 or not isinstance(item[0], str)
            or not _sha(item[1]) for item in self.source_hashes
        ):
            raise ValueError("execution source binding is malformed")
        for value in (
            self.row_receipt_sha256, self.model_config_sha256,
            self.model_weights_sha256, self.derangement_sha256,
        ):
            if not _sha(value):
                raise ValueError("execution input hash is malformed")
        if type(self.row_role_file_sha256) is not tuple or tuple(
            item[0] for item in self.row_role_file_sha256
        ) != ("fit", "select", "ood") or any(not _sha(item[1]) for item in self.row_role_file_sha256):
            raise ValueError("execution row roles differ from fresh-row authority")
        if type(self.programs) is not tuple or tuple(
            item.arm for item in self.programs
        ) != canary.ARM_NAMES[1:]:
            raise ValueError("execution programs differ from exact stored arm order")
        if type(self.authorized_for_forward) is not bool:
            raise ValueError("authorized_for_forward must be exact boolean")
        if self.authorized_for_forward or self.inference_ruling_sha256 is not None or (
            self.independent_audit_sha256 is not None
        ):
            raise ValueError("unresolved v1 authority must remain forward-disabled without rulings")


class ImmutableProgramBank:
    __slots__ = ("_programs", "_hashes")

    def __init__(self, programs: Mapping[str, TensorPreservingSquaredAttention]):
        if tuple(programs) != canary.ARM_NAMES[1:] or any(
            not isinstance(value, TensorPreservingSquaredAttention) for value in programs.values()
        ):
            raise ValueError("program bank must contain exact stored bracket arm order")
        self._programs = dict(programs)
        self._hashes = {name: tensor_program.program_state_sha256(value)
                        for name, value in self._programs.items()}

    def validate(self, authority: ExecutionAuthority) -> None:
        expected = {item.arm: item.state_sha256 for item in authority.programs}
        current = {name: tensor_program.program_state_sha256(value)
                   for name, value in self._programs.items()}
        if current != self._hashes or current != expected:
            raise RuntimeError("materialized bracket program mutated or differs from authority")
        for name, program in self._programs.items():
            receipt = program.cost_receipt()
            binding = next(item for item in authority.programs if item.arm == name)
            if (receipt.total_stored_values, receipt.native_calls_per_forward,
                receipt.token_table_values, receipt.total_input_support) != (
                binding.stored_values, binding.native_calls_per_forward,
                binding.token_table_values, binding.total_input_support,
            ):
                raise RuntimeError("materialized bracket program receipt changed")

    def callback(self, arm: str, authority: ExecutionAuthority):
        self.validate(authority)
        if arm not in self._programs:
            raise ValueError("native arm has no replacement callback")
        return canary.make_attention_replacement(self._programs[arm])


def materialize_program_bank(
    native_attention: torch.nn.Module, permutation: torch.Tensor,
) -> ImmutableProgramBank:
    programs = {
        tensor_program.BracketTensorArm.STORED_ALL_HEADS.value:
            tensor_program.build_bracket_tensor_program(
                native_attention, tensor_program.BracketTensorArm.STORED_ALL_HEADS,
            ),
        tensor_program.BracketTensorArm.DELETE_H8.value:
            tensor_program.build_bracket_tensor_program(
                native_attention, tensor_program.BracketTensorArm.DELETE_H8,
            ),
        tensor_program.BracketTensorArm.DERANGED_H8.value:
            tensor_program.build_bracket_tensor_program(
                native_attention, tensor_program.BracketTensorArm.DERANGED_H8,
                permutation=permutation,
            ),
    }
    return ImmutableProgramBank(programs)


def run_one_batch(
    model: torch.nn.Module, tokens: torch.Tensor, arm: str,
    bank: ImmutableProgramBank, authority: ExecutionAuthority,
    *, require_production: bool = True,
) -> tuple[torch.Tensor, campaign.ForwardClosure]:
    require_launch_ready(authority)  # Must fail before model or row use in current v1.
    plan = canary.build_circuit_plan()
    callbacks = {} if arm == "native" else {arm: bank.callback(arm, authority)}
    owner = campaign.CircuitForwardOwner(
        plan=plan, arm=arm, attention_replacements=callbacks,
    )
    logits = owner.run(model, tokens, require_production=require_production)
    canary.validate_forward_closure(owner.closure, arm, document_count=tokens.shape[0])
    return logits, owner.closure


@dataclass(frozen=True)
class RoleSufficientStatistics:
    role: str
    document_ids: tuple[str, ...]
    counts: torch.Tensor       # [D,6]
    ce_sums: torch.Tensor      # [4,D,6]
    teacher_kl_sums: torch.Tensor
    correct_sums: torch.Tensor
    replay_max_abs_logit: float

    def validate(self) -> None:
        if self.role not in ROLE_ORDER or len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("role/document identity is malformed")
        shape = (len(self.document_ids), len(CELL_ORDER))
        if self.counts.dtype != torch.int64 or tuple(self.counts.shape) != shape or (
            self.counts.device.type != "cpu" or bool((self.counts < 0).any())
        ):
            raise ValueError("role count currency changed")
        expected = (len(canary.ARM_NAMES), *shape)
        for value in (self.ce_sums, self.teacher_kl_sums, self.correct_sums):
            if value.dtype != torch.float64 or value.device.type != "cpu" or (
                tuple(value.shape) != expected or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("role sufficient-statistic currency changed")
        if not isinstance(self.replay_max_abs_logit, float) or self.replay_max_abs_logit < 0:
            raise ValueError("replay maximum logit error is malformed")


def collect_role_statistics(
    role: str, rows: torch.Tensor, document_ids: tuple[str, ...],
    cell_masks: Mapping[str, torch.Tensor], logits_by_arm: Mapping[str, torch.Tensor],
) -> RoleSufficientStatistics:
    if role not in ROLE_ORDER or tuple(logits_by_arm) != canary.ARM_NAMES or tuple(
        cell_masks
    ) != CELL_ORDER[:-1]:
        raise ValueError("role arms/cells differ from frozen order")
    documents = rows.shape[0]
    if rows.dtype != torch.long or tuple(rows.shape) != (documents, 257) or (
        len(document_ids) != documents
    ):
        raise ValueError("role rows/document identities changed")
    expected_logits = (documents, 256, 50_304)
    if any(value.dtype != torch.float32 or tuple(value.shape) != expected_logits
           or not bool(torch.isfinite(value).all()) for value in logits_by_arm.values()):
        raise ValueError("role logits currency changed")
    scored = torch.zeros((documents, 256), dtype=torch.bool, device=rows.device)
    scored[:, 64:256] = True
    masks = [cell_masks[name].to(rows.device) & scored for name in CELL_ORDER[:-1]] + [scored]
    if any(mask.dtype != torch.bool or tuple(mask.shape) != (documents, 256) for mask in masks):
        raise ValueError("role mask currency changed")
    targets = rows[:, 1:].to(next(iter(logits_by_arm.values())).device)
    native_logp = F.log_softmax(logits_by_arm["native"], dim=-1)
    native_p = native_logp.exp()
    counts = torch.stack([mask.sum(1) for mask in masks], 1).cpu().to(torch.int64)
    ce, kl, correct = [], [], []
    for arm in canary.ARM_NAMES:
        logits = logits_by_arm[arm]
        logp = F.log_softmax(logits, dim=-1)
        per_ce = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        per_kl = (native_p * (native_logp - logp)).sum(-1)
        per_correct = logits.argmax(-1).eq(targets).to(torch.float64)
        ce.append(torch.stack([(per_ce * mask).sum(1) for mask in masks], 1).double().cpu())
        kl.append(torch.stack([(per_kl * mask).sum(1) for mask in masks], 1).double().cpu())
        correct.append(torch.stack([(per_correct * mask).sum(1) for mask in masks], 1).cpu())
    replay_error = float((
        logits_by_arm[tensor_program.BracketTensorArm.STORED_ALL_HEADS.value]
        - logits_by_arm["native"]
    ).abs().max().item())
    result = RoleSufficientStatistics(
        role, document_ids, counts, torch.stack(ce), torch.stack(kl),
        torch.stack(correct), replay_error,
    )
    result.validate(); return result


def document_means(stats: RoleSufficientStatistics, arm: str) -> torch.Tensor:
    """Signed CE(arm)-CE(native), document-balanced with NaN for no support."""
    stats.validate()
    index = canary.ARM_NAMES.index(arm)
    denominator = stats.counts.double()
    value = (stats.ce_sums[index] - stats.ce_sums[0]) / denominator.clamp_min(1)
    return value.masked_fill(denominator == 0, float("nan"))


def bootstrap_means(
    values: torch.Tensor, *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Generic shared-document bootstrap primitive; not an inference ruling."""
    if values.dtype != torch.float64 or values.device.type != "cpu" or values.ndim != 2 \
            or draws != BOOTSTRAP_DRAWS or seed != BOOTSTRAP_SEED:
        raise ValueError("bootstrap input/registered constants changed")
    finite = torch.isfinite(values)
    if bool((finite.sum(0) < 2).any()):
        raise ValueError("every bootstrap coordinate needs at least two documents")
    point = torch.nansum(values, 0) / finite.sum(0)
    generator = torch.Generator().manual_seed(seed)
    replicates = torch.empty(draws, values.shape[1], dtype=torch.float64)
    for draw in range(draws):
        index = torch.randint(values.shape[0], (values.shape[0],), generator=generator)
        sampled = values[index]; okay = torch.isfinite(sampled)
        replicates[draw] = torch.nansum(sampled, 0) / okay.sum(0).clamp_min(1)
    maximum_error = (replicates - point).abs().max(1).values
    critical = float(torch.sort(maximum_error).values[BOOTSTRAP_ORDER_INDEX])
    return point, replicates, critical


__all__ = (
    "BOOTSTRAP_DRAWS", "CELL_ORDER", "ExecutionAuthority", "ImmutableProgramBank",
    "NO_GO_BLOCKERS", "ProgramAuthority", "RoleSufficientStatistics",
    "bootstrap_means", "collect_role_statistics", "document_means",
    "materialize_program_bank", "require_launch_ready", "run_one_batch",
)
