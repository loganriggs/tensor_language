"""Outcome-blind execution/scoring primitives for the L13H8 bracket canary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

import bracket_closure_canary_v1 as canary
import bracket_closure_masks_v1 as mask_module
import bracket_closure_tensor_v1 as tensor_program
import circuit_campaign_runtime as campaign
from tensor_preserving_attention import TensorPreservingSquaredAttention


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_026_083_013
BOOTSTRAP_ORDER_INDEX = 18_999
BATCH_SIZE = 4
ROLE_ORDER = ("select", "ood")
CELL_ORDER = (
    "compatible_closer", "incompatible_closer", "no_opener",
    "quote_control", "punctuation_control", "all",
)
NO_GO_BLOCKERS: tuple[str, ...] = ()
REPLAY_MAX_ABS_LOGIT = 1e-4
REPLAY_POOLED_TEACHER_KL = 1e-8
COLLATERAL_CE = 0.01
OOD_EXTRACTION_POINT = 0.60
OOD_EXTRACTION_LCB = 0.40
OOD_SELECT_RETENTION = 0.50


def require_launch_ready(authority: "ExecutionAuthority") -> None:
    if type(authority) is not ExecutionAuthority:
        raise ValueError("launch requires an exact ExecutionAuthority")
    if not authority.authorized_for_forward:
        raise RuntimeError("bracket execution lacks source-bound ruling/audit authority")


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
    row_support_sha256: tuple[tuple[str, str], ...]
    row_document_ids_sha256: tuple[tuple[str, str], ...]
    delimiter_family_names: tuple[str, ...]
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
        if type(self.row_support_sha256) is not tuple or tuple(
            item[0] for item in self.row_support_sha256
        ) != ("fit", "select", "ood") or any(not _sha(item[1]) for item in self.row_support_sha256):
            raise ValueError("execution support roles differ from fresh-row authority")
        if type(self.row_document_ids_sha256) is not tuple or tuple(
            item[0] for item in self.row_document_ids_sha256
        ) != ("fit", "select", "ood") or any(
            not _sha(item[1]) for item in self.row_document_ids_sha256
        ):
            raise ValueError("execution document roles differ from fresh-row authority")
        if type(self.delimiter_family_names) is not tuple or len(
            self.delimiter_family_names
        ) < 2 or any(not isinstance(name, str) or not name for name in self.delimiter_family_names) \
                or len(set(self.delimiter_family_names)) != len(self.delimiter_family_names):
            raise ValueError("execution delimiter families are malformed")
        if type(self.programs) is not tuple or tuple(
            item.arm for item in self.programs
        ) != canary.ARM_NAMES[1:]:
            raise ValueError("execution programs differ from exact stored arm order")
        if type(self.authorized_for_forward) is not bool:
            raise ValueError("authorized_for_forward must be exact boolean")
        if self.authorized_for_forward:
            if not _sha(self.inference_ruling_sha256) or not _sha(self.independent_audit_sha256):
                raise ValueError("enabled authority needs exact ruling and independent audit hashes")
        elif self.inference_ruling_sha256 is not None or self.independent_audit_sha256 is not None:
            raise ValueError("disabled authority cannot carry promotive ruling/audit hashes")


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


def _logical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


@dataclass(frozen=True)
class RoleMaterialization:
    role: str
    rows: torch.Tensor
    document_ids: tuple[str, ...]
    masks: mask_module.BracketMasks

    def validate(self, authority: ExecutionAuthority) -> None:
        if self.role not in ROLE_ORDER or self.rows.device.type != "cpu" or (
            self.rows.dtype != torch.long or self.rows.ndim != 2 or self.rows.shape[1] != 257
            or not self.rows.is_contiguous() or len(self.document_ids) != self.rows.shape[0]
            or len(set(self.document_ids)) != len(self.document_ids)
        ):
            raise ValueError("role materialization currency changed")
        self.masks.validate()
        support = dict(authority.row_support_sha256)[self.role]
        documents = dict(authority.row_document_ids_sha256)[self.role]
        if canary.support_sha256(self.rows, self.masks) != support or _logical_sha(
            list(self.document_ids)
        ) != documents:
            raise RuntimeError("role materialization differs from authority")


def _slice_masks(masks: mask_module.BracketMasks, start: int, stop: int) -> mask_module.BracketMasks:
    return mask_module.BracketMasks(*(
        getattr(masks, field)[start:stop].contiguous()
        for field in (
            "compatible", "incompatible", "no_opener", "quote_control",
            "punctuation_control", "family_index", "depth", "distance", "domain_index",
        )
    ))


def execute_loaded_roles(
    model: torch.nn.Module, roles: tuple[RoleMaterialization, RoleMaterialization],
    permutation: torch.Tensor, authority: ExecutionAuthority, *, source_guard,
) -> tuple[RoleSufficientStatistics, RoleSufficientStatistics, dict[
    tuple[str, str], tuple[campaign.ForwardClosure, ...]
]]:
    """Own the exact two-role/four-arm forward topology after a preflight-only load."""
    require_launch_ready(authority)
    if tuple(role.role for role in roles) != ROLE_ORDER or not callable(source_guard):
        raise ValueError("execution roles/guard differ from frozen topology")
    for role in roles:
        role.validate(authority)
    if set(roles[0].document_ids) & set(roles[1].document_ids):
        raise RuntimeError("SELECT/OOD documents overlap")
    try:
        attention = model.transformer.h[tensor_program.TARGET_SITE].attn
        device = next(model.parameters()).device
    except (AttributeError, IndexError, StopIteration) as error:
        raise ValueError("execution model does not expose exact L13 attention") from error
    bank = materialize_program_bank(attention, permutation)
    bank.validate(authority)
    closures: dict[tuple[str, str], list[campaign.ForwardClosure]] = {
        (role, arm): [] for role in ROLE_ORDER for arm in canary.ARM_NAMES
    }
    output = []
    for role in roles:
        source_guard(); bank.validate(authority); role.validate(authority)
        batches = []
        for start in range(0, role.rows.shape[0], BATCH_SIZE):
            stop = min(start + BATCH_SIZE, role.rows.shape[0])
            tokens = role.rows[start:stop, :-1].to(device=device, non_blocking=False)
            logits_by_arm: dict[str, torch.Tensor] = {}
            for arm in canary.ARM_NAMES:
                logits, closure = run_one_batch(
                    model, tokens, arm, bank, authority, require_production=True,
                )
                logits_by_arm[arm] = logits
                closures[(role.role, arm)].append(closure)
            batches.append(collect_role_statistics(
                role.role, role.rows[start:stop].contiguous(), role.document_ids[start:stop],
                _slice_masks(role.masks, start, stop), authority.delimiter_family_names,
                logits_by_arm,
            ))
            del logits_by_arm, tokens
        output.append(merge_role_statistics(tuple(batches)))
    source_guard(); bank.validate(authority)
    frozen_closures = {key: tuple(value) for key, value in closures.items()}
    validate_execution_ledgers(
        frozen_closures, {role.role: role.rows.shape[0] for role in roles},
    )
    return output[0], output[1], frozen_closures


@dataclass(frozen=True)
class RoleSufficientStatistics:
    role: str
    document_ids: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    counts: torch.Tensor       # [D,C]
    ce_sums: torch.Tensor      # [4,D,C]
    teacher_kl_sums: torch.Tensor
    correct_sums: torch.Tensor
    replay_max_abs_logit: float

    def validate(self) -> None:
        if self.role not in ROLE_ORDER or len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("role/document identity is malformed")
        if type(self.coordinate_names) is not tuple or not self.coordinate_names or any(
            not isinstance(name, str) or not name for name in self.coordinate_names
        ) or len(set(self.coordinate_names)) != len(self.coordinate_names):
            raise ValueError("role coordinate registry is malformed")
        shape = (len(self.document_ids), len(self.coordinate_names))
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


def registered_coordinate_names(family_names: tuple[str, ...]) -> tuple[str, ...]:
    if type(family_names) is not tuple or len(family_names) < 2 or any(
        not isinstance(name, str) or not name for name in family_names
    ) or len(set(family_names)) != len(family_names):
        raise ValueError("score family registry is malformed")
    return tuple(
        name
        for domain in ("prose", "code")
        for name in (
            *(f"{domain}:{cell}" for cell in CELL_ORDER),
            *(f"{domain}:family:{family}:compatible_closer" for family in family_names),
        )
    )


def score_coordinate_masks(
    masks: mask_module.BracketMasks, family_names: tuple[str, ...],
) -> Mapping[str, torch.Tensor]:
    """Return the exact domain/base/family score registry; never used for execution."""
    masks.validate()
    registered_coordinate_names(family_names)
    observed = masks.family_index[masks.family_index.ge(0)]
    if observed.numel() == 0 or int(observed.max()) >= len(family_names):
        raise ValueError("score masks do not fit authority family registry")
    scored = torch.zeros_like(masks.compatible)
    scored[:, 64:256] = True
    output: dict[str, torch.Tensor] = {}
    base = dict(masks.named_cells()); base["all"] = scored
    for domain_index, domain in enumerate(mask_module.BracketDomain):
        in_domain = masks.domain_index.eq(domain_index)
        for cell in CELL_ORDER:
            output[f"{domain.value}:{cell}"] = base[cell] & in_domain & scored
        for family_index, family_name in enumerate(family_names):
            output[f"{domain.value}:family:{family_name}:compatible_closer"] = (
                masks.compatible & masks.family_index.eq(family_index) & in_domain & scored
            )
    if tuple(output) != registered_coordinate_names(family_names):
        raise AssertionError("score coordinate construction order changed")
    return output


def collect_role_statistics(
    role: str, rows: torch.Tensor, document_ids: tuple[str, ...],
    masks: mask_module.BracketMasks, family_names: tuple[str, ...],
    logits_by_arm: Mapping[str, torch.Tensor],
) -> RoleSufficientStatistics:
    if role not in ROLE_ORDER or tuple(logits_by_arm) != canary.ARM_NAMES:
        raise ValueError("role arms differ from frozen order")
    documents = rows.shape[0]
    if rows.dtype != torch.long or tuple(rows.shape) != (documents, 257) or (
        len(document_ids) != documents
    ):
        raise ValueError("role rows/document identities changed")
    expected_logits = (documents, 256, 50_304)
    if any(value.dtype != torch.float32 or tuple(value.shape) != expected_logits
           or not bool(torch.isfinite(value).all()) for value in logits_by_arm.values()):
        raise ValueError("role logits currency changed")
    logits_device = next(iter(logits_by_arm.values())).device
    coordinate_masks = score_coordinate_masks(masks, family_names)
    names = tuple(coordinate_masks)
    mask_values = [coordinate_masks[name].to(logits_device) for name in names]
    if any(mask.dtype != torch.bool or tuple(mask.shape) != (documents, 256)
           for mask in mask_values):
        raise ValueError("role mask currency changed")
    targets = rows[:, 1:].to(logits_device)
    # The physical facade returns one float32 30*tanh(logit/30) output.  Convert that
    # exact published currency to float64 before normalization and every reduction.
    native_logp = F.log_softmax(logits_by_arm["native"].double(), dim=-1)
    native_p = native_logp.exp()
    counts = torch.stack([mask.sum(1) for mask in mask_values], 1).cpu().to(torch.int64)
    ce, kl, correct = [], [], []
    for arm in canary.ARM_NAMES:
        logits = logits_by_arm[arm]
        logp = F.log_softmax(logits.double(), dim=-1)
        per_ce = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        per_kl = (native_p * (native_logp - logp)).sum(-1)
        per_correct = logits.argmax(-1).eq(targets).to(torch.float64)
        ce.append(torch.stack([(per_ce * mask).sum(1) for mask in mask_values], 1).double().cpu())
        kl.append(torch.stack([(per_kl * mask).sum(1) for mask in mask_values], 1).double().cpu())
        correct.append(torch.stack([(per_correct * mask).sum(1) for mask in mask_values], 1).cpu())
    replay_error = float((
        logits_by_arm[tensor_program.BracketTensorArm.STORED_ALL_HEADS.value]
        - logits_by_arm["native"]
    ).abs().max().item())
    result = RoleSufficientStatistics(
        role, document_ids, names, counts, torch.stack(ce), torch.stack(kl),
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


def _arm_document_ce(stats: RoleSufficientStatistics) -> torch.Tensor:
    denominator = stats.counts.double().unsqueeze(0)
    return (stats.ce_sums / denominator.clamp_min(1)).masked_fill(
        denominator == 0, float("nan")
    )


def _role_arm_bootstrap(stats: RoleSufficientStatistics, generator: torch.Generator) -> tuple[
    torch.Tensor, torch.Tensor,
]:
    """Document-balanced arm CE; all arms/coordinates share each role draw."""
    stats.validate()
    values = _arm_document_ce(stats)  # [A,D,C]
    finite = torch.isfinite(values)
    if not torch.equal(finite, finite[:1].expand_as(finite)) or bool(
        (finite[0].sum(0) < 2).any()
    ):
        raise ValueError("arm support differs or has fewer than two source documents")
    point = torch.nansum(values, 1) / finite.sum(1)
    output = torch.empty(
        BOOTSTRAP_DRAWS, len(canary.ARM_NAMES), len(stats.coordinate_names),
        dtype=torch.float64,
    )
    for draw in range(BOOTSTRAP_DRAWS):
        index = torch.randint(values.shape[1], (values.shape[1],), generator=generator)
        sampled = values[:, index]; okay = torch.isfinite(sampled)
        if bool((okay[:, :, :].sum(1) == 0).any()):
            raise RuntimeError("role bootstrap replicate lost coordinate support")
        output[draw] = torch.nansum(sampled, 1) / okay.sum(1)
    return point, output


@dataclass(frozen=True)
class ExecutionIntegrity:
    source_replayed: bool
    row_receipt_replayed: bool
    role_disjointness_replayed: bool
    program_replayed: bool
    common_support_replayed: bool
    call_ledgers_replayed: bool
    finite_outputs: bool

    def passed(self) -> bool:
        values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if any(type(value) is not bool for value in values):
            raise ValueError("integrity flags must be exact booleans")
        return all(values)


@dataclass(frozen=True)
class BracketScore:
    coordinate_order: tuple[str, ...]
    point: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    simultaneous_critical: float
    replay: tuple[tuple[str, float, float], ...]
    decisions: tuple[tuple[str, bool], ...]
    promoted: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "bracket_closure_execution_v1_result",
            "inference": {
                "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                "order_index": BOOTSTRAP_ORDER_INDEX,
                "simultaneous_critical": self.simultaneous_critical,
            },
            "coordinates": {
                name: {"point": self.point[index], "lower": self.lower[index],
                       "upper": self.upper[index]}
                for index, name in enumerate(self.coordinate_order)
            },
            "replay": {role: {"max_abs_logit": maximum, "pooled_teacher_kl": kl}
                       for role, maximum, kl in self.replay},
            "decisions": dict(self.decisions),
            "promoted": self.promoted,
            "claim_boundary": (
                "exact zero-native stored extraction/OOD/removal evidence only; "
                "no compression, simplification, standalone, or stack-algorithm credit"
            ),
        }


def _metric_coordinates(
    role_points: Mapping[str, torch.Tensor], role_replicates: Mapping[str, torch.Tensor],
    names: tuple[str, ...], family_names: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Recompute every nonlinear contrast for every bootstrap realization."""
    arm = {name: index for index, name in enumerate(canary.ARM_NAMES)}
    index = {name: position for position, name in enumerate(names)}
    points: dict[str, float] = {}; replicates: dict[str, torch.Tensor] = {}
    extraction: dict[tuple[str, str], tuple[float, torch.Tensor]] = {}
    recovery: dict[tuple[str, str], tuple[float, torch.Tensor]] = {}
    for role in ROLE_ORDER:
        point_arm, replicate_arm = role_points[role], role_replicates[role]
        for domain in ("prose", "code"):
            def damage(cell: str, arm_name: str) -> tuple[float, torch.Tensor]:
                column = index[f"{domain}:{cell}"]
                value = point_arm[arm[arm_name], column] - point_arm[arm["native"], column]
                reps = replicate_arm[:, arm[arm_name], column] - replicate_arm[:, arm["native"], column]
                return float(value), reps
            deletion, deletion_reps = damage("compatible_closer", canary.ARM_NAMES[2])
            stored, stored_reps = damage("compatible_closer", canary.ARM_NAMES[1])
            spectral, spectral_reps = damage("compatible_closer", canary.ARM_NAMES[3])
            if deletion <= 0 or bool((deletion_reps <= 0).any()):
                raise RuntimeError("normalized extraction is undefined for a nonpositive deletion stake")
            e = (deletion - stored) / deletion
            e_reps = (deletion_reps - stored_reps) / deletion_reps
            r = (deletion - spectral) / deletion
            r_reps = (deletion_reps - spectral_reps) / deletion_reps
            extraction[(role, domain)] = (e, e_reps)
            recovery[(role, domain)] = (r, r_reps)
            controls = [damage(cell, canary.ARM_NAMES[2]) for cell in CELL_ORDER[1:5]]
            specificity = deletion - max(value for value, _ in controls)
            specificity_reps = deletion_reps - torch.stack(
                [value for _, value in controls], 1
            ).max(1).values
            collateral, collateral_reps = damage("all", canary.ARM_NAMES[2])
            role_prefix = f"{role}:{domain}"
            metrics = {
                f"{role_prefix}:deletion_stake": (deletion, deletion_reps),
                f"{role_prefix}:specificity": (specificity, specificity_reps),
                f"{role_prefix}:collateral_margin": (
                    COLLATERAL_CE - collateral, COLLATERAL_CE - collateral_reps,
                ),
                f"{role_prefix}:true_vs_null": (e - r, e_reps - r_reps),
                f"{role_prefix}:spectral_margin": (
                    0.5 * e - r, torch.full_like(r_reps, 0.5 * e) - r_reps,
                ),
            }
            if role == "ood":
                metrics[f"{role_prefix}:extraction_minus_lcb_floor"] = (
                    e - OOD_EXTRACTION_LCB, e_reps - OOD_EXTRACTION_LCB,
                )
            for key, (value, reps) in metrics.items():
                points[key] = float(value); replicates[key] = reps
        if role == "select":
            for domain in ("prose", "code"):
                point_arm, replicate_arm = role_points[role], role_replicates[role]
                for family_name in family_names:
                    column = index[f"{domain}:family:{family_name}:compatible_closer"]
                    key = f"select:{domain}:family:{family_name}:deletion_stake"
                    points[key] = float(
                        point_arm[arm[canary.ARM_NAMES[2]], column] - point_arm[0, column]
                    )
                    replicates[key] = (
                        replicate_arm[:, arm[canary.ARM_NAMES[2]], column]
                        - replicate_arm[:, 0, column]
                    )
    for domain in ("prose", "code"):
        e_ood, reps_ood = extraction[("ood", domain)]
        e_select, reps_select = extraction[("select", domain)]
        key = f"ood:{domain}:select_retention"
        points[key] = e_ood - OOD_SELECT_RETENTION * e_select
        replicates[key] = reps_ood - OOD_SELECT_RETENTION * reps_select
    return points, replicates


def score_roles(
    select: RoleSufficientStatistics, ood: RoleSufficientStatistics,
    integrity: ExecutionIntegrity, family_names: tuple[str, ...],
) -> BracketScore:
    """One authoritative joint-family score from unopened role statistics."""
    if (select.role, ood.role) != ROLE_ORDER or select.coordinate_names != ood.coordinate_names:
        raise ValueError("score roles/order/coordinate registry changed")
    if select.coordinate_names != registered_coordinate_names(family_names):
        raise ValueError("score coordinates differ from authority")
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    role_points, role_replicates = {}, {}
    for stats in (select, ood):
        point, reps = _role_arm_bootstrap(stats, generator)
        role_points[stats.role] = point; role_replicates[stats.role] = reps
    points, reps = _metric_coordinates(
        role_points, role_replicates, select.coordinate_names, family_names,
    )
    order = tuple(points)
    point_vector = torch.tensor([points[name] for name in order], dtype=torch.float64)
    replicate_matrix = torch.stack([reps[name] for name in order], 1)
    if not bool(torch.isfinite(point_vector).all()) or not bool(torch.isfinite(replicate_matrix).all()):
        raise RuntimeError("score coordinates are nonfinite")
    critical = float(torch.sort(
        (replicate_matrix - point_vector).abs().max(1).values
    ).values[BOOTSTRAP_ORDER_INDEX])
    lower, upper = point_vector - critical, point_vector + critical
    bounds = {name: (float(point_vector[i]), float(lower[i]), float(upper[i]))
              for i, name in enumerate(order)}
    replay = []
    replay_pass = True
    stored_index = canary.ARM_NAMES.index(canary.ARM_NAMES[1])
    all_columns = [i for i, name in enumerate(select.coordinate_names) if name.endswith(":all")]
    for stats in (select, ood):
        pooled_kl = float(stats.teacher_kl_sums[stored_index, :, all_columns].sum()
                          / stats.counts[:, all_columns].sum())
        replay.append((stats.role, stats.replay_max_abs_logit, pooled_kl))
        replay_pass &= stats.replay_max_abs_logit <= REPLAY_MAX_ABS_LOGIT and (
            pooled_kl <= REPLAY_POOLED_TEACHER_KL
        )
    decisions: dict[str, bool] = {
        "common_integrity": integrity.passed(), "replay": replay_pass,
    }
    for role in ROLE_ORDER:
        role_checks = []
        for domain in ("prose", "code"):
            prefix = f"{role}:{domain}"
            required = (
                f"{prefix}:deletion_stake", f"{prefix}:specificity",
                f"{prefix}:collateral_margin", f"{prefix}:true_vs_null",
                f"{prefix}:spectral_margin",
            )
            domain_pass = all(bounds[name][1] > 0 for name in required[:2] + required[3:]) \
                and bounds[required[2]][1] >= 0
            if role == "select":
                family_pass = all(
                    bounds[f"select:{domain}:family:{family}:deletion_stake"][0] > 0
                    for family in family_names
                )
                domain_pass &= family_pass
            else:
                e, _ = extraction_from_arm_means(
                    role_points[role], select.coordinate_names, domain,
                )
                domain_pass &= e >= OOD_EXTRACTION_POINT and bounds[
                    f"{prefix}:extraction_minus_lcb_floor"
                ][1] >= 0 and bounds[f"ood:{domain}:select_retention"][1] >= 0
            decisions[f"{role}:{domain}"] = bool(domain_pass)
            role_checks.append(bool(domain_pass))
        decisions[role] = all(role_checks)
    promoted = all(decisions.values())
    return BracketScore(
        order, tuple(float(value) for value in point_vector),
        tuple(float(value) for value in lower), tuple(float(value) for value in upper),
        critical, tuple(replay), tuple(decisions.items()), promoted,
    )


def extraction_from_arm_means(
    arm_means: torch.Tensor, coordinate_names: tuple[str, ...], domain: str,
) -> tuple[float, float]:
    index = {name: position for position, name in enumerate(coordinate_names)}
    column = index[f"{domain}:compatible_closer"]
    native, stored, deletion = (float(arm_means[i, column]) for i in range(3))
    stake = deletion - native
    if stake <= 0:
        raise RuntimeError("extraction requires positive deletion stake")
    return (deletion - stored) / stake, stake


def merge_role_statistics(
    batches: tuple[RoleSufficientStatistics, ...],
) -> RoleSufficientStatistics:
    """Merge batch-local sufficient statistics without changing document currency."""
    if type(batches) is not tuple or not batches:
        raise ValueError("role statistic merge needs a nonempty exact tuple")
    for batch in batches:
        if type(batch) is not RoleSufficientStatistics:
            raise ValueError("role statistic merge received an untyped batch")
        batch.validate()
    role, names = batches[0].role, batches[0].coordinate_names
    if any(batch.role != role or batch.coordinate_names != names for batch in batches):
        raise ValueError("role statistic batches do not share role/coordinates")
    document_ids = tuple(document for batch in batches for document in batch.document_ids)
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("role statistic batches repeat source documents")
    result = RoleSufficientStatistics(
        role, document_ids, names,
        torch.cat([batch.counts for batch in batches], 0),
        torch.cat([batch.ce_sums for batch in batches], 1),
        torch.cat([batch.teacher_kl_sums for batch in batches], 1),
        torch.cat([batch.correct_sums for batch in batches], 1),
        max(batch.replay_max_abs_logit for batch in batches),
    )
    result.validate()
    return result


def validate_execution_ledgers(
    closures: Mapping[tuple[str, str], tuple[campaign.ForwardClosure, ...]],
    expected_documents: Mapping[str, int],
) -> None:
    """Replay exact four-arm physical calls and document coverage for both roles."""
    expected_keys = tuple((role, arm) for role in ROLE_ORDER for arm in canary.ARM_NAMES)
    if tuple(closures) != expected_keys or tuple(expected_documents) != ROLE_ORDER or any(
        type(expected_documents[role]) is not int or expected_documents[role] <= 0
        for role in ROLE_ORDER
    ):
        raise ValueError("execution ledger registry changed")
    for role, arm in expected_keys:
        entries = closures[(role, arm)]
        if type(entries) is not tuple or not entries:
            raise ValueError("execution ledger arm has no batches")
        documents = 0
        for closure in entries:
            canary.validate_forward_closure(
                closure, arm, document_count=closure.document_count,
            )
            documents += closure.document_count
        if documents != expected_documents[role]:
            raise RuntimeError("execution ledger document coverage changed")


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
        if bool((okay.sum(0) == 0).any()):
            raise RuntimeError("bootstrap replicate lost all support for a coordinate")
        replicates[draw] = torch.nansum(sampled, 0) / okay.sum(0).clamp_min(1)
    maximum_error = (replicates - point).abs().max(1).values
    critical = float(torch.sort(maximum_error).values[BOOTSTRAP_ORDER_INDEX])
    return point, replicates, critical


def joint_role_bootstrap(
    select_values: torch.Tensor, ood_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Independent role draws; all coordinates within a role share multiplicities."""
    for value in (select_values, ood_values):
        if value.dtype != torch.float64 or value.device.type != "cpu" or value.ndim != 2 \
                or bool((torch.isfinite(value).sum(0) < 2).any()):
            raise ValueError("joint role bootstrap input is malformed")
    point = torch.cat([
        torch.nansum(value, 0) / torch.isfinite(value).sum(0)
        for value in (select_values, ood_values)
    ])
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    output = torch.empty(BOOTSTRAP_DRAWS, point.numel(), dtype=torch.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        pieces = []
        for value in (select_values, ood_values):
            index = torch.randint(value.shape[0], (value.shape[0],), generator=generator)
            sampled = value[index]; finite = torch.isfinite(sampled)
            if bool((finite.sum(0) == 0).any()):
                raise RuntimeError("joint bootstrap replicate lost coordinate support")
            pieces.append(torch.nansum(sampled, 0) / finite.sum(0))
        output[draw] = torch.cat(pieces)
    critical = float(torch.sort((output - point).abs().max(1).values).values[
        BOOTSTRAP_ORDER_INDEX
    ])
    return point, output, critical


__all__ = (
    "BOOTSTRAP_DRAWS", "BracketScore", "CELL_ORDER", "ExecutionAuthority",
    "ExecutionIntegrity", "ImmutableProgramBank", "NO_GO_BLOCKERS", "ProgramAuthority",
    "RoleSufficientStatistics",
    "bootstrap_means", "collect_role_statistics", "document_means",
    "joint_role_bootstrap", "materialize_program_bank", "merge_role_statistics",
    "registered_coordinate_names", "require_launch_ready", "run_one_batch",
    "score_coordinate_masks", "score_roles",
    "validate_execution_ledgers",
)
