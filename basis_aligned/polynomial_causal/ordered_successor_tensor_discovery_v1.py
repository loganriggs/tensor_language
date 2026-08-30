"""Pure discovery contract for the autonomous L8H7 ordered-successor tensor.

This module loads no rows, model, tokenizer, checkpoint, or outcome.  It freezes the
arm registry, literal storage price, authority schema, common execution-ledger checks,
support hashing, and per-document sufficient-statistic currency needed by a later
production adapter.  It deliberately has no CLI or publication function.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import circuit_campaign_runtime as campaign
import circuit_successor_tensor as tensor
from ordered_successor_masks_v1 import OrderedLexicon, SuccessorMasks


SCHEMA = "ordered_successor_tensor_discovery_v1"
SITE_COUNT = 18
TARGET_SITE = 8
TARGET_HEAD = 7
STATE_DIM = 1152
QK_RANK = 128
NATIVE_VALUE_RANK = 128
SAVED_VALUE_DIM = 128
TOKEN_VOCAB = 50_257
LOGIT_VOCAB = 50_304
ROW_LENGTH = 257
SCORED_START = 64
SCORED_STOP = 256
RANK_LADDER = (8, 16, 32, 64, 96, 128)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_026_083_013
MAX_ERROR_ORDER_INDEX = 18_999  # zero-based 95th order statistic; no interpolation
SOURCE_CLOSURE = (
    "basis_aligned/polynomial_causal/ORDERED_SUCCESSOR_TENSOR_DISCOVERY_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/circuit_campaign_runtime.py",
    "basis_aligned/polynomial_causal/circuit_successor_tensor.py",
    "basis_aligned/polynomial_causal/ordered_successor_masks_v1.py",
    "basis_aligned/polynomial_causal/ordered_successor_tensor_discovery_v1.py",
)

FULL_REPLAY = "full_attention8_replay"
HEAD_DELETED = "head8_7_deleted"
CURRENT_ONLY = "head8_7_current_only_r128"
V1_ONLY = "head8_7_v1_only_r128"


class CandidateKind(str, Enum):
    TRUE = "true"
    SPECTRAL_NULL = "spectral_null"


@dataclass(frozen=True)
class CandidateSpec:
    rank: int
    kind: CandidateKind

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank not in RANK_LADDER:
            raise ValueError("candidate rank is outside the frozen ladder")
        if type(self.kind) is not CandidateKind:
            raise ValueError("candidate kind must be a CandidateKind")

    @property
    def arm(self) -> str:
        return f"head8_7_both_r{self.rank}_{self.kind.value}"

    @property
    def stored_parameters(self) -> int:
        return tensor.autonomous_successor_parameter_count(
            STATE_DIM, SAVED_VALUE_DIM, QK_RANK, self.rank, STATE_DIM,
            include_current=True, include_saved=True,
        )


CANDIDATES = tuple(
    CandidateSpec(rank, kind)
    for rank in RANK_LADDER
    for kind in (CandidateKind.TRUE, CandidateKind.SPECTRAL_NULL)
)
ARM_NAMES = (
    "native",
    FULL_REPLAY,
    HEAD_DELETED,
    *(candidate.arm for candidate in CANDIDATES),
    CURRENT_ONLY,
    V1_ONLY,
)
PROMOTIVE_ARMS = tuple(
    candidate.arm for candidate in CANDIDATES if candidate.kind is CandidateKind.TRUE
)
NULL_BY_TRUE = {
    CandidateSpec(rank, CandidateKind.TRUE).arm:
    CandidateSpec(rank, CandidateKind.SPECTRAL_NULL).arm
    for rank in RANK_LADDER
}


def arm_stored_parameters(arm: str) -> int | None:
    """Return target-component storage; controls without a deployed target return None."""

    if arm == "native" or arm == FULL_REPLAY:
        return tensor.autonomous_successor_parameter_count(
            STATE_DIM, SAVED_VALUE_DIM, QK_RANK, NATIVE_VALUE_RANK, STATE_DIM,
            include_current=True, include_saved=True,
        )
    if arm == HEAD_DELETED:
        return 0
    if arm == CURRENT_ONLY:
        return tensor.autonomous_successor_parameter_count(
            STATE_DIM, SAVED_VALUE_DIM, QK_RANK, NATIVE_VALUE_RANK, STATE_DIM,
            include_current=True, include_saved=False,
        )
    if arm == V1_ONLY:
        return tensor.autonomous_successor_parameter_count(
            STATE_DIM, SAVED_VALUE_DIM, QK_RANK, NATIVE_VALUE_RANK, STATE_DIM,
            include_current=False, include_saved=True,
        )
    matches = tuple(candidate for candidate in CANDIDATES if candidate.arm == arm)
    if len(matches) != 1:
        raise ValueError("arm is outside the frozen successor registry")
    return matches[0].stored_parameters


def build_circuit_plan() -> campaign.CircuitPlan:
    """Build the exact hook-free topology: every nonnative arm replaces attention 8."""

    arms = [campaign.ArmPlan.build(
        "native", campaign.ArmKind.NATIVE, site_count=SITE_COUNT,
    )]
    for name in ARM_NAMES[1:]:
        arms.append(campaign.ArmPlan.build(
            name,
            campaign.ArmKind.CANDIDATE,
            site_count=SITE_COUNT,
            attention_replacements={TARGET_SITE: name},
        ))
    return campaign.CircuitPlan(SCHEMA, SITE_COUNT, tuple(arms))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_git_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class ProgramBinding:
    arm: str
    sha256: str
    stored_parameters: int

    def __post_init__(self) -> None:
        if self.arm not in ARM_NAMES[1:]:
            raise ValueError("program binding arm is not one registered replacement")
        if not _is_sha256(self.sha256):
            raise ValueError("program binding needs one lowercase SHA256")
        expected = arm_stored_parameters(self.arm)
        if type(self.stored_parameters) is not int or self.stored_parameters != expected:
            raise ValueError("program binding stored price differs from the frozen formula")


@dataclass(frozen=True)
class DiscoveryAuthority:
    """Pre-forward bindings. There is intentionally no OOD role in this authority."""

    source_commit: str
    source_files: tuple[tuple[str, str], ...]
    select_rows_sha256: str
    select_support_sha256: str
    select_documents: int
    lexicon_registry_sha256: str
    model_config_sha256: str
    model_weights_sha256: str
    shared_bus_producer_sha256: str
    shared_bus_producer_stored_parameters: int
    programs: tuple[ProgramBinding, ...]

    def __post_init__(self) -> None:
        if not _is_git_hash(self.source_commit):
            raise ValueError("authority source commit must be one full git hash")
        if type(self.source_files) is not tuple or not self.source_files or any(
            type(item) is not tuple or len(item) != 2 or not isinstance(item[0], str)
            or not item[0] or not _is_sha256(item[1]) for item in self.source_files
        ):
            raise ValueError("authority source closure is malformed")
        paths = tuple(item[0] for item in self.source_files)
        if paths != SOURCE_CLOSURE:
            raise ValueError("authority source closure differs from the exact path set")
        for value in (
            self.select_rows_sha256,
            self.select_support_sha256,
            self.lexicon_registry_sha256,
            self.model_config_sha256,
            self.model_weights_sha256,
            self.shared_bus_producer_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("authority hash field is malformed")
        if type(self.select_documents) is not int or self.select_documents <= 0:
            raise ValueError("authority SELECT document count must be positive")
        expected_producer = tensor.shared_bus_producer_parameter_count(
            STATE_DIM, SAVED_VALUE_DIM,
        )
        if type(self.shared_bus_producer_stored_parameters) is not int or (
            self.shared_bus_producer_stored_parameters != expected_producer
        ):
            raise ValueError("authority shared-bus producer price changed")
        if type(self.programs) is not tuple or any(
            type(binding) is not ProgramBinding for binding in self.programs
        ):
            raise ValueError("authority program registry is malformed")
        expected = ARM_NAMES[1:]
        if tuple(binding.arm for binding in self.programs) != expected:
            raise ValueError("authority programs must exactly follow the frozen arm order")
        hashes = tuple(binding.sha256 for binding in self.programs)
        if len(set(hashes)) != len(hashes):
            raise ValueError("authority program hashes must be arm-distinct")


def tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value) or value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("hashed tensor must be one contiguous CPU tensor")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_lexicon_registry(lexicons: Sequence[OrderedLexicon]) -> tuple[OrderedLexicon, ...]:
    if type(lexicons) not in (tuple, list) or not lexicons or any(
        type(lexicon) is not OrderedLexicon for lexicon in lexicons
    ):
        raise ValueError("lexicon registry must contain exact OrderedLexicon objects")
    frozen = tuple(lexicons)
    names = tuple(lexicon.name for lexicon in frozen)
    if len(set(names)) != len(names):
        raise ValueError("lexicon names must be unique")
    owner: dict[int, str] = {}
    for lexicon in frozen:
        for item in lexicon.items:
            for token_id in item:
                if token_id >= TOKEN_VOCAB:
                    raise ValueError("lexicon token ID is outside tokenizer support")
                if token_id in owner:
                    raise ValueError("token ID appears in two discovery lexicons")
                owner[token_id] = lexicon.name
    return frozen


def lexicon_registry_sha256(lexicons: Sequence[OrderedLexicon]) -> str:
    frozen = validate_lexicon_registry(lexicons)
    payload = [
        {"name": lexicon.name, "items": [list(item) for item in lexicon.items]}
        for lexicon in frozen
    ]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def support_sha256(
    rows: torch.Tensor,
    lexicons: Sequence[OrderedLexicon],
    masks: Mapping[str, SuccessorMasks],
) -> str:
    """Bind exact rows, registry, and every ordered cell/pair-index realization."""

    frozen = validate_lexicon_registry(lexicons)
    if not torch.is_tensor(rows) or rows.device.type != "cpu" or rows.dtype != torch.long or (
        rows.ndim != 2 or rows.shape[1] != ROW_LENGTH or not rows.is_contiguous()
    ) or bool((rows < 0).any()) or bool((rows >= TOKEN_VOCAB).any()):
        raise ValueError("support rows must be contiguous CPU int64 [N,257] tokenizer IDs")
    if tuple(masks) != tuple(lexicon.name for lexicon in frozen):
        raise ValueError("mask registry must exactly follow lexicon order")
    digest = hashlib.sha256()
    digest.update(tensor_sha256(rows).encode("ascii"))
    digest.update(lexicon_registry_sha256(frozen).encode("ascii"))
    for lexicon in frozen:
        realized = masks[lexicon.name]
        realized.validate_partition()
        digest.update(lexicon.name.encode("utf-8"))
        digest.update(tensor_sha256(realized.eligible_target.contiguous()).encode("ascii"))
        for name, mask in realized.named_cells().items():
            digest.update(name.encode("ascii"))
            digest.update(tensor_sha256(mask.contiguous()).encode("ascii"))
        digest.update(tensor_sha256(realized.pair_index.contiguous()).encode("ascii"))
    return digest.hexdigest()


def validate_forward_closure(closure: campaign.ForwardClosure, arm: str, documents: int) -> None:
    """Require literal one-forward component calls for one arm/batch."""

    if type(closure) is not campaign.ForwardClosure or arm not in ARM_NAMES:
        raise ValueError("forward closure or arm is outside the discovery contract")
    if closure.circuit != SCHEMA or closure.arm != arm or closure.document_count != documents:
        raise ValueError("forward closure identity differs from the discovery batch")
    if not closure.closed or closure.attempted_outer_forwards != 1 or (
        closure.completed_outer_forwards != 1 or closure.outer_returns != 1
    ):
        raise ValueError("forward closure is not one successful outer forward")
    if len(closure.sites) != SITE_COUNT:
        raise ValueError("forward closure site topology changed")
    for ledger in closure.sites:
        replace = arm != "native" and ledger.site == TARGET_SITE
        expected_attention = (0, 1) if replace else (1, 0)
        if (ledger.native_attention_calls, ledger.replacement_attention_calls) != expected_attention:
            raise ValueError("attention call ledger differs from the frozen arm")
        if (ledger.native_mlp_calls, ledger.replacement_mlp_calls) != (1, 0):
            raise ValueError("MLP call ledger differs from the frozen arm")
    if arm != "native" and not closure.candidate_native_call_prohibition_passed:
        raise ValueError("replacement arm called native attention 8")


@dataclass(frozen=True)
class DocumentCellStatistics:
    count: torch.Tensor
    ce_sum: torch.Tensor
    native_kl_sum: torch.Tensor
    top1_change_sum: torch.Tensor
    successor_margin_sum: torch.Tensor

    def __post_init__(self) -> None:
        values = (
            self.count,
            self.ce_sum,
            self.native_kl_sum,
            self.top1_change_sum,
            self.successor_margin_sum,
        )
        if any(not torch.is_tensor(value) or value.ndim != 1 for value in values):
            raise ValueError("document sufficient statistics must be rank-1 tensors")
        if len({len(value) for value in values}) != 1 or self.count.dtype != torch.int64:
            raise ValueError("document sufficient-statistic currency is malformed")
        if any(value.dtype != torch.float64 for value in values[1:]) or any(
            not bool(torch.isfinite(value).all()) for value in values
        ) or bool((self.count < 0).any()):
            raise ValueError("document sufficient statistics must be finite and nonnegative-count")


def ordered_item_margin_logits(
    logits: torch.Tensor,
    masks: SuccessorMasks,
    lexicon: OrderedLexicon,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive source/target item-logit means from exact pair indices and arm logits."""

    if not torch.is_tensor(logits) or logits.ndim != 3 or not logits.dtype.is_floating_point or (
        not bool(torch.isfinite(logits).all())
    ):
        raise ValueError("logits must be one finite [batch,position,vocabulary] tensor")
    expected = logits.shape[:2]
    if masks.pair_index.shape != expected or masks.pair_index.device != logits.device:
        raise ValueError("successor pair indices must align with logits")
    if any(token >= logits.shape[-1] for item in lexicon.items for token in item):
        raise ValueError("lexicon token ID is outside logit support")
    source = torch.zeros(expected, dtype=logits.dtype, device=logits.device)
    target = torch.zeros_like(source)
    for index, (source_ids, target_ids) in enumerate(
        zip(lexicon.items[:-1], lexicon.items[1:])
    ):
        selected = masks.pair_index == index
        if not bool(selected.any()):
            continue
        source_mean = logits[..., list(source_ids)].mean(-1)
        target_mean = logits[..., list(target_ids)].mean(-1)
        source[selected] = source_mean[selected]
        target[selected] = target_mean[selected]
    if not torch.equal(masks.pair_index.ge(0), masks.eligible_target):
        raise ValueError("successor pair indices do not cover eligible targets")
    return source, target


def document_cell_statistics(
    native_logits: torch.Tensor,
    arm_logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    source_item_logits: torch.Tensor,
    target_item_logits: torch.Tensor,
) -> DocumentCellStatistics:
    """Return float64 per-document sums for a common fixed support.

    This lower-level primitive is shape-generic for known-answer tests.  A production
    adapter must separately enforce `[B,256,50304]`, rows `[B,257]`, and support
    positions 64--255 before calling it.
    """

    if not torch.is_tensor(native_logits) or native_logits.ndim != 3 or (
        not torch.is_tensor(arm_logits) or arm_logits.shape != native_logits.shape
    ) or native_logits.dtype != arm_logits.dtype or native_logits.device != arm_logits.device:
        raise ValueError("native and arm logits must have identical tensor currency")
    batch, positions, vocabulary = native_logits.shape
    expected = (batch, positions)
    if targets.shape != expected or targets.dtype != torch.long or targets.device != (
        native_logits.device
    ) or mask.shape != expected or mask.dtype != torch.bool or mask.device != (
        native_logits.device
    ):
        raise ValueError("targets and mask must align with logits")
    if source_item_logits.shape != expected or target_item_logits.shape != expected or (
        source_item_logits.device != native_logits.device
        or target_item_logits.device != native_logits.device
    ):
        raise ValueError("item-logit margins must align with logits")
    if not bool(torch.isfinite(native_logits).all()) or not bool(torch.isfinite(arm_logits).all()) or (
        not bool(torch.isfinite(source_item_logits).all())
        or not bool(torch.isfinite(target_item_logits).all())
    ) or bool((targets < 0).any()) or bool((targets >= vocabulary).any()):
        raise ValueError("scoring inputs must be finite and targets in vocabulary")

    native_logp = F.log_softmax(native_logits.float(), dim=-1)
    arm_logp = F.log_softmax(arm_logits.float(), dim=-1)
    ce = F.cross_entropy(
        arm_logits.float().transpose(1, 2), targets, reduction="none",
    ).double()
    kl = (native_logp.exp() * (native_logp - arm_logp)).sum(-1).double()
    changed = (native_logits.argmax(-1) != arm_logits.argmax(-1)).double()
    margin = (target_item_logits - source_item_logits).double()
    weight = mask.double()
    return DocumentCellStatistics(
        count=mask.sum(-1).cpu().to(torch.int64).contiguous(),
        ce_sum=(ce * weight).sum(-1).cpu().contiguous(),
        native_kl_sum=(kl * weight).sum(-1).cpu().contiguous(),
        top1_change_sum=(changed * weight).sum(-1).cpu().contiguous(),
        successor_margin_sum=(margin * weight).sum(-1).cpu().contiguous(),
    )


def validate_production_batch(
    rows: torch.Tensor,
    logits_by_arm: Mapping[str, torch.Tensor],
) -> None:
    """Fail closed on the exact unsliced production/common-denominator currency."""

    if not torch.is_tensor(rows) or rows.device.type != "cpu" or rows.dtype != torch.long or (
        rows.ndim != 2 or rows.shape[1] != ROW_LENGTH or not rows.is_contiguous()
    ) or bool((rows < 0).any()) or bool((rows >= TOKEN_VOCAB).any()):
        raise ValueError("production rows must be contiguous CPU int64 [B,257]")
    if tuple(logits_by_arm) != ARM_NAMES:
        raise ValueError("production logits must exactly follow the frozen arm order")
    expected = (rows.shape[0], ROW_LENGTH - 1, LOGIT_VOCAB)
    currency = None
    for arm, logits in logits_by_arm.items():
        if not torch.is_tensor(logits) or tuple(logits.shape) != expected or (
            logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all())
        ):
            raise ValueError(f"production logits are malformed for arm {arm}")
        observed = (logits.device, logits.dtype)
        if currency is None:
            currency = observed
        elif observed != currency:
            raise ValueError("all production arms must share device/dtype currency")


__all__ = (
    "ARM_NAMES",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CANDIDATES",
    "CURRENT_ONLY",
    "CandidateKind",
    "CandidateSpec",
    "DiscoveryAuthority",
    "DocumentCellStatistics",
    "FULL_REPLAY",
    "HEAD_DELETED",
    "MAX_ERROR_ORDER_INDEX",
    "NULL_BY_TRUE",
    "PROMOTIVE_ARMS",
    "ProgramBinding",
    "RANK_LADDER",
    "SCHEMA",
    "SOURCE_CLOSURE",
    "V1_ONLY",
    "arm_stored_parameters",
    "build_circuit_plan",
    "document_cell_statistics",
    "lexicon_registry_sha256",
    "ordered_item_margin_logits",
    "support_sha256",
    "tensor_sha256",
    "validate_forward_closure",
    "validate_lexicon_registry",
    "validate_production_batch",
)
