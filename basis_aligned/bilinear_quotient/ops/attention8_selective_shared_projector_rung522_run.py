#!/usr/bin/env python3
"""Fail-closed scientific runner for rung 522.

The runner learns rank-4 projectors inside attention8 from exact causal CE
responses.  It trains and freezes every registered frame before TEST can be
evaluated.  Rank is a matched capacity control; the scientific claim is
held-out circuit reuse, selectivity, stability, and removal specificity.

Prediction A: task-conditioned frames fitted to two circuit masks predict the
omitted circuit's signed causal response selectively in VALIDATION and TEST.
Prediction B: this selectivity exceeds recovery-only, Haar, and label-null
controls under the registered matched statistics.
Prediction C: the validation-selected all-three frame reuses on r.2.0.1 and
separates the quartet from all 28 other fingerprint circuits without generic
outside-union CE damage.
Prediction D: mean-centered removal with that frozen frame preserves the
quartet ordering and sign while sparing the 28 other fingerprint circuits.

Null: any hash, instrument, fit-health, liveness, leakage, uncertainty,
stability, response, selectivity, fingerprint, or removal gate fails closed at
the registered stage.  A small frame or response recovery alone is not a
circuit result.

Price: at most 103*200 = 20,600 projected forwards and 20,600 backwards;
9,422 inference-only forwards before removal; at most 2,000 removal forwards.
"""

# BQGATE: EXPERIMENT
# pred_a: leave-one-circuit-out selective signed responses pass every frozen cell
# pred_b: task-conditioned frames beat recovery-only Haar and label-null controls
# pred_c: one medoid frame reuses on the fourth circuit without generic damage
# pred_d: frozen mean-centered removal selectively changes the quartet

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence


REGISTERED_PREDICTIONS = {
    "pred_a": "leave-one-circuit-out selective signed responses pass every frozen cell",
    "pred_b": "task-conditioned frames beat recovery-only Haar and label-null controls",
    "pred_c": "one medoid frame reuses on the fourth circuit without generic damage",
    "pred_d": "frozen mean-centered removal selectively changes the quartet",
}

REPO = Path("/workspace/tensor_language")
ROOT = REPO / "basis_aligned/bilinear_quotient"
OPS = ROOT / "ops"
POLY = REPO / "basis_aligned/polynomial_causal"
DEFAULT_OUTPUT = ROOT / "attention8_selective_shared_projector_rung522_results.json"
DEFAULT_WORK = ROOT / "attention8_selective_shared_projector_rung522_work"

FROZEN_HASHES = {
    POLY / "ATTENTION8_SELECTIVE_SHARED_PROJECTOR_RUNG522_PREREGISTRATION.md":
        "27bc74c3e19ac310f0ed88f1527a1df44ff52d8990d980971415b32b503126f5",
    POLY / "ATTENTION8_SELECTIVE_SHARED_PROJECTOR_RUNG522_PREFLIGHT_ADDENDUM.md":
        "4f75c97dcdce1e652030cb933301c10540aa750d9a78cf5049c15aae48546ca6",
    OPS / "attention8_selective_shared_projector_rung522_math.py":
        "6cff6f7726dd8f76e786d64abf913fc31adbdfec101a97741a1aa3396f8431c2",
    OPS / "attention8_selective_shared_projector_rung522_scheduler.py":
        "d840318d5b675ce762f6c9a0d451c11550c6520b97ffbb762672ec703af5540f",
    OPS / "attention8_selective_shared_projector_rung522_protocol.py":
        "e05d409806aab33d3b0c13eb87ebd188be82762d53849505c7fc10f4df5e3c47",
    OPS / "attention8_selective_shared_projector_rung522_state_guard.py":
        "028a21352506236ae99c4181925494ed144993fd2186cb14d61fb8a16fe00d9c",
    OPS / "attention8_selective_shared_projector_rung522_archive.py":
        "02680d4912d48d4199b6aaa607d1c77120822217e8e56b40a61d80bddb33dec9",
    OPS / "attention8_selective_shared_projector_rung522_validation_gates.py":
        "54894c5c56883aa6062f21e485379f91d55139ad779d08ec54ab5432cd1c8452",
    OPS / "attention8_selective_shared_projector_rung522_sparse_fingerprint_null.py":
        "8315bc2ebfb367a97519ed71d448368267481d64620f098dd5226bee71da9acd",
    OPS / "attention8_shared_private_das_rung521.py":
        "d5ca962c16cd8f454adac79916a9cf3272b91debac0d27ebba2ce77804fb9ebd",
    OPS / "das_shared_private_lib.py":
        "edcf3d750e8fbdcb2ae479bcc6e68bd7ccc5078217b62cf981570656b6a773e4",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    ROOT / "attention8_selective_shared_projector_rung522_gpu_smoke.json":
        "6790dc40e481e8d46432d12b5637870ce80fdd4251538f61f47560b7e4a8fdd7",
}

D = 1152
TOKENS = 256
RANK = 4
CAPTURE_BATCH = 6
EVALUATION_ROWS = 6
UPDATES = 200
REAL_SEEDS = tuple(range(52200, 52205))
NULL_SEEDS = tuple(range(52300, 52316))
HAAR_SEEDS = tuple(range(52400, 52420))
FITTED_TAGS = ("r.2.0.2", "r.2.1.1", "r.2.2.1")
REUSE_TAG = "r.2.0.1"
QUARTET_TAGS = (REUSE_TAG,) + FITTED_TAGS
EXPECTED_FIT_COUNTS = {
    "real_leave_one_out": 15,
    "recovery_only": 15,
    "target_oracle": 20,
    "label_null": 48,
    "all_three": 5,
}
EXPECTED_TOTAL_FRAMES = 103
OPTIMIZATION_FORWARD_CEILING = 20600
OPTIMIZATION_BACKWARD_CEILING = 20600
INFERENCE_FORWARD_CEILING = 9422
REMOVAL_FORWARD_CEILING = 2000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _validate_frozen_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in FROZEN_HASHES.items():
        if not path.is_file():
            raise RuntimeError(f"frozen rung522 dependency is absent: {path}")
        actual = _sha256_file(path)
        observed[str(path.relative_to(REPO))] = actual
        if actual != expected:
            raise RuntimeError(
                f"frozen rung522 dependency changed: {path}; expected {expected}, got {actual}"
            )
    smoke = json.loads(
        (ROOT / "attention8_selective_shared_projector_rung522_gpu_smoke.json").read_text()
    )
    if not smoke.get("passed") or smoke.get("scientific_metrics_retained") is not False:
        raise RuntimeError("managed rung522 smoke is not a passing no-science receipt")
    if smoke.get("model_science_opened") is not False:
        raise RuntimeError("managed smoke unexpectedly claims to have opened science")
    return observed


_PREIMPORT_HASHES = _validate_frozen_hashes()
if os.environ.get("BQLIB_DRYRUN") == "1":
    print(
        "DRYRUN OK: rung522 full runner; 103 frames before TEST; "
        "20600 forwards + 20600 backwards; 9422 inference; TEST seal required",
        flush=True,
    )
    raise SystemExit(0)


import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

for _path in (OPS, POLY, ROOT, REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import attention8_shared_private_das_rung521 as stage_a  # noqa: E402
import attention8_selective_shared_projector_rung522_math as core  # noqa: E402
import attention8_selective_shared_projector_rung522_protocol as protocol  # noqa: E402
import attention8_selective_shared_projector_rung522_scheduler as scheduler  # noqa: E402
import attention8_selective_shared_projector_rung522_state_guard as state_guard  # noqa: E402
import attention8_selective_shared_projector_rung522_archive as archive  # noqa: E402
import attention8_selective_shared_projector_rung522_validation_gates as validation_gates  # noqa: E402
import attention8_selective_shared_projector_rung522_sparse_fingerprint_null as sparse_null  # noqa: E402
import bilin18_observed_model_facade as facade  # noqa: E402


@dataclass
class CallLedger:
    optimization_forwards: int = 0
    optimization_backwards: int = 0
    inference_forwards: int = 0
    removal_forwards: int = 0
    inference_by_bucket: dict[str, int] = field(default_factory=dict)

    def charge(self, kind: str, count: int = 1, *, bucket: str | None = None) -> None:
        if count < 0:
            raise ValueError("cannot charge a negative model-call count")
        if kind == "optimization_forward":
            self.optimization_forwards += count
            if self.optimization_forwards > OPTIMIZATION_FORWARD_CEILING:
                raise RuntimeError("rung522 optimization-forward ceiling exceeded")
        elif kind == "optimization_backward":
            self.optimization_backwards += count
            if self.optimization_backwards > OPTIMIZATION_BACKWARD_CEILING:
                raise RuntimeError("rung522 optimization-backward ceiling exceeded")
        elif kind == "inference_forward":
            if bucket not in state_guard.INFERENCE_LEDGER:
                raise RuntimeError(f"unregistered inference bucket {bucket!r}")
            self.inference_forwards += count
            self.inference_by_bucket[bucket] = self.inference_by_bucket.get(bucket, 0) + count
            if self.inference_forwards > INFERENCE_FORWARD_CEILING:
                raise RuntimeError("rung522 inference-forward ceiling exceeded")
        elif kind == "removal_forward":
            self.removal_forwards += count
            if self.removal_forwards > REMOVAL_FORWARD_CEILING:
                raise RuntimeError("rung522 removal-forward ceiling exceeded")
        else:
            raise ValueError(f"unknown ledger kind {kind!r}")

    def snapshot(self) -> dict[str, object]:
        return {
            "optimization_forwards": self.optimization_forwards,
            "optimization_backwards": self.optimization_backwards,
            "inference_forwards": self.inference_forwards,
            "removal_forwards": self.removal_forwards,
            "inference_by_bucket": dict(sorted(self.inference_by_bucket.items())),
        }

    def assert_pretest_registered_price(self) -> None:
        expected = {
            "native_capture": 131,
            "native_replay": 131,
            "self_donor": 2,
            "fit_d0_full_attention8": 95,
            "fit_health": 206,
            "full_attention8_comparator": 36,
            "prediction_a": 2_988,
            "recovery_only": 540,
            "haar": 720,
            "all_three_selection_and_test": 180,
        }
        if self.optimization_forwards != OPTIMIZATION_FORWARD_CEILING or (
            self.optimization_backwards != OPTIMIZATION_BACKWARD_CEILING
        ):
            raise RuntimeError("pre-TEST fit ledger is not exactly 20,600/20,600")
        if self.inference_by_bucket != expected:
            raise RuntimeError(
                f"pre-TEST inference bucket ledger changed: {self.inference_by_bucket} != {expected}"
            )
        if self.inference_forwards != sum(expected.values()) or self.removal_forwards != 0:
            raise RuntimeError("pre-TEST inference/removal totals changed")

    def assert_final_registered_price(self, *, expected_removal_forwards: int = 36) -> None:
        if self.optimization_forwards != OPTIMIZATION_FORWARD_CEILING:
            raise RuntimeError("final optimization-forward count is not exactly 20,600")
        if self.optimization_backwards != OPTIMIZATION_BACKWARD_CEILING:
            raise RuntimeError("final optimization-backward count is not exactly 20,600")
        expected = dict(state_guard.INFERENCE_LEDGER)
        if self.inference_by_bucket != expected:
            raise RuntimeError(
                f"final inference bucket ledger changed: {self.inference_by_bucket} != {expected}"
            )
        if self.inference_forwards != INFERENCE_FORWARD_CEILING:
            raise RuntimeError("final inference count is not exactly 9,422")
        if expected_removal_forwards not in (0, 36):
            raise ValueError("registered removal count must be zero or the complete 36-call sweep")
        if self.removal_forwards != expected_removal_forwards:
            raise RuntimeError(
                "final removal count differs from the registered conditional sweep: "
                f"{self.removal_forwards} != {expected_removal_forwards}"
            )


def _atomic_json(path: Path, value: Mapping[str, object], *, refuse_overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"refusing to overwrite scientific receipt: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as sink:
        json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
        sink.write("\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)


def _execute(
    model,
    tokens: torch.Tensor,
    *,
    edit: Callable[[torch.Tensor], torch.Tensor] | None = None,
    capture: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, object]]:
    dispatch_calls = 0
    module_calls = 0
    captured = None
    edit_rms = None

    def count_module(_module, _inputs, _output):
        nonlocal module_calls
        module_calls += 1

    def attention(event):
        nonlocal dispatch_calls, captured, edit_rms
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 8:
            dispatch_calls += 1
            if capture:
                if captured is not None:
                    raise RuntimeError("attention8 capture occurred more than once")
                captured = write.detach().float().cpu().clone()
            if edit is not None:
                changed = edit(write)
                if changed.shape != write.shape or changed.dtype != write.dtype or changed.device != write.device:
                    raise RuntimeError("attention8 intervention changed tensor metadata")
                difference = (changed.detach().float() - write.detach().float()).reshape(write.shape[0], -1)
                edit_rms = difference.square().mean(1).sqrt().cpu()
                write = changed
        return write, first_value

    def mlp(event):
        return event.block.mlp(event.state)

    handle = model.transformer.h[8].attn.register_forward_hook(count_module)
    try:
        logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=False)
    finally:
        handle.remove()
    if dispatch_calls != 1 or module_calls != 1:
        raise RuntimeError(
            f"attention8 execution count changed: dispatch={dispatch_calls}, module={module_calls}"
        )
    return logits, captured, {
        "attention8_dispatch_calls": dispatch_calls,
        "attention8_module_calls": module_calls,
        "per_sequence_edit_rms": edit_rms,
    }


def _per_token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).view_as(targets)


def _row_pool(mask: torch.Tensor, row_mask: torch.Tensor) -> tuple[int, ...]:
    eligible = mask.view(1000, TOKENS).any(1) & row_mask
    rows = tuple(eligible.nonzero().flatten().tolist())
    if not rows:
        raise RuntimeError("registered member/control role has no eligible row")
    return rows


@dataclass(frozen=True)
class TargetPairs:
    """Ordered matched member/control positions for one target and data cell."""

    target: str
    members: torch.Tensor
    controls: torch.Tensor
    tiers: torch.Tensor

    def __post_init__(self) -> None:
        if any(value.device.type != "cpu" or value.ndim != 1 for value in (
            self.members, self.controls, self.tiers
        )):
            raise ValueError("target-pair arrays must be one-dimensional CPU tensors")
        if not (self.members.numel() == self.controls.numel() == self.tiers.numel() > 0):
            raise ValueError("target-pair arrays must have one equal positive length")
        if self.members.unique().numel() != self.members.numel():
            raise ValueError("member positions must be unique")
        if self.controls.unique().numel() != self.controls.numel():
            raise ValueError("control positions must be unique")

    def mask(self, kind: str) -> torch.Tensor:
        indices = self.members if kind == "member" else self.controls
        result = torch.zeros(1000 * TOKENS, dtype=torch.bool)
        result[indices] = True
        return result

    def row_pool(self, kind: str, row_mask: torch.Tensor) -> tuple[int, ...]:
        return _row_pool(self.mask(kind), row_mask)

    def identity(self) -> dict[str, object]:
        return {
            "target": self.target,
            "count": int(self.members.numel()),
            "members_sha256": stage_a._tensor_sha256(self.members),
            "controls_sha256": stage_a._tensor_sha256(self.controls),
            "tiers_sha256": stage_a._tensor_sha256(self.tiers),
        }


def _pairs_from_match(target: str, match: Mapping[str, torch.Tensor]) -> TargetPairs:
    return TargetPairs(
        target,
        match["members"].long().contiguous(),
        match["controls"].long().contiguous(),
        match["tiers"].long().contiguous(),
    )


def _combined_fit_pairs(design: dict, target: str) -> TargetPairs:
    first = design["cells"]["fit_half0"]["exclusive"][target]
    second = design["cells"]["fit_half1"]["exclusive"][target]
    return TargetPairs(
        target,
        torch.cat((first["members"], second["members"])).long().contiguous(),
        torch.cat((first["controls"], second["controls"])).long().contiguous(),
        torch.cat((first["tiers"], second["tiers"])).long().contiguous(),
    )


@dataclass(frozen=True)
class LabelNullDesign:
    seed: int
    pairs: Mapping[str, TargetPairs]
    permutation_sha256: str
    moved_nonzero_count: int
    maximum_possible_moved_nonzero_count: int
    identity_sha256: str


def _parent_masks(data: dict) -> dict[str, torch.Tensor]:
    result = {}
    for target in QUARTET_TAGS:
        mask = torch.zeros(1000 * TOKENS, dtype=torch.bool)
        mask[data["leaves"][target]["slice"].long()] = True
        result[target] = mask
    return result


def build_label_null_designs(
    data: dict,
    descriptors: Mapping[str, torch.Tensor],
) -> dict[int, LabelNullDesign]:
    """Construct all 16 maximal-movement FIT-only null designs.

    VALIDATION and TEST arrays are neither accepted nor returned. Matching is
    performed separately inside FIT halves so controls remain in the member's
    frozen data cell.
    """
    fit_flat = stage_a._flat_row_mask(data["row_masks"]["fit"])
    fit_positions = fit_flat.nonzero().flatten()
    membership_codes = torch.zeros(fit_positions.numel(), dtype=torch.int64)
    parent_codes = torch.zeros_like(membership_codes)
    parents = _parent_masks(data)
    for bit, target in enumerate(QUARTET_TAGS):
        membership_codes |= data["full_masks"][target][fit_positions].long() << bit
        parent_codes |= parents[target][fit_positions].long() << bit

    results: dict[int, LabelNullDesign] = {}
    for seed in NULL_SEEDS:
        permutation = protocol.permute_four_bit_memberships(
            membership_codes,
            token_classes=descriptors["token_class"][fit_positions],
            token_positions=fit_positions % TOKENS,
            ce_deciles=descriptors["ce_decile"][fit_positions],
            parent_slice_codes=parent_codes,
            seed=seed,
            position_ids=fit_positions,
        )
        if not permutation.attains_maximum_possible_movement:
            raise RuntimeError(f"label-null seed {seed} missed maximum feasible movement")
        assigned_codes = torch.tensor(permutation.permuted_codes, dtype=torch.int64)
        permuted_global_codes = torch.zeros(1000 * TOKENS, dtype=torch.int64)
        permuted_global_codes[fit_positions] = assigned_codes
        permuted_union = permuted_global_codes != 0
        by_target: dict[str, TargetPairs] = {}
        for bit, target in enumerate(QUARTET_TAGS):
            # Exclusive membership is exactly the singleton four-bit code.
            exclusive = permuted_global_codes == (1 << bit)
            matches = []
            for cell, low, high in (("fit_half0", 0, 3), ("fit_half1", 3, 6)):
                half = stage_a._flat_row_mask(
                    (data["folds"] >= low) & (data["folds"] < high)
                )
                positive = exclusive & half
                pool = parents[target] & half & ~permuted_union
                matches.append(
                    stage_a._construct_controls(
                        positive,
                        pool,
                        dict(descriptors),
                        tag=target,
                        cell=f"rung522:null:{seed}:{cell}:exclusive",
                    )
                )
            by_target[target] = TargetPairs(
                target,
                torch.cat(tuple(match["members"] for match in matches)).long(),
                torch.cat(tuple(match["controls"] for match in matches)).long(),
                torch.cat(tuple(match["tiers"] for match in matches)).long(),
            )
        identity = {
            "seed": seed,
            "permutation_sha256": permutation.sha256,
            "moved_nonzero_count": permutation.moved_nonzero_count,
            "maximum_possible_moved_nonzero_count": (
                permutation.maximum_possible_moved_nonzero_count
            ),
            "pairs": {target: by_target[target].identity() for target in QUARTET_TAGS},
        }
        identity_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        results[seed] = LabelNullDesign(
            seed,
            by_target,
            permutation.sha256,
            permutation.moved_nonzero_count,
            permutation.maximum_possible_moved_nonzero_count,
            identity_sha256,
        )
    if tuple(sorted(results)) != NULL_SEEDS:
        raise RuntimeError("label-null design census changed")
    return results


@dataclass(frozen=True)
class SwapArm:
    ensemble: str
    map_index: int
    direction: str
    recipient_row: int

    @property
    def cell(self) -> str:
        return f"{self.ensemble}:{self.direction}"


def _swap_arm_plan(recipient_rows: Sequence[int]) -> tuple[SwapArm, ...]:
    arms = []
    for ensemble, offset in (("D0", 0), ("D1", 4)):
        for direction in ("forward", "reverse"):
            for local_map in range(4):
                for row in recipient_rows:
                    arms.append(SwapArm(ensemble, offset + local_map, direction, int(row)))
    expected = 16 * len(recipient_rows)
    if len(arms) != expected or len(set(arms)) != expected:
        raise RuntimeError("swap arm plan is incomplete or duplicated")
    return tuple(arms)


@dataclass(frozen=True)
class SwapEvaluation:
    split: str
    kind: str
    map_responses: Mapping[str, torch.Tensor]
    cell_responses: Mapping[str, torch.Tensor]
    forward_calls: int
    minimum_edit_rms: float
    response_sha256: str


@dataclass(frozen=True)
class RemovalEvaluation:
    split: str
    response: torch.Tensor
    forward_calls: int
    minimum_edit_rms: float
    response_sha256: str


def _donor_rows(donor_map: torch.Tensor, selected_rows: torch.Tensor) -> torch.Tensor:
    donor = donor_map[selected_rows * TOKENS]
    if bool((donor < 0).any()) or bool((donor % TOKENS != 0).any()):
        raise RuntimeError("row-coherent donor map is invalid")
    return donor // TOKENS


class Rung522Instrument:
    """Exact model boundary, native caches, and charged forward execution."""

    def __init__(
        self,
        model,
        data: dict,
        design: dict,
        ledger: CallLedger,
        state: state_guard.ProtocolState,
    ) -> None:
        self.model = model
        self.data = data
        self.design = design
        self.ledger = ledger
        self.state = state
        self.device = next(model.parameters()).device
        self.captures: dict[str, torch.Tensor] = {}
        self.native_ce: dict[str, torch.Tensor] = {}
        self.split_rows: dict[str, torch.Tensor] = {}
        self.row_to_local: dict[str, torch.Tensor] = {}
        self.full_fit_d0: torch.Tensor | None = None

    @torch.no_grad()
    def _capture_split(self, split: str) -> dict[str, object]:
        self.state.authorize_split_access(split.upper())
        if split in self.captures:
            raise RuntimeError(f"{split} native state was already captured")
        rows = self.data["row_masks"][split].nonzero().flatten()
        row_to_local = torch.full((1000,), -1, dtype=torch.int64)
        row_to_local[rows] = torch.arange(rows.numel())
        writes = torch.empty((rows.numel(), TOKENS, D), dtype=torch.float32)
        native_ce = torch.empty((rows.numel(), TOKENS), dtype=torch.float32)
        replay_exact = True
        self_donor_exact = False
        for start in range(0, rows.numel(), CAPTURE_BATCH):
            chosen = rows[start : start + CAPTURE_BATCH]
            batch = self.data["rows"][chosen]
            tokens = batch[:, :TOKENS].to(self.device)
            targets = batch[:, 1 : TOKENS + 1].to(self.device)
            logits, captured, _ = _execute(self.model, tokens, capture=True)
            self.ledger.charge("inference_forward", bucket="native_capture")
            self.state.record_inference_events(1)
            if captured is None:
                raise RuntimeError("attention8 native capture is absent")
            # This intentionally bypasses the dispatch facade used above.  A
            # second call through the same path would only test repeatability;
            # the literal block loop can catch a shared dispatch error.
            replay, replay_attention8_calls = stage_a._direct_logits(self.model, tokens)
            self.ledger.charge("inference_forward", bucket="native_replay")
            self.state.record_inference_events(1)
            if replay_attention8_calls != 1:
                raise RuntimeError(
                    "independent native replay did not execute attention8 exactly once"
                )
            replay_exact &= bool(torch.equal(logits, replay))
            if start == 0:
                native_write = captured.to(self.device)
                self_logits, _, self_diag = _execute(
                    self.model,
                    tokens,
                    edit=lambda _write, value=native_write: value,
                )
                self.ledger.charge("inference_forward", bucket="self_donor")
                self.state.record_inference_events(1)
                rms = self_diag["per_sequence_edit_rms"]
                self_donor_exact = bool(
                    torch.equal(logits, self_logits)
                    and rms is not None
                    and not bool((rms != 0).any())
                )
                del native_write, self_logits
            stop = start + chosen.numel()
            writes[start:stop] = captured
            native_ce[start:stop] = _per_token_ce(logits, targets).cpu()
            del logits, replay, captured
        if not replay_exact:
            raise RuntimeError(f"{split} independent native replay changed logits")
        if not self_donor_exact:
            raise RuntimeError(f"{split} self-donor activation/logits are not an exact no-op")
        self.captures[split] = writes
        self.native_ce[split] = native_ce
        self.split_rows[split] = rows
        self.row_to_local[split] = row_to_local
        return {
            "rows": int(rows.numel()),
            "native_replay_exact": replay_exact,
            "self_donor_exact": self_donor_exact,
        }

    def capture_pretest_splits(self) -> dict[str, object]:
        return {
            split: self._capture_split(split)
            for split in ("fit", "validation")
        }

    def capture_test_after_open(self) -> dict[str, object]:
        self.state.authorize_split_access("TEST")
        return self._capture_split("test")

    def _writes_for_rows(self, split: str, rows: torch.Tensor) -> torch.Tensor:
        local = self.row_to_local[split][rows]
        if bool((local < 0).any()):
            raise RuntimeError(f"requested native write outside {split}")
        return self.captures[split][local]

    def donor_writes(self, split: str, rows: torch.Tensor, donor_map: torch.Tensor) -> torch.Tensor:
        return self._writes_for_rows(split, _donor_rows(donor_map, rows))

    @torch.no_grad()
    def precompute_full_fit_d0(self) -> dict[str, object]:
        """Save the exact per-map whole-attention8 FIT response in 95 calls."""
        if "fit" not in self.captures:
            raise RuntimeError("FIT native state must be captured first")
        if self.full_fit_d0 is not None:
            raise RuntimeError("whole-attention8 FIT responses were already computed")
        rows = self.split_rows["fit"]
        maps = tuple(self.design["donors"]["fit"]["maps"][:4])
        if len(maps) != 4:
            raise RuntimeError("FIT D0 no longer contains four donor maps")
        result = torch.empty((4, rows.numel(), TOKENS), dtype=torch.float32)
        minimum_edit_rms = math.inf
        calls = 0
        for start in range(0, rows.numel(), CAPTURE_BATCH):
            chosen = rows[start : start + CAPTURE_BATCH]
            count = chosen.numel()
            token_blocks = []
            donor_blocks = []
            arm_ids = []
            for map_index, donor_map in enumerate(maps):
                token_blocks.append(self.data["rows"][chosen, :TOKENS])
                donor_blocks.append(self.donor_writes("fit", chosen, donor_map))
                arm_ids.extend((map_index, int(row)) for row in chosen.tolist())
            tokens = torch.cat(token_blocks).to(self.device)
            targets = torch.cat(
                [self.data["rows"][chosen, 1 : TOKENS + 1] for _ in maps]
            ).to(self.device)
            donor = torch.cat(donor_blocks).to(self.device)
            logits, _, diagnostics = _execute(
                self.model, tokens, edit=lambda _write, value=donor: value
            )
            self.ledger.charge("inference_forward", bucket="fit_d0_full_attention8")
            self.state.record_inference_events(1)
            calls += 1
            rms = diagnostics["per_sequence_edit_rms"]
            if rms is None or bool((rms <= 0).any()):
                raise RuntimeError("whole-attention8 FIT target contains a dead donor arm")
            minimum_edit_rms = min(minimum_edit_rms, float(rms.min()))
            local = self.row_to_local["fit"][chosen]
            native = self.native_ce["fit"][local].to(self.device)
            deltas = (
                _per_token_ce(logits, targets)
                - native.unsqueeze(0).expand(4, -1, -1).reshape(4 * count, TOKENS)
            ).view(4, count, TOKENS)
            # arm_ids makes the map-major physical order explicit; never infer
            # an ensemble/direction axis from an undocumented reshape.
            for physical, (map_index, row) in enumerate(arm_ids):
                expected_physical = map_index * count + (row == chosen).nonzero().item()
                if physical != expected_physical:
                    raise RuntimeError("FIT arm-ID reconstruction changed")
            result[:, start : start + count] = deltas.cpu()
            del logits, tokens, targets, donor, deltas
        if calls != math.ceil(rows.numel() / CAPTURE_BATCH):
            raise RuntimeError("whole-attention8 FIT call count changed")
        self.full_fit_d0 = result
        return {
            "forward_calls": calls,
            "minimum_edit_rms": minimum_edit_rms,
            "response_sha256": stage_a._tensor_sha256(result),
        }

    def projected_delta(
        self,
        split: str,
        rows: torch.Tensor,
        donor_map: torch.Tensor,
        frame: torch.Tensor,
        *,
        optimization: bool,
        inference_bucket: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.data["rows"][rows, :TOKENS].to(self.device)
        targets = self.data["rows"][rows, 1 : TOKENS + 1].to(self.device)
        donor = self.donor_writes(split, rows, donor_map).to(self.device)

        def edit(write: torch.Tensor) -> torch.Tensor:
            return core.daslib.projection_interchange(write, donor, frame, validate=False)

        logits, _, diagnostics = _execute(self.model, tokens, edit=edit)
        if optimization:
            if inference_bucket is not None:
                raise RuntimeError("optimization calls cannot enter an inference bucket")
            self.ledger.charge("optimization_forward")
        else:
            self.ledger.charge("inference_forward", bucket=inference_bucket)
        if optimization:
            self.state.record_optimization_events(1, 0)
        else:
            self.state.record_inference_events(1)
        rms = diagnostics["per_sequence_edit_rms"]
        if rms is None or bool((rms <= 0).any()):
            raise RuntimeError("projected intervention contains a dead sequence")
        local = self.row_to_local[split][rows]
        delta = _per_token_ce(logits, targets) - self.native_ce[split][local].to(self.device)
        return delta, rms

    @torch.no_grad()
    def evaluate_swap(
        self,
        split: str,
        *,
        frame: torch.Tensor | None,
        inference_bucket: str,
    ) -> SwapEvaluation:
        """Evaluate all D0/D1 x forward/reverse cells with explicit arm IDs."""
        self.state.authorize_split_access(split.upper())
        if split not in self.captures:
            raise RuntimeError(f"{split} native state has not been captured")
        rows = self.split_rows[split]
        donor_design = self.design["donors"][split]
        maps = tuple(donor_design["maps"])
        inverse_maps = tuple(donor_design["inverse_maps"])
        if len(maps) != 8 or len(inverse_maps) != 8:
            raise RuntimeError("evaluation requires eight forward and inverse donor maps")
        map_responses = {
            f"{ensemble}:{direction}": torch.full(
                (4, rows.numel(), TOKENS), float("nan"), dtype=torch.float32
            )
            for ensemble in ("D0", "D1")
            for direction in ("forward", "reverse")
        }
        frame_device = None if frame is None else frame.to(self.device, dtype=torch.float32)
        calls = 0
        minimum_edit_rms = math.inf
        for start in range(0, rows.numel(), EVALUATION_ROWS):
            chosen = rows[start : start + EVALUATION_ROWS]
            arms = _swap_arm_plan(chosen.tolist())
            token_blocks = []
            target_blocks = []
            donor_blocks = []
            for arm in arms:
                row_tensor = torch.tensor([arm.recipient_row], dtype=torch.int64)
                token_blocks.append(self.data["rows"][row_tensor, :TOKENS])
                target_blocks.append(self.data["rows"][row_tensor, 1 : TOKENS + 1])
                donor_map = (
                    maps[arm.map_index]
                    if arm.direction == "forward"
                    else inverse_maps[arm.map_index]
                )
                donor_blocks.append(self.donor_writes(split, row_tensor, donor_map))
            tokens = torch.cat(token_blocks).to(self.device)
            targets = torch.cat(target_blocks).to(self.device)
            donors = torch.cat(donor_blocks).to(self.device)

            def edit(write: torch.Tensor) -> torch.Tensor:
                if frame_device is None:
                    return donors
                return core.daslib.projection_interchange(
                    write, donors, frame_device, validate=False
                )

            logits, _, diagnostics = _execute(self.model, tokens, edit=edit)
            self.ledger.charge("inference_forward", bucket=inference_bucket)
            self.state.record_inference_events(1)
            calls += 1
            rms = diagnostics["per_sequence_edit_rms"]
            if rms is None or bool((rms <= 0).any()):
                raise RuntimeError("evaluation contains a dead donor arm")
            minimum_edit_rms = min(minimum_edit_rms, float(rms.min()))
            ce = _per_token_ce(logits, targets).cpu()
            for physical_index, arm in enumerate(arms):
                local_map = arm.map_index % 4
                local_row = self.row_to_local[split][arm.recipient_row]
                if int(local_row) < 0:
                    raise RuntimeError("evaluation arm references a row outside its split")
                delta = ce[physical_index] - self.native_ce[split][local_row]
                map_responses[arm.cell][local_map, local_row] = delta
            del logits, tokens, targets, donors, ce
        expected_calls = math.ceil(rows.numel() / EVALUATION_ROWS)
        if calls != expected_calls:
            raise RuntimeError("evaluation forward-call count changed")
        incomplete = [
            name for name, values in map_responses.items()
            if not bool(torch.isfinite(values).all())
        ]
        if incomplete:
            raise RuntimeError(f"evaluation left unfilled/nonfinite arm cells: {incomplete}")
        cells = {name: values.mean(0) for name, values in map_responses.items()}
        digest = hashlib.sha256()
        for name in sorted(map_responses):
            digest.update(name.encode())
            digest.update(map_responses[name].contiguous().numpy().tobytes())
        return SwapEvaluation(
            split=split,
            kind="whole_attention8" if frame is None else "rank4_projector",
            map_responses=map_responses,
            cell_responses=cells,
            forward_calls=calls,
            minimum_edit_rms=minimum_edit_rms,
            response_sha256=digest.hexdigest(),
        )

    def fit_projection_mean(self, frame: torch.Tensor) -> torch.Tensor:
        """Mean of yQ over every native FIT row and predicted-token position."""
        if "fit" not in self.captures:
            raise RuntimeError("FIT native state must be captured before computing mu_Q")
        frame_cpu = frame.detach().cpu().float()
        if tuple(frame_cpu.shape) != (D, RANK):
            raise ValueError("removal frame shape changed")
        mean = (self.captures["fit"] @ frame_cpu).double().mean((0, 1)).float()
        if tuple(mean.shape) != (RANK,) or not bool(torch.isfinite(mean).all()):
            raise RuntimeError("FIT projection mean is absent or non-finite")
        return mean.contiguous()

    @torch.no_grad()
    def evaluate_removal(
        self,
        frame: torch.Tensor,
        fit_projection_mean: torch.Tensor,
    ) -> RemovalEvaluation:
        """Execute the frozen mean-centered action on TEST exactly once."""
        self.state.authorize_split_access("TEST")
        if "test" not in self.captures:
            raise RuntimeError("TEST native state must be captured inside the open sweep")
        frame_device = frame.to(self.device, dtype=torch.float32)
        mean_device = fit_projection_mean.to(self.device, dtype=torch.float32)
        if tuple(frame_device.shape) != (D, RANK) or tuple(mean_device.shape) != (RANK,):
            raise ValueError("removal frame or FIT mean shape changed")
        rows = self.split_rows["test"]
        response = torch.empty((rows.numel(), TOKENS), dtype=torch.float32)
        calls = 0
        minimum_edit_rms = math.inf
        for start in range(0, rows.numel(), EVALUATION_ROWS):
            chosen = rows[start : start + EVALUATION_ROWS]
            tokens = self.data["rows"][chosen, :TOKENS].to(self.device)
            targets = self.data["rows"][chosen, 1 : TOKENS + 1].to(self.device)

            def edit(write: torch.Tensor) -> torch.Tensor:
                coordinates = write @ frame_device
                return write - (coordinates - mean_device) @ frame_device.mT

            logits, _, diagnostics = _execute(self.model, tokens, edit=edit)
            self.ledger.charge("removal_forward")
            self.state.record_inference_events(1, removal=True)
            calls += 1
            rms = diagnostics["per_sequence_edit_rms"]
            if rms is None or bool((rms <= 0).any()):
                raise RuntimeError("mean-centered removal contains a dead sequence")
            minimum_edit_rms = min(minimum_edit_rms, float(rms.min()))
            local = self.row_to_local["test"][chosen]
            stop = start + chosen.numel()
            response[start:stop] = (
                _per_token_ce(logits, targets).cpu() - self.native_ce["test"][local]
            )
            del logits, tokens, targets
        expected = math.ceil(rows.numel() / EVALUATION_ROWS)
        if calls != expected:
            raise RuntimeError("removal forward-call count changed")
        return RemovalEvaluation(
            split="test",
            response=response,
            forward_calls=calls,
            minimum_edit_rms=minimum_edit_rms,
            response_sha256=stage_a._tensor_sha256(response),
        )


def _make_balanced_scheduler(
    spec: state_guard.FrameSpec,
    pairs: Mapping[str, TargetPairs],
    row_mask: torch.Tensor,
) -> scheduler.BalancedRowScheduler:
    member_rows = {
        target: pairs[target].row_pool("member", row_mask)
        for target in spec.training_targets
    }
    control_rows = {
        target: pairs[target].row_pool("control", row_mask)
        for target in spec.training_targets
    }
    if spec.family == "target_oracle":
        return scheduler.single_target_oracle_scheduler(
            spec.training_targets[0], member_rows, control_rows, seed=spec.seed
        )
    if len(spec.training_targets) == 2:
        return scheduler.two_target_scheduler(
            spec.training_targets, member_rows, control_rows, seed=spec.seed
        )
    if len(spec.training_targets) == 3:
        return scheduler.all_three_scheduler(
            spec.training_targets, member_rows, control_rows, seed=spec.seed
        )
    raise RuntimeError(f"unsupported registered training layout for {spec.frame_id}")


class ProjectedResponseCallback:
    """One exact scheduled model call yielding one response per target."""

    def __init__(
        self,
        instrument: Rung522Instrument,
        spec: state_guard.FrameSpec,
        *,
        split: str,
        pairs: Mapping[str, TargetPairs],
        balanced: scheduler.BalancedRowScheduler,
        full_by_map: torch.Tensor,
        optimization: bool,
        fixed_health_batch: bool = False,
    ) -> None:
        self.instrument = instrument
        self.spec = spec
        self.split = split
        self.pairs = dict(pairs)
        self.balanced = balanced
        self.full_by_map = full_by_map
        self.optimization = optimization
        self.fixed_health_batch = fixed_health_batch
        self._position_masks = {
            (target, kind): self.pairs[target].mask(kind).view(1000, TOKENS)
            for target in spec.training_targets
            for kind in ("member", "control")
        }
        expected_maps = 4 if split == "fit" else 1
        if tuple(full_by_map.shape[:1]) != (expected_maps,):
            raise ValueError("full response map axis does not match callback split")
        if tuple(full_by_map.shape[1:]) != (
            instrument.split_rows[split].numel(), TOKENS
        ):
            raise ValueError("full response row/position axes changed")

    def __call__(self, frame: torch.Tensor, step: int) -> Mapping[str, core.TargetResponse]:
        # fit_projector uses -1 for both health calls. The registered health
        # callback must always use its independently hashed batch zero.
        update = 0 if self.fixed_health_batch else step
        if update < 0:
            raise RuntimeError("negative update reached a non-health scheduler")
        scheduled = self.balanced.batch(update)
        map_index = 0 if self.fixed_health_batch else scheduled.donor_map_index
        donor_map = self.instrument.design["donors"][self.split]["maps"][map_index]
        rows = torch.tensor(
            [role.row_index for role in scheduled.roles], dtype=torch.int64
        )
        delta, _ = self.instrument.projected_delta(
            self.split,
            rows,
            donor_map,
            frame,
            optimization=self.optimization,
            inference_bucket=None if self.optimization else "fit_health",
        )
        responses: dict[str, core.TargetResponse] = {}
        for target in self.spec.training_targets:
            projected_members = []
            full_members = []
            projected_controls = []
            for physical_index, role in enumerate(scheduled.roles):
                if role.target != target:
                    continue
                mask = self._position_masks[(target, role.kind)][role.row_index]
                if not bool(mask.any()):
                    raise RuntimeError(
                        f"scheduled {role.kind} row has no eligible positions for {target}"
                    )
                if role.kind == "member":
                    projected_members.append(delta[physical_index, mask])
                    local = self.instrument.row_to_local[self.split][role.row_index]
                    if int(local) < 0:
                        raise RuntimeError("scheduled member row is outside callback split")
                    full_members.append(
                        self.full_by_map[map_index, local, mask].to(frame.device)
                    )
                else:
                    projected_controls.append(delta[physical_index, mask])
            if not projected_members or not projected_controls:
                raise RuntimeError(f"balanced callback lost a member/control role for {target}")
            responses[target] = core.TargetResponse(
                full_member=torch.cat(full_members),
                projected_member=torch.cat(projected_members),
                projected_control=torch.cat(projected_controls),
            )
        if tuple(sorted(responses)) != tuple(sorted(self.spec.training_targets)):
            raise RuntimeError("response callback changed registered target identities")
        return responses


@dataclass(frozen=True)
class FittedFrameRecord:
    frame_id: str
    frame_sha256: str
    fit_scheduler_sha256: str
    validation_scheduler_sha256: str
    healthy: bool
    health_failures: tuple[str, ...]
    initial_validation_objective: float
    final_validation_objective: float
    initial_window_mean: float
    final_window_mean: float
    orthonormality_error: float
    projector_distance_from_initialization: float
    maximizing_target_counts: Mapping[str, int]
    loss_history_sha256: str
    fit_selected_batches_sha256: str
    validation_health_batch_sha256: str
    fit_record_sha256: str
    health_record_sha256: str
    fit_scheduler_payload: Mapping[str, object]
    validation_scheduler_payload: Mapping[str, object]
    fit_record_payload: Mapping[str, object]
    health_record_payload: Mapping[str, object]


def _scheduler_payload(
    balanced: scheduler.BalancedRowScheduler,
) -> dict[str, object]:
    """Materialize the exact scheduler object whose bytes define its fingerprint."""
    payload: dict[str, object] = {
        "namespace": scheduler.SCHEDULER_NAMESPACE,
        "mode": balanced.mode,
        "seed": balanced.seed,
        "donor_map_rule": "update_mod_4",
        "roles": [
            {
                "name": role.name,
                "target": role.target,
                "kind": role.kind,
                "replica": role.replica,
                "permutation": list(role.permutation),
            }
            for role in balanced.roles
        ],
    }
    if _sha256_json(payload) != balanced.fingerprint:
        raise RuntimeError("materialized scheduler differs from its frozen fingerprint")
    return payload


def _batch_zero_rows(payload: Mapping[str, object]) -> dict[str, int]:
    roles = payload.get("roles")
    if not isinstance(roles, list):
        raise RuntimeError("scheduler payload has no role list")
    result: dict[str, int] = {}
    for role in roles:
        if not isinstance(role, Mapping):
            raise RuntimeError("scheduler role payload is malformed")
        permutation = role.get("permutation")
        if not isinstance(permutation, list) or not permutation:
            raise RuntimeError("scheduler role has no permutation")
        result[str(role["name"])] = int(permutation[0])
    return result


def _scheduled_batches_sha256(
    balanced: scheduler.BalancedRowScheduler, updates: Sequence[int]
) -> str:
    return _sha256_json([
        {
            "update": batch.update,
            "donor_map_index": batch.donor_map_index,
            "roles": [asdict(role) for role in batch.roles],
        }
        for batch in (balanced.batch(update) for update in updates)
    ])


def fit_one_registered_frame(
    instrument: Rung522Instrument,
    spec: state_guard.FrameSpec,
    state: state_guard.ProtocolState,
    training_callback: ProjectedResponseCallback,
    health_callback: ProjectedResponseCallback,
) -> tuple[torch.Tensor, FittedFrameRecord]:
    """Run exactly 200 audited projector updates and two fixed health calls."""
    state.authorize_training(
        spec.frame_id,
        split="FIT",
        training_targets=spec.training_targets,
        health_targets=spec.health_targets,
    )
    if not training_callback.optimization or training_callback.fixed_health_batch:
        raise RuntimeError("training callback is not the registered optimizer callback")
    if health_callback.optimization or not health_callback.fixed_health_batch:
        raise RuntimeError("health callback is not the registered fixed VALIDATION callback")
    if training_callback.spec != spec or health_callback.spec != spec:
        raise RuntimeError("callbacks do not belong to the registered frame")
    coefficient = 0.0 if spec.family == "recovery_only" else 24.0
    config = core.OptimizerConfig(control_coefficient=coefficient)
    if config.updates != UPDATES or config.rank != RANK:
        raise RuntimeError("optimizer configuration changed")
    initial = core.deterministic_haar_frame(
        D, RANK, spec.seed, dtype=torch.float32, device=instrument.device
    )
    raw = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam(
        (raw,),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    with torch.no_grad():
        initial_health = core.exact_max_target_objective(
            health_callback(initial, -1),
            control_coefficient=coefficient,
            epsilon=config.loss_epsilon,
        )
        initial_validation = float(initial_health.maximum)

    history: list[float] = []
    maximizing: list[str] = []
    for update in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        frame = core.differentiable_qr_retraction(raw)
        before_forward = instrument.ledger.optimization_forwards
        before_backward = instrument.ledger.optimization_backwards
        responses = training_callback(frame, update)
        if instrument.ledger.optimization_forwards != before_forward + 1:
            raise RuntimeError("one optimizer update did not use exactly one forward")
        objective = core.exact_max_target_objective(
            responses,
            control_coefficient=coefficient,
            epsilon=config.loss_epsilon,
        )
        if not bool(torch.isfinite(objective.maximum).detach().cpu()):
            raise FloatingPointError(f"non-finite objective at {spec.frame_id}/{update}")
        objective.maximum.backward()
        instrument.ledger.charge("optimization_backward")
        state.record_optimization_events(0, 1)
        if instrument.ledger.optimization_backwards != before_backward + 1:
            raise RuntimeError("one optimizer update did not use exactly one backward")
        if raw.grad is None or not bool(torch.isfinite(raw.grad).all().detach().cpu()):
            raise FloatingPointError(f"absent/non-finite frame gradient at {spec.frame_id}/{update}")
        core.assert_parameters_have_no_gradients(tuple(instrument.model.parameters()))
        optimizer.step()
        history.append(float(objective.maximum.detach()))
        maximizing.append(objective.maximizing_target)

    final = core.differentiable_qr_retraction(raw.detach()).cpu().contiguous()
    with torch.no_grad():
        final_health = core.exact_max_target_objective(
            health_callback(final.to(instrument.device), -1),
            control_coefficient=coefficient,
            epsilon=config.loss_epsilon,
        )
        final_validation = float(final_health.maximum)
    orthonormality = float(
        (final.mT @ final - torch.eye(RANK, dtype=torch.float32)).abs().amax()
    )
    initial_cpu = initial.detach().cpu().float()
    overlap = (initial_cpu.mT @ final).square().sum()
    distance = float((2 * RANK - 2 * overlap).clamp_min(0).sqrt())
    window = config.health_window
    initial_window = sum(history[:window]) / window
    final_window = sum(history[-window:]) / window
    failures = []
    if not all(math.isfinite(value) for value in history):
        failures.append("nonfinite_loss")
    if final_window >= initial_window:
        failures.append("final_window_not_below_initial_window")
    if final_validation >= initial_validation:
        failures.append("validation_not_better_than_initialization")
    if orthonormality > config.orthonormality_atol:
        failures.append("orthonormality")
    if distance <= config.minimum_projector_distance:
        failures.append("projector_did_not_move")
    counts = {
        target: maximizing.count(target) for target in sorted(set(maximizing))
    }
    loss_history_sha256 = _sha256_json(history)
    fit_selected_batches_sha256 = _scheduled_batches_sha256(
        training_callback.balanced, range(UPDATES)
    )
    validation_health_batch_sha256 = _scheduled_batches_sha256(
        health_callback.balanced, (0,)
    )
    fit_scheduler_payload = _scheduler_payload(training_callback.balanced)
    validation_scheduler_payload = _scheduler_payload(health_callback.balanced)
    spec_payload = asdict(spec)
    fit_record_payload = {
        "frame_id": spec.frame_id,
        "spec": spec_payload,
        "frame_sha256": archive.tensor_sha256(final),
        "fit_scheduler_sha256": training_callback.balanced.fingerprint,
        "fit_batch_zero_selected_row_ids": _batch_zero_rows(fit_scheduler_payload),
        "coefficient": coefficient,
        "optimizer": asdict(config),
        "loss_history": history,
        "maximizing_targets": maximizing,
    }
    health_record_payload = {
        "frame_id": spec.frame_id,
        "spec": spec_payload,
        "frame_sha256": archive.tensor_sha256(final),
        "validation_scheduler_sha256": health_callback.balanced.fingerprint,
        "validation_batch_zero_selected_row_ids": _batch_zero_rows(
            validation_scheduler_payload
        ),
        "healthy": not failures,
        "failures": failures,
        "initial_validation_objective": initial_validation,
        "final_validation_objective": final_validation,
        "initial_window_mean": initial_window,
        "final_window_mean": final_window,
        "orthonormality_error": orthonormality,
        "projector_distance_from_initialization": distance,
    }
    fit_record_sha256 = _sha256_json(fit_record_payload)
    health_record_sha256 = _sha256_json(health_record_payload)
    record = FittedFrameRecord(
        frame_id=spec.frame_id,
        frame_sha256=archive.tensor_sha256(final),
        fit_scheduler_sha256=training_callback.balanced.fingerprint,
        validation_scheduler_sha256=health_callback.balanced.fingerprint,
        healthy=not failures,
        health_failures=tuple(failures),
        initial_validation_objective=initial_validation,
        final_validation_objective=final_validation,
        initial_window_mean=initial_window,
        final_window_mean=final_window,
        orthonormality_error=orthonormality,
        projector_distance_from_initialization=distance,
        maximizing_target_counts=counts,
        loss_history_sha256=loss_history_sha256,
        fit_selected_batches_sha256=fit_selected_batches_sha256,
        validation_health_batch_sha256=validation_health_batch_sha256,
        fit_record_sha256=fit_record_sha256,
        health_record_sha256=health_record_sha256,
        fit_scheduler_payload=fit_scheduler_payload,
        validation_scheduler_payload=validation_scheduler_payload,
        fit_record_payload=fit_record_payload,
        health_record_payload=health_record_payload,
    )
    combined_scheduler_hash = hashlib.sha256(
        (record.fit_scheduler_sha256 + record.validation_scheduler_sha256).encode()
    ).hexdigest()
    state.register_frozen_frame(
        state_guard.FrozenFrame(
            spec=spec,
            frame_sha256=record.frame_sha256,
            scheduler_sha256=combined_scheduler_hash,
        )
    )
    del initial, raw, optimizer
    return final, record


def _exclusive_pairs_for_cell(design: dict, cell: str) -> dict[str, TargetPairs]:
    matches = design["cells"][cell]["exclusive"]
    if set(matches) != set(QUARTET_TAGS):
        raise RuntimeError(f"{cell} exclusive-pair census changed")
    return {
        target: _pairs_from_match(target, matches[target]) for target in QUARTET_TAGS
    }


def _training_pairs_for_spec(
    spec: state_guard.FrameSpec,
    real_fit_pairs: Mapping[str, TargetPairs],
    null_designs: Mapping[int, LabelNullDesign],
) -> Mapping[str, TargetPairs]:
    if spec.family == "label_null":
        return null_designs[spec.seed].pairs
    return real_fit_pairs


def train_all_registered_frames(
    instrument: Rung522Instrument,
    validation_full: SwapEvaluation,
    null_designs: Mapping[int, LabelNullDesign],
) -> tuple[dict[str, torch.Tensor], dict[str, FittedFrameRecord]]:
    """Fit the exact registered 103-frame census before TEST can open."""
    if instrument.full_fit_d0 is None:
        raise RuntimeError("whole-attention8 FIT targets must be cached before fitting")
    real_fit_pairs = {
        target: _combined_fit_pairs(instrument.design, target) for target in QUARTET_TAGS
    }
    validation_pairs = _exclusive_pairs_for_cell(instrument.design, "validation")
    validation_full_map0 = validation_full.map_responses["D0:forward"][:1]
    frames: dict[str, torch.Tensor] = {}
    records: dict[str, FittedFrameRecord] = {}
    specs = tuple(state_guard.EXPECTED_FRAME_SPECS.values())
    if len(specs) != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError("registered frame specification census changed")
    for index, spec in enumerate(specs):
        fit_pairs = _training_pairs_for_spec(spec, real_fit_pairs, null_designs)
        fit_scheduler = _make_balanced_scheduler(
            spec, fit_pairs, instrument.data["row_masks"]["fit"]
        )
        health_scheduler = _make_balanced_scheduler(
            spec, validation_pairs, instrument.data["row_masks"]["validation"]
        )
        training_callback = ProjectedResponseCallback(
            instrument,
            spec,
            split="fit",
            pairs=fit_pairs,
            balanced=fit_scheduler,
            full_by_map=instrument.full_fit_d0,
            optimization=True,
        )
        health_callback = ProjectedResponseCallback(
            instrument,
            spec,
            split="validation",
            pairs=validation_pairs,
            balanced=health_scheduler,
            full_by_map=validation_full_map0,
            optimization=False,
            fixed_health_batch=True,
        )
        frame, record = fit_one_registered_frame(
            instrument,
            spec,
            instrument.state,
            training_callback,
            health_callback,
        )
        if record.frame_id in frames:
            raise RuntimeError("duplicate fitted frame ID")
        frames[record.frame_id] = frame
        records[record.frame_id] = record
        print(
            f"RUNG522 FRAME {index + 1:03d}/103 {record.frame_id} "
            f"healthy={record.healthy} frame={record.frame_sha256[:12]}",
            flush=True,
        )
        if instrument.device.type == "cuda" and (index + 1) % 10 == 0:
            torch.cuda.empty_cache()
    if set(frames) != set(state_guard.EXPECTED_FRAME_SPECS) or set(records) != set(frames):
        raise RuntimeError("fitted frame census differs from the registered 103 objects")
    if instrument.state.frame_count != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError("protocol state did not freeze all 103 frames")
    return frames, records


def _split_local_indices(
    global_indices: torch.Tensor,
    row_to_local: torch.Tensor,
) -> torch.Tensor:
    local_rows = row_to_local[global_indices // TOKENS]
    if bool((local_rows < 0).any()):
        raise RuntimeError("metric pair contains a position outside its data split")
    return local_rows * TOKENS + global_indices % TOKENS


def _pair_effects(
    response: torch.Tensor,
    pairs: TargetPairs,
    row_to_local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if response.device.type != "cpu" or response.ndim != 2 or response.shape[1] != TOKENS:
        raise ValueError("saved response must be a CPU [rows, tokens] tensor")
    flat = response.reshape(-1)
    members = _split_local_indices(pairs.members, row_to_local)
    controls = _split_local_indices(pairs.controls, row_to_local)
    return flat[members].double(), flat[controls].double()


def _row_pair_squares(
    member_effects: torch.Tensor,
    control_effects: torch.Tensor,
    pairs: TargetPairs,
) -> protocol.RowPairSquares:
    if member_effects.shape != control_effects.shape or member_effects.numel() != pairs.members.numel():
        raise ValueError("effect vectors no longer align with the frozen matched pairs")
    member_rows = pairs.members // TOKENS
    unique_rows = member_rows.unique(sorted=True)
    member_ss = []
    member_counts = []
    control_ss = []
    control_counts = []
    for row in unique_rows.tolist():
        selected = member_rows == row
        member_ss.append(float(member_effects[selected].square().sum()))
        member_counts.append(int(selected.sum()))
        control_ss.append(float(control_effects[selected].square().sum()))
        control_counts.append(int(selected.sum()))
    return protocol.RowPairSquares.from_sequences(
        member_ss,
        member_counts,
        control_ss,
        control_counts,
        pair_ids=unique_rows.tolist(),
    )


def score_response_cell(
    projected: torch.Tensor,
    full_attention8: torch.Tensor,
    pairs: TargetPairs,
    row_to_local: torch.Tensor,
    *,
    cell_id: str,
    selectivity_comparison: torch.Tensor | None = None,
    run_bootstrap: bool = True,
) -> dict[str, object]:
    """Compute the frozen A-cell metrics from saved per-token responses."""
    projected_member, projected_control = _pair_effects(projected, pairs, row_to_local)
    full_member, full_control = _pair_effects(full_attention8, pairs, row_to_local)
    signed = core.signed_response_metrics(projected_member, full_member)
    selectivity = protocol.selectivity_from_effects(projected_member, projected_control)
    full_selectivity = protocol.selectivity_from_effects(full_member, full_control)
    rows = _row_pair_squares(projected_member, projected_control, pairs)
    comparison_rows = None
    if selectivity_comparison is not None and not run_bootstrap:
        raise ValueError("a selectivity comparison requires the paired bootstrap")
    if selectivity_comparison is not None:
        comparison_member, comparison_control = _pair_effects(
            selectivity_comparison, pairs, row_to_local
        )
        comparison_rows = _row_pair_squares(
            comparison_member, comparison_control, pairs
        )
    bootstrap = (
        protocol.deterministic_row_bootstrap(
            rows, cell_id=cell_id, comparison=comparison_rows
        )
        if run_bootstrap else None
    )
    exact = pairs.tiers <= 1
    exact_token = None
    if run_bootstrap and int(exact.sum()) >= 32:
        exact_members = projected_member[exact]
        exact_controls = projected_control[exact]
        exact_pairs = TargetPairs(
            pairs.target,
            pairs.members[exact],
            pairs.controls[exact],
            pairs.tiers[exact],
        )
        exact_rows = _row_pair_squares(exact_members, exact_controls, exact_pairs)
        exact_bootstrap = protocol.deterministic_row_bootstrap(
            exact_rows, cell_id=f"{cell_id}:exact-token-tier0-or1"
        )
        exact_summary = protocol.selectivity_from_effects(exact_members, exact_controls)
        exact_token = {
            "pair_count": int(exact.sum()),
            "member_rms": exact_summary.member_rms,
            "control_rms": exact_summary.control_rms,
            "concentration": exact_summary.concentration,
            "fourfold_margin_lower95": exact_bootstrap.fourfold_margin_lower95_higher,
            "passes": bool(
                exact_summary.concentration >= 4
                and exact_bootstrap.fourfold_margin_lower95_higher > 0
            ),
        }
    result = {
        **signed,
        "member_rms": selectivity.member_rms,
        "control_rms": selectivity.control_rms,
        "concentration": selectivity.concentration,
        "bounded_selectivity": selectivity.bounded_selectivity,
        "fourfold_margin": selectivity.fourfold_margin,
        "fourfold_margin_lower95": (
            None if bootstrap is None else bootstrap.fourfold_margin_lower95_higher
        ),
        "bounded_selectivity_improvement_lower95": (
            None if bootstrap is None
            else bootstrap.bounded_selectivity_improvement_lower95_higher
        ),
        "full_attention8_concentration": full_selectivity.concentration,
        "concentration_improvement_over_full_attention8": (
            selectivity.concentration - full_selectivity.concentration
        ),
        "pair_count": int(pairs.members.numel()),
        "member_row_clusters": len(rows),
        "bootstrap_sha256": None if bootstrap is None else bootstrap.sha256,
        "exact_token_tier0_or1": exact_token,
    }
    result["base_gates_pass"] = bool(run_bootstrap and
        result["signed_cosine"] >= 0.75
        and result["relative_residual"] <= 0.55
        and result["aligned_recovery"] > 0
        and result["member_rms"] >= 0.02
        and result["concentration"] >= 4
        and result["concentration_improvement_over_full_attention8"] >= 1
        and result["fourfold_margin_lower95"] > 0
        and (exact_token is None or exact_token["passes"])
    )
    return result


@dataclass(frozen=True)
class CompactSwapEvaluation:
    split: str
    kind: str
    cell_responses: Mapping[str, torch.Tensor]
    forward_calls: int
    minimum_edit_rms: float
    response_sha256: str


def _compact_swap(value: SwapEvaluation) -> CompactSwapEvaluation:
    return CompactSwapEvaluation(
        split=value.split,
        kind=value.kind,
        cell_responses={
            name: tensor.detach().cpu().contiguous()
            for name, tensor in value.cell_responses.items()
        },
        forward_calls=value.forward_calls,
        minimum_edit_rms=value.minimum_edit_rms,
        response_sha256=value.response_sha256,
    )


def _score_target_cells(
    evaluation: CompactSwapEvaluation,
    full_attention8: CompactSwapEvaluation,
    pairs: TargetPairs,
    row_to_local: torch.Tensor,
    *,
    frame_id: str,
    comparison: CompactSwapEvaluation | None = None,
    run_bootstrap: bool,
) -> dict[str, dict[str, object]]:
    if set(evaluation.cell_responses) != set(validation_gates.VALIDATION_CELLS):
        raise RuntimeError("swap evaluation cell census changed")
    if set(full_attention8.cell_responses) != set(evaluation.cell_responses):
        raise RuntimeError("whole-attention8 comparator cells changed")
    if comparison is not None and set(comparison.cell_responses) != set(
        evaluation.cell_responses
    ):
        raise RuntimeError("paired selectivity-control cells changed")
    return {
        cell: score_response_cell(
            response,
            full_attention8.cell_responses[cell],
            pairs,
            row_to_local,
            cell_id=f"{evaluation.split}:{frame_id}:{pairs.target}:{cell}",
            selectivity_comparison=(
                None if comparison is None else comparison.cell_responses[cell]
            ),
            run_bootstrap=run_bootstrap,
        )
        for cell, response in evaluation.cell_responses.items()
    }


def _joint_statistic_from_cells(cells: Mapping[str, Mapping[str, object]]) -> float:
    selectivities = [float(cell["bounded_selectivity"]) for cell in cells.values()]
    recoveries = [float(cell["aligned_recovery"]) for cell in cells.values()]
    return protocol.bounded_joint_statistic(selectivities, recoveries).product


def evaluate_validation_suite(
    instrument: Rung522Instrument,
    frames: Mapping[str, torch.Tensor],
    records: Mapping[str, FittedFrameRecord],
    validation_full: SwapEvaluation,
) -> dict[str, object]:
    """Run and score the complete frozen pre-TEST VALIDATION suite."""
    full = _compact_swap(validation_full)
    evaluations: dict[str, CompactSwapEvaluation] = {}
    a_families = {"real_leave_one_out", "target_oracle", "label_null"}
    for spec in state_guard.EXPECTED_FRAME_SPECS.values():
        if spec.family in a_families:
            evaluations[spec.frame_id] = _compact_swap(instrument.evaluate_swap(
                "validation", frame=frames[spec.frame_id], inference_bucket="prediction_a"
            ))
    for spec in state_guard.EXPECTED_FRAME_SPECS.values():
        if spec.family == "recovery_only":
            evaluations[spec.frame_id] = _compact_swap(instrument.evaluate_swap(
                "validation", frame=frames[spec.frame_id], inference_bucket="recovery_only"
            ))
    haar_frames = {
        seed: core.deterministic_haar_frame(
            D, RANK, seed, dtype=torch.float32, device=torch.device("cpu")
        )
        for seed in HAAR_SEEDS
    }
    haar_evaluations = {
        seed: _compact_swap(instrument.evaluate_swap(
            "validation", frame=frame, inference_bucket="haar"
        ))
        for seed, frame in haar_frames.items()
    }
    for spec in state_guard.EXPECTED_FRAME_SPECS.values():
        if spec.family == "all_three":
            evaluations[spec.frame_id] = _compact_swap(instrument.evaluate_swap(
                "validation",
                frame=frames[spec.frame_id],
                inference_bucket="all_three_selection_and_test",
            ))

    pairs = _exclusive_pairs_for_cell(instrument.design, "validation")
    row_to_local = instrument.row_to_local["validation"]
    real: dict[str, dict[int, dict[str, object]]] = {target: {} for target in FITTED_TAGS}
    recovery: dict[str, dict[int, dict[str, object]]] = {
        target: {} for target in FITTED_TAGS
    }
    oracles: dict[str, dict[int, dict[str, object]]] = {
        target: {} for target in FITTED_TAGS
    }
    label_null: dict[str, dict[int, dict[str, object]]] = {
        target: {} for target in FITTED_TAGS
    }
    haar: dict[str, dict[int, dict[str, object]]] = {target: {} for target in FITTED_TAGS}
    haar_joint: dict[str, list[float]] = {target: [] for target in FITTED_TAGS}
    label_null_joint: dict[str, list[float]] = {target: [] for target in FITTED_TAGS}

    for omitted in FITTED_TAGS:
        for seed in REAL_SEEDS:
            real_id = f"real_leave_one_out:{omitted}:{seed}"
            recovery_id = f"recovery_only:{omitted}:{seed}"
            oracle_id = f"target_oracle:{omitted}:{seed}"
            recovery_cells = _score_target_cells(
                evaluations[recovery_id], full, pairs[omitted], row_to_local,
                frame_id=recovery_id, run_bootstrap=False,
            )
            real_cells = _score_target_cells(
                evaluations[real_id], full, pairs[omitted], row_to_local,
                frame_id=real_id, comparison=evaluations[recovery_id], run_bootstrap=True,
            )
            oracle_cells = _score_target_cells(
                evaluations[oracle_id], full, pairs[omitted], row_to_local,
                frame_id=oracle_id, run_bootstrap=False,
            )
            real[omitted][seed] = {
                "healthy": records[real_id].healthy,
                "cells": real_cells,
                "response_sha256": evaluations[real_id].response_sha256,
            }
            recovery[omitted][seed] = {
                "healthy": records[recovery_id].healthy,
                "cells": recovery_cells,
                "response_sha256": evaluations[recovery_id].response_sha256,
            }
            oracles[omitted][seed] = {
                "healthy": records[oracle_id].healthy,
                "cells": oracle_cells,
                "response_sha256": evaluations[oracle_id].response_sha256,
            }
        for null_seed in NULL_SEEDS:
            frame_id = f"label_null:{null_seed}:{omitted}"
            cells = _score_target_cells(
                evaluations[frame_id], full, pairs[omitted], row_to_local,
                frame_id=frame_id, run_bootstrap=False,
            )
            label_null[omitted][null_seed] = {
                "healthy": records[frame_id].healthy,
                "cells": cells,
                "response_sha256": evaluations[frame_id].response_sha256,
            }
            label_null_joint[omitted].append(_joint_statistic_from_cells(cells))
        for seed in HAAR_SEEDS:
            frame_id = f"haar:{seed}"
            cells = _score_target_cells(
                haar_evaluations[seed], full, pairs[omitted], row_to_local,
                frame_id=frame_id, run_bootstrap=False,
            )
            haar[omitted][seed] = {"healthy": True, "cells": cells}
            haar_joint[omitted].append(_joint_statistic_from_cells(cells))

    reserved_oracles = {}
    for seed in REAL_SEEDS:
        frame_id = f"target_oracle:{REUSE_TAG}:{seed}"
        reserved_oracles[seed] = {
            "healthy": records[frame_id].healthy,
            "cells": _score_target_cells(
                evaluations[frame_id], full, pairs[REUSE_TAG], row_to_local,
                frame_id=frame_id, run_bootstrap=False,
            ),
            "response_sha256": evaluations[frame_id].response_sha256,
        }

    all_three = {}
    for seed in REAL_SEEDS:
        frame_id = f"all_three:{seed}"
        all_three[seed] = {
            "frame_id": frame_id,
            "healthy": records[frame_id].healthy,
            "targets": {
                target: {
                    "cells": _score_target_cells(
                        evaluations[frame_id], full, pairs[target], row_to_local,
                        frame_id=f"{frame_id}:{target}", run_bootstrap=True,
                    )
                }
                for target in FITTED_TAGS
            },
            "response_sha256": evaluations[frame_id].response_sha256,
        }

    return {
        "real": real,
        "recovery_only": recovery,
        "oracles": oracles,
        "reserved_oracles": reserved_oracles,
        "label_null": label_null,
        "label_null_fit_health": {
            seed: {
                omitted: records[f"label_null:{seed}:{omitted}"].healthy
                for omitted in FITTED_TAGS
            }
            for seed in NULL_SEEDS
        },
        "haar": haar,
        "haar_joint": haar_joint,
        "label_null_joint": label_null_joint,
        "all_three": all_three,
        "real_frames": {
            seed: {
                omitted: frames[f"real_leave_one_out:{omitted}:{seed}"]
                for omitted in FITTED_TAGS
            }
            for seed in REAL_SEEDS
        },
        "label_null_frames": {
            seed: {
                omitted: frames[f"label_null:{seed}:{omitted}"]
                for omitted in FITTED_TAGS
            }
            for seed in NULL_SEEDS
        },
        "haar_hashes": {
            seed: archive.tensor_sha256(frame) for seed, frame in haar_frames.items()
        },
        "full_attention8_response_sha256": validation_full.response_sha256,
    }


def evaluate_test_suite(
    instrument: Rung522Instrument,
    frames: Mapping[str, torch.Tensor],
    records: Mapping[str, FittedFrameRecord],
    selected_all_three_frame_id: str,
) -> dict[str, object]:
    """Execute the one-way TEST model sweep; no selection occurs here."""
    if not instrument.state.test_open:
        raise RuntimeError("TEST suite requires an already-open one-way protocol state")
    test_full_raw = instrument.evaluate_swap(
        "test", frame=None, inference_bucket="full_attention8_comparator"
    )
    full = _compact_swap(test_full_raw)
    evaluations: dict[str, CompactSwapEvaluation] = {}
    a_families = {"real_leave_one_out", "target_oracle", "label_null"}
    for spec in state_guard.EXPECTED_FRAME_SPECS.values():
        if spec.family in a_families:
            evaluations[spec.frame_id] = _compact_swap(instrument.evaluate_swap(
                "test", frame=frames[spec.frame_id], inference_bucket="prediction_a"
            ))
    for spec in state_guard.EXPECTED_FRAME_SPECS.values():
        if spec.family == "recovery_only":
            evaluations[spec.frame_id] = _compact_swap(instrument.evaluate_swap(
                "test", frame=frames[spec.frame_id], inference_bucket="recovery_only"
            ))
    haar_frames = {
        seed: core.deterministic_haar_frame(
            D, RANK, seed, dtype=torch.float32, device=torch.device("cpu")
        )
        for seed in HAAR_SEEDS
    }
    haar_evaluations = {
        seed: _compact_swap(instrument.evaluate_swap(
            "test", frame=frame, inference_bucket="haar"
        ))
        for seed, frame in haar_frames.items()
    }
    selected = _compact_swap(instrument.evaluate_swap(
        "test",
        frame=frames[selected_all_three_frame_id],
        inference_bucket="all_three_selection_and_test",
    ))

    pairs = _exclusive_pairs_for_cell(instrument.design, "test")
    row_to_local = instrument.row_to_local["test"]
    real: dict[str, dict[int, dict[str, object]]] = {target: {} for target in FITTED_TAGS}
    recovery: dict[str, dict[int, dict[str, object]]] = {
        target: {} for target in FITTED_TAGS
    }
    oracles: dict[str, dict[int, dict[str, object]]] = {
        target: {} for target in FITTED_TAGS
    }
    label_null: dict[str, dict[int, dict[str, object]]] = {
        target: {} for target in FITTED_TAGS
    }
    haar: dict[str, dict[int, dict[str, object]]] = {target: {} for target in FITTED_TAGS}
    haar_joint: dict[str, list[float]] = {target: [] for target in FITTED_TAGS}
    label_null_joint: dict[str, list[float]] = {target: [] for target in FITTED_TAGS}
    for omitted in FITTED_TAGS:
        for seed in REAL_SEEDS:
            real_id = f"real_leave_one_out:{omitted}:{seed}"
            recovery_id = f"recovery_only:{omitted}:{seed}"
            oracle_id = f"target_oracle:{omitted}:{seed}"
            recovery_cells = _score_target_cells(
                evaluations[recovery_id], full, pairs[omitted], row_to_local,
                frame_id=recovery_id, run_bootstrap=False,
            )
            real_cells = _score_target_cells(
                evaluations[real_id], full, pairs[omitted], row_to_local,
                frame_id=real_id, comparison=evaluations[recovery_id], run_bootstrap=True,
            )
            oracle_cells = _score_target_cells(
                evaluations[oracle_id], full, pairs[omitted], row_to_local,
                frame_id=oracle_id, run_bootstrap=False,
            )
            real[omitted][seed] = {
                "healthy": records[real_id].healthy, "cells": real_cells,
                "response_sha256": evaluations[real_id].response_sha256,
            }
            recovery[omitted][seed] = {
                "healthy": records[recovery_id].healthy, "cells": recovery_cells,
                "response_sha256": evaluations[recovery_id].response_sha256,
            }
            oracles[omitted][seed] = {
                "healthy": records[oracle_id].healthy, "cells": oracle_cells,
                "response_sha256": evaluations[oracle_id].response_sha256,
            }
        for null_seed in NULL_SEEDS:
            frame_id = f"label_null:{null_seed}:{omitted}"
            cells = _score_target_cells(
                evaluations[frame_id], full, pairs[omitted], row_to_local,
                frame_id=frame_id, run_bootstrap=False,
            )
            label_null[omitted][null_seed] = {
                "healthy": records[frame_id].healthy, "cells": cells,
                "response_sha256": evaluations[frame_id].response_sha256,
            }
            label_null_joint[omitted].append(_joint_statistic_from_cells(cells))
        for seed in HAAR_SEEDS:
            cells = _score_target_cells(
                haar_evaluations[seed], full, pairs[omitted], row_to_local,
                frame_id=f"haar:{seed}", run_bootstrap=False,
            )
            haar[omitted][seed] = {"healthy": True, "cells": cells}
            haar_joint[omitted].append(_joint_statistic_from_cells(cells))

    reserved_oracles = {}
    for seed in REAL_SEEDS:
        frame_id = f"target_oracle:{REUSE_TAG}:{seed}"
        reserved_oracles[seed] = {
            "healthy": records[frame_id].healthy,
            "cells": _score_target_cells(
                evaluations[frame_id], full, pairs[REUSE_TAG], row_to_local,
                frame_id=frame_id, run_bootstrap=False,
            ),
            "response_sha256": evaluations[frame_id].response_sha256,
        }
    selected_reuse_cells = _score_target_cells(
        selected, full, pairs[REUSE_TAG], row_to_local,
        frame_id=f"{selected_all_three_frame_id}:reserved-reuse", run_bootstrap=True,
    )
    return {
        "real": real,
        "recovery_only": recovery,
        "oracles": oracles,
        "reserved_oracles": reserved_oracles,
        "label_null": label_null,
        "haar": haar,
        "haar_joint": haar_joint,
        "label_null_joint": label_null_joint,
        "selected_all_three": {
            "frame_id": selected_all_three_frame_id,
            "healthy": records[selected_all_three_frame_id].healthy,
            "reuse_cells": selected_reuse_cells,
            "evaluation": selected,
        },
        "full_attention8_response_sha256": test_full_raw.response_sha256,
    }


def _prefixed_cells(
    validation_cells: Mapping[str, Mapping[str, object]],
    test_cells: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if set(validation_cells) != set(validation_gates.VALIDATION_CELLS) or set(
        test_cells
    ) != set(validation_gates.VALIDATION_CELLS):
        raise RuntimeError("cannot combine incomplete VALIDATION/TEST cells")
    return {
        **{f"validation:{name}": value for name, value in validation_cells.items()},
        **{f"test:{name}": value for name, value in test_cells.items()},
    }


def combined_final_ab_inputs(
    validation: Mapping[str, object], test: Mapping[str, object]
) -> dict[str, object]:
    """Join saved half metrics into the exact eight-cell final A/B input."""
    combined: dict[str, object] = {}
    for family in ("real", "recovery_only", "oracles"):
        by_fold = {}
        for fold in FITTED_TAGS:
            by_seed = {}
            for seed in REAL_SEEDS:
                left = validation[family][fold][seed]
                right = test[family][fold][seed]
                if left["healthy"] is not right["healthy"]:
                    raise RuntimeError(f"{family}/{fold}/{seed} health changed across halves")
                by_seed[seed] = {
                    "healthy": left["healthy"],
                    "cells": _prefixed_cells(left["cells"], right["cells"]),
                }
            by_fold[fold] = by_seed
        combined[family] = by_fold
    reserved = {}
    for seed in REAL_SEEDS:
        left = validation["reserved_oracles"][seed]
        right = test["reserved_oracles"][seed]
        reserved[seed] = {
            "healthy": left["healthy"],
            "cells": _prefixed_cells(left["cells"], right["cells"]),
        }
    combined["reserved_oracles"] = reserved

    haar_joint = {fold: [] for fold in FITTED_TAGS}
    label_null_joint = {fold: [] for fold in FITTED_TAGS}
    for fold in FITTED_TAGS:
        for seed in HAAR_SEEDS:
            cells = _prefixed_cells(
                validation["haar"][fold][seed]["cells"],
                test["haar"][fold][seed]["cells"],
            )
            haar_joint[fold].append(_joint_statistic_from_cells(cells))
        for seed in NULL_SEEDS:
            cells = _prefixed_cells(
                validation["label_null"][fold][seed]["cells"],
                test["label_null"][fold][seed]["cells"],
            )
            label_null_joint[fold].append(_joint_statistic_from_cells(cells))
    combined["haar_joint"] = haar_joint
    combined["label_null_joint"] = label_null_joint
    combined["real_frames"] = validation["real_frames"]
    combined["label_null_frames"] = validation["label_null_frames"]
    return combined


def fingerprint_pairs(design: dict, cell: str) -> dict[str, TargetPairs]:
    matches = design["cells"][cell]["fingerprint"]
    if tuple(matches) != tuple(stage_a.FINGERPRINT_TAGS):
        # Dict construction order is frozen in rung521, but compare sets too so
        # an informative failure survives a benign serialization reordering.
        if set(matches) != set(stage_a.FINGERPRINT_TAGS):
            raise RuntimeError("32-circuit fingerprint census changed")
    return {
        target: _pairs_from_match(target, matches[target])
        for target in stage_a.FINGERPRINT_TAGS
    }


def fingerprint_coordinates(
    response: torch.Tensor,
    pairs_by_target: Mapping[str, TargetPairs],
    row_to_local: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Compute the common 32-circuit RMS-member-minus-control coordinates."""
    if set(pairs_by_target) != set(stage_a.FINGERPRINT_TAGS):
        raise ValueError("fingerprint scoring requires exactly the frozen 32 circuits")
    result = {}
    for target in stage_a.FINGERPRINT_TAGS:
        member, control = _pair_effects(response, pairs_by_target[target], row_to_local)
        member_rms = float(member.square().mean().sqrt())
        control_rms = float(control.square().mean().sqrt())
        result[target] = {
            "member_rms": member_rms,
            "control_rms": control_rms,
            "coordinate": member_rms - control_rms,
            "signed_mean_member_minus_control": float(member.mean() - control.mean()),
        }
    return result


def quartet_separation(coordinates: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if set(coordinates) != set(stage_a.FINGERPRINT_TAGS):
        raise ValueError("quartet separation requires the frozen 32 coordinates")
    quartet_min = min(float(coordinates[target]["coordinate"]) for target in QUARTET_TAGS)
    nonquartet_max = max(
        float(coordinates[target]["coordinate"])
        for target in stage_a.FINGERPRINT_TAGS
        if target not in QUARTET_TAGS
    )
    return {
        "minimum_quartet_coordinate": quartet_min,
        "maximum_nonquartet_coordinate": nonquartet_max,
        "separation": quartet_min - nonquartet_max,
    }


def outside_union_damage(
    response: torch.Tensor,
    data: dict,
    split: str,
    exclusive_pairs: Mapping[str, TargetPairs],
    row_to_local: torch.Tensor,
) -> dict[str, float | bool]:
    rows = data["row_masks"][split].nonzero().flatten()
    global_positions = (
        rows[:, None] * TOKENS + torch.arange(TOKENS, dtype=torch.int64)[None, :]
    ).reshape(-1)
    outside = ~data["quartet_union"][global_positions]
    outside_rms = float(response.reshape(-1)[outside].double().square().mean().sqrt())
    quartet_member_rms = []
    for target in QUARTET_TAGS:
        member, _ = _pair_effects(response, exclusive_pairs[target], row_to_local)
        quartet_member_rms.append(float(member.square().mean().sqrt()))
    smallest = min(quartet_member_rms)
    ratio = outside_rms / max(smallest, 1e-30)
    return {
        "outside_union_rms": outside_rms,
        "smallest_quartet_member_rms": smallest,
        "outside_to_smallest_quartet_ratio": ratio,
        "passes_at_most_25_percent": ratio <= 0.25,
    }


def _pairs_member_anchored_in_fold(
    pairs: Mapping[str, TargetPairs],
    folds: torch.Tensor,
    fold: int,
) -> dict[str, TargetPairs]:
    result = {}
    for target, value in pairs.items():
        selected = folds[value.members // TOKENS] == fold
        if not bool(selected.any()):
            raise RuntimeError(f"{target} has no member-anchored pairs in fold {fold}")
        result[target] = TargetPairs(
            target,
            value.members[selected].contiguous(),
            value.controls[selected].contiguous(),
            value.tiers[selected].contiguous(),
        )
    return result


def score_removal_fold_without_null(
    removal: RemovalEvaluation,
    selected_swap: CompactSwapEvaluation,
    pairs: Mapping[str, TargetPairs],
    row_to_local: torch.Tensor,
    *,
    fold: int,
) -> dict[str, object]:
    """Score the deterministic D clauses before the common-response null."""
    coordinates = fingerprint_coordinates(removal.response, pairs, row_to_local)
    separation = quartet_separation(coordinates)
    nonquartet_member = torch.tensor([
        float(coordinates[target]["member_rms"])
        for target in stage_a.FINGERPRINT_TAGS
        if target not in QUARTET_TAGS
    ], dtype=torch.float64)
    median_nonquartet = float(nonquartet_member.median())
    magnitude_ratios = {
        target: float(coordinates[target]["member_rms"]) / max(median_nonquartet, 1e-30)
        for target in QUARTET_TAGS
    }
    sign_checks = {}
    for target in QUARTET_TAGS:
        removal_sign = float(coordinates[target]["signed_mean_member_minus_control"])
        target_checks = {}
        for cell, response in selected_swap.cell_responses.items():
            member, control = _pair_effects(response, pairs[target], row_to_local)
            swap_sign = float(member.mean() - control.mean())
            same_nonzero_sign = removal_sign != 0 and swap_sign != 0 and (
                math.copysign(1.0, removal_sign) == math.copysign(1.0, swap_sign)
            )
            target_checks[cell] = {
                "removal_signed_mean_member_minus_control": removal_sign,
                "swap_signed_mean_member_minus_control": swap_sign,
                "same_nonzero_sign": same_nonzero_sign,
            }
        sign_checks[target] = target_checks
    deterministic_pass = bool(
        all(float(coordinates[target]["coordinate"]) > 0 for target in QUARTET_TAGS)
        and separation["separation"] > 0
        and all(value >= 2 for value in magnitude_ratios.values())
        and all(
            cell["same_nonzero_sign"]
            for target in sign_checks.values()
            for cell in target.values()
        )
    )
    return {
        "fold": fold,
        "coordinates": coordinates,
        "quartet_separation": separation,
        "median_nonquartet_member_rms": median_nonquartet,
        "quartet_to_median_nonquartet_member_rms": magnitude_ratios,
        "sign_checks": sign_checks,
        "deterministic_clauses_pass": deterministic_pass,
    }


def fingerprint_null_distribution(
    response: torch.Tensor,
    pairs_by_target: Mapping[str, TargetPairs],
    data: dict,
    descriptors: Mapping[str, torch.Tensor],
    split: str,
    *,
    cell_id: str,
    replicates: int = 20_000,
    fold_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    """Common-response coarse-stratum null for the 32-circuit max statistic."""
    if not isinstance(replicates, int) or not 1 <= replicates <= 20_000:
        raise ValueError("fingerprint null replicates must lie in 1..20000")
    rows = data["row_masks"][split].nonzero().flatten()
    if tuple(response.shape) != (rows.numel(), TOKENS) or response.device.type != "cpu":
        raise ValueError("fingerprint-null response does not match its CPU split")
    global_positions = (
        rows[:, None] * TOKENS + torch.arange(TOKENS, dtype=torch.int64)[None, :]
    ).reshape(-1)
    row_to_local = torch.full((1000,), -1, dtype=torch.int64)
    row_to_local[rows] = torch.arange(rows.numel())
    local_pairs = {
        target: sparse_null.IndexPairs.from_sequences(
            _split_local_indices(pairs.members, row_to_local),
            _split_local_indices(pairs.controls, row_to_local),
        )
        for target, pairs in pairs_by_target.items()
    }
    local_folds = None
    if fold_ids is not None:
        if fold_ids.device.type != "cpu" or tuple(fold_ids.shape) != (1000,):
            raise ValueError("fold_ids must be the frozen 1000-row CPU fold vector")
        local_folds = fold_ids[global_positions // TOKENS]
    null = sparse_null.evaluate_sparse_affine_fingerprint_null(
        response,
        global_positions,
        token_classes=descriptors["token_class"][global_positions],
        position_bins=descriptors["position_bin"][global_positions],
        ce_deciles=descriptors["ce_decile"][global_positions],
        fold_ids=local_folds,
        circuit_pairs=local_pairs,
        quartet_tags=QUARTET_TAGS,
        cell_id=cell_id,
        replicates=replicates,
    )
    observed_coordinates = fingerprint_coordinates(response, pairs_by_target, row_to_local)
    observed = quartet_separation(observed_coordinates)["separation"]
    q95 = null.null_q95_higher
    return {
        "cell_id": cell_id,
        "replicates": replicates,
        "observed_separation": observed,
        "null_q95_higher": q95,
        "observed_positive": observed > 0,
        "observed_strictly_above_q95": observed > q95,
        "passes": bool(observed > 0 and observed > q95),
        "queried_position_count": null.queried_position_count,
        "full_position_count": null.full_position_count,
        "maximum_materialized_sparse_map_elements": (
            null.maximum_materialized_sparse_map_elements
        ),
        "algorithm_definition_sha256": null.algorithm_definition_sha256,
        "null_samples_sha256": null.statistic_vector_sha256,
        "algorithm_and_samples_sha256": null.algorithm_and_statistic_sha256,
        "first_permutation_sha256": null.first_full_map_sha256,
        "last_permutation_sha256": null.last_full_map_sha256,
    }


def score_prediction_c(
    test_suite: Mapping[str, object],
    instrument: Rung522Instrument,
    descriptors: Mapping[str, torch.Tensor],
    selected_seed: int,
) -> dict[str, object]:
    selected_info = test_suite["selected_all_three"]
    if not isinstance(selected_info, Mapping):
        raise RuntimeError("selected all-three TEST record is malformed")
    selected = selected_info["evaluation"]
    if not isinstance(selected, CompactSwapEvaluation):
        raise RuntimeError("selected all-three TEST evaluation is absent")
    reuse_cells = selected_info["reuse_cells"]
    reserved_oracles = test_suite["reserved_oracles"]
    if not isinstance(reuse_cells, Mapping) or not isinstance(reserved_oracles, Mapping):
        raise RuntimeError("reserved reuse/oracle TEST metrics are malformed")
    oracle_cells = reserved_oracles[selected_seed]["cells"]
    reuse_cell_gates = {}
    for cell in validation_gates.VALIDATION_CELLS:
        metric = reuse_cells[cell]
        oracle = oracle_cells[cell]
        oracle_live = bool(
            float(oracle["member_rms"]) >= 0.02
            and float(oracle["aligned_recovery"]) >= 0.05
        )
        half_oracle = float(metric["aligned_recovery"]) >= 0.5 * float(
            oracle["aligned_recovery"]
        )
        reuse_cell_gates[cell] = {
            "metrics": metric,
            "same_seed_oracle_member_rms": oracle["member_rms"],
            "same_seed_oracle_aligned_recovery": oracle["aligned_recovery"],
            "oracle_live": oracle_live,
            "at_least_half_oracle_recovery": half_oracle,
            "passes": bool(metric["base_gates_pass"] and oracle_live and half_oracle),
        }

    fingerprint = fingerprint_pairs(instrument.design, "test")
    exclusive = _exclusive_pairs_for_cell(instrument.design, "test")
    row_to_local = instrument.row_to_local["test"]
    cell_results = {}
    for cell, response in selected.cell_responses.items():
        coordinates = fingerprint_coordinates(response, fingerprint, row_to_local)
        separation = quartet_separation(coordinates)
        null = fingerprint_null_distribution(
            response,
            fingerprint,
            instrument.data,
            descriptors,
            "test",
            cell_id=f"test:selected-all-three:{selected_seed}:{cell}",
        )
        damage = outside_union_damage(
            response, instrument.data, "test", exclusive, row_to_local
        )
        cell_results[cell] = {
            "coordinates": coordinates,
            "quartet_separation": separation,
            "fingerprint_null": null,
            "outside_union_damage": damage,
            "passes": bool(null["passes"] and damage["passes_at_most_25_percent"]),
        }
    return {
        "selected_seed": selected_seed,
        "reuse_cell_gates": reuse_cell_gates,
        "fingerprint_cells": cell_results,
        "passes": bool(
            all(value["passes"] for value in reuse_cell_gates.values())
            and all(value["passes"] for value in cell_results.values())
        ),
    }


def score_prediction_d(
    removal: RemovalEvaluation,
    test_suite: Mapping[str, object],
    instrument: Rung522Instrument,
    descriptors: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    selected_info = test_suite["selected_all_three"]
    selected = selected_info["evaluation"]
    if not isinstance(selected, CompactSwapEvaluation):
        raise RuntimeError("selected swap evaluation is absent for removal scoring")
    all_pairs = fingerprint_pairs(instrument.design, "test")
    results = {}
    for fold in (8, 9):
        pairs = _pairs_member_anchored_in_fold(all_pairs, instrument.data["folds"], fold)
        deterministic = score_removal_fold_without_null(
            removal,
            selected,
            pairs,
            instrument.row_to_local["test"],
            fold=fold,
        )
        null = fingerprint_null_distribution(
            removal.response,
            pairs,
            instrument.data,
            descriptors,
            "test",
            cell_id=f"removal:fold{fold}",
            fold_ids=instrument.data["folds"],
        )
        pair_counts = {
            target: {
                "count": int(value.members.numel()),
                "members_sha256": stage_a._tensor_sha256(value.members),
                "controls_sha256": stage_a._tensor_sha256(value.controls),
                "cross_fold_pair_count": int((
                    instrument.data["folds"][value.members // TOKENS]
                    != instrument.data["folds"][value.controls // TOKENS]
                ).sum()),
            }
            for target, value in pairs.items()
        }
        results[f"fold{fold}"] = {
            **deterministic,
            "fingerprint_null": null,
            "pair_census": pair_counts,
            "passes": bool(deterministic["deterministic_clauses_pass"] and null["passes"]),
        }
    return {
        "folds": results,
        "passes": all(value["passes"] for value in results.values()),
    }


FINGERPRINT_DEFINITION = {
    "namespace": "rung522-32-circuit-fingerprint-v1",
    "circuits": list(stage_a.FINGERPRINT_TAGS),
    "quartet": list(QUARTET_TAGS),
    "coordinate": "rms_member_delta_ce_minus_rms_matched_control_delta_ce",
    "separation": "minimum_quartet_coordinate_minus_maximum_nonquartet_coordinate",
    "null": "common_affine_response_permutation_within_frozen_coarse_strata",
    "replicates": 20_000,
}

TEST_SWEEP_PLAN = {
    "namespace": "rung522-one-way-test-sweep-v1",
    "order": [
        "native_test_capture_independent_replay_and_self_donor",
        "whole_attention8_comparator",
        "83_prediction_a_frames",
        "15_recovery_only_frames",
        "20_haar_frames",
        "one_validation_selected_all_three_frame",
        "score_final_a_b_and_prediction_c_from_saved_outputs",
        "conditional_complete_mean_centered_removal_if_a_b_c_pass",
        "close_test_forever",
    ],
    "post_test_fitting_allowed": False,
    "selection_after_test_allowed": False,
    "test_rows_per_model_call": EVALUATION_ROWS,
}


def _frame_artifacts(
    frames: Mapping[str, torch.Tensor],
    records: Mapping[str, FittedFrameRecord],
) -> tuple[archive.FrameArtifact, ...]:
    if set(frames) != set(state_guard.EXPECTED_FRAME_SPECS) or set(records) != set(frames):
        raise RuntimeError("cannot archive an incomplete frame/record census")
    artifacts = []
    for frame_id in sorted(frames):
        record = records[frame_id]
        spec = state_guard.EXPECTED_FRAME_SPECS[frame_id]
        if record.frame_id != frame_id:
            raise RuntimeError("frame record identifier differs from its archive key")
        artifacts.append(archive.FrameArtifact(
            spec=spec,
            frame=frames[frame_id],
            tensor_sha256=record.frame_sha256,
            fit_scheduler_payload=record.fit_scheduler_payload,
            validation_scheduler_payload=record.validation_scheduler_payload,
            fit_record_payload=record.fit_record_payload,
            health_record_payload=record.health_record_payload,
        ))
    return tuple(artifacts)


def _public_suite(value: Mapping[str, object]) -> dict[str, object]:
    """Drop only duplicate tensor objects; retain every scalar decision input/hash."""
    result = {
        key: item for key, item in value.items()
        if key not in {"real_frames", "label_null_frames"}
    }
    selected = result.get("selected_all_three")
    if isinstance(selected, Mapping):
        result["selected_all_three"] = {
            key: item for key, item in selected.items() if key != "evaluation"
        }
    return result


def _archive_ledger(ledger: CallLedger) -> archive.CallLedgerSnapshot:
    return archive.CallLedgerSnapshot(
        optimization_forward_events=ledger.optimization_forwards,
        optimization_backward_events=ledger.optimization_backwards,
        inference_forward_events=ledger.inference_forwards,
        inference_by_bucket=dict(ledger.inference_by_bucket),
        removal_inference_forward_events=ledger.removal_forwards,
    )


def _null_design_receipts(
    null_designs: Mapping[int, LabelNullDesign],
) -> dict[int, dict[str, object]]:
    return {
        seed: {
            "permutation_sha256": design.permutation_sha256,
            "identity_sha256": design.identity_sha256,
            "moved_nonzero_count": design.moved_nonzero_count,
            "maximum_possible_moved_nonzero_count": (
                design.maximum_possible_moved_nonzero_count
            ),
            "pair_identities": {
                target: design.pairs[target].identity() for target in QUARTET_TAGS
            },
        }
        for seed, design in sorted(null_designs.items())
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-directory", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args(argv)
    if os.environ.get("BQLIB_NO_MODEL") == "1":
        raise RuntimeError("BQLIB_NO_MODEL forbids rung522 scientific execution")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite scientific result: {args.output}")
    if args.work_directory.exists():
        raise FileExistsError(
            f"refusing to reuse scientific work directory: {args.work_directory}"
        )
    args.work_directory.mkdir(parents=True, exist_ok=False)
    started = time.time()

    data, design, preflight = stage_a.preflight()
    descriptors = design["descriptors"]
    null_designs = build_label_null_designs(data, descriptors)
    null_receipts = _null_design_receipts(null_designs)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32)
    ledger = CallLedger()
    protocol_state = state_guard.ProtocolState()
    instrument = Rung522Instrument(model, data, design, ledger, protocol_state)

    capture_pretest = instrument.capture_pretest_splits()
    fit_full = instrument.precompute_full_fit_d0()
    validation_full = instrument.evaluate_swap(
        "validation", frame=None, inference_bucket="full_attention8_comparator"
    )
    frames, records = train_all_registered_frames(
        instrument, validation_full, null_designs
    )

    frame_archive_path = args.work_directory / "frames_pretest.pt"
    loaded_archive = archive.write_frame_archive(
        frame_archive_path, _frame_artifacts(frames, records)
    )
    validation_suite = evaluate_validation_suite(
        instrument, frames, records, validation_full
    )
    provisional = validation_gates.evaluate_provisional_validation_gates(
        real=validation_suite["real"],
        recovery_only=validation_suite["recovery_only"],
        oracles=validation_suite["oracles"],
        haar_joint=validation_suite["haar_joint"],
        label_null_joint=validation_suite["label_null_joint"],
        real_frames=validation_suite["real_frames"],
        label_null_frames=validation_suite["label_null_frames"],
        label_null_fit_health=validation_suite["label_null_fit_health"],
        reserved_oracles=validation_suite["reserved_oracles"],
        all_three=validation_suite["all_three"],
    )
    ledger.assert_pretest_registered_price()
    public_validation = _public_suite(validation_suite)
    common_result: dict[str, object] = {
        "schema_version": 1,
        "rung": 522,
        "claim_level": (
            "held-out task-conditioned attention8 subspace extraction and selective manipulation"
        ),
        "registered_predictions": REGISTERED_PREDICTIONS,
        "checkpoint": checkpoint.__dict__,
        "dependency_sha256": _PREIMPORT_HASHES,
        "stage_a_preflight": preflight,
        "capture_pretest": capture_pretest,
        "fit_full_attention8": fit_full,
        "label_null_designs": null_receipts,
        "frame_archive": {
            "path": str(loaded_archive.path.resolve()),
            "file_sha256": loaded_archive.file_sha256,
            "content_sha256": loaded_archive.content_sha256,
            "frame_count": len(loaded_archive.frames),
        },
        "frame_health": {
            frame_id: {
                "healthy": record.healthy,
                "failures": list(record.health_failures),
                "frame_sha256": record.frame_sha256,
                "fit_record_sha256": record.fit_record_sha256,
                "health_record_sha256": record.health_record_sha256,
            }
            for frame_id, record in sorted(records.items())
        },
        "validation_outputs": public_validation,
        "provisional_validation_decision": asdict(provisional),
        "pretest_call_ledger": ledger.snapshot(),
    }

    if not provisional.pretest_passes:
        result = {
            **common_result,
            "status": "terminal_pretest_validation_failure",
            "test_opened": False,
            "test_closed": False,
            "pretest_manifest_created": False,
            "predictions": {
                "a": provisional.prediction_a_passes,
                "b": provisional.prediction_b_passes,
                "c": None,
                "d": None,
            },
            "execution_price": {
                **ledger.snapshot(),
                "runtime_seconds": time.time() - started,
            },
        }
        _atomic_json(args.output, result)
        print(json.dumps({
            "status": result["status"],
            "output": str(args.output),
            "test_opened": False,
            "prediction_a": provisional.prediction_a_passes,
            "prediction_b": provisional.prediction_b_passes,
        }, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()
        return result

    eligible = provisional.eligible_all_three_frame_ids
    preliminary_selected, _ = archive.geometry_only_grassmann_medoid(
        loaded_archive, eligible
    )
    fit_mu_q = instrument.fit_projection_mean(frames[preliminary_selected])
    archived_pretest_ledger = _archive_ledger(ledger)
    validation_evidence_path = args.work_directory / "validation_evidence.json"
    _atomic_json(validation_evidence_path, {
        "schema": archive.VALIDATION_EVIDENCE_SCHEMA,
        "validation_outputs": public_validation,
        "provisional_validation_decision": asdict(provisional),
        "call_ledger": asdict(archived_pretest_ledger),
    })
    pretest_manifest_path = args.work_directory / "pretest_manifest.json"
    manifest = archive.write_pretest_manifest(
        pretest_manifest_path,
        archive_path=frame_archive_path,
        null_hashes={seed: value.identity_sha256 for seed, value in null_designs.items()},
        validation_decisions=provisional,
        validation_provisional_gates_passed=provisional.pretest_passes,
        validation_evidence_path=validation_evidence_path,
        haar_hashes=validation_suite["haar_hashes"],
        eligible_all_three_frame_ids=eligible,
        fit_mu_q=fit_mu_q,
        fit_mu_q_source_split="FIT",
        call_ledger=archived_pretest_ledger,
        fingerprint_definition_sha256=_sha256_json(FINGERPRINT_DEFINITION),
        test_sweep_plan_sha256=_sha256_json(TEST_SWEEP_PLAN),
    )
    if manifest.selected_all_three_frame_id != preliminary_selected:
        raise RuntimeError("manifest medoid differs from the precomputed geometry-only medoid")

    instrument.state = manifest.protocol_state
    instrument.state.open_test_once()
    capture_test = instrument.capture_test_after_open()
    test_suite = evaluate_test_suite(
        instrument, frames, records, manifest.selected_all_three_frame_id
    )
    combined = combined_final_ab_inputs(validation_suite, test_suite)
    final_ab = validation_gates.evaluate_final_validation_test_gates(
        real=combined["real"],
        recovery_only=combined["recovery_only"],
        oracles=combined["oracles"],
        reserved_oracles=combined["reserved_oracles"],
        haar_joint=combined["haar_joint"],
        label_null_joint=combined["label_null_joint"],
        real_frames=combined["real_frames"],
        label_null_frames=combined["label_null_frames"],
    )
    prediction_c = score_prediction_c(
        test_suite, instrument, descriptors, manifest.selected_all_three_seed
    )
    removal = None
    prediction_d = None
    abc_pass = bool(
        final_ab.prediction_a_passes
        and final_ab.prediction_b_passes
        and prediction_c["passes"]
    )
    if abc_pass:
        removal = instrument.evaluate_removal(
            frames[manifest.selected_all_three_frame_id], fit_mu_q
        )
        prediction_d = score_prediction_d(
            removal, test_suite, instrument, descriptors
        )
    instrument.state.close_test()
    ledger.assert_final_registered_price(
        expected_removal_forwards=36 if removal is not None else 0
    )

    if not final_ab.prediction_a_passes or not final_ab.prediction_b_passes:
        status = "terminal_test_a_or_b_failure"
    elif not prediction_c["passes"]:
        status = "terminal_prediction_c_failure"
    elif prediction_d is not None and prediction_d["passes"]:
        status = "adoption_evidence_a_through_d_passed"
    else:
        status = "terminal_prediction_d_failure"
    result = {
        **common_result,
        "status": status,
        "test_opened": True,
        "test_closed": True,
        "pretest_manifest_created": True,
        "pretest_manifest": {
            "path": str(manifest.path.resolve()),
            "file_sha256": manifest.file_sha256,
            "selected_all_three_frame_id": manifest.selected_all_three_frame_id,
            "selected_all_three_seed": manifest.selected_all_three_seed,
        },
        "capture_test": capture_test,
        "test_outputs": _public_suite(test_suite),
        "final_validation_test_decision": asdict(final_ab),
        "prediction_c": prediction_c,
        "prediction_d": prediction_d,
        "removal": None if removal is None else {
            "forward_calls": removal.forward_calls,
            "minimum_edit_rms": removal.minimum_edit_rms,
            "response_sha256": removal.response_sha256,
        },
        "predictions": {
            "a": final_ab.prediction_a_passes,
            "b": final_ab.prediction_b_passes,
            "c": bool(prediction_c["passes"]),
            "d": None if prediction_d is None else bool(prediction_d["passes"]),
        },
        "execution_price": {
            **ledger.snapshot(),
            "runtime_seconds": time.time() - started,
        },
    }
    _atomic_json(args.output, result)
    print(json.dumps({
        "status": status,
        "output": str(args.output),
        "selected_frame": manifest.selected_all_three_frame_id,
        "prediction_a": result["predictions"]["a"],
        "prediction_b": result["predictions"]["b"],
        "prediction_c": result["predictions"]["c"],
        "prediction_d": result["predictions"]["d"],
        "inference_forwards": ledger.inference_forwards,
        "removal_forwards": ledger.removal_forwards,
    }, indent=2), flush=True)
    del model
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    main()
