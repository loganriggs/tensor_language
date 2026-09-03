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
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
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
        "7333c3f8ae07c5469c6e2159583db73b8f779be29d6b2de4f8f1f51c0a4e4679",
    OPS / "attention8_selective_shared_projector_rung522_math.py":
        "6cff6f7726dd8f76e786d64abf913fc31adbdfec101a97741a1aa3396f8431c2",
    OPS / "attention8_selective_shared_projector_rung522_scheduler.py":
        "d840318d5b675ce762f6c9a0d451c11550c6520b97ffbb762672ec703af5540f",
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
import bilin18_observed_model_facade as facade  # noqa: E402


@dataclass
class CallLedger:
    optimization_forwards: int = 0
    optimization_backwards: int = 0
    inference_forwards: int = 0
    removal_forwards: int = 0

    def charge(self, kind: str, count: int = 1) -> None:
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
            self.inference_forwards += count
            if self.inference_forwards > INFERENCE_FORWARD_CEILING:
                raise RuntimeError("rung522 inference-forward ceiling exceeded")
        elif kind == "removal_forward":
            self.removal_forwards += count
            if self.removal_forwards > REMOVAL_FORWARD_CEILING:
                raise RuntimeError("rung522 removal-forward ceiling exceeded")
        else:
            raise ValueError(f"unknown ledger kind {kind!r}")


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
            self.ledger.charge("inference_forward")
            self.state.record_inference_events(1)
            if captured is None:
                raise RuntimeError("attention8 native capture is absent")
            replay, _, _ = _execute(self.model, tokens)
            self.ledger.charge("inference_forward")
            self.state.record_inference_events(1)
            replay_exact &= bool(torch.equal(logits, replay))
            if start == 0:
                native_write = captured.to(self.device)
                self_logits, _, self_diag = _execute(
                    self.model,
                    tokens,
                    edit=lambda _write, value=native_write: value,
                )
                self.ledger.charge("inference_forward")
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
            self.ledger.charge("inference_forward")
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.data["rows"][rows, :TOKENS].to(self.device)
        targets = self.data["rows"][rows, 1 : TOKENS + 1].to(self.device)
        donor = self.donor_writes(split, rows, donor_map).to(self.device)

        def edit(write: torch.Tensor) -> torch.Tensor:
            return core.daslib.projection_interchange(write, donor, frame, validate=False)

        logits, _, diagnostics = _execute(self.model, tokens, edit=edit)
        self.ledger.charge("optimization_forward" if optimization else "inference_forward")
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
    orthonormality = float(core.daslib.orthonormality_error(final))
    distance = float(core.daslib.projector_frobenius_distance(initial.cpu(), final))
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
    record = FittedFrameRecord(
        frame_id=spec.frame_id,
        frame_sha256=stage_a._tensor_sha256(final),
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


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-directory", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args(argv)
    if os.environ.get("BQLIB_NO_MODEL") == "1":
        raise RuntimeError("BQLIB_NO_MODEL forbids rung522 scientific execution")
    # The implementation is intentionally fail-closed until the training,
    # pretest-freeze, evaluation, and scoring stages below are complete and
    # independently tested.  This explicit stop prevents a partial runner from
    # reaching the model merely because it parses and passes the static gate.
    raise RuntimeError(
        "RUNG522 SCIENCE CLOSED: full 103-frame training and TEST-seal stages are under implementation"
    )


if __name__ == "__main__":
    main()
