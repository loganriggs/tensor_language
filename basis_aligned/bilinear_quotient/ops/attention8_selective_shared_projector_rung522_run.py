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
        rms = diagnostics["per_sequence_edit_rms"]
        if rms is None or bool((rms <= 0).any()):
            raise RuntimeError("projected intervention contains a dead sequence")
        local = self.row_to_local[split][rows]
        delta = _per_token_ce(logits, targets) - self.native_ce[split][local].to(self.device)
        return delta, rms


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
