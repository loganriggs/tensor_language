#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by this targeted runner.
"""Exact MLP11 x MLP12 restoration factorial after inducing Task-14 head 11.3.

Let F(S) be signed donor recovery when head 11.3 is donor-patched and the
final-token outputs of modules in S are restored to their native recipient
values.  The immutable downstream-reader screen supplies F(empty), F({M11}),
and F({M12}); this run opens only F({M11,M12}).  The exact interaction is

    I = F({M11,M12}) - F({M11}) - F({M12}) + F(empty).

Negative I means the joint restoration removes more recovery than the two
individual restorations additively predict; positive I means redundancy.

Registered predictions: (A) newly captured native logits replay the frozen
parent within 1e-4, otherwise invalid; (B) nonlinear grouping if |I| >= 0.10
in at least two of four A1/A2 direction cells with P/C mean absolute interaction
<= 0.10; (C) additive use if |I| <= 0.03 in all four target cells and P/C mean
absolute interaction <= 0.03.  A valid result satisfying neither B nor C is
explicitly inconclusive.  Maximum new price is 12 forwards, 384 examples, zero
backwards/updates, and 1,024 retained raw logit bytes. Managed queue only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Mapping, Protocol, Sequence

import circuit_fast_screen_producer as producer
import run_task14_head11_3_downstream_module_reader_screen as parent_reader


ROOT = Path(__file__).resolve().parent.parent
READER_RESULT = ROOT / "circuits/followups/task14_head11_3_downstream_module_reader_screen_v1_result.json"
RESULT = ROOT / "circuits/followups/task14_head11_3_mlp11_mlp12_reader_factorial_v1_result.json"
READER_RESULT_SHA256 = "677e5e2eccdc13fcc7bd2053be5d9c450d2af22d36d20760334d9bd2f50a0ffa"
PRIOR_ART_SHA256 = "b3a4cea63ca468146f511c2950e09ec4b32cb72a71ebc5320c8f2c0ebfd1298e"
RESTORE_SITES = ("mlp:11", "mlp:12")
REPLAY_ATOL = 1.0e-4
GROUPED_CELL_ABS_MIN = 0.10
GROUPED_MIN_CELL_COUNT = 2
GROUPED_CONTROL_MAX = 0.10
ADDITIVE_CELL_ABS_MAX = 0.03
ADDITIVE_CONTROL_MAX = 0.03


class FactorialError(ValueError):
    """The frozen four-corner contract or execution is invalid."""


class FactorialBackend(Protocol):
    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput: ...

    def induce_and_restore_both(
        self, batch: producer.ModelBatch, *, donor_cache: Mapping[tuple[str, str], object],
        recipient_cache: Mapping[tuple[str, str], object],
    ) -> producer.BatchOutput: ...


class Task14MlpFactorialTorchBackend(parent_reader.Task14ReaderTorchBackend):
    """Install two temporary output hooks around the existing exact forward."""

    def induce_and_restore_both(self, batch, *, donor_cache, recipient_cache):
        combined = {}
        for row_id in batch.row_ids:
            key = (row_id, parent_reader.HEAD_SITE)
            if key not in donor_cache:
                raise FactorialError(f"donor cache lacks {key}")
            combined[key] = donor_cache[key]
        handles = []
        for site in RESTORE_SITES:
            layer = int(site.split(":")[1])

            def restore(_module, _arguments, output, *, frozen_site=site):
                return self._replace(output, batch, frozen_site, recipient_cache)

            handles.append(self.model.transformer.h[layer].mlp.register_forward_hook(restore))
        try:
            return self._forward(
                batch, capture=False, patch_heads=(11, (3,)), donor_cache=combined,
            )
        finally:
            for handle in handles:
                handle.remove()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load():
    rows, v2 = parent_reader._load()
    if _sha256(READER_RESULT) != READER_RESULT_SHA256:
        raise FactorialError("immutable downstream-reader result hash changed")
    reader = json.loads(READER_RESULT.read_text())
    if reader.get("schema") != "task14_head11_3_downstream_reader_result_v1" \
            or reader.get("authority_sha256") != parent_reader.AUTHORITY_SHA256 \
            or reader.get("parent_sha256") != parent_reader.PARENT_SHA256:
        raise FactorialError("downstream-reader result identity changed")
    native, head = parent_reader._parent_maps(v2)
    singles = {site: {} for site in RESTORE_SITES}
    for item in reader.get("evidence", []):
        site = item.get("site_id")
        if site in singles:
            singles[site][str(item["row_id"])] = (
                float(item["answer_logit"]), float(item["foil_logit"]),
            )
    if any(len(values) != len(rows) for values in singles.values()):
        raise FactorialError("reader parent lacks an exact single-restoration corner")
    return rows, native, head, singles


def compile_dryrun() -> dict[str, object]:
    rows, _native, _head, _singles = _load()
    calls = 3 * math.ceil(len(rows) / parent_reader.BATCH_SIZE)
    return {
        "schema": "task14_head11_3_mlp11_mlp12_factorial_dryrun_v1",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "authority_sha256": parent_reader.AUTHORITY_SHA256,
        "reader_result_sha256": READER_RESULT_SHA256,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "equation": "I=F({M11,M12})-F({M11})-F({M12})+F(empty)",
        "maximum_new_price": {
            "forward_calls": calls, "example_evaluations": calls * parent_reader.BATCH_SIZE,
            "backward_calls": 0, "model_updates": 0,
            "raw_numeric_evidence_bytes": len(rows) * 2 * 4,
        },
        "bars": {
            "native_replay_atol": REPLAY_ATOL,
            "grouped_cell_abs_min": GROUPED_CELL_ABS_MIN,
            "grouped_min_cell_count": GROUPED_MIN_CELL_COUNT,
            "grouped_control_max": GROUPED_CONTROL_MAX,
            "additive_cell_abs_max": ADDITIVE_CELL_ABS_MAX,
            "additive_control_max": ADDITIVE_CONTROL_MAX,
        },
    }


def run_science(*, backend: FactorialBackend | None = None, device="cuda", clock=time.perf_counter):
    rows, native, head, singles = _load()
    executor = backend if backend is not None else Task14MlpFactorialTorchBackend.load(device)
    recipient_cache, donor_cache = {}, {}
    replay_error = 0.0
    forwards = evaluations = 0
    started = clock()
    for side, cache in (("base", recipient_cache), ("donor", donor_cache)):
        for chunk in parent_reader._chunks(rows):
            batch = parent_reader._batch(chunk, side)
            output = executor.native(batch, capture=True)
            forwards += 1
            evaluations += len(chunk)
            if len(output.answer_foil) != len(chunk):
                raise FactorialError("native output count differs from its batch")
            cache.update(output.captured)
            for row_id, observed in zip(batch.row_ids, output.answer_foil):
                replay_error = max(replay_error, *(
                    abs(a - b) for a, b in zip(parent_reader._pair(observed), native[(row_id, side)])
                ))
    required = {(str(row["row_id"]), site) for row in rows for site in RESTORE_SITES}
    if not required.issubset(recipient_cache):
        raise FactorialError("recipient capture lacks MLP11 or MLP12")
    both = {}
    for chunk in parent_reader._chunks(rows):
        batch = parent_reader._batch(chunk, "base")
        output = executor.induce_and_restore_both(
            batch, donor_cache=donor_cache, recipient_cache=recipient_cache,
        )
        forwards += 1
        evaluations += len(chunk)
        if len(output.answer_foil) != len(chunk):
            raise FactorialError("joint output count differs from its batch")
        both.update({row_id: parent_reader._pair(pair) for row_id, pair in zip(batch.row_ids, output.answer_foil)})

    scale = statistics.median(
        parent_reader._margin(native[(str(row["row_id"]), "donor")])
        + parent_reader._margin(native[(str(row["row_id"]), "base")])
        for row in rows if row["transform_id"] in {"A1", "A2"}
    )
    cells, controls, evidence = {}, {"P": [], "C": []}, []
    for row in rows:
        row_id, family = str(row["row_id"]), str(row["transform_id"])
        args = family, native[(row_id, "base")], native[(row_id, "donor")]
        values = {
            "none": parent_reader._recovery(*args, head[row_id], scale),
            "mlp11": parent_reader._recovery(*args, singles["mlp:11"][row_id], scale),
            "mlp12": parent_reader._recovery(*args, singles["mlp:12"][row_id], scale),
            "both": parent_reader._recovery(*args, both[row_id], scale),
        }
        interaction = values["both"] - values["mlp11"] - values["mlp12"] + values["none"]
        evidence.append({"row_id": row_id, "family": family, **values,
                         "interaction": interaction,
                         "both_answer_logit": both[row_id][0], "both_foil_logit": both[row_id][1]})
        if family in {"A1", "A2"}:
            cells.setdefault(str(row["capability_cell_id"]), []).append(interaction)
        else:
            controls[family].append(abs(interaction))
    cell_means = {key: statistics.fmean(values) for key, values in sorted(cells.items())}
    control_means = {key: statistics.fmean(values) for key, values in controls.items()}
    valid = replay_error <= REPLAY_ATOL
    grouped = (
        sum(abs(value) >= GROUPED_CELL_ABS_MIN for value in cell_means.values())
        >= GROUPED_MIN_CELL_COUNT and max(control_means.values()) <= GROUPED_CONTROL_MAX
    )
    additive = (
        max(abs(value) for value in cell_means.values()) <= ADDITIVE_CELL_ABS_MAX
        and max(control_means.values()) <= ADDITIVE_CONTROL_MAX
    )
    terminal = "invalid" if not valid else (
        "nonlinear_grouping_screen" if grouped else "additive_null" if additive else "inconclusive"
    )
    return {
        "schema": "task14_head11_3_mlp11_mlp12_factorial_result_v1",
        "screen_tier_only": True, "execution_policy": "managed_queue_only",
        "authority_sha256": parent_reader.AUTHORITY_SHA256,
        "reader_result_sha256": READER_RESULT_SHA256,
        "prior_art_sha256": PRIOR_ART_SHA256, "terminal": terminal,
        "predictions": {"pred_a_native_replay": valid,
                        "pred_b_nonlinear_grouping": grouped,
                        "pred_c_additive_use": additive},
        "native_replay_max_abs_error": replay_error, "target_scale": scale,
        "target_cell_mean_interaction": cell_means,
        "control_mean_absolute_interaction": control_means, "evidence": evidence,
        "active_new_price": {"forward_calls": forwards, "example_evaluations": evaluations,
                             "backward_calls": 0, "model_updates": 0,
                             "raw_numeric_evidence_bytes": len(rows) * 2 * 4},
        "serial_seconds": clock() - started,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise FactorialError(f"{name} must be absent or exactly 1")
    if args.dry_run or any(os.environ.get(name) == "1" for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL")):
        print(json.dumps(compile_dryrun(), sort_keys=True))
        return
    if RESULT.exists():
        raise FactorialError(f"refusing to overwrite existing result: {RESULT}")
    result = run_science()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({key: result[key] for key in ("terminal", "predictions", "active_new_price", "serial_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
