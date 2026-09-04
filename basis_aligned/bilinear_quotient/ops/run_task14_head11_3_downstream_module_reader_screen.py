#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by this targeted runner.
"""Locate downstream module outputs that use the Task-14 head-11.3 effect.

The established intervention replaces head 3 before attention 11's output
projection with its natural donor value.  For each frozen downstream site, this
screen repeats that intervention but restores exactly that module's final-token
output to its native recipient value.  The signed score is

    head-only donor recovery - recovery after restoring the module output.

A positive score means the module's response amplified the transported number
signal; a negative score means it suppressed it.  This is a FIT-authority causal
screen, not held-out identification of a semantic subspace.

Registered predictions
----------------------
pred_a_native_replay: newly captured base/donor logits replay the immutable v2
    evidence to maximum absolute error <= 1e-4.  Failure is instrument-invalid.
pred_b_one_shared_amplifier: at least one frozen site has mean signed loss >=
    0.20 and loss >= 0.10 in every A1/A2 direction cell, while its incremental
    P and C movement is <= 0.10.
pred_c_no_strong_single_module: every frozen site has mean absolute A1/A2 cell
    effect <= 0.10.  This opposes pred_b and supports residual bypass or a
    distributed downstream computation; the gap is inconclusive.

The strong scientific null is a valid run where pred_c passes; the gap between
pred_b and pred_c is inconclusive. Frozen sites are MLP11 and attention/MLP
outputs in layers 12--17. Maximum price: 60 forward calls,
1,920 example evaluations, no backward calls or updates, and 13,312 retained
raw logit bytes.  GPU execution is only through ops/enqueue.sh.
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

import circuit_fast_screen_candidate_task14_agreement as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer


ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT / "circuits/fast_screens/task14_subject_verb_agreement_full_state_v2_result.json"
RESULT = ROOT / "circuits/followups/task14_head11_3_downstream_module_reader_screen_v1_result.json"
PARENT_SHA256 = "3c87e3973e1a7627f504ce26dfdaa3d7c48536f27a522e36c9e85741f09555c1"
AUTHORITY_SHA256 = "9b8ede7d17b0358467438b7f8fda7703bba1c93c9c594d55454404c1bb6e21cc"
PRIOR_ART_SHA256 = "12224e464ea1030f2ae421419f7bb1540b356068e386202a5919a108232cd3b2"
HEAD_SITE = "attn:11:head:03"
SITES = ("mlp:11",) + tuple(
    f"{kind}:{layer:02d}" for layer in range(12, 18) for kind in ("attn", "mlp")
)
BATCH_SIZE = 32
REPLAY_ATOL = 1.0e-4
AMPLIFIER_MEAN_MIN = 0.20
AMPLIFIER_CELL_MIN = 0.10
CONTROL_INCREMENT_MAX = 0.10
NO_SINGLE_MEAN_ABS_MAX = 0.10


class ReaderScreenError(ValueError):
    """Frozen evidence or execution violated the registered contract."""


class ReaderBackend(Protocol):
    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput: ...

    def induce_and_restore(
        self, batch: producer.ModelBatch, *, restore_site: str,
        donor_cache: Mapping[tuple[str, str], object],
        recipient_cache: Mapping[tuple[str, str], object],
    ) -> producer.BatchOutput: ...


class Task14ReaderTorchBackend(producer.Bilin18TorchBackend):
    """Reuse the exact producer forward with simultaneous head and module patches."""

    def induce_and_restore(
        self, batch: producer.ModelBatch, *, restore_site: str,
        donor_cache: Mapping[tuple[str, str], object],
        recipient_cache: Mapping[tuple[str, str], object],
    ) -> producer.BatchOutput:
        site = producer._site(restore_site)
        combined: dict[tuple[str, str], object] = {}
        for row_id in batch.row_ids:
            for key, source in (
                ((row_id, HEAD_SITE), donor_cache),
                ((row_id, restore_site), recipient_cache),
            ):
                if key not in source:
                    raise ReaderScreenError(f"activation cache lacks {key}")
                combined[key] = source[key]
        return self._forward(
            batch, capture=False, patch_site=site,
            patch_heads=(11, (3,)), donor_cache=combined,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> tuple[list[dict[str, object]], dict[str, object]]:
    if _sha256(PARENT) != PARENT_SHA256:
        raise ReaderScreenError("immutable v2 evidence hash changed")
    rows = candidate.build_rows()
    if candidate.validate_rows(rows) != AUTHORITY_SHA256:
        raise ReaderScreenError("Task 14 FIT authority changed")
    parent = json.loads(PARENT.read_text())
    if parent.get("authority_sha256") != AUTHORITY_SHA256 or parent.get("terminal") != "screen":
        raise ReaderScreenError("v2 evidence is not the expected passing parent")
    return rows, parent


def _batch(rows: Sequence[Mapping[str, object]], side: str) -> producer.ModelBatch:
    if side not in {"base", "donor"}:
        raise ReaderScreenError("side must be base or donor")
    return producer.ModelBatch(
        row_ids=tuple(str(row["row_id"]) for row in rows),
        side=side,  # type: ignore[arg-type]
        token_rows=tuple(tuple(int(x) for x in row[f"{side}_ids"]) for row in rows),
        answer_ids=tuple(int(row[f"{side}_answer_id"]) for row in rows),
        foil_ids=tuple(int(row[f"{side}_foil_id"]) for row in rows),
        semantic_positions=tuple(int(row[f"{side}_semantic_position"]) for row in rows),
    )


def _chunks(rows: Sequence[dict[str, object]]) -> tuple[list[dict[str, object]], ...]:
    return tuple(list(rows[start:start + BATCH_SIZE]) for start in range(0, len(rows), BATCH_SIZE))


def _pair(value: object) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(
        type(item) not in {int, float} or not math.isfinite(float(item)) for item in value
    ):
        raise ReaderScreenError("backend returned a malformed logit pair")
    return float(value[0]), float(value[1])


def _margin(pair: Sequence[float]) -> float:
    return float(pair[0]) - float(pair[1])


def _parent_maps(parent: Mapping[str, object]):
    run = parent.get("run")
    if not isinstance(run, dict):
        raise ReaderScreenError("parent lacks run evidence")
    native = {
        (str(item["row_id"]), str(item["side"])):
            (float(item["answer_logit"]), float(item["foil_logit"]))
        for item in run.get("native_logits", [])
    }
    head = {
        str(item["row_id"]): (float(item["answer_logit"]), float(item["foil_logit"]))
        for item in run.get("intervention_logits", [])
        if item.get("site", {}).get("site_id") == HEAD_SITE
    }
    if len(native) != 256 or len(head) != 128:
        raise ReaderScreenError("parent lacks exact native/head coverage")
    return native, head


def compile_dryrun() -> dict[str, object]:
    rows, _parent = _load()
    calls = 2 * math.ceil(len(rows) / BATCH_SIZE) + len(SITES) * math.ceil(len(rows) / BATCH_SIZE)
    return {
        "schema": "task14_head11_3_downstream_reader_dryrun_v1",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "authority_sha256": AUTHORITY_SHA256, "parent_sha256": PARENT_SHA256,
        "prior_art_sha256": PRIOR_ART_SHA256, "head_site": HEAD_SITE,
        "restore_sites": list(SITES),
        "maximum_price": {
            "forward_calls": calls, "example_evaluations": calls * BATCH_SIZE,
            "backward_calls": 0, "model_updates": 0,
            "raw_numeric_evidence_bytes": len(SITES) * len(rows) * 2 * 4,
        },
        "bars": {
            "native_replay_atol": REPLAY_ATOL,
            "shared_amplifier_mean_loss_min": AMPLIFIER_MEAN_MIN,
            "shared_amplifier_each_cell_loss_min": AMPLIFIER_CELL_MIN,
            "control_increment_max": CONTROL_INCREMENT_MAX,
            "no_single_module_mean_abs_max": NO_SINGLE_MEAN_ABS_MAX,
        },
    }


def _recovery(
    family: str, base: tuple[float, float], donor: tuple[float, float],
    changed: tuple[float, float], target_scale: float,
) -> float:
    base_margin, changed_margin = _margin(base), _margin(changed)
    if family in {"A1", "A2"}:
        return kernel.signed_pairwise_donor_recovery(
            -base_margin, _margin(donor), -changed_margin,
        )
    return (changed_margin - base_margin) / target_scale


def run_science(
    *, backend: ReaderBackend | None = None, device: str = "cuda", clock=time.perf_counter,
) -> dict[str, object]:
    rows, parent = _load()
    native_parent, head_parent = _parent_maps(parent)
    executor = backend if backend is not None else Task14ReaderTorchBackend.load(device)
    recipient_cache: dict[tuple[str, str], object] = {}
    donor_cache: dict[tuple[str, str], object] = {}
    replay_error = 0.0
    forwards = evaluations = 0
    started = clock()
    for side, cache in (("base", recipient_cache), ("donor", donor_cache)):
        for chunk in _chunks(rows):
            batch = _batch(chunk, side)
            output = executor.native(batch, capture=True)
            forwards += 1
            evaluations += len(chunk)
            cache.update(output.captured)
            if len(output.answer_foil) != len(chunk):
                raise ReaderScreenError("native output count differs from its batch")
            for row_id, observed in zip(batch.row_ids, output.answer_foil):
                expected = native_parent[(row_id, side)]
                replay_error = max(replay_error, *(abs(a - b) for a, b in zip(_pair(observed), expected)))
    required_donor = {(str(row["row_id"]), HEAD_SITE) for row in rows}
    required_recipient = {(str(row["row_id"]), site) for row in rows for site in SITES}
    if not required_donor.issubset(donor_cache) or not required_recipient.issubset(recipient_cache):
        raise ReaderScreenError("native capture lacks a required head or module output")

    target_denominators = [
        _margin(native_parent[(str(row["row_id"]), "donor")])
        + _margin(native_parent[(str(row["row_id"]), "base")])
        for row in rows if row["transform_id"] in {"A1", "A2"}
    ]
    target_scale = statistics.median(target_denominators)
    if target_scale <= kernel.MIN_DONOR_DENOMINATOR:
        raise ReaderScreenError("target effect scale is invalid")

    evidence = []
    summaries = []
    for site in SITES:
        restored: dict[str, tuple[float, float]] = {}
        for chunk in _chunks(rows):
            batch = _batch(chunk, "base")
            output = executor.induce_and_restore(
                batch, restore_site=site, donor_cache=donor_cache,
                recipient_cache=recipient_cache,
            )
            forwards += 1
            evaluations += len(chunk)
            if len(output.answer_foil) != len(chunk):
                raise ReaderScreenError("restored output count differs from its batch")
            restored.update({row_id: _pair(pair) for row_id, pair in zip(batch.row_ids, output.answer_foil)})
        cells: dict[str, list[float]] = {}
        controls: dict[str, list[float]] = {"P": [], "C": []}
        for row in rows:
            row_id, family = str(row["row_id"]), str(row["transform_id"])
            base, donor = native_parent[(row_id, "base")], native_parent[(row_id, "donor")]
            head_recovery = _recovery(family, base, donor, head_parent[row_id], target_scale)
            restored_recovery = _recovery(family, base, donor, restored[row_id], target_scale)
            loss = head_recovery - restored_recovery
            evidence.append({
                "site_id": site, "row_id": row_id, "family": family,
                "head_only_recovery": head_recovery,
                "restored_recovery": restored_recovery,
                "signed_recovery_loss": loss,
                "answer_logit": restored[row_id][0], "foil_logit": restored[row_id][1],
            })
            if family in {"A1", "A2"}:
                cells.setdefault(str(row["capability_cell_id"]), []).append(loss)
            else:
                controls[family].append(abs(loss))
        cell_means = {key: statistics.fmean(values) for key, values in sorted(cells.items())}
        mean_signed = statistics.fmean(cell_means.values())
        mean_abs = statistics.fmean(abs(value) for value in cell_means.values())
        control_means = {key: statistics.fmean(values) for key, values in controls.items()}
        summaries.append({
            "site_id": site, "target_cell_mean_signed_loss": cell_means,
            "mean_signed_target_loss": mean_signed,
            "mean_absolute_target_cell_effect": mean_abs,
            "control_mean_absolute_increment": control_means,
            "shared_amplifier": (
                mean_signed >= AMPLIFIER_MEAN_MIN
                and min(cell_means.values()) >= AMPLIFIER_CELL_MIN
                and max(control_means.values()) <= CONTROL_INCREMENT_MAX
            ),
        })

    instrument_valid = replay_error <= REPLAY_ATOL
    shared = [item["site_id"] for item in summaries if item["shared_amplifier"]]
    no_single = all(
        item["mean_absolute_target_cell_effect"] <= NO_SINGLE_MEAN_ABS_MAX
        for item in summaries
    )
    terminal = (
        "invalid" if not instrument_valid else
        "shared_amplifier_screen" if shared else
        "no_strong_single_module_null" if no_single else "inconclusive"
    )
    return {
        "schema": "task14_head11_3_downstream_reader_result_v1",
        "screen_tier_only": True, "execution_policy": "managed_queue_only",
        "authority_sha256": AUTHORITY_SHA256, "parent_sha256": PARENT_SHA256,
        "prior_art_sha256": PRIOR_ART_SHA256, "terminal": terminal,
        "predictions": {
            "pred_a_native_replay": instrument_valid,
            "pred_b_one_shared_amplifier": bool(shared),
            "pred_c_no_strong_single_module": no_single,
        },
        "native_replay_max_abs_error": replay_error,
        "target_scale": target_scale, "shared_amplifier_sites": shared,
        "site_summaries": summaries, "evidence": evidence,
        "active_price": {
            "forward_calls": forwards, "example_evaluations": evaluations,
            "backward_calls": 0, "model_updates": 0,
            "raw_numeric_evidence_bytes": len(SITES) * len(rows) * 2 * 4,
        },
        "serial_seconds": clock() - started,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise ReaderScreenError(f"{name} must be absent or exactly 1")
    if args.dry_run or any(os.environ.get(name) == "1" for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL")):
        print(json.dumps(compile_dryrun(), sort_keys=True))
        return
    if RESULT.exists():
        raise ReaderScreenError(f"refusing to overwrite existing result: {RESULT}")
    result = run_science()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({key: result[key] for key in ("terminal", "predictions", "active_price", "serial_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
