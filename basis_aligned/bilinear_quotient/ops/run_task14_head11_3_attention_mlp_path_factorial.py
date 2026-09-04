#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by this targeted runner.
"""Downstream attention-path x MLP-path factorial after inducing head 11.3.

Let F(S) be signed donor recovery after donor-patching head 11.3 and restoring
all recipient-native final-token outputs in site group S.  The four corners are
none, MLP11--17, attention12--17, and both.  Their exact interaction is

 I = F(both) - F(MLP path) - F(attention path) + F(none).

Registered predictions: (A) native replay error <=1e-4, otherwise invalid;
(B) nonlinear cross-path grouping if |I|>=0.10 in at least two A1/A2 direction
cells and every P/C mean absolute interaction or group loss <=0.10; (C)
additive-or-direct use if |I|<=0.03 in every target cell and every P/C mean
absolute interaction or group loss <=0.03.
Within C, mean absolute group loss >=0.10 identifies additive path use, whereas
both group losses <=0.05 identifies direct residual readout.  The gap is
explicitly inconclusive.  Maximum new price: 20 forwards, 640 examples, zero
backwards/updates, and 3,072 retained raw logit bytes. Managed queue only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Mapping, Protocol, Sequence

import circuit_fast_screen_producer as producer
import run_task14_head11_3_downstream_module_reader_screen as parent_reader


ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "circuits/followups/task14_head11_3_attention_mlp_path_factorial_v1_result.json"
SINGLE_RESULT = ROOT / "circuits/followups/task14_head11_3_downstream_module_reader_screen_v1_result.json"
SINGLE_RESULT_SHA256 = "677e5e2eccdc13fcc7bd2053be5d9c450d2af22d36d20760334d9bd2f50a0ffa"
PRIOR_ART_SHA256 = "99a021437ab31ed49f08dcc5df34716efe824b23f4cf63d353bf3cce5a735c9e"
MLP_SITES = tuple(f"mlp:{layer:02d}" for layer in range(11, 18))
ATTENTION_SITES = tuple(f"attn:{layer:02d}" for layer in range(12, 18))
ARMS = {"mlp_path": MLP_SITES, "attention_path": ATTENTION_SITES,
        "both": MLP_SITES + ATTENTION_SITES}
REPLAY_ATOL = 1e-4
GROUPED_ABS_MIN = 0.10
GROUPED_MIN_CELLS = 2
GROUPED_CONTROL_MAX = 0.10
ADDITIVE_ABS_MAX = 0.03
ADDITIVE_CONTROL_MAX = 0.03
PATH_USE_MEAN_ABS_MIN = 0.10
DIRECT_READ_MEAN_ABS_MAX = 0.05


class PathFactorialError(ValueError):
    """Frozen evidence or path intervention violated the contract."""


class PathBackend(Protocol):
    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput: ...
    def induce_and_restore(self, batch: producer.ModelBatch, *, restore_sites: Sequence[str],
                           donor_cache: Mapping[tuple[str, str], object],
                           recipient_cache: Mapping[tuple[str, str], object]) -> producer.BatchOutput: ...


def _restore_output(backend, batch, site, recipient_cache, output):
    """Apply the exact output hook shape for one frozen module site."""
    kind = site.split(":", 1)[0]
    if kind == "mlp":
        return backend._replace(output, batch, site, recipient_cache)
    if kind == "attn" and isinstance(output, tuple) and len(output) >= 2:
        changed = backend._replace(output[0], batch, site, recipient_cache)
        return (changed,) + output[1:]
    raise PathFactorialError("module hook output does not match its frozen site kind")


class Task14PathTorchBackend(parent_reader.Task14ReaderTorchBackend):
    """Restore any frozen module group using temporary hooks around the exact forward."""

    def induce_and_restore(self, batch, *, restore_sites, donor_cache, recipient_cache):
        selected = tuple(restore_sites)
        if not selected or len(selected) != len(set(selected)) or any(
            site not in MLP_SITES + ATTENTION_SITES for site in selected
        ):
            raise PathFactorialError("restore group is empty, duplicate, or outside the frozen paths")
        head_cache = {}
        for row_id in batch.row_ids:
            key = (row_id, parent_reader.HEAD_SITE)
            if key not in donor_cache:
                raise PathFactorialError(f"donor cache lacks {key}")
            head_cache[key] = donor_cache[key]
        handles = []
        for site in selected:
            kind, layer_text = site.split(":")
            module = getattr(self.model.transformer.h[int(layer_text)], kind)
            def restore(_module, _arguments, output, *, frozen_site=site):
                return _restore_output(self, batch, frozen_site, recipient_cache, output)
            handles.append(module.register_forward_hook(restore))
        try:
            return self._forward(batch, capture=False, patch_heads=(11, (3,)), donor_cache=head_cache)
        finally:
            for handle in handles:
                handle.remove()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load():
    rows, parent = parent_reader._load()
    if _sha256(SINGLE_RESULT) != SINGLE_RESULT_SHA256:
        raise PathFactorialError("single-module screen hash changed")
    single = json.loads(SINGLE_RESULT.read_text())
    if single.get("authority_sha256") != parent_reader.AUTHORITY_SHA256:
        raise PathFactorialError("single-module evidence has wrong authority")
    native, head = parent_reader._parent_maps(parent)
    return rows, native, head


def compile_dryrun():
    rows, _native, _head = _load()
    calls = (2 + len(ARMS)) * len(parent_reader._chunks(rows))
    return {
        "schema": "task14_head11_3_attention_mlp_path_factorial_dryrun_v1",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "authority_sha256": parent_reader.AUTHORITY_SHA256,
        "single_result_sha256": SINGLE_RESULT_SHA256, "prior_art_sha256": PRIOR_ART_SHA256,
        "equation": "I=F(both)-F(MLP_path)-F(attention_path)+F(none)",
        "arms": {key: list(value) for key, value in ARMS.items()},
        "maximum_new_price": {"forward_calls": calls,
                              "example_evaluations": calls * parent_reader.BATCH_SIZE,
                              "backward_calls": 0, "model_updates": 0,
                              "raw_numeric_evidence_bytes": len(ARMS) * len(rows) * 2 * 4},
        "bars": {"native_replay_atol": REPLAY_ATOL, "grouped_abs_min": GROUPED_ABS_MIN,
                 "grouped_min_cells": GROUPED_MIN_CELLS,
                 "grouped_all_control_terms_max": GROUPED_CONTROL_MAX,
                 "additive_abs_max": ADDITIVE_ABS_MAX,
                 "additive_all_control_terms_max": ADDITIVE_CONTROL_MAX,
                 "path_use_mean_abs_min": PATH_USE_MEAN_ABS_MIN,
                 "direct_read_mean_abs_max": DIRECT_READ_MEAN_ABS_MAX},
    }


def run_science(*, backend: PathBackend | None = None, device="cuda", clock=time.perf_counter):
    rows, native, head = _load()
    executor = backend if backend is not None else Task14PathTorchBackend.load(device)
    recipient_cache, donor_cache = {}, {}
    replay_error = 0.0
    forwards = evaluations = 0
    started = clock()
    for side, cache in (("base", recipient_cache), ("donor", donor_cache)):
        for chunk in parent_reader._chunks(rows):
            batch = parent_reader._batch(chunk, side)
            output = executor.native(batch, capture=True)
            forwards += 1; evaluations += len(chunk)
            if len(output.answer_foil) != len(chunk):
                raise PathFactorialError("native output count differs from batch")
            cache.update(output.captured)
            for row_id, observed in zip(batch.row_ids, output.answer_foil):
                replay_error = max(replay_error, *(
                    abs(a-b) for a,b in zip(parent_reader._pair(observed), native[(row_id, side)])
                ))
    required = {(str(row["row_id"]), site) for row in rows
                for site in MLP_SITES + ATTENTION_SITES}
    if not required.issubset(recipient_cache):
        raise PathFactorialError("recipient capture lacks a frozen path output")
    arm_pairs = {}
    for arm, sites in ARMS.items():
        arm_pairs[arm] = {}
        for chunk in parent_reader._chunks(rows):
            batch = parent_reader._batch(chunk, "base")
            output = executor.induce_and_restore(batch, restore_sites=sites,
                                                 donor_cache=donor_cache,
                                                 recipient_cache=recipient_cache)
            forwards += 1; evaluations += len(chunk)
            if len(output.answer_foil) != len(chunk):
                raise PathFactorialError("path-arm output count differs from batch")
            arm_pairs[arm].update({rid: parent_reader._pair(pair)
                                   for rid, pair in zip(batch.row_ids, output.answer_foil)})
    scale = statistics.median(
        parent_reader._margin(native[(str(row["row_id"]), "donor")])
        + parent_reader._margin(native[(str(row["row_id"]), "base")])
        for row in rows if row["transform_id"] in {"A1", "A2"})
    by_cell, controls, evidence = {}, {"P": [], "C": []}, []
    for row in rows:
        rid, family = str(row["row_id"]), str(row["transform_id"])
        args = family, native[(rid, "base")], native[(rid, "donor")]
        f = {"none": parent_reader._recovery(*args, head[rid], scale)}
        f.update({arm: parent_reader._recovery(*args, pairs[rid], scale)
                  for arm, pairs in arm_pairs.items()})
        interaction = f["both"] - f["mlp_path"] - f["attention_path"] + f["none"]
        losses = {"mlp_path_loss": f["none"]-f["mlp_path"],
                  "attention_path_loss": f["none"]-f["attention_path"],
                  "both_paths_loss": f["none"]-f["both"]}
        evidence.append({"row_id": rid, "family": family, **f, **losses,
                         "interaction": interaction})
        if family in {"A1", "A2"}:
            by_cell.setdefault(str(row["capability_cell_id"]), []).append((interaction, losses))
        else:
            controls[family].append((interaction, losses))
    cells = {}
    for cell, records in sorted(by_cell.items()):
        cells[cell] = {"interaction": statistics.fmean(x[0] for x in records)}
        for key in ("mlp_path_loss", "attention_path_loss", "both_paths_loss"):
            cells[cell][key] = statistics.fmean(x[1][key] for x in records)
    control_means = {}
    for family, records in controls.items():
        control_means[family] = {
            "interaction": statistics.fmean(abs(x[0]) for x in records),
            **{key: statistics.fmean(abs(x[1][key]) for x in records)
               for key in ("mlp_path_loss", "attention_path_loss", "both_paths_loss")},
        }
    maximum_control_term = max(value for family in control_means.values() for value in family.values())
    valid = replay_error <= REPLAY_ATOL
    grouped = (sum(abs(x["interaction"]) >= GROUPED_ABS_MIN for x in cells.values())
               >= GROUPED_MIN_CELLS and maximum_control_term <= GROUPED_CONTROL_MAX)
    additive = (max(abs(x["interaction"]) for x in cells.values()) <= ADDITIVE_ABS_MAX
                and maximum_control_term <= ADDITIVE_CONTROL_MAX)
    mean_path_abs = {key: statistics.fmean(abs(x[key]) for x in cells.values())
                     for key in ("mlp_path_loss", "attention_path_loss")}
    subtype = None
    if additive:
        subtype = ("additive_path_use" if max(mean_path_abs.values()) >= PATH_USE_MEAN_ABS_MIN
                   else "direct_residual_read" if max(mean_path_abs.values()) <= DIRECT_READ_MEAN_ABS_MAX
                   else "weak_additive_path_use")
    terminal = "invalid" if not valid else (
        "nonlinear_cross_path_screen" if grouped else "additive_or_direct_null" if additive else "inconclusive")
    return {"schema": "task14_head11_3_attention_mlp_path_factorial_result_v1",
            "screen_tier_only": True, "execution_policy": "managed_queue_only",
            "authority_sha256": parent_reader.AUTHORITY_SHA256,
            "single_result_sha256": SINGLE_RESULT_SHA256,
            "prior_art_sha256": PRIOR_ART_SHA256, "terminal": terminal,
            "predictions": {"pred_a_native_replay": valid,
                            "pred_b_nonlinear_cross_path_grouping": grouped,
                            "pred_c_additive_or_direct_use": additive},
            "additive_subtype": subtype, "native_replay_max_abs_error": replay_error,
            "target_scale": scale, "target_cells": cells,
            "target_mean_absolute_path_loss": mean_path_abs,
            "control_mean_absolute_terms": control_means, "evidence": evidence,
            "active_new_price": {"forward_calls": forwards, "example_evaluations": evaluations,
                                 "backward_calls": 0, "model_updates": 0,
                                 "raw_numeric_evidence_bytes": len(ARMS)*len(rows)*2*4},
            "serial_seconds": clock()-started}


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise PathFactorialError(f"{name} must be absent or exactly 1")
    if args.dry_run or any(os.environ.get(n)=="1" for n in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL")):
        print(json.dumps(compile_dryrun(), sort_keys=True)); return
    if RESULT.exists(): raise PathFactorialError(f"refusing to overwrite {RESULT}")
    result = run_science(); RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":"))+"\n")
    print(json.dumps({k: result[k] for k in ("terminal","predictions","active_new_price","serial_seconds")}, sort_keys=True))


if __name__ == "__main__": main()
