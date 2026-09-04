#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by this targeted runner.
"""Early-MLP x late-MLP restoration factorial after inducing head 11.3.

F(S) is signed donor recovery after head-11.3 donor patching and restoration
of recipient-native MLP outputs in S.  Immutable results supply F(empty),
F(MLP11--12), and F(MLP11--17); only F(MLP13--17) is newly evaluated.

 I = F(all) - F(early) - F(late) + F(empty).

Registered predictions: (A) native replay <=1e-4, otherwise invalid; (B)
compensating/nonlinear use if |I|>=0.10 in at least two A1/A2 direction cells;
(C) additive split if |I|<=0.03 in every target cell and both early and late
groups have mean absolute target loss >=0.05.  Every scientific terminal also
requires every P/C corner effect, group loss, and interaction <=0.10.  A valid
result satisfying neither B nor C is inconclusive.  Maximum new price: 12
forwards, 384 examples, zero backwards/updates, 1,024 retained logit bytes.
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
import run_task14_head11_3_attention_mlp_path_factorial as path_parent
import run_task14_head11_3_downstream_module_reader_screen as reader


ROOT = Path(__file__).resolve().parent.parent
PAIR_RESULT = ROOT / "circuits/followups/task14_head11_3_mlp11_mlp12_reader_factorial_v1_result.json"
PATH_RESULT = ROOT / "circuits/followups/task14_head11_3_attention_mlp_path_factorial_v1_result.json"
RESULT = ROOT / "circuits/followups/task14_head11_3_early_late_mlp_factorial_v1_result.json"
PAIR_SHA256 = "ec0c1e0e2e7091f4e4441ca23257f63d39d18f922c6a1009c08842af5e10269d"
PATH_SHA256 = "dd27508472e862a01485829ebb6e8d398c1be56434e0708ae09d03794452c8a8"
PRIOR_ART_SHA256 = "b42867dec1d6db189fcd9bfa236654a34a0d4e36ddd78d0ebdbfd0331c6d2df7"
EARLY = ("mlp:11", "mlp:12")
LATE = tuple(f"mlp:{layer:02d}" for layer in range(13, 18))
REPLAY_ATOL = 1e-4
INTERACTION_STRONG = 0.10
INTERACTION_STRONG_CELLS = 2
INTERACTION_ADDITIVE = 0.03
ADDITIVE_GROUP_LOSS_MIN = 0.05
CONTROL_TERM_MAX = 0.10


class SplitError(ValueError):
    """Frozen evidence or execution violated the early/late split contract."""


class SplitBackend(Protocol):
    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput: ...
    def induce_and_restore(self, batch: producer.ModelBatch, *, restore_sites: Sequence[str],
                           donor_cache: Mapping[tuple[str, str], object],
                           recipient_cache: Mapping[tuple[str, str], object]) -> producer.BatchOutput: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load():
    rows, native, head = path_parent._load()
    if _sha256(PAIR_RESULT) != PAIR_SHA256 or _sha256(PATH_RESULT) != PATH_SHA256:
        raise SplitError("immutable parent result hash changed")
    pair, path = json.loads(PAIR_RESULT.read_text()), json.loads(PATH_RESULT.read_text())
    if pair.get("authority_sha256") != reader.AUTHORITY_SHA256 \
            or path.get("authority_sha256") != reader.AUTHORITY_SHA256:
        raise SplitError("parent result authority changed")
    pair_rows = {str(x["row_id"]): x for x in pair.get("evidence", [])}
    path_rows = {str(x["row_id"]): x for x in path.get("evidence", [])}
    if len(pair_rows) != len(rows) or len(path_rows) != len(rows):
        raise SplitError("parent results lack exact row coverage")
    prior = {}
    for row in rows:
        rid = str(row["row_id"])
        if abs(float(pair_rows[rid]["none"]) - float(path_rows[rid]["none"])) > 1e-12:
            raise SplitError("parent head-only corners disagree")
        prior[rid] = {"empty": float(pair_rows[rid]["none"]),
                      "early": float(pair_rows[rid]["both"]),
                      "all": float(path_rows[rid]["mlp_path"])}
    return rows, native, head, prior


def compile_dryrun():
    rows, _native, _head, _prior = _load()
    calls = 3 * len(reader._chunks(rows))
    return {"schema": "task14_head11_3_early_late_mlp_factorial_dryrun_v1",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "authority_sha256": reader.AUTHORITY_SHA256,
            "pair_result_sha256": PAIR_SHA256, "path_result_sha256": PATH_SHA256,
            "prior_art_sha256": PRIOR_ART_SHA256,
            "equation": "I=F(all)-F(early)-F(late)+F(empty)",
            "groups": {"early": list(EARLY), "late": list(LATE)},
            "maximum_new_price": {"forward_calls": calls,
                                  "example_evaluations": calls*reader.BATCH_SIZE,
                                  "backward_calls": 0, "model_updates": 0,
                                  "raw_numeric_evidence_bytes": len(rows)*2*4},
            "bars": {"native_replay_atol": REPLAY_ATOL,
                     "nonlinear_abs_min": INTERACTION_STRONG,
                     "nonlinear_min_cells": INTERACTION_STRONG_CELLS,
                     "additive_abs_max": INTERACTION_ADDITIVE,
                     "additive_each_group_mean_abs_loss_min": ADDITIVE_GROUP_LOSS_MIN,
                     "all_control_terms_max": CONTROL_TERM_MAX}}


def run_science(*, backend: SplitBackend | None = None, device="cuda", clock=time.perf_counter):
    rows, native, _head, prior = _load()
    executor = backend if backend is not None else path_parent.Task14PathTorchBackend.load(device)
    recipient_cache, donor_cache = {}, {}
    replay_error = 0.0; forwards = evaluations = 0; started = clock()
    for side, cache in (("base",recipient_cache),("donor",donor_cache)):
        for chunk in reader._chunks(rows):
            batch = reader._batch(chunk, side); output = executor.native(batch, capture=True)
            forwards += 1; evaluations += len(chunk)
            if len(output.answer_foil) != len(chunk): raise SplitError("native output count differs")
            cache.update(output.captured)
            for rid, observed in zip(batch.row_ids, output.answer_foil):
                replay_error = max(replay_error, *(abs(a-b) for a,b in
                    zip(reader._pair(observed), native[(rid,side)])))
    required = {(str(row["row_id"]), site) for row in rows for site in LATE}
    if not required.issubset(recipient_cache): raise SplitError("late MLP capture is incomplete")
    late_pairs = {}
    for chunk in reader._chunks(rows):
        batch = reader._batch(chunk,"base")
        output = executor.induce_and_restore(batch, restore_sites=LATE,
                                             donor_cache=donor_cache,
                                             recipient_cache=recipient_cache)
        forwards += 1; evaluations += len(chunk)
        if len(output.answer_foil) != len(chunk): raise SplitError("late-arm output count differs")
        late_pairs.update({rid: reader._pair(pair) for rid,pair in zip(batch.row_ids,output.answer_foil)})
    scale = statistics.median(reader._margin(native[(str(row["row_id"]),"donor")])
        + reader._margin(native[(str(row["row_id"]),"base")])
        for row in rows if row["transform_id"] in {"A1","A2"})
    target, controls, evidence = {}, {"P":[],"C":[]}, []
    for row in rows:
        rid, family = str(row["row_id"]), str(row["transform_id"])
        late = reader._recovery(family,native[(rid,"base")],native[(rid,"donor")],late_pairs[rid],scale)
        f = {**prior[rid], "late": late}
        interaction = f["all"]-f["early"]-f["late"]+f["empty"]
        losses = {"early_loss":f["empty"]-f["early"],
                  "late_loss":f["empty"]-f["late"], "all_loss":f["empty"]-f["all"]}
        record = {"row_id":rid,"family":family,**f,**losses,"interaction":interaction}
        evidence.append(record)
        if family in {"A1","A2"}: target.setdefault(str(row["capability_cell_id"]),[]).append(record)
        else: controls[family].append(record)
    cells = {cell:{key:statistics.fmean(x[key] for x in records)
                   for key in ("early_loss","late_loss","all_loss","interaction")}
             for cell,records in sorted(target.items())}
    control = {family:{key:statistics.fmean(abs(x[key]) for x in records)
                       for key in ("empty","early","late","all","early_loss","late_loss","all_loss","interaction")}
               for family,records in controls.items()}
    max_control = max(v for family in control.values() for v in family.values())
    valid = replay_error <= REPLAY_ATOL
    nonlinear = (sum(abs(x["interaction"])>=INTERACTION_STRONG for x in cells.values())
                 >= INTERACTION_STRONG_CELLS and max_control<=CONTROL_TERM_MAX)
    mean_abs_loss = {group:statistics.fmean(abs(x[f"{group}_loss"]) for x in cells.values())
                     for group in ("early","late")}
    additive = (max(abs(x["interaction"]) for x in cells.values())<=INTERACTION_ADDITIVE
                and min(mean_abs_loss.values())>=ADDITIVE_GROUP_LOSS_MIN
                and max_control<=CONTROL_TERM_MAX)
    terminal = "invalid" if not valid else (
        "compensating_or_nonlinear_screen" if nonlinear else "additive_split_screen" if additive else "inconclusive")
    return {"schema":"task14_head11_3_early_late_mlp_factorial_result_v1",
            "screen_tier_only":True,"execution_policy":"managed_queue_only",
            "authority_sha256":reader.AUTHORITY_SHA256,"pair_result_sha256":PAIR_SHA256,
            "path_result_sha256":PATH_SHA256,"prior_art_sha256":PRIOR_ART_SHA256,
            "terminal":terminal,"predictions":{"pred_a_native_replay":valid,
                "pred_b_compensating_or_nonlinear":nonlinear,"pred_c_additive_split":additive},
            "native_replay_max_abs_error":replay_error,"target_scale":scale,
            "target_cells":cells,"target_mean_absolute_group_loss":mean_abs_loss,
            "control_mean_absolute_terms":control,"evidence":evidence,
            "active_new_price":{"forward_calls":forwards,"example_evaluations":evaluations,
                "backward_calls":0,"model_updates":0,"raw_numeric_evidence_bytes":len(rows)*2*4},
            "serial_seconds":clock()-started}


def main(argv:Sequence[str]|None=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true"); args=parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN","BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None,"1"}: raise SplitError(f"{name} must be absent or exactly 1")
    if args.dry_run or any(os.environ.get(n)=="1" for n in ("BQLIB_DRYRUN","BQLIB_NO_MODEL")):
        print(json.dumps(compile_dryrun(),sort_keys=True)); return
    if RESULT.exists(): raise SplitError(f"refusing to overwrite {RESULT}")
    result=run_science(); RESULT.parent.mkdir(parents=True,exist_ok=True)
    RESULT.write_text(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps({k:result[k] for k in ("terminal","predictions","active_new_price","serial_seconds")},sort_keys=True))


if __name__=="__main__": main()
