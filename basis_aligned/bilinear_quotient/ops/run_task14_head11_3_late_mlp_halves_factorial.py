#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by this targeted runner.
"""Split the compensating MLP13--17 path into MLP13--14 and MLP15--17.

F(S) is head-11.3 donor recovery after restoring recipient-native outputs in
S.  The immutable early/late result supplies F(empty) and F(13--17); this run
opens only F(13--14) and F(15--17).  The exact interaction is

 I = F(13--17) - F(13--14) - F(15--17) + F(empty).

Registered predictions: (A) native replay <=1e-4, otherwise invalid; (B) one
half explains the directional compensation if its four-cell loss vector has
relative L2 error <=0.35 against the all-late loss, the other half has RMS
loss <=0.05, and interaction RMS <=0.03; (C) genuine cross-half interaction
if interaction RMS >=0.08, at least two cells have |I|>=0.05, and its L2 norm
is >=50% of the all-late loss norm.  Every scientific terminal additionally
requires every P/C corner effect, loss, and interaction <=0.10.  Anything in
the gap is inconclusive.  Maximum new price: 16 forwards, 512 examples, zero
backwards/updates, and 2,048 retained raw-logit bytes. Managed queue only.
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
import run_task14_head11_3_attention_mlp_path_factorial as path_parent
import run_task14_head11_3_downstream_module_reader_screen as reader


ROOT=Path(__file__).resolve().parent.parent
PARENT=ROOT/"circuits/followups/task14_head11_3_early_late_mlp_factorial_v1_result.json"
RESULT=ROOT/"circuits/followups/task14_head11_3_late_mlp_halves_factorial_v1_result.json"
PARENT_SHA256="b5ef36fa7aad8b0893be841861b46466d0f20ae977e6dde66acc9d765f6557db"
PRIOR_ART_SHA256="7c32b484174be55d28df2918406ecc3ae2aa11c16b920b3222f5705956d0a8ed"
FIRST=("mlp:13","mlp:14")
SECOND=("mlp:15","mlp:16","mlp:17")
ARMS={"mlp13_14":FIRST,"mlp15_17":SECOND}
REPLAY_ATOL=1e-4
DOMINANT_REL_L2_MAX=0.35
OTHER_RMS_MAX=0.05
DOMINANT_INTERACTION_RMS_MAX=0.03
NONLINEAR_INTERACTION_RMS_MIN=0.08
NONLINEAR_CELL_ABS_MIN=0.05
NONLINEAR_MIN_CELLS=2
NONLINEAR_NORM_RATIO_MIN=0.50
CONTROL_TERM_MAX=0.10


class HalfError(ValueError): pass


class HalfBackend(Protocol):
    def native(self,batch:producer.ModelBatch,*,capture:bool)->producer.BatchOutput:...
    def induce_and_restore(self,batch:producer.ModelBatch,*,restore_sites:Sequence[str],
                           donor_cache:Mapping[tuple[str,str],object],
                           recipient_cache:Mapping[tuple[str,str],object])->producer.BatchOutput:...


def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def _load():
    rows,native,head=path_parent._load()
    if _sha(PARENT)!=PARENT_SHA256: raise HalfError("immutable early/late result hash changed")
    parent=json.loads(PARENT.read_text())
    if parent.get("authority_sha256")!=reader.AUTHORITY_SHA256: raise HalfError("parent authority changed")
    records={str(x["row_id"]):x for x in parent.get("evidence",[])}
    if len(records)!=len(rows): raise HalfError("parent lacks exact row coverage")
    prior={rid:{"empty":float(x["empty"]),"all_late":float(x["late"])} for rid,x in records.items()}
    return rows,native,head,prior


def compile_dryrun():
    rows,_native,_head,_prior=_load(); calls=(2+len(ARMS))*len(reader._chunks(rows))
    return {"schema":"task14_head11_3_late_mlp_halves_factorial_dryrun_v1",
      "model_loaded":False,"gpu_accessed":False,"queue_touched":False,
      "authority_sha256":reader.AUTHORITY_SHA256,"parent_result_sha256":PARENT_SHA256,
      "prior_art_sha256":PRIOR_ART_SHA256,"groups":{k:list(v) for k,v in ARMS.items()},
      "equation":"I=F(13-17)-F(13-14)-F(15-17)+F(empty)",
      "maximum_new_price":{"forward_calls":calls,"example_evaluations":calls*reader.BATCH_SIZE,
        "backward_calls":0,"model_updates":0,"raw_numeric_evidence_bytes":len(ARMS)*len(rows)*2*4},
      "bars":{"native_replay_atol":REPLAY_ATOL,"dominant_relative_l2_max":DOMINANT_REL_L2_MAX,
        "other_half_rms_max":OTHER_RMS_MAX,"dominant_interaction_rms_max":DOMINANT_INTERACTION_RMS_MAX,
        "nonlinear_interaction_rms_min":NONLINEAR_INTERACTION_RMS_MIN,
        "nonlinear_cell_abs_min":NONLINEAR_CELL_ABS_MIN,"nonlinear_min_cells":NONLINEAR_MIN_CELLS,
        "nonlinear_norm_ratio_min":NONLINEAR_NORM_RATIO_MIN,"all_control_terms_max":CONTROL_TERM_MAX}}


def _norm(values): return math.sqrt(sum(x*x for x in values))


def run_science(*,backend:HalfBackend|None=None,device="cuda",clock=time.perf_counter):
    rows,native,_head,prior=_load()
    executor=backend if backend is not None else path_parent.Task14PathTorchBackend.load(device)
    recipient_cache,donor_cache={},{}; replay_error=0.; forwards=evaluations=0; started=clock()
    for side,cache in (("base",recipient_cache),("donor",donor_cache)):
      for chunk in reader._chunks(rows):
        batch=reader._batch(chunk,side); output=executor.native(batch,capture=True)
        forwards+=1; evaluations+=len(chunk)
        if len(output.answer_foil)!=len(chunk): raise HalfError("native output count differs")
        cache.update(output.captured)
        for rid,observed in zip(batch.row_ids,output.answer_foil):
          replay_error=max(replay_error,*(abs(a-b) for a,b in zip(reader._pair(observed),native[(rid,side)])))
    required={(str(row["row_id"]),site) for row in rows for site in FIRST+SECOND}
    if not required.issubset(recipient_cache): raise HalfError("late MLP capture is incomplete")
    arm_pairs={}
    for arm,sites in ARMS.items():
      arm_pairs[arm]={}
      for chunk in reader._chunks(rows):
        batch=reader._batch(chunk,"base")
        output=executor.induce_and_restore(batch,restore_sites=sites,donor_cache=donor_cache,
                                           recipient_cache=recipient_cache)
        forwards+=1; evaluations+=len(chunk)
        if len(output.answer_foil)!=len(chunk): raise HalfError("half-arm output count differs")
        arm_pairs[arm].update({rid:reader._pair(pair) for rid,pair in zip(batch.row_ids,output.answer_foil)})
    scale=statistics.median(reader._margin(native[(str(row["row_id"]),"donor")])
      +reader._margin(native[(str(row["row_id"]),"base")]) for row in rows
      if row["transform_id"] in {"A1","A2"})
    targets,controls,evidence={}, {"P":[],"C":[]}, []
    for row in rows:
      rid,family=str(row["row_id"]),str(row["transform_id"]); args=family,native[(rid,"base")],native[(rid,"donor")]
      f={**prior[rid],**{arm:reader._recovery(*args,pairs[rid],scale) for arm,pairs in arm_pairs.items()}}
      interaction=f["all_late"]-f["mlp13_14"]-f["mlp15_17"]+f["empty"]
      losses={"all_late_loss":f["empty"]-f["all_late"],
              "mlp13_14_loss":f["empty"]-f["mlp13_14"],
              "mlp15_17_loss":f["empty"]-f["mlp15_17"]}
      record={"row_id":rid,"family":family,**f,**losses,"interaction":interaction}; evidence.append(record)
      (targets.setdefault(str(row["capability_cell_id"]),[]) if family in {"A1","A2"}
       else controls[family]).append(record)
    keys=("all_late_loss","mlp13_14_loss","mlp15_17_loss","interaction")
    cells={cell:{key:statistics.fmean(x[key] for x in records) for key in keys}
           for cell,records in sorted(targets.items())}
    control_keys=("empty","all_late","mlp13_14","mlp15_17")+keys
    control={family:{key:statistics.fmean(abs(x[key]) for x in records) for key in control_keys}
             for family,records in controls.items()}
    max_control=max(v for family in control.values() for v in family.values())
    all_vector=[x["all_late_loss"] for x in cells.values()]; denom=max(_norm(all_vector),1e-12)
    diagnostics={}
    for half in ("mlp13_14","mlp15_17"):
      vector=[x[f"{half}_loss"] for x in cells.values()]
      diagnostics[half]={"rms":_norm(vector)/2,
        "relative_l2_to_all_late":_norm([a-b for a,b in zip(vector,all_vector)])/denom,
        "cosine_to_all_late":sum(a*b for a,b in zip(vector,all_vector))/(max(_norm(vector),1e-12)*denom)}
    interaction_vector=[x["interaction"] for x in cells.values()]
    interaction_rms=_norm(interaction_vector)/2
    dominant=[]
    for half,other in (("mlp13_14","mlp15_17"),("mlp15_17","mlp13_14")):
      if (diagnostics[half]["relative_l2_to_all_late"]<=DOMINANT_REL_L2_MAX
          and diagnostics[other]["rms"]<=OTHER_RMS_MAX
          and interaction_rms<=DOMINANT_INTERACTION_RMS_MAX): dominant.append(half)
    control_ok=max_control<=CONTROL_TERM_MAX; valid=replay_error<=REPLAY_ATOL
    nonlinear=(interaction_rms>=NONLINEAR_INTERACTION_RMS_MIN
      and sum(abs(x)>=NONLINEAR_CELL_ABS_MIN for x in interaction_vector)>=NONLINEAR_MIN_CELLS
      and _norm(interaction_vector)/denom>=NONLINEAR_NORM_RATIO_MIN and control_ok)
    one_half=bool(dominant) and control_ok
    terminal="invalid" if not valid else (
      "one_half_explains_screen" if one_half else "cross_half_interaction_screen" if nonlinear else "inconclusive")
    return {"schema":"task14_head11_3_late_mlp_halves_factorial_result_v1",
      "screen_tier_only":True,"execution_policy":"managed_queue_only",
      "authority_sha256":reader.AUTHORITY_SHA256,"parent_result_sha256":PARENT_SHA256,
      "prior_art_sha256":PRIOR_ART_SHA256,"terminal":terminal,
      "predictions":{"pred_a_native_replay":valid,"pred_b_one_half_explains":one_half,
                     "pred_c_cross_half_interaction":nonlinear},"dominant_half":dominant,
      "native_replay_max_abs_error":replay_error,"target_scale":scale,"target_cells":cells,
      "half_diagnostics":diagnostics,"interaction_rms":interaction_rms,
      "interaction_norm_ratio_to_all_late":_norm(interaction_vector)/denom,
      "control_mean_absolute_terms":control,"evidence":evidence,
      "active_new_price":{"forward_calls":forwards,"example_evaluations":evaluations,
        "backward_calls":0,"model_updates":0,"raw_numeric_evidence_bytes":len(ARMS)*len(rows)*2*4},
      "serial_seconds":clock()-started}


def main(argv:Sequence[str]|None=None):
  parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true"); args=parser.parse_args(argv)
  for name in ("BQLIB_DRYRUN","BQLIB_NO_MODEL"):
    if os.environ.get(name) not in {None,"1"}: raise HalfError(f"{name} must be absent or exactly 1")
  if args.dry_run or any(os.environ.get(n)=="1" for n in ("BQLIB_DRYRUN","BQLIB_NO_MODEL")):
    print(json.dumps(compile_dryrun(),sort_keys=True)); return
  if RESULT.exists(): raise HalfError(f"refusing to overwrite {RESULT}")
  result=run_science(); RESULT.parent.mkdir(parents=True,exist_ok=True)
  RESULT.write_text(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n")
  print(json.dumps({k:result[k] for k in ("terminal","predictions","active_new_price","serial_seconds")},sort_keys=True))


if __name__=="__main__": main()
