#!/usr/bin/env python3
"""Scale-relative replay audit for the response-complement experiment."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_finiteness_and_price pred_b_rank0_scale_relative_replay_closes pred_c_behavior_mode_replay_confirms_v1 pred_d_control_reduction_null_is_material pred_e_target_complement_was_not_the_failure
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as v1
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_five_mlp_generic_subspace_complement_v2_scale_audit.json"
OLD=ROOT/"circuits/followups/temporal_five_mlp_generic_subspace_complement_program_v1_result.json"
OUT=ROOT/"circuits/followups/temporal_five_mlp_generic_subspace_complement_v2_scale_audit_result.json"
EXPECTED={"prior":"62078405d7b4164b58e96ecb54de9418d3b7643c429cd0b7ba1842e2646e765d","old":"6b45e6c58b2d8ca208368ebc04247c06574511b77be1b7fdcdca1d2999031f86","runner":"85f556968935ae49e23aa52a5b7cb62aa62e15d9def27c24374ac57d0ab05f4f"}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")

def main():
    if {"prior":sha(PRIOR),"old":sha(OLD),"runner":sha(Path(v1.__file__))}!=EXPECTED: raise RuntimeError("authority changed")
    old=json.loads(OLD.read_text()); dry={"candidate_id":"temporal_auxiliary.five_mlp_generic_subspace_complement_v2_scale_audit","dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"model_forwards_max":8,"fit_updates":0,"model_updates":0,"transformer_backwards":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(dry,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    tic=time.perf_counter(); backend=producer.Bilin18TorchBackend.load("cuda"); torch=backend.torch
    tcap,icap=json.loads(v1.TCAP.read_text()),json.loads(v1.ICAP.read_text()); tr,ir,*_=greedy.rows_and_controls(tcap,icap); rows=tr+ir
    disc=matched.control_rows()["discovery"]
    def cap(rs):
        bb,db=das._batch(backend,rs,side="base"),das._batch(backend,rs,side="donor")
        bo,bc=atlasrun.capture_native(backend,bb); do,dc=atlasrun.capture_native(backend,db); return bb,bo,bc,do,dc
    disc_batch,_dbo,dbc,_ddo,ddc=cap(disc); batch,bo,bc,do,dc=cap(rows)
    bases,_=v1.fit_bases(backend,disc_batch,dbc,ddc)
    ordinary,_=atlasrun.run_patch(backend,batch,dc,v1.SITES); rank0=v1.run_complement(backend,batch,bc,dc,bases,0)
    bs,ds,ordinary_state,rs=[atlasrun.states(torch,backend,x,rows) for x in (bo,do,ordinary,rank0)]
    error=rs-ordinary_state; reference_norm=float(ordinary_state.norm()); max_abs=float(error.abs().max()); rse=float(error.square().sum()/ordinary_state.square().sum())
    reader,orientation,reader_ok=greedy.physical_reader(backend,json.loads(v1.WEIGHTS.read_text()))
    full_b=v1.margins(backend,ds,rows)-v1.margins(backend,bs,rows); db=v1.margins(backend,rs,rows)-v1.margins(backend,bs,rows); full_m=(ds-bs)@reader; dm=(rs-bs)@reader
    ids={"temporal":slice(0,len(tr)),"iswas":slice(len(tr),len(rows))}; cells={}
    for task,ix in ids.items():
        cells[f"{task}_behavior"]=float((db[ix]-full_b[ix]).square().sum()/full_b[ix].square().sum())
        for j in range(2): cells[f"{task}_mode{j+1}"]=float((dm[ix,j]-full_m[ix,j]).square().sum()/full_m[ix,j].square().sum())
    worst=max(cells.values()); reports=old["reports"]; pa=reader_ok and orientation<=1e-6 and all(math.isfinite(x) for x in (max_abs,reference_norm,rse,worst)); pb=max_abs/max(reference_norm,1e-30)<=1e-6 and rse<=1e-10; pc=abs(worst-reports["0"]["worst_target_residual"])<=1e-6
    pd=any(reports["8"]["validation_control_fraction"][t]>.5*reports["0"]["validation_control_fraction"][t] for t in ids); pe=reports["8"]["worst_target_residual"]<=.05 and all(reports["8"]["behavior_signed_projection"][t]>=.85 for t in ids)
    preds={"pred_a_authority_alignment_finiteness_and_price":bool(pa),"pred_b_rank0_scale_relative_replay_closes":bool(pb),"pred_c_behavior_mode_replay_confirms_v1":bool(pc),"pred_d_control_reduction_null_is_material":bool(pd),"pred_e_target_complement_was_not_the_failure":bool(pe)}; terminal="generic_subspace_miss" if all(preds.values()) else "invalid" if not(pa and pb and pc) else "inconclusive"
    result={"schema":"temporal_five_mlp_generic_subspace_complement_scale_audit_result_v2","started_utc":now(),"finished_utc":now(),"serial_seconds":time.perf_counter()-tic,"authority_sha256":EXPECTED,"instrument":{"max_abs":max_abs,"reference_norm":reference_norm,"max_abs_over_reference_norm":max_abs/max(reference_norm,1e-30),"relative_squared_error":rse,"rank0_worst_target_residual":worst},"predictions":preds,"terminal":terminal,"price":{"model_forwards":6,"fit_updates":0,"model_updates":0,"transformer_backwards":0}}
    atomic_create_json(OUT,result); print(json.dumps(result,sort_keys=True))
if __name__=="__main__": main()
