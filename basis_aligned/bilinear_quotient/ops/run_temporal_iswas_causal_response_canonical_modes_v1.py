#!/usr/bin/env python3
"""Canonical shared/task-specific modes of the frozen temporal/is-was causal matrix."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_matrix_coverage_finiteness_and_price pred_b_leading_mode_is_cross_task_shared pred_c_leading_modes_encode_intervention_direction pred_d_weight_prediction_preserves_canonical_subspaces pred_e_response_is_non_scalar_but_rank8_complete
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path
import numpy as np
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_iswas_causal_response_canonical_modes_v1.json"
HANKEL=ROOT/"circuits/followups/temporal_iswas_q8_finite_causal_hankel_v1_result.json"
SHARED=ROOT/"circuits/followups/temporal_q8_iswas_cdas_shared_specific_causal_v2_result.json"
OVERLAP=ROOT/"circuits/followups/temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1_result.json"
WEIGHTS=ROOT/"circuits/followups/temporal_auxiliary_will_had_h3_multicue_weight_interface_v1_result.json"
OUT=ROOT/"circuits/followups/temporal_iswas_causal_response_canonical_modes_v1_result.json"
CANDIDATE_ID="cross_task.temporal_iswas_causal_response_canonical_modes_v1"
EXPECTED={"prior":"cb0a42954765264c609a5fc8a783dad1d7032d7e262f094a79c73e206b7b9546","hankel":"f8fa10c21c30cd3420648641b4a284ba3cb41152872db8cf77d25213c597bb62","shared":"bd302cb0d104db5afe43906885dff52f851a03e638c6ff30de9d87224ce235bc","overlap":"883861b7392a8b1214491bef2fab80bfd670dfca98d87f91cb73fcb22bf624e6","weights":"5bf804ee1e61f918edceb0dc9e31ac68fa157de384a943391c4ca4eeb672246a"}
TASKS=("temporal","iswas");RANKS=(1,2,4,8)
PRICE={"model_forwards":0,"example_evaluations":0,"matrix_cells":1024,"fit_updates":0,"model_updates":0,"transformer_backwards":0}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def cosine(x,y):
    z=np.linalg.norm(x)*np.linalg.norm(y);return float(np.dot(x,y)/z) if z else float("nan")
def rms(x):return float(np.sqrt(np.mean(np.square(x))))

def main():
    paths={"prior":PRIOR,"hankel":HANKEL,"shared":SHARED,"overlap":OVERLAP,"weights":WEIGHTS}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("canonical response authority changed")
    prior,hankel,shared,overlap,weights=[json.loads(p.read_text()) for p in (PRIOR,HANKEL,SHARED,OVERLAP,WEIGHTS)]
    if prior.get("candidate_id")!=CANDIDATE_ID or hankel.get("terminal")!="screen" or shared.get("terminal")!="screen" or overlap.get("terminal")!="screen" or weights.get("terminal")!="partial":raise RuntimeError("parent terminal changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"source_states":32,"reader_states":32,"tasks":list(TASKS),"ranks":list(RANKS),**PRICE}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();metadata=hankel["row_metadata"]
    exact=np.full((32,32),np.nan);predicted=np.full((32,32),np.nan);seen=set()
    for record in hankel["records"]:
        key=(record["source_index"],record["target_index"])
        if key in seen:raise RuntimeError("duplicate matrix cell")
        seen.add(key);exact[key]=record["exact_effect"];predicted[key]=record["predicted_effect"]
    task_index={task:np.array([i for i,x in enumerate(metadata) if x["task"]==task]) for task in TASKS}
    diag_rms={task:rms(exact[np.ix_(task_index[task],task_index[task])]) for task in TASKS}
    scale=np.empty((32,32))
    for i,a in enumerate(metadata):
        for j,b in enumerate(metadata):scale[i,j]=math.sqrt(diag_rms[a["task"]]*diag_rms[b["task"]])
    balanced=exact/scale;balanced_pred=predicted/scale
    u,s,vh=np.linalg.svd(balanced,full_matrices=False);up,sp,vhp=np.linalg.svd(balanced_pred,full_matrices=False)
    direction=np.array([1. if x["direction"] in ("future_to_anterior","present_to_past") else -1. for x in metadata])
    modes=[]
    for k in range(8):
        source_energy={task:float(np.square(u[task_index[task],k]).sum()) for task in TASKS}
        target_energy={task:float(np.square(vh[k,task_index[task]]).sum()) for task in TASKS}
        source_family={f"{task}_{family}":float(np.square(u[[i for i,x in enumerate(metadata) if x["task"]==task and x["family"]==family],k]).sum()) for task in TASKS for family in ("A1","A2")}
        target_family={f"{task}_{family}":float(np.square(vh[k,[i for i,x in enumerate(metadata) if x["task"]==task and x["family"]==family]]).sum()) for task in TASKS for family in ("A1","A2")}
        modes.append({"mode":k+1,"singular_value":float(s[k]),"energy_fraction":float(s[k]**2/np.square(s).sum()),"source_task_energy":source_energy,"target_task_energy":target_energy,"source_family_energy":source_family,"target_family_energy":target_family,"source_direction_abs_cosine":abs(cosine(u[:,k],direction)),"target_direction_abs_cosine":abs(cosine(vh[k],direction)),"cross_task_shared":min(*source_energy.values(),*target_energy.values())>=.10})
    reconstruction={}
    for rank in RANKS:
        approx=(u[:,:rank]*s[:rank])@vh[:rank];by={}
        for a in TASKS:
            for b in TASKS:
                block=balanced[np.ix_(task_index[a],task_index[b])];guess=approx[np.ix_(task_index[a],task_index[b])]
                by[f"{a}_to_{b}"]={"relative_rmse":rms(block-guess)/rms(block),"cosine":cosine(block.reshape(-1),guess.reshape(-1))}
        reconstruction[str(rank)]=by
    source_sv=np.linalg.svd(u[:,:8].T@up[:,:8],compute_uv=False);target_sv=np.linalg.svd(vh[:8]@vhp[:8].T,compute_uv=False)
    agreement={"source_principal_cosines":source_sv.tolist(),"target_principal_cosines":target_sv.tolist(),"mean_squared_cosine":float((np.square(source_sv).mean()+np.square(target_sv).mean())/2)}
    raw_s=np.linalg.svd(exact,compute_uv=False)
    finite=len(seen)==1024 and np.isfinite(exact).all() and np.isfinite(predicted).all() and all(len(x)==16 for x in task_index.values()) and all(v>0 for v in diag_rms.values())
    pred_a=bool(finite and shared["component_norms"]["shared"]>0 and overlap["overlap"]["squared_projection_rho"]>0 and weights["known_late_attention_percentiles"]["L15H5"]==1.0)
    pred_b=any(x["cross_task_shared"] for x in modes[:4])
    pred_c=max(max(x["source_direction_abs_cosine"],x["target_direction_abs_cosine"]) for x in modes[:4])>=.70
    pred_d=agreement["mean_squared_cosine"]>=.90
    rank1_bad=any(x["relative_rmse"]>.10 for x in reconstruction["1"].values());rank8_good=all(x["relative_rmse"]<=.01 for x in reconstruction["8"].values());pred_e=rank1_bad and rank8_good
    predictions={"pred_a_authority_matrix_coverage_finiteness_and_price":pred_a,"pred_b_leading_mode_is_cross_task_shared":bool(pred_b),"pred_c_leading_modes_encode_intervention_direction":bool(pred_c),"pred_d_weight_prediction_preserves_canonical_subspaces":bool(pred_d),"pred_e_response_is_non_scalar_but_rank8_complete":bool(pred_e)}
    terminal="invalid" if not pred_a else "canonical_shared_response_modes" if all(predictions.values()) else "predictive_interface_without_shared_vocabulary" if not pred_b else "partial_canonicalization"
    result={"schema":"temporal_iswas_causal_response_canonical_modes_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"cpu_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,"task_diagonal_rms":diag_rms,"raw_singular_values":raw_s.tolist(),"balanced_singular_values":s.tolist(),"modes":modes,"reconstruction":reconstruction,"weight_prediction_subspace_agreement":agreement,"predictions":predictions,"terminal":terminal,"price":PRICE}
    atomic_create_json(OUT,result);print(json.dumps({"candidate_id":CANDIDATE_ID,"task_diagonal_rms":diag_rms,"balanced_singular_values":s[:8].tolist(),"modes":modes,"reconstruction":reconstruction,"weight_prediction_subspace_agreement":agreement,"predictions":predictions,"terminal":terminal,"price":PRICE},sort_keys=True))

if __name__=="__main__":main()
