#!/usr/bin/env python3
"""Exhaustive response-composition lattice for activation-conditioned readers."""

# BQGATE: EXPERIMENT pred_a_authority_pool_replay_self_clamp_finiteness_price pred_b_ten_site_pool_is_sufficient pred_c_an_eight_site_or_smaller_program_exists pred_d_selected_program_meets_all_cells pred_e_complete_zero_fit_lattice
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import attention_source_destination_eval as attention_eval
import run_temporal_iswas_canonical_downstream_response_removal_atlas_v1 as atlas
import run_temporal_iswas_upstream_full_response_mode_atlas_v1 as response

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_iswas_downstream_ten_site_response_lattice_v1.json"
ATLAS_RESULT=ROOT/"circuits/followups/temporal_iswas_canonical_downstream_response_removal_atlas_v1_result.json"
ATLAS_RUNNER=ROOT/"ops/run_temporal_iswas_canonical_downstream_response_removal_atlas_v1.py"
FRONTIER_RESULT=ROOT/"circuits/followups/temporal_iswas_v11_writer_frontier_holdout_v1_result.json"
FRONTIER_RUNNER=ROOT/"ops/run_temporal_iswas_v11_writer_frontier_holdout_v1.py"
OUT=ROOT/"circuits/followups/temporal_iswas_downstream_ten_site_response_lattice_v1_result.json"
CANDIDATE_ID="cross_task.temporal_iswas_downstream_ten_site_response_lattice_v1"
POOL=("L13H6","L15H1","L15H5","L17H2","MLP12","MLP13","MLP14","MLP15","MLP16","MLP17")
ARMS=tuple(range(1<<len(POOL)))
MAX_FORWARDS,MAX_EVALUATIONS=1036,29008
EXPECTED={"prior":"032fc186bb1e0acf8ea2334f26214f83779ced885064827e2ccf8d12ea417a29","atlas_result":"13bc3b7406c27b5ee0fa28482aa5786889f765b172abbd85f897a842e7981d62","atlas_runner":"2e0309b9d39bbd5a61c2d9da428164667c038699f4c623100d0ef8214c5dc5d6","frontier_result":"9262c379be1a485b826a0f414c822e75832a1162a69881a4b40d2322e26ab07b","frontier_runner":"08c9b1e85b9599c4cac2195b22ab2f04514c5bf4c861fff3192e3d1bf8e1431d"}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def tensor_sha(tensor):return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()
def run_mixed(backend,batch,values,call):
    handles=[];head_width=int(backend.model.config.n_embd//backend.model.config.n_head)
    for site in atlas.SITES:
        kind,_layer,_head=response.site_parts(site);hook=response.patch_hook(batch,values[site],site,head_width);module=atlas.site_module(backend,site)
        handles.append(module.register_forward_pre_hook(hook) if kind=="attn" else module.register_forward_hook(hook))
    try:return call()
    finally:
        for handle in handles:handle.remove()
def main():
    paths={"prior":PRIOR,"atlas_result":ATLAS_RESULT,"atlas_runner":ATLAS_RUNNER,"frontier_result":FRONTIER_RESULT,"frontier_runner":FRONTIER_RUNNER}
    if {key:sha(value) for key,value in paths.items()}!=EXPECTED:raise RuntimeError("downstream lattice authority changed")
    prior,aresult,fresult,weights,tcap,icap=[json.loads(path.read_text()) for path in (PRIOR,ATLAS_RESULT,FRONTIER_RESULT,atlas.WEIGHT_RESULT,atlas.frontier.TEMPORAL_CAPABILITY,atlas.frontier.ISWAS_CAPABILITY)]
    if prior.get("candidate_id")!=CANDIDATE_ID or aresult.get("terminal")!="novel_reader_null" or fresult.get("terminal")!="conditional_writer_screen":raise RuntimeError("authority terminal changed")
    selected=tuple(site for site in atlas.SITES if any(abs(aresult["site_metrics"][site]["tasks"]["pooled"][family]["signed_projection"])>=.005 for family in ("behavior","mode1","mode2")))
    if set(selected)!=set(POOL):raise RuntimeError(f"activation-conditioned pool changed: {selected}")
    rows=sum((atlas.frontier.sealed_rows(atlas.frontier.temporal,tcap,panel) for panel in ("A1","A2")),[])+sum((atlas.frontier.sealed_rows(atlas.frontier.iswas,icap,panel) for panel in ("A1","A2")),[])
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"rows":len(rows),"pool":list(POOL),"arms":len(ARMS),"model_forwards_max":MAX_FORWARDS,"example_evaluations_max":MAX_EVALUATIONS,"fit_updates":0,"model_updates":0,"transformer_backwards":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if len(rows)!=28 or len(ARMS)!=1024:raise RuntimeError("frozen population changed")
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
    base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor")
    base_output,base_up=response.capture(backend,base_batch);donor_output,donor_up=response.capture(backend,donor_batch)
    base_down_output,base_down=atlas.capture_downstream(backend,lambda:backend.native(base_batch,capture=True))
    live_pair,live_down=atlas.capture_downstream(backend,lambda:response.run_patch(backend,base_batch,donor_up,atlas.WRITER_SITES));live_output=live_pair[0]
    self_output=run_mixed(backend,base_batch,base_down,lambda:backend.native(base_batch,capture=True))
    live_replay_pair=run_mixed(backend,base_batch,live_down,lambda:response.run_patch(backend,base_batch,donor_up,atlas.WRITER_SITES));live_replay=live_replay_pair[0]
    base_state,donor_state,live_state,self_state,replay_state=(response.states(torch,backend,x,rows) for x in (base_output,donor_output,live_output,self_output,live_replay))
    identity_error=max(float((response.states(torch,backend,base_down_output,rows)-base_state).abs().max()),float((self_state-base_state).abs().max()),float((replay_state-live_state).abs().max()))
    reconstruction=0.0
    for layer in atlas.LAYERS:
        _replay,captured=attention_eval.capture_layer_attention(backend,base_batch,layer);reconstruction=max(reconstruction,float((captured["head_output"].reshape_as(base_down[f"L{layer}H0"])-base_down[f"L{layer}H0"]).abs().max()))
    family,_singular,_energy=atlas.frontier.parent.family_builder.build_family(backend,json.loads(atlas.frontier.parent.SUBSPACE.read_text()));q=family[8]
    gain=math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float()) for layer in atlas.LAYERS);raw_modes,orientation_error,_wrong=atlas.frontier.parent.overlap.residual_modes(backend,q,gain);state_basis=torch.linalg.qr(raw_modes,mode="reduced").Q
    reader_coordinates=torch.as_tensor(weights["mode_artifacts"]["reader_coordinates"],device=backend.device).float();physical_reader=state_basis@reader_coordinates;reader_hash_ok=tensor_sha(physical_reader)==weights["mode_artifacts"]["physical_reader_covectors_sha256"]
    answer=torch.as_tensor([row["donor_answer_id"] for row in rows],device=backend.device);foil=torch.as_tensor([row["donor_foil_id"] for row in rows],device=backend.device);index=torch.arange(len(rows),device=backend.device)
    def margin(state):
        logits=das.head_logits(backend,state);return logits[index,answer]-logits[index,foil]
    base_margin,donor_margin,live_margin=margin(base_state),margin(donor_state),margin(live_state);full_behavior=live_margin-base_margin;full_modes=(live_state-base_state)@physical_reader
    task_indices={"temporal":torch.arange(0,14,device=backend.device),"iswas":torch.arange(14,28,device=backend.device)}
    metrics={};forwards,evaluations=12,336
    for mask in ARMS:
        values={site:(live_down[site] if site in POOL and mask&(1<<POOL.index(site)) else base_down[site]) for site in atlas.SITES};pair=run_mixed(backend,base_batch,values,lambda:response.run_patch(backend,base_batch,donor_up,atlas.WRITER_SITES));state=response.states(torch,backend,pair[0],rows);behavior=margin(state)-base_margin;modes=(state-base_state)@physical_reader;tasks={};cell_residuals=[]
        for task,ids in task_indices.items():
            b=response.vector_stats(torch,behavior[ids],full_behavior[ids]);b["squared_residual"]=float((behavior[ids]-full_behavior[ids]).square().sum()/full_behavior[ids].square().sum());tasks[task]={"behavior":b};cell_residuals.append(b["squared_residual"])
            for mode in range(2):
                m=response.vector_stats(torch,modes[ids,mode],full_modes[ids,mode]);m["squared_residual"]=float((modes[ids,mode]-full_modes[ids,mode]).square().sum()/full_modes[ids,mode].square().sum());tasks[task][f"mode{mode+1}"]=m;cell_residuals.append(m["squared_residual"])
        direction=float(((behavior/full_behavior)>0).float().mean());metrics[str(mask)]={"sites":[site for bit,site in enumerate(POOL) if mask&(1<<bit)],"site_count":int(mask.bit_count()),"tasks":tasks,"behavior_direction_fraction":direction,"worst_six_cell_residual":max(cell_residuals)};forwards+=1;evaluations+=len(rows)
    def eligible(item):return item[1]["behavior_direction_fraction"]>=.90 and item[1]["worst_six_cell_residual"]<=.15
    eligible_arms=[item for item in metrics.items() if eligible(item)];selected_mask,selected_metrics=min(eligible_arms,key=lambda item:(item[1]["site_count"],item[1]["worst_six_cell_residual"],int(item[0]))) if eligible_arms else (None,None)
    full=metrics[str((1<<len(POOL))-1)];pred_b=full["worst_six_cell_residual"]<=.10 and full["behavior_direction_fraction"]>=.95;pred_c=selected_metrics is not None and selected_metrics["site_count"]<=8;pred_d=selected_metrics is not None and eligible((selected_mask,selected_metrics))
    finite=all(math.isfinite(value) for arm in metrics.values() for task in arm["tasks"].values() for family in task.values() for value in family.values())
    pred_a=orientation_error<=1e-6 and reader_hash_ok and identity_error<=1e-4 and reconstruction<=5e-4 and finite and forwards==MAX_FORWARDS and evaluations==MAX_EVALUATIONS
    pred_e=len(metrics)==1024 and set(metrics)=={str(x) for x in ARMS}
    predictions={"pred_a_authority_pool_replay_self_clamp_finiteness_price":bool(pred_a),"pred_b_ten_site_pool_is_sufficient":bool(pred_b),"pred_c_an_eight_site_or_smaller_program_exists":bool(pred_c),"pred_d_selected_program_meets_all_cells":bool(pred_d),"pred_e_complete_zero_fit_lattice":bool(pred_e)}
    terminal="invalid" if not pred_a else "compact_response_program" if all(predictions.values()) else "pool_insufficient" if not pred_b else "distributed_downstream_chain" if not pred_c else "selection_null"
    result={"schema":"temporal_iswas_downstream_ten_site_response_lattice_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"orientation_max_abs":orientation_error,"physical_reader_hash_ok":reader_hash_ok,"identity_self_and_live_replay_max_abs":identity_error,"attention_reconstruction_max_abs":reconstruction},"pool":list(POOL),"selected_mask":int(selected_mask) if selected_mask is not None else None,"selected_metrics":selected_metrics,"eligible_arm_count":len(eligible_arms),"full_pool_metrics":full,"arm_metrics":metrics,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":forwards,"example_evaluations":evaluations,"fit_updates":0,"model_updates":0,"transformer_backwards":0},"serial_seconds":time.perf_counter()-started}
    atomic_create_json(OUT,result);print(json.dumps({key:result[key] for key in ("candidate_id","instrument","pool","selected_mask","selected_metrics","eligible_arm_count","full_pool_metrics","predictions","terminal","price")},sort_keys=True))
if __name__=="__main__":main()
