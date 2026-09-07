#!/usr/bin/env python3
"""Split the late attention correction into a small head-level program."""

# BQGATE: EXPERIMENT pred_a_authority_split_exactness_finiteness_and_price pred_b_selected_heads_improve_mlp_chain_and_singleton_on_confirmation pred_c_selected_heads_preserve_behavior_on_both_panels pred_d_selected_heads_preserve_q8_on_both_panels pred_e_zero_fit_small_head_program_and_full_replay
from datetime import datetime, timezone
import hashlib,json,math,os,time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_downstream_response_greedy_program_v1 as parent_runner
import run_iswas_mlp8_main_mlp9_aux_composition_v1 as composition
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/iswas_mlp8_attn15_remainder_head_greedy_v1.json"
PARENT=ROOT/"circuits/followups/iswas_mlp8_downstream_response_greedy_program_v1_result.json"
PARENT_RUNNER=ROOT/"ops/run_iswas_mlp8_downstream_response_greedy_program_v1.py"
CAPABILITY=ROOT/"circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
BUILDER=ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11.py"
OUT=ROOT/"circuits/followups/iswas_mlp8_attn15_remainder_head_greedy_v1_result.json"
CANDIDATE_ID="cross_task.iswas_mlp8_attn15_remainder_head_greedy_v1"
EXPECTED={"prior":"457d00d9c62a99a496ac169ca85f2474f132d770e1a4b4b9255294e31ad5434d","parent":"fbd5416376ce9d0a510393a3685a6c733aa071efed2340264172123fe6b6ab54","parent_runner":"90354f9524f0281c530ab5c68fff45ab98c65e4d6cd83e25448ae3a81ccf05f0","capability":"6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2","builder":"fbd47713fafcb87fc30ba339d175f7d06770ce36b93b6035e8848455529344ec"}
HEADS=(0,1,2,3,4,6,7,8);FIXED_COMPONENTS=("mlp10","mlp11","mlp12","mlp13","mlp14");TARGET_COMPONENTS=FIXED_COMPONENTS+("attn15_remainder",)
MAX_STEPS=5
MAX_FORWARDS,MAX_EVALUATIONS,SELECTION_CANDIDATES=46,718,30

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")

def run_heads(backend,batch,base_hidden,complement,positions,base,live,heads,*,actuate=True):
    heads=set(heads);handles=[];n_head=int(backend.model.config.n_head);head_dim=int(backend.model.config.n_embd//n_head)
    for layer in composition.ATTN_CLAMPS:
        def patch(_module,args,layer=layer):
            raw=args[0];changed=raw.clone().view(raw.shape[0],raw.shape[1],n_head,head_dim);b=base["attn"][layer].view_as(changed);l=live["attn"][layer].view_as(changed)
            for i,query in enumerate(batch.semantic_positions):
                q=slice(None,int(query)+1);changed[i,q]=b[i,q].to(changed);seed=list(composition.SELECTED.get(layer,()))
                if seed:changed[i,q,seed]=l[i,q,seed].to(changed)
                if layer==15 and heads:
                    hs=sorted(heads);changed[i,q,hs]=l[i,q,hs].to(changed)
            return (changed.reshape_as(raw),)+tuple(args[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    for layer in composition.MLP_CLAMPS:
        def patch(_module,_args,output,layer=layer):
            changed=output.clone()
            for i,query in enumerate(batch.semantic_positions):
                q=slice(None,int(query)+1);value=live["mlp"][layer][i,q] if layer==9 or f"mlp{layer}" in FIXED_COMPONENTS else base["mlp"][layer][i,q];changed[i,q]=value.to(changed)
            return changed
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch))
    if actuate:
        import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(converter.actuation_hook(base_hidden,complement,positions)))
    try:return backend.native(batch,capture=True)
    finally:
        for h in handles:h.remove()

def main():
    paths={"prior":PRIOR,"parent":PARENT,"parent_runner":PARENT_RUNNER,"capability":CAPABILITY,"builder":BUILDER}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("attention15 head authority changed")
    prior,parent,capability,subspace=[json.loads(p.read_text()) for p in (PRIOR,PARENT,CAPABILITY,weight.SUBSPACE)]
    allowed={x for ids in capability["jointly_capable_row_ids"].values() for x in ids};rows=[r for r in candidate.build_rows() if r["family"] in ("A1","A2") and r["row_id"] in allowed];discovery,confirmation=parent_runner.split_rows(rows)
    if prior.get("candidate_id")!=CANDIDATE_ID or parent.get("terminal")!="screen" or parent.get("selected_union")!=["mlp13","mlp10","mlp11","mlp14","mlp12","attn15_remainder"] or len(discovery)!=16 or len(confirmation)!=14:raise RuntimeError("parent or split changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"head_pool":list(HEADS),"fixed_components":list(FIXED_COMPONENTS),"discovery_rows":16,"confirmation_rows":14,"maximum_steps":MAX_STEPS,"selection_candidates":SELECTION_CANDIDATES,"model_forwards_max":MAX_FORWARDS,"example_evaluations_max":MAX_EVALUATIONS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
    family,_s,_e=family_builder.build_family(backend,subspace);gain=math.prod(float(backend.model.transformer.h[k].lambdas[0].detach().float()) for k in range(12,18));modes,orientation_error,_wrong=overlap.residual_modes(backend,family[8],gain);q8=torch.linalg.qr(modes,mode="reduced").Q
    batch,bh,comp,pos,base_output,base,live=parent_runner.prepare(backend,discovery,q8);base18=parent_runner.state(base_output,discovery,backend);target=parent_runner.run_components(backend,batch,bh,comp,pos,base,live,TARGET_COMPONENTS);target18=parent_runner.state(target,discovery,backend)
    self_output=run_heads(backend,batch,bh,comp,pos,base,base,(),actuate=False);self_error=float((parent_runner.state(self_output,discovery,backend)-base18).abs().max())
    selected=[];remaining=list(HEADS);trace=[];prefix=[{"step":0,"selected":[],"objective":parent_runner.losses(backend,discovery,base18,target18,parent_runner.state(run_heads(backend,batch,bh,comp,pos,base,live,()),discovery,backend),q8)[0]}]
    for step in range(1,MAX_STEPS+1):
        trials=[]
        for head in remaining:
            union=tuple(selected+[head]);output=run_heads(backend,batch,bh,comp,pos,base,live,union);objective,panels=parent_runner.losses(backend,discovery,base18,target18,parent_runner.state(output,discovery,backend),q8);trials.append({"head":head,"objective":objective,"panels":panels})
        winner=min(trials,key=lambda x:(x["objective"],HEADS.index(x["head"])));selected.append(winner["head"]);remaining.remove(winner["head"]);trace.append({"step":step,"winner":winner["head"],"winner_objective":winner["objective"],"candidate_objectives":{str(x["head"]):x["objective"] for x in trials}});prefix.append({"step":step,"selected":list(selected),"objective":winner["objective"]})
    best_prefix=min(prefix,key=lambda x:(x["objective"],x["step"]));selected_heads=tuple(best_prefix["selected"]);best_singleton=(trace[0]["winner"],)
    cbatch,cbh,ccomp,cpos,cbase_output,cbase,clive=parent_runner.prepare(backend,confirmation,q8);cbase18=parent_runner.state(cbase_output,confirmation,backend);ctarget=parent_runner.run_components(backend,cbatch,cbh,ccomp,cpos,cbase,clive,TARGET_COMPONENTS);ctarget18=parent_runner.state(ctarget,confirmation,backend)
    arm_heads={"mlp_chain":(),"best_singleton":best_singleton,"selected_heads":selected_heads,"full_remainder":HEADS};reports={};states={}
    for arm,heads in arm_heads.items():
        output=run_heads(backend,cbatch,cbh,ccomp,cpos,cbase,clive,heads);states[arm]=parent_runner.state(output,confirmation,backend);reports[arm]=parent_runner.report(backend,confirmation,cbase18,ctarget18,states[arm],q8)
    replay=float((states["full_remainder"]-ctarget18).norm()/(ctarget18-cbase18).norm());combined=lambda arm:sum(reports[arm][p][k]**2 for p in ("A1","A2") for k in ("behavior_relative_rmse","q8_relative_rmse"));finite=all(math.isfinite(x) for arm in reports.values() for panel in arm.values() for x in panel.values())
    pred_a=orientation_error<=1e-6 and self_error<=.05 and finite and MAX_FORWARDS==46 and MAX_EVALUATIONS==718
    pred_b=combined("selected_heads")<combined("mlp_chain") and combined("selected_heads")<=combined("best_singleton")
    pred_c=all(reports["selected_heads"][p]["behavior_abs_fraction"]>=.90 and reports["selected_heads"][p]["behavior_relative_rmse"]<=.20 for p in ("A1","A2"))
    pred_d=all(reports["selected_heads"][p]["q8_norm_fraction"]>=.90 and reports["selected_heads"][p]["q8_relative_rmse"]<=.25 for p in ("A1","A2"))
    pred_e=len(selected_heads)<=MAX_STEPS and replay<=2e-4 and sum(len(HEADS)-i for i in range(MAX_STEPS))==SELECTION_CANDIDATES
    predictions={"pred_a_authority_split_exactness_finiteness_and_price":bool(pred_a),"pred_b_selected_heads_improve_mlp_chain_and_singleton_on_confirmation":bool(pred_b),"pred_c_selected_heads_preserve_behavior_on_both_panels":bool(pred_c),"pred_d_selected_heads_preserve_q8_on_both_panels":bool(pred_d),"pred_e_zero_fit_small_head_program_and_full_replay":bool(pred_e)};terminal="invalid" if not pred_a or not pred_e else "screen" if all(predictions.values()) else "distributed_heads_or_negligible_correction"
    result={"schema":"iswas_mlp8_attn15_remainder_head_greedy_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"f_linear_orientation_max_abs":orientation_error,"base_self_clamp_max_abs":self_error,"full_remainder_target_relative_norm":replay},"greedy_trace":trace,"prefixes":prefix,"best_prefix":best_prefix,"best_singleton":list(best_singleton),"selected_heads":list(selected_heads),"confirmation_reports":reports,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":MAX_FORWARDS,"example_evaluations":MAX_EVALUATIONS,"selection_candidates":SELECTION_CANDIDATES,"fit_updates":0,"transformer_backwards":0,"model_updates":0}}
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("candidate_id","instrument","best_prefix","best_singleton","selected_heads","confirmation_reports","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__":main()
