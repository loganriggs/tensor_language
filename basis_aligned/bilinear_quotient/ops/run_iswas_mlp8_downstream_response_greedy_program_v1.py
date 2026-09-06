#!/usr/bin/env python3
"""Greedily compress the distributed downstream response branch."""

# BQGATE: EXPERIMENT pred_a_authority_split_exactness_finiteness_and_price pred_b_selected_union_beats_seed_and_singleton_on_confirmation pred_c_selected_union_recovers_behavior_on_both_panels pred_d_selected_union_recovers_q8_on_both_panels pred_e_zero_fit_small_program_and_full_pool_replay
from datetime import datetime, timezone
import hashlib,json,math,os,time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_mlp8_main_mlp9_aux_composition_v1 as composition
import run_iswas_mlp8_missing_response_group_factorial_v1 as group_factorial
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/iswas_mlp8_downstream_response_greedy_program_v1.json"
PARENT=ROOT/"circuits/followups/iswas_mlp8_missing_response_group_factorial_v1_result.json"
PARENT_RUNNER=ROOT/"ops/run_iswas_mlp8_missing_response_group_factorial_v1.py"
CAPABILITY=ROOT/"circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
BUILDER=ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11.py"
OUT=ROOT/"circuits/followups/iswas_mlp8_downstream_response_greedy_program_v1_result.json"
CANDIDATE_ID="cross_task.iswas_mlp8_downstream_response_greedy_program_v1"
EXPECTED={"prior":"478aa51013a6956862d0269a93f31c8349f1e4e4f94a26283c7e25869f022c7a","parent":"bfb7ef88f2eea3b3df1e5e7eb8620aa970d72bc2ec94943fb0d9d96134a5488b",
    "parent_runner":"5bf4fa02f98f9ce7a9389c666c7510cef6212e69d8be9917f82dc3c4602f8929",
    "capability":"6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2",
    "builder":"fbd47713fafcb87fc30ba339d175f7d06770ce36b93b6035e8848455529344ec"}
POOL=("attn10_all","attn11_remainder","attn12_all","attn13_all","attn14_all","attn15_remainder","mlp10","mlp11","mlp12","mlp13","mlp14")
MAX_STEPS=6
MAX_FORWARDS,MAX_EVALUATIONS,SELECTION_CANDIDATES=66,1038,51

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def cosine(x,y):
    d=float(x.norm()*y.norm());return float((x*y).sum())/d if d else 0.0

def split_rows(rows):
    discovery=[];confirmation=[]
    for panel in ("A1","A2"):
        panel_rows=[r for r in rows if r["family"]==panel]
        discovery.extend(panel_rows[:8]);confirmation.extend(panel_rows[8:])
    return discovery,confirmation

def run_components(backend,batch,base_hidden,complement,positions,base,live,components,*,actuate=True):
    components=set(components);handles=[];n_head=int(backend.model.config.n_head);head_dim=int(backend.model.config.n_embd//n_head)
    for layer in composition.ATTN_CLAMPS:
        def patch(_module,args,layer=layer):
            raw=args[0];changed=raw.clone().view(raw.shape[0],raw.shape[1],n_head,head_dim)
            b=base["attn"][layer].view_as(changed);l=live["attn"][layer].view_as(changed)
            for i,query in enumerate(batch.semantic_positions):
                q=slice(None,int(query)+1);changed[i,q]=b[i,q].to(changed)
                seed_heads=composition.SELECTED.get(layer,())
                if seed_heads:
                    hs=list(seed_heads);changed[i,q,hs]=l[i,q,hs].to(changed)
                all_label=f"attn{layer}_all";remainder_label=f"attn{layer}_remainder"
                if all_label in components:changed[i,q]=l[i,q].to(changed)
                elif remainder_label in components:
                    hs=[h for h in range(n_head) if h not in seed_heads];changed[i,q,hs]=l[i,q,hs].to(changed)
            return (changed.reshape_as(raw),)+tuple(args[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    for layer in composition.MLP_CLAMPS:
        def patch(_module,_args,output,layer=layer):
            changed=output.clone()
            for i,query in enumerate(batch.semantic_positions):
                q=slice(None,int(query)+1);value=base["mlp"][layer][i,q]
                if layer==9 or f"mlp{layer}" in components:value=live["mlp"][layer][i,q]
                changed[i,q]=value.to(changed)
            return changed
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch))
    if actuate:handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(converter.actuation_hook(base_hidden,complement,positions)))
    try:return backend.native(batch,capture=True)
    finally:
        for h in handles:h.remove()

def prepare(backend,rows,q8):
    base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor")
    _bo,base_hidden=weight.capture_mlp8(backend,base_batch);_do,donor_hidden=weight.capture_mlp8(backend,donor_batch)
    down=backend.model.transformer.h[8].mlp.Down.weight.detach().float();_u,_s,vh=backend.torch.linalg.svd(q8.T@down,full_matrices=False)
    hd=donor_hidden["hidden"].float()-base_hidden["hidden"].float();complement=hd-weight.project(hd,vh,vh.shape[0])
    positions=[weight.postcue_positions(r) for r in rows]
    base_output,base=composition.capture_modules(backend,lambda:backend.native(base_batch,capture=True))
    _live_output,live=composition.capture_modules(backend,lambda:weight.run_hidden_patch(backend,base_batch,base_hidden["hidden"],complement,positions))
    return base_batch,base_hidden["hidden"],complement,positions,base_output,base,live

def state(output,rows,backend):return composition.state(output,rows,backend)
def margins(backend,x,rows):
    index=backend.torch.arange(len(rows),device=backend.device);answers=backend.torch.as_tensor([r["donor_answer_id"] for r in rows],device=backend.device);foils=backend.torch.as_tensor([r["donor_foil_id"] for r in rows],device=backend.device)
    logits=das.head_logits(backend,x);return logits[index,answers]-logits[index,foils]

def losses(backend,rows,base18,target18,value18,q8):
    base_margin=margins(backend,base18,rows);target_effect=margins(backend,target18,rows)-base_margin;effect=margins(backend,value18,rows)-base_margin
    target_q=(target18-base18)@q8;q=(value18-base18)@q8;items={}
    for panel in ("A1","A2"):
        mask=backend.torch.as_tensor([r["family"]==panel for r in rows],device=backend.device)
        b=float(((effect[mask]-target_effect[mask])**2).mean()/target_effect[mask].square().mean())
        z=float(((q[mask]-target_q[mask])**2).mean()/target_q[mask].square().mean())
        items[panel]={"behavior_relative_mse":b,"q8_relative_mse":z}
    objective=sum(.25*(v["behavior_relative_mse"]+v["q8_relative_mse"]) for v in items.values())
    return objective,items

def report(backend,rows,base18,target18,value18,q8):
    base_margin=margins(backend,base18,rows);target_effect=margins(backend,target18,rows)-base_margin;effect=margins(backend,value18,rows)-base_margin
    target_q=(target18-base18)@q8;q=(value18-base18)@q8;result={}
    for panel in ("A1","A2"):
        mask=backend.torch.as_tensor([r["family"]==panel for r in rows],device=backend.device);te=target_effect[mask];e=effect[mask];tq=target_q[mask];z=q[mask]
        result[panel]={"behavior_cosine":cosine(e,te),"behavior_abs_fraction":float(e.abs().mean()/te.abs().mean()),"behavior_relative_rmse":float(backend.torch.sqrt(((e-te)**2).mean()/te.square().mean())),
            "q8_cosine":cosine(z.reshape(-1),tq.reshape(-1)),"q8_norm_fraction":float(z.norm()/tq.norm()),"q8_relative_rmse":float(backend.torch.sqrt(((z-tq)**2).mean()/tq.square().mean()))}
    return result

def main():
    paths={"prior":PRIOR,"parent":PARENT,"parent_runner":PARENT_RUNNER,"capability":CAPABILITY,"builder":BUILDER}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("greedy response authority changed")
    prior,parent,capability,subspace=[json.loads(p.read_text()) for p in (PRIOR,PARENT,CAPABILITY,weight.SUBSPACE)]
    allowed={x for ids in capability["jointly_capable_row_ids"].values() for x in ids};rows=[r for r in candidate.build_rows() if r["family"] in ("A1","A2") and r["row_id"] in allowed]
    discovery,confirmation=split_rows(rows)
    if prior.get("candidate_id")!=CANDIDATE_ID or parent.get("terminal")!="distributed_downstream" or len(discovery)!=16 or len(confirmation)!=14 or set(r["row_id"] for r in discovery)&set(r["row_id"] for r in confirmation):raise RuntimeError("parent or split changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"pool":list(POOL),"discovery_rows":16,"confirmation_rows":14,"maximum_steps":MAX_STEPS,"selection_candidates":SELECTION_CANDIDATES,"model_forwards_max":MAX_FORWARDS,"example_evaluations_max":MAX_EVALUATIONS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
    family,_s,_e=family_builder.build_family(backend,subspace);gain=math.prod(float(backend.model.transformer.h[k].lambdas[0].detach().float()) for k in range(12,18));modes,orientation_error,_wrong=overlap.residual_modes(backend,family[8],gain);q8=torch.linalg.qr(modes,mode="reduced").Q
    batch,bh,comp,pos,base_output,base,live=prepare(backend,discovery,q8);base18=state(base_output,discovery,backend)
    target_output=group_factorial.run(backend,batch,bh,comp,pos,base,live,("downstream_response_remainder",));target18=state(target_output,discovery,backend)
    self_output=run_components(backend,batch,bh,comp,pos,base,base,(),actuate=False);self_error=float((state(self_output,discovery,backend)-base18).abs().max())
    selected=[];remaining=list(POOL);trace=[];prefix=[{"step":0,"selected":[],"objective":losses(backend,discovery,base18,target18,base18,q8)[0]}]
    for step in range(1,MAX_STEPS+1):
        trials=[]
        for component in remaining:
            union=tuple(selected+[component]);output=run_components(backend,batch,bh,comp,pos,base,live,union);objective,panels=losses(backend,discovery,base18,target18,state(output,discovery,backend),q8)
            trials.append({"component":component,"objective":objective,"panels":panels})
        winner=min(trials,key=lambda x:(x["objective"],POOL.index(x["component"])));selected.append(winner["component"]);remaining.remove(winner["component"])
        trace.append({"step":step,"winner":winner["component"],"winner_objective":winner["objective"],"candidate_objectives":{x["component"]:x["objective"] for x in trials}});prefix.append({"step":step,"selected":list(selected),"objective":winner["objective"]})
    best_prefix=min(prefix,key=lambda x:(x["objective"],x["step"]));selected_union=tuple(best_prefix["selected"]);best_singleton=(trace[0]["winner"],)
    cbatch,cbh,ccomp,cpos,cbase_output,cbase,clive=prepare(backend,confirmation,q8);cbase18=state(cbase_output,confirmation,backend)
    ctarget_output=group_factorial.run(backend,cbatch,cbh,ccomp,cpos,cbase,clive,("downstream_response_remainder",));ctarget18=state(ctarget_output,confirmation,backend)
    arm_components={"seed":(),"best_singleton":best_singleton,"selected_union":selected_union,"full_pool":POOL};reports={};states={}
    for arm,components in arm_components.items():
        output=run_components(backend,cbatch,cbh,ccomp,cpos,cbase,clive,components);states[arm]=state(output,confirmation,backend);reports[arm]=report(backend,confirmation,cbase18,ctarget18,states[arm],q8)
    replay=float((states["full_pool"]-ctarget18).norm()/(ctarget18-cbase18).norm());combined=lambda arm:sum(reports[arm][p][k]**2 for p in ("A1","A2") for k in ("behavior_relative_rmse","q8_relative_rmse"))
    finite=all(math.isfinite(x) for arm in reports.values() for panel in arm.values() for x in panel.values());forwards,evaluations=MAX_FORWARDS,MAX_EVALUATIONS
    pred_a=orientation_error<=1e-6 and self_error<=.05 and finite and forwards<=MAX_FORWARDS and evaluations<=MAX_EVALUATIONS
    pred_b=combined("selected_union")<combined("seed") and combined("selected_union")<=combined("best_singleton")
    pred_c=all(reports["selected_union"][p]["behavior_abs_fraction"]>=.75 and reports["selected_union"][p]["behavior_relative_rmse"]<=.35 for p in ("A1","A2"))
    pred_d=all(reports["selected_union"][p]["q8_norm_fraction"]>=.70 and reports["selected_union"][p]["q8_relative_rmse"]<=.40 for p in ("A1","A2"))
    pred_e=len(selected_union)<=MAX_STEPS and replay<=2e-4 and sum(len(POOL)-i for i in range(MAX_STEPS))==SELECTION_CANDIDATES
    predictions={"pred_a_authority_split_exactness_finiteness_and_price":bool(pred_a),"pred_b_selected_union_beats_seed_and_singleton_on_confirmation":bool(pred_b),"pred_c_selected_union_recovers_behavior_on_both_panels":bool(pred_c),"pred_d_selected_union_recovers_q8_on_both_panels":bool(pred_d),"pred_e_zero_fit_small_program_and_full_pool_replay":bool(pred_e)}
    terminal="invalid" if not pred_a or not pred_e else "screen" if all(predictions.values()) else "insufficient_greedy_program"
    result={"schema":"iswas_mlp8_downstream_response_greedy_program_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"f_linear_orientation_max_abs":orientation_error,"base_self_clamp_max_abs":self_error,"full_pool_target_relative_norm":replay,"discovery_confirmation_overlap":0},"greedy_trace":trace,"prefixes":prefix,"best_prefix":best_prefix,"best_singleton":list(best_singleton),"selected_union":list(selected_union),"confirmation_reports":reports,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":forwards,"example_evaluations":evaluations,"selection_candidates":SELECTION_CANDIDATES,"fit_updates":0,"transformer_backwards":0,"model_updates":0}}
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("candidate_id","instrument","best_prefix","best_singleton","selected_union","confirmation_reports","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__":main()
