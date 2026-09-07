#!/usr/bin/env python3
"""Fresh leave-one-component-out audit of the release-candidate program."""

# BQGATE: EXPERIMENT pred_a_authority_capability_self_clamp_finiteness_and_price pred_b_complete_program_matches_frozen_parent pred_c_every_frozen_component_has_nonzero_necessity pred_d_every_downstream_mlp_is_material_on_both_panels pred_e_zero_fit_exhaustive_component_inventory
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_mlp8_main_mlp9_aux_composition_v1 as composition
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/iswas_mlp8_parsimonious_program_fresh_v12_leave_one_out_v1.json"
PARENT=ROOT/"circuits/followups/iswas_mlp8_parsimonious_program_fresh_v12_confirmation_v1_result.json"
PARENT_RUNNER=ROOT/"ops/run_iswas_mlp8_parsimonious_program_fresh_v12_confirmation_v1.py"
CAPABILITY=ROOT/"circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
BUILDER=ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
OUT=ROOT/"circuits/followups/iswas_mlp8_parsimonious_program_fresh_v12_leave_one_out_v1_result.json"
CANDIDATE_ID="cross_task.iswas_mlp8_parsimonious_program_fresh_v12_leave_one_out_v1"
EXPECTED={"prior":"a96f549ac1f57f9ce8f1b7aace60acd4141d9ba16a0eeee6623881294bd80b96","parent":"3c8db411ee92701337df613870b139fc6c0d745e79ff7272280380b753108ebf","parent_runner":"53f43e47327e73a944d48d9e98e452a296779df1a1d948fdab68cce7175afb11","capability":"67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4","builder":"2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2"}
COMPONENTS=("attn9_main_h1h4","mlp9","attn11_aux_h1h3","mlp10","mlp11","mlp12","mlp13","mlp14","attn15_aux_h5","attn15_late_h1")
HEAD_COMPONENTS={"attn9_main_h1h4":(9,(1,4)),"attn11_aux_h1h3":(11,(1,3)),"attn15_aux_h5":(15,(5,)),"attn15_late_h1":(15,(1,))}
MAX_FORWARDS,MAX_EVALUATIONS=16,480

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def cosine(x,y):
    d=float(x.norm()*y.norm());return float((x*y).sum())/d if d else 0.0

def run_program(backend,batch,base_hidden,complement,positions,base,live,omit=frozenset(),*,actuate=True):
    omit=set(omit);handles=[];n_head=int(backend.model.config.n_head);head_dim=int(backend.model.config.n_embd//n_head)
    for layer in composition.ATTN_CLAMPS:
        def patch(_module,args,layer=layer):
            raw=args[0];changed=raw.clone().view(raw.shape[0],raw.shape[1],n_head,head_dim);b=base["attn"][layer].view_as(changed);l=live["attn"][layer].view_as(changed)
            for i,query in enumerate(batch.semantic_positions):
                q=slice(None,int(query)+1);changed[i,q]=b[i,q].to(changed)
                for label,(site,heads) in HEAD_COMPONENTS.items():
                    if site==layer and label not in omit:
                        hs=list(heads);changed[i,q,hs]=l[i,q,hs].to(changed)
            return (changed.reshape_as(raw),)+tuple(args[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    for layer in composition.MLP_CLAMPS:
        def patch(_module,_args,output,layer=layer):
            changed=output.clone();label=f"mlp{layer}"
            for i,query in enumerate(batch.semantic_positions):
                q=slice(None,int(query)+1);changed[i,q]=(live if label not in omit else base)["mlp"][layer][i,q].to(changed)
            return changed
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch))
    if actuate:handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(converter.actuation_hook(base_hidden,complement,positions)))
    try:return backend.native(batch,capture=True)
    finally:
        for h in handles:h.remove()

def main():
    paths={"prior":PRIOR,"parent":PARENT,"parent_runner":PARENT_RUNNER,"capability":CAPABILITY,"builder":BUILDER}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("leave-one-out authority changed")
    prior,parent,capability,subspace=[json.loads(p.read_text()) for p in (PRIOR,PARENT,CAPABILITY,weight.SUBSPACE)];allowed={x for ids in capability["jointly_capable_row_ids"].values() for x in ids};rows=[r for r in candidate.build_rows() if r["family"] in ("A1","A2") and r["row_id"] in allowed]
    if prior.get("candidate_id")!=CANDIDATE_ID or parent.get("terminal")!="release_candidate" or tuple(prior["frozen_components"])!=COMPONENTS or len(rows)!=30:raise RuntimeError("parent, inventory, or rows changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"rows":30,"components":list(COMPONENTS),"model_forwards_max":MAX_FORWARDS,"example_evaluations_max":MAX_EVALUATIONS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch;model=backend.model
    family,_s,_e=family_builder.build_family(backend,subspace);gain=math.prod(float(model.transformer.h[k].lambdas[0].detach().float()) for k in range(12,18));modes,orientation_error,_wrong=overlap.residual_modes(backend,family[8],gain);q8=torch.linalg.qr(modes,mode="reduced").Q;base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor");_bo,bh=weight.capture_mlp8(backend,base_batch);_do,dh=weight.capture_mlp8(backend,donor_batch);down=model.transformer.h[8].mlp.Down.weight.detach().float();_u,_s,vh=torch.linalg.svd(q8.T@down,full_matrices=False);delta=dh["hidden"].float()-bh["hidden"].float();complement=delta-weight.project(delta,vh,vh.shape[0]);positions=[weight.postcue_positions(r) for r in rows]
    base_output,base=composition.capture_modules(backend,lambda:backend.native(base_batch,capture=True));live_output,live=composition.capture_modules(backend,lambda:weight.run_hidden_patch(backend,base_batch,bh["hidden"],complement,positions));program=run_program(backend,base_batch,bh["hidden"],complement,positions,base,live);removed={c:run_program(backend,base_batch,bh["hidden"],complement,positions,base,live,(c,)) for c in COMPONENTS};self_output=run_program(backend,base_batch,bh["hidden"],complement,positions,base,base,actuate=False)
    state=lambda o:composition.state(o,rows,backend);base18,live18,program18=state(base_output),state(live_output),state(program);removed18={k:state(v) for k,v in removed.items()};self_error=float((state(self_output)-base18).abs().max());index=torch.arange(len(rows),device=backend.device);answers=torch.as_tensor([r["donor_answer_id"] for r in rows],device=backend.device);foils=torch.as_tensor([r["donor_foil_id"] for r in rows],device=backend.device);margin=lambda x:das.head_logits(backend,x)[index,answers]-das.head_logits(backend,x)[index,foils];program_effect=margin(program18)-margin(base18);live_effect=margin(live18)-margin(base18);program_q=(program18-base18)@q8;masks={p:torch.as_tensor([r["family"]==p for r in rows],device=backend.device) for p in ("A1","A2")};reports={}
    for component,value in removed18.items():
        be=margin(program18)-margin(value);z=(program18-value)@q8;reports[component]={}
        for panel,mask in masks.items():reports[component][panel]={"behavior_abs_fraction_of_program":float(be[mask].abs().mean()/program_effect[mask].abs().mean()),"behavior_cosine_to_program":cosine(be[mask],program_effect[mask]),"q8_norm_fraction_of_program":float(z[mask].norm()/program_q[mask].norm()),"q8_cosine_to_program":cosine(z[mask].reshape(-1),program_q[mask].reshape(-1))}
    finite=all(math.isfinite(x) for r in reports.values() for p in r.values() for x in p.values());parent_fraction={p:parent["reports"]["parsimonious"][p]["behavior_abs_fraction_of_live"] for p in masks};current_fraction={p:float(program_effect[masks[p]].abs().mean()/live_effect[masks[p]].abs().mean()) for p in masks};parent_replay=max(abs(current_fraction[p]-parent_fraction[p]) for p in masks)
    pred_a=orientation_error<=1e-6 and self_error<=.05 and finite
    pred_b=parent_replay<=1e-6
    pred_c=all(any(reports[c][p][k]>1e-6 for p in masks for k in ("behavior_abs_fraction_of_program","q8_norm_fraction_of_program")) for c in COMPONENTS)
    pred_d=all((reports[f"mlp{k}"][p]["behavior_abs_fraction_of_program"]>=.02 or reports[f"mlp{k}"][p]["q8_norm_fraction_of_program"]>=.02) for k in range(10,15) for p in masks)
    pred_e=set(reports)==set(COMPONENTS);predictions={"pred_a_authority_capability_self_clamp_finiteness_and_price":bool(pred_a),"pred_b_complete_program_matches_frozen_parent":bool(pred_b),"pred_c_every_frozen_component_has_nonzero_necessity":bool(pred_c),"pred_d_every_downstream_mlp_is_material_on_both_panels":bool(pred_d),"pred_e_zero_fit_exhaustive_component_inventory":bool(pred_e)};terminal="invalid" if not pred_a or not pred_b else "component_resolved_release" if all(predictions.values()) else "redundant_or_nonmaterial_component"
    result={"schema":"iswas_mlp8_parsimonious_program_fresh_v12_leave_one_out_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"f_linear_orientation_max_abs":orientation_error,"base_self_clamp_max_abs":self_error,"parent_behavior_fraction_replay_max_abs":parent_replay},"reports":reports,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":MAX_FORWARDS,"example_evaluations":MAX_EVALUATIONS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}}
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("candidate_id","instrument","reports","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__":main()
