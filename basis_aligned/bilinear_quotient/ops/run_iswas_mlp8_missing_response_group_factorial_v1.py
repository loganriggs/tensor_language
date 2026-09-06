#!/usr/bin/env python3
"""Localize the natural-program deficit to attention9 or downstream responses."""

# BQGATE: EXPERIMENT pred_a_authority_self_clamp_full_replay_finiteness_and_price pred_b_downstream_response_remainder_is_material pred_c_attention9_remainder_is_quantified pred_d_group_interaction_is_measured pred_e_zero_fit_exhaustive_two_group_factorial
from datetime import datetime, timezone
import hashlib,json,math,os,time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_mlp8_main_mlp9_aux_composition_v1 as composition
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/iswas_mlp8_missing_response_group_factorial_v1.json"
PARENT=ROOT/"circuits/followups/iswas_mlp8_main_mlp9_aux_composition_v1_result.json"
PARENT_RUNNER=ROOT/"ops/run_iswas_mlp8_main_mlp9_aux_composition_v1.py"
CAPABILITY=ROOT/"circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
BUILDER=ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11.py"
OUT=ROOT/"circuits/followups/iswas_mlp8_missing_response_group_factorial_v1_result.json"
CANDIDATE_ID="cross_task.iswas_mlp8_missing_response_group_factorial_v1"
GROUPS=("attention9_remainder","downstream_response_remainder")
EXPECTED={"prior":"d4a19c90c0ea1254cd5b79bd10e6f73f10cb1281c701fa36f593109953acbef6","parent":"96a60c343508c553c58102d541f6e525cce2e1c312aeefec61ee181a075cd876",
    "parent_runner":"a32e886d9e92dfd9eb11654b6874ecb509f70e0c5f82b74b90e9c1af67ff08f4",
    "capability":"6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2",
    "builder":"fbd47713fafcb87fc30ba339d175f7d06770ce36b93b6035e8848455529344ec"}
MAX_FORWARDS,MAX_EVALUATIONS=9,270

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def cosine(x,y):
    d=float(x.norm()*y.norm());return float((x*y).sum())/d if d else 0.0
def arm_name(s):return "+".join(s) if s else "seed"

def run(backend,batch,base_hidden,complement,positions,base,live,groups,*,actuate=True):
    groups=set(groups);handles=[];n_head=int(backend.model.config.n_head);head_dim=int(backend.model.config.n_embd//n_head)
    for layer in composition.ATTN_CLAMPS:
        def patch(_module,args,layer=layer):
            raw=args[0];changed=raw.clone().view(raw.shape[0],raw.shape[1],n_head,head_dim)
            b=base["attn"][layer].view_as(changed);l=live["attn"][layer].view_as(changed)
            for i,query in enumerate(batch.semantic_positions):
                q=slice(None,int(query)+1);changed[i,q]=b[i,q].to(changed)
                seed_heads=composition.SELECTED.get(layer,())
                if seed_heads:
                    hs=list(seed_heads);changed[i,q,hs]=l[i,q,hs].to(changed)
                if layer==9 and "attention9_remainder" in groups:
                    hs=[h for h in range(n_head) if h not in seed_heads];changed[i,q,hs]=l[i,q,hs].to(changed)
                if layer!=9 and "downstream_response_remainder" in groups:
                    changed[i,q]=l[i,q].to(changed)
            return (changed.reshape_as(raw),)+tuple(args[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    for layer in composition.MLP_CLAMPS:
        def patch(_module,_args,output,layer=layer):
            changed=output.clone()
            for i,query in enumerate(batch.semantic_positions):
                q=slice(None,int(query)+1);value=base["mlp"][layer][i,q]
                if layer==9 or "downstream_response_remainder" in groups:value=live["mlp"][layer][i,q]
                changed[i,q]=value.to(changed)
            return changed
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch))
    if actuate:handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
        converter.actuation_hook(base_hidden,complement,positions)))
    try:return backend.native(batch,capture=True)
    finally:
        for h in handles:h.remove()

def main():
    paths={"prior":PRIOR,"parent":PARENT,"parent_runner":PARENT_RUNNER,"capability":CAPABILITY,"builder":BUILDER}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("group-factor authority changed")
    prior,parent,capability,subspace=[json.loads(p.read_text()) for p in (PRIOR,PARENT,CAPABILITY,weight.SUBSPACE)]
    allowed={x for ids in capability["jointly_capable_row_ids"].values() for x in ids}
    rows=[r for r in candidate.build_rows() if r["family"] in ("A1","A2") and r["row_id"] in allowed]
    positions=[weight.postcue_positions(r) for r in rows]
    if prior.get("candidate_id")!=CANDIDATE_ID or parent.get("terminal")!="missing_branch" or len(rows)!=30:raise RuntimeError("parent changed")
    subsets=((),(GROUPS[0],),(GROUPS[1],),GROUPS)
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,
        "rows":len(rows),"groups":list(GROUPS),"arms":[arm_name(s) for s in subsets],"model_forwards_max":MAX_FORWARDS,
        "example_evaluations_max":MAX_EVALUATIONS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
    family,_s,_e=family_builder.build_family(backend,subspace);gain=math.prod(float(backend.model.transformer.h[k].lambdas[0].detach().float()) for k in range(12,18))
    modes,orientation_error,_wrong=overlap.residual_modes(backend,family[8],gain);q8=torch.linalg.qr(modes,mode="reduced").Q
    base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor")
    _bo,base_hidden=weight.capture_mlp8(backend,base_batch);_do,donor_hidden=weight.capture_mlp8(backend,donor_batch)
    down=backend.model.transformer.h[8].mlp.Down.weight.detach().float();_u,_s,vh=torch.linalg.svd(q8.T@down,full_matrices=False)
    hd=donor_hidden["hidden"].float()-base_hidden["hidden"].float();complement=hd-weight.project(hd,vh,vh.shape[0])
    base_output,base=composition.capture_modules(backend,lambda:backend.native(base_batch,capture=True))
    live_output,live=composition.capture_modules(backend,lambda:weight.run_hidden_patch(backend,base_batch,base_hidden["hidden"],complement,positions))
    outputs={arm_name(s):run(backend,base_batch,base_hidden["hidden"],complement,positions,base,live,s) for s in subsets}
    self_clamp=run(backend,base_batch,base_hidden["hidden"],complement,positions,base,base,(),actuate=False)
    forwards,evaluations=9,9*len(rows);state=lambda o:composition.state(o,rows,backend)
    base18,live18=state(base_output),state(live_output);states={k:state(v) for k,v in outputs.items()}
    index=torch.arange(len(rows),device=backend.device);answers=torch.as_tensor([r["donor_answer_id"] for r in rows],device=backend.device);foils=torch.as_tensor([r["donor_foil_id"] for r in rows],device=backend.device)
    margin=lambda x:das.head_logits(backend,x)[index,answers]-das.head_logits(backend,x)[index,foils]
    live_effect=margin(live18)-margin(base18);live_q8=(live18-base18)@q8;masks={p:torch.as_tensor([r["family"]==p for r in rows],device=backend.device) for p in ("A1","A2")}
    metrics={p:{} for p in masks}
    for panel,mask in masks.items():
        for name,value in states.items():
            effect=(margin(value)-margin(base18))[mask];coord=((value-base18)@q8)[mask]
            metrics[panel][name]={"behavior_fraction_of_live":float(effect.abs().mean()/live_effect[mask].abs().mean()),"behavior_cosine_to_live":cosine(effect,live_effect[mask]),
                "q8_norm_fraction_of_live":float(coord.norm()/live_q8[mask].norm()),"q8_cosine_to_live":cosine(coord.reshape(-1),live_q8[mask].reshape(-1))}
    full=arm_name(GROUPS);replay=float((states[full]-live18).norm()/(live18-base18).norm());self_error=float((state(self_clamp)-base18).abs().max())
    seed=states["seed"]-base18;attn=states[GROUPS[0]]-states["seed"];downstream=states[GROUPS[1]]-states["seed"]
    interaction=float((states[full]-base18-seed-attn-downstream).norm()/(live18-base18).norm())
    finite=all(math.isfinite(v) for p in metrics.values() for a in p.values() for v in a.values())
    pred_a=orientation_error<=1e-6 and self_error<=.05 and replay<=2e-4 and finite and forwards<=MAX_FORWARDS and evaluations<=MAX_EVALUATIONS
    pred_b=all((metrics[p][GROUPS[1]]["behavior_fraction_of_live"]-metrics[p]["seed"]["behavior_fraction_of_live"]>=.25) or
        (metrics[p][GROUPS[1]]["q8_norm_fraction_of_live"]-metrics[p]["seed"]["q8_norm_fraction_of_live"]>=.25) for p in masks)
    pred_c=all(GROUPS[0] in metrics[p] for p in masks);pred_d=math.isfinite(interaction);pred_e=set(outputs)=={arm_name(s) for s in subsets}
    predictions={"pred_a_authority_self_clamp_full_replay_finiteness_and_price":bool(pred_a),"pred_b_downstream_response_remainder_is_material":bool(pred_b),
        "pred_c_attention9_remainder_is_quantified":bool(pred_c),"pred_d_group_interaction_is_measured":bool(pred_d),"pred_e_zero_fit_exhaustive_two_group_factorial":bool(pred_e)}
    terminal="invalid" if not pred_a else "distributed_downstream" if all(predictions.values()) else "attention9_or_mixed"
    result={"schema":"iswas_mlp8_missing_response_group_factorial_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only",
        "started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,
        "instrument":{"f_linear_orientation_max_abs":orientation_error,"base_self_clamp_max_abs":self_error,"full_group_live_relative_norm":replay,"group_interaction_relative_norm":interaction,"rows":len(rows)},
        "metrics":metrics,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":forwards,"example_evaluations":evaluations,"fit_updates":0,"transformer_backwards":0,"model_updates":0}}
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("candidate_id","instrument","metrics","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__":main()
