#!/usr/bin/env python3
"""Exact ten-route effect game using a frozen native M11 reader on v22."""

# BQGATE: EXPERIMENT pred_a_authority_reader_hooks_closures_enumeration_finiteness_and_exact_price pred_b_frozen_ten_route_union_is_selective_on_behavior_and_q pred_c_replicated_four_head_union_is_selective pred_d_effect_is_genuinely_distributed pred_e_three_of_four_heads_replicate_as_singletons
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v22 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v22_reader_contracted_writer_effect_game_v1.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v22_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v22.py"
V21_ATLAS = ROOT / "circuits/followups/temporal_iswas_v21_m11_complete_upstream_writer_module_head_atlas_v1_result.json"
V21_AUDIT = ROOT / "circuits/followups/temporal_iswas_v21_m11_complete_upstream_writer_module_head_atlas_v1_instrument_audit.json"
HR_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_writer_reader_factor_split_v1_result.json"
STATE_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_hr_factorial_state_program_crossfit_v1_result.json"
GAIN_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_hr_gain_factorization_crossfit_v1_result.json"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v22_reader_contracted_writer_effect_game_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v22_reader_contracted_writer_effect_game_v1"
EXPECTED = {
    "prior": "45391823f550f1f3f9ea464343d6e42a5223ab8b06ef333965dbba66bc188c42",
    "capability": "692914652431d33389f88818532c6d29aa16b3f1d151ed4728eb7aece7c3b437",
    "builder": "d84f50b334c6a2b2af47273da7668fdea8269bbc615378985fd49547f2f65499",
    "v21_atlas": "b873575f36716a42a32167caa79230d0d6765754afa057e7541e73916282451d",
    "v21_audit": "0fe6ed8b69498cbe68d00457cd797b2ecdb2b8ed06d02855b6e27b1a3cbf2b69",
    "hr_result": "e334a580b72806a3976408cc16efb36aa05d066c3a7db28cb74da8f96c5ccc71",
    "state_result": "6825aea2585394e2570317069d13e1de19bc97058d149638e8f46b62f845bb7d",
    "gain_result": "584d3ed65ecb74ae06f58b331a821a7f5958daf89ae6db3a9313eee5ef0120a3",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
}
ROUTES = ("mlp:1", "mlp:3", "mlp:2", "mlp:4", "mlp:6", "mlp:0",
          "L9H1", "L9H4", "L8H1", "L11H3")
HEAD_ROUTES = ("L9H1", "L9H4", "L8H1", "L11H3")
PANELS = ("A1", "A2", "P", "C")
INPUT = "input_residual"
ATTENTION_LAYERS, MLP_LAYERS, HEADS = tuple(range(12)), tuple(range(11)), tuple(range(9))
ALL_MODULES = (INPUT,) + tuple(site for layer in range(12) for site in
    ((f"attn:{layer}", f"mlp:{layer}") if layer < 11 else (f"attn:{layer}",)))
BARS = {"closure_max_abs":1e-4,"closure_h_relative":1e-4,"shapley_efficiency":1e-8,
        "full_union_recovery":.60,"head_union_recovery":.50,"union_direction":.90,
        "union_control_leak":.15,"distributed_shapley":.05,"singleton_behavior_recovery":.15,
        "singleton_q_recovery":.05,"singleton_direction":.75,"singleton_control_leak":.10,
        "singleton_head_count":3}
PRICE = {"checkpoint_loads":1,"model_forwards_exact":1027,"sequence_evaluations_exact":65728,
         "transformer_backwards":1,"model_updates":0,"fit_parameters":0}
PREDICTION_KEYS = ("pred_a_authority_reader_hooks_closures_enumeration_finiteness_and_exact_price",
    "pred_b_frozen_ten_route_union_is_selective_on_behavior_and_q",
    "pred_c_replicated_four_head_union_is_selective","pred_d_effect_is_genuinely_distributed",
    "pred_e_three_of_four_heads_replicate_as_singletons")

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")

def full_forward(backend,batch):
    torch,F,model=backend.torch,backend.F,backend.model
    tokens,lengths=backend._tensor_batch(batch); x=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)); x0,v1=x,None
    for block in model.transformer.h: x,v1=block(x,v1,x0)
    logits=30.0*torch.tanh(model.lm_head(F.rms_norm(x,(model.config.n_embd,)))/30.0)
    return logits,lengths,x

def capture_native(backend,batch,make_leaf=False):
    cache,saved,calls,handles={}, {}, {"m11_input":0,"leaf":0}, []
    handles.append(backend.model.transformer.wte.register_forward_hook(
        lambda _m,_a,o: cache.__setitem__(INPUT,o.detach().clone())))
    for layer in ATTENTION_LAYERS:
        def save_attention(_m,args,layer=layer): cache[f"attn:{layer}"]=args[0].detach().clone()
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(save_attention))
    for layer in MLP_LAYERS:
        def save_mlp(_m,_a,o,layer=layer): cache[f"mlp:{layer}"]=o.detach().clone()
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(save_mlp))
    def save_input(_m,args): calls["m11_input"]+=1; saved["x"]=args[0].detach().clone()
    handles.append(backend.model.transformer.h[11].mlp.register_forward_pre_hook(save_input))
    if make_leaf:
        def leaf_hook(_m,_a,o):
            calls["leaf"]+=1; leaf=o.detach().requires_grad_(True); saved["leaf"]=leaf; return leaf
        handles.append(backend.model.transformer.h[11].mlp.register_forward_hook(leaf_hook))
    try: logits,lengths,final=full_forward(backend,batch)
    finally:
        for handle in handles: handle.remove()
    if set(cache)!=set(ALL_MODULES) or calls["m11_input"]!=1 or calls["leaf"]!=int(make_leaf):
        raise RuntimeError("native writer capture incomplete")
    return logits,lengths,final,cache,saved,calls

def replace_prefix(batch,values,output):
    changed=output.clone()
    for index,query in enumerate(batch.semantic_positions): changed[index,:int(query)+1]=values[index,:int(query)+1].to(changed)
    return changed

def parse_head(site): return int(site[1:site.index("H")]),int(site[site.index("H")+1:])

def run_patch(backend,batch,cache,routes=(),all_sites=False):
    routes=set(routes); saved={}; handles=[]
    module_sites=set(ALL_MODULES if all_sites else (s for s in routes if s.startswith("mlp:")))
    head_groups={}
    if all_sites: head_groups={layer:HEADS for layer in ATTENTION_LAYERS}
    else:
        for site in routes:
            if site.startswith("L"):
                layer,head=parse_head(site); head_groups.setdefault(layer,[]).append(head)
    if INPUT in module_sites:
        handles.append(backend.model.transformer.wte.register_forward_hook(
            lambda _m,_a,o:replace_prefix(batch,cache[INPUT],o)))
    for site in module_sites:
        if not site.startswith("mlp:"): continue
        layer=int(site.split(":")[1])
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(
            lambda _m,_a,o,site=site:replace_prefix(batch,cache[site],o)))
    width=int(backend.model.config.n_embd//backend.model.config.n_head)
    for layer,heads in head_groups.items():
        def patch_heads(_m,args,layer=layer,heads=tuple(heads)):
            changed=args[0].clone(); values=cache[f"attn:{layer}"]
            for index,query in enumerate(batch.semantic_positions):
                for head in heads:
                    left,right=head*width,(head+1)*width
                    changed[index,:int(query)+1,left:right]=values[index,:int(query)+1,left:right].to(changed)
            return (changed,)+tuple(args[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch_heads))
    handles.append(backend.model.transformer.h[11].mlp.register_forward_pre_hook(
        lambda _m,args:saved.__setitem__("x",args[0].detach().clone())))
    try: logits,lengths,final=full_forward(backend,batch)
    finally:
        for handle in handles: handle.remove()
    if "x" not in saved: raise RuntimeError("patched M11 input missing")
    return logits,lengths,final,saved["x"]

def margin(torch,logits,lengths,rows,device):
    indices=torch.arange(len(rows),device=device); positions=torch.as_tensor([n-1 for n in lengths],device=device)
    answer=torch.as_tensor([r["donor_answer_id"] for r in rows],device=device)
    foil=torch.as_tensor([r["donor_foil_id"] for r in rows],device=device)
    return logits[indices,positions,answer]-logits[indices,positions,foil]

def hidden(model,x):
    mlp=model.transformer.h[11].mlp
    return (x.float()@mlp.Left.weight.detach().float().T)*(x.float()@mlp.Right.weight.detach().float().T)

def metrics(torch,value,target):
    x,y=value.reshape(-1).double(),target.reshape(-1).double(); xx,yy,xy=float(x@x),float(y@y),float(x@y)
    return {"signed_projection":xy/max(yy,1e-30),"cosine":xy/math.sqrt(max(xx*yy,1e-30)),
            "relative_residual":math.sqrt(float((x-y)@(x-y))/max(yy,1e-30)),
            "norm_ratio":math.sqrt(xx/max(yy,1e-30)),
            "direction_fraction":float(((value*target)>0).float().mean())}

def rms_ratio(value,target): return math.sqrt(float(value.float().square().mean())/max(float(target.float().square().mean()),1e-30))

def shapley(values):
    values=np.asarray(values,dtype=np.float64); n=len(ROUTES); factorial=math.factorial; phi=np.zeros(n)
    for i in range(n):
        bit=1<<i
        for mask in range(1<<n):
            if mask&bit: continue
            size=mask.bit_count(); weight=factorial(size)*factorial(n-size-1)/factorial(n)
            phi[i]+=weight*(values[mask|bit]-values[mask])
    interactions=np.zeros((n,n))
    for i,j in itertools.combinations(range(n),2):
        bi,bj=1<<i,1<<j
        for mask in range(1<<n):
            if mask&(bi|bj): continue
            size=mask.bit_count(); weight=factorial(size)*factorial(n-size-2)/factorial(n-1)
            interactions[i,j]+=weight*(values[mask|bi|bj]-values[mask|bi]-values[mask|bj]+values[mask])
        interactions[j,i]=interactions[i,j]
    return {"allocations":{ROUTES[i]:float(phi[i]) for i in range(n)},
            "pair_interactions":{f"{ROUTES[i]}|{ROUTES[j]}":float(interactions[i,j]) for i,j in itertools.combinations(range(n),2)},
            "efficiency_residual":float(abs(phi.sum()-(values[-1]-values[0])))}

def finite(value):
    if isinstance(value,dict): return all(finite(x) for x in value.values())
    if isinstance(value,(list,tuple)): return all(finite(x) for x in value)
    return not isinstance(value,(int,float)) or isinstance(value,bool) or math.isfinite(float(value))

def main():
    paths={"prior":PRIOR,"capability":CAPABILITY,"builder":BUILDER,"v21_atlas":V21_ATLAS,"v21_audit":V21_AUDIT,
           "hr_result":HR_RESULT,"state_result":STATE_RESULT,"gain_result":GAIN_RESULT,"producer":PRODUCER,"das":DAS}
    observed={name:sha(path) for name,path in paths.items()}; capability=json.loads(CAPABILITY.read_text())
    rows=fresh.build_rows(); counts={p:sum(r["transform_id"]==p for r in rows) for p in PANELS}
    authority_ok=bool(observed==EXPECTED and counts=={p:16 for p in PANELS} and capability.get("terminal")=="screen"
        and all(capability.get("predictions",{}).values()) and all(len(capability["jointly_capable_row_ids"][p])==16 for p in ("A1","A2"))
        and len(ROUTES)==10 and len(set(ROUTES))==10 and set(HEAD_ROUTES)<=set(ROUTES) and len(ALL_MODULES)==24)
    dry={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,
         "authority_ok":authority_ok,"rows":counts,"routes":ROUTES,"coalitions":1<<len(ROUTES),"bars":BARS,"price":PRICE}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(dry,sort_keys=True)); return
    if not authority_ok: raise RuntimeError("v22 reader-contracted game authority invalid")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=now(),time.perf_counter(); backend=producer.Bilin18TorchBackend.load("cuda"); torch=backend.torch
    for parameter in backend.model.parameters(): parameter.requires_grad_(False)
    base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor")
    base_logits,lengths,base_final,base_cache,base_saved,base_calls=capture_native(backend,base_batch,True)
    base_margin=margin(torch,base_logits,lengths,rows,backend.device); base_margin.sum().backward()
    if base_saved["leaf"].grad is None: raise RuntimeError("frozen native reader gradient missing")
    reader=base_saved["leaf"].grad.detach().float()@backend.model.transformer.h[11].mlp.Down.weight.detach().float()
    with torch.no_grad():
        donor_logits,donor_lengths,donor_final,donor_cache,donor_saved,donor_calls=capture_native(backend,donor_batch,False)
        donor_margin=margin(torch,donor_logits,donor_lengths,rows,backend.device)
        base_h,donor_h=hidden(backend.model,base_saved["x"]),hidden(backend.model,donor_saved["x"])
        mask=torch.zeros(base_h.shape[:2],dtype=torch.bool,device=backend.device)
        for i,query in enumerate(base_batch.semantic_positions): mask[i,:int(query)+1]=True
        native_q=((donor_h-base_h)*reader*mask.unsqueeze(-1)).sum(dim=(1,2)); native_behavior=donor_margin-base_margin.detach()
        indices={p:torch.as_tensor([i for i,r in enumerate(rows) if r["transform_id"]==p],device=backend.device) for p in PANELS}
        target_idx=torch.cat((indices["A1"],indices["A2"])); target_behavior=native_behavior[target_idx]; target_q=native_q[target_idx]
        half_indices={"first":torch.as_tensor([i for i,r in enumerate(rows) if r["transform_id"] in ("A1","A2") and int(r["group_number"])%4<2],device=backend.device),
                      "second":torch.as_tensor([i for i,r in enumerate(rows) if r["transform_id"] in ("A1","A2") and int(r["group_number"])%4>=2],device=backend.device)}

        def report(logits,run_lengths,x):
            behavior=margin(torch,logits,run_lengths,rows,backend.device)-base_margin.detach()
            q=((hidden(backend.model,x)-base_h)*reader*mask.unsqueeze(-1)).sum(dim=(1,2))
            result={"target":{"behavior":metrics(torch,behavior[target_idx],target_behavior),"q":metrics(torch,q[target_idx],target_q)},
                    "cells":{},"controls":{},"halves":{}}
            for panel in ("A1","A2"):
                for parity in (0,1):
                    selected=torch.as_tensor([i for i,r in enumerate(rows) if r["transform_id"]==panel and int(r["group_number"])%2==parity],device=backend.device)
                    result["cells"][f"{panel}:{parity}"]={"behavior":metrics(torch,behavior[selected],native_behavior[selected]),
                                                             "q":metrics(torch,q[selected],native_q[selected])}
            for panel in ("P","C"):
                selected=indices[panel]; result["controls"][panel]={"behavior_leak_ratio":rms_ratio(behavior[selected],target_behavior),
                                                                    "q_leak_ratio":rms_ratio(q[selected],target_q)}
            for name,selected in half_indices.items():
                result["halves"][name]={"behavior":metrics(torch,behavior[selected],native_behavior[selected]),
                                        "q":metrics(torch,q[selected],native_q[selected])}
            return result,behavior,q

        all_attention={layer:HEADS for layer in ATTENTION_LAYERS}
        self_logits,self_lengths,self_final,self_x=run_patch(backend,base_batch,base_cache,all_sites=True)
        closure_logits,closure_lengths,closure_final,closure_x=run_patch(backend,base_batch,donor_cache,all_sites=True)
        self_behavior=margin(torch,self_logits,self_lengths,rows,backend.device)-base_margin.detach()
        self_error=max(float((self_final-base_final.detach()).abs().max()),float(self_behavior.abs().max()),float((self_x-base_saved["x"]).abs().max()))
        closure_behavior=margin(torch,closure_logits,closure_lengths,rows,backend.device)-donor_margin
        closure_abs=max(float((closure_final-donor_final).abs().max()),float(closure_behavior.abs().max()),float((closure_x-donor_saved["x"]).abs().max()))
        closure_h_rel=float(torch.linalg.vector_norm(hidden(backend.model,closure_x)-donor_h)/torch.linalg.vector_norm(donor_h).clamp_min(1e-30))
        forwards=4
        game_names=("behavior","q")+tuple(f"{p}:{d}:{kind}" for p in ("A1","A2") for d in (0,1) for kind in ("behavior","q"))\
                   +tuple(f"half:{half}:{kind}" for half in ("first","second") for kind in ("behavior","q"))
        games={name:np.zeros(1<<len(ROUTES),dtype=np.float64) for name in game_names}
        coalition_summaries=[None]*(1<<len(ROUTES)); key_reports={}; prefix_masks=[]; running=0
        for i in range(len(ROUTES)): running|=1<<i; prefix_masks.append(running)
        key_masks={1<<i for i in range(len(ROUTES))}|set(prefix_masks)|{sum(1<<ROUTES.index(s) for s in HEAD_ROUTES),(1<<len(ROUTES))-1}
        for subset in range(1,1<<len(ROUTES)):
            selected=tuple(ROUTES[i] for i in range(len(ROUTES)) if subset&(1<<i))
            logits,run_lengths,_final,x=run_patch(backend,base_batch,donor_cache,selected,False); arm,behavior,q=report(logits,run_lengths,x); forwards+=1
            games["behavior"][subset]=arm["target"]["behavior"]["signed_projection"]; games["q"][subset]=arm["target"]["q"]["signed_projection"]
            for key,value in arm["cells"].items():
                for kind in ("behavior","q"): games[f"{key}:{kind}"][subset]=value[kind]["signed_projection"]
            for half,value in arm["halves"].items():
                for kind in ("behavior","q"): games[f"half:{half}:{kind}"][subset]=value[kind]["signed_projection"]
            coalition_summaries[subset]={"mask":subset,"routes":selected,"behavior":float(games["behavior"][subset]),
                                         "q":float(games["q"][subset]),"controls":arm["controls"]}
            if subset in key_masks: key_reports[str(subset)]=arm
            if subset%64==0: print(json.dumps({"progress_subset":subset,"of":(1<<len(ROUTES))-1}),flush=True)
    coalition_summaries[0]={"mask":0,"routes":[],"behavior":0.0,"q":0.0,
                            "controls":{"P":{"behavior_leak_ratio":0.0,"q_leak_ratio":0.0},"C":{"behavior_leak_ratio":0.0,"q_leak_ratio":0.0}}}
    game_reports={name:shapley(values) for name,values in games.items()}; max_efficiency=max(v["efficiency_residual"] for v in game_reports.values())
    full_mask=(1<<len(ROUTES))-1; head_mask=sum(1<<ROUTES.index(s) for s in HEAD_ROUTES)
    full_report,head_report=key_reports[str(full_mask)],key_reports[str(head_mask)]
    singleton_reports={ROUTES[i]:key_reports[str(1<<i)] for i in range(len(ROUTES))}
    prefix_reports={str(i+1):key_reports[str(mask)] for i,mask in enumerate(prefix_masks)}
    def selective(report,recovery,control):
        return bool(report["target"]["behavior"]["signed_projection"]>=recovery and report["target"]["q"]["signed_projection"]>=recovery
            and report["target"]["behavior"]["direction_fraction"]>=BARS["union_direction"]
            and report["target"]["q"]["direction_fraction"]>=BARS["union_direction"]
            and all(report["controls"][p][kind]<=control for p in ("P","C") for kind in ("behavior_leak_ratio","q_leak_ratio")))
    singleton_pass={site:bool(value["target"]["behavior"]["signed_projection"]>=BARS["singleton_behavior_recovery"]
        and value["target"]["q"]["signed_projection"]>=BARS["singleton_q_recovery"]
        and value["target"]["behavior"]["direction_fraction"]>=BARS["singleton_direction"]
        and value["target"]["q"]["direction_fraction"]>=BARS["singleton_direction"]
        and all(value["controls"][p][kind]<=BARS["singleton_control_leak"] for p in ("P","C") for kind in ("behavior_leak_ratio","q_leak_ratio")))
        for site,value in singleton_reports.items() if site in HEAD_ROUTES}
    distributed=[]
    for route in ROUTES:
        if (game_reports["behavior"]["allocations"][route]>=BARS["distributed_shapley"]
                and game_reports["q"]["allocations"][route]>=BARS["distributed_shapley"]
                and all(game_reports[f"half:{half}:{kind}"]["allocations"][route]>0
                        for half in ("first","second") for kind in ("behavior","q"))): distributed.append(route)
    A=bool(authority_ok and base_saved["leaf"].grad is not None and float(reader.abs().max())>0
        and self_error<=BARS["closure_max_abs"] and closure_abs<=BARS["closure_max_abs"] and closure_h_rel<=BARS["closure_h_relative"]
        and max_efficiency<=BARS["shapley_efficiency"] and finite([coalition_summaries,key_reports,game_reports])
        and forwards==PRICE["model_forwards_exact"] and forwards*len(rows)==PRICE["sequence_evaluations_exact"])
    B=selective(full_report,BARS["full_union_recovery"],BARS["union_control_leak"])
    C=selective(head_report,BARS["head_union_recovery"],BARS["union_control_leak"])
    D=len(distributed)>=2; E=sum(singleton_pass.values())>=BARS["singleton_head_count"]
    predictions=dict(zip(PREDICTION_KEYS,map(bool,(A,B,C,D,E))))
    terminal=("invalid_instrument" if not A else "reader_contracted_distributed_writer_program" if all((B,C,D,E))
              else "four_head_writer_program" if C and E else "head_singleton_replication_failure" if not E
              else "distributed_nonselective_or_q_mismatch")
    result={"schema":"temporal_iswas_v22_reader_contracted_writer_effect_game_result_v1","candidate_id":CANDIDATE_ID,
        "started_utc":started_utc,"finished_utc":now(),"serial_seconds":time.perf_counter()-started,
        "authority_sha256":observed,"evidence_status":"fresh_v22_reader_contracted_fixed_route_game",
        "population":{"counts":counts,"row_ids":[r["row_id"] for r in rows]},"routes":list(ROUTES),
        "reader_gradient_max_abs":float(reader.abs().max()),"native_q_rms":float(target_q.float().square().mean().sqrt()),
        "closures":{"self_max_abs":self_error,"donor_max_abs":closure_abs,"donor_h_relative":closure_h_rel},
        "coalition_summaries":coalition_summaries,"full_union_report":full_report,"head_union_report":head_report,
        "singleton_reports":singleton_reports,"singleton_pass":singleton_pass,"prefix_reports":prefix_reports,
        "game_reports":game_reports,"max_shapley_efficiency_residual":max_efficiency,"distributed_routes":distributed,
        "predictions":predictions,"terminal":terminal,"bars":BARS,"price":PRICE,"model_forwards":forwards,
        "v22_selected_subset":None,"v22_tuned_rank_or_dose":False,"future_confirmation_required_for_game_selected_subset":True}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ("closures","full_union_report","head_union_report","singleton_pass",
        "distributed_routes","max_shapley_efficiency_residual","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__": main()
