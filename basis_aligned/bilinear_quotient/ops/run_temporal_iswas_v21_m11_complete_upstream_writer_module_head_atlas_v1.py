#!/usr/bin/env python3
"""Exhaustive v21 complete-module/head atlas into the exact M11 writer state."""

# BQGATE: EXPERIMENT pred_a_authority_population_hooks_closures_finiteness_and_exact_price pred_b_complete_upstream_union_is_selective_and_sufficient pred_c_at_least_one_complete_module_is_a_selective_writer pred_d_at_least_one_complete_head_is_a_selective_writer pred_e_weight_ranked_heads_are_causally_enriched
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v21 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v15_all_layer_complete_module_response_atlas_v1 as module_parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v21_m11_complete_upstream_writer_module_head_atlas_v1.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v21_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v21.py"
HR_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_writer_reader_factor_split_v1_result.json"
STATE_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_hr_factorial_state_program_crossfit_v1_result.json"
GAIN_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_hr_gain_factorization_crossfit_v1_result.json"
WEIGHT_PLAN = ROOT / "circuits/prior_art/temporal_iswas_v20_hr_weight_capability_occupancy_causal_atlas_v1.json"
MODULE_PARENT = ROOT / "ops/run_temporal_iswas_v15_all_layer_complete_module_response_atlas_v1.py"
HEAD_PARENT = ROOT / "ops/run_temporal_iswas_v15_attention8_9_11_complete_head_atlas_v1.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v21_m11_complete_upstream_writer_module_head_atlas_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v21_m11_complete_upstream_writer_module_head_atlas_v1"
EXPECTED = {
    "prior": "6a864b70b7f133f6f55c96e884e49e5a89ccec0d5ff5ccf89dca3de524f91251",
    "capability": "9819224e3d83cc99039bbbe64f45e725050aa2381535b2ec655b08ca772a79d7",
    "builder": "4289795a9886c3c36561f47ca202cb790072dd05c324d96ca9fbd5e837a23dc8",
    "hr_result": "e334a580b72806a3976408cc16efb36aa05d066c3a7db28cb74da8f96c5ccc71",
    "state_result": "6825aea2585394e2570317069d13e1de19bc97058d149638e8f46b62f845bb7d",
    "gain_result": "584d3ed65ecb74ae06f58b331a821a7f5958daf89ae6db3a9313eee5ef0120a3",
    "weight_plan": "4467613bd38d62e3ecccd56390054ef6f7b3eb18ef262fcec28e6c247d97fa47",
    "module_parent": "450835763a1db0ea5586265a70775b57551a6820aa7107d4f4c50ea5e2cc3958",
    "head_parent": "eb89c0948b03f9df15ce35cdbf07c8483c87a3404b8dd36efb6ebc1a873b2282",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
}
INPUT = "input_residual"
ATTENTION_LAYERS, MLP_LAYERS, HEADS = tuple(range(12)), tuple(range(11)), tuple(range(9))
MODULE_SITES = (INPUT,) + tuple(
    site for layer in range(12) for site in ((f"attn:{layer}", f"mlp:{layer}") if layer < 11 else (f"attn:{layer}",))
)
HEAD_SITES = tuple(f"L{layer}H{head}" for layer in ATTENTION_LAYERS for head in HEADS)
PANELS = ("A1", "A2", "P", "C")
BARS = {"closure_max_abs": 1e-4, "union_target_recovery": .75, "union_target_direction": .90,
        "singleton_target_recovery": .10, "singleton_target_direction": .75,
        "control_leak_ratio": .15, "localized_head_recovery": .30, "known_head_top_k": 12}
PRICE = {"checkpoint_loads": 1, "model_forwards_exact": 136, "sequence_evaluations_exact": 8704,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = ("pred_a_authority_population_hooks_closures_finiteness_and_exact_price",
    "pred_b_complete_upstream_union_is_selective_and_sufficient",
    "pred_c_at_least_one_complete_module_is_a_selective_writer",
    "pred_d_at_least_one_complete_head_is_a_selective_writer",
    "pred_e_weight_ranked_heads_are_causally_enriched")

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

def capture(backend, batch):
    cache, saved, handles = {}, {}, []
    handles.append(backend.model.transformer.wte.register_forward_hook(
        lambda _module, _arguments, output: cache.__setitem__(INPUT, output.detach().clone())))
    for layer in ATTENTION_LAYERS:
        def save_attention(_module, arguments, layer=layer):
            cache[f"attn:{layer}"] = arguments[0].detach().clone()
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(save_attention))
    for layer in MLP_LAYERS:
        def save_mlp(_module, _arguments, output, layer=layer):
            cache[f"mlp:{layer}"] = output.detach().clone()
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(save_mlp))
    handles.append(backend.model.transformer.h[11].mlp.register_forward_pre_hook(
        lambda _module, arguments: saved.__setitem__("m11_input", arguments[0].detach().clone())))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()
    if set(cache) != set(MODULE_SITES) or "m11_input" not in saved:
        raise RuntimeError("incomplete upstream/M11 capture")
    return output, cache, saved["m11_input"]

def replace_prefix(batch, values, output):
    changed = output.clone()
    for index, query in enumerate(batch.semantic_positions):
        changed[index, :int(query)+1] = values[index, :int(query)+1].to(changed)
    return changed

def run_patch(backend, batch, cache, module_sites=(), head_groups=None):
    head_groups = head_groups or {}; handles, saved = [], {}
    if INPUT in module_sites:
        handles.append(backend.model.transformer.wte.register_forward_hook(
            lambda _module, _arguments, output: replace_prefix(batch, cache[INPUT], output)))
    for site in module_sites:
        if site == INPUT or site.startswith("attn:"): continue
        layer = int(site.split(":")[1])
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(
            lambda _module, _arguments, output, site=site: replace_prefix(batch, cache[site], output)))
    width = int(backend.model.config.n_embd // backend.model.config.n_head)
    for layer, heads in head_groups.items():
        def patch_heads(_module, arguments, layer=layer, heads=tuple(heads)):
            changed = arguments[0].clone(); values = cache[f"attn:{layer}"]
            for index, query in enumerate(batch.semantic_positions):
                stop = int(query)+1
                for head in heads:
                    left, right = head*width, (head+1)*width
                    changed[index, :stop, left:right] = values[index, :stop, left:right].to(changed)
            return (changed,) + tuple(arguments[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch_heads))
    handles.append(backend.model.transformer.h[11].mlp.register_forward_pre_hook(
        lambda _module, arguments: saved.__setitem__("m11_input", arguments[0].detach().clone())))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()
    if "m11_input" not in saved: raise RuntimeError("M11 writer endpoint was not captured")
    return output, saved["m11_input"]

def writer_state(model, x):
    mlp = model.transformer.h[11].mlp
    return (x.float() @ mlp.Left.weight.detach().float().T) * (x.float() @ mlp.Right.weight.detach().float().T)

def vector_metrics(torch, value, target):
    x, y = value.reshape(-1).double(), target.reshape(-1).double()
    xx, yy, xy = float(x@x), float(y@y), float(x@y)
    return {"signed_projection": xy/max(yy,1e-30), "cosine": xy/math.sqrt(max(xx*yy,1e-30)),
            "relative_residual": math.sqrt(float((x-y)@(x-y))/max(yy,1e-30)),
            "norm_ratio": math.sqrt(xx/max(yy,1e-30))}

def behavior_metrics(torch, value, target):
    report = vector_metrics(torch,value,target)
    report["direction_fraction"] = float(((value*target)>0).float().mean())
    return report

def writer_metrics(torch, value, target, mask):
    x, y = value[mask].reshape(-1), target[mask].reshape(-1)
    report = vector_metrics(torch,x,y)
    dots = [(value[row,mask[row]]*target[row,mask[row]]).sum() for row in range(len(value))]
    report["direction_fraction"] = float(torch.stack(dots).gt(0).float().mean())
    return report

def rms_ratio(torch, value, reference, mask=None):
    x = value[mask] if mask is not None else value
    return math.sqrt(float(x.float().square().mean()) / max(float(reference.float().square().mean()),1e-30))

def finite(value):
    if isinstance(value, dict): return all(finite(x) for x in value.values())
    if isinstance(value, (list, tuple)): return all(finite(x) for x in value)
    return not isinstance(value,(int,float)) or isinstance(value,bool) or math.isfinite(float(value))

def main():
    paths = {"prior":PRIOR,"capability":CAPABILITY,"builder":BUILDER,"hr_result":HR_RESULT,
             "state_result":STATE_RESULT,"gain_result":GAIN_RESULT,"weight_plan":WEIGHT_PLAN,
             "module_parent":MODULE_PARENT,"head_parent":HEAD_PARENT,"producer":PRODUCER,"das":DAS}
    observed = {name:sha(path) for name,path in paths.items()}
    capability, hr, state_result, gain = (json.loads(path.read_text()) for path in (CAPABILITY,HR_RESULT,STATE_RESULT,GAIN_RESULT))
    rows = fresh.build_rows(); counts = {p:sum(r["transform_id"]==p for r in rows) for p in PANELS}
    jointly = capability.get("jointly_capable_row_ids",{})
    authority_ok = bool(observed==EXPECTED and counts=={p:16 for p in PANELS} and len({r["row_id"] for r in rows})==64
        and all(len(jointly.get(p,()))==16 for p in ("A1","A2")) and capability.get("terminal")=="screen"
        and hr.get("terminal")=="writer_or_interaction_dominated_context_sign"
        and state_result.get("terminal")=="factorized_coupled_cell_program"
        and gain.get("terminal")=="gain_not_identified_by_marginal_strength"
        and len(MODULE_SITES)==24 and len(HEAD_SITES)==108)
    dry = {"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,
           "queue_touched":False,"authority_ok":authority_ok,"rows":counts,"module_sites":list(MODULE_SITES),
           "head_sites":len(HEAD_SITES),"bars":BARS,"price":PRICE}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":
        print(json.dumps(dry,sort_keys=True)); return
    if not authority_ok: raise RuntimeError("v21 exhaustive writer-atlas authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch; base_batch = das._batch(backend,rows,side="base"); donor_batch = das._batch(backend,rows,side="donor")
    base_output, base_cache, base_x = capture(backend,base_batch)
    donor_output, donor_cache, donor_x = capture(backend,donor_batch); forwards=2
    if not all(base_cache[s].shape==donor_cache[s].shape for s in MODULE_SITES): raise RuntimeError("unaligned caches")
    base_final = module_parent.states(torch,backend,base_output,rows); donor_final = module_parent.states(torch,backend,donor_output,rows)
    answer=torch.as_tensor([r["donor_answer_id"] for r in rows],device=backend.device)
    foil=torch.as_tensor([r["donor_foil_id"] for r in rows],device=backend.device); arange=torch.arange(len(rows),device=backend.device)
    margins=lambda final: das.head_logits(backend,final)[arange,answer]-das.head_logits(backend,final)[arange,foil]
    base_margin, donor_margin = margins(base_final), margins(donor_final)
    base_h, donor_h = writer_state(backend.model,base_x), writer_state(backend.model,donor_x)
    native_behavior, native_h = donor_margin-base_margin, donor_h-base_h
    mask=torch.zeros(base_h.shape[:2],dtype=torch.bool,device=backend.device)
    for i,query in enumerate(base_batch.semantic_positions): mask[i,:int(query)+1]=True
    panel_indices={p:torch.as_tensor([i for i,r in enumerate(rows) if r["transform_id"]==p],device=backend.device) for p in PANELS}
    target_idx=torch.cat((panel_indices["A1"],panel_indices["A2"])); target_behavior=native_behavior[target_idx]
    target_h=native_h[target_idx]; target_mask=mask[target_idx]
    target_behavior_rms=target_behavior.float(); target_h_rms=target_h[target_mask]

    def report(output,x):
        final=module_parent.states(torch,backend,output,rows); behavior=margins(final)-base_margin
        h=writer_state(backend.model,x)-base_h
        result={"target":{"behavior":behavior_metrics(torch,behavior[target_idx],target_behavior),
                           "writer":writer_metrics(torch,h[target_idx],target_h,target_mask)},"cells":{},"controls":{}}
        for panel in PANELS:
            indices=panel_indices[panel]
            for parity in (0,1):
                local=torch.as_tensor([int(rows[int(i)]["group_number"])%2==parity for i in indices],device=backend.device)
                selected=indices[local]; key=f"{panel}:{parity}"
                if panel in ("A1","A2"):
                    result["cells"][key]={"behavior":behavior_metrics(torch,behavior[selected],native_behavior[selected]),
                        "writer":writer_metrics(torch,h[selected],native_h[selected],mask[selected])}
                else:
                    result["cells"][key]={"behavior_leak_ratio":rms_ratio(torch,behavior[selected],target_behavior_rms),
                        "writer_leak_ratio":rms_ratio(torch,h[selected],target_h_rms,mask[selected])}
            if panel in ("P","C"):
                result["controls"][panel]={"behavior_leak_ratio":rms_ratio(torch,behavior[indices],target_behavior_rms),
                    "writer_leak_ratio":rms_ratio(torch,h[indices],target_h_rms,mask[indices])}
        return result,final,behavior,h

    all_attention={layer:HEADS for layer in ATTENTION_LAYERS}
    all_nonattention=tuple(s for s in MODULE_SITES if not s.startswith("attn:"))
    self_output,self_x=run_patch(backend,base_batch,base_cache,all_nonattention,all_attention); forwards+=1
    donor_union_output,donor_union_x=run_patch(backend,base_batch,donor_cache,all_nonattention,all_attention); forwards+=1
    self_report,self_final,self_behavior,self_h=report(self_output,self_x)
    union_report,union_final,union_behavior,union_h=report(donor_union_output,donor_union_x)
    self_error=max(float((self_final-base_final).abs().max()),float(self_behavior.abs().max()),
                   float((self_x-base_x).abs().max()),float(self_h.abs().max()))
    donor_error=max(float((union_final-donor_final).abs().max()),float((union_behavior-native_behavior).abs().max()),
                    float((donor_union_x-donor_x).abs().max()),float((union_h-native_h).abs().max()))
    module_reports={}
    for site in MODULE_SITES:
        if site.startswith("attn:"):
            layer=int(site.split(":")[1]); output,x=run_patch(backend,base_batch,donor_cache,(),{layer:HEADS})
        else: output,x=run_patch(backend,base_batch,donor_cache,(site,),{})
        module_reports[site]=report(output,x)[0]; forwards+=1
    head_reports={}
    for layer in ATTENTION_LAYERS:
        for head in HEADS:
            site=f"L{layer}H{head}"; output,x=run_patch(backend,base_batch,donor_cache,(),{layer:(head,)})
            head_reports[site]=report(output,x)[0]; forwards+=1

    def selective(value,threshold):
        target=value["target"]
        return bool(abs(target["behavior"]["signed_projection"])>=threshold
            and abs(target["writer"]["signed_projection"])>=threshold
            and target["behavior"]["direction_fraction"]>=BARS["singleton_target_direction"]
            and target["writer"]["direction_fraction"]>=BARS["singleton_target_direction"]
            and all(value["controls"][p][kind]<=BARS["control_leak_ratio"] for p in ("P","C")
                    for kind in ("behavior_leak_ratio","writer_leak_ratio")))
    def score(value):
        return min(abs(value["target"]["behavior"]["signed_projection"]),abs(value["target"]["writer"]["signed_projection"]))
    selective_modules=[s for s,v in module_reports.items() if selective(v,BARS["singleton_target_recovery"])]
    selective_heads=[s for s,v in head_reports.items() if selective(v,BARS["singleton_target_recovery"])]
    module_ranking=sorted(MODULE_SITES,key=lambda s:(score(module_reports[s]),s),reverse=True)
    head_ranking=sorted(HEAD_SITES,key=lambda s:(score(head_reports[s]),s),reverse=True)
    union_selective=bool(union_report["target"]["behavior"]["signed_projection"]>=BARS["union_target_recovery"]
        and union_report["target"]["writer"]["signed_projection"]>=BARS["union_target_recovery"]
        and union_report["target"]["behavior"]["direction_fraction"]>=BARS["union_target_direction"]
        and union_report["target"]["writer"]["direction_fraction"]>=BARS["union_target_direction"]
        and all(union_report["controls"][p][kind]<=BARS["control_leak_ratio"] for p in ("P","C")
                for kind in ("behavior_leak_ratio","writer_leak_ratio")))
    A=bool(self_error<=BARS["closure_max_abs"] and donor_error<=BARS["closure_max_abs"]
        and finite([self_report,union_report,module_reports,head_reports]) and forwards==PRICE["model_forwards_exact"]
        and forwards*len(rows)==PRICE["sequence_evaluations_exact"])
    B,C,D=union_selective,bool(selective_modules),bool(selective_heads)
    known={"L7H7","L9H4"}; E=bool(known & set(selective_heads) and known & set(head_ranking[:BARS["known_head_top_k"]]))
    predictions=dict(zip(PREDICTION_KEYS,map(bool,(A,B,C,D,E))))
    localized=[s for s in selective_heads if score(head_reports[s])>=BARS["localized_head_recovery"]]
    terminal=("invalid_instrument" if not A else "upstream_union_not_selective" if not B
              else "localized_upstream_writer_head" if all((C,D,E)) and localized
              else "module_without_head_localization" if C and not D else "distributed_upstream_writer_set")
    result={"schema":"temporal_iswas_v21_m11_complete_upstream_writer_module_head_atlas_result_v1",
        "candidate_id":CANDIDATE_ID,"started_utc":started_utc,"finished_utc":now(),
        "serial_seconds":time.perf_counter()-started,"authority_sha256":observed,
        "evidence_status":"v21_exhaustive_complete_component_causal_writer_atlas",
        "population":{"counts":counts,"row_ids":[r["row_id"] for r in rows]},"self_closure_max_abs":self_error,
        "donor_union_closure_max_abs":donor_error,"union_report":union_report,"module_reports":module_reports,
        "head_reports":head_reports,"selective_modules":selective_modules,"selective_heads":selective_heads,
        "localized_heads":localized,"module_ranking":module_ranking,"head_ranking":head_ranking,
        "known_head_ranks":{s:head_ranking.index(s)+1 for s in sorted(known)},"predictions":predictions,
        "terminal":terminal,"bars":BARS,"price":PRICE,"model_forwards":forwards,
        "greedy_outcome_opened":False,"rank_or_dose_tuned":False}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ("self_closure_max_abs","donor_union_closure_max_abs",
        "union_report","selective_modules","selective_heads","localized_heads","known_head_ranks",
        "module_ranking","head_ranking","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__": main()
