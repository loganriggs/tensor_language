#!/usr/bin/env python3
"""Causally test weight-ranked upstream writers with complete response patches."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_self_patch_reconstruction_price pred_b_known_l9_positive_control pred_c_weight_ranking_enriches_causal_writers pred_d_novel_top_writer_is_material pred_e_l7_mode_split_matches_weight_prediction
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path

import numpy as np
import attention_source_destination_eval as attention_eval
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_q8_finite_causal_hankel_v1 as parent
import run_temporal_iswas_two_mode_weight_pullback_v1 as atlas

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_iswas_upstream_full_response_mode_atlas_v1.json"
ATLAS_RESULT=ROOT/"circuits/followups/temporal_iswas_two_mode_weight_pullback_v3_result.json"
ATLAS_RUNNER=ROOT/"ops/run_temporal_iswas_two_mode_weight_pullback_v1.py"
PARENT_RUNNER=ROOT/"ops/run_temporal_iswas_q8_finite_causal_hankel_v1.py"
OUT=ROOT/"circuits/followups/temporal_iswas_upstream_full_response_mode_atlas_v1_result.json"
CANDIDATE_ID="cross_task.temporal_iswas_upstream_full_response_mode_atlas_v1"
LAYERS=(6,7,9)
SITES=tuple([f"L{layer}H{head}" for layer in LAYERS for head in range(9)]+[f"MLP{layer}" for layer in LAYERS])
MAX_FORWARDS,MAX_EVALUATIONS=36,1152
EXPECTED={"prior":"18e9dc5715240980e0d656188b3a8cd4fa6b43b1646ef8fc002be44f28c14fc1","atlas_result":"c8ab608fa116342f9cbc8af4955e6087faa0f1eee9dd74dacb5c0ec168c5bf4d","atlas_runner":"abfedf5b2347bc49a13d7008c4e815a33d6c1629f300d9e720f0876c4e983788","parent_runner":"e9303c6fc1a11af4c103c49c5d47b2fdf0937714a80fd33d0549df2fa7216950"}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def tensor_sha(tensor):return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()
def site_parts(site):
    if site.startswith("MLP"):return "mlp",int(site[3:]),None
    layer,head=site[1:].split("H");return "attn",int(layer),int(head)
def site_module(backend,site):
    kind,layer,_head=site_parts(site);block=backend.model.transformer.h[layer]
    return block.attn.c_proj if kind=="attn" else block.mlp

def capture(backend,batch):
    cache,handles={},[]
    for site in SITES:
        kind,_layer,head=site_parts(site)
        if kind=="attn":
            def save(_module,arguments,site=site):cache[site]=arguments[0].detach().clone()
            handles.append(site_module(backend,site).register_forward_pre_hook(save))
        else:
            def save(_module,_arguments,output,site=site):cache[site]=output.detach().clone()
            handles.append(site_module(backend,site).register_forward_hook(save))
    def save_h3(_module,arguments):cache["pre_h3"]=arguments[0].detach().clone()
    handles.append(backend.model.transformer.h[11].attn.c_v.register_forward_pre_hook(save_h3))
    try:output=backend.native(batch,capture=True)
    finally:
        for handle in handles:handle.remove()
    if set(cache)!=(set(SITES)|{"pre_h3"}):raise RuntimeError("incomplete response capture")
    return output,cache

def patch_hook(batch,values,site,head_width):
    kind,_layer,head=site_parts(site)
    if kind=="attn":
        def hook(_module,arguments):
            changed=arguments[0].clone();start=head*head_width;stop=start+head_width
            for index,query in enumerate(batch.semantic_positions):changed[index,:int(query)+1,start:stop]=values[index,:int(query)+1,start:stop].to(changed)
            return (changed,)+tuple(arguments[1:])
    else:
        def hook(_module,_arguments,output):
            changed=output.clone()
            for index,query in enumerate(batch.semantic_positions):changed[index,:int(query)+1]=values[index,:int(query)+1].to(changed)
            return changed
    return hook

def run_patch(backend,batch,values_by_site,sites):
    handles=[];head_width=int(backend.model.config.n_embd//backend.model.config.n_head);pre_h3=[]
    for site in sites:
        kind,_layer,_head=site_parts(site);hook=patch_hook(batch,values_by_site[site],site,head_width)
        handles.append(site_module(backend,site).register_forward_pre_hook(hook) if kind=="attn" else site_module(backend,site).register_forward_hook(hook))
    handles.append(backend.model.transformer.h[11].attn.c_v.register_forward_pre_hook(lambda _m,a:pre_h3.append(a[0].detach().clone())))
    try:output=backend.native(batch,capture=True)
    finally:
        for handle in handles:handle.remove()
    if len(pre_h3)!=1:raise RuntimeError("pre-H3 capture count changed")
    return output,pre_h3[0]

def states(torch,backend,output,rows):
    return torch.stack([torch.as_tensor(output.captured[(row["row_id"],"resid:18")]) for row in rows]).to(backend.device).float()
def query_rows(tensor,batch):return tensor[list(range(len(batch.row_ids))),list(batch.semantic_positions)].float()
def vector_stats(torch,x,y):
    denominator=float(y@y);xn=float(x.norm());yn=float(y.norm())
    return {"signed_projection":float(x@y)/denominator if denominator else 0.0,"cosine":float(x@y)/(xn*yn) if xn*yn else 0.0,"norm_ratio":xn/yn if yn else 0.0}
def ranks(values):
    order=np.argsort(np.asarray(values),kind="stable");out=np.empty(len(values));out[order]=np.arange(len(values));return out
def spearman(a,b):return float(np.corrcoef(ranks(a),ranks(b))[0,1])

def main():
    paths={"prior":PRIOR,"atlas_result":ATLAS_RESULT,"atlas_runner":ATLAS_RUNNER,"parent_runner":PARENT_RUNNER}
    if {key:sha(value) for key,value in paths.items()}!=EXPECTED:raise RuntimeError("upstream causal atlas authority changed")
    parent_paths={"prior":parent.PRIOR,"shared_causal":parent.SHARED_CAUSAL,"temporal_capability":parent.TEMPORAL_CAPABILITY,"subspace":parent.SUBSPACE,"iswas":parent.ISWAS,"v2_capability":parent.V2_CAPABILITY,"v3_capability":parent.V3_CAPABILITY,"temporal_builder":parent.TEMPORAL_BUILDER,"v2_builder":parent.V2_BUILDER,"v3_builder":parent.V3_BUILDER,"atlas_runner":parent.ATLAS_RUNNER,"analytic_runner":parent.ANALYTIC_RUNNER,"overlap_runner":parent.OVERLAP_RUNNER}
    if {key:sha(value) for key,value in parent_paths.items()}!=parent.EXPECTED:raise RuntimeError("parent factor authorities changed")
    prior,atlas_result=json.loads(PRIOR.read_text()),json.loads(ATLAS_RESULT.read_text())
    if prior.get("candidate_id")!=CANDIDATE_ID or atlas_result.get("terminal")!="mode_specific_weight_screen":raise RuntimeError("authority terminal changed")
    allowed={x for ids in json.loads(parent.TEMPORAL_CAPABILITY.read_text())["jointly_capable_row_ids"].values() for x in ids};all_temporal=parent.temporal.build_rows();temporal=[]
    for panel in ("A1","A2"):temporal.extend([row for row in all_temporal if row["transform_id"]==panel and row["row_id"] in allowed][:8])
    iswas=parent.select_iswas_rows();rows=temporal+iswas
    if len(temporal)!=16 or len(iswas)!=16 or len(SITES)!=30:raise RuntimeError("row or site population changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"rows":32,"sites":list(SITES),"model_forwards_max":MAX_FORWARDS,"example_evaluations_max":MAX_EVALUATIONS,"fit_updates":0,"model_updates":0,"transformer_backwards":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
    base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor")
    base_output,base_cache=capture(backend,base_batch);donor_output,donor_cache=capture(backend,donor_batch)
    aligned=all(base_cache[key].shape==donor_cache[key].shape for key in base_cache)
    if not aligned:raise RuntimeError("base/donor response shapes changed")
    self_output,self_pre=run_patch(backend,base_batch,base_cache,SITES)
    base_state,donor_state,self_state=(states(torch,backend,out,rows) for out in (base_output,donor_output,self_output))
    base_pre,donor_pre=query_rows(base_cache["pre_h3"],base_batch),query_rows(donor_cache["pre_h3"],donor_batch)
    self_pre=query_rows(self_pre,base_batch)
    family,_singular,_energy=parent.family_builder.build_family(backend,json.loads(parent.SUBSPACE.read_text()));q=family[8]
    gain=math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float()) for layer in range(12,18))
    raw_modes,orientation_error,_wrong=parent.overlap.residual_modes(backend,q,gain);state_basis=torch.linalg.qr(raw_modes,mode="reduced").Q
    source_coordinates=torch.as_tensor(atlas_result["mode_artifacts"]["source_coordinates"],device=backend.device).float();physical_source=state_basis@source_coordinates
    physical_hash_ok=tensor_sha(physical_source)==atlas_result["mode_artifacts"]["physical_source_covectors_sha256"]
    width=int(backend.model.transformer.h[11].attn.head_dim);value_rows=backend.model.transformer.h[11].attn.c_v.weight.detach().float()[3*width:4*width]
    h3_inputs=[]
    for mode in range(2):
        dual=raw_modes.T@physical_source[:,mode];h3_inputs.append(value_rows.T@(q@dual))
    h3_inputs=torch.stack(h3_inputs,dim=1)
    answer=torch.as_tensor([row["donor_answer_id"] for row in rows],device=backend.device);foil=torch.as_tensor([row["donor_foil_id"] for row in rows],device=backend.device);index=torch.arange(len(rows),device=backend.device)
    def margin(state):
        logits=das.head_logits(backend,state);return logits[index,answer]-logits[index,foil]
    base_margin,donor_margin=margin(base_state),margin(donor_state)
    task_indices={"temporal":torch.arange(0,16,device=backend.device),"iswas":torch.arange(16,32,device=backend.device),"pooled":torch.arange(32,device=backend.device)}
    weight={mode:{row["label"]:row["score"] for row in atlas_result["upstream_rankings"][mode]} for mode in ("mode1","mode2")}
    reconstruction=0.0
    for layer in LAYERS:
        _replay,captured=attention_eval.capture_layer_attention(backend,base_batch,layer)
        reconstruction=max(reconstruction,float((captured["head_output"].reshape_as(base_cache[f"L{layer}H0"])-base_cache[f"L{layer}H0"]).abs().max()))
    metrics={};forwards,evaluations=6,192
    full_pre=(donor_pre-base_pre)@h3_inputs;full_final=(donor_state-base_state)@physical_source
    for site in SITES:
        output,pre=run_patch(backend,base_batch,donor_cache,(site,));state=states(torch,backend,output,rows);pre=query_rows(pre,base_batch)
        patch_pre=(pre-base_pre)@h3_inputs;patch_final=(state-base_state)@physical_source;recovery=(margin(state)-base_margin)/(donor_margin-base_margin)
        site_result={"behavior":{"mean_recovery":float(recovery.mean()),"mean_absolute_recovery":float(recovery.abs().mean()),"direction_fraction":float((recovery>0).float().mean())},"tasks":{}}
        for task,ids in task_indices.items():
            site_result["tasks"][task]={}
            for mode in range(2):
                site_result["tasks"][task][f"mode{mode+1}"]={"pre_h3":vector_stats(torch,patch_pre[ids,mode],full_pre[ids,mode]),"final_state":vector_stats(torch,patch_final[ids,mode],full_final[ids,mode])}
        metrics[site]=site_result;forwards+=1;evaluations+=32
    self_error=max(float((self_pre-base_pre).abs().max()),float((self_state-base_state).abs().max()),float((margin(self_state)-base_margin).abs().max()))
    correlations={};layer_contrasts={}
    for mode in ("mode1","mode2"):
        causal=[abs(metrics[site]["tasks"]["pooled"][mode]["pre_h3"]["signed_projection"]) for site in SITES];correlations[mode]=spearman([weight[mode][site] for site in SITES],causal)
        layer_contrasts[mode]={}
        for layer in LAYERS:
            heads=[f"L{layer}H{head}" for head in range(9)];high=max(heads,key=lambda site:weight[mode][site]);low=min(heads,key=lambda site:weight[mode][site]);layer_contrasts[mode][str(layer)]={"high":high,"low":low,"high_causal":abs(metrics[high]["tasks"]["pooled"][mode]["pre_h3"]["signed_projection"]),"low_causal":abs(metrics[low]["tasks"]["pooled"][mode]["pre_h3"]["signed_projection"])}
    temporal_rank=sorted(SITES,key=lambda site:max(abs(metrics[site]["tasks"]["temporal"][mode]["pre_h3"]["signed_projection"]) for mode in ("mode1","mode2")),reverse=True)
    known=any(site in temporal_rank[:3] and max(metrics[site]["tasks"]["temporal"][mode]["pre_h3"]["cosine"] for mode in ("mode1","mode2"))>0 for site in ("L9H1","L9H4"))
    enrich=any(correlations[mode]>=.30 and sum(row["high_causal"]>row["low_causal"] for row in layer_contrasts[mode].values())>=2 for mode in ("mode1","mode2"))
    novel=any(metrics[site]["tasks"][task][mode]["pre_h3"]["cosine"]>=.50 and abs(metrics[site]["tasks"][task][mode]["pre_h3"]["signed_projection"])>=.10 for site in ("L6H7","L7H7","L7H8") for task in ("temporal","iswas") for mode in ("mode1","mode2"))
    def selectivity(site):
        a=abs(metrics[site]["tasks"]["temporal"]["mode1"]["pre_h3"]["signed_projection"]);b=abs(metrics[site]["tasks"]["temporal"]["mode2"]["pre_h3"]["signed_projection"]);return a/(b+1e-12)
    mode_split=selectivity("L7H7")>selectivity("L7H8")
    finite=all(math.isfinite(value) for site in metrics.values() for task in site["tasks"].values() for mode in task.values() for family in mode.values() for value in family.values())
    pred_a=aligned and physical_hash_ok and orientation_error<=1e-6 and self_error<=1e-4 and reconstruction<=5e-4 and forwards<=MAX_FORWARDS and evaluations<=MAX_EVALUATIONS and finite
    predictions={"pred_a_authority_alignment_self_patch_reconstruction_price":bool(pred_a),"pred_b_known_l9_positive_control":bool(known),"pred_c_weight_ranking_enriches_causal_writers":bool(enrich),"pred_d_novel_top_writer_is_material":bool(novel),"pred_e_l7_mode_split_matches_weight_prediction":bool(mode_split)}
    terminal="invalid" if not pred_a else "screen" if all(predictions.values()) else "weight_incidence_only" if known else "causal_writer_null"
    result={"schema":"temporal_iswas_upstream_full_response_mode_atlas_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"parent_authority_sha256":parent.EXPECTED,"dryrun":dryrun,"instrument":{"aligned_shapes":aligned,"physical_source_hash_ok":physical_hash_ok,"orientation_max_abs":orientation_error,"self_patch_max_abs":self_error,"attention_reconstruction_max_abs":reconstruction},"correlations":correlations,"layer_contrasts":layer_contrasts,"temporal_causal_ranking":temporal_rank,"novel_selectivity":{"L7H7":selectivity("L7H7"),"L7H8":selectivity("L7H8")},"site_metrics":metrics,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":forwards,"example_evaluations":evaluations,"fit_updates":0,"model_updates":0,"transformer_backwards":0}}
    atomic_create_json(OUT,result);print(json.dumps({key:result[key] for key in ("candidate_id","instrument","correlations","layer_contrasts","temporal_causal_ranking","novel_selectivity","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__":main()
