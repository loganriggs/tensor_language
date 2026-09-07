#!/usr/bin/env python3
"""Complete downstream response-removal atlas for canonical cross-task modes."""

# BQGATE: EXPERIMENT pred_a_authority_inventory_replay_self_clamp_finiteness_price pred_b_known_l15_readers_are_material pred_c_weight_scores_enrich_causal_readers pred_d_novel_l17h4_is_material pred_e_complete_zero_fit_atlas
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import attention_source_destination_eval as attention_eval
import run_temporal_iswas_v11_writer_frontier_holdout_v1 as frontier
import run_temporal_iswas_upstream_full_response_mode_atlas_v1 as response

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_iswas_canonical_downstream_response_removal_atlas_v1.json"
FRONTIER_RESULT=ROOT/"circuits/followups/temporal_iswas_v11_writer_frontier_holdout_v1_result.json"
FRONTIER_RUNNER=ROOT/"ops/run_temporal_iswas_v11_writer_frontier_holdout_v1.py"
WEIGHT_RESULT=ROOT/"circuits/followups/temporal_iswas_two_mode_weight_pullback_v3_result.json"
WEIGHT_RUNNER=ROOT/"ops/run_temporal_iswas_two_mode_weight_pullback_v1.py"
OUT=ROOT/"circuits/followups/temporal_iswas_canonical_downstream_response_removal_atlas_v1_result.json"
CANDIDATE_ID="cross_task.temporal_iswas_canonical_downstream_response_removal_atlas_v1"
LAYERS=tuple(range(12,18))
SITES=tuple([f"L{layer}H{head}" for layer in LAYERS for head in range(9)]+[f"MLP{layer}" for layer in LAYERS])
WRITER_SITES=tuple(site for bit,name in enumerate(frontier.discovery_runner.GROUPS) if 30&(1<<bit) for site in frontier.discovery_runner.GROUPS[name])
MAX_FORWARDS,MAX_EVALUATIONS=71,1988
EXPECTED={"prior":"e07228dd1c4ec02766cfb931d5ec6076b833b3944e8172cb5b586396b4afcaa8","frontier_result":"9262c379be1a485b826a0f414c822e75832a1162a69881a4b40d2322e26ab07b","frontier_runner":"08c9b1e85b9599c4cac2195b22ab2f04514c5bf4c861fff3192e3d1bf8e1431d","weight_result":"c8ab608fa116342f9cbc8af4955e6087faa0f1eee9dd74dacb5c0ec168c5bf4d","weight_runner":"abfedf5b2347bc49a13d7008c4e815a33d6c1629f300d9e720f0876c4e983788"}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def tensor_sha(tensor):return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()
def site_module(backend,site):
    kind,layer,_head=response.site_parts(site);block=backend.model.transformer.h[layer]
    return block.attn.c_proj if kind=="attn" else block.mlp
def capture_downstream(backend,call):
    cache,handles={},[]
    for site in SITES:
        kind,_layer,_head=response.site_parts(site)
        if kind=="attn":
            def save(_module,arguments,site=site):cache[site]=arguments[0].detach().clone()
            handles.append(site_module(backend,site).register_forward_pre_hook(save))
        else:
            def save(_module,_arguments,output,site=site):cache[site]=output.detach().clone()
            handles.append(site_module(backend,site).register_forward_hook(save))
    try:output=call()
    finally:
        for handle in handles:handle.remove()
    if set(cache)!=set(SITES):raise RuntimeError("incomplete downstream capture")
    return output,cache
def run_clamped(backend,batch,base_values,sites,call):
    handles=[];head_width=int(backend.model.config.n_embd//backend.model.config.n_head)
    for site in sites:
        kind,_layer,_head=response.site_parts(site);hook=response.patch_hook(batch,base_values[site],site,head_width)
        handles.append(site_module(backend,site).register_forward_pre_hook(hook) if kind=="attn" else site_module(backend,site).register_forward_hook(hook))
    try:return call()
    finally:
        for handle in handles:handle.remove()
def main():
    paths={"prior":PRIOR,"frontier_result":FRONTIER_RESULT,"frontier_runner":FRONTIER_RUNNER,"weight_result":WEIGHT_RESULT,"weight_runner":WEIGHT_RUNNER}
    if {key:sha(value) for key,value in paths.items()}!=EXPECTED:raise RuntimeError("downstream response atlas authority changed")
    prior,fresult,weights,tcap,icap=[json.loads(path.read_text()) for path in (PRIOR,FRONTIER_RESULT,WEIGHT_RESULT,frontier.TEMPORAL_CAPABILITY,frontier.ISWAS_CAPABILITY)]
    if prior.get("candidate_id")!=CANDIDATE_ID or fresult.get("terminal")!="conditional_writer_screen" or weights.get("terminal")!="mode_specific_weight_screen":raise RuntimeError("authority terminal changed")
    rows=sum((frontier.sealed_rows(frontier.temporal,tcap,panel) for panel in ("A1","A2")),[])+sum((frontier.sealed_rows(frontier.iswas,icap,panel) for panel in ("A1","A2")),[])
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"rows":len(rows),"writer_sites":list(WRITER_SITES),"downstream_sites":len(SITES),"model_forwards_max":MAX_FORWARDS,"example_evaluations_max":MAX_EVALUATIONS,"fit_updates":0,"model_updates":0,"transformer_backwards":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if len(rows)!=28 or len(SITES)!=60 or len(WRITER_SITES)!=5:raise RuntimeError("frozen inventory changed")
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
    base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor")
    base_output,base_up=response.capture(backend,base_batch);donor_output,donor_up=response.capture(backend,donor_batch)
    base_down_output,base_down=capture_downstream(backend,lambda:backend.native(base_batch,capture=True))
    live_pair,live_down=capture_downstream(backend,lambda:response.run_patch(backend,base_batch,donor_up,WRITER_SITES));live_output=live_pair[0]
    self_output=run_clamped(backend,base_batch,base_down,SITES,lambda:backend.native(base_batch,capture=True))
    base_state,donor_state,live_state,self_state=(response.states(torch,backend,x,rows) for x in (base_output,donor_output,live_output,self_output))
    identity_error=max(response.states(torch,backend,base_down_output,rows).sub(base_state).abs().max().item(),(self_state-base_state).abs().max().item())
    reconstruction=0.0
    for layer in LAYERS:
        _replay,captured=attention_eval.capture_layer_attention(backend,base_batch,layer)
        reconstruction=max(reconstruction,float((captured["head_output"].reshape_as(base_down[f"L{layer}H0"])-base_down[f"L{layer}H0"]).abs().max()))
    family,_singular,_energy=frontier.parent.family_builder.build_family(backend,json.loads(frontier.parent.SUBSPACE.read_text()));q=family[8]
    gain=math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float()) for layer in LAYERS)
    raw_modes,orientation_error,_wrong=frontier.parent.overlap.residual_modes(backend,q,gain);state_basis=torch.linalg.qr(raw_modes,mode="reduced").Q
    reader_coordinates=torch.as_tensor(weights["mode_artifacts"]["reader_coordinates"],device=backend.device).float();physical_reader=state_basis@reader_coordinates
    reader_hash_ok=tensor_sha(physical_reader)==weights["mode_artifacts"]["physical_reader_covectors_sha256"]
    answer=torch.as_tensor([row["donor_answer_id"] for row in rows],device=backend.device);foil=torch.as_tensor([row["donor_foil_id"] for row in rows],device=backend.device);index=torch.arange(len(rows),device=backend.device)
    def margin(state):
        logits=das.head_logits(backend,state);return logits[index,answer]-logits[index,foil]
    base_margin,donor_margin,live_margin=margin(base_state),margin(donor_state),margin(live_state);full_behavior=live_margin-base_margin;full_modes=(live_state-base_state)@physical_reader
    recovery=(live_margin-base_margin)/(donor_margin-base_margin);frontier_replay=abs(float(recovery.mean())-fresult["arm_metrics"]["30"]["behavior"]["mean_recovery"])
    task_indices={"temporal":torch.arange(0,14,device=backend.device),"iswas":torch.arange(14,28,device=backend.device),"pooled":torch.arange(28,device=backend.device)}
    metrics={};forwards,evaluations=11,308
    for site in SITES:
        removed_pair=run_clamped(backend,base_batch,base_down,(site,),lambda:response.run_patch(backend,base_batch,donor_up,WRITER_SITES));removed_state=response.states(torch,backend,removed_pair[0],rows);removed_margin=margin(removed_state)
        necessity_modes=(live_state-removed_state)@physical_reader;necessity_behavior=live_margin-removed_margin;tasks={}
        for task,ids in task_indices.items():
            tasks[task]={"behavior":response.vector_stats(torch,necessity_behavior[ids],full_behavior[ids])}
            for mode in range(2):tasks[task][f"mode{mode+1}"]=response.vector_stats(torch,necessity_modes[ids,mode],full_modes[ids,mode])
        metrics[site]={"tasks":tasks};forwards+=1;evaluations+=len(rows)
    static={mode:{row["label"]:row["score"] for row in weights["downstream_rankings"][mode]} for mode in ("mode1","mode2")}
    correlations={};layer_contrasts={}
    for mode in ("mode1","mode2"):
        causal=[abs(metrics[site]["tasks"]["pooled"][mode]["signed_projection"]) for site in SITES];correlations[mode]=response.spearman([static[mode][site] for site in SITES],causal);layer_contrasts[mode]={}
        for layer in LAYERS:
            heads=[f"L{layer}H{head}" for head in range(9)];high=max(heads,key=lambda site:static[mode][site]);low=min(heads,key=lambda site:static[mode][site]);layer_contrasts[mode][str(layer)]={"high":high,"low":low,"high_causal":abs(metrics[high]["tasks"]["pooled"][mode]["signed_projection"]),"low_causal":abs(metrics[low]["tasks"]["pooled"][mode]["signed_projection"])}
    known=any(all(metrics[site]["tasks"][task][mode]["signed_projection"]>=.05 for task in ("temporal","iswas")) for site in ("L15H5","L15H1") for mode in ("mode1","mode2"))
    enrich=any(correlations[mode]>=.20 and sum(x["high_causal"]>x["low_causal"] for x in layer_contrasts[mode].values())>=4 for mode in ("mode1","mode2"))
    novel=any(metrics["L17H4"]["tasks"][task][mode]["signed_projection"]>=.03 and metrics["L17H4"]["tasks"][task][mode]["cosine"]>=.40 for task in ("temporal","iswas") for mode in ("mode1","mode2"))
    finite=all(math.isfinite(value) for site in metrics.values() for task in site["tasks"].values() for family in task.values() for value in family.values())
    pred_a=orientation_error<=1e-6 and reader_hash_ok and identity_error<=1e-4 and reconstruction<=5e-4 and frontier_replay<=1e-6 and finite and forwards==MAX_FORWARDS and evaluations==MAX_EVALUATIONS
    pred_e=len(metrics)==60 and set(metrics)==set(SITES)
    predictions={"pred_a_authority_inventory_replay_self_clamp_finiteness_price":bool(pred_a),"pred_b_known_l15_readers_are_material":bool(known),"pred_c_weight_scores_enrich_causal_readers":bool(enrich),"pred_d_novel_l17h4_is_material":bool(novel),"pred_e_complete_zero_fit_atlas":bool(pred_e)}
    terminal="invalid" if not pred_a or not known else "causal_downstream_weight_screen" if all(predictions.values()) else "weight_incidence_only" if not enrich else "novel_reader_null" if not novel else "partial_downstream_screen"
    result={"schema":"temporal_iswas_canonical_downstream_response_removal_atlas_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"orientation_max_abs":orientation_error,"physical_reader_hash_ok":reader_hash_ok,"native_identity_and_self_clamp_max_abs":identity_error,"attention_reconstruction_max_abs":reconstruction,"frontier_behavior_replay_abs":frontier_replay},"correlations":correlations,"layer_contrasts":layer_contrasts,"site_metrics":metrics,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":forwards,"example_evaluations":evaluations,"fit_updates":0,"model_updates":0,"transformer_backwards":0},"serial_seconds":time.perf_counter()-started}
    atomic_create_json(OUT,result);print(json.dumps({"candidate_id":CANDIDATE_ID,"instrument":result["instrument"],"correlations":correlations,"layer_contrasts":layer_contrasts,"known":{site:metrics[site] for site in ("L15H5","L15H1")},"novel":{"L17H4":metrics["L17H4"]},"predictions":predictions,"terminal":terminal,"price":result["price"]},sort_keys=True))
if __name__=="__main__":main()
