#!/usr/bin/env python3
"""Activation-condition exact MLP weight tensors on the localized circuit."""

# BQGATE: EXPERIMENT pred_a_authority_basis_exact_tensor_replay_finiteness_and_price pred_b_all_selected_mlps_write_q8_under_task_activation pred_c_activation_conditioned_magnitude_matches_or_beats_static_ranking pred_d_q8_read_and_context_complement_are_both_quantified pred_e_zero_fit_exact_weight_contraction
from datetime import datetime,timezone
import hashlib,json,math,os,statistics,time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap
import subspace_weight_atlas as atlas

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/iswas_mlp8_activation_conditioned_q8_mlp_tensor_v1.json"
WEIGHT_ATLAS=ROOT/"circuits/followups/iswas_mlp8_downstream_q8_weight_tensor_atlas_v1_result.json"
MODULE_PARENT=ROOT/"circuits/followups/iswas_mlp8_downstream_response_greedy_program_v1_result.json"
CAPABILITY=ROOT/"circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
BUILDER=ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11.py"
LIBRARY=ROOT/"ops/subspace_weight_atlas.py"
OUT=ROOT/"circuits/followups/iswas_mlp8_activation_conditioned_q8_mlp_tensor_v1_result.json"
CANDIDATE_ID="cross_task.iswas_mlp8_activation_conditioned_q8_mlp_tensor_v1"
EXPECTED={"prior":"acba535d6a428cc8e4a77c63966216cf191fcbb2e93121a5e9a0421ef0c29369","weight_atlas":"03e9f70bf93d96304ffef2c2f08f30633c2f044612026681f3a99ac13aa5784f","module_parent":"fbd5416376ce9d0a510393a3685a6c733aa071efed2340264172123fe6b6ab54","capability":"6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2","builder":"fbd47713fafcb87fc30ba339d175f7d06770ce36b93b6035e8848455529344ec","library":"2e7d3a546813a6029eca6fae455ad5abd03b429fcee92432ca6fe06e835e83f5"}
LAYERS=tuple(range(10,15));MAX_FORWARDS,MAX_EVALUATIONS=4,120

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def ranks(xs):
    order=sorted(range(len(xs)),key=lambda i:(xs[i],i));out=[0.0]*len(xs)
    for rank,i in enumerate(order):out[i]=float(rank)
    return out
def spearman(a,b):return statistics.correlation(ranks(a),ranks(b))

def capture(backend,call):
    inputs={};outputs={};handles=[]
    for layer in LAYERS:
        def pre(_module,args,layer=layer):inputs[layer]=args[0].detach().clone()
        def post(_module,_args,out,layer=layer):outputs[layer]=out.detach().clone()
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_pre_hook(pre));handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(post))
    try:result=call()
    finally:
        for h in handles:h.remove()
    return result,inputs,outputs

def flatten_prefix(cache,rows):
    return {layer:cache[layer].new_tensor([]) if not rows else __import__("torch").cat([cache[layer][i,:int(row["base_semantic_position"])+1] for i,row in enumerate(rows)],dim=0).float() for layer in LAYERS}

def main():
    paths={"prior":PRIOR,"weight_atlas":WEIGHT_ATLAS,"module_parent":MODULE_PARENT,"capability":CAPABILITY,"builder":BUILDER,"library":LIBRARY}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("activation-conditioned tensor authority changed")
    prior,weight_atlas,parent,capability,subspace=[json.loads(p.read_text()) for p in (PRIOR,WEIGHT_ATLAS,MODULE_PARENT,CAPABILITY,weight.SUBSPACE)]
    allowed={x for ids in capability["jointly_capable_row_ids"].values() for x in ids};rows=[r for r in candidate.build_rows() if r["family"] in ("A1","A2") and r["row_id"] in allowed]
    if prior.get("candidate_id")!=CANDIDATE_ID or weight_atlas.get("terminal")!="screen" or parent.get("terminal")!="screen" or len(rows)!=30:raise RuntimeError("parents or rows changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"rows":30,"layers":list(LAYERS),"model_forwards_max":MAX_FORWARDS,"example_evaluations_max":MAX_EVALUATIONS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch;model=backend.model
    family,_s,_e=family_builder.build_family(backend,subspace);gain=math.prod(float(model.transformer.h[k].lambdas[0].detach().float()) for k in range(12,18));modes,orientation_error,_wrong=overlap.residual_modes(backend,family[8],gain);q8=torch.linalg.qr(modes,mode="reduced").Q;read=q8.T
    base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor");_bo,bh=weight.capture_mlp8(backend,base_batch);_do,dh=weight.capture_mlp8(backend,donor_batch)
    down=model.transformer.h[8].mlp.Down.weight.detach().float();_u,_s,vh=torch.linalg.svd(q8.T@down,full_matrices=False);delta=dh["hidden"].float()-bh["hidden"].float();complement=delta-weight.project(delta,vh,vh.shape[0]);positions=[weight.postcue_positions(r) for r in rows]
    _base,base_inputs,base_outputs=capture(backend,lambda:backend.native(base_batch,capture=False));_live,live_inputs,live_outputs=capture(backend,lambda:weight.run_hidden_patch(backend,base_batch,bh["hidden"],complement,positions))
    b_in,l_in,b_out,l_out=[flatten_prefix(c,rows) for c in (base_inputs,live_inputs,base_outputs,live_outputs)];causal={r["label"]:r["causal_singleton_gain"] for r in weight_atlas["mlp_rows"]};reports={};max_exact=0.0
    for layer in LAYERS:
        mlp=model.transformer.h[layer].mlp;full=atlas.activation_conditioned_mlp_write(mlp,read,b_in[layer],l_in[layer])["response"];actual=(l_out[layer]-b_out[layer])@q8;exact=float((full-actual).norm()/actual.norm().clamp_min(1e-30));max_exact=max(max_exact,exact);din=l_in[layer]-b_in[layer];q_delta=(din@q8)@q8.T
        q_response=atlas.activation_conditioned_mlp_write(mlp,read,b_in[layer],b_in[layer]+q_delta)["response"];c_response=atlas.activation_conditioned_mlp_write(mlp,read,b_in[layer],b_in[layer]+din-q_delta)["response"];interaction=full-q_response-c_response
        norm=float(full.norm());reports[f"mlp{layer}"]={"prefix_vectors":int(full.shape[0]),"full_q8_response_rms":float(torch.sqrt(full.square().mean())),"q8_input_response_norm_fraction":float(q_response.norm()/norm),"complement_input_response_norm_fraction":float(c_response.norm()/norm),"bilinear_interaction_norm_fraction":float(interaction.norm()/norm),"q8_input_response_cosine":float((q_response*full).sum()/(q_response.norm()*full.norm()).clamp_min(1e-30)),"exact_cached_response_relative_norm":exact,"causal_singleton_gain":causal[f"mlp{layer}"]}
    response_corr=spearman([reports[f"mlp{k}"]["full_q8_response_rms"] for k in LAYERS],[causal[f"mlp{k}"] for k in LAYERS]);qread_corr=spearman([reports[f"mlp{k}"]["q8_input_response_norm_fraction"] for k in LAYERS],[causal[f"mlp{k}"] for k in LAYERS]);finite=all(math.isfinite(v) for r in reports.values() for v in r.values() if isinstance(v,float))
    pred_a=orientation_error<=1e-6 and max_exact<=2e-4 and finite
    pred_b=all(reports[f"mlp{k}"]["full_q8_response_rms"]>0 for k in LAYERS)
    pred_c=max(response_corr,qread_corr)>=.70
    pred_d=all(all(key in reports[f"mlp{k}"] for key in ("q8_input_response_norm_fraction","complement_input_response_norm_fraction","bilinear_interaction_norm_fraction")) for k in LAYERS)
    pred_e=True;predictions={"pred_a_authority_basis_exact_tensor_replay_finiteness_and_price":bool(pred_a),"pred_b_all_selected_mlps_write_q8_under_task_activation":bool(pred_b),"pred_c_activation_conditioned_magnitude_matches_or_beats_static_ranking":bool(pred_c),"pred_d_q8_read_and_context_complement_are_both_quantified":bool(pred_d),"pred_e_zero_fit_exact_weight_contraction":pred_e};terminal="invalid" if not pred_a else "screen" if all(predictions.values()) else "downstream_sensitivity_required"
    result={"schema":"iswas_mlp8_activation_conditioned_q8_mlp_tensor_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"f_linear_orientation_max_abs":orientation_error,"maximum_exact_cached_response_relative_norm":max_exact,"prefix_vectors":sum(int(r["prefix_vectors"]) for r in reports.values())},"reports":reports,"rank_correlations":{"full_q8_response_rms":response_corr,"q8_input_response_norm_fraction":qread_corr,"static_shared_io_tensor_reference":.70},"predictions":predictions,"terminal":terminal,"price":{"model_forwards":MAX_FORWARDS,"example_evaluations":MAX_EVALUATIONS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}}
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("candidate_id","instrument","reports","rank_correlations","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__":main()
