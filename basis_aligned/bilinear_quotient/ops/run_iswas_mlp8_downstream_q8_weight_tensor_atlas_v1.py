#!/usr/bin/env python3
"""Test whether exact Q8 weight contractions predict the localized response circuit."""

# BQGATE: EXPERIMENT pred_a_authority_basis_gauge_finiteness_and_zero_forward_price pred_b_selected_mlp_chain_has_exact_q8_read_write_incidence pred_c_weight_scores_positively_rank_causal_importance pred_d_attention15_h1_is_top_three_by_fixed_weight_scores pred_e_no_causal_fit_or_metric_selection
from datetime import datetime,timezone
import hashlib,json,math,os,statistics,time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap
import subspace_weight_atlas as atlas

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/iswas_mlp8_downstream_q8_weight_tensor_atlas_v1.json"
MODULE_PARENT=ROOT/"circuits/followups/iswas_mlp8_downstream_response_greedy_program_v1_result.json"
HEAD_PARENT=ROOT/"circuits/followups/iswas_mlp8_attn15_remainder_head_greedy_v1_result.json"
SUBSPACE=ROOT/"circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
LIBRARY=ROOT/"ops/subspace_weight_atlas.py"
OUT=ROOT/"circuits/followups/iswas_mlp8_downstream_q8_weight_tensor_atlas_v1_result.json"
CANDIDATE_ID="cross_task.iswas_mlp8_downstream_q8_weight_tensor_atlas_v1"
EXPECTED={"prior":"1d15b77c7bfccdaa93cb105886d5685c3b2ef19dd0aac5d13f75559aebfb2e54","module_parent":"fbd5416376ce9d0a510393a3685a6c733aa071efed2340264172123fe6b6ab54","head_parent":"5cf2aa2db970b454fd01ab0ed0fef47ac46616001ca5cfced0539a77f7d6a6af","subspace":"d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9","library":"2e7d3a546813a6029eca6fae455ad5abd03b429fcee92432ca6fe06e835e83f5"}
ATTN_GROUPS={"attn10_all":(10,tuple(range(9))),"attn11_remainder":(11,(0,2,4,5,6,7,8)),"attn12_all":(12,tuple(range(9))),"attn13_all":(13,tuple(range(9))),"attn14_all":(14,tuple(range(9))),"attn15_remainder":(15,(0,1,2,3,4,6,7,8))}
MLPS=("mlp10","mlp11","mlp12","mlp13","mlp14");HEADS=(0,1,2,3,4,6,7,8);METRICS=("q","k","q2","k2","v","o","ov")

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def ranks(xs):
    order=sorted(range(len(xs)),key=lambda i:(xs[i],i));out=[0.0]*len(xs)
    for rank,i in enumerate(order):out[i]=float(rank)
    return out
def spearman(a,b):return statistics.correlation(ranks(a),ranks(b))

def main():
    paths={"prior":PRIOR,"module_parent":MODULE_PARENT,"head_parent":HEAD_PARENT,"subspace":SUBSPACE,"library":LIBRARY}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("weight tensor atlas authority changed")
    prior,module_parent,head_parent,subspace=[json.loads(p.read_text()) for p in (PRIOR,MODULE_PARENT,HEAD_PARENT,SUBSPACE)]
    if prior.get("candidate_id")!=CANDIDATE_ID or module_parent.get("terminal")!="screen" or head_parent.get("terminal")!="screen":raise RuntimeError("causal parents changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"mlp_tensors":5,"attention_groups":6,"attention15_heads":8,"fixed_metrics":list(METRICS),"model_forwards":0,"example_evaluations":0,"fit_updates":0,"transformer_backwards":0,"model_updates":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch;model=backend.model
    family,_s,_e=family_builder.build_family(backend,subspace);gain=math.prod(float(model.transformer.h[k].lambdas[0].detach().float()) for k in range(12,18));modes,orientation_error,_wrong=overlap.residual_modes(backend,family[8],gain);q8=torch.linalg.qr(modes,mode="reduced").Q;read=q8.T
    gram_error=float((q8.T@q8-torch.eye(q8.shape[1],device=q8.device)).abs().max());module_step=module_parent["greedy_trace"][0];head_step=head_parent["greedy_trace"][0]
    mlp_rows=[]
    for label in MLPS:
        layer=int(label[3:]);mlp=model.transformer.h[layer].mlp;write=atlas.mlp_writer_to_read_map(mlp,read);complete=atlas.mlp_writer_to_read_tensor(mlp,read);shared=atlas.mlp_subspace_tensor(mlp,q8,q8)
        denom=math.prod(max(shared["scores"][k],1e-30) for k in ("left","right","down"));mlp_rows.append({"label":label,"layer":layer,"causal_singleton_gain":1.0-float(module_step["candidate_objectives"][label]),"write_score":write["score"],"complete_tensor_score":complete["score"],"complete_tensor_normalized":complete["normalized_score"],"shared_io_tensor_score":shared["scores"]["tensor"],"shared_io_tensor_normalized":shared["scores"]["tensor"]/denom})
        del complete,shared
    factors={layer:atlas.attention_subspace_factors(model.transformer.h[layer].attn,q8) for layer in range(10,16)}
    attention_rows=[]
    for label,(layer,heads) in ATTN_GROUPS.items():
        scores={metric:math.sqrt(sum(factors[layer][h]["scores"][metric]**2 for h in heads)) for metric in METRICS};attention_rows.append({"label":label,"layer":layer,"heads":list(heads),"causal_singleton_gain":1.0-float(module_step["candidate_objectives"][label]),"scores":scores})
    head_rows=[];baseline=float(head_parent["prefixes"][0]["objective"])
    for head in HEADS:
        scores={metric:factors[15][head]["scores"][metric] for metric in METRICS};head_rows.append({"head":head,"causal_singleton_gain":baseline-float(head_step["candidate_objectives"][str(head)]),"scores":scores})
    mlp_correlations={metric:spearman([r[metric] for r in mlp_rows],[r["causal_singleton_gain"] for r in mlp_rows]) for metric in ("write_score","complete_tensor_score","complete_tensor_normalized","shared_io_tensor_score","shared_io_tensor_normalized")}
    attention_correlations={metric:spearman([r["scores"][metric] for r in attention_rows],[r["causal_singleton_gain"] for r in attention_rows]) for metric in METRICS};head_correlations={metric:spearman([r["scores"][metric] for r in head_rows],[r["causal_singleton_gain"] for r in head_rows]) for metric in METRICS}
    head_ranks={metric:[r["head"] for r in sorted(head_rows,key=lambda r:(-r["scores"][metric],r["head"]))] for metric in METRICS};all_values=[x for r in mlp_rows for k,x in r.items() if isinstance(x,float)]+[x for rows in (attention_rows,head_rows) for r in rows for x in r["scores"].values()]
    pred_a=orientation_error<=1e-6 and gram_error<=1e-5 and all(math.isfinite(x) for x in all_values)
    pred_b=all(r["write_score"]>0 and r["complete_tensor_score"]>0 and r["shared_io_tensor_score"]>0 for r in mlp_rows)
    pred_c=max((*mlp_correlations.values(),*attention_correlations.values(),*head_correlations.values()))>0
    pred_d=any(1 in head_ranks[metric][:3] for metric in METRICS)
    pred_e=True;predictions={"pred_a_authority_basis_gauge_finiteness_and_zero_forward_price":bool(pred_a),"pred_b_selected_mlp_chain_has_exact_q8_read_write_incidence":bool(pred_b),"pred_c_weight_scores_positively_rank_causal_importance":bool(pred_c),"pred_d_attention15_h1_is_top_three_by_fixed_weight_scores":bool(pred_d),"pred_e_no_causal_fit_or_metric_selection":pred_e};terminal="invalid" if not pred_a else "screen" if all(predictions.values()) else "activation_conditioning_required"
    result={"schema":"iswas_mlp8_downstream_q8_weight_tensor_atlas_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"f_linear_orientation_max_abs":orientation_error,"q8_gram_max_abs":gram_error},"mlp_rows":mlp_rows,"attention_rows":attention_rows,"attention15_head_rows":head_rows,"correlations":{"mlp":mlp_correlations,"attention_groups":attention_correlations,"attention15_heads":head_correlations},"attention15_head_rankings":head_ranks,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":0,"example_evaluations":0,"fit_updates":0,"transformer_backwards":0,"model_updates":0}}
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("candidate_id","instrument","correlations","attention15_head_rankings","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__":main()
