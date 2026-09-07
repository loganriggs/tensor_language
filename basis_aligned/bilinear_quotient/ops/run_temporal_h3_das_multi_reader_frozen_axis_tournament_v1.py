#!/usr/bin/env python3
"""Evaluate frozen DAS axes across independent downstream readers and banks."""

# BQGATE: EXPERIMENT pred_a_authority_capability_closure_finiteness_and_price pred_b_pooled_aligned_beats_dim_on_mean_and_worst_multi_reader_objective pred_c_at_least_one_noise_or_kl_axis_improves_dim_full_vocabulary pred_d_pooled_aligned_beats_dim_at_each_reader_family pred_e_zero_new_fit_and_complete_frozen_method_inventory
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fit_v1
import circuit_candidate_temporal_auxiliary_fresh_cues_v8 as test_v8
import circuit_candidate_temporal_auxiliary_fresh_cues_v9 as test_v9
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_unit_greedy as g
import run_temporal_auxiliary_will_had_block11h3_regularized_cdas_v1 as regularized
import run_temporal_auxiliary_will_had_h3_tensor_anchored_regularized_rank7_v2 as evaluator

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_h3_das_multi_reader_frozen_axis_tournament_v1.json"
AXIS=ROOT/"circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_cdas_axis_necessity_v2_result.json"
REGULARIZED=ROOT/"circuits/followups/temporal_auxiliary_will_had_block11h3_regularized_cdas_v1_result.json"
ALIGNED=ROOT/"circuits/followups/temporal_auxiliary_will_had_block11h3_aligned_objective_cdas_v1_result.json"
POOLED=ROOT/"circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
CAP8=ROOT/"circuits/followups/temporal_auxiliary_will_had_fresh_v8_capability_v1_result.json"
CAP9=ROOT/"circuits/followups/temporal_auxiliary_will_had_fresh_v9_capability_v1_result.json"
V1=ROOT/"ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py";V8=ROOT/"ops/circuit_candidate_temporal_auxiliary_fresh_cues_v8.py";V9=ROOT/"ops/circuit_candidate_temporal_auxiliary_fresh_cues_v9.py"
EVALUATOR=ROOT/"ops/run_temporal_auxiliary_will_had_h3_tensor_anchored_regularized_rank7_v2.py";UNIT_LIB=ROOT/"ops/circuit_unit_greedy.py"
OUT=ROOT/"circuits/followups/temporal_h3_das_multi_reader_frozen_axis_tournament_v1_result.json"
CANDIDATE_ID="temporal_auxiliary.will_vs_had.h3_das_multi_reader_frozen_axis_tournament_v1"
EXPECTED={"prior":"8f1fc3c8ad0b1e8c82b5b76b44e5050d98f89c22c9b9931511320fc404d36c1c","axis":"4f345907c41222ebeec33c3a860052a0a8f39166655da0354f69e79a1c577fc5","regularized":"f7d53dd6530dbdbebba7610236adc862b3c595bd83fb6c1b24d8fd4365543163","aligned":"3aea84323bae1c2e46a430ef5f08b838826504693e6b1ba8a05027ca065b379d","pooled":"d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9","cap8":"fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff","cap9":"828d8b15d9bcf048de32d74384e2f4bc62972f20289515f8eb6c576302262392","v1":"5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9","v8":"13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf","v9":"9b771713c5803082c95a3566bc41120587e60f99c4c8bacc291602516bbe01a5","evaluator":"561b40b093e0a46469fa95b01c23e1b0d7d294201aaccb63b918b86900359303","unit_lib":"8a5389dfee9f239509ef2440409854b1115e46a729193cf0ceba548443a1d254"}
METHODS=("dim","unregularized","noise","kl","kl_noise","aligned_0.3","pooled_aligned");MAX_FORWARDS,MAX_EVALUATIONS,MAX_RECORDS=366,11316,4340

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")

def rows_for(builder,capability):
    allowed={x for ids in capability["jointly_capable_row_ids"].values() for x in ids};return [r for r in builder.build_rows() if r["transform_id"] in ("A1","A2") and r["row_id"] in allowed]

def cell_score(item,panel):
    behavior=item["behavior_fraction_of_full_h3"][panel];transport=item["downstream_fraction_of_full_h3"][panel];kl=item["full_vocabulary_kl"][panel]
    pieces={"behavior_match_squared":(1.0-behavior["h3_regularized_rank7"])**2,"behavior_complement_squared":behavior["h3_regularized_rank7_orthogonal"]**2,"l15_transport_match_squared":(1.0-transport)**2,"full_vocab_match_fraction":kl["student_match_fraction"],"full_vocab_complement_fraction":kl["complement_inert_fraction"]}
    return sum(pieces.values()),pieces

def main():
    paths={"prior":PRIOR,"axis":AXIS,"regularized":REGULARIZED,"aligned":ALIGNED,"pooled":POOLED,"cap8":CAP8,"cap9":CAP9,"v1":V1,"v8":V8,"v9":V9,"evaluator":EVALUATOR,"unit_lib":UNIT_LIB}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("multi-reader tournament authority changed")
    prior,axis,reg,aligned,pooled,cap8,cap9=[json.loads(p.read_text()) for p in (PRIOR,AXIS,REGULARIZED,ALIGNED,POOLED,CAP8,CAP9)]
    rows8,rows9=rows_for(test_v8,cap8),rows_for(test_v9,cap9)
    if prior.get("candidate_id")!=CANDIDATE_ID or reg.get("terminal")!="null" or aligned.get("terminal")!="null" or pooled.get("terminal")!="task_conditioned" or cap8.get("terminal")!="manifest" or cap9.get("terminal")!="manifest" or len(rows8)!=60 or len(rows9)!=64:raise RuntimeError("parents, capabilities, or populations changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"methods":list(METHODS),"banks":{"v8":len(rows8),"v9":len(rows9)},"reader_families":["behavior","attention15_transport","full_vocabulary"],"model_forwards_max":MAX_FORWARDS,"example_evaluations_max":MAX_EVALUATIONS,"evaluation_records_max":MAX_RECORDS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
    fit_rows=[r for r in fit_v1.build_rows() if r["transform_id"]=="A1"][0::2];prep=g.prepare(backend,fit_rows);q_dim=g.diff_in_means_direction(backend,prep,regularized.UNIT)
    aligned_fit=next(item for item in aligned["fits"] if float(item["weight"])==.3);coordinates={"unregularized":axis["axis_artifact"]["coordinates"],"noise":reg["fits"]["noise"]["coordinates"],"kl":reg["fits"]["kl"]["coordinates"],"kl_noise":reg["fits"]["kl_noise"]["coordinates"],"aligned_0.3":aligned_fit["coordinates"],"pooled_aligned":pooled["axis_artifacts"]["pooled_aligned_rank1"]};axes={"dim":torch.linalg.qr(q_dim,mode="reduced").Q,**{name:torch.linalg.qr(torch.tensor(values,device=backend.device).float().unsqueeze(1),mode="reduced").Q for name,values in coordinates.items()}}
    evaluations={};max_instrument=0.0;forwards=2;example_evaluations=2*len(fit_rows);records=0
    for method,q in axes.items():
        evaluations[method]={}
        for bank,rows in (("v8",rows8),("v9",rows9)):
            item=evaluator.evaluate_bank(backend,bank,rows,q,q@q.T);records+=len(item.pop("records"));forwards+=item.pop("forwards");example_evaluations+=item.pop("evaluations");max_instrument=max(max_instrument,*[float(v) for v in item["instrument"].values()]);evaluations[method][bank]=item
    objectives={}
    for method in METHODS:
        cells=[];by_cell={}
        for bank in ("v8","v9"):
            for panel in ("A1","A2"):
                score,pieces=cell_score(evaluations[method][bank],panel);cells.append(score);by_cell[f"{bank}_{panel}"]={"joint":score,**pieces}
        objectives[method]={"mean":sum(cells)/len(cells),"worst":max(cells),"by_cell":by_cell}
    vocab=lambda method:sum(v for cell in objectives[method]["by_cell"].values() for k,v in cell.items() if k in ("full_vocab_match_fraction","full_vocab_complement_fraction"))/8
    finite=all(math.isfinite(x) for method in objectives.values() for cell in method["by_cell"].values() for x in cell.values());pred_a=max_instrument<=5e-4 and finite and forwards==MAX_FORWARDS and example_evaluations==MAX_EVALUATIONS and records==MAX_RECORDS
    pred_b=objectives["pooled_aligned"]["mean"]<objectives["dim"]["mean"] and objectives["pooled_aligned"]["worst"]<objectives["dim"]["worst"]
    pred_c=min(vocab(m) for m in ("noise","kl","kl_noise"))<vocab("dim")
    families=("behavior_match_squared","behavior_complement_squared","l15_transport_match_squared","full_vocab_match_fraction","full_vocab_complement_fraction");pred_d=all(sum(objectives["pooled_aligned"]["by_cell"][c][f] for c in objectives["dim"]["by_cell"])<sum(objectives["dim"]["by_cell"][c][f] for c in objectives["dim"]["by_cell"]) for f in families)
    pred_e=tuple(evaluations)==METHODS;predictions={"pred_a_authority_capability_closure_finiteness_and_price":bool(pred_a),"pred_b_pooled_aligned_beats_dim_on_mean_and_worst_multi_reader_objective":bool(pred_b),"pred_c_at_least_one_noise_or_kl_axis_improves_dim_full_vocabulary":bool(pred_c),"pred_d_pooled_aligned_beats_dim_at_each_reader_family":bool(pred_d),"pred_e_zero_new_fit_and_complete_frozen_method_inventory":bool(pred_e)};terminal="invalid" if not pred_a else "optimizer_licensed" if all(predictions.values()) else "reader_objective_rejected" if not pred_b else "partial_multi_reader_gain"
    result={"schema":"temporal_h3_das_multi_reader_frozen_axis_tournament_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"maximum_closure_or_reconstruction_abs":max_instrument},"objectives":objectives,"full_vocabulary_mean":{m:vocab(m) for m in METHODS},"evaluations":evaluations,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":forwards,"example_evaluations":example_evaluations,"evaluation_records":records,"fit_updates":0,"transformer_backwards":0,"model_updates":0}}
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("candidate_id","instrument","objectives","full_vocabulary_mean","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__":main()
