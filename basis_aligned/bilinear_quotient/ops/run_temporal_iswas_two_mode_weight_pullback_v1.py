#!/usr/bin/env python3
"""Pull canonical cross-task response modes into exact writer/reader weight interfaces."""

# BQGATE: EXPERIMENT pred_a_authority_factor_replay_finiteness_and_price pred_b_orthogonal_gauge_preserves_physical_modes pred_c_known_writer_and_readers_are_enriched pred_d_modes_split_weight_rankings pred_e_complete_weight_inventory
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import causal_response_weight_pullback as pullback
import run_temporal_iswas_q8_finite_causal_hankel_v1 as parent

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_iswas_two_mode_weight_pullback_v1.json"
CANONICAL=ROOT/"circuits/followups/temporal_iswas_causal_response_canonical_modes_v1_result.json"
PULLBACK=ROOT/"ops/causal_response_weight_pullback.py"
PARENT=ROOT/"ops/run_temporal_iswas_q8_finite_causal_hankel_v1.py"
OUT=ROOT/"circuits/followups/temporal_iswas_two_mode_weight_pullback_v1_result.json"
CANDIDATE_ID="cross_task.temporal_iswas_two_mode_weight_pullback_v1"
EXPECTED={"prior":"18901667ea4076803a3faeeef8d3d70798f008a56413c25f8cecab1717de2b88","canonical":"35631562bf5cc62ae4b58350fd4b7be0111ba0a9fd1c9aedbde1d093740eacf7","pullback":"6439b4af8972fb4cf3cdc061425d8064c612945e152bf12199c317767b9ffa1b","parent":"e9303c6fc1a11af4c103c49c5d47b2fdf0937714a80fd33d0549df2fa7216950"}
PRICE={"model_forwards":10,"example_evaluations":96,"fit_updates":0,"model_updates":0,"transformer_backwards":0,"upstream_scores":220,"downstream_scores":120}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def tensor_sha(t):return hashlib.sha256(t.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()
def vector_matrix_score(torch,covector,matrix):
    denominator=float(covector.norm())*float(torch.linalg.matrix_norm(matrix))
    return float((covector@matrix).norm())/denominator if denominator else 0.0
def section_score(torch,matrix,covector):
    denominator=float(torch.linalg.matrix_norm(matrix))*float(covector.norm())
    return float((matrix@covector).norm())/denominator if denominator else 0.0
def percentile(rows,key):
    order=sorted(range(len(rows)),key=lambda i:(rows[i][key],rows[i]["label"]));den=max(1,len(rows)-1)
    for rank,i in enumerate(order):rows[i][key+"_percentile"]=rank/den
    return sorted(rows,key=lambda x:(-x[key],x["label"]))
def spearman(a,b):
    import numpy as np
    def ranks(x):
        order=np.argsort(np.asarray(x),kind="stable");out=np.empty(len(x));out[order]=np.arange(len(x));return out
    return float(np.corrcoef(ranks(a),ranks(b))[0,1])

def main(*, candidate_id=CANDIDATE_ID, out=OUT, normalized_gauge=False,
         gauge_float64=False, repair_authority=None):
    if {"prior":sha(PRIOR),"canonical":sha(CANONICAL),"pullback":sha(PULLBACK),"parent":sha(PARENT)}!=EXPECTED:raise RuntimeError("two-mode pullback authority changed")
    parent_paths={"prior":parent.PRIOR,"shared_causal":parent.SHARED_CAUSAL,"temporal_capability":parent.TEMPORAL_CAPABILITY,"subspace":parent.SUBSPACE,"iswas":parent.ISWAS,"v2_capability":parent.V2_CAPABILITY,"v3_capability":parent.V3_CAPABILITY,"temporal_builder":parent.TEMPORAL_BUILDER,"v2_builder":parent.V2_BUILDER,"v3_builder":parent.V3_BUILDER,"atlas_runner":parent.ATLAS_RUNNER,"analytic_runner":parent.ANALYTIC_RUNNER,"overlap_runner":parent.OVERLAP_RUNNER}
    if {k:sha(v) for k,v in parent_paths.items()}!=parent.EXPECTED:raise RuntimeError("parent factor authorities changed")
    prior,canonical,subspace,iswas,temporal_cap=[json.loads(p.read_text()) for p in (PRIOR,CANONICAL,parent.SUBSPACE,parent.ISWAS,parent.TEMPORAL_CAPABILITY)];parent_result=json.loads(parent.OUT.read_text())
    if prior.get("candidate_id")!=CANDIDATE_ID or canonical.get("terminal")!="partial_canonicalization" or subspace.get("terminal")!="task_conditioned" or iswas.get("terminal")!="screen" or temporal_cap.get("terminal")!="manifest":raise RuntimeError("authority terminal changed")
    dryrun={"candidate_id":candidate_id,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"commands":32,"readers":32,"canonical_modes":2,"upstream_modules_per_mode":110,"downstream_modules_per_mode":60,**PRICE}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if out.exists():raise FileExistsError(f"refusing overwrite: {out}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
    family,_singular,_energy=parent.family_builder.build_family(backend,subspace);q=family[8]
    gain=math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float()) for layer in range(12,18))
    raw_modes,orientation_error,_wrong=parent.overlap.residual_modes(backend,q,gain);state_basis=torch.linalg.qr(raw_modes,mode="reduced").Q
    allowed={x for ids in temporal_cap["jointly_capable_row_ids"].values() for x in ids};all_temporal=parent.temporal.build_rows();temporal_rows=[]
    for panel in ("A1","A2"):temporal_rows.extend([r for r in all_temporal if r["transform_id"]==panel and r["row_id"] in allowed][:8])
    iswas_rows=parent.select_iswas_rows()
    if len(temporal_rows)!=16 or len(iswas_rows)!=16:raise RuntimeError("row population changed")
    writes=[];contexts=[];answers=[];foils=[];metadata=[];reconstruction=capture_identity=q8_closure=0.;forwards=evaluations=0
    for panel in ("A1","A2"):
        rows=[r for r in temporal_rows if r["transform_id"]==panel];base_batch=parent.das._batch(backend,rows,side="base");donor_batch=parent.das._batch(backend,rows,side="donor")
        base_output,base8=parent.attention_eval.capture_layer_attention(backend,base_batch,8,call=lambda:backend.native(base_batch,capture=True));_donor_output,donor8=parent.attention_eval.capture_layer_attention(backend,donor_batch,8);base11_output,base_h3=parent.attention_eval.capture_layer_attention(backend,base_batch,11)
        destinations=parent.onset.positions_for_group(base_batch,donor_batch,"subject_onset");writer_hook=parent.mediation.fixed_source_delta_hook(backend,base_batch,donor_batch,base8,donor8,destinations,("cue",),selected_heads=(1,));handle=backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:_writer_output,writer_h3=parent.attention_eval.capture_layer_attention(backend,base_batch,11)
        finally:handle.remove()
        forwards+=4;evaluations+=4*len(rows);reconstruction=max(reconstruction,*(float(x["reconstruction_max_abs"]) for x in (base8,donor8,base_h3,writer_h3)));capture_identity=max(capture_identity,parent.upstream.pair_error(base_output,base11_output))
        for index,(row,query) in enumerate(zip(rows,base_batch.semantic_positions)):
            query=int(query);groups=parent.atlas.source_partition(base_batch.token_rows[index],donor_batch.token_rows[index],query,destinations[index]);exact=(writer_h3["head_output"][index,query,3].float()-base_h3["head_output"][index,query,3].float())@q;complete=exact.new_zeros(8);suffix=exact.new_zeros(8)
            for group in parent.atlas.GROUPS:
                for factor_name in parent.atlas.FACTORS:
                    coordinate=parent.atlas.factor_head(base_h3,writer_h3,index,query,groups[group],factor_name)@q;complete+=coordinate
                    if group in ("subject_onset","post_subject","self"):suffix+=coordinate
            q8_closure=max(q8_closure,float((exact-complete).abs().max()));writes.append(raw_modes@suffix);contexts.append(torch.as_tensor(base_output.captured[(row["row_id"],"resid:18")],device=backend.device).float());answers.append(int(row["donor_answer_id"]));foils.append(int(row["donor_foil_id"]));metadata.append({"task":"temporal","row_id":row["row_id"],"family":panel,"direction":row["direction_id"]})
    base_batch=parent.das._batch(backend,iswas_rows,side="base");donor_batch=parent.das._batch(backend,iswas_rows,side="donor");base_output=backend.native(base_batch,capture=True);donor_output=backend.native(donor_batch,capture=True);forwards+=2;evaluations+=2*len(iswas_rows)
    base18=torch.stack([torch.as_tensor(base_output.captured[(r["row_id"],"resid:18")]) for r in iswas_rows]).to(backend.device).float();donor18=torch.stack([torch.as_tensor(donor_output.captured[(r["row_id"],"resid:18")]) for r in iswas_rows]).to(backend.device).float()
    import numpy as np
    axis_values=np.asarray(iswas["basis"]["values_column_major"],dtype=np.float32)
    if hashlib.sha256(axis_values.tobytes()).hexdigest()!=iswas["basis"]["sha256"]:raise RuntimeError("iswas axis changed")
    axis=torch.as_tensor(axis_values,device=backend.device).reshape(1152,1);axis=axis/axis.norm();shared_axis=state_basis@(state_basis.T@axis);iswas_writes=((donor18-base18)@axis)@shared_axis.T
    for index,row in enumerate(iswas_rows):writes.append(iswas_writes[index]);contexts.append(base18[index]);answers.append(int(row["donor_answer_id"]));foils.append(int(row["donor_foil_id"]));metadata.append({"task":"iswas","row_id":row["row_id"],"family":row["family"],"direction":row["direction_id"]})
    if metadata!=parent_result["row_metadata"]:raise RuntimeError("metadata changed")
    writes=torch.stack(writes);contexts=torch.stack(contexts);commands=writes@state_basis;readers=torch.stack([parent.analytic.analytic_reader(backend,contexts[i],answers[i],foils[i],state_basis.T)[0] for i in range(32)])
    source_scale=torch.tensor([canonical["task_diagonal_rms"][x["task"]]**-.5 for x in metadata],device=backend.device);target_scale=source_scale.clone();modes=pullback.pullback(torch,commands,readers,state_basis,source_scale,target_scale,rank=2)
    exact=torch.full((32,32),float("nan"),device=backend.device)
    for record in parent_result["records"]:exact[record["source_index"],record["target_index"]]=record["exact_effect"]
    scale=source_scale[:,None]*target_scale[None,:];balanced_exact=exact*scale;u,s,vh=torch.linalg.svd(balanced_exact,full_matrices=False);exact_rank2=(u[:,:2]*s[:2])@vh[:2];replay_rel=float((modes["rank_response"]-exact_rank2).norm()/exact_rank2.norm())
    gen=torch.Generator(device="cpu").manual_seed(20260907);gauge=torch.linalg.qr(torch.randn((8,8),generator=gen).to(backend.device)).Q;rot=pullback.pullback(torch,commands@gauge,readers@gauge,state_basis@gauge,source_scale,target_scale,rank=2)
    if normalized_gauge:
        if gauge_float64:
            commands64,readers64,state_basis64=commands.double(),readers.double(),state_basis.double()
            source_scale64,target_scale64=source_scale.double(),target_scale.double()
            modes_gauge=pullback.pullback(torch,commands64,readers64,state_basis64,source_scale64,target_scale64,rank=2)
            gen64=torch.Generator(device="cpu").manual_seed(20260907);gauge64=torch.linalg.qr(torch.randn((8,8),generator=gen64,dtype=torch.float64).to(backend.device)).Q
            rot_gauge=pullback.pullback(torch,commands64@gauge64,readers64@gauge64,state_basis64@gauge64,source_scale64,target_scale64,rank=2)
        else:
            modes_gauge,rot_gauge=modes,rot
        gauge_error=max(float((torch.linalg.qr(modes_gauge[k],mode="reduced").Q@torch.linalg.qr(modes_gauge[k],mode="reduced").Q.T-torch.linalg.qr(rot_gauge[k],mode="reduced").Q@torch.linalg.qr(rot_gauge[k],mode="reduced").Q.T).abs().max()) for k in ("physical_source_covectors","physical_reader_covectors"))
    else:
        gauge_error=max(float((modes[k]@modes[k].T-rot[k]@rot[k].T).abs().max()) for k in ("physical_source_covectors","physical_reader_covectors"))
    head=backend.model.transformer.h[11].attn;width=int(head.head_dim);value_rows=head.c_v.weight.detach().float()[3*width:4*width];upstream={};downstream={}
    for mode in range(2):
        physical_source=modes["physical_source_covectors"][:,mode];h3_dual=raw_modes.T@physical_source;h3_input=value_rows.T@(q@h3_dual);up=[]
        for layer in range(11):
            block=backend.model.transformer.h[layer]
            for h in range(int(block.attn.n_head)):
                start=h*int(block.attn.head_dim);matrix=block.attn.c_proj.weight.detach().float()[:,start:start+int(block.attn.head_dim)];up.append({"label":f"L{layer}H{h}","kind":"attention","layer":layer,"head":h,"score":vector_matrix_score(torch,h3_input,matrix)})
            up.append({"label":f"MLP{layer}","kind":"mlp","layer":layer,"score":vector_matrix_score(torch,h3_input,block.mlp.Down.weight.detach().float())})
        upstream[f"mode{mode+1}"]=percentile(up,"score")
        physical_reader=modes["physical_reader_covectors"][:,mode];heads=[];mlps=[]
        for layer in range(12,18):
            block=backend.model.transformer.h[layer]
            for h in range(int(block.attn.n_head)):
                factors={}
                for name in ("c_q","c_k","c_q2","c_k2","c_v"):
                    matrix=getattr(block.attn,name).weight.detach().float()[h*width:(h+1)*width];factors[name[2:]]=section_score(torch,matrix,physical_reader)
                heads.append({"label":f"L{layer}H{h}","kind":"attention","layer":layer,"head":h,"score":max(factors.values()),"factors":factors})
            left=section_score(torch,block.mlp.Left.weight.detach().float(),physical_reader);right=section_score(torch,block.mlp.Right.weight.detach().float(),physical_reader);mlps.append({"label":f"MLP{layer}","kind":"mlp","layer":layer,"score":max(left,right),"left":left,"right":right})
        downstream[f"mode{mode+1}"]=percentile(heads+mlps,"score")
    up_by={m:{x["label"]:x for x in rows} for m,rows in upstream.items()};down_by={m:{x["label"]:x for x in rows} for m,rows in downstream.items()};labels_up=sorted(up_by["mode1"]);labels_down=sorted(down_by["mode1"]);rank_correlations={"upstream":spearman([up_by["mode1"][x]["score"] for x in labels_up],[up_by["mode2"][x]["score"] for x in labels_up]),"downstream":spearman([down_by["mode1"][x]["score"] for x in labels_down],[down_by["mode2"][x]["score"] for x in labels_down])}
    known={"L8H1_upstream_percentiles":[up_by[m]["L8H1"]["score_percentile"] for m in ("mode1","mode2")],"L15H5_downstream_percentiles":[down_by[m]["L15H5"]["score_percentile"] for m in ("mode1","mode2")],"L15H1_downstream_percentiles":[down_by[m]["L15H1"]["score_percentile"] for m in ("mode1","mode2")]}
    finite=all(math.isfinite(x["score"]) for rows in list(upstream.values())+list(downstream.values()) for x in rows);pred_a=orientation_error<=1e-6 and reconstruction<=5e-4 and capture_identity<=1e-4 and q8_closure<=5e-4 and replay_rel<=.01 and forwards==10 and evaluations==96 and finite
    pred_b=gauge_error<=1e-5;pred_c=max(known["L8H1_upstream_percentiles"])>=.90 and max(known["L15H5_downstream_percentiles"])>=.75 and max(known["L15H1_downstream_percentiles"])>=.75;pred_d=min(rank_correlations.values())<.90;pred_e=all(len(x)==110 for x in upstream.values()) and all(len(x)==60 for x in downstream.values())
    predictions={"pred_a_authority_factor_replay_finiteness_and_price":bool(pred_a),"pred_b_orthogonal_gauge_preserves_physical_modes":bool(pred_b),"pred_c_known_writer_and_readers_are_enriched":bool(pred_c),"pred_d_modes_split_weight_rankings":bool(pred_d),"pred_e_complete_weight_inventory":bool(pred_e)};terminal="invalid" if not pred_a or not pred_b else "mode_specific_weight_screen" if all(predictions.values()) else "weight_pullback_null" if not pred_c else "partial_weight_pullback"
    result={"schema":"temporal_iswas_two_mode_weight_pullback_result_v2" if normalized_gauge else "temporal_iswas_two_mode_weight_pullback_result_v1","candidate_id":candidate_id,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"repair_authority_sha256":repair_authority,"parent_authority_sha256":parent.EXPECTED,"dryrun":{**dryrun,"candidate_id":candidate_id},"instrument":{"orientation_max_abs":orientation_error,"attention_reconstruction_max_abs":reconstruction,"capture_identity_max_abs":capture_identity,"q8_factor_closure_max_abs":q8_closure,"rank2_exact_response_relative_error":replay_rel,"orthogonal_gauge_projector_max_abs":gauge_error},"mode_artifacts":{"singular_values":[float(x) for x in modes["singular_values"][:8]],"physical_source_covectors_sha256":tensor_sha(modes["physical_source_covectors"]),"physical_reader_covectors_sha256":tensor_sha(modes["physical_reader_covectors"]),"source_coordinates":modes["source_coordinates"].cpu().tolist(),"reader_coordinates":modes["reader_coordinates"].cpu().tolist()},"known_components":known,"rank_correlations":rank_correlations,"upstream_rankings":upstream,"downstream_rankings":downstream,"predictions":predictions,"terminal":terminal,"price":PRICE}
    atomic_create_json(out,result);print(json.dumps({"candidate_id":candidate_id,"instrument":result["instrument"],"known_components":known,"rank_correlations":rank_correlations,"top_upstream":{m:[x["label"] for x in rows[:10]] for m,rows in upstream.items()},"top_downstream":{m:[x["label"] for x in rows[:10]] for m,rows in downstream.items()},"predictions":predictions,"terminal":terminal,"price":PRICE},sort_keys=True))

if __name__=="__main__":main()
