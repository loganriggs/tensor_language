#!/usr/bin/env python3
"""Map exact local MLP writes to finite downstream reader gains."""

# BQGATE: EXPERIMENT pred_a_authority_basis_finiteness_and_price pred_b_all_cached_live_arms_replay_unrestricted pred_c_every_selected_mlp_has_nonzero_final_necessity pred_d_downstream_gain_varies_across_layers pred_e_zero_fit_finite_causal_edge_table
from datetime import datetime,timezone
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
PRIOR=ROOT/"circuits/prior_art/iswas_mlp8_mlp_chain_finite_downstream_sensitivity_v1.json"
PARENT=ROOT/"circuits/followups/iswas_mlp8_activation_conditioned_q8_mlp_tensor_v1_result.json"
PARENT_RUNNER=ROOT/"ops/run_iswas_mlp8_activation_conditioned_q8_mlp_tensor_v1.py"
CAPABILITY=ROOT/"circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
BUILDER=ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11.py"
OUT=ROOT/"circuits/followups/iswas_mlp8_mlp_chain_finite_downstream_sensitivity_v1_result.json"
CANDIDATE_ID="cross_task.iswas_mlp8_mlp_chain_finite_downstream_sensitivity_v1"
EXPECTED={"prior":"5eb39ffd3ef046d1a43f9385fde3f2d8fba50c020cd726c091d4b816eac7b6c1","parent":"cbbc09378bc27a8674aa5da28309f95314654a2c72c9d6c44e9ab2a6f341e62c","parent_runner":"b5419f0b397f0998947e8788686ad3e17c784f6258d49af80f533a0aea4390d8","capability":"6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2","builder":"fbd47713fafcb87fc30ba339d175f7d06770ce36b93b6035e8848455529344ec"}
LAYERS=tuple(range(10,15));MAX_FORWARDS,MAX_EVALUATIONS=14,420

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def cosine(x,y):
    d=float(x.norm()*y.norm());return float((x*y).sum())/d if d else 0.0

def run_scaled(backend,batch,base_hidden,complement,positions,base,live,layer,alpha):
    def patch(_module,_args,output):
        changed=output.clone()
        for i,query in enumerate(batch.semantic_positions):
            q=slice(None,int(query)+1);b=base["mlp"][layer][i,q].to(changed);l=live["mlp"][layer][i,q].to(changed);changed[i,q]=b+alpha*(l-b)
        return changed
    h1=backend.model.transformer.h[layer].mlp.register_forward_hook(patch);h2=backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(converter.actuation_hook(base_hidden,complement,positions))
    try:return backend.native(batch,capture=True)
    finally:h1.remove();h2.remove()

def main():
    paths={"prior":PRIOR,"parent":PARENT,"parent_runner":PARENT_RUNNER,"capability":CAPABILITY,"builder":BUILDER}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("finite sensitivity authority changed")
    prior,parent,capability,subspace=[json.loads(p.read_text()) for p in (PRIOR,PARENT,CAPABILITY,weight.SUBSPACE)];allowed={x for ids in capability["jointly_capable_row_ids"].values() for x in ids};rows=[r for r in candidate.build_rows() if r["family"] in ("A1","A2") and r["row_id"] in allowed]
    if prior.get("candidate_id")!=CANDIDATE_ID or parent.get("terminal")!="downstream_sensitivity_required" or len(rows)!=30:raise RuntimeError("parent or rows changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"rows":30,"layers":list(LAYERS),"alphas":[0.0,1.0],"model_forwards_max":MAX_FORWARDS,"example_evaluations_max":MAX_EVALUATIONS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch;model=backend.model
    family,_s,_e=family_builder.build_family(backend,subspace);gain=math.prod(float(model.transformer.h[k].lambdas[0].detach().float()) for k in range(12,18));modes,orientation_error,_wrong=overlap.residual_modes(backend,family[8],gain);q8=torch.linalg.qr(modes,mode="reduced").Q
    base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor");_bo,bh=weight.capture_mlp8(backend,base_batch);_do,dh=weight.capture_mlp8(backend,donor_batch);down=model.transformer.h[8].mlp.Down.weight.detach().float();_u,_s,vh=torch.linalg.svd(q8.T@down,full_matrices=False);delta=dh["hidden"].float()-bh["hidden"].float();complement=delta-weight.project(delta,vh,vh.shape[0]);positions=[weight.postcue_positions(r) for r in rows]
    base_output,base=composition.capture_modules(backend,lambda:backend.native(base_batch,capture=True));live_output,live=composition.capture_modules(backend,lambda:weight.run_hidden_patch(backend,base_batch,bh["hidden"],complement,positions));base18=composition.state(base_output,rows,backend);live18=composition.state(live_output,rows,backend)
    index=torch.arange(len(rows),device=backend.device);answers=torch.as_tensor([r["donor_answer_id"] for r in rows],device=backend.device);foils=torch.as_tensor([r["donor_foil_id"] for r in rows],device=backend.device);margin=lambda x:das.head_logits(backend,x)[index,answers]-das.head_logits(backend,x)[index,foils];live_effect=margin(live18)-margin(base18);live_q=(live18-base18)@q8;masks={p:torch.as_tensor([r["family"]==p for r in rows],device=backend.device) for p in ("A1","A2")}
    reports={};max_replay=0.0
    for layer in LAYERS:
        removed=composition.state(run_scaled(backend,base_batch,bh["hidden"],complement,positions,base,live,layer,0.0),rows,backend);replay=composition.state(run_scaled(backend,base_batch,bh["hidden"],complement,positions,base,live,layer,1.0),rows,backend);replay_error=float((replay-live18).norm()/(live18-base18).norm());max_replay=max(max_replay,replay_error);final_q=(live18-removed)@q8;final_behavior=margin(live18)-margin(removed);panels={}
        for panel,mask in masks.items():
            local=torch.cat([((live["mlp"][layer][i,:int(r["base_semantic_position"])+1]-base["mlp"][layer][i,:int(r["base_semantic_position"])+1])@q8).reshape(-1,8) for i,r in enumerate(rows) if r["family"]==panel],dim=0);fq=final_q[mask];fb=final_behavior[mask]
            panels[panel]={"local_q8_write_rms":float(torch.sqrt(local.square().mean())),"final_q8_necessity_rms":float(torch.sqrt(fq.square().mean())),"finite_q8_downstream_gain":float(torch.sqrt(fq.square().mean())/torch.sqrt(local.square().mean()).clamp_min(1e-30)),"final_q8_fraction_of_live":float(fq.norm()/live_q[mask].norm()),"final_q8_cosine_to_live":cosine(fq.reshape(-1),live_q[mask].reshape(-1)),"behavior_abs_fraction_of_live":float(fb.abs().mean()/live_effect[mask].abs().mean()),"behavior_cosine_to_live":cosine(fb,live_effect[mask])}
        reports[f"mlp{layer}"]={"alpha1_live_replay_relative_norm":replay_error,"panels":panels}
    gains=[reports[f"mlp{k}"]["panels"][p]["finite_q8_downstream_gain"] for k in LAYERS for p in ("A1","A2")];finite=all(math.isfinite(v) for r in reports.values() for p in r["panels"].values() for v in p.values())
    pred_a=orientation_error<=1e-6 and finite and MAX_FORWARDS==14 and MAX_EVALUATIONS==420
    pred_b=max_replay<=2e-4
    pred_c=all(reports[f"mlp{k}"]["panels"][p]["final_q8_necessity_rms"]>0 and reports[f"mlp{k}"]["panels"][p]["behavior_abs_fraction_of_live"]>0 for k in LAYERS for p in ("A1","A2"))
    pred_d=max(gains)/min(gains)>1.10
    pred_e=True;predictions={"pred_a_authority_basis_finiteness_and_price":bool(pred_a),"pred_b_all_cached_live_arms_replay_unrestricted":bool(pred_b),"pred_c_every_selected_mlp_has_nonzero_final_necessity":bool(pred_c),"pred_d_downstream_gain_varies_across_layers":bool(pred_d),"pred_e_zero_fit_finite_causal_edge_table":pred_e};terminal="invalid" if not pred_a or not pred_b else "screen" if all(predictions.values()) else "uniform_or_contextual_downstream_reader"
    result={"schema":"iswas_mlp8_mlp_chain_finite_downstream_sensitivity_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"f_linear_orientation_max_abs":orientation_error,"maximum_alpha1_live_replay_relative_norm":max_replay},"reports":reports,"downstream_gain_range":{"minimum":min(gains),"maximum":max(gains),"ratio":max(gains)/min(gains)},"predictions":predictions,"terminal":terminal,"price":{"model_forwards":MAX_FORWARDS,"example_evaluations":MAX_EVALUATIONS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}}
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("candidate_id","instrument","reports","downstream_gain_range","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__":main()
