#!/usr/bin/env python3
"""Fresh end-to-end confirmation of the frozen parsimonious response program."""

# BQGATE: EXPERIMENT pred_a_authority_capability_self_clamp_finiteness_and_price pred_b_parsimonious_program_recovers_behavior_on_both_panels pred_c_parsimonious_program_recovers_q8_on_both_panels pred_d_parsimonious_program_predicts_full_vocabulary_effect pred_e_fidelity_variant_is_no_worse_and_zero_fit
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_attn15_remainder_head_greedy_v1 as head_program
import run_iswas_mlp8_main_mlp9_aux_composition_v1 as composition
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/iswas_mlp8_parsimonious_program_fresh_v12_confirmation_v1.json"
HEAD_RESULT=ROOT/"circuits/followups/iswas_mlp8_attn15_remainder_head_greedy_v1_result.json"
HEAD_RUNNER=ROOT/"ops/run_iswas_mlp8_attn15_remainder_head_greedy_v1.py"
SENSITIVITY=ROOT/"circuits/followups/iswas_mlp8_mlp_chain_finite_downstream_sensitivity_v1_result.json"
CAPABILITY=ROOT/"circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
BUILDER=ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
OUT=ROOT/"circuits/followups/iswas_mlp8_parsimonious_program_fresh_v12_confirmation_v1_result.json"
CANDIDATE_ID="cross_task.iswas_mlp8_parsimonious_program_fresh_v12_confirmation_v1"
EXPECTED={"prior":"fc0b90becd73367fe4993136e352f51ae4e144d5a1d16f0fa56cc71d2ea3fbb5","head_result":"5cf2aa2db970b454fd01ab0ed0fef47ac46616001ca5cfced0539a77f7d6a6af","head_runner":"6c562f558f8e80d2c14355eb51144f5b02fe418bd64400a01ea9bfb14e9d2cd2","sensitivity":"b3302b9e83b17117335f59bfc0eed32d5e994669eb5ff71dafad8e1170c306cf","capability":"67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4","builder":"2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2"}
PARSIMONIOUS=(1,);FIDELITY=(1,0,8,7,2);MAX_FORWARDS,MAX_EVALUATIONS=7,210

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def cosine(x,y):
    d=float(x.norm()*y.norm());return float((x*y).sum())/d if d else 0.0

def main():
    paths={"prior":PRIOR,"head_result":HEAD_RESULT,"head_runner":HEAD_RUNNER,"sensitivity":SENSITIVITY,"capability":CAPABILITY,"builder":BUILDER}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("fresh program authority changed")
    prior,head_result,sensitivity,capability,subspace=[json.loads(p.read_text()) for p in (PRIOR,HEAD_RESULT,SENSITIVITY,CAPABILITY,weight.SUBSPACE)];allowed={x for ids in capability["jointly_capable_row_ids"].values() for x in ids};rows=[r for r in candidate.build_rows() if r["family"] in ("A1","A2") and r["row_id"] in allowed]
    if prior.get("candidate_id")!=CANDIDATE_ID or head_result.get("terminal")!="screen" or sensitivity.get("terminal")!="screen" or len(rows)!=30 or {p:sum(r["family"]==p for r in rows) for p in ("A1","A2")}!={"A1":14,"A2":16}:raise RuntimeError("frozen program or population changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"rows":30,"panel_counts":{"A1":14,"A2":16},"parsimonious_heads":list(PARSIMONIOUS),"fidelity_heads":list(FIDELITY),"model_forwards_max":MAX_FORWARDS,"example_evaluations_max":MAX_EVALUATIONS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch;model=backend.model
    family,_s,_e=family_builder.build_family(backend,subspace);gain=math.prod(float(model.transformer.h[k].lambdas[0].detach().float()) for k in range(12,18));modes,orientation_error,_wrong=overlap.residual_modes(backend,family[8],gain);q8=torch.linalg.qr(modes,mode="reduced").Q
    base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor");_bo,bh=weight.capture_mlp8(backend,base_batch);_do,dh=weight.capture_mlp8(backend,donor_batch);down=model.transformer.h[8].mlp.Down.weight.detach().float();_u,_s,vh=torch.linalg.svd(q8.T@down,full_matrices=False);delta=dh["hidden"].float()-bh["hidden"].float();complement=delta-weight.project(delta,vh,vh.shape[0]);positions=[weight.postcue_positions(r) for r in rows]
    base_output,base=composition.capture_modules(backend,lambda:backend.native(base_batch,capture=True));live_output,live=composition.capture_modules(backend,lambda:weight.run_hidden_patch(backend,base_batch,bh["hidden"],complement,positions));outputs={"parsimonious":head_program.run_heads(backend,base_batch,bh["hidden"],complement,positions,base,live,PARSIMONIOUS),"fidelity":head_program.run_heads(backend,base_batch,bh["hidden"],complement,positions,base,live,FIDELITY)};self_output=head_program.run_heads(backend,base_batch,bh["hidden"],complement,positions,base,base,(),actuate=False)
    state=lambda o:composition.state(o,rows,backend);base18,live18=state(base_output),state(live_output);states={k:state(v) for k,v in outputs.items()};self_error=float((state(self_output)-base18).abs().max());index=torch.arange(len(rows),device=backend.device);answers=torch.as_tensor([r["donor_answer_id"] for r in rows],device=backend.device);foils=torch.as_tensor([r["donor_foil_id"] for r in rows],device=backend.device);logits=lambda x:das.head_logits(backend,x);margin=lambda x:logits(x)[index,answers]-logits(x)[index,foils];live_effect=margin(live18)-margin(base18);live_q=(live18-base18)@q8;live_vocab=logits(live18)-logits(base18);live_vocab=live_vocab-live_vocab.mean(dim=-1,keepdim=True);masks={p:torch.as_tensor([r["family"]==p for r in rows],device=backend.device) for p in ("A1","A2")};reports={}
    for arm,value in states.items():
        effect=margin(value)-margin(base18);coord=(value-base18)@q8;vocab=logits(value)-logits(base18);vocab=vocab-vocab.mean(dim=-1,keepdim=True);reports[arm]={}
        for panel,mask in masks.items():
            e,le=effect[mask],live_effect[mask];z,lz=coord[mask],live_q[mask];v,lv=vocab[mask],live_vocab[mask];reports[arm][panel]={"behavior_abs_fraction_of_live":float(e.abs().mean()/le.abs().mean()),"behavior_cosine_to_live":cosine(e,le),"behavior_relative_rmse":float(torch.sqrt(((e-le)**2).mean()/le.square().mean())),"direction_fraction":float(((e*le)>0).float().mean()),"q8_norm_fraction_of_live":float(z.norm()/lz.norm()),"q8_cosine_to_live":cosine(z.reshape(-1),lz.reshape(-1)),"q8_relative_rmse":float(torch.sqrt(((z-lz)**2).mean()/lz.square().mean())),"full_vocab_cosine_to_live":cosine(v.reshape(-1),lv.reshape(-1)),"full_vocab_relative_rmse":float(torch.sqrt(((v-lv)**2).mean()/lv.square().mean()))}
    finite=all(math.isfinite(x) for arm in reports.values() for panel in arm.values() for x in panel.values());joint=lambda arm:sum(reports[arm][p][k]**2 for p in ("A1","A2") for k in ("behavior_relative_rmse","q8_relative_rmse","full_vocab_relative_rmse"))
    pred_a=orientation_error<=1e-6 and self_error<=.05 and finite
    pred_b=all(reports["parsimonious"][p]["behavior_abs_fraction_of_live"]>=.80 and reports["parsimonious"][p]["behavior_cosine_to_live"]>=.98 and reports["parsimonious"][p]["direction_fraction"]>=.90 for p in masks)
    pred_c=all(reports["parsimonious"][p]["q8_norm_fraction_of_live"]>=.80 and reports["parsimonious"][p]["q8_cosine_to_live"]>=.95 for p in masks)
    pred_d=all(reports["parsimonious"][p]["full_vocab_cosine_to_live"]>=.90 and reports["parsimonious"][p]["full_vocab_relative_rmse"]<=.50 for p in masks)
    pred_e=joint("fidelity")<=1.05*joint("parsimonious");predictions={"pred_a_authority_capability_self_clamp_finiteness_and_price":bool(pred_a),"pred_b_parsimonious_program_recovers_behavior_on_both_panels":bool(pred_b),"pred_c_parsimonious_program_recovers_q8_on_both_panels":bool(pred_c),"pred_d_parsimonious_program_predicts_full_vocabulary_effect":bool(pred_d),"pred_e_fidelity_variant_is_no_worse_and_zero_fit":bool(pred_e)};terminal="invalid" if not pred_a else "release_candidate" if all(predictions.values()) else "fresh_program_shortfall"
    result={"schema":"iswas_mlp8_parsimonious_program_fresh_v12_confirmation_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"f_linear_orientation_max_abs":orientation_error,"base_self_clamp_max_abs":self_error,"rows":len(rows)},"program":{"semantic_seed":"attention9 H1/H4 + MLP9 + attention11 H1/H3 + attention15 H5","downstream_mlps":[10,11,12,13,14],"late_attention_parsimonious":[1],"late_attention_fidelity":list(FIDELITY)},"reports":reports,"joint_error":{"parsimonious":joint("parsimonious"),"fidelity":joint("fidelity")},"predictions":predictions,"terminal":terminal,"price":{"model_forwards":MAX_FORWARDS,"example_evaluations":MAX_EVALUATIONS,"fit_updates":0,"transformer_backwards":0,"model_updates":0}}
    atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("candidate_id","instrument","program","reports","joint_error","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__":main()
