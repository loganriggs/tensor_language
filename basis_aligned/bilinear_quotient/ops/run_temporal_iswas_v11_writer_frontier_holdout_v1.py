#!/usr/bin/env python3
"""Sealed-row confirmation of the conditional canonical writer frontier."""

# BQGATE: EXPERIMENT pred_a_authority_disjoint_capability_replay_self_patch_finiteness_price pred_b_frontier_predicts_all_cells pred_c_frontier_beats_l9_baseline pred_d_l7h8_has_conditional_increment pred_e_fidelity_set_is_no_worse
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path

import numpy as np
import circuit_candidate_temporal_auxiliary_fresh_cues_v11 as temporal
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as iswas
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_q8_finite_causal_hankel_v1 as parent
import run_temporal_iswas_upstream_full_response_mode_atlas_v1 as screen_runner
import run_temporal_iswas_v11_writer_group_factorial_v1 as discovery_runner

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_iswas_v11_writer_frontier_holdout_v1.json"
DISCOVERY=ROOT/"circuits/followups/temporal_iswas_v11_writer_group_factorial_v1_result.json"
DISCOVERY_RUNNER=ROOT/"ops/run_temporal_iswas_v11_writer_group_factorial_v1.py"
TEMPORAL_BUILDER=ROOT/"ops/circuit_candidate_temporal_auxiliary_fresh_cues_v11.py"
TEMPORAL_CAPABILITY=ROOT/"circuits/followups/temporal_auxiliary_will_had_fresh_v11_capability_v1_result.json"
ISWAS_BUILDER=ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11.py"
ISWAS_CAPABILITY=ROOT/"circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
OUT=ROOT/"circuits/followups/temporal_iswas_v11_writer_frontier_holdout_v1_result.json"
CANDIDATE_ID="cross_task.temporal_iswas_v11_writer_frontier_holdout_v1"
ARMS=(4,28,30,31)
EXPECTED={"prior":"176014d8211a1261d9a6b8998e6b2cd4b2a999b701c4634ea95b310e922e7af0","discovery":"a3379673ef5b72aa3b50a1e2a5f32f733f78d81b961d149daeefdb4e20eea0bb","discovery_runner":"ddbec08031825b6cc3455b6768541d9e0ad9cb24b3b390bb9232af49e61af14f","temporal_builder":"f75b17669a5fc5299d21f5b44e91530c03c71d75181683c7b6728cb95c862450","temporal_capability":"0330dc5a4f85bc68c4da6f98af2f4208335e65c644ddedd5d8cc487368091026","iswas_builder":"fbd47713fafcb87fc30ba339d175f7d06770ce36b93b6035e8848455529344ec","iswas_capability":"6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2"}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def tensor_sha(tensor):return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()
def sealed_rows(builder,capability,panel):
    allowed=set(capability["jointly_capable_row_ids"][panel]);rows=[row for row in builder.build_rows() if row.get("transform_id",row.get("family"))==panel and row["row_id"] in allowed]
    return rows[7:14]
def main():
    paths={"prior":PRIOR,"discovery":DISCOVERY,"discovery_runner":DISCOVERY_RUNNER,"temporal_builder":TEMPORAL_BUILDER,"temporal_capability":TEMPORAL_CAPABILITY,"iswas_builder":ISWAS_BUILDER,"iswas_capability":ISWAS_CAPABILITY}
    if {key:sha(value) for key,value in paths.items()}!=EXPECTED:raise RuntimeError("writer frontier holdout authority changed")
    discovery=json.loads(DISCOVERY.read_text());tcap=json.loads(TEMPORAL_CAPABILITY.read_text());icap=json.loads(ISWAS_CAPABILITY.read_text())
    rows=sum((sealed_rows(temporal,tcap,panel) for panel in ("A1","A2")),[])+sum((sealed_rows(iswas,icap,panel) for panel in ("A1","A2")),[])
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"rows":len(rows),"arms":list(ARMS),"selected_mask":30,"fidelity_mask":31,"model_forwards_max":10,"example_evaluations_max":280,"fit_updates":0,"model_updates":0,"transformer_backwards":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    discovery_rows=sum((discovery_runner.selected_rows(temporal,tcap,panel) for panel in ("A1","A2")),[])+sum((discovery_runner.selected_rows(iswas,icap,panel) for panel in ("A1","A2")),[])
    discovery_ids={row["row_id"] for row in discovery_rows};sealed_ids={row["row_id"] for row in rows}
    disjoint=len(discovery_rows)==len(discovery_ids)==28 and len(rows)==len(sealed_ids)==28 and not discovery_ids&sealed_ids
    capability_ok=discovery.get("terminal")=="writer_transfer_null" and tcap.get("terminal")=="manifest" and icap.get("terminal")=="screen"
    if len(rows)!=28 or len(ARMS)!=4:raise RuntimeError("frozen holdout population changed")
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
    base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor")
    base_output,base_cache=screen_runner.capture(backend,base_batch);donor_output,donor_cache=screen_runner.capture(backend,donor_batch)
    aligned=all(base_cache[key].shape==donor_cache[key].shape for key in base_cache)
    group_names=tuple(discovery_runner.GROUPS);union_sites=tuple(site for sites in discovery_runner.GROUPS.values() for site in sites)
    self_output,self_pre=screen_runner.run_patch(backend,base_batch,base_cache,union_sites)
    base_state,donor_state,self_state=(screen_runner.states(torch,backend,output,rows) for output in (base_output,donor_output,self_output))
    base_pre,donor_pre=screen_runner.query_rows(base_cache["pre_h3"],base_batch),screen_runner.query_rows(donor_cache["pre_h3"],donor_batch)
    self_pre=screen_runner.query_rows(self_pre,base_batch)
    reconstruction=0.0
    for layer in screen_runner.LAYERS:
        _replay,captured=screen_runner.attention_eval.capture_layer_attention(backend,base_batch,layer)
        reconstruction=max(reconstruction,float((captured["head_output"].reshape_as(base_cache[f"L{layer}H0"])-base_cache[f"L{layer}H0"]).abs().max()))
    atlas_result=json.loads(screen_runner.ATLAS_RESULT.read_text());subspace=json.loads(parent.SUBSPACE.read_text())
    family,_singular,_energy=parent.family_builder.build_family(backend,subspace);q=family[8]
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
    base_margin,donor_margin=margin(base_state),margin(donor_state);full_pre=(donor_pre-base_pre)@h3_inputs;full_final=(donor_state-base_state)@physical_source
    task_indices={"temporal":torch.arange(0,14,device=backend.device),"iswas":torch.arange(14,28,device=backend.device)}
    outputs={};forwards,evaluations=6,168
    for mask in ARMS:
        sites=tuple(site for bit,name in enumerate(group_names) if mask&(1<<bit) for site in discovery_runner.GROUPS[name])
        output,pre=screen_runner.run_patch(backend,base_batch,donor_cache,sites);state=screen_runner.states(torch,backend,output,rows)
        outputs[mask]=(screen_runner.query_rows(pre,base_batch),state);forwards+=1;evaluations+=28
    arm_metrics={}
    for mask,(pre,state) in outputs.items():
        patch_pre=(pre-base_pre)@h3_inputs;patch_final=(state-base_state)@physical_source;recovery=(margin(state)-base_margin)/(donor_margin-base_margin);tasks={}
        for task,ids in task_indices.items():
            tasks[task]={}
            for mode in range(2):
                stats=screen_runner.vector_stats(torch,patch_pre[ids,mode],full_pre[ids,mode]);stats["squared_residual"]=float(((patch_pre[ids,mode]-full_pre[ids,mode]).square().sum())/(full_pre[ids,mode].square().sum()))
                tasks[task][f"mode{mode+1}"]={"pre_h3":stats,"final_state":screen_runner.vector_stats(torch,patch_final[ids,mode],full_final[ids,mode])}
        arm_metrics[str(mask)]={"groups":[name for bit,name in enumerate(group_names) if mask&(1<<bit)],"behavior":{"mean_recovery":float(recovery.mean()),"direction_fraction":float((recovery>0).float().mean())},"tasks":tasks}
    def residual(mask,task,mode):return arm_metrics[str(mask)]["tasks"][task][mode]["pre_h3"]["squared_residual"]
    cells=((task,mode) for task in task_indices for mode in ("mode1","mode2"))
    pred_b=all(residual(30,task,mode)<=.20 and arm_metrics["30"]["tasks"][task][mode]["pre_h3"]["cosine"]>=.90 for task,mode in cells) and arm_metrics["30"]["behavior"]["direction_fraction"]>=.90
    improved_cells=sum(residual(30,task,mode)<=.90*residual(4,task,mode) for task in task_indices for mode in ("mode1","mode2"));pred_c=improved_cells>=3
    conditional_reduction=1.0-residual(30,"iswas","mode2")/residual(28,"iswas","mode2");pred_d=conditional_reduction>=.20
    mean_residual={str(mask):float(np.mean([residual(mask,task,mode) for task in task_indices for mode in ("mode1","mode2")])) for mask in ARMS};pred_e=mean_residual["31"]<=1.05*mean_residual["30"]
    self_error=max(float((self_pre-base_pre).abs().max()),float((self_state-base_state).abs().max()),float((margin(self_state)-base_margin).abs().max()))
    finite=all(math.isfinite(value) for metrics in arm_metrics.values() for task in metrics["tasks"].values() for mode in task.values() for family in mode.values() for value in family.values()) and all(math.isfinite(value) for value in mean_residual.values()) and math.isfinite(conditional_reduction)
    pred_a=disjoint and capability_ok and aligned and physical_hash_ok and orientation_error<=1e-6 and reconstruction<=5e-4 and self_error<=1e-4 and finite and forwards==10 and evaluations==280 and len(arm_metrics)==4
    predictions={"pred_a_authority_disjoint_capability_replay_self_patch_finiteness_price":bool(pred_a),"pred_b_frontier_predicts_all_cells":bool(pred_b),"pred_c_frontier_beats_l9_baseline":bool(pred_c),"pred_d_l7h8_has_conditional_increment":bool(pred_d),"pred_e_fidelity_set_is_no_worse":bool(pred_e)}
    terminal="invalid" if not pred_a else "conditional_writer_screen" if all(predictions.values()) else "conditional_increment_null" if pred_b and pred_c and not pred_d else "frontier_transfer_null"
    result={"schema":"temporal_iswas_v11_writer_frontier_holdout_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"authority_sha256":EXPECTED,"dryrun":dryrun,"population":{"discovery_row_ids":sorted(discovery_ids),"sealed_row_ids":sorted(sealed_ids),"disjoint":disjoint},"instrument":{"capability_ok":capability_ok,"aligned_shapes":aligned,"physical_source_hash_ok":physical_hash_ok,"orientation_max_abs":orientation_error,"attention_reconstruction_max_abs":reconstruction,"self_patch_max_abs":self_error},"arm_metrics":arm_metrics,"mean_four_cell_residual":mean_residual,"improved_cells_over_l9":improved_cells,"iswas_mode2_l7h8_conditional_residual_reduction":conditional_reduction,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":forwards,"example_evaluations":evaluations,"fit_updates":0,"model_updates":0,"transformer_backwards":0},"serial_seconds":time.perf_counter()-started}
    atomic_create_json(OUT,result);print(json.dumps({key:result[key] for key in ("candidate_id","instrument","mean_four_cell_residual","improved_cells_over_l9","iswas_mode2_l7h8_conditional_residual_reduction","predictions","terminal","price")},sort_keys=True))
if __name__=="__main__":main()
