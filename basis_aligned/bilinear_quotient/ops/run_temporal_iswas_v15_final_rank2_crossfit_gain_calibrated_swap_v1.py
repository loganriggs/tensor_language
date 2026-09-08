#!/usr/bin/env python3
"""Outcome-free cross-fit scalar calibration of A1/A2 final-state payload swaps."""

# BQGATE: EXPERIMENT pred_a_authority_geometry_crossfit_finiteness_and_price pred_b_construction_gains_are_stable_and_reciprocal pred_c_calibrated_swap_preserves_both_constructions pred_d_calibration_improves_unscaled_amplitude pred_e_calibrated_swap_is_control_selective
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import entry12_finite_router_contract as finite_router
import entry12_response_basis_contract as basis_contract
import final_rank2_manipulation_contract as manipulation
import residual_state_mediation_executor as state_executor
import run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1 as dependency
import run_temporal_iswas_v15_final_rank2_removal_construction_swap_v1 as previous_runner
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent
from two_by_two_dependency_contract import vector_metrics

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_iswas_v15_final_rank2_crossfit_gain_calibrated_swap_v1.json"
UNSCALED=ROOT/"circuits/followups/temporal_iswas_v15_final_rank2_removal_construction_swap_v1_result.json"
DIRECT=ROOT/"circuits/followups/temporal_iswas_v15_entry12_rank2_direct_residual_final_head_route_v1_result.json"
BASIS_RESULT=ROOT/"circuits/followups/temporal_iswas_v15_entry12_shared_target_control_response_basis_v1_result.json"
ORACLES=ROOT/"circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
OUT=ROOT/"circuits/followups/temporal_iswas_v15_final_rank2_crossfit_gain_calibrated_swap_v1_result.json"
CANDIDATE_ID="temporal_auxiliary.iswas_v15_final_rank2_crossfit_gain_calibrated_swap_v1"
EXPECTED={
 "prior":"51d0d81843e2d76886ee14609fdda14fb4f6f5f9d575031b2fddcc71ce0ebec0",
 "unscaled":"df0f6b4d3a43ea5a2e88281bdf7c95afab99e9329cae7edba5c9aee86480bded",
 "direct":"d6dd9592660cfad1c632a7b6d580b15faa04f3e762629d76984129e4872a56e8",
 "basis_result":"ffaec3f0bede415f7b9b1664ae7ddf1667978dc26f42cfa2f81e9c82f0f77467",
 "oracles":"dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
 "manipulation":"77b109846d489f06489221bd327b97e72bf48f25bc8a898968b8eb5d36c67e30",
 "previous_runner":"c33e962f4543f8b7569d21a6eea2d0bea513fa9c52f15aefcda719b52ba59834",
 "finite_router":"48007c4950bd9116601afbfe66e5e4a6d9bb7dbef65f90afc2dd405714809222",
 "basis_contract":"cc05ed73baf02c7ec6f0ec2b9e64a831a1fff6462ab4d8a5ba1f39bc5f70dab1",
 "state_executor":"ae3dc83c4a1954575778fa88744f964efc712e8bc483de078e366a90afee9720",
 "dependency":"4d435dfa6c6f29a34de2b5ba7679aafb3b622cdf0fbf2ea779c58bea982b1fe3",
 "parent":"0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
 "v15":"7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645"}
FILES={"prior":PRIOR,"unscaled":UNSCALED,"direct":DIRECT,"basis_result":BASIS_RESULT,"oracles":ORACLES,
 "manipulation":ROOT/"ops/final_rank2_manipulation_contract.py",
 "previous_runner":ROOT/"ops/run_temporal_iswas_v15_final_rank2_removal_construction_swap_v1.py",
 "finite_router":ROOT/"ops/entry12_finite_router_contract.py","basis_contract":ROOT/"ops/entry12_response_basis_contract.py",
 "state_executor":ROOT/"ops/residual_state_mediation_executor.py","dependency":ROOT/"ops/run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.py",
 "parent":ROOT/"ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py","v15":ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"}
PRICE_MAX={"native_capture_forwards":10,"differentiable_transformer_forwards":14,"transformer_backward_forwards":0,
           "model_updates":0,"example_evaluations":1400,"fit_parameters":4}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def finite(value):
 if isinstance(value,dict):return all(finite(item) for item in value.values())
 if isinstance(value,(list,tuple)):return all(finite(item) for item in value)
 return not isinstance(value,float) or math.isfinite(value)

def panel_mean_norm(torch,parallel,context,panel):
 indices=context["panel_indices"][panel]
 values=parallel[indices].norm(dim=1)
 if not len(values) or float(values.mean())<=0:raise RuntimeError("payload norm is empty or zero")
 return float(values.mean())

def main():
 observed={n:sha(p) for n,p in FILES.items()}; unscaled=json.loads(UNSCALED.read_text()); direct=json.loads(DIRECT.read_text()); basis_result=json.loads(BASIS_RESULT.read_text())
 authority=observed==EXPECTED and unscaled.get("terminal")=="final_rank2_construction_split" and direct.get("terminal")=="direct_residual_final_head_route"
 dry={"candidate_id":CANDIDATE_ID,"dryrun":True,"authority_ok":authority,"expected_differentiable_forwards":14,"price_max":PRICE_MAX}
 if not authority:raise RuntimeError(f"authority changed: {observed}")
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 start=time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda")
 for parameter in backend.model.parameters():parameter.requires_grad_(False)
 counters={n:0 for n in PRICE_MAX};native=backend.native
 def counted(batch,*,capture):counters["native_capture_forwards"]+=1;counters["example_evaluations"]+=len(batch.row_ids);return native(batch,capture=capture)
 backend.native=counted;rows=v15.build_rows();bank=parent.capture_bank(backend,rows,counters,factors=False)
 contexts={p:parent.attach_references(backend,parent.subset_context(bank,parent.row_indices(rows,parity=p)),counters) for p in (0,1)}
 off={p:state_executor.execute(parent,backend,c,counters,{},capture=True) for p,c in contexts.items()}
 oracle_data=json.loads(ORACLES.read_text());experts={panel:{} for panel in ("A1","A2")}
 for panel,expert in (("A1","A1_oracle"),("A2","A2_oracle")):
  bases=dependency.bases_for_evaluation(backend.torch,backend.device,oracle_data,expert)
  for p,c in contexts.items():experts[panel][p]=state_executor.execute(parent,backend,c,counters,bases[p],capture=True)

 reports={};gains={};controls={};geometry_error=orthogonality=head_replay=0.0;disjoint=True
 for held in (0,1):
  train=1-held;tc=contexts[train];base_entry=off[train][1]["entry12"]
  deltas={p:experts[p][train][1]["entry12"].float()-base_entry.float() for p in ("A1","A2")}
  a1=previous_runner.stack_prefix(backend.torch,deltas["A1"],tc,"A1").mean(0);a2=previous_runner.stack_prefix(backend.torch,deltas["A2"],tc,"A2").mean(0)
  p_rows=backend.torch.cat([previous_runner.stack_prefix(backend.torch,deltas[p],tc,"P") for p in ("A1","A2")],0)
  bases,geom=basis_contract.fit_bases(backend.torch,a1,a2,p_rows,relative_threshold=1e-6);union=bases["construction_union_rank_le_2"]
  orthogonality=max(orthogonality,float((union.T@union-backend.torch.eye(2,device=union.device)).abs().max()))
  geometry_error=max(geometry_error,max(abs(a-b)/max(abs(b),1e-30) for a,b in zip(geom["union_singular_values"],basis_result["fits"][str(held)]["union_singular_values"])))
  states={}
  for split in (train,held):
   context=contexts[split];entries={p:experts[p][split][1]["entry12"] for p in ("A1","A2")};truth=finite_router.labels_for_rows(backend.torch,context["rows"],device=backend.device)
   entry=finite_router.routed_absolute(backend.torch,off[split][1]["entry12"],entries,union,truth,context["base_batch"].semantic_positions)
   on=state_executor.execute(parent,backend,context,counters,{},absolute={"entry12":entry})
   head_replay=max(head_replay,previous_runner.maxdiff(previous_runner.decode(backend,on["state"])["logits"],on["logits"]))
   index=backend.torch.arange(len(context["rows"]),device=backend.device);pos=backend.torch.tensor(context["base_batch"].semantic_positions,device=backend.device)
   gamma=backend.torch.tensor(direct["direct_state_law"][str(split)]["recurrent_product"],device=backend.device)
   frozen_on=off[split][0]["state"].float()+gamma*(entry[index,pos].float()-off[split][1]["entry12"][index,pos].float())
   states[split]={"live_suffix":(on["state"],manipulation.projected_delta(off[split][0]["state"],on["state"],union)),
                  "frozen_suffix":(frozen_on,manipulation.projected_delta(off[split][0]["state"],frozen_on,union))}
  gains[str(held)]={};reports[str(held)]={};controls[str(held)]={}
  for variant in ("live_suffix","frozen_suffix"):
   train_parallel=states[train][variant][1]
   norms={p:panel_mean_norm(backend.torch,train_parallel,contexts[train],p) for p in ("A1","A2")}
   fit={"A1_from_A2":norms["A1"]/norms["A2"],"A2_from_A1":norms["A2"]/norms["A1"]}
   gains[str(held)][variant]={"fit_parity":train,"mean_payload_norms":norms,**fit,"reciprocal_error":abs(fit["A1_from_A2"]*fit["A2_from_A1"]-1)}
   held_on,held_parallel=states[held][variant];context=contexts[held];off_output=off[held][0]
   swapped=previous_runner.decode(backend,manipulation.paired_scaled_payload_swap(backend.torch,off_output["state"],held_parallel,context["rows"],fit))
   on_output=previous_runner.decode(backend,held_on)
   reports[str(held)][variant]={}
   for panel in ("A1","A2"):
    full=previous_runner.panel_vector(context,on_output,off_output,panel);response=previous_runner.panel_vector(context,swapped,off_output,panel)
    metric=vector_metrics(response,full);old=unscaled["reports"][str(held)][variant][panel]["construction_swap"]["signed_projection"]
    reports[str(held)][variant][panel]={"calibrated_swap":metric,"unscaled_signed_projection":old,
                                      "absolute_error_improvement":abs(old-1)-abs(metric["signed_projection"]-1)}
   controls[str(held)][variant]={p:previous_runner.control(backend,context,swapped,off_output,p) for p in ("P","C")}
 counters["fit_parameters"]=4;native_closure=max(c["manual_native_max_abs_error"] for c in contexts.values())
 stability={v:{k:abs(gains["0"][v][k]-gains["1"][v][k])/max(abs(gains["0"][v][k]),abs(gains["1"][v][k]),1e-30)
               for k in ("A1_from_A2","A2_from_A1")} for v in ("live_suffix","frozen_suffix")}
 A=bool(authority and disjoint and native_closure<=1e-4 and head_replay<=1e-4 and geometry_error<=1e-5 and orthogonality<=1e-5
        and finite({"reports":reports,"gains":gains,"controls":controls}) and counters["differentiable_transformer_forwards"]==14
        and all(counters[n]<=PRICE_MAX[n] for n in PRICE_MAX))
 B=all(item["reciprocal_error"]<=1e-6 and item["A1_from_A2"]>0 and item["A2_from_A1"]>0 for folds in gains.values() for item in folds.values()) and all(x<=.2 for v in stability.values() for x in v.values())
 C=all(.75<=item["calibrated_swap"]["signed_projection"]<=1.25 and item["calibrated_swap"]["cosine"]>=.95 and item["calibrated_swap"]["direction_fraction"]>=.875 for folds in reports.values() for variants in folds.values() for item in variants.values())
 D=all(item["absolute_error_improvement"]>0 for folds in reports.values() for variants in folds.values() for item in variants.values())
 E=all(item["top1_flip_count"]==0 and item["mean_kl"]<=.02 for folds in controls.values() for variants in folds.values() for item in variants.values())
 predictions=dict(zip(("pred_a_authority_geometry_crossfit_finiteness_and_price","pred_b_construction_gains_are_stable_and_reciprocal",
  "pred_c_calibrated_swap_preserves_both_constructions","pred_d_calibration_improves_unscaled_amplitude","pred_e_calibrated_swap_is_control_selective"),map(bool,(A,B,C,D,E))))
 terminal="invalid" if not A else "shared_payload_with_construction_gain" if all((B,C,D,E)) else "gain_calibrated_swap_null"
 result={"schema":"temporal_iswas_v15_final_rank2_crossfit_gain_calibrated_swap_result_v1","candidate_id":CANDIDATE_ID,
  "started_utc":datetime.now(timezone.utc).isoformat(),"serial_seconds":time.perf_counter()-start,"authority_sha256":EXPECTED,
  "gains":gains,"gain_crossfold_relative_difference":stability,"reports":reports,"controls":controls,
  "instrument":{"manual_native_max_abs_error":native_closure,"exact_head_replay_max_abs_error":head_replay,
                "geometry_relative_max_error":geometry_error,"orthogonality_max_abs_error":orthogonality,"train_held_disjoint":disjoint},
  "predictions":predictions,"terminal":terminal,"price":{**counters,"maxima":PRICE_MAX}}
 atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("predictions","terminal","gains","gain_crossfold_relative_difference","reports","controls","instrument","price")},sort_keys=True))

if __name__=="__main__":main()
