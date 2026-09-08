#!/usr/bin/env python3
"""Final-state necessity, sufficiency, and A1/A2 payload interchange."""

# BQGATE: EXPERIMENT pred_a_authority_geometry_replay_finiteness_and_price pred_b_final_rank2_removal_is_necessary pred_c_final_rank2_projection_is_sufficient pred_d_construction_payload_swap_preserves_function pred_e_manipulations_are_control_selective
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
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent
from two_by_two_dependency_contract import vector_metrics

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_final_rank2_removal_construction_swap_v1.json"
DIRECT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_rank2_direct_residual_final_head_route_v1_result.json"
BASIS_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_shared_target_control_response_basis_v1_result.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_final_rank2_removal_construction_swap_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_final_rank2_removal_construction_swap_v1"
EXPECTED = {
    "prior": "66d85f698c9148acb254d377841f4533ac8ddeb0070be817b32d887763f2c53b",
    "direct": "d6dd9592660cfad1c632a7b6d580b15faa04f3e762629d76984129e4872a56e8",
    "basis_result": "ffaec3f0bede415f7b9b1664ae7ddf1667978dc26f42cfa2f81e9c82f0f77467",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "manipulation": "6d27835846aa1c48b86fc3be7c2027e2a5ec312b158ca6a8f321ca45eb76440a",
    "finite_router": "48007c4950bd9116601afbfe66e5e4a6d9bb7dbef65f90afc2dd405714809222",
    "basis_contract": "cc05ed73baf02c7ec6f0ec2b9e64a831a1fff6462ab4d8a5ba1f39bc5f70dab1",
    "state_executor": "ae3dc83c4a1954575778fa88744f964efc712e8bc483de078e366a90afee9720",
    "dependency": "4d435dfa6c6f29a34de2b5ba7679aafb3b622cdf0fbf2ea779c58bea982b1fe3",
    "parent": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "v15": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
}
FILES = {"prior": PRIOR, "direct": DIRECT, "basis_result": BASIS_RESULT, "oracles": ORACLES,
         "manipulation": ROOT / "ops/final_rank2_manipulation_contract.py",
         "finite_router": ROOT / "ops/entry12_finite_router_contract.py",
         "basis_contract": ROOT / "ops/entry12_response_basis_contract.py",
         "state_executor": ROOT / "ops/residual_state_mediation_executor.py",
         "dependency": ROOT / "ops/run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.py",
         "parent": ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py",
         "v15": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"}
PRICE_MAX = {"native_capture_forwards": 10, "differentiable_transformer_forwards": 12,
             "transformer_backward_forwards": 0, "model_updates": 0,
             "example_evaluations": 1200, "fit_parameters": 0}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def maxdiff(left, right): return float((left - right).abs().max())


def stack_prefix(torch, tensor, context, panel):
    indices = context["panel_indices"][panel].detach().cpu().tolist()
    return torch.cat([tensor[index, :int(context["base_batch"].semantic_positions[index]) + 1].float()
                      for index in indices], dim=0)


def decode(backend, state):
    logits = 30.0 * backend.torch.tanh(
        backend.model.lm_head(backend.F.rms_norm(state, (backend.model.config.n_embd,))) / 30.0)
    return {"state": state.detach(), "logits": logits.detach()}


def panel_vector(context, output, background, panel):
    indices = context["panel_indices"][panel]
    margin = lambda item: item["logits"][context["index"], context["answer"]] - item["logits"][context["index"], context["foil"]]
    return (margin(output) - margin(background))[indices].detach().cpu().tolist()


def control(backend, context, changed, background, panel):
    indices = context["panel_indices"][panel]
    lp = backend.F.log_softmax(changed["logits"][indices], -1); lq = backend.F.log_softmax(background["logits"][indices], -1)
    kl = (lq.exp() * (lq - lp)).sum(-1); flips = changed["logits"][indices].argmax(-1) != background["logits"][indices].argmax(-1)
    return {"mean_kl": float(kl.mean()), "top1_flip_count": int(flips.sum())}


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    direct = json.loads(DIRECT.read_text()); basis_result = json.loads(BASIS_RESULT.read_text())
    authority = observed == EXPECTED and direct.get("terminal") == "direct_residual_final_head_route"
    dry = {"candidate_id":CANDIDATE_ID,"dryrun":True,"authority_ok":authority,
           "expected_differentiable_forwards":12,"price_max":PRICE_MAX}
    if not authority: raise RuntimeError(f"authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    start=time.perf_counter(); backend=producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters(): parameter.requires_grad_(False)
    counters={name:0 for name in PRICE_MAX}; native=backend.native
    def counted(batch,*,capture):
        counters["native_capture_forwards"]+=1; counters["example_evaluations"]+=len(batch.row_ids); return native(batch,capture=capture)
    backend.native=counted; rows=v15.build_rows(); bank=parent.capture_bank(backend,rows,counters,factors=False)
    contexts={parity:parent.attach_references(backend,parent.subset_context(bank,parent.row_indices(rows,parity=parity)),counters) for parity in (0,1)}
    off={parity:state_executor.execute(parent,backend,context,counters,{},capture=True) for parity,context in contexts.items()}
    oracle_data=json.loads(ORACLES.read_text()); experts={panel:{} for panel in ("A1","A2")}
    for panel,expert in (("A1","A1_oracle"),("A2","A2_oracle")):
        bases=dependency.bases_for_evaluation(backend.torch,backend.device,oracle_data,expert)
        for parity,context in contexts.items(): experts[panel][parity]=state_executor.execute(parent,backend,context,counters,bases[parity],capture=True)

    reports={}; control_reports={}; geometry={}; geometry_error=orthogonality=head_replay=0.0
    for held in (0,1):
        train=1-held; train_context=contexts[train]; base_entry=off[train][1]["entry12"]
        deltas={panel:experts[panel][train][1]["entry12"].float()-base_entry.float() for panel in ("A1","A2")}
        a1=stack_prefix(backend.torch,deltas["A1"],train_context,"A1").mean(0); a2=stack_prefix(backend.torch,deltas["A2"],train_context,"A2").mean(0)
        p=backend.torch.cat([stack_prefix(backend.torch,deltas[panel],train_context,"P") for panel in ("A1","A2")],0)
        bases,geom=basis_contract.fit_bases(backend.torch,a1,a2,p,relative_threshold=1e-6); union=bases["construction_union_rank_le_2"]
        geometry[str(held)]={"rank":int(union.shape[1]),**geom}; orthogonality=max(orthogonality,float((union.T@union-backend.torch.eye(2,device=union.device)).abs().max()))
        geometry_error=max(geometry_error,max(abs(a-b)/max(abs(b),1e-30) for a,b in zip(geom["union_singular_values"],basis_result["fits"][str(held)]["union_singular_values"])))
        context=contexts[held]; off_output=off[held][0]; entries={panel:experts[panel][held][1]["entry12"] for panel in ("A1","A2")}
        truth=finite_router.labels_for_rows(backend.torch,context["rows"],device=backend.device)
        gold_entry=finite_router.routed_absolute(backend.torch,off[held][1]["entry12"],entries,union,truth,context["base_batch"].semantic_positions)
        on=state_executor.execute(parent,backend,context,counters,{},absolute={"entry12":gold_entry})
        head_replay=max(head_replay,maxdiff(decode(backend,off_output["state"])["logits"],off_output["logits"]),maxdiff(decode(backend,on["state"])["logits"],on["logits"]))
        live=manipulation.removal_and_sufficiency(off_output["state"],on["state"],union)
        gamma=backend.torch.tensor(direct["direct_state_law"][str(held)]["recurrent_product"],device=backend.device)
        index=backend.torch.arange(len(context["rows"]),device=backend.device); pos=backend.torch.tensor(context["base_batch"].semantic_positions,device=backend.device)
        entry_delta=gold_entry[index,pos].float()-off[held][1]["entry12"][index,pos].float()
        frozen_on=off_output["state"].float()+gamma*entry_delta
        frozen=manipulation.removal_and_sufficiency(off_output["state"],frozen_on,union)
        variants={"live_suffix":{**live,"on":on["state"]},"frozen_suffix":{**frozen,"on":frozen_on}}
        reports[str(held)]={}; control_reports[str(held)]={}
        for variant,states in variants.items():
            outputs={name:decode(backend,state) for name,state in states.items() if name in ("removed","sufficient","on")}
            outputs["swapped"]=decode(backend,manipulation.paired_payload_swap(backend.torch,off_output["state"],states["parallel"],context["rows"]))
            item={}
            for panel in ("A1","A2"):
                full=panel_vector(context,outputs["on"],off_output,panel); removed=panel_vector(context,outputs["removed"],off_output,panel)
                loss=[a-b for a,b in zip(full,removed)]
                item[panel]={"removal_loss":vector_metrics(loss,full),
                             "sufficiency":vector_metrics(panel_vector(context,outputs["sufficient"],off_output,panel),full),
                             "construction_swap":vector_metrics(panel_vector(context,outputs["swapped"],off_output,panel),full)}
            reports[str(held)][variant]=item
            control_reports[str(held)][variant]={condition:{panel:control(backend,context,outputs[condition],off_output,panel) for panel in ("P","C")}
                                                  for condition in ("removed","sufficient","swapped")}
    counters["fit_parameters"]=0; native_closure=max(c["manual_native_max_abs_error"] for c in contexts.values())
    A=bool(authority and native_closure<=1e-4 and head_replay<=1e-4 and geometry_error<=1e-5 and orthogonality<=1e-5
           and finite({"reports":reports,"controls":control_reports}) and counters["differentiable_transformer_forwards"]==12
           and all(counters[n]<=PRICE_MAX[n] for n in PRICE_MAX))
    gate=lambda metric:all(reports[str(f)][v][p][metric]["signed_projection"]>=.75 and reports[str(f)][v][p][metric]["direction_fraction"]>=.875
                           for f in (0,1) for v in ("live_suffix","frozen_suffix") for p in ("A1","A2"))
    B=gate("removal_loss"); C=gate("sufficiency"); D=gate("construction_swap")
    E=all(item["top1_flip_count"]==0 and item["mean_kl"]<=.02 for folds in control_reports.values() for variants in folds.values() for conditions in variants.values() for item in conditions.values())
    predictions=dict(zip(("pred_a_authority_geometry_replay_finiteness_and_price","pred_b_final_rank2_removal_is_necessary",
        "pred_c_final_rank2_projection_is_sufficient","pred_d_construction_payload_swap_preserves_function","pred_e_manipulations_are_control_selective"),map(bool,(A,B,C,D,E))))
    terminal="invalid" if not A else "final_rank2_reusable_payload" if all((B,C,D,E)) else "final_rank2_construction_split" if B and C else "final_rank2_manipulation_null"
    result={"schema":"temporal_iswas_v15_final_rank2_removal_construction_swap_result_v1","candidate_id":CANDIDATE_ID,
            "started_utc":datetime.now(timezone.utc).isoformat(),"serial_seconds":time.perf_counter()-start,"authority_sha256":EXPECTED,
            "geometry":geometry,"reports":reports,"controls":control_reports,
            "instrument":{"manual_native_max_abs_error":native_closure,"exact_head_replay_max_abs_error":head_replay,
                          "geometry_relative_max_error":geometry_error,"orthogonality_max_abs_error":orthogonality},
            "predictions":predictions,"terminal":terminal,"price":{**counters,"maxima":PRICE_MAX}}
    atomic_create_json(OUT,result); print(json.dumps({k:result[k] for k in ("predictions","terminal","reports","controls","instrument","price")},sort_keys=True))


if __name__=="__main__": main()
