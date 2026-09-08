#!/usr/bin/env python3
"""Full layer-15 module and singleton-head mediation atlas for admitted oracle routes."""

# BQGATE: EXPERIMENT pred_a_admission_authority_capture_replay_closure_finiteness_and_price pred_b_complete_attention15_module_mediates_live_transfer pred_c_l15h5_is_stable_dominant_singleton_mediator pred_d_singleton_reset_losses_approximately_compose pred_e_l15h5_mediation_is_control_selective pred_f_l15h5_mediation_transfers_to_v16
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import attention15_head_module_mediation_executor as executor
import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16 as v16
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import head_response_mediation_contract as contract
import head_response_mediation_scorer as scorer
import run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1 as dependency
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_attention15_head_module_mediation_atlas_v1.json"
ADMISSION = ROOT / "circuits/followups/temporal_iswas_v15_attention15_head_module_mediation_atlas_v1_admission.json"
DEPENDENCY = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1_result.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_attention15_head_module_mediation_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_attention15_head_module_mediation_atlas_v1"
EXPECTED = {
 "prior":"16227c886d9dd3e14aa863591dfe624b773565371c422e2e87e00da02cbc52b9",
 "admission":"5bc43016ac9df0cb7fe9a9275c044999f68952aa8c797b28344e13b328ab747f",
 "dependency":"f80e82ad090f6fada1f98624f1e6040dcad3b1ae432f2f48caec98d75c15558f",
 "oracles":"dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
 "contract":"9a3097fd1baf0654892a3ce5d425729d1c8778fc13f23b91ec7f972c8b8b46fb",
 "scorer":"9d0b56bd68a75e939d6cd0cc900de82d905d4d7131b56261d17759e83a28ba2f",
 "executor":"a10ec267bce00cc82343df4ee2e70045c3525c466ae8076136fd2ab9f1ef7deb",
 "parent":"0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
 "v15":"7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
 "v16":"5b1cb38cc62b5682c03505a5090e962d4773015efb44f8fad145515f5066b549",
}
FILES = {"prior":PRIOR,"admission":ADMISSION,"dependency":DEPENDENCY,"oracles":ORACLES,
 "contract":ROOT/"ops/head_response_mediation_contract.py","scorer":ROOT/"ops/head_response_mediation_scorer.py",
 "executor":ROOT/"ops/attention15_head_module_mediation_executor.py","parent":ROOT/"ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py",
 "v15":ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
 "v16":ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16.py"}
EXPERTS={"A1_oracle":"A1","A2_oracle":"A2"}; HEADS={"module":tuple(range(9)),**{f"L15H{h}":(h,) for h in range(9)}}
PRICE_MAX={"native_capture_forwards":20,"differentiable_transformer_forwards":120,"transformer_backward_forwards":0,"model_updates":0,"example_evaluations":12000,"fit_parameters":0}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(x):
 if isinstance(x,dict): return all(finite(v) for v in x.values())
 if isinstance(x,(list,tuple)): return all(finite(v) for v in x)
 return not isinstance(x,float) or math.isfinite(x)
def margin(context,out,panel):
 l=out["logits"]; m=l[context["index"],context["answer"]]-l[context["index"],context["foil"]]
 return (m[context["panel_indices"][panel]]-context["base_margin"][context["panel_indices"][panel]]).detach().cpu().tolist()
def maxdiff(torch,a,b): return float((a-b).abs().max())
def pair_control(backend,context,changed,background,panel):
 t,F=backend.torch,backend.F; ix=context["panel_indices"][panel]
 lp=F.log_softmax(changed["logits"][ix],-1); lq=F.log_softmax(background["logits"][ix],-1)
 kl=(lq.exp()*(lq-lp)).sum(-1); flips=changed["logits"][ix].argmax(-1)!=background["logits"][ix].argmax(-1)
 return {"mean_kl":float(kl.mean()),"top1_flip_count":int(flips.sum())}

def main():
 observed={k:sha(p) for k,p in FILES.items()}; prior=json.loads(PRIOR.read_text()); admission=json.loads(ADMISSION.read_text()); oracles=json.loads(ORACLES.read_text())
 authority=observed==EXPECTED and prior.get("candidate_id")==CANDIDATE_ID and admission.get("admitted") is True
 dry={"candidate_id":CANDIDATE_ID,"dryrun":True,"authority_ok":authority,"v15_mediators":list(HEADS),"v16_mediators":["L15H5"],"price_max":PRICE_MAX,"expected_differentiable_forwards":120}
 if not authority: raise RuntimeError(f"authority changed: {observed}")
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(dry,sort_keys=True)); return
 if OUT.exists(): raise FileExistsError(OUT)
 start=time.perf_counter(); backend=producer.Bilin18TorchBackend.load("cuda")
 for p in backend.model.parameters(): p.requires_grad_(False)
 counters={k:0 for k in PRICE_MAX}; native=backend.native
 def counted(batch,*,capture): counters["native_capture_forwards"]+=1; counters["example_evaluations"]+=len(batch.row_ids); return native(batch,capture=capture)
 backend.native=counted; versions={"v15":v15.build_rows(),"v16":[r for r in v16.build_rows() if r["transform_id"] in ("A1","A2","P")]}; results={}; replay=0.0; closure=0.0
 for version,rows in versions.items():
  bank=parent.capture_bank(backend,rows,counters,factors=False); contexts={p:parent.attach_references(backend,parent.subset_context(bank,parent.row_indices(rows,parity=p)),counters) for p in (0,1)}; results[version]={}
  for expert,panel in EXPERTS.items():
   bases=dependency.bases_for_evaluation(backend.torch,backend.device,oracles,expert); results[version][expert]={}
   for parity,context in contexts.items():
    off,cap0=executor.capture_live_response(parent,backend,context,counters,{}) ; on,cap1=executor.capture_live_response(parent,backend,context,counters,bases[parity]); caps={"00":cap0,"11":cap1,"00_output":off,"11_output":on}; med_names=HEADS if version=="v15" else {"L15H5":HEADS["L15H5"]}; reports={}
    for name,heads in med_names.items():
     cells=executor.execute_mediator_cells(parent,backend,context,counters,bases[parity],caps,heads); panels=[x for x in ("A1","A2","P","C") if len(context["panel_indices"][x])]; item={}
     for q in panels:
      vectors={c:margin(context,o,q) for c,o in cells.items()}; target=[a-b for a,b in zip(vectors["11"],vectors["00"])]; item[q]=scorer.score_cells(vectors,target)
     item["controls"]={q:{"rescue":pair_control(backend,context,cells["01"],cells["00"],q),"reset":pair_control(backend,context,cells["10"],cells["11"],q)} for q in ("P","C") if len(context["panel_indices"][q])}; reports[name]=item
     if name=="module":
      tensors=contract.build_absolute_set_cells(backend.torch,cap0,cap1,heads=heads,semantic_positions=context["base_batch"].semantic_positions); s0=executor.execute_absolute_response(parent,backend,context,counters,{},tensors["00"]); s1=executor.execute_absolute_response(parent,backend,context,counters,bases[parity],tensors["11"]); replay=max(replay,maxdiff(backend.torch,s0["logits"],off["logits"]),maxdiff(backend.torch,s1["logits"],on["logits"]))
    if version=="v15":
     own={h:reports[h][panel] for h in HEADS if h!="module"}; reports["rank_reset"]=scorer.deterministic_head_ranking(own,"head_reset_loss"); reports["rank_rescue"]=scorer.deterministic_head_ranking(own,"head_rescue"); reports["composition"]=scorer.singleton_module_composition(own,reports["module"][panel]); closure=max(closure,*(reports[h][q]["closure_max_abs_error"] for h in HEADS for q in ("A1","A2","P","C")))
    results[version][expert][str(parity)]=reports
 counters["fit_parameters"]=0
 own=lambda v,e,p:results[v][e][str(p)]["L15H5"][EXPERTS[e]]
 module=lambda e,p:results["v15"][e][str(p)]["module"][EXPERTS[e]]
 A=authority and replay<=1e-4 and closure<=1e-12 and finite(results) and all(counters[k]<=PRICE_MAX[k] for k in PRICE_MAX)
 B=all(module(e,p)["metrics"][x]["signed_projection"]>=.10 and module(e,p)["metrics"][x]["direction_fraction"]>=.75 for e in EXPERTS for p in (0,1) for x in ("head_reset_loss","head_rescue"))
 C=all(own("v15",e,p)["metrics"]["head_reset_loss"]["signed_projection"]>0 and own("v15",e,p)["metrics"]["head_reset_loss"]["direction_fraction"]>=.75 and results["v15"][e][str(p)]["rank_reset"].index("L15H5")<2 and results["v15"][e][str(p)]["rank_rescue"].index("L15H5")<2 for e in EXPERTS for p in (0,1))
 D=all(results["v15"][e][str(p)]["composition"]["metrics"]["cosine"]>=.90 and results["v15"][e][str(p)]["composition"]["metrics"]["relative_l2_error"]<=.25 for e in EXPERTS for p in (0,1))
 E=all(results["v15"][e][str(p)]["L15H5"]["controls"][q][arm]["top1_flip_count"]==0 and results["v15"][e][str(p)]["L15H5"]["controls"][q][arm]["mean_kl"]<=.02 for e in EXPERTS for p in (0,1) for q in ("P","C") for arm in ("rescue","reset"))
 F=all(own("v16",e,p)["metrics"]["head_reset_loss"]["signed_projection"]>=.10 and own("v16",e,p)["metrics"]["head_reset_loss"]["direction_fraction"]>=.75 and results["v16"][e][str(p)]["L15H5"]["controls"]["P"]["reset"]["top1_flip_count"]==0 for e in EXPERTS for p in (0,1))
 predictions={"pred_a_admission_authority_capture_replay_closure_finiteness_and_price":A,"pred_b_complete_attention15_module_mediates_live_transfer":B,"pred_c_l15h5_is_stable_dominant_singleton_mediator":C,"pred_d_singleton_reset_losses_approximately_compose":D,"pred_e_l15h5_mediation_is_control_selective":E,"pred_f_l15h5_mediation_transfers_to_v16":F}
 terminal="invalid" if not A else "attention15_bypass" if not B else "weight_reader_rejected" if not C else "interaction_conditioned_reader" if not D else "greedy_head_union_licensed"
 result={"schema":"temporal_iswas_v15_attention15_head_module_mediation_atlas_result_v1","candidate_id":CANDIDATE_ID,"started_utc":datetime.now(timezone.utc).isoformat(),"serial_seconds":time.perf_counter()-start,"authority_sha256":EXPECTED,"reports":results,"instrument":{"self_clamp_replay_max_abs_error":replay,"factorial_closure_max_abs_error":closure},"predictions":predictions,"terminal":terminal,"price":{**counters,"maxima":PRICE_MAX},"v16_scope":"OOD_TEXT_REUSE_NEW_INTERVENTION"}; atomic_create_json(OUT,result); print(json.dumps({"predictions":predictions,"terminal":terminal,"instrument":result["instrument"],"price":result["price"]},sort_keys=True))

if __name__=="__main__": main()
