#!/usr/bin/env python3
"""Select noisy multi-reader H3 DAS on a complete held-out construction family."""

# BQGATE: EXPERIMENT pred_a_authority_closure_finiteness_and_price pred_b_v8_rejects_known_row_holdout_shortcut pred_c_family_selector_never_worsens_pooled_candidate pred_d_nonzero_update_improves_sealed_or_zero_update_is_safe pred_e_selected_beats_dim_every_reader_family
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fit_v1
import circuit_candidate_temporal_auxiliary_fresh_cues_v2 as fit_v2
import circuit_candidate_temporal_auxiliary_fresh_cues_v8 as select_v8
import circuit_candidate_temporal_auxiliary_fresh_cues_v10 as test_v10
import circuit_candidate_temporal_auxiliary_fresh_cues_v11 as test_v11
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_unit_greedy as g
import run_temporal_h3_das_noisy_worst_environment_multi_reader_v1 as fitlib
import run_temporal_auxiliary_will_had_h3_tensor_anchored_regularized_rank7_v2 as evaluator

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_h3_das_complete_family_holdout_v1.json"
POOLED=ROOT/"circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
PREVIOUS=ROOT/"circuits/followups/temporal_h3_das_noisy_worst_environment_multi_reader_v1_result.json"
CAP8=ROOT/"circuits/followups/temporal_auxiliary_will_had_fresh_v8_capability_v1_result.json"
CAP10=ROOT/"circuits/followups/temporal_auxiliary_will_had_fresh_v10_capability_v1_result.json"
CAP11=ROOT/"circuits/followups/temporal_auxiliary_will_had_fresh_v11_capability_v1_result.json"
V1=ROOT/"ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py";V2=ROOT/"ops/circuit_candidate_temporal_auxiliary_fresh_cues_v2.py";V8=ROOT/"ops/circuit_candidate_temporal_auxiliary_fresh_cues_v8.py";V10=ROOT/"ops/circuit_candidate_temporal_auxiliary_fresh_cues_v10.py";V11=ROOT/"ops/circuit_candidate_temporal_auxiliary_fresh_cues_v11.py"
FITLIB=ROOT/"ops/run_temporal_h3_das_noisy_worst_environment_multi_reader_v1.py";EVALUATOR=ROOT/"ops/run_temporal_auxiliary_will_had_h3_tensor_anchored_regularized_rank7_v2.py";UNIT_LIB=ROOT/"ops/circuit_unit_greedy.py"
OUT=ROOT/"circuits/followups/temporal_h3_das_complete_family_holdout_v1_result.json"
CANDIDATE_ID="temporal_auxiliary.will_vs_had.h3_das_complete_family_holdout_v1"
EXPECTED={"prior":"b4162b2934be4105af3099f1af7ea0d783100f0733140e7814a1559c91ccb5ad","pooled":"d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9","previous":"b569c29349ec66d29883235df12366ceeb86aa3fdd06cdd5c45df34096327f4f","cap8":"fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff","cap10":"9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad","cap11":"0330dc5a4f85bc68c4da6f98af2f4208335e65c644ddedd5d8cc487368091026","v1":"5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9","v2":"adbfaf91ed2889cc42da85255edf9f5074f1002e9ad93dc1d4ff706de66d1144","v8":"13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf","v10":"e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494","v11":"f75b17669a5fc5299d21f5b44e91530c03c71d75181683c7b6728cb95c862450","fitlib":"bf806b077da4e2fb43612f8f3b5ca318e0d45edac6b2c26ecc2f6accd218b413","evaluator":"561b40b093e0a46469fa95b01c23e1b0d7d294201aaccb63b918b86900359303","unit_lib":"556cfe8bef376dc57d053bf1eacba9ac81d6c9b05b4405908400af03d233eb74"}
METHODS=("selected","pooled_aligned","dim")
PRICE={"model_forwards":2678,"example_evaluations":46498,"transformer_backward_forwards":2400,"model_updates":150,"fit_parameters":128,"evaluation_records":1890}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def tensor_sha(t):return hashlib.sha256(t.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()
def rows_for(builder,cap):
    allowed={x for ids in cap["jointly_capable_row_ids"].values() for x in ids}
    return [r for r in builder.build_rows() if r["transform_id"] in ("A1","A2") and r["row_id"] in allowed]

def score_evaluations(evaluations):
    cells=[];by_cell={}
    for bank in ("v10","v11"):
        for panel in ("A1","A2"):
            score,pieces=fitlib.cell_score(evaluations[bank],panel);cells.append(score);by_cell[f"{bank}_{panel}"]={"joint":score,**pieces}
    return {"mean":sum(cells)/len(cells),"worst":max(cells),"by_cell":by_cell}

def main():
    paths={"prior":PRIOR,"pooled":POOLED,"previous":PREVIOUS,"cap8":CAP8,"cap10":CAP10,"cap11":CAP11,"v1":V1,"v2":V2,"v8":V8,"v10":V10,"v11":V11,"fitlib":FITLIB,"evaluator":EVALUATOR,"unit_lib":UNIT_LIB}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("complete-family DAS authority changed")
    prior,pooled,previous,cap8,cap10,cap11=[json.loads(p.read_text()) for p in (PRIOR,POOLED,PREVIOUS,CAP8,CAP10,CAP11)]
    groups={(bank,panel):[r for r in builder.build_rows() if r["transform_id"]==panel][0::2] for bank,builder in (("v1",fit_v1),("v2",fit_v2)) for panel in ("A1","A2")}
    rows8,rows10,rows11=rows_for(select_v8,cap8),rows_for(test_v10,cap10),rows_for(test_v11,cap11)
    if prior.get("candidate_id")!=CANDIDATE_ID or pooled.get("terminal")!="task_conditioned" or previous.get("terminal")!="task_memorization" or any(c.get("terminal")!="manifest" for c in (cap8,cap10,cap11)) or [len(rows8),len(rows10),len(rows11)]!=[60,63,63] or any(len(x)!=16 for x in groups.values()):raise RuntimeError("authority terminal or population changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"rank":1,"gradient_environments":[f"{b}_{p}_even" for b,p in groups],"selection_family":{"v8":len(rows8)},"sealed_families":{"v10":len(rows10),"v11":len(rows11)},"methods":list(METHODS),"restarts":list(fitlib.RESTARTS),"steps_per_restart":fitlib.STEPS,"checkpoints":list(fitlib.CHECKPOINTS),"pooled_step_zero_eligible":True,**PRICE}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
    for parameter in backend.model.parameters():parameter.requires_grad_(False)
    anchor=torch.tensor(pooled["axis_artifacts"]["pooled_aligned_rank1"],device=backend.device).float().unsqueeze(1);anchor=anchor/anchor.norm()
    prep=g.prepare(backend,groups[("v1","A1")]);dim=g.diff_in_means_direction(backend,prep,("attn:11:head:03",));dim=dim/dim.norm()
    generator=torch.Generator(device="cpu").manual_seed(907);random=torch.randn((128,1),generator=generator).to(backend.device);random=random/random.norm()
    train=[fitlib.attach_targets(backend,rows) for rows in groups.values()]
    selection=[fitlib.attach_targets(backend,[r for r in rows8 if r["transform_id"]==panel]) for panel in ("A1","A2")]
    closure=max(c["manual_base_margin_max_abs"] for c in train+selection)
    with torch.no_grad():
        baseline_train=fitlib.objective(backend,train,anchor,anchor,grad=False)
        baseline_selection=fitlib.objective(backend,selection,anchor,anchor,grad=False)
    fits=[fitlib.fit_restart(backend,train,selection,initial,anchor,name) for name,initial in zip(fitlib.RESTARTS,(anchor,dim,random))]
    selected=min(fits,key=lambda x:x["best"]["selection"]);q=selected["axis"]
    axes={"selected":q,"pooled_aligned":anchor,"dim":dim};raw_evaluations={};scores={};max_instrument=0.;records=forwards=evaluations=0
    for method,axis in axes.items():
        raw_evaluations[method]={}
        for bank,rows in (("v10",rows10),("v11",rows11)):
            report=evaluator.evaluate_bank(backend,bank,rows,axis,axis@axis.T);records+=len(report["records"]);forwards+=report["forwards"];evaluations+=report["evaluations"];max_instrument=max(max_instrument,*report["instrument"].values());raw_evaluations[method][bank]=report
        scores[method]=score_evaluations(raw_evaluations[method])
    old_v8=[];pooled_v8=[]
    for panel in ("A1","A2"):
        x,_=fitlib.cell_score(previous["evaluations"]["v8"],panel);old_v8.append(x)
        pooled_v8.append(previous["sealed"]["frozen_pooled_aligned"]["by_cell"][f"v8_{panel}"]["joint"])
    selected_step=selected["best"]["step"];moved=selected_step!=0
    families=("behavior_match_squared","behavior_complement_squared","l15_transport_match_squared","full_vocab_match_fraction","full_vocab_complement_fraction")
    family_sums={m:{f:sum(c[f] for c in scores[m]["by_cell"].values()) for f in families} for m in METHODS}
    finite=all(math.isfinite(v) for fit in fits for point in fit["trace"] for v in point.values() if isinstance(v,(int,float)))
    pred_a=closure<=1e-4 and max_instrument<=1e-4 and finite and forwards==156 and evaluations==4914 and records==1890
    pred_b=max(old_v8)>max(pooled_v8)
    pred_c=selected["best"]["selection"]<=float(baseline_selection[1].max())+1e-7
    sealed_improves=scores["selected"]["mean"]<scores["pooled_aligned"]["mean"] and scores["selected"]["worst"]<scores["pooled_aligned"]["worst"]
    sealed_safe=scores["selected"]["mean"]<=scores["pooled_aligned"]["mean"]+1e-7 and scores["selected"]["worst"]<=scores["pooled_aligned"]["worst"]+1e-7
    pred_d=sealed_improves if moved else sealed_safe
    pred_e=all(family_sums["selected"][f]<family_sums["dim"][f] for f in families)
    predictions={"pred_a_authority_closure_finiteness_and_price":bool(pred_a),"pred_b_v8_rejects_known_row_holdout_shortcut":bool(pred_b),"pred_c_family_selector_never_worsens_pooled_candidate":bool(pred_c),"pred_d_nonzero_update_improves_sealed_or_zero_update_is_safe":bool(pred_d),"pred_e_selected_beats_dim_every_reader_family":bool(pred_e)}
    terminal="invalid" if not pred_a else "family_selector_rejects_update" if not moved and pred_d else "cross_family_candidate" if moved and sealed_improves and pred_e else "residual_family_memorization" if moved and not sealed_safe else "partial_cross_family"
    result={"schema":"temporal_h3_das_complete_family_holdout_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"training_and_selection_manual_margin_max_abs":closure,"sealed_max_closure_or_reconstruction_abs":max_instrument},"selection":{"pooled_baseline":{"joint":float(baseline_selection[0]),"worst":float(baseline_selection[1].max())},"known_shortcut_v8":{"mean":sum(old_v8)/2,"worst":max(old_v8)},"pooled_v8_tournament":{"mean":sum(pooled_v8)/2,"worst":max(pooled_v8)},"selected_restart":selected["name"],"selected_step":selected_step,"moved_from_pooled":moved,"selected_axis_sha256":tensor_sha(q),"selected_axis":q.flatten().cpu().tolist(),"fits":[{k:v for k,v in fit.items() if k!="axis"} for fit in fits]},"train":{"pooled_baseline":{"joint":float(baseline_train[0]),"worst":float(baseline_train[1].max())}},"sealed":{"scores":scores,"family_sums":family_sums},"evaluations":{m:{b:{k:v for k,v in report.items() if k not in ("records","forwards","evaluations")} for b,report in banks.items()} for m,banks in raw_evaluations.items()},"predictions":predictions,"terminal":terminal,"price":PRICE}
    atomic_create_json(OUT,result);print(json.dumps({"candidate_id":CANDIDATE_ID,"instrument":result["instrument"],"selection":{k:v for k,v in result["selection"].items() if k not in ("selected_axis","fits")},"sealed":result["sealed"],"predictions":predictions,"terminal":terminal,"price":PRICE},sort_keys=True))

if __name__=="__main__":main()
