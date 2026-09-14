#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 3forward640seq; exact H3+H7 score,cached,joint interaction on capable +1/copy;0fits.
"""A exactness; B cached transfer; C multiplicative interaction; D selective arm."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys


RUNNER=Path(__file__).resolve(); OPS=RUNNER.parent; ROOT=RUNNER.parents[3]
POLY=ROOT/"basis_aligned/polynomial_causal"
TARGET_ROWS=POLY/"NUMERIC_SEQUENCE_NONADJACENT_OOD_V1_ROWS.json"
TARGET_CAP=POLY/"NUMERIC_SEQUENCE_NONADJACENT_OOD_CAPABILITY_V1_RESULT.json"
CONTROL_ROWS=POLY/"NUMERIC_SUCCESSOR_SPACED_OOD_V1_ROWS.json"
CONTROL_CAP=POLY/"NUMERIC_SUCCESSOR_SPACED_OOD_CAPABILITY_V1_RESULT.json"
PREREG=POLY/"NUMERIC_SEQUENCE_PAYLOAD_ROUTER_INTERACTION_V1_PREREGISTRATION.md"
PRIOR=ROOT/"basis_aligned/bilinear_quotient/circuits/prior_art/numeric_sequence_payload_router_interaction_v1.json"
BINDING=POLY/"NUMERIC_SEQUENCE_PAYLOAD_ROUTER_INTERACTION_V1_BINDING.json"
OUT=POLY/"NUMERIC_SEQUENCE_PAYLOAD_ROUTER_INTERACTION_V1_RESULT.json"
REQUESTED_DRY=bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
ARMS=("score","cached","joint")
PRICE={"forwards":3,"sequences":640,"interventions":384,"fits":0}
BARS={"minimum_native_accuracy":.85,"minimum_donorward_fraction":.75,
      "minimum_donor_answer_win_fraction":.50,"minimum_mean_donor_margin_effect":0.,
      "minimum_mean_donor_ce_gain":0.,"minimum_joint_over_best_single":1.10,
      "minimum_control_answer_preservation":.85,"maximum_control_absolute_mean_ce_change":.10,
      "maximum_control_median_margin_change_fraction":.25,"minimum_control_norm_fraction":.10,
      "maximum_native_replay_rse":1e-10,"maximum_source_sum_rse":1e-10,
      "maximum_cached_decomposition_rse":1e-10,"maximum_installed_term_error":1e-5}


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def load_bound():
    binding=json.loads(BINDING.read_text())
    paths={"target_rows":TARGET_ROWS,"target_capability":TARGET_CAP,"control_rows":CONTROL_ROWS,
           "control_capability":CONTROL_CAP,"preregistration":PREREG,"prior_art":PRIOR}
    for name,expected in binding["files"].items():
        if digest(paths[name])!=expected: raise RuntimeError(f"bound {name} changed")
    target=json.loads(TARGET_ROWS.read_text()); control=json.loads(CONTROL_ROWS.read_text())
    if canonical(target["rows"])!=target["row_manifest_sha256"] or canonical(control["rows"])!=control["row_manifest_sha256"]:
        raise RuntimeError("row manifest changed")
    if binding["target_manifest_sha256"]!=target["row_manifest_sha256"] or binding["control_manifest_sha256"]!=control["row_manifest_sha256"] or binding["price"]!=PRICE:
        raise RuntimeError("binding changed")
    for path in (TARGET_CAP,CONTROL_CAP):
        result=json.loads(path.read_text())
        if result["terminal"]!="capability_pass" or not all(result["predictions"].values()): raise RuntimeError("capability changed")
    controls=[row for row in control["rows"] if row["program_role"] in ("digit_copy_spaced_control","word_copy_spaced_control")]
    if len(target["rows"])!=32 or len(controls)!=32: raise RuntimeError("row coverage changed")
    rows=[{"role":"target","construction":row["construction"],"representation":row["representation"],**row} for row in target["rows"]]
    rows += [{"role":"control","construction":row["construction"],
              "representation":"digit" if row["program_role"].startswith("digit_") else "number_word",**row} for row in controls]
    return rows,target["row_manifest_sha256"],control["row_manifest_sha256"]


def plan():
    rows,_,_=load_bound()
    return {"schema":"numeric_sequence_payload_router_interaction_v1_plan","model_loaded":False,
            "gpu_accessed":False,"queue_touched":False,"row_pairs":len(rows),"unique_endpoints":2*len(rows),
            "fixed_layer":8,"fixed_heads":[3,7],"arms":list(ARMS),"bars":BARS,"price":PRICE,
            "predicates":["pred_a_exact_authority_and_price","pred_b_cached_payload_transfers_plus_one",
                          "pred_c_joint_score_cached_interaction","pred_d_one_target_arm_is_copy_selective"]}


def main():
    rows,target_manifest,control_manifest=load_bound()
    if REQUESTED_DRY:
        print(json.dumps(plan(),indent=2,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    sys.path.insert(0,str(OPS))
    import torch
    import run_attn8_h3_h7_cross_behavior_factor_interchange_v2 as exact
    import numeric_sequence_semantic_positions_rung577 as semantic
    from circuit_fast_screen_managed_runner import atomic_create_json
    torch.set_num_threads(2)
    endpoints=[]; endpoint_index={}
    for row in rows:
        for side in ("base","donor"):
            item=dict(row[side]); mapping=semantic.endpoint_mapping(item["ids"])
            item["query_position"]=mapping["query_position"]
            item["source_positions"]=[x["token_position"] for x in mapping["value_positions"]]
            endpoint_index[(row["row_id"],side)]=len(endpoints); endpoints.append(item)
    model,checkpoint=exact.r573.facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
    tokens,finals,positions=exact._pad(endpoints,"cuda")
    with torch.inference_mode(): native_full=exact.r573.native_logits(model,tokens)
    native=native_full[torch.arange(len(endpoints),device="cuda"),finals]
    replay,captures,diagnostics=exact._capture_forward(model,tokens,finals,positions)
    examples=[]; replacements=[]; specs=[]
    for row in rows:
        for direction,recipient_side,donor_side in (("base_to_donor","base","donor"),("donor_to_base","donor","base")):
            ri=endpoint_index[(row["row_id"],recipient_side)]; di=endpoint_index[(row["row_id"],donor_side)]
            recipient={name:value[ri] for name,value in captures.items()}; donor={name:value[di] for name,value in captures.items()}
            for arm in ARMS:
                examples.append(endpoints[ri]); replacements.append(exact._replace(recipient,donor,arm)); specs.append((row,direction,arm,ri,di))
    patch_tokens,patch_finals,_=exact._pad(examples,"cuda")
    patched,norms,patch_diagnostics,installed_error=exact._patched_forward(model,patch_tokens,patch_finals,torch.stack(replacements))
    evidence=[]
    for index,(row,direction,arm,ri,di) in enumerate(specs):
        recipient_answer=int(endpoints[ri]["answer_id"]); donor_answer=int(endpoints[di]["answer_id"])
        before=replay[ri]; after=patched[index]
        donor_before=exact._margin(before,donor_answer,recipient_answer); donor_after=exact._margin(after,donor_answer,recipient_answer)
        recipient_before=-donor_before; recipient_after=-donor_after
        evidence.append({"row_id":row["row_id"],"role":row["role"],"construction":row["construction"],
            "representation":row["representation"],"direction":direction,"arm":arm,
            "native_recipient_correct":recipient_before>0,"native_donor_correct":exact._margin(replay[di],donor_answer,recipient_answer)>0,
            "post_recipient_correct":recipient_after>0,"donor_margin_effect":donor_after-donor_before,
            "donor_answer_win":donor_after>0,"donor_ce_gain":exact._ce(before,donor_answer)-exact._ce(after,donor_answer),
            "recipient_margin_change":recipient_after-recipient_before,
            "recipient_ce_change":exact._ce(after,recipient_answer)-exact._ce(before,recipient_answer),
            "intervention_norm":float(norms[index])})
    grouped=defaultdict(list)
    for row in evidence: grouped[(row["role"],row["construction"],row["representation"],row["direction"],row["arm"])].append(row)
    reports={}; target_cells={arm:[] for arm in ARMS}; control_cells={arm:[] for arm in ARMS}
    for key,items in sorted(grouped.items()):
        report={"n":len(items),"recipient_native_accuracy":statistics.fmean(x["native_recipient_correct"] for x in items),
                "donor_native_accuracy":statistics.fmean(x["native_donor_correct"] for x in items),
                "median_intervention_norm":statistics.median(x["intervention_norm"] for x in items)}
        if key[0]=="target":
            report.update({"donorward_fraction":statistics.fmean(x["donor_margin_effect"]>0 for x in items),
                "donor_answer_win_fraction":statistics.fmean(x["donor_answer_win"] for x in items),
                "mean_donor_margin_effect":statistics.fmean(x["donor_margin_effect"] for x in items),
                "mean_donor_ce_gain":statistics.fmean(x["donor_ce_gain"] for x in items)})
            report["passes_transfer"]=(min(report["recipient_native_accuracy"],report["donor_native_accuracy"])>=BARS["minimum_native_accuracy"]
                and report["donorward_fraction"]>=BARS["minimum_donorward_fraction"]
                and report["donor_answer_win_fraction"]>=BARS["minimum_donor_answer_win_fraction"]
                and report["mean_donor_margin_effect"]>BARS["minimum_mean_donor_margin_effect"]
                and report["mean_donor_ce_gain"]>BARS["minimum_mean_donor_ce_gain"])
            target_cells[key[-1]].append((key,report))
        else: control_cells[key[-1]].append((key,report,items))
        reports["|".join(key)]=report
    target_pass={arm:len(target_cells[arm])==8 and all(x[1]["passes_transfer"] for x in target_cells[arm]) for arm in ARMS}
    target_scale={arm:statistics.median(abs(x["donor_margin_effect"]) for x in evidence if x["role"]=="target" and x["arm"]==arm) for arm in ARMS}
    target_norm={arm:statistics.median(x["intervention_norm"] for x in evidence if x["role"]=="target" and x["arm"]==arm) for arm in ARMS}
    control_pass={}
    for arm in ARMS:
        passes=[]
        for key,report,items in control_cells[arm]:
            report.update({"recipient_answer_preservation":statistics.fmean(x["post_recipient_correct"] for x in items),
                "absolute_mean_recipient_ce_change":abs(statistics.fmean(x["recipient_ce_change"] for x in items)),
                "median_absolute_margin_change_fraction":statistics.median(abs(x["recipient_margin_change"]) for x in items)/max(target_scale[arm],1e-12),
                "median_intervention_norm_fraction":report["median_intervention_norm"]/max(target_norm[arm],1e-12)})
            report["passes_control"]=(min(report["recipient_native_accuracy"],report["donor_native_accuracy"])>=BARS["minimum_native_accuracy"]
                and report["recipient_answer_preservation"]>=BARS["minimum_control_answer_preservation"]
                and report["absolute_mean_recipient_ce_change"]<=BARS["maximum_control_absolute_mean_ce_change"]
                and report["median_absolute_margin_change_fraction"]<=BARS["maximum_control_median_margin_change_fraction"]
                and report["median_intervention_norm_fraction"]>=BARS["minimum_control_norm_fraction"])
            reports["|".join(key)]=report; passes.append(report["passes_control"])
        control_pass[arm]=len(passes)==8 and all(passes)
    by_cell={(key[1],key[2],key[3],key[4]):report for key,report in sum(target_cells.values(),[])}
    synergy=[]
    for construction in ("fresh_series","fresh_instruction"):
        for representation in ("digit","number_word"):
            for direction in ("base_to_donor","donor_to_base"):
                joint=by_cell[(construction,representation,direction,"joint")]["mean_donor_margin_effect"]
                single=max(by_cell[(construction,representation,direction,"score")]["mean_donor_margin_effect"],
                           by_cell[(construction,representation,direction,"cached")]["mean_donor_margin_effect"])
                synergy.append(joint/max(single,1e-12))
    replay_rse=float((native-replay).square().sum())/max(float(native.square().sum()),1e-30)
    source_rse=max(diagnostics["head_source_sum_relative_squared_error"],patch_diagnostics["head_source_sum_relative_squared_error"])
    cached_rse=max(diagnostics["value_split_relative_squared_error"],patch_diagnostics["value_split_relative_squared_error"])
    pred_a=(replay_rse<=BARS["maximum_native_replay_rse"] and source_rse<=BARS["maximum_source_sum_rse"]
        and cached_rse<=BARS["maximum_cached_decomposition_rse"] and installed_error<=BARS["maximum_installed_term_error"]
        and len(endpoints)==128 and len(evidence)==384 and min(x["intervention_norm"] for x in evidence)>0)
    pred_b=bool(pred_a and target_pass["cached"])
    pred_c=bool(pred_a and target_pass["joint"] and min(synergy)>=BARS["minimum_joint_over_best_single"])
    selective=[arm for arm in ARMS if target_pass[arm] and control_pass[arm]]
    pred_d=bool(pred_a and selective)
    predictions={"pred_a_exact_authority_and_price":bool(pred_a),"pred_b_cached_payload_transfers_plus_one":pred_b,
                 "pred_c_joint_score_cached_interaction":pred_c,"pred_d_one_target_arm_is_copy_selective":pred_d}
    terminal=("invalid" if not pred_a else "selective_cached_relation_carrier" if pred_b and "cached" in selective else
              "selective_payload_router_interaction" if pred_c and "joint" in selective else
              "selective_score_router" if "score" in selective else
              "shared_final_source_factor_without_relation_selectivity" if any(target_pass.values()) else
              "relation_state_outside_final_source_factors")
    result={"schema":"numeric_sequence_payload_router_interaction_v1_result","terminal":terminal,
            "predictions":predictions,"target_arm_pass":target_pass,"control_arm_pass":control_pass,
            "selective_arms":selective,"joint_over_best_single_range":[min(synergy),max(synergy)],
            "exactness":{"native_replay_relative_squared_error":replay_rse,"head_source_sum_relative_squared_error":source_rse,
                         "cached_decomposition_relative_squared_error":cached_rse,"installed_term_max_absolute_error":installed_error},
            "cell_reports":reports,"price":{**PRICE,"observed_forwards":3,"observed_sequences":640},
            "target_manifest_sha256":target_manifest,"control_manifest_sha256":control_manifest,
            "checkpoint_weights_sha256":checkpoint.weights_sha256,"runner_sha256":digest(RUNNER),"evidence":evidence}
    atomic_create_json(OUT,result); print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
