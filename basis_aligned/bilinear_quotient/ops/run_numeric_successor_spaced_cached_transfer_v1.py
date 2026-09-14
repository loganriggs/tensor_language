#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 3forward288seq; exact bidirectional L8H3+H7 cached term; capable spaced authority;0fits.
"""A exact authority; B bidirectional list transfer; C copy controls; D live/selective."""
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
ROWS=POLY/"NUMERIC_SUCCESSOR_SPACED_OOD_V1_ROWS.json"
CAPABILITY=POLY/"NUMERIC_SUCCESSOR_SPACED_OOD_CAPABILITY_V1_RESULT.json"
PREREG=POLY/"NUMERIC_SUCCESSOR_SPACED_CACHED_TRANSFER_V1_PREREGISTRATION.md"
PRIOR=ROOT/"basis_aligned/bilinear_quotient/circuits/prior_art/numeric_successor_spaced_cached_transfer_v1.json"
BINDING=POLY/"NUMERIC_SUCCESSOR_SPACED_CACHED_TRANSFER_V1_BINDING.json"
OUT=POLY/"NUMERIC_SUCCESSOR_SPACED_CACHED_TRANSFER_V1_RESULT.json"
REQUESTED_DRY=bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
PRICE={"forwards":3,"sequences":288,"interventions":96,"fits":0}
BARS={"minimum_native_accuracy":.85,"minimum_donorward_fraction":.75,
      "minimum_donor_answer_win_fraction":.50,"minimum_mean_donor_margin_effect":0.,
      "minimum_mean_donor_ce_gain":0.,"minimum_control_answer_preservation":.85,
      "maximum_control_absolute_mean_ce_change":.10,
      "maximum_control_median_margin_change_fraction":.25,
      "minimum_control_intervention_norm_fraction":.10,
      "maximum_native_replay_rse":1e-10,"maximum_source_sum_rse":1e-10,
      "maximum_cached_decomposition_rse":1e-10,"maximum_installed_term_error":1e-5}


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def load_bound():
    binding=json.loads(BINDING.read_text())
    paths={"rows":ROWS,"capability":CAPABILITY,"preregistration":PREREG,"prior_art":PRIOR}
    for name,expected in binding["files"].items():
        if digest(paths[name])!=expected: raise RuntimeError(f"bound {name} changed")
    rows=json.loads(ROWS.read_text()); capability=json.loads(CAPABILITY.read_text())
    if canonical(rows["rows"])!=rows["row_manifest_sha256"]: raise RuntimeError("row manifest changed")
    if binding["row_manifest_sha256"]!=rows["row_manifest_sha256"] or binding["price"]!=PRICE:
        raise RuntimeError("binding changed")
    if capability["terminal"]!="capability_pass" or not all(capability["predictions"].values()):
        raise RuntimeError("capability did not pass")
    return rows,capability


def plan():
    rows,_=load_bound()
    return {"schema":"numeric_successor_spaced_cached_transfer_v1_plan","model_loaded":False,
            "gpu_accessed":False,"queue_touched":False,"rows":rows["row_count"],"directions":96,
            "fixed_layer":8,"fixed_heads":[3,7],"factor":"final_source_cached_value_only",
            "bars":BARS,"price":PRICE,"predicates":["pred_a_exact_authority_and_price",
                "pred_b_bidirectional_numbered_list_transfer","pred_c_digit_word_copy_controls_preserved",
                "pred_d_capable_and_live_interventions"]}


def _list_positions(text,ids,enc):
    positions=[]; prefix=""
    for line in text.splitlines(keepends=True):
        label=line.split(".",1)[0]
        if label.isdigit():
            position=len(enc.encode(prefix)); token=enc.encode(line)[0]
            if ids[position]!=token or enc.decode([ids[position]]).strip()!=label: raise RuntimeError("list map changed")
            positions.append(position)
        prefix+=line
    if len(positions)!=3 or enc.encode(prefix)!=ids: raise RuntimeError("list source count changed")
    return positions


def main():
    frozen,capability=load_bound()
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
    for row in frozen["rows"]:
        for side in ("base","donor"):
            item=dict(row[side])
            item["query_position"]=len(item["ids"])-1
            item["source_positions"]=(_list_positions(item["text"],item["ids"],semantic.ENC)
                if row["program_role"].startswith("list_") else
                [x["token_position"] for x in semantic.endpoint_mapping(item["ids"])["value_positions"]])
            endpoint_index[(row["row_id"],side)]=len(endpoints); endpoints.append(item)
    model,checkpoint=exact.r573.facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
    tokens,finals,positions=exact._pad(endpoints,"cuda")
    with torch.inference_mode(): native_full=exact.r573.native_logits(model,tokens)
    native=native_full[torch.arange(len(endpoints),device="cuda"),finals]
    replay,captures,diagnostics=exact._capture_forward(model,tokens,finals,positions)
    examples=[]; replacements=[]; specs=[]
    for row in frozen["rows"]:
        for direction,recipient_side,donor_side in (("base_to_donor","base","donor"),("donor_to_base","donor","base")):
            ri=endpoint_index[(row["row_id"],recipient_side)]; di=endpoint_index[(row["row_id"],donor_side)]
            recipient={name:value[ri] for name,value in captures.items()}
            donor={name:value[di] for name,value in captures.items()}
            examples.append(endpoints[ri]); replacements.append(exact._replace(recipient,donor,"cached"))
            specs.append((row,direction,ri,di))
    patch_tokens,patch_finals,_=exact._pad(examples,"cuda")
    patched,norms,patch_diagnostics,installed_error=exact._patched_forward(
        model,patch_tokens,patch_finals,torch.stack(replacements))
    evidence=[]
    for index,(row,direction,ri,di) in enumerate(specs):
        recipient_answer=int(endpoints[ri]["answer_id"]); donor_answer=int(endpoints[di]["answer_id"])
        before=replay[ri]; after=patched[index]
        donor_before=exact._margin(before,donor_answer,recipient_answer)
        donor_after=exact._margin(after,donor_answer,recipient_answer)
        recipient_before=-donor_before; recipient_after=-donor_after
        evidence.append({"row_id":row["row_id"],"construction":row["construction"],
            "program_role":row["program_role"],"direction":direction,
            "native_recipient_correct":recipient_before>0,"native_donor_correct":
                exact._margin(replay[di],donor_answer,recipient_answer)>0,
            "post_recipient_correct":recipient_after>0,"donor_margin_effect":donor_after-donor_before,
            "donor_answer_win":donor_after>0,"donor_ce_gain":exact._ce(before,donor_answer)-exact._ce(after,donor_answer),
            "recipient_margin_change":recipient_after-recipient_before,
            "recipient_ce_change":exact._ce(after,recipient_answer)-exact._ce(before,recipient_answer),
            "intervention_norm":float(norms[index])})
    grouped=defaultdict(list)
    for row in evidence: grouped[(row["construction"],row["program_role"],row["direction"])].append(row)
    target_rows=[x for x in evidence if x["program_role"]=="list_step_two_spaced"]
    target_scale=statistics.median(abs(x["donor_margin_effect"]) for x in target_rows)
    target_norm=statistics.median(x["intervention_norm"] for x in target_rows)
    reports={}
    positive_pass=True; control_pass=True; live=True
    for key,items in sorted(grouped.items()):
        common={"n":len(items),"recipient_native_accuracy":statistics.fmean(x["native_recipient_correct"] for x in items),
                "donor_native_accuracy":statistics.fmean(x["native_donor_correct"] for x in items),
                "median_intervention_norm_fraction_of_list":statistics.median(x["intervention_norm"] for x in items)/max(target_norm,1e-12)}
        if key[1]=="list_step_two_spaced":
            report={**common,"donorward_fraction":statistics.fmean(x["donor_margin_effect"]>0 for x in items),
                    "donor_answer_win_fraction":statistics.fmean(x["donor_answer_win"] for x in items),
                    "mean_donor_margin_effect":statistics.fmean(x["donor_margin_effect"] for x in items),
                    "mean_donor_ce_gain":statistics.fmean(x["donor_ce_gain"] for x in items)}
            report["passes"]=(min(report["recipient_native_accuracy"],report["donor_native_accuracy"])>=BARS["minimum_native_accuracy"]
                and report["donorward_fraction"]>=BARS["minimum_donorward_fraction"]
                and report["donor_answer_win_fraction"]>=BARS["minimum_donor_answer_win_fraction"]
                and report["mean_donor_margin_effect"]>BARS["minimum_mean_donor_margin_effect"]
                and report["mean_donor_ce_gain"]>BARS["minimum_mean_donor_ce_gain"])
            positive_pass &= report["passes"]
        else:
            report={**common,"recipient_answer_preservation":statistics.fmean(x["post_recipient_correct"] for x in items),
                    "absolute_mean_recipient_ce_change":abs(statistics.fmean(x["recipient_ce_change"] for x in items)),
                    "median_absolute_margin_change_fraction_of_list":statistics.median(abs(x["recipient_margin_change"]) for x in items)/max(target_scale,1e-12)}
            report["passes"]=(min(report["recipient_native_accuracy"],report["donor_native_accuracy"])>=BARS["minimum_native_accuracy"]
                and report["recipient_answer_preservation"]>=BARS["minimum_control_answer_preservation"]
                and report["absolute_mean_recipient_ce_change"]<=BARS["maximum_control_absolute_mean_ce_change"]
                and report["median_absolute_margin_change_fraction_of_list"]<=BARS["maximum_control_median_margin_change_fraction"])
            control_pass &= report["passes"]
            live &= common["median_intervention_norm_fraction_of_list"]>=BARS["minimum_control_intervention_norm_fraction"]
        reports["|".join(key)]=report
    replay_rse=float((native-replay).square().sum())/max(float(native.square().sum()),1e-30)
    source_rse=max(diagnostics["head_source_sum_relative_squared_error"],patch_diagnostics["head_source_sum_relative_squared_error"])
    cached_rse=max(diagnostics["value_split_relative_squared_error"],patch_diagnostics["value_split_relative_squared_error"])
    pred_a=(replay_rse<=BARS["maximum_native_replay_rse"] and source_rse<=BARS["maximum_source_sum_rse"]
            and cached_rse<=BARS["maximum_cached_decomposition_rse"] and installed_error<=BARS["maximum_installed_term_error"]
            and len(evidence)==96 and min(x["intervention_norm"] for x in evidence)>0)
    pred_d=bool(all(x["accuracy"]>=BARS["minimum_native_accuracy"] for x in capability["cell_reports"].values()) and live)
    predictions={"pred_a_exact_authority_and_price":bool(pred_a),
                 "pred_b_bidirectional_numbered_list_transfer":bool(pred_a and positive_pass),
                 "pred_c_digit_word_copy_controls_preserved":bool(pred_a and control_pass),
                 "pred_d_capable_and_live_interventions":bool(pred_a and pred_d)}
    terminal=("invalid" if not predictions["pred_a_exact_authority_and_price"] else
              "selective_bidirectional_cached_transfer" if all(predictions.values()) else
              "shared_numeric_copy_payload" if predictions["pred_b_bidirectional_numbered_list_transfer"] and predictions["pred_d_capable_and_live_interventions"] else
              "bidirectional_transfer_null")
    result={"schema":"numeric_successor_spaced_cached_transfer_v1_result","terminal":terminal,
            "predictions":predictions,"exactness":{"native_replay_relative_squared_error":replay_rse,
            "head_source_sum_relative_squared_error":source_rse,"cached_decomposition_relative_squared_error":cached_rse,
            "installed_term_max_absolute_error":installed_error},"cell_reports":reports,
            "price":{**PRICE,"observed_forwards":3,"observed_sequences":288},
            "row_manifest_sha256":frozen["row_manifest_sha256"],"checkpoint_weights_sha256":checkpoint.weights_sha256,
            "runner_sha256":digest(RUNNER),"evidence":evidence}
    atomic_create_json(OUT,result); print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__": main()
