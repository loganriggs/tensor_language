#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 4forwards576seq; immutable bracket vector and scalar transfer;0fits.
"""A instrument; B vector transfer; C pair recurrence; D controls; E scalar law."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys


RUNNER = Path(__file__).resolve()
OPS = RUNNER.parent
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
ROWS = POLY / "BRACKET_NESTED_PENDING_OOD_V1_ROWS.json"
CAPABILITY = POLY / "BRACKET_NESTED_PENDING_OOD_CAPABILITY_V1_RESULT.json"
PREREG = POLY / "BRACKET_NESTED_PENDING_PROGRAM_TRANSFER_V1_PREREGISTRATION.md"
BINDING = POLY / "BRACKET_NESTED_PENDING_PROGRAM_TRANSFER_V1_BINDING.json"
ARTIFACT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/bracket_l13h8_ordered_pair_displacement_artifact_v1.json"
SCALAR = ROOT / "basis_aligned/bilinear_quotient/circuits/followups/bracket_ordered_pair_suffix_free_scalar_feasibility_v1_result.json"
OUT = POLY / "BRACKET_NESTED_PENDING_PROGRAM_TRANSFER_V1_RESULT.json"
REQUESTED_DRY = bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
PRICE = {"forwards": 4, "sequences": 576, "fits": 0}
BARS = {
    "maximum_replay_logit_error": 1e-5,
    "minimum_overall_cosine": 0.80,
    "maximum_overall_relative_l2": 0.60,
    "minimum_overall_sign_agreement": 0.90,
    "minimum_norm_ratio": 0.50,
    "maximum_norm_ratio": 1.50,
    "minimum_pair_cosine": 0.70,
    "maximum_pair_relative_l2": 0.60,
    "minimum_pair_sign_agreement": 0.90,
    "maximum_control_logit_change": 1e-5,
    "minimum_pair_exact_positive_fraction": 0.90,
}


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"rows": ROWS, "capability": CAPABILITY, "preregistration": PREREG,
             "artifact": ARTIFACT, "scalar": SCALAR}
    for name, expected in binding["files"].items():
        if digest(paths[name]) != expected: raise RuntimeError(f"bound {name} changed")
    rows = json.loads(ROWS.read_text())
    capability = json.loads(CAPABILITY.read_text())
    if capability["terminal"] != "capability_pass" or not all(capability["predictions"].values()):
        raise RuntimeError("nested capability is not open")
    if binding["row_manifest_sha256"] != rows["row_manifest_sha256"] or binding["price"] != PRICE or binding["bars"] != BARS:
        raise RuntimeError("binding changed")
    artifact, scalar = json.loads(ARTIFACT.read_text()), json.loads(SCALAR.read_text())
    if artifact["terminal"] != "prototype_artifact" or not all(artifact["predictions"].values()): raise RuntimeError("program invalid")
    if scalar["terminal"] != "feasibility_screen" or not all(scalar["score"]["predictions"].values()): raise RuntimeError("scalar audit invalid")
    tables = list(scalar["score"]["cross_family_scalar_tables"].values())
    scalars = {pair: sum(table[pair] for table in tables) / len(tables) for pair in tables[0]}
    if scalars != binding["ordered_pair_scalars"]: raise RuntimeError("frozen scalar table changed")
    return rows, artifact, scalars


def plan():
    rows, _, scalars = load_bound()
    return {"schema":"bracket_nested_pending_program_transfer_v1_plan","model_loaded":False,"gpu_accessed":False,"queue_touched":False,
            "rows":rows["row_count"],"endpoints":rows["endpoint_count"],"ordered_pair_scalars":scalars,"bars":BARS,"price":PRICE,
            "predicates":["pred_a_exact_instrument","pred_b_immutable_vector_program_transfers","pred_c_each_ordered_pair_recurs","pred_d_zero_dispatch_controls","pred_e_six_scalar_effect_law_transfers"]}


def metrics(actual, predicted):
    dot=sum(a*b for a,b in zip(actual,predicted)); an=math.sqrt(sum(a*a for a in actual)); pn=math.sqrt(sum(p*p for p in predicted))
    return {"count":len(actual),"cosine":dot/max(an*pn,1e-30),"relative_l2_error":math.sqrt(sum((a-p)**2 for a,p in zip(actual,predicted)))/max(an,1e-30),
            "sign_agreement":sum((a>0)==(p>0) for a,p in zip(actual,predicted))/len(actual),"predicted_to_actual_norm_ratio":pn/max(an,1e-30)}


def passes(value, pair=False):
    return (value["cosine"] >= BARS["minimum_pair_cosine" if pair else "minimum_overall_cosine"]
            and value["relative_l2_error"] <= BARS["maximum_pair_relative_l2" if pair else "maximum_overall_relative_l2"]
            and value["sign_agreement"] >= BARS["minimum_pair_sign_agreement" if pair else "minimum_overall_sign_agreement"]
            and (pair or BARS["minimum_norm_ratio"] <= value["predicted_to_actual_norm_ratio"] <= BARS["maximum_norm_ratio"]))


def main():
    frozen, artifact, scalars = load_bound()
    if REQUESTED_DRY: print(json.dumps(plan(),indent=2,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    sys.path.insert(0,str(OPS))
    import torch
    import run_bracket_l13h8_source_region_payload_factorial as exact
    from circuit_fast_screen_managed_runner import atomic_create_json
    torch.set_num_threads(2)
    torch_module,F,facade=exact._dependencies()
    model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
    rows=frozen["rows"]; endpoints=[(row,side) for row in rows for side in ("base","donor")]
    length=max(len(row[f"{side}_ids"]) for row,side in endpoints)
    tokens=torch.full((len(endpoints),length),50256,dtype=torch.long,device="cuda"); finals=[]; sources=[]
    for i,(row,side) in enumerate(endpoints):
        ids=row[f"{side}_ids"];tokens[i,:len(ids)]=torch.tensor(ids,device="cuda");finals.append(len(ids)-1);sources.append(row[f"{side}_open_position"])
    finals_t=torch.tensor(finals,device="cuda");sources_t=torch.tensor(sources,device="cuda");ar=torch.arange(len(endpoints),device="cuda")
    with torch.inference_mode():
        direct=exact.native_logits(model,tokens,torch_module,F)
        replay,factors=exact.factor_forward(model,tokens,finals_t,{},torch_module,F,facade)
        terms=factors["p"][ar,sources_t].unsqueeze(-1)*factors["u"][ar,sources_t]
        donor_terms=terms[ar^1]
        vectors={key:torch.tensor(value["coordinates"],dtype=torch.float32,device="cuda") for key,value in artifact["prototypes"].items()}
        program_terms=[]
        for i,(row,side) in enumerate(endpoints):
            if row["program_role"]=="target":
                other="donor" if side=="base" else "base"; key=f"{int(row[f'{side}_answer_id'])}->{int(row[f'{other}_answer_id'])}"
                program_terms.append(terms[i]+vectors[key])
            else: program_terms.append(terms[i])
        exact_logits=exact.factor_forward(model,tokens,finals_t,{},torch_module,F,facade,replacement_terms=donor_terms,source_positions=sources_t)[0]
        program_logits=exact.factor_forward(model,tokens,finals_t,{},torch_module,F,facade,replacement_terms=torch.stack(program_terms),source_positions=sources_t)[0]
    replay_error=max(float((direct[i,finals[i]]-replay[i,finals[i]]).abs().max()) for i in range(len(endpoints)))
    records=[]
    for i,(row,side) in enumerate(endpoints):
        recipient=int(row[f"{side}_answer_id"]);other="donor" if side=="base" else "base";donor=int(row[f"{other}_answer_id"]);pair=f"{recipient}->{donor}"
        record={"row_id":row["row_id"],"side":side,"program_role":row["program_role"],"ordered_pair":pair,
                "control_max_logit_change":float((program_logits[i,finals[i]]-replay[i,finals[i]]).abs().max()),
                "native_closer_margin":float(exact.closer_margin(replay[i,finals[i]],recipient))}
        if row["program_role"]=="target":
            direction="base_to_donor" if side=="base" else "donor_to_base"
            record["exact_effect"]=float(exact.endpoint_change(replay[i,finals[i]],exact_logits[i,finals[i]],row,direction))
            record["program_effect"]=float(exact.endpoint_change(replay[i,finals[i]],program_logits[i,finals[i]],row,direction))
            record["scalar_prediction"]=float(scalars[pair])
        records.append(record)
    targets=[r for r in records if r["program_role"]=="target"];controls=[r for r in records if r["program_role"]=="control"]
    overall=metrics([r["exact_effect"] for r in targets],[r["program_effect"] for r in targets])
    scalar_overall=metrics([r["exact_effect"] for r in targets],[r["scalar_prediction"] for r in targets])
    by_pair={pair:metrics([r["exact_effect"] for r in targets if r["ordered_pair"]==pair],[r["program_effect"] for r in targets if r["ordered_pair"]==pair]) for pair in sorted(scalars)}
    positive={pair:sum(r["exact_effect"]>0 for r in targets if r["ordered_pair"]==pair)/12 for pair in sorted(scalars)}
    pred_a=bool(replay_error<=BARS["maximum_replay_logit_error"] and len(records)==144 and len(targets)==len(controls)==72)
    pred_b=bool(pred_a and passes(overall)); pred_c=bool(pred_a and all(passes(v,True) for v in by_pair.values()))
    control_max=max(r["control_max_logit_change"] for r in controls)
    pred_d=bool(pred_a and control_max<=BARS["maximum_control_logit_change"] and all(r["native_closer_margin"]>0 for r in controls))
    pred_e=bool(pred_a and passes(scalar_overall) and all(v>=BARS["minimum_pair_exact_positive_fraction"] for v in positive.values()))
    predictions={"pred_a_exact_instrument":pred_a,"pred_b_immutable_vector_program_transfers":pred_b,"pred_c_each_ordered_pair_recurs":pred_c,"pred_d_zero_dispatch_controls":pred_d,"pred_e_six_scalar_effect_law_transfers":pred_e}
    result={"schema":"bracket_nested_pending_program_transfer_v1_result","terminal":"program_and_scalar_transfer" if all(predictions.values()) else "null" if pred_a and pred_d else "invalid",
            "predictions":predictions,"overall_vector_program":overall,"by_ordered_pair":by_pair,"scalar_overall":scalar_overall,"exact_positive_fraction_by_pair":positive,
            "instrument":{"native_factor_replay_max_logit_error":replay_error,"control_max_logit_change":control_max},"price":{**PRICE,"observed_forwards":4,"observed_sequences":576},
            "row_manifest_sha256":frozen["row_manifest_sha256"],"checkpoint_weights_sha256":checkpoint.weights_sha256,"runner_sha256":digest(RUNNER)}
    atomic_create_json(OUT,result);print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__": main()
