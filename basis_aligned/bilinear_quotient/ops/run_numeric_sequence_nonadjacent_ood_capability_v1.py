#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 1forward64seq; fresh nonadjacent digit/word +1 capability only;0fits.
"""A frozen authority; B digit capability; C word capability; D exact price."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys


RUNNER=Path(__file__).resolve(); OPS=RUNNER.parent; ROOT=RUNNER.parents[3]
POLY=ROOT/"basis_aligned/polynomial_causal"
ROWS=POLY/"NUMERIC_SEQUENCE_NONADJACENT_OOD_V1_ROWS.json"
PREREG=POLY/"NUMERIC_SEQUENCE_NONADJACENT_OOD_CAPABILITY_V1_PREREGISTRATION.md"
PRIOR=ROOT/"basis_aligned/bilinear_quotient/circuits/prior_art/numeric_sequence_nonadjacent_ood_capability_v1.json"
BINDING=POLY/"NUMERIC_SEQUENCE_NONADJACENT_OOD_CAPABILITY_V1_BINDING.json"
OUT=POLY/"NUMERIC_SEQUENCE_NONADJACENT_OOD_CAPABILITY_V1_RESULT.json"
REQUESTED_DRY=bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
MINIMUM_ACCURACY=.85
PRICE={"forwards":1,"sequences":64,"causal_interventions":0,"fits":0}


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def load_bound():
    binding=json.loads(BINDING.read_text())
    paths={"rows":ROWS,"builder":POLY/"build_numeric_sequence_nonadjacent_ood_v1_rows.py",
           "preregistration":PREREG,"prior_art":PRIOR}
    for name,expected in binding["files"].items():
        if digest(paths[name])!=expected: raise RuntimeError(f"bound {name} changed")
    frozen=json.loads(ROWS.read_text())
    if canonical(frozen["rows"])!=frozen["row_manifest_sha256"]: raise RuntimeError("row manifest changed")
    if binding["row_manifest_sha256"]!=frozen["row_manifest_sha256"] or binding["price"]!=PRICE: raise RuntimeError("binding changed")
    return frozen


def plan():
    frozen=load_bound()
    return {"schema":"numeric_sequence_nonadjacent_ood_capability_v1_plan","model_loaded":False,
            "gpu_accessed":False,"queue_touched":False,"rows":frozen["row_count"],"endpoints":frozen["endpoint_count"],
            "minimum_accuracy_each_cell":MINIMUM_ACCURACY,"price":PRICE,
            "predicates":["pred_a_frozen_nonadjacent_authority","pred_b_native_digit_capability",
                          "pred_c_native_number_word_capability","pred_d_exact_capability_price"]}


def main():
    frozen=load_bound()
    if REQUESTED_DRY:
        print(json.dumps(plan(),indent=2,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    sys.path.insert(0,str(OPS))
    import torch
    import torch.nn.functional as F
    import run_bracket_l13h8_source_region_payload_factorial as exact
    from circuit_fast_screen_managed_runner import atomic_create_json
    torch.set_num_threads(2)
    model,checkpoint=exact._dependencies()[2].load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
    endpoints=[(row,side) for row in frozen["rows"] for side in ("base","donor")]
    length=max(len(row[side]["ids"]) for row,side in endpoints)
    tokens=torch.full((len(endpoints),length),50256,dtype=torch.long,device="cuda"); finals=[]
    for index,(row,side) in enumerate(endpoints):
        ids=row[side]["ids"]; tokens[index,:len(ids)]=torch.tensor(ids,dtype=torch.long,device="cuda"); finals.append(len(ids)-1)
    with torch.inference_mode(): logits=exact.native_logits(model,tokens,torch,F)
    cells=defaultdict(list)
    for index,(row,side) in enumerate(endpoints):
        other="donor" if side=="base" else "base"; answer=int(row[side]["answer_id"]); contrast=int(row[other]["answer_id"])
        margin=float(logits[index,finals[index],answer]-logits[index,finals[index],contrast])
        cells[f'{row["construction"]}|{row["representation"]}'].append({"row_id":row["row_id"],"side":side,"margin":margin,"correct":margin>0})
    reports={key:{"n":len(items),"accuracy":sum(x["correct"] for x in items)/len(items),
                  "mean_partner_margin":sum(x["margin"] for x in items)/len(items)} for key,items in sorted(cells.items())}
    pred_a=(len(frozen["rows"])==32 and len(endpoints)==64 and len(reports)==4 and {x["n"] for x in reports.values()}=={16})
    passes=lambda x:x["accuracy"]>=MINIMUM_ACCURACY and x["mean_partner_margin"]>0
    digits={k:v for k,v in reports.items() if k.endswith("|digit")}; words={k:v for k,v in reports.items() if k.endswith("|number_word")}
    predictions={"pred_a_frozen_nonadjacent_authority":bool(pred_a),
                 "pred_b_native_digit_capability":bool(pred_a and len(digits)==2 and all(passes(x) for x in digits.values())),
                 "pred_c_native_number_word_capability":bool(pred_a and len(words)==2 and all(passes(x) for x in words.values())),
                 "pred_d_exact_capability_price":True}
    result={"schema":"numeric_sequence_nonadjacent_ood_capability_v1_result",
            "terminal":"capability_pass" if all(predictions.values()) else "capability_null",
            "predictions":predictions,"cell_reports":reports,
            "price":{**PRICE,"observed_forwards":1,"observed_sequences":len(endpoints)},
            "row_manifest_sha256":frozen["row_manifest_sha256"],"checkpoint_weights_sha256":checkpoint.weights_sha256,
            "runner_sha256":digest(RUNNER)}
    atomic_create_json(OUT,result); print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
