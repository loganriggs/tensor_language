#!/usr/bin/env python3
"""Broad OOD targets and controls for the three-vector absolute bracket program."""

# BQGATE: EXPERIMENT pred_a_immutable_artifact_capability_and_replay pred_b_absolute_program_transfers_both_families pred_c_every_ordered_pair_recurs pred_d_no_edit_controls_are_exact_noops pred_e_fixed_price_and_boundary
from __future__ import annotations
import hashlib,json,math,os,sys
from datetime import datetime,timezone
from pathlib import Path
import circuit_fast_screen_candidate_bracket_l13h8_ordered_pair_displacement_program as authority
import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_ordered_pair_displacement_program_ood_validation as parent
import run_bracket_l13h8_source_region_payload_factorial as exact

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/bracket_absolute_term_program_ood_control_validation_v2.json";ARTIFACT=ROOT/"circuits/followups/bracket_l13h8_closer_absolute_term_program_v1_artifact.json";FRESH=ROOT/"circuits/followups/bracket_l13h8_closer_absolute_term_program_v1_result.json";CAP=ROOT/"circuits/fast_screens/bracket_l13h8_ordered_pair_program_ood_capability_v1_result.json";DISP=ROOT/"circuits/followups/bracket_l13h8_ordered_pair_displacement_program_ood_validation_v1_result.json";OUT=ROOT/"circuits/followups/bracket_absolute_term_program_ood_control_validation_v2_result.json"
CANDIDATE_ID="bracket.pending_opener.absolute_term_program_ood_control_validation_v2"
EXPECTED={PRIOR:"2e30563cb9c6c6d88f97becbab041a022354c32225e77959444dd18360160185",ARTIFACT:"c365a9d3e5fccc8ba8463099571c1fbdf059e49ed8426cf0e32dddec30644930",FRESH:"c3b80c69d51453e91dea1c45f7df095d7e5821de9441d1dea67a169e07263f10",CAP:"a1bb465af45b6d7c4059370629d117017aa01e028ed0a985b58d8bbb46da5622",DISP:"3b267f069647824fb7557e9784c63becb0366f94fe4d274fea343ae2bc802e5f"}
BARS={"maximum_replay_error":1e-4,"minimum_cosine":.65,"maximum_relative_l2":.90,"minimum_norm_ratio":.25,"maximum_norm_ratio":2.,"minimum_sign_agreement":.75,"minimum_family_cosine":.60,"minimum_family_sign_agreement":.70,"minimum_pair_positive_fraction":.75,"minimum_pair_sign_agreement":.70,"maximum_control_logit_change":1e-4}
def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _load():
 for p,h in EXPECTED.items():
  if _sha(p)!=h:raise ValueError(f"immutable input changed: {p}")
 artifact,fresh,cap,disp=(json.loads(p.read_text()) for p in (ARTIFACT,FRESH,CAP,DISP))
 if artifact.get("terminal")!="prototype_artifact" or fresh.get("terminal")!="program_screen" or cap.get("terminal")!="capability_pass" or disp.get("terminal")!="program_screen":raise ValueError("parent status invalid")
 return artifact,cap
def compile_plan():
 _load();rows=authority.build_ood_rows()
 return {"schema":"bracket_absolute_term_program_ood_control_validation_plan_v2","candidate_id":CANDIDATE_ID,"prior_art_sha256":EXPECTED[PRIOR],"absolute_artifact_sha256":EXPECTED[ARTIFACT],"fresh_result_sha256":EXPECTED[FRESH],"capability_sha256":EXPECTED[CAP],"displacement_result_sha256":EXPECTED[DISP],"rows":len(rows),"endpoints":2*len(rows),"target_endpoints":144,"control_endpoints":216,"target_families":list(authority.TARGET_FAMILIES),"control_families":list(authority.CONTROL_FAMILIES),"bars":dict(BARS),"price":{"physical_model_forwards":3,"example_evaluations":1080,"backwards":0,"fits":0,"parameter_updates":0,"prototype_changes":0}}
def evaluate(model,torch,F,facade):
 artifact,cap=_load();rows=authority.build_ood_rows();endpoints,tokens,finals,sources=parent._pad(rows,torch,next(model.parameters()).device);replay,factors=exact.factor_forward(model,tokens,finals,{},torch,F,facade);arange=torch.arange(len(endpoints),device=tokens.device);native=factors["p"][arange,sources].unsqueeze(-1)*factors["u"][arange,sources];donor=native[arange^1];prototypes={k:torch.tensor(v["coordinates"],dtype=torch.float32,device=tokens.device) for k,v in artifact["prototypes"].items()};exact_terms=[];program_terms=[];dispatch=[]
 for i,(row,side) in enumerate(endpoints):
  other="donor" if side=="base" else "base"
  if row["program_role"]=="target":exact_terms.append(donor[i]);program_terms.append(prototypes[str(row[f"{other}_answer_id"])]);dispatch.append(str(row[f"{other}_answer_id"]))
  else:exact_terms.append(native[i]);program_terms.append(native[i]);dispatch.append("no_edit_native")
 exact_terms=torch.stack(exact_terms);program_terms=torch.stack(program_terms);exact_logits=exact.factor_forward(model,tokens,finals,{},torch,F,facade,replacement_terms=exact_terms,source_positions=sources)[0];program_logits=exact.factor_forward(model,tokens,finals,{},torch,F,facade,replacement_terms=program_terms,source_positions=sources)[0];native_margins={(r["row_id"],r["side"]):r["closer_margin"] for r in cap["evidence"]};records=[]
 for i,(row,side) in enumerate(endpoints):
  q=int(finals[i]);other="donor" if side=="base" else "base";recipient=row[f"{side}_answer_id"];direction="base_to_donor" if side=="base" else "donor_to_base";record={"row_id":row["row_id"],"family_id":row["family_id"],"program_role":row["program_role"],"side":side,"ordered_pair":f'{recipient}->{row[f"{other}_answer_id"]}',"dispatch":dispatch[i],"native_margin_replay_absolute_error":abs(exact.closer_margin(replay[i,q],recipient)-native_margins[(row["row_id"],side)]),"program_max_absolute_logit_change":float((program_logits[i,q]-replay[i,q]).abs().max()),"program_recipient_correct":bool(exact.closer_margin(program_logits[i,q],recipient)>0)}
  if row["program_role"]=="target":record.update({"exact_donorward_effect":exact.endpoint_change(replay[i,q],exact_logits[i,q],row,direction),"absolute_program_donorward_effect":exact.endpoint_change(replay[i,q],program_logits[i,q],row,direction)})
  records.append(record)
 return records
def _metrics(rows):
 a=[r["exact_donorward_effect"] for r in rows];p=[r["absolute_program_donorward_effect"] for r in rows];an=math.sqrt(sum(x*x for x in a));pn=math.sqrt(sum(x*x for x in p));return {"count":len(rows),"cosine":sum(x*y for x,y in zip(a,p))/(an*pn),"relative_l2_error":math.sqrt(sum((x-y)**2 for x,y in zip(a,p)))/an,"predicted_to_actual_norm_ratio":pn/an,"sign_agreement":sum((x>0)==(y>0) for x,y in zip(a,p))/len(rows),"positive_program_fraction":sum(x>0 for x in p)/len(rows)}
def score(records):
 targets=[r for r in records if r["program_role"]=="target"];controls=[r for r in records if r["program_role"]=="control"];overall=_metrics(targets);families={f:_metrics([r for r in targets if r["family_id"]==f]) for f in authority.TARGET_FAMILIES};pairs={f"{a}->{b}":_metrics([r for r in targets if r["ordered_pair"]==f"{a}->{b}"]) for a,b in authority.ORDERED_PAIRS};replay=max(r["native_margin_replay_absolute_error"] for r in records);instrument=len(records)==360 and len(targets)==144 and len(controls)==216 and replay<=BARS["maximum_replay_error"];overall_pass=overall["cosine"]>=BARS["minimum_cosine"] and overall["relative_l2_error"]<=BARS["maximum_relative_l2"] and BARS["minimum_norm_ratio"]<=overall["predicted_to_actual_norm_ratio"]<=BARS["maximum_norm_ratio"] and overall["sign_agreement"]>=BARS["minimum_sign_agreement"] and all(v["cosine"]>=BARS["minimum_family_cosine"] and v["sign_agreement"]>=BARS["minimum_family_sign_agreement"] for v in families.values());pair_pass=all(v["count"]==24 and v["positive_program_fraction"]>=BARS["minimum_pair_positive_fraction"] and v["sign_agreement"]>=BARS["minimum_pair_sign_agreement"] for v in pairs.values());control_max=max(r["program_max_absolute_logit_change"] for r in controls);control_pass=control_max<=BARS["maximum_control_logit_change"] and all(r["program_recipient_correct"] for r in controls) and all(r["dispatch"]=="no_edit_native" for r in controls);pred={"pred_a_immutable_artifact_capability_and_replay":instrument,"pred_b_absolute_program_transfers_both_families":instrument and overall_pass,"pred_c_every_ordered_pair_recurs":instrument and pair_pass,"pred_d_no_edit_controls_are_exact_noops":instrument and control_pass,"pred_e_fixed_price_and_boundary":compile_plan()["price"]=={"physical_model_forwards":3,"example_evaluations":1080,"backwards":0,"fits":0,"parameter_updates":0,"prototype_changes":0}};terminal="program_screen" if all(pred.values()) else "null" if instrument else "invalid";return {"overall":overall,"by_family":families,"by_ordered_pair":pairs,"native_margin_replay_max_absolute_error":replay,"control_max_absolute_logit_change":control_max,"negative_boundary_retained":"pair-centered selective necessity remains false","predictions":pred,"terminal":terminal}
def main():
 plan=compile_plan()
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1" or "--dry-run" in sys.argv:print(json.dumps(plan,sort_keys=True));return
 if OUT.exists():raise ValueError(f"refusing overwrite {OUT}")
 torch,F,facade=exact._dependencies();model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 with torch.no_grad():records=evaluate(model,torch,F,facade)
 scored=score(records);payload=managed.atomic_create_json(OUT,{"schema":"bracket_absolute_term_program_ood_control_validation_result_v2","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,"score":scored,"evidence":records,"terminal":scored["terminal"]});print(json.dumps({"terminal":scored["terminal"],"predictions":scored["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
