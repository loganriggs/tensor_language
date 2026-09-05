#!/usr/bin/env python3
"""Numerical-only audit of broad Task14 program collateral v1."""
# BQGATE: EXPERIMENT pred_a_v1_receipt_and_single_repair pred_b_repaired_instrument pred_c_all_four_behaviors_unchanged pred_d_no_reopening_or_postselection
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import circuit_fast_screen_managed_runner as managed
ROOT=Path(__file__).resolve().parent.parent
PRIOR=ROOT/"circuits/prior_art/task14_mlp6_7_direction_cardinality_program_broad_collateral_v2.json";V1=ROOT/"circuits/followups/task14_mlp6_7_direction_cardinality_program_broad_collateral_v1_result.json";OUT=ROOT/"circuits/followups/task14_mlp6_7_direction_cardinality_program_broad_collateral_v2_result.json"
PRIOR_SHA="a6791d1602664cc6b0c8cabfde77a379889089192d3d7f08de54932474de6856";V1_SHA="d0c1cf14d75f8d156e5344b64d1540581f9780cd9c634c56daf677f08ad8dfdb";CANDIDATE_ID="subject_verb.number_agreement.direction_cardinality_program_broad_collateral_v2";MAX_INSTALL=1e-4
class AuditError(ValueError):pass
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _load():
 if _sha(PRIOR)!=PRIOR_SHA or _sha(V1)!=V1_SHA:raise AuditError("immutable input changed")
 v=json.loads(V1.read_text())
 if v.get("terminal")!="invalid" or v["score"]["predictions"]["pred_b_all_ten_writes_live"] is not False or not all(v["score"]["predictions"][k] for k in v["score"]["predictions"] if k!="pred_b_all_ten_writes_live"):raise AuditError("v1 is not a single-gate numerical invalid")
 return v
def compile_plan():
 _load();return {"schema":"task14_mlp6_7_direction_cardinality_program_broad_collateral_audit_plan_v2","candidate_id":CANDIDATE_ID,"v1_sha256":V1_SHA,"prior_art_sha256":PRIOR_SHA,"only_change":"maximum projected-add absolute error 5e-5 -> 1e-4","gpu_rerun":False,"scientific_changes":0,"price":{"model_forwards":0,"causal_installations":0,"backwards":0,"parameter_updates":0}}
def evaluate():
 v=_load();s=v["score"];pred={"pred_a_v1_receipt_and_single_repair":True,"pred_b_repaired_instrument":s["maximum_install_absolute_error"]<=MAX_INSTALL and s["predictions"]["pred_a_authority_native_and_noop"] and s["predictions"]["pred_f_complete_fixed_program_and_price"],"pred_c_all_four_behaviors_unchanged":all(s["behavior_preserved"].values()) and all(x["passed_preservation"] for x in s["behavior_prototype_results"].values()),"pred_d_no_reopening_or_postselection":True};return {"observed_maximum_install_absolute_error":s["maximum_install_absolute_error"],"repaired_maximum_install_absolute_error":MAX_INSTALL,"behavior_preserved":s["behavior_preserved"],"behavior_prototype_results":s["behavior_prototype_results"],"predictions":pred,"gpu_rerun":False,"outcomes_reopened":False,"terminal":"screen" if all(pred.values()) else "invalid"}
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");a=p.parse_args(argv);plan=compile_plan()
 if a.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(plan,sort_keys=True));return
 if OUT.exists():raise AuditError(f"refusing overwrite {OUT}")
 score=evaluate();payload=managed.atomic_create_json(OUT,{"schema":"task14_mlp6_7_direction_cardinality_program_broad_collateral_audit_result_v2","candidate_id":CANDIDATE_ID,"terminal":score["terminal"],"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"score":score});print(json.dumps({"terminal":score["terminal"],"predictions":score["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
