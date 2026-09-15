#!/usr/bin/env python3
"""Register donor-free bracket score and payload interaction compression."""
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; BQ=REPO/"basis_aligned/bilinear_quotient"
sys.path.insert(0,str(BQ))
from circuit_registry_v2 import _atomic_json,_lock,circuit_path,design_key,execution_key,file_sha256,rebuild_registry_v2,validate_v2

TAG="task.bracket_pending_opener"; OLD="pending_opener_state.v33"; NEW="pending_opener_state.v34"
EVENT="pending_opener.layered_key_payload_rank2_live_query.held.v1"
RESULT=HERE/"BRACKET_LAYERED_PENDING_KEY_RANK2_INTERACTION_V1_RESULT.json"
ARTIFACTS={
 "layered_pending_rows_builder_v1":("basis_aligned/polynomial_causal/build_bracket_layered_pending_ood_v1_rows.py","builder"),
 "layered_pending_rows_test_v1":("basis_aligned/polynomial_causal/test_build_bracket_layered_pending_ood_v1_rows.py","test"),
 "layered_pending_rows_v1":("basis_aligned/polynomial_causal/BRACKET_LAYERED_PENDING_OOD_V1_ROWS.json","rows"),
 "layered_key_rank2_v1_prereg":("basis_aligned/polynomial_causal/BRACKET_LAYERED_PENDING_KEY_RANK2_INTERACTION_V1_PREREGISTRATION.md","preregistration"),
 "layered_key_rank2_v1_binding":("basis_aligned/polynomial_causal/BRACKET_LAYERED_PENDING_KEY_RANK2_INTERACTION_V1_BINDING.json","binding"),
 "layered_key_rank2_v1_runner":("basis_aligned/bilinear_quotient/ops/run_bracket_layered_pending_key_rank2_interaction_v1.py","runner"),
 "layered_key_rank2_v1_test":("basis_aligned/bilinear_quotient/ops/test_run_bracket_layered_pending_key_rank2_interaction_v1.py","test"),
 "layered_key_rank2_v1_result":("basis_aligned/polynomial_causal/BRACKET_LAYERED_PENDING_KEY_RANK2_INTERACTION_V1_RESULT.json","result")}
def art(path,kind): return {"path":path,"sha256":file_sha256(REPO/path),"kind":kind,"status":"frozen"}

def main():
 result=json.loads(RESULT.read_text()); assert result["terminal"]=="key_rank2_payload_rank2_interaction_transfer" and all(result["predictions"].values()); assert result["price"]["observed_forwards"]==8 and result["price"]["observed_sequences"]==2304
 path=circuit_path(TAG); current=json.loads(path.read_text())
 if any(e["event_id"]==EVENT for e in current["evidence_events"]): validate_v2(current); rebuild_registry_v2(); print("already registered"); return
 with _lock("registry"):
  record=json.loads(path.read_text()); previous=next(c for c in record["claims"] if c["claim_id"]==OLD)
  for aid,spec in ARTIFACTS.items(): record["artifacts"][aid]=art(*spec)
  split="pending_opener_fifth_layered_v1"
  record["split_plans"].append({"split_plan_id":split,"unit":"fresh lexical group and exact token sequence across third-authority training and fifth layered-construction test","partition_artifact_id":"layered_pending_rows_v1","builder_artifact_id":"layered_pending_rows_builder_v1","seed":None,"groups":{"THIRD_AUTHORITY_TRAIN":72,"FIFTH_CONSTRUCTION_TEST":72},"leakage_group_keys":["exact token sequence","lexical group","construction template","row id"],"sealed_before_outcomes":True,"sealed_at":"2026-09-15T00:27:28Z"})
  claim=deepcopy(previous); claim.update({"claim_id":NEW,"revision":34,"supersedes":OLD,"evidence_event_ids":[*previous["evidence_event_ids"],EVENT],"next_missing":"Third-authority centered rank-two tables for both normalized pre-rotary keys and the projected payload transfer to a fresh fifth construction with no donor-state access: live-recipient-query score relative error .1897 and downstream effect relative error .0966, improving .2288 over recipient score, with controls .0596 of target RMS. This closes the local three-type L13H8 source generator conditional on native recipient query/payload and the counterfactual delimiter type. Next combine this donor-free generator with the already transferred six-scalar suffix effect law and confirm a complete compact behavioral predictor on a sixth construction; do not revisit source rank, gains, full-term vectors, or quantization."})
  claim["causal_variable"]["operation"]="use rank-two delimiter-type tables for both rotary key factors and projected payload difference; contract keys with the live recipient query, multiply the two score factors, and write the resulting source term without donor-state access"
  claim["counterfactual_families"].extend([{"family_id":"layered_pending_stack_top","role":"interchange","changes":["inner stack-top delimiter type","correct immediate closer"],"holds_fixed":["outer brace opener","middle pending delimiter","new layered surface","length and positions"],"builder_artifact_id":"layered_pending_rows_builder_v1","control_ids":["exact joint ceiling","single-key arms","recipient-score baseline"],"split_plan_id":split,"status":"validated"},{"family_id":"layered_middle_change_inner_fixed","role":"invariance","changes":["middle pending delimiter type"],"holds_fixed":["outer brace opener","inner stack-top delimiter","correct closer","new words and aligned length"],"builder_artifact_id":"layered_pending_rows_builder_v1","control_ids":["donor-free program","native capability","exact joint ceiling"],"split_plan_id":split,"status":"validated"}]); claim["split_plan_ids"]=[*claim.get("split_plan_ids",[]),split]
  site={"site_id":"attention13.head8.opener_type_key_payload_rank2_live_query","tensor_path":"(<q1_r,K1_type(d)>/128)(<q2_r,K2_type(d)>/128) times (u_r plus rank-two type payload difference)","shape":["batch",1,1152],"intervention":"install donor-free rank-two type-key/type-payload source term before native suffix","ceiling_event_ids":[EVENT]}; claim["candidate_sites"].append(site); record["claims"].append(claim)
  km=result["key_source_metrics"]; fm=result["score_factor_metrics"]; em=result["effect_metrics"]["both"]
  event={"event_id":EVENT,"claim_id":NEW,"test_type":"cross_family_transfer","stage":"complete","verdict":"held","failure_kind":None,"family_ids":["layered_pending_stack_top","layered_middle_change_inner_fixed"],"site_id":site["site_id"],"split_plan_id":split,"evaluation_role":"fresh_fifth_construction_donor_free_confirmation","metrics":[{"name":"key1_relative_l2","estimate":km["key1"]["relative_l2_error"],"ci95":None,"bar":"<=0.50"},{"name":"key2_relative_l2","estimate":km["key2"]["relative_l2_error"],"ci95":None,"bar":"<=0.50"},{"name":"score_relative_l2","estimate":fm["score"]["relative_l2_error"],"ci95":None,"bar":"<=0.60"},{"name":"compressed_effect_relative_l2","estimate":em["relative_l2_error"],"ci95":None,"bar":"<=0.40"},{"name":"improvement_over_recipient_score","estimate":result["relative_l2_improvement_over_recipient_score"],"ci95":None,"bar":">=0.10"},{"name":"control_to_target_rms","estimate":result["control_to_target_rms"],"ci95":None,"bar":"<=0.50"}],"prereg_artifact_id":"layered_key_rank2_v1_prereg","result_artifact_id":"layered_key_rank2_v1_result","input_artifact_ids":list(ARTIFACTS),"seed":None,"checkpoint_sha256":result["checkpoint_sha256"],"supersedes_event_id":None,"replicates_event_id":None,"sections":["basis_aligned/polynomial_causal/BRACKET_LAYERED_PENDING_KEY_RANK2_INTERACTION_V1_PREREGISTRATION.md"],"notes":"Three outcome-blind forced rank-two SVDs; no donor state in the fully compressed arm and no gradient, gain, update, full-term-vector reuse, or quantization."}
  event["design_key"]=design_key(record,event); event["execution_key"]=execution_key(record,event); record["evidence_events"].append(event); validate_v2(record); _atomic_json(path,record)
 rebuild_registry_v2(); final=json.loads(path.read_text()); validate_v2(final); print(json.dumps({"status":"registered","claim_id":NEW,"event_id":EVENT},indent=2))
if __name__=="__main__": main()
