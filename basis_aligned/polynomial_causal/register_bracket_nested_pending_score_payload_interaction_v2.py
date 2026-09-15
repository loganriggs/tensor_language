#!/usr/bin/env python3
"""Register the selective bracket L13H8 score-payload interaction."""
from copy import deepcopy
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; BQ=REPO/'basis_aligned/bilinear_quotient'; sys.path.insert(0,str(BQ))
from circuit_registry_v2 import _atomic_json,_lock,circuit_path,design_key,execution_key,file_sha256,rebuild_registry_v2,validate_v2
TAG='task.bracket_pending_opener'; OLD='pending_opener_state.v31'; NEW='pending_opener_state.v32'; EVENT='pending_opener.nested_score_payload_interaction.held.v2'
ARTIFACTS={
 'nested_score_payload_v1_prereg':('basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_SCORE_PAYLOAD_INTERACTION_V1_PREREGISTRATION.md','preregistration'),
 'nested_score_payload_v1_binding':('basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_SCORE_PAYLOAD_INTERACTION_V1_BINDING.json','binding'),
 'nested_score_payload_v1_runner':('basis_aligned/bilinear_quotient/ops/run_bracket_nested_pending_score_payload_interaction_v1.py','runner'),
 'nested_score_payload_v1_result':('basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_SCORE_PAYLOAD_INTERACTION_V1_RESULT.json','result'),
 'nested_score_payload_v2_prereg':('basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_SCORE_PAYLOAD_INTERACTION_V2_PREREGISTRATION.md','preregistration'),
 'nested_score_payload_v2_binding':('basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_SCORE_PAYLOAD_INTERACTION_V2_BINDING.json','binding'),
 'nested_score_payload_v2_runner':('basis_aligned/bilinear_quotient/ops/run_bracket_nested_pending_score_payload_interaction_v2.py','runner'),
 'nested_score_payload_v2_result':('basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_SCORE_PAYLOAD_INTERACTION_V2_RESULT.json','result')}
def art(p,k): return {'path':p,'sha256':file_sha256(REPO/p),'kind':k,'status':'frozen'}
def main():
 result=json.loads((HERE/'BRACKET_NESTED_PENDING_SCORE_PAYLOAD_INTERACTION_V2_RESULT.json').read_text()); assert result['terminal']=='score_payload_interaction_required'; assert result['predictions']=={'pred_a_exact_instrument':True,'pred_b_joint_target_live':True,'pred_c_additive_score_payload_sufficient':False,'pred_d_material_score_payload_interaction':True,'pred_e_control_selectivity':True}; assert result['price']['forwards']==6 and result['price']['sequences']==864
 path=circuit_path(TAG); current=json.loads(path.read_text())
 if any(e['event_id']==EVENT for e in current['evidence_events']): validate_v2(current); rebuild_registry_v2(); print('already registered'); return
 with _lock('registry'):
  record=json.loads(path.read_text()); previous=next(c for c in record['claims'] if c['claim_id']==OLD)
  for aid,spec in ARTIFACTS.items(): record['artifacts'][aid]=art(*spec)
  claim=deepcopy(previous); claim.update({'claim_id':NEW,'revision':32,'supersedes':OLD,'evidence_event_ids':[*previous['evidence_event_ids'],EVENT],
   'next_missing':'The nested L13H8 opener edit has a selective material score×payload interaction: (pd-pr)(ud-ur) is 37.34% of the exact source-delta norm and the downstream behavioral interaction is 37.06% of joint effect norm. Additive score+payload omits it and has .386 relative error. Next freeze an outcome-blind low-rank/ordered-pair representation of payload differences multiplied by the live score delta and confirm on a fourth construction; do not scale/refit the failed full displacement vectors or quantize.'})
  site={'site_id':'attention13.head8.opener_score_payload_interaction','tensor_path':'(p_d-p_r) scalar times (u_d-u_r) 1152D projected opener-value difference at the active source','shape':['batch',1,1152],'intervention':'install score-only, payload-only, additive, joint, or exact bilinear interaction source term before native suffix','ceiling_event_ids':[EVENT]}; claim['candidate_sites'].append(site); record['claims'].append(claim); o=result['overall']
  event={'event_id':EVENT,'claim_id':NEW,'test_type':'composition','stage':'complete','verdict':'held','failure_kind':None,'family_ids':['nested_two_pending_stack_top','outer_pending_type_change_inner_fixed'],'site_id':site['site_id'],'split_plan_id':'pending_opener_three_value_fresh_split_r545_v1','evaluation_role':'opened_authority_exact_score_payload_factorial','metrics':[
   {'name':'source_interaction_over_joint_delta_norm','estimate':o['source_interaction_over_joint_delta_norm'],'ci95':None,'bar':'>=0.10'},
   {'name':'behavioral_interaction_over_joint_norm','estimate':o['behavioral_interaction_over_joint_norm'],'ci95':None,'bar':'descriptive'},
   {'name':'additive_vs_joint_cosine','estimate':o['additive_vs_joint']['cosine'],'ci95':None,'bar':'>=0.95 for additive sufficiency'},
   {'name':'additive_vs_joint_relative_l2','estimate':o['additive_vs_joint']['relative_l2_error'],'ci95':None,'bar':'<=0.25 for additive sufficiency'},
   {'name':'maximum_control_to_target_rms','estimate':result['maximum_control_to_target_joint_rms'],'ci95':None,'bar':'<=0.50'},
   {'name':'source_closure_error','estimate':result['instrument']['source_term_closure_max_absolute_error'],'ci95':None,'bar':'<=1e-5'}],
   'prereg_artifact_id':'nested_score_payload_v2_prereg','result_artifact_id':'nested_score_payload_v2_result','input_artifact_ids':list(ARTIFACTS),'seed':None,'checkpoint_sha256':result['checkpoint_sha256'],'supersedes_event_id':None,'replicates_event_id':None,'sections':['basis_aligned/polynomial_causal/BRACKET_NESTED_PENDING_SCORE_PAYLOAD_INTERACTION_V2_PREREGISTRATION.md'],'notes':'V1 causal arms were instrument-invalid only because closure was accumulated in FP32. V2 changed the independent closure audit to FP64; interventions and outcomes were unchanged. No fit, gain, rank sweep, or quantization.'}
  event['design_key']=design_key(record,event); event['execution_key']=execution_key(record,event); record['evidence_events'].append(event); validate_v2(record); _atomic_json(path,record)
 rebuild_registry_v2(); final=json.loads(path.read_text()); validate_v2(final); print(json.dumps({'status':'registered','claim_id':NEW,'event_id':EVENT},indent=2))
if __name__=='__main__': main()
