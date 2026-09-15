#!/usr/bin/env python3
"""Register the held native L11H3 weight axis."""
from copy import deepcopy
import json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; BQ=REPO/'basis_aligned/bilinear_quotient'; sys.path.insert(0,str(BQ))
from circuit_registry_v2 import _atomic_json,_lock,circuit_path,design_key,execution_key,file_sha256,rebuild_registry_v2,validate_v2
TAG='task.subject_verb.number_agreement'; OLD='grammatical_subject_number.v28'; NEW='grammatical_subject_number.v29'; EVENT='subject_number_native_weight_axis.v1.complete.held'
RESULT=BQ/'circuits/fast_screens/subject_number_native_weight_axis_v1_result.json'; AUDIT=HERE/'SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_AUDIT.json'
ARTIFACTS={
 'subject_number_native_axis_builder_v1':('basis_aligned/polynomial_causal/build_subject_number_native_weight_axis_v1.py','builder'),
 'subject_number_native_axis_artifact_v1':('basis_aligned/polynomial_causal/SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_ARTIFACT.json','artifact'),
 'subject_number_native_axis_prereg_v1':('basis_aligned/polynomial_causal/SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_PREREGISTRATION.md','preregistration'),
 'subject_number_native_axis_binding_v1':('basis_aligned/polynomial_causal/SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_BINDING.json','binding'),
 'subject_number_native_axis_runner_v1':('basis_aligned/bilinear_quotient/ops/run_subject_number_native_weight_axis_v1.py','runner'),
 'subject_number_native_axis_test_v1':('basis_aligned/bilinear_quotient/ops/test_subject_number_native_weight_axis_v1.py','test'),
 'subject_number_native_axis_result_v1':('basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_native_weight_axis_v1_result.json','result'),
 'subject_number_native_axis_auditor_v1':('basis_aligned/polynomial_causal/audit_subject_number_native_weight_axis_v1.py','audit'),
 'subject_number_native_axis_audit_v1':('basis_aligned/polynomial_causal/SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_AUDIT.json','audit')}
def art(p,k): return {'path':p,'sha256':file_sha256(REPO/p),'kind':k,'status':'frozen'}
def m(n,e,b): return {'name':n,'estimate':e,'ci95':None,'bar':b}
def main():
 r=json.loads(RESULT.read_text()); a=json.loads(AUDIT.read_text()); assert r['terminal']=='native_weight_axis_held' and a['all_checks_pass']; path=circuit_path(TAG)
 with _lock('registry'):
  record=json.loads(path.read_text()); assert record['claims'][-1]['claim_id']==OLD
  for aid,spec in ARTIFACTS.items(): record['artifacts'][aid]=art(*spec)
  previous=record['claims'][-1]; claim=deepcopy(previous); claim.update({'claim_id':NEW,'revision':29,'supersedes':OLD,'evidence_event_ids':[*previous['evidence_event_ids'],EVENT],'next_missing':'The fitted 1152D subject write axis is now replaced by the native top left singular direction of the L11H3 output projection while preserving fresh causal effects at .99945 cosine versus the held axis law. Next derive direction and background-cardinality variables from native activations; do not fit axis mixtures/scales, add scalar-table variants, rescue weak reverse composition, or use quantization.'}); record['claims'].append(claim)
  s=r['score']; event={'event_id':EVENT,'claim_id':NEW,'test_type':'compiled_equivalence','stage':'complete','verdict':'held','failure_kind':None,'family_ids':[],'site_id':'L11H3.projected_write.direction_cardinality_rank1_program','split_plan_id':'task14_fresh_matched_natural_split_v1','evaluation_role':'fresh_native_weight_axis_substitution','metrics':[m('prior_axis_cosine',r['plan']['native_axis_prior_cosine'],'>=.94'),m('native_axis_vs_law_cosine',s['native_axis_vs_law_axis']['cosine'],'>=.90'),m('native_axis_vs_law_relative_l2',s['native_axis_vs_law_axis']['relative_l2_error'],'<=.50'),m('native_axis_vs_law_sign',s['native_axis_vs_law_axis']['sign_agreement'],'>=.90'),m('native_axis_vs_native_cosine',s['native_axis_vs_native']['cosine'],'>=.65'),m('native_axis_vs_native_relative_l2',s['native_axis_vs_native']['relative_l2_error'],'<=.90'),m('native_axis_vs_native_sign',s['native_axis_vs_native']['sign_agreement'],'>=.70')],'prereg_artifact_id':'subject_number_native_axis_prereg_v1','result_artifact_id':'subject_number_native_axis_result_v1','input_artifact_ids':list(ARTIFACTS),'seed':None,'checkpoint_sha256':r['checkpoint_weights_sha256'],'supersedes_event_id':None,'replicates_event_id':None,'sections':[],'notes':'Axis is the checkpoint-weight-only top left singular vector of L11H3 output projection; prior axis fixes gauge sign only. Four-scalar law unchanged; zero causal fits, gradients, updates, or quantization.'}; event['design_key']=design_key(record,event); event['execution_key']=execution_key(record,event); record['evidence_events'].append(event); validate_v2(record); _atomic_json(path,record)
 rebuild_registry_v2(); print(json.dumps({'status':'registered','claim_id':NEW,'event_id':EVENT},indent=2))
if __name__=='__main__': main()
