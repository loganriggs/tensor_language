#!/usr/bin/env python3
"""Register fresh confirmation of the subject-number rank-one program."""
from copy import deepcopy
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; BQ=REPO/'basis_aligned/bilinear_quotient'
sys.path.insert(0,str(BQ))
from circuit_registry_v2 import _atomic_json,_lock,circuit_path,design_key,execution_key,file_sha256,rebuild_registry_v2,validate_v2
TAG='task.subject_verb.number_agreement'; OLD='grammatical_subject_number.v24'; NEW='grammatical_subject_number.v25'
EVENT='task14.direction_cardinality_rank1.fresh_confirmation.held.v1'
ARTIFACTS={
 'task14_rank1_fresh_authority_v1':('basis_aligned/bilinear_quotient/ops/circuit_fast_screen_candidate_subject_number_rank1_fresh_confirmation.py','rows'),
 'task14_rank1_fresh_native_prereg_v1':('basis_aligned/polynomial_causal/SUBJECT_NUMBER_RANK1_FRESH_CONFIRMATION_NATIVE_CAPABILITY_V1_PREREGISTRATION.md','preregistration'),
 'task14_rank1_fresh_native_binding_v1':('basis_aligned/polynomial_causal/SUBJECT_NUMBER_RANK1_FRESH_CONFIRMATION_NATIVE_CAPABILITY_V1_BINDING.json','binding'),
 'task14_rank1_fresh_native_runner_v1':('basis_aligned/bilinear_quotient/ops/run_subject_number_rank1_fresh_confirmation_native_capability_v1.py','runner'),
 'task14_rank1_fresh_native_result_v1':('basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_rank1_fresh_confirmation_native_capability_v1_result.json','result'),
 'task14_rank1_fresh_license_v1':('basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_rank1_fresh_confirmation_v1_license.json','artifact'),
 'task14_rank1_fresh_prereg_v1':('basis_aligned/polynomial_causal/SUBJECT_NUMBER_RANK1_FRESH_CONFIRMATION_V1_PREREGISTRATION.md','preregistration'),
 'task14_rank1_fresh_binding_v1':('basis_aligned/polynomial_causal/SUBJECT_NUMBER_RANK1_FRESH_CONFIRMATION_V1_BINDING.json','binding'),
 'task14_rank1_fresh_runner_v1':('basis_aligned/bilinear_quotient/ops/run_subject_number_rank1_fresh_confirmation_v1.py','runner'),
 'task14_rank1_fresh_result_v1':('basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_rank1_fresh_confirmation_v1_result.json','result')}
def artifact(p,k): return {'path':p,'sha256':file_sha256(REPO/p),'kind':k,'status':'frozen'}
def bind(r,e): e['design_key']=design_key(r,e); e['execution_key']=execution_key(r,e); return e
def main():
 result=json.loads((BQ/'circuits/fast_screens/subject_number_rank1_fresh_confirmation_v1_result.json').read_text())
 cap=json.loads((BQ/'circuits/fast_screens/subject_number_rank1_fresh_confirmation_native_capability_v1_result.json').read_text())
 assert cap['terminal']=='pass' and all(x['passed'] for x in cap['cells'].values())
 assert result['terminal']=='rank1_fresh_confirmed' and all(result['score']['predictions'].values())
 path=circuit_path(TAG); existing=json.loads(path.read_text())
 if any(e['event_id']==EVENT for e in existing['evidence_events']): validate_v2(existing); rebuild_registry_v2(); print('already registered'); return
 with _lock('registry'):
  record=json.loads(path.read_text())
  for aid,spec in ARTIFACTS.items():
   value=artifact(*spec)
   if aid in record['artifacts'] and record['artifacts'][aid]!=value: raise ValueError(aid)
   record['artifacts'][aid]=value
  previous=next(c for c in record['claims'] if c['claim_id']==OLD); claim=deepcopy(previous)
  claim.update({'claim_id':NEW,'revision':25,'supersedes':OLD,'evidence_event_ids':[*previous['evidence_event_ids'],EVENT],
   'next_missing':'The ten direction-by-cardinality L11H3 writes are prospectively confirmed as one shared 1152D axis plus ten scalars on new nouns and constructions without refit. Storage remains 10.09% of the write bank and 25.07% of the charged interface. Do not repeat rank, gain, reader, program, lexical-transfer, or quantization work; move to another unresolved circuit unless independent evidence targets the upstream selector.'})
  site=next(s for s in claim['candidate_sites'] if s['site_id']=='L11H3.projected_write.direction_cardinality_rank1_program')
  site['ceiling_event_ids']=[*site['ceiling_event_ids'],EVENT]
  record['claims'].append(claim); s=result['score']
  event={'event_id':EVENT,'claim_id':NEW,'test_type':'cross_family_transfer','stage':'complete','verdict':'held','failure_kind':None,'family_ids':[],
   'site_id':site['site_id'],'split_plan_id':'task14_fresh_matched_natural_split_v1','evaluation_role':'prospective_lexical_and_construction_confirmation',
   'metrics':[
    {'name':'native_capability_minimum_cell_accuracy','estimate':min(x['accuracy'] for x in cap['cells'].values()),'ci95':None,'bar':'>=0.75'},
    {'name':'fresh_rank1_vs_original_cosine','estimate':s['rank1_vs_original']['cosine'],'ci95':None,'bar':'>=0.98'},
    {'name':'fresh_rank1_vs_original_relative_l2','estimate':s['rank1_vs_original']['relative_l2_error'],'ci95':None,'bar':'<=0.20'},
    {'name':'fresh_rank1_vs_native_cosine','estimate':s['rank1_vs_native']['cosine'],'ci95':None,'bar':'>=0.75'},
    {'name':'fresh_rank1_vs_native_sign','estimate':s['rank1_vs_native']['sign_agreement'],'ci95':None,'bar':'>=0.75'},
    {'name':'fresh_worst_template_cosine','estimate':min(x['cosine'] for x in s['template_rank1_vs_native'].values()),'ci95':None,'bar':'>=0.65'}],
   'prereg_artifact_id':'task14_rank1_fresh_prereg_v1','result_artifact_id':'task14_rank1_fresh_result_v1','input_artifact_ids':list(ARTIFACTS),
   'seed':None,'checkpoint_sha256':result['checkpoint_weights_sha256'],'supersedes_event_id':None,'replicates_event_id':'task14.mlp6_7.direction_cardinality_rank1_compression.held.v1',
   'sections':['basis_aligned/polynomial_causal/SUBJECT_NUMBER_RANK1_FRESH_CONFIRMATION_V1_PREREGISTRATION.md'],
   'notes':'Fresh authority used two unseen constructions and 16 unseen one-token noun pairs. Rank-one coordinates were unchanged; no refit, gain, rank sweep, or quantization.'}
  record['evidence_events'].append(bind(record,event)); validate_v2(record); _atomic_json(path,record)
 rebuild_registry_v2(); final=json.loads(path.read_text()); validate_v2(final); print(json.dumps({'status':'registered','claim_id':NEW,'event_id':EVENT},indent=2))
if __name__=='__main__': main()
