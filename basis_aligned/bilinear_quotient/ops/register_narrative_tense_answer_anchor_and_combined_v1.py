#!/usr/bin/env python3
"""Register answer-anchor null and combined-authority capability license."""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import circuit_registry_v2 as registry

BQ=Path(__file__).resolve().parents[1]; REPO=BQ.parents[1]
PATH=BQ/'circuits/task_narrative_tense_past_vs_present.json'
CHECKPOINT='680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3'
FILES={
 'answer_anchor_authority_v1':('basis_aligned/bilinear_quotient/ops/circuit_fast_screen_candidate_narrative_tense_answer_anchor_authority.py','75a357e09bd3b00c22b21e42c42f18829a17538716d03975ecfc747a105e2bb0','dataset_authority'),
 'answer_anchor_prereg_v1':('basis_aligned/bilinear_quotient/circuits/prior_art/NARRATIVE_TENSE_ANSWER_ANCHOR_NATIVE_CAPABILITY_V1_PREREGISTRATION.md','909d9a5c9449c2d36e6e88a25a2b28b83e6414dd72ab5c7c9541047a1cc8d4d1','preregistration'),
 'answer_anchor_result_v1':('basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_answer_anchor_native_capability_v1_result.json','4781c70fce70c6a9cdbae36bcf625f4406dea30f8873759c060d295394158f40','screen_result'),
 'combined_authority_v1':('basis_aligned/bilinear_quotient/ops/circuit_fast_screen_candidate_narrative_tense_combined_authority.py','dea01f20355999af3104a54c701a6d0ee15c3192990f6fe448ed897eb0a70f12','dataset_authority'),
 'combined_prereg_v1':('basis_aligned/bilinear_quotient/circuits/prior_art/NARRATIVE_TENSE_COMBINED_NATIVE_CAPABILITY_V1_PREREGISTRATION.md','56355b0231074cb3739a2f87ba16c4e88d5c6615f7825cdef26c9f3f86b22300','preregistration'),
 'combined_result_v1':('basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_combined_native_capability_v1_result.json','f3eadd7040ca8cf884df7aa46b80fc3026b42ff3f172af78711d29b2e5825dd1','screen_result'),
 'combined_capability_v1':('basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_combined_native_capability_v1_capability.json','5c1bd1e19130c17e2ce4f9b761809cf1788fd5afb32351f7edb264451f84c714','capability_result'),
 'combined_license_v1':('basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_combined_native_capability_v1_license.json','2805e7e49bdc1bb6d63a9e61758c5598474980eccdfc6f3ccc10015d331eb040','capability_license'),
}
def m(n,e,b): return {'name':n,'estimate':e,'ci95':None,'bar':b}
def add(record, revision, event_id, verdict, failure, result, prereg, metrics, notes, inputs, next_missing):
 c=copy.deepcopy(record['claims'][-1]);c.update(claim_id=f'narrative_tense_at_final_position.v{revision}',revision=revision,supersedes=record['claims'][-1]['claim_id'])
 c['evidence_event_ids']=c['evidence_event_ids']+[event_id];c['next_missing']=next_missing;record['claims'].append(c)
 e={'event_id':event_id,'test_type':'capability','stage':'complete','verdict':verdict,'failure_kind':failure,'site_id':None,'result_artifact_id':result,'prereg_artifact_id':prereg,'metrics':metrics,'supersedes_event_id':None,'notes':notes,'claim_id':c['claim_id'],'family_ids':['a1_direct_narration','a2_relative_clause','p_surface_rewrite','c_same_answer_rewrite'],'evaluation_role':'frozen FIT screen','input_artifact_ids':inputs,'split_plan_id':'narrative_tense_fit_v1','seed':None,'checkpoint_sha256':CHECKPOINT,'replicates_event_id':None,'sections':[]}
 e['design_key']=registry.design_key(record,e);e['execution_key']=registry.execution_key(record,e);record['evidence_events'].append(e)
def build():
 r=json.loads(PATH.read_text())
 if r['claims'][-1]['claim_id']!='narrative_tense_at_final_position.v7': raise ValueError('authority moved')
 for k,(p,h,t) in FILES.items():
  if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h: raise ValueError('artifact drift '+k)
  r['artifacts'][k]={'path':p,'sha256':h,'kind':t,'status':'frozen'}
 add(r,8,'narrative_tense.answer_anchor_native_capability.v1.complete.null','null','scientific_null','answer_anchor_result_v1','answer_anchor_prereg_v1',[m('minimum_FIT_A1_cell_accuracy',.75,'>=.875'),m('minimum_FIT_A2_cell_accuracy',1.0,'>=.875'),m('holdout_opened',0,'requires all FIT cells'),m('license_emitted',0,'must equal 1')],{'localization':'answer-clause anchor repaired A2 to 4/4 in every cell; alternate A1 wording and past controls failed','holdout':'sealed'},['answer_anchor_authority_v1'],'combine the independently passing direct A1 and answer-anchored A2 on fresh lexical material; causal work remains blocked until a license issues')
 add(r,9,'narrative_tense.combined_native_capability.v1.held','held',None,'combined_result_v1','combined_prereg_v1',[m('FIT_cell_pass_count',16,'must equal 16'),m('HOLDOUT_cell_pass_count',16,'must equal 16'),m('minimum_cell_accuracy',1.0,'>=.875'),m('endpoint_accuracy',1.0,'128/128'),m('license_emitted',1,'must equal 1')],{'scientific_scope':'native capability for the exact outcome-derived construction package','license_sha256':FILES['combined_license_v1'][1],'controls':'fixed-present modifier rewrites; does not license arbitrary past controls'},['combined_authority_v1','combined_capability_v1','combined_license_v1'],'run an exact license-bound L11H3 carrier confirmation on the combined authority; interpret selectivity only against its fixed-present P/C controls')
 registry.validate_v2(r);return r
def main():
 v=build()
 with registry._lock('registry'):
  if json.loads(PATH.read_text())['claims'][-1]['claim_id']!='narrative_tense_at_final_position.v7':raise ValueError('authority moved during publication')
  registry._atomic_json(PATH,v)
 registry.rebuild_registry_v2();print(json.dumps({'written':str(PATH.relative_to(REPO)),'gpu_used':False}))
if __name__=='__main__':main()
