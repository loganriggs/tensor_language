#!/usr/bin/env python3
"""Register the licensed L11H3 post-change effective-value carrier."""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import circuit_registry_v2 as registry
BQ=Path(__file__).resolve().parents[1];REPO=BQ.parents[1]
PATH=BQ/'circuits/task_narrative_tense_past_vs_present.json'
FILES={
 'combined_carrier_prereg_v1':('basis_aligned/bilinear_quotient/circuits/prior_art/NARRATIVE_TENSE_L11H3_COMBINED_CARRIER_V1_PREREGISTRATION.md','a77ea11d5de98e0c6e7aed2a28f56d3ba013d72e1eb4d4073a14e376c3a16d73','preregistration'),
 'combined_carrier_result_v1':('basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_l11h3_combined_carrier_v1_result.json','9ffef5a3f87eed13ed40649e7748d6fabd6f0043cc732bf94cf779862b0f6e2b','screen_result')}
def m(n,e,b):return {'name':n,'estimate':e,'ci95':None,'bar':b}
def build():
 r=json.loads(PATH.read_text())
 if r['claims'][-1]['claim_id']!='narrative_tense_at_final_position.v9':raise ValueError('authority moved')
 for k,(p,h,t) in FILES.items():
  if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h:raise ValueError('drift '+k)
  r['artifacts'][k]={'path':p,'sha256':h,'kind':t,'status':'frozen'}
 c=copy.deepcopy(r['claims'][-1]);c.update(claim_id='narrative_tense_at_final_position.v10',revision=10,supersedes='narrative_tense_at_final_position.v9',status='weights_translated')
 eid='narrative_tense.l11h3_combined_carrier.v1.held'
 c['evidence_event_ids']=c['evidence_event_ids']+[eid]
 c['next_missing']='compress the licensed post-last-change L11H3 effective-value carrier by exact self-relative suffix positions and test their interaction; do not repeat whole-R carrier localization'
 r['claims'].append(c)
    e={'event_id':eid,'test_type':'compiled_equivalence','stage':'complete','verdict':'held','failure_kind':None,'site_id':'attention.block11.head3.pre_output_projection.final_position','result_artifact_id':'combined_carrier_result_v1','prereg_artifact_id':'combined_carrier_prereg_v1','metrics':[m('minimum_R_joint_margin_fraction_of_complete',.936,'>=.50'),m('minimum_R_joint_CE_fraction_of_complete',.940,'>=.50'),m('minimum_R_effective_value_margin_fraction_of_R_joint',.947,'>=.70'),m('minimum_R_effective_value_CE_fraction_of_R_joint',.928,'>=.70'),m('minimum_post_last_margin_fraction_of_R_value',.945,'>=.70'),m('minimum_post_last_CE_fraction_of_R_value',.954,'>=.70'),m('maximum_between_change_margin_fraction_of_R_value',.054,'descriptive'),m('target_donorward_fraction',1.0,'>=.75'),m('P_C_selectivity_all_arms_pass',1,'must equal 1'),m('source_sum_max_error',0.0,'<=5e-5')],'supersedes_event_id':None,'notes':{'terminal':'unchanged_carrier_value_post_last_change','authority_scope':'licensed combined A1/A2 with fixed-present P/C controls','interpretation':'L11H3 reads tense chiefly from contextual effective values on unchanged suffix tokens after the last explicit tense change'},'claim_id':c['claim_id'],'family_ids':['a1_direct_narration','a2_relative_clause','p_surface_rewrite','c_same_answer_rewrite'],'evaluation_role':'frozen FIT screen','input_artifact_ids':['combined_authority_v1','combined_capability_v1','combined_license_v1'],'split_plan_id':'narrative_tense_fit_v1','seed':None,'checkpoint_sha256':'680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3','replicates_event_id':None,'sections':[]}
 e['design_key']=registry.design_key(r,e);e['execution_key']=registry.execution_key(r,e);r['evidence_events'].append(e);registry.validate_v2(r);return r
def main():
 v=build()
 with registry._lock('registry'):
  if json.loads(PATH.read_text())['claims'][-1]['claim_id']!='narrative_tense_at_final_position.v9':raise ValueError('moved')
  registry._atomic_json(PATH,v)
 registry.rebuild_registry_v2();print(json.dumps({'written':str(PATH.relative_to(REPO)),'gpu_used':False}))
if __name__=='__main__':main()
