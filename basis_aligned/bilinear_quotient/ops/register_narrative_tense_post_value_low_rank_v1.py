#!/usr/bin/env python3
"""Register rank-one narrative interaction compression."""
import copy,hashlib,json
from pathlib import Path
import circuit_registry_v2 as registry
BQ=Path(__file__).resolve().parents[1];REPO=BQ.parents[1];PATH=BQ/'circuits/task_narrative_tense_past_vs_present.json';CK='680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3'
H={'post_value_low_rank_prereg_v1':('basis_aligned/bilinear_quotient/circuits/prior_art/NARRATIVE_TENSE_L11H3_POST_VALUE_LOW_RANK_V1_PREREGISTRATION.md','56491e8ee55b0f1861a9f4959d493daeedd45ea42e8ca90b481ed4605657c3b9','preregistration'),'post_value_low_rank_result_v1':('basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_l11h3_post_value_low_rank_v1_result.json','13cc9b128d909343bdf3ff1869c21ca4945183a31a3d7c1a239d879db2a14585','screen_result')}
def m(n,e,b):return {'name':n,'estimate':e,'ci95':None,'bar':b}
def build():
 r=json.loads(PATH.read_text())
 if r['claims'][-1]['claim_id']!='narrative_tense_at_final_position.v11':raise ValueError('moved')
 for k,(p,h,t) in H.items():
  if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h:raise ValueError('drift '+k)
  r['artifacts'][k]={'path':p,'sha256':h,'kind':t,'status':'frozen'}
 c=copy.deepcopy(r['claims'][-1]);c.update(claim_id='narrative_tense_at_final_position.v12',revision=12,supersedes='narrative_tense_at_final_position.v11',status='weights_translated');eid='narrative_tense.l11h3_post_value_low_rank.v1.held';c['evidence_event_ids']+=[eid];c['next_missing']='rank-one uncentered FIT basis prospectively transfers the exact six-source value carrier to lexical HOLDOUT; next test must independently distinguish a signed tense axis from construction or answer-token direction, using fixed-basis counterfactual scaling or cross-authority reuse rather than refitting';r['claims'].append(c)
 e={'event_id':eid,'test_type':'compiled_equivalence','stage':'complete','verdict':'held','failure_kind':None,'site_id':'attention.block11.head3.pre_output_projection.final_position','result_artifact_id':'post_value_low_rank_result_v1','prereg_artifact_id':'post_value_low_rank_prereg_v1','metrics':[m('selected_rank',1,'smallest FIT eligible in [1,2,4,8]'),m('FIT_margin_fraction_range',[.898,1.140],'>=.80 each cell'),m('FIT_CE_fraction_range',[.895,1.166],'>=.80 each cell'),m('HOLDOUT_margin_fraction_range',[.917,1.162],'>=.80 each cell'),m('HOLDOUT_CE_fraction_range',[.897,1.225],'>=.80 each cell'),m('target_donorward_fraction',1.0,'>=.75'),m('maximum_HOLDOUT_control_answer_margin_change',.027,'<=.25 x smallest target full-post'),m('native_reinstall_max_error',0.0,'<=5e-5')],'supersedes_event_id':None,'notes':{'terminal':'low_rank_1_transfer','basis':'uncentered right singular vector fit from 16 target FIT post-value deltas only','holdout':'rank frozen before groups 8:16; no fallback','interpretation':'distributed six-source L11H3 interaction occupies a prospectively reusable rank-one residual direction'},'claim_id':c['claim_id'],'family_ids':['a1_direct_narration','a2_relative_clause','p_surface_rewrite','c_same_answer_rewrite'],'evaluation_role':'frozen FIT screen','input_artifact_ids':['combined_authority_v1','combined_license_v1'],'split_plan_id':'narrative_tense_fit_v1','seed':None,'checkpoint_sha256':CK,'replicates_event_id':None,'sections':[]};e['design_key']=registry.design_key(r,e);e['execution_key']=registry.execution_key(r,e);r['evidence_events'].append(e);registry.validate_v2(r);return r
def main():
 v=build()
 with registry._lock('registry'):
  if json.loads(PATH.read_text())['claims'][-1]['claim_id']!='narrative_tense_at_final_position.v11':raise ValueError('moved')
  registry._atomic_json(PATH,v)
 registry.rebuild_registry_v2();print(json.dumps({'written':str(PATH.relative_to(REPO)),'gpu_used':False}))
if __name__=='__main__':main()
