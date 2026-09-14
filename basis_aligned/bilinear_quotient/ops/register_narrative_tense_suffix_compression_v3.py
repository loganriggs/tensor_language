#!/usr/bin/env python3
"""Register two invalid suffix instruments and the valid V3 compression null."""
import copy,hashlib,json
from pathlib import Path
import circuit_registry_v2 as registry
BQ=Path(__file__).resolve().parents[1];REPO=BQ.parents[1];PATH=BQ/'circuits/task_narrative_tense_past_vs_present.json';CK='680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3'
H={'v1p':('basis_aligned/bilinear_quotient/circuits/prior_art/NARRATIVE_TENSE_L11H3_SUFFIX_COMPRESSION_V1_PREREGISTRATION.md','2cc91706cf33e087423905cf5c39c68a81ded58b23a06b6aaac352a23f148067','preregistration'),'v1r':('basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_l11h3_suffix_compression_v1_result.json','4078e6131e371796e6e32275544875a80ea28e1ce120f5ed760321c695b49894','screen_result'),'v2p':('basis_aligned/bilinear_quotient/circuits/prior_art/NARRATIVE_TENSE_L11H3_SUFFIX_COMPRESSION_V2_PREREGISTRATION.md','25ad229b15ec2941b7b51da6d5f22ef36a46efdb44f27ef0ac7576406f1c76fb','preregistration'),'v2r':('basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_l11h3_suffix_compression_v2_result.json','2989bf61708105610f908986cf20ad26a0212f7f3f8e89bfbf9de3bf9e923242','screen_result'),'v3p':('basis_aligned/bilinear_quotient/circuits/prior_art/NARRATIVE_TENSE_L11H3_SUFFIX_COMPRESSION_V3_PREREGISTRATION.md','381fb13cc87f81e6aa5efa6e46ef71995eba50cf16e1cf5dff6001efab0f5e32','preregistration'),'v3r':('basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_l11h3_suffix_compression_v3_result.json','0cc9935194488f6e7a84a7ab326637493cecb060356582672eadf90ab772c53b','screen_result')}
def m(n,e,b):return {'name':n,'estimate':e,'ci95':None,'bar':b}
def event(r,eid,stage,verdict,failure,res,pre,metrics,notes,claim):
 e={'event_id':eid,'test_type':'compiled_equivalence','stage':stage,'verdict':verdict,'failure_kind':failure,'site_id':'attention.block11.head3.pre_output_projection.final_position','result_artifact_id':res,'prereg_artifact_id':pre,'metrics':metrics,'supersedes_event_id':None,'notes':notes,'claim_id':claim,'family_ids':['a1_direct_narration','a2_relative_clause','p_surface_rewrite','c_same_answer_rewrite'],'evaluation_role':'frozen FIT screen','input_artifact_ids':['combined_authority_v1','combined_license_v1'],'split_plan_id':'narrative_tense_fit_v1','seed':None,'checkpoint_sha256':CK,'replicates_event_id':None,'sections':[]};e['design_key']=registry.design_key(r,e);e['execution_key']=registry.execution_key(r,e);return e
def build():
 r=json.loads(PATH.read_text())
 if r['claims'][-1]['claim_id']!='narrative_tense_at_final_position.v10':raise ValueError('moved')
 for k,(p,h,t) in H.items():
  if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h:raise ValueError('drift '+k)
  r['artifacts']['suffix_compression_'+k]={'path':p,'sha256':h,'kind':t,'status':'frozen'}
 c=copy.deepcopy(r['claims'][-1]);c.update(claim_id='narrative_tense_at_final_position.v11',revision=11,supersedes='narrative_tense_at_final_position.v10')
 ids=['narrative_tense.l11h3_suffix_compression.v1.invalid','narrative_tense.l11h3_suffix_compression.v2.invalid','narrative_tense.l11h3_suffix_compression.v3.complete.null'];c['evidence_event_ids']+=ids;c['next_missing']='six-token post-change value carrier is distributed: common role/of/the suffix is insufficient and construction-specific early3 is material; move to an independently defined residual-basis compression or another circuit, not more source-position slicing';r['claims'].append(c)
 r['evidence_events'].append(event(r,ids[0],'invalid','invalid','invalid_instrument','suffix_compression_v1r','suffix_compression_v1p',[m('native_reinstall_max_error',5.1975250244140625e-5,'<=5e-5')],{'reason':'generic replacement path and cross-batch native reference; no suffix evidence'},c['claim_id']))
 r['evidence_events'].append(event(r,ids[1],'invalid','invalid','invalid_instrument','suffix_compression_v2r','suffix_compression_v2p',[m('native_reinstall_max_error',5.1975250244140625e-5,'<=5e-5')],{'reason':'native mask corrected but reference remained cross-batch; no suffix evidence'},c['claim_id']))
 r['evidence_events'].append(event(r,ids[2],'complete','null','scientific_null','suffix_compression_v3r','suffix_compression_v3p',[m('native_reinstall_max_error',0.0,'<=5e-5'),m('shared_suffix_margin_fraction_range',[.603,.698],'>=.85 every cell'),m('shared_suffix_CE_fraction_range',[.626,.737],'>=.85 every cell'),m('early3_margin_fraction_range',[.315,.378],'descriptive'),m('early3_CE_fraction_range',[.365,.408],'descriptive'),m('role_fraction_of_suffix_range',[.791,.882],'>=.70 every cell'),m('of_the_fraction_of_suffix_range',[.157,.235],'>=.70 every cell'),m('maximum_absolute_role_by_of_the_interaction_fraction',.039,'>=.20 every cell'),m('P_C_selectivity',1,'all candidate arms pass')],{'terminal':'shared_suffix_compression_null','interpretation':'post-change carrier is distributed across all six answer-clause prefix positions; role is strongest within the common suffix but insufficient for the registered carrier compression','exact_head_additivity_max_error':3.0517578125e-5},c['claim_id']))
 registry.validate_v2(r);return r
def main():
 v=build()
 with registry._lock('registry'):
  if json.loads(PATH.read_text())['claims'][-1]['claim_id']!='narrative_tense_at_final_position.v10':raise ValueError('moved')
  registry._atomic_json(PATH,v)
 registry.rebuild_registry_v2();print(json.dumps({'written':str(PATH.relative_to(REPO)),'gpu_used':False}))
if __name__=='__main__':main()
