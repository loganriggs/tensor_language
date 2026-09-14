#!/usr/bin/env python3
"""Register the global signed narrative tense scalar program."""
import copy,hashlib,json
from pathlib import Path
import circuit_registry_v2 as registry
BQ=Path(__file__).resolve().parents[1];REPO=BQ.parents[1];PATH=BQ/'circuits/task_narrative_tense_past_vs_present.json';CK='680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3'
H={'rank1_scalar_prereg_v1':('basis_aligned/bilinear_quotient/circuits/prior_art/NARRATIVE_TENSE_L11H3_RANK1_SCALAR_PROGRAM_V1_PREREGISTRATION.md','540aaf34ccffc9d8d43c3007933a3911f31cea47dfd50f412289c03779f6fad2','preregistration'),'rank1_scalar_result_v1':('basis_aligned/bilinear_quotient/circuits/fast_screens/narrative_tense_l11h3_rank1_scalar_program_v1_result.json','b8abb9abddcc4e62befd586cd1f68a4ba926ee58727d83c8fd9f6000b4008f16','screen_result')}
def m(n,e,b):return {'name':n,'estimate':e,'ci95':None,'bar':b}
def build():
 r=json.loads(PATH.read_text())
 if r['claims'][-1]['claim_id']!='narrative_tense_at_final_position.v12':raise ValueError('moved')
 for k,(p,h,t) in H.items():
  if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h:raise ValueError('drift')
  r['artifacts'][k]={'path':p,'sha256':h,'kind':t,'status':'frozen'}
 c=copy.deepcopy(r['claims'][-1]);c.update(claim_id='narrative_tense_at_final_position.v13',revision=13,supersedes='narrative_tense_at_final_position.v12',status='weights_translated');eid='narrative_tense.l11h3_rank1_scalar_program.v1.held';c['evidence_event_ids']+=[eid];c['next_missing']='the six-source L11H3 value interaction compresses to one signed residual axis plus one global magnitude; sign alone selects past-to-present versus present-to-past and negation reverses every heldout row. Native state generation and downstream decoding remain external; move to another circuit rather than refitting amplitude.';r['claims'].append(c)
 e={'event_id':eid,'test_type':'composition','stage':'complete','verdict':'held','failure_kind':None,'site_id':'attention.block11.head3.pre_output_projection.final_position','result_artifact_id':'rank1_scalar_result_v1','prereg_artifact_id':'rank1_scalar_prereg_v1','metrics':[m('FIT_direction_sign_agreement',1.0,'must equal 1 each direction'),m('FIT_direction_means',[677.361,-731.770],'opposite signs'),m('global_absolute_amplitude',704.566,'frozen FIT mean absolute coefficient'),m('HOLDOUT_global_margin_recovery_range',[1.009,1.729],'>=.80 each cell'),m('HOLDOUT_global_CE_recovery_range',[1.023,1.587],'>=.80 each cell'),m('HOLDOUT_donorward_fraction',1.0,'>=.75'),m('wrong_sign_anti_donor_fraction',1.0,'>=.75'),m('native_reinstall_max_error',0.0,'<=5e-5')],'supersedes_event_id':None,'notes':{'terminal':'global_signed_scalar_program','program':'one rank-one 128D axis, one global positive magnitude, and direction sign','construction_conditioning':'unnecessary; pooled direction and global-signed programs both pass','scope':'licensed A1/A2 target manipulation; endogenous P/C selectivity established in v12'},'claim_id':c['claim_id'],'family_ids':['a1_direct_narration','a2_relative_clause'],'evaluation_role':'frozen FIT screen','input_artifact_ids':['combined_authority_v1','combined_license_v1'],'split_plan_id':'narrative_tense_fit_v1','seed':None,'checkpoint_sha256':CK,'replicates_event_id':None,'sections':[]};e['design_key']=registry.design_key(r,e);e['execution_key']=registry.execution_key(r,e);r['evidence_events'].append(e);registry.validate_v2(r);return r
def main():
 v=build()
 with registry._lock('registry'):
  if json.loads(PATH.read_text())['claims'][-1]['claim_id']!='narrative_tense_at_final_position.v12':raise ValueError('moved')
  registry._atomic_json(PATH,v)
 registry.rebuild_registry_v2();print(json.dumps({'written':str(PATH.relative_to(REPO)),'gpu_used':False}))
if __name__=='__main__':main()
