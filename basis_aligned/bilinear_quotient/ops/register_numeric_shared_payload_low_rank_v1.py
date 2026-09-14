#!/usr/bin/env python3
"""Register the audited cross-authority numeric cached-basis null."""
import copy,hashlib,json
from pathlib import Path
import circuit_registry_v2 as registry
BQ=Path(__file__).resolve().parents[1];REPO=BQ.parents[1];CK='680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3'
FILES={'numeric_shared_low_rank_prereg_v1':('basis_aligned/polynomial_causal/NUMERIC_SHARED_PAYLOAD_LOW_RANK_V1_PREREGISTRATION.md','b91630b900887d7e82fd059914502969040f32fd06f4f19ed4bd8bf55e3c9941','preregistration'),'numeric_shared_low_rank_result_v1':('basis_aligned/polynomial_causal/NUMERIC_SHARED_PAYLOAD_LOW_RANK_V1_RESULT.json','7ff0352b075aef8afe1cd3c75671b521cc431dd7f39daf7c329095071be40857','screen_result'),'numeric_shared_low_rank_audit_v1':('basis_aligned/polynomial_causal/NUMERIC_SHARED_PAYLOAD_LOW_RANK_V1_AUDIT.json','0f301409eafd119a0afb99c483892b2746998e62ab01dfbfc894a9855368800d','postexecution_audit')}
TASKS=[('task_numbered_list_index_successor.json','numbered_list_index_successor.v12','numbered_list_index_successor.v13','numbered_list_index_successor','numeric_successor_spaced_ood_v1'),('task_numeric_sequence_continuation.json','numeric_sequence_continuation.v10','numeric_sequence_continuation.v11','numeric_sequence_continuation','numeric_sequence_nonadjacent_ood_v1')]
def m(n,e,b):return {'name':n,'estimate':e,'ci95':None,'bar':b}
def build(path,old,new,stem,split):
 r=json.loads(path.read_text())
 if r['claims'][-1]['claim_id']!=old:raise ValueError('moved '+old)
 for k,(p,h,t) in FILES.items():
  if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h:raise ValueError('drift '+k)
  r['artifacts'][k]={'path':p,'sha256':h,'kind':t,'status':'frozen'}
 c=copy.deepcopy(r['claims'][-1]);c.update(claim_id=new,revision=int(new.rsplit('v',1)[1]),supersedes=old,status='weights_translated');eid=stem+'.shared_payload_low_rank.v1.complete.null';c['evidence_event_ids']+=[eid];c['next_missing']='exact L8H3+H7 cached payload is shared functionally, but a rank-two list basis transfers neither to +1 nor copy (only 9.9% test delta energy); cached directions are context/representation specific. Do not fit a broader post-outcome union basis. Move to an independently defined circuit or consumer decomposition.';r['claims'].append(c)
 fam=[x['family_id'] for x in c['counterfactual_families']]
 e={'event_id':eid,'test_type':'cross_family_transfer','stage':'complete','verdict':'null','failure_kind':'scientific_null','site_id':'final_label_l0_value_through_l8h3_h7','result_artifact_id':'numeric_shared_low_rank_result_v1','prereg_artifact_id':'numeric_shared_low_rank_prereg_v1','metrics':[m('selected_list_FIT_rank',2,'smallest rank with >=.80 recovery'),m('list_rank2_margin_recovery_range',[.998,1.004],'>=.80'),m('list_rank2_CE_recovery_range',[.992,1.011],'>=.80'),m('cross_authority_projection_energy',.099,'descriptive'),m('plus1_margin_recovery_range',[.050,.192],'>=.80 each cell'),m('plus1_CE_recovery_range',[.092,.259],'>=.80 each cell'),m('copy_margin_recovery_range',[-.070,.099],'>=.80 each cell'),m('copy_CE_recovery_range',[-.034,.097],'>=.80 each cell'),m('minimum_test_donorward_fraction',.5,'>=.75'),m('audited_actual_sequences',736,'must equal preregistered 736')],'supersedes_event_id':None,'notes':{'terminal':'cross_authority_null','scientific_status':'valid; independent CPU audit corrects result observed_sequences metadata from 576 to actual 736','interpretation':'same exact cached-term interface does not mean a common low-rank cached vector basis across list, +1, and copy contexts','audit_artifact_id':'numeric_shared_low_rank_audit_v1'},'claim_id':new,'family_ids':fam,'evaluation_role':'frozen FIT screen','input_artifact_ids':['numeric_shared_low_rank_audit_v1'],'split_plan_id':split,'seed':None,'checkpoint_sha256':CK,'replicates_event_id':None,'sections':[]};e['design_key']=registry.design_key(r,e);e['execution_key']=registry.execution_key(r,e);r['evidence_events'].append(e);registry.validate_v2(r);return r
def main():
 vals=[]
 for fn,old,new,stem,split in TASKS:
  p=BQ/'circuits'/fn;vals.append((p,old,build(p,old,new,stem,split)))
 with registry._lock('registry'):
  for p,old,v in vals:
   if json.loads(p.read_text())['claims'][-1]['claim_id']!=old:raise ValueError('moved during publication')
  for p,old,v in vals:registry._atomic_json(p,v)
 registry.rebuild_registry_v2();print(json.dumps({'written':[str(p.relative_to(REPO)) for p,_,_ in vals],'gpu_used':False}))
if __name__=='__main__':main()
