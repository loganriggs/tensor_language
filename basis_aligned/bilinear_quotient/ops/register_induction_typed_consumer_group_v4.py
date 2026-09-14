#!/usr/bin/env python3
"""Register typed multi-module induction consumer null and correction trail."""
import copy,hashlib,json
from pathlib import Path
import circuit_registry_v2 as registry
BQ=Path(__file__).resolve().parents[1];REPO=BQ.parents[1];PATH=BQ/'circuits/task_induction_selector_payload.json';CK='680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3'
HASH=[('95df0732d2b97d50007185ed7965fb0fddfa4d76de0e77812a4fd05db545b5a1','344245ce35dd41d3d973556a71040bd4884cd7d4c4a59fa34f9f883232ceb27e','b7a9688bd07a47eb87670082d31093b344bd358c98b455242bd923bcd4b4e430'),('4bbaf0ccb4ff2a020146f8fccfccba017773529563d4ba8951445b17f84bda07','136ede92d00a56120900bd67eb1c5c731d036ffc7d104a889bfa465285378368','840f4e400d57da289c4e5e34dbb84891a202b12e447963b35bf83db12d79e18f'),('3bdbf0bbadeb94ce4cdb49d07fdc4477c2d0ebc6e4eb7a56cb67061e0b911a3e','9f7e08af074430c07ff70be2a8b474767888579d2812597d11c4e2af3188eb23','de2614f17fd7c2b8128d55f398346ff70610af3b8a0b40d1b6ac4d9cb72549d9'),('9ea58527371059f25863381d7169a553f37d8e9a3ad86f2cd0b2e23f5acf886d','c968e6b4fdd79ffe9c65b3a5c610fe3bb9eb5d4896a28a7a6ebefb0712194fdf','c2e22db3f26862bd22fd6efb7cf4a76fbc935d3a26b7087184aef0bafe722f8c')]
def m(n,e,b):return {'name':n,'estimate':e,'ci95':None,'bar':b}
def ev(r,eid,stage,verdict,failure,v,metrics,notes,claim):
 e={'event_id':eid,'test_type':'compiled_equivalence','stage':stage,'verdict':verdict,'failure_kind':failure,'site_id':None,'result_artifact_id':f'typed_consumer_v{v}_result','prereg_artifact_id':f'typed_consumer_v{v}_prereg','metrics':metrics,'supersedes_event_id':None,'notes':notes,'claim_id':claim,'family_ids':[x['family_id'] for x in r['claims'][-1]['counterfactual_families']],'evaluation_role':'frozen FIT screen','input_artifact_ids':['contextual_consumer_rows',f'typed_consumer_v{v}_binding'],'split_plan_id':r['claims'][-1]['split_plan_ids'][0],'seed':None,'checkpoint_sha256':CK,'replicates_event_id':None,'sections':[]};e['design_key']=registry.design_key(r,e);e['execution_key']=registry.execution_key(r,e);return e
def build():
 r=json.loads(PATH.read_text())
 if r['claims'][-1]['claim_id']!='induction_selector_and_payload.v12':raise ValueError('moved')
 for i,(hp,hb,hr) in enumerate(HASH,1):
  for suffix,h,kind in [('prereg',hp,'preregistration'),('binding',hb,'binding'),('result',hr,'screen_result')]:
   p=f'basis_aligned/polynomial_causal/INDUCTION_TYPED_CONSUMER_GROUP_RESPONSE_V{i}_{"PREREGISTRATION.md" if suffix=="prereg" else "BINDING.json" if suffix=="binding" else "RESULT.json"}'
   if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=h:raise ValueError('drift '+p)
   r['artifacts'][f'typed_consumer_v{i}_{suffix}']={'path':p,'sha256':h,'kind':kind,'status':'frozen'}
 c=copy.deepcopy(r['claims'][-1]);c.update(claim_id='induction_selector_and_payload.v13',revision=13,supersedes='induction_selector_and_payload.v12');ids=[f'induction.typed_consumer_group.v{i}.invalid' for i in range(1,4)]+['induction.typed_consumer_group.v4.complete.null'];c['evidence_event_ids']+=ids;c['next_missing']='singleton and architecture-defined multi-module native response restoration are both null. Early MLP8-12 removes 37.9-49.9% of vocabulary response, but necessarily causes >.10 answer CE damage in at least one cell; no typed group is an eligible mediator. Do not tune groups or bars. A future induction continuation needs a non-oracle factorization of answer-preserving versus collateral response, derived independently; otherwise move circuits.';r['claims'].append(c)
 r['evidence_events'].append(ev(r,ids[0],'invalid','invalid','invalid_instrument',1,[m('actual_forwards',33,'registered 30'),m('actual_sequences',1056,'registered 960')],{'reason':'native capture omitted from price'},c['claim_id']))
 r['evidence_events'].append(ev(r,ids[1],'invalid','invalid','invalid_instrument',2,[m('actual_forwards',33,'registered 33')],{'reason':'instrument predicate retained stale 66/2112 literal'},c['claim_id']))
 r['evidence_events'].append(ev(r,ids[2],'invalid','invalid','invalid_instrument',3,[m('runtime_instrument',1,'passed')],{'reason':'result omitted per-candidate CE/choice reports needed to reproduce no-eligible terminal; not registered as scientific authority'},c['claim_id']))
 r['evidence_events'].append(ev(r,ids[3],'complete','null','scientific_null',4,[m('eligible_DISCOVERY_groups',0,'>=1 to open CONFIRM'),m('early_mlp_vocab_RMS_reduction_range',[.379,.499],'>=.25'),m('early_all_vocab_RMS_reduction_range',[.420,.532],'>=.25'),m('all_mlp_vocab_RMS_reduction_range',[.313,.644],'>=.25'),m('minimum_positive_CE_damage_early_mlp',.194,'<=.10 every cell'),m('minimum_positive_CE_damage_early_all',.149,'<=.10 every cell'),m('actual_forwards',33,'must equal 33'),m('actual_sequences',1056,'must equal 1056')],{'terminal':'complete','interpretation':'response is concentrated in early MLPs but cannot be restored as complete native module writes without removing needed answer computation','confirm':'sealed because no discovery group eligible'},c['claim_id']))
 registry.validate_v2(r);return r
def main():
 v=build()
 with registry._lock('registry'):
  if json.loads(PATH.read_text())['claims'][-1]['claim_id']!='induction_selector_and_payload.v12':raise ValueError('moved')
  registry._atomic_json(PATH,v)
 registry.rebuild_registry_v2();print(json.dumps({'written':str(PATH.relative_to(REPO)),'gpu_used':False}))
if __name__=='__main__':main()
