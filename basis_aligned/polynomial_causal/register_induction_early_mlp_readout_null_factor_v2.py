#!/usr/bin/env python3
"""Register the induction early-MLP readout-null factorization null."""
from copy import deepcopy
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; BQ=REPO/'basis_aligned/bilinear_quotient'; sys.path.insert(0,str(BQ))
from circuit_registry_v2 import _atomic_json,_lock,circuit_path,design_key,execution_key,file_sha256,rebuild_registry_v2,validate_v2
TAG='task.induction_selector_payload'; OLD='induction_selector_and_payload.v13'; NEW='induction_selector_and_payload.v14'; EVENT='induction.early_mlp_readout_null_factor.v2.complete.null'
ARTIFACTS={
 'induction_readout_null_v1_prereg':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_READOUT_NULL_FACTOR_V1_PREREGISTRATION.md','preregistration'),
 'induction_readout_null_v1_binding':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_READOUT_NULL_FACTOR_V1_BINDING.json','binding'),
 'induction_readout_null_v1_runner':('basis_aligned/bilinear_quotient/ops/run_induction_early_mlp_readout_null_factor_v1.py','runner'),
 'induction_readout_null_v2_prereg':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_READOUT_NULL_FACTOR_V2_PREREGISTRATION.md','preregistration'),
 'induction_readout_null_v2_binding':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_READOUT_NULL_FACTOR_V2_BINDING.json','binding'),
 'induction_readout_null_v2_runner':('basis_aligned/bilinear_quotient/ops/run_induction_early_mlp_readout_null_factor_v2.py','runner'),
 'induction_readout_null_v2_result':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_READOUT_NULL_FACTOR_V2_RESULT.json','result')}
def artifact(p,k): return {'path':p,'sha256':file_sha256(REPO/p),'kind':k,'status':'frozen'}
def main():
 result=json.loads((HERE/'INDUCTION_EARLY_MLP_READOUT_NULL_FACTOR_V2_RESULT.json').read_text()); assert result['terminal']=='readout_null_factor_null'; assert result['predictions']=={'pred_a_exact_instrument':True,'pred_b_readout_null_answer_preserving':False,'pred_c_readout_null_removes_collateral':True,'pred_d_parallel_null_partition':True}; assert result['price']['forwards']==18 and result['price']['sequences']==576
 path=circuit_path(TAG); current=json.loads(path.read_text())
 if any(e['event_id']==EVENT for e in current['evidence_events']): validate_v2(current); rebuild_registry_v2(); print('already registered'); return
 with _lock('registry'):
  record=json.loads(path.read_text()); previous=next(c for c in record['claims'] if c['claim_id']==OLD)
  for aid,spec in ARTIFACTS.items(): record['artifacts'][aid]=artifact(*spec)
  claim=deepcopy(previous); claim.update({'claim_id':NEW,'revision':14,'supersedes':OLD,'evidence_event_ids':[*previous['evidence_event_ids'],EVENT],
   'next_missing':'Direct LM-head answer-axis factorization of the early MLP8-12 response is null: the exact readout-null remainder removes 31.4-50.3% of vocabulary RMS but still exceeds +.10 answer CE damage in both opened splits. Raw readout orthogonality does not survive later nonlinear consumers. Do not tune the axis, group, or bars; reopen induction only with an independently derived suffix-transported sensitivity/factorization, otherwise move circuits.'})
  site={'site_id':'early_mlp8_12_answer_parallel_null_factor','tensor_path':'native-minus-edited MLP8-12 final-query writes split by per-row LM-head answer contrast','shape':['batch',5,1152],'intervention':'restore either the exact scalar answer-parallel projection or its exact orthogonal remainder','ceiling_event_ids':[]}; claim['candidate_sites'].append(site); record['claims'].append(claim)
  reductions=[v for split in result['decisions'].values() for v in split['reductions'].values()]; positives=[v['mean_ce_damage'] for split in result['reports'].values() for v in split['null'].values() if v['mean_ce_damage']>0]
  event={'event_id':EVENT,'claim_id':NEW,'test_type':'compiled_equivalence','stage':'complete','verdict':'null','failure_kind':'scientific_null','family_ids':['selector_payload_joint_answer_preserved'],'site_id':site['site_id'],'split_plan_id':claim['split_plan_ids'][0],'evaluation_role':'opened_row_explanatory_factorization','metrics':[
   {'name':'partition_max_absolute_error','estimate':result['instrument']['maximum_partition_error'],'ci95':None,'bar':'<=1e-5'},
   {'name':'readout_null_vocab_rms_reduction_min','estimate':min(reductions),'ci95':None,'bar':'>=0.25 each cell'},
   {'name':'readout_null_vocab_rms_reduction_max','estimate':max(reductions),'ci95':None,'bar':'descriptive'},
   {'name':'minimum_positive_readout_null_CE_damage','estimate':min(positives),'ci95':None,'bar':'<=0.10 each cell'},
   {'name':'mean_answer_parallel_component_norm','estimate':result['component_norms']['parallel']['mean_installed_norm'],'ci95':None,'bar':'descriptive'},
   {'name':'mean_readout_null_component_norm','estimate':result['component_norms']['null']['mean_installed_norm'],'ci95':None,'bar':'descriptive'}],
   'prereg_artifact_id':'induction_readout_null_v2_prereg','result_artifact_id':'induction_readout_null_v2_result','input_artifact_ids':list(ARTIFACTS),'seed':None,'checkpoint_sha256':result['checkpoint_sha256'],'supersedes_event_id':None,'replicates_event_id':None,'sections':['basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_READOUT_NULL_FACTOR_V2_PREREGISTRATION.md'],'notes':'V1 emitted only a dry-run receipt and opened no model outcomes. V2 changed only dry-run detection. Factorization used no fitted basis, gradient, gain, rank sweep, or quantization.'}
  event['design_key']=design_key(record,event); event['execution_key']=execution_key(record,event); record['evidence_events'].append(event); validate_v2(record); _atomic_json(path,record)
 rebuild_registry_v2(); final=json.loads(path.read_text()); validate_v2(final); print(json.dumps({'status':'registered','claim_id':NEW,'event_id':EVENT},indent=2))
if __name__=='__main__': main()
