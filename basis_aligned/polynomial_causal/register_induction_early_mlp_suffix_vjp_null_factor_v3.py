#!/usr/bin/env python3
"""Register the induction suffix-VJP-null factorization boundary."""
from copy import deepcopy
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; BQ=REPO/'basis_aligned/bilinear_quotient'; sys.path.insert(0,str(BQ))
from circuit_registry_v2 import _atomic_json,_lock,circuit_path,design_key,execution_key,file_sha256,rebuild_registry_v2,validate_v2
TAG='task.induction_selector_payload'; OLD='induction_selector_and_payload.v14'; NEW='induction_selector_and_payload.v15'; EVENT='induction.early_mlp_suffix_vjp_null_factor.v3.complete.null'
ARTIFACTS={
 'induction_suffix_vjp_v1_prereg':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_SUFFIX_VJP_NULL_FACTOR_V1_PREREGISTRATION.md','preregistration'),
 'induction_suffix_vjp_v1_binding':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_SUFFIX_VJP_NULL_FACTOR_V1_BINDING.json','binding'),
 'induction_suffix_vjp_v1_runner':('basis_aligned/bilinear_quotient/ops/run_induction_early_mlp_suffix_vjp_null_factor_v1.py','runner'),
 'induction_suffix_vjp_v2_prereg':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_SUFFIX_VJP_NULL_FACTOR_V2_PREREGISTRATION.md','preregistration'),
 'induction_suffix_vjp_v2_binding':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_SUFFIX_VJP_NULL_FACTOR_V2_BINDING.json','binding'),
 'induction_suffix_vjp_v2_runner':('basis_aligned/bilinear_quotient/ops/run_induction_early_mlp_suffix_vjp_null_factor_v2.py','runner'),
 'induction_suffix_vjp_answer_audit':('basis_aligned/polynomial_causal/INDUCTION_SUFFIX_VJP_ANSWER_BINDING_AUDIT_V1.json','audit'),
 'induction_suffix_vjp_v3_prereg':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_SUFFIX_VJP_NULL_FACTOR_V3_PREREGISTRATION.md','preregistration'),
 'induction_suffix_vjp_v3_binding':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_SUFFIX_VJP_NULL_FACTOR_V3_BINDING.json','binding'),
 'induction_suffix_vjp_v3_runner':('basis_aligned/bilinear_quotient/ops/run_induction_early_mlp_suffix_vjp_null_factor_v3.py','runner'),
 'induction_suffix_vjp_v3_result':('basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_SUFFIX_VJP_NULL_FACTOR_V3_RESULT.json','result')}
def art(p,k): return {'path':p,'sha256':file_sha256(REPO/p),'kind':k,'status':'frozen'}
def main():
 result=json.loads((HERE/'INDUCTION_EARLY_MLP_SUFFIX_VJP_NULL_FACTOR_V3_RESULT.json').read_text()); assert result['terminal']=='suffix_vjp_null_factor_null'; assert result['predictions']=={'pred_a_exact_instrument':True,'pred_b_suffix_null_answer_preserving':False,'pred_c_suffix_null_removes_collateral':True,'pred_d_vjp_null_partition':True}; assert result['price']['forwards']==18 and result['price']['sequences']==576 and result['price']['backwards']==3
 path=circuit_path(TAG); current=json.loads(path.read_text())
 if any(e['event_id']==EVENT for e in current['evidence_events']): validate_v2(current); rebuild_registry_v2(); print('already registered'); return
 with _lock('registry'):
  record=json.loads(path.read_text()); previous=next(c for c in record['claims'] if c['claim_id']==OLD)
  for aid,spec in ARTIFACTS.items(): record['artifacts'][aid]=art(*spec)
  claim=deepcopy(previous); claim.update({'claim_id':NEW,'revision':15,'supersedes':OLD,'evidence_event_ids':[*previous['evidence_event_ids'],EVENT],
   'next_missing':'Both direct LM-head and exact local suffix-CE-VJP one-axis splits of the MLP8-12 response are null. The suffix-null remainder removes 26.8-49.2% vocabulary RMS in every opened cell but still incurs +.266 to +.705 CE damage in failing discovery cells and +.277 to +.478 in failing confirmation cells. Do not tune axes, groups, or bars. Reopen induction only with independently derived finite-amplitude nonlinear consumer coordinates; otherwise move circuits.'})
  site={'site_id':'early_mlp8_12_suffix_ce_vjp_factor','tensor_path':'native-minus-edited MLP8-12 writes split by local recipient-CE VJP through the native suffix','shape':['batch',5,1152],'intervention':'restore scalar VJP projection or its exact orthogonal remainder','ceiling_event_ids':[]}; claim['candidate_sites'].append(site); record['claims'].append(claim)
  reductions=[v for split in result['decisions'].values() for v in split['reductions'].values()]; positives=[v['mean_ce_damage'] for split in result['reports'].values() for v in split['null'].values() if v['mean_ce_damage']>0]
  event={'event_id':EVENT,'claim_id':NEW,'test_type':'compiled_equivalence','stage':'complete','verdict':'null','failure_kind':'scientific_null','family_ids':['selector_payload_joint_answer_preserved'],'site_id':site['site_id'],'split_plan_id':claim['split_plan_ids'][0],'evaluation_role':'opened_row_suffix_sensitivity_factorization','metrics':[
   {'name':'partition_max_absolute_error','estimate':result['instrument']['maximum_partition_error'],'ci95':None,'bar':'<=1e-5'},
   {'name':'suffix_null_vocab_rms_reduction_min','estimate':min(reductions),'ci95':None,'bar':'>=0.25 each cell'},
   {'name':'suffix_null_vocab_rms_reduction_max','estimate':max(reductions),'ci95':None,'bar':'descriptive'},
   {'name':'minimum_positive_suffix_null_CE_damage','estimate':min(positives),'ci95':None,'bar':'<=0.10 each cell'},
   {'name':'maximum_positive_suffix_null_CE_damage','estimate':max(positives),'ci95':None,'bar':'descriptive'},
   {'name':'mean_suffix_sensitive_component_norm','estimate':result['component_norms']['parallel']['mean_installed_norm'],'ci95':None,'bar':'descriptive'},
   {'name':'mean_suffix_null_component_norm','estimate':result['component_norms']['null']['mean_installed_norm'],'ci95':None,'bar':'descriptive'}],
   'prereg_artifact_id':'induction_suffix_vjp_v3_prereg','result_artifact_id':'induction_suffix_vjp_v3_result','input_artifact_ids':list(ARTIFACTS),'seed':None,'checkpoint_sha256':result['checkpoint_sha256'],'supersedes_event_id':None,'replicates_event_id':None,'sections':['basis_aligned/polynomial_causal/INDUCTION_EARLY_MLP_SUFFIX_VJP_NULL_FACTOR_V3_PREREGISTRATION.md'],'notes':'V1/V2 were runtime-invalid and serialized no result. V3 used exact local reverse-mode CE sensitivity; no fitting, gain, rank sweep, weight update, or quantization.'}
  event['design_key']=design_key(record,event); event['execution_key']=execution_key(record,event); record['evidence_events'].append(event); validate_v2(record); _atomic_json(path,record)
 rebuild_registry_v2(); final=json.loads(path.read_text()); validate_v2(final); print(json.dumps({'status':'registered','claim_id':NEW,'event_id':EVENT},indent=2))
if __name__=='__main__': main()
