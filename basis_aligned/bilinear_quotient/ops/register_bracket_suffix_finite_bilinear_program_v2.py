#!/usr/bin/env python3
"""Register the source-only bracket suffix boundary."""
from copy import deepcopy
import json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];POLY=REPO/'basis_aligned/polynomial_causal';BQ=REPO/'basis_aligned/bilinear_quotient';sys.path.insert(0,str(BQ))
from circuit_registry_v2 import _atomic_json,_lock,circuit_path,design_key,execution_key,file_sha256,rebuild_registry_v2,validate_v2

TAG='task.bracket_pending_opener';OLD='pending_opener_state.v37';NEW='pending_opener_state.v38';EVENT='pending_opener.suffix_finite_bilinear_program.v2.complete.source_only'
ARTIFACTS={
 'suffix_finite_bilinear_v1_prereg':('basis_aligned/polynomial_causal/BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V1_PREREGISTRATION.md','preregistration'),
 'suffix_finite_bilinear_v1_binding':('basis_aligned/polynomial_causal/BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V1_BINDING.json','binding'),
 'suffix_finite_bilinear_v1_runner':('basis_aligned/bilinear_quotient/ops/run_bracket_suffix_finite_bilinear_program_v1.py','runner'),
 'suffix_finite_bilinear_v1_result':('basis_aligned/polynomial_causal/BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V1_RESULT.json','result'),
 'suffix_finite_bilinear_v2_prereg':('basis_aligned/polynomial_causal/BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_PREREGISTRATION.md','preregistration'),
 'suffix_finite_bilinear_v2_binding':('basis_aligned/polynomial_causal/BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_BINDING.json','binding'),
 'suffix_finite_bilinear_v2_runner':('basis_aligned/bilinear_quotient/ops/run_bracket_suffix_finite_bilinear_program_v2.py','runner'),
 'suffix_finite_bilinear_v2_result':('basis_aligned/polynomial_causal/BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_RESULT.json','result'),
 'suffix_finite_bilinear_v2_audit':('basis_aligned/polynomial_causal/BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_AUDIT.json','audit'),
}
def artifact(path,kind):return {'path':path,'sha256':file_sha256(REPO/path),'kind':kind,'status':'frozen'}
def metric(name,estimate,bar):return {'name':name,'estimate':estimate,'ci95':None,'bar':bar}

def build(path):
 result=json.loads((POLY/'BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_RESULT.json').read_text());audit=json.loads((POLY/'BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_AUDIT.json').read_text());assert result['terminal']=='suffix_source_only' and audit['all_checks_pass']
 record=json.loads(path.read_text());assert record['claims'][-1]['claim_id']==OLD
 for artifact_id,spec in ARTIFACTS.items():record['artifacts'][artifact_id]=artifact(*spec)
 previous=record['claims'][-1];claim=deepcopy(previous);claim.update({'claim_id':NEW,'revision':38,'supersedes':OLD,'evidence_event_ids':[*previous['evidence_event_ids'],EVENT],'next_missing':'Exact finite-amplitude MLP13-17 interaction coordinates are unnecessary for the bracket suffix: after exact L13H8 source replacement, removing the entire source-induced finite MLP response preserves the behavioral effect on sixth and seventh constructions. This localizes the remaining error to donor-free source generation or its direct readout calibration. Do not add or tune suffix MLP interaction terms; derive an architecture-native improvement to the rank-two source generator or readout and test it prospectively.'})
 site={'site_id':'mlp13_17.source_background_finite_response','tensor_path':'online exact MLP response from recipient-native normalized query state to exact-source-edited normalized query state','shape':['batch',5,1152],'intervention':'retain none, quadratic, left cross, right cross, or both cross terms while all attention remains live','ceiling_event_ids':[]};claim['candidate_sites'].append(site);record['claims'].append(claim)
 sixth=result['sixth']['reports']['source_only'];seventh=result['seventh']['reports']['source_only']
 event={'event_id':EVENT,'claim_id':NEW,'test_type':'cross_family_transfer','stage':'complete','verdict':'held','failure_kind':None,'family_ids':['embedded_pending_stack_top','cascade_pending_stack_top'],'site_id':site['site_id'],'split_plan_id':'pending_opener_seventh_cascade_v1','evaluation_role':'sixth_selection_then_unchanged_seventh_confirmation','metrics':[metric('sixth_source_only_relative_l2',sixth['overall']['relative_l2_error'],'<=.25 overall and <=.35 each pair'),metric('seventh_source_only_relative_l2',seventh['overall']['relative_l2_error'],'<=.25 overall and <=.35 each pair'),metric('sixth_source_only_cosine',sixth['overall']['cosine'],'>=.95 overall and >=.85 each pair'),metric('seventh_source_only_cosine',seventh['overall']['cosine'],'>=.95 overall and >=.85 each pair'),metric('sixth_control_to_target_rms',sixth['control_to_exact_target_rms'],'<=.50'),metric('seventh_control_to_target_rms',seventh['control_to_exact_target_rms'],'<=.50'),metric('maximum_FP64_closure_RSE',audit['maximum_closure_rse'],'<=1e-10'),metric('forwards',result['price']['observed_forwards'],'12')],'prereg_artifact_id':'suffix_finite_bilinear_v2_prereg','result_artifact_id':'suffix_finite_bilinear_v2_result','input_artifact_ids':list(ARTIFACTS),'seed':None,'checkpoint_sha256':result['checkpoint_sha256'],'supersedes_event_id':None,'replicates_event_id':None,'sections':['basis_aligned/polynomial_causal/BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_PREREGISTRATION.md'],'notes':'V1 was invalid only because candidate liveness included the untouched exact reference. V2 excludes that reference and leaves every causal intervention, row, bar, and price unchanged. Source-only is the fixed first passing arm. Zero fits, gradients, updates, gains, or quantization.'};event['design_key']=design_key(record,event);event['execution_key']=execution_key(record,event);record['evidence_events'].append(event);validate_v2(record);return record

def main():
 path=circuit_path(TAG);current=json.loads(path.read_text())
 if any(e['event_id']==EVENT for e in current['evidence_events']):validate_v2(current);rebuild_registry_v2();print('already registered');return
 with _lock('registry'):_atomic_json(path,build(path))
 rebuild_registry_v2();print(json.dumps({'status':'registered','claim_id':NEW,'event_id':EVENT},indent=2))
if __name__=='__main__':main()
