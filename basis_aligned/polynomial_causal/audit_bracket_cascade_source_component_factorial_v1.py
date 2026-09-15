#!/usr/bin/env python3
"""CPU audit and interaction accounting for the bracket source factorial."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUNNER=ROOT/'basis_aligned/bilinear_quotient/ops/run_bracket_cascade_source_component_factorial_v1.py';RESULT=ROOT/'basis_aligned/polynomial_causal/BRACKET_CASCADE_SOURCE_COMPONENT_FACTORIAL_V1_RESULT.json';OUT=ROOT/'basis_aligned/polynomial_causal/BRACKET_CASCADE_SOURCE_COMPONENT_FACTORIAL_V1_AUDIT.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rms(v):return math.sqrt(sum(x*x for x in v)/len(v))
def main():
 r=json.loads(RESULT.read_text());rows=[x for x in r['records'] if x['program_role']=='target'];error=[x['full_effect']-x['exact_effect'] for x in rows];den=rms(error)
 main_effects={name:rms([x[name+'_effect']-x['exact_effect'] for x in rows])/den for name in ('key1','key2','payload')}
 pairwise={
  'key1_key2':rms([x['key12_effect']-x['key1_effect']-x['key2_effect']+x['exact_effect'] for x in rows])/den,
  'key1_payload':rms([x['key1_payload_effect']-x['key1_effect']-x['payload_effect']+x['exact_effect'] for x in rows])/den,
  'key2_payload':rms([x['key2_payload_effect']-x['key2_effect']-x['payload_effect']+x['exact_effect'] for x in rows])/den,
 }
 checks={'runner_hash_bound':r['runner_sha256']==sha(RUNNER),'valid_terminal':r['terminal']=='source_factorial_partial_ceiling','literal_verdict':r['predictions']=={'pred_a_exact_instrument_and_capability':True,'pred_b_exact_joint_parent_live':True,'pred_c_factorial_corner_selected':True,'pred_d_full_donor_free_transfers':False,'pred_e_interaction_error_small':True,'pred_f_control_selectivity':True},'fixed_selection':r['selected']=='key12','all_partial_corners_pass':all(r['decisions'][x] for x in ('key12','key1_payload','key2_payload','payload','key1','key2')),'full_corner_fails':not r['decisions']['full'],'interaction_small':r['three_way_interaction_to_full_error_rms']<=.25,'exact_price':r['price']['observed_forwards']==10 and r['price']['observed_sequences']==1440 and r['fits']==r['backwards']==r['updates']==0 and not r['quantized'],'control_selectivity':all(v<=.5 for v in r['control_to_target_rms'].values())}
 a={'schema':'bracket_cascade_source_component_factorial_v1_audit','result_sha256':sha(RESULT),'runner_sha256':sha(RUNNER),'checks':checks,'all_checks_pass':all(checks.values()),'main_effect_rms_fraction_of_full_error':main_effects,'pairwise_interaction_rms_fraction_of_full_error':pairwise,'three_way_interaction_rms_fraction_of_full_error':r['three_way_interaction_to_full_error_rms'],'finding':'All partial approximation corners pass and only the fully donor-free source fails. The key2 approximation supplies the dominant single-factor error; key1, payload, pairwise interactions, and the three-way interaction are smaller. Exact payload with both approximate keys is the maximally compressed passing diagnostic ceiling.'};OUT.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n');print(json.dumps(a,indent=2,sort_keys=True))
if __name__=='__main__':main()
