#!/usr/bin/env python3
"""CPU audit of the corrected bracket suffix finite-response experiment."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUNNER=ROOT/'basis_aligned/bilinear_quotient/ops/run_bracket_suffix_finite_bilinear_program_v2.py'
RESULT=ROOT/'basis_aligned/polynomial_causal/BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_RESULT.json'
OUT=ROOT/'basis_aligned/polynomial_causal/BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_AUDIT.json'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 r=json.loads(RESULT.read_text());six=r['sixth'];seven=r['seventh'];source='source_only'
 checks={
  'runner_hash_bound':r['runner_sha256']==sha(RUNNER),
  'valid_terminal':r['terminal']=='suffix_source_only',
  'literal_verdict':r['predictions']=={'pred_a_exact_finite_suffix_instrument':True,'pred_b_sixth_program_selected':True,'pred_c_seventh_program_transfers':True,'pred_d_bilinear_interaction_compression':False,'pred_e_control_selectivity':True},
  'fixed_first_selection':r['selected']==source,
  'exact_instrument':six['instrument'] and seven['instrument'] and six['replay_error']<=1e-5 and seven['replay_error']<=1e-5 and max(six['maximum_closure_rse'],seven['maximum_closure_rse'])<=1e-10,
  'candidate_liveness':all(v>=.9 for panel in (six,seven) for arm,v in panel['active_fraction'].items() if arm!='exact'),
  'source_only_transfer':six['reports'][source]['passes'] and seven['reports'][source]['passes'],
  'exact_parents_live':six['parent_live'] and seven['parent_live'],
  'exact_price':r['price']['observed_forwards']==12 and r['price']['observed_sequences']==1728 and r['fits']==r['backwards']==r['updates']==0 and not r['quantized'],
 }
 a={'schema':'bracket_suffix_finite_bilinear_program_v2_audit','result_sha256':sha(RESULT),'runner_sha256':sha(RUNNER),'checks':checks,'all_checks_pass':all(checks.values()),'sixth_source_only':six['reports'][source],'seventh_source_only':seven['reports'][source],'maximum_closure_rse':max(six['maximum_closure_rse'],seven['maximum_closure_rse']),'finding':'The exact L13H8 donor source retains the bracket effect across sixth and seventh constructions even when the full source-induced MLP13-17 finite response is removed online. The suffix interaction terms are active but unnecessary under the frozen selection and transfer bars.'}
 OUT.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n');print(json.dumps(a,indent=2,sort_keys=True))

if __name__=='__main__':main()
