#!/usr/bin/env python3
"""CPU audit of the oriented downstream bilinear FIT null."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUNNER=ROOT/'basis_aligned/bilinear_quotient/ops/run_numeric_downstream_oriented_bilinear_v1.py'
RESULT=ROOT/'basis_aligned/polynomial_causal/NUMERIC_DOWNSTREAM_ORIENTED_BILINEAR_V1_RESULT.json'
OUT=ROOT/'basis_aligned/polynomial_causal/NUMERIC_DOWNSTREAM_ORIENTED_BILINEAR_V1_AUDIT.json'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result=json.loads(RESULT.read_text()); reports=result['fit_reports']
    counts={name:{section:sum(bool(x['passed']) for x in report[section].values())
                  for section in ('targets','copies','action_gaps')} for name,report in reports.items()}
    checks={
      'runner_hash_bound':result['runner_sha256']==sha(RUNNER),
      'exact_three_term_closure':max(result['fit_exactness'].values())<=1e-10,
      'fit_only':result['opened_splits']==['FIT'] and not result['forbidden_splits_opened'],
      'complete_candidate_panel':len(reports)==16 and all(len(r[x])==12 for r in reports.values() for x in ('targets','copies','action_gaps')),
      'no_candidate_passed':not any(r['passed_without_nulls'] for r in reports.values()),
      'terminal_consistent':result['terminal']=='fit_null' and result['provisional'] is None and result['selected'] is None,
      'predicate_consistent':result['predictions']=={'pred_a_exact_oriented_decomposition':True,'pred_b_fit_candidate_and_active_nulls':False,'pred_c_select_replication':False,'pred_d_oriented_interaction_compression':False},
      'price_consistent':result['price']['observed_forwards']==487 and result['price']['maximum_forwards']==632,
      'no_optimization':result['fits']==0 and result['model_backwards']==0 and result['model_weights_updated'] is False,
    }
    audit={'schema':'numeric_downstream_oriented_bilinear_v1_audit','result_sha256':sha(RESULT),
           'runner_sha256':sha(RUNNER),'checks':checks,'all_checks_pass':all(checks.values()),
           'maximum_exactness_error':max(result['fit_exactness'].values()),'candidate_pass_counts':counts,
           'finding':'The native left/right orientation split is exact, but no oriented, self, or joint MLP response passes the frozen FIT action/selectivity gates; SELECT remains sealed.'}
    OUT.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print(json.dumps(audit,indent=2,sort_keys=True))


if __name__=='__main__': main()
