#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_fixed pred_c_learned
"""Frozen finite-response fixed/learned programs, same opened48docs and source donors.
Both native replays<1e-5. Per-arm adoption screen requires all fullpath/mode
errors<.10 and natural effecterror<.10, CEadded<.02 in bothdomains.
No native-outcome program selection; explicit prior failures retained.
Same3686products14,067,072coefficients,96captures total, no fitting.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(captures=96,arms=['finite_fixed','finite_learned'],fit=False)));return
 sys.path.insert(0,str(P));from full_layer_candidate_evaluation import evaluate
 start=time.monotonic();rows=[]
 for name in ('FINITE_FIXED','FINITE_LEARNED'):
  output=f'FULL_QUADRATIC_{name}_NATIVE_V1.json';evaluate(f'FULL_QUADRATIC_{name}_PROGRAM_V1.pt',output);result=json.loads((P/output).read_text());rows.append(dict(arm=name.lower(),predictions=result['predictions'],natural_summary=result['natural_summary'],primary=[r for r in result['summary'] if r['family']=='same_cohort']))
 passes=lambda r:all(r['predictions'].values())
 pred=dict(pred_a_instrument=all(r['predictions']['pred_a_replay'] for r in rows),pred_b_fixed=passes(rows[0]),pred_c_learned=passes(rows[1]))
 (P/'FULL_QUADRATIC_FINITE_RESPONSE_NATIVE_V1.json').write_text(json.dumps(dict(predictions=pred,records=rows,seconds=time.monotonic()-start,scope='Both finite-response fitting alternatives tested under identical opened-panel rules; no selection/adoption from training improvements alone.'),indent=2)+'\n');print(json.dumps(dict(predictions=pred)),flush=True)
if __name__=='__main__':main()
