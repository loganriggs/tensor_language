#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_omission pred_c_bound
"""Exact output-span oracle diagnostic; 48 captures, no fitting.
pred_a_instrument old native program checks and Pythagorean identity<1e-8.
pred_b_omission rank32 omitted output directions explain>=half context squared error both domains.
pred_c_bound oracle linear error <= actual program linear error all domains/families.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,oracle=True,fit=False)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_OUTPUT_SPAN_CONTROLS_V1.pt';native.SOURCE_CONTEXT_FAMILIES=True;native.OUTPUT_SPAN_CONTROLS=True;native.OUTPUT_NAME='MIDPOINT_OUTPUT_SPAN_NATIVE_RAW_V1.json';native.main()
 r=json.loads((P/native.OUTPUT_NAME).read_text());rows=[];checks=[]
 for d in ['fineweb','code']:
  for f in ['source_only','context_only']:
   for name in ['baseline','rank8','rank32']:
    rr=[v for v in r['records'] if v['domain']==d and v['family']==f and v['candidate']=='span_'+name]
    sums={k:sum(v[k] for v in rr) for k in ['linear_reference_energy','linear_error_energy','in_span_error_energy','program_error_energy']};checks.extend(v['pythagorean_relative_error'] for v in rr)
    rows.append(dict(domain=d,family=f,program=name,omitted_squared_error_fraction=sums['linear_error_energy']/sums['program_error_energy'],linear_output_floor=(sums['linear_error_energy']/sums['linear_reference_energy'])**.5,linear_program_error=(sums['program_error_energy']/sums['linear_reference_energy'])**.5))
 pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'] and max(checks)<1e-8,pred_b_omission=all(v['omitted_squared_error_fraction']>=.5 for v in rows if v['program']=='rank32' and v['family']=='context_only'),pred_c_bound=all(v['linear_output_floor']<=v['linear_program_error']+1e-10 for v in rows))
 out=P/'MIDPOINT_OUTPUT_SPAN_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,records=rows,pythagorean=max(checks),scope='Exact native output-span projection oracle. Linear metric lower bound, not a bound after nonlinearities and not a compressed algorithm. No native states used to choose span.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
