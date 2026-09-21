#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_alignment pred_c_increment
"""Native error composition diagnostic;48captures,no fitting.
pred_a_instrument pooled replay and additive squared-error identity<1e-8.
pred_b_alignment separate256 cross term>=10%total fullswaperror bothdomains.
pred_c_increment separate256 incremental error>=10%total fullswaperror bothdomains.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,diagnostic='error_composition',fit=False)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_ROLE_SHARED_LINEAR_GRAPHS_V1.pt';native.SOURCE_CONTEXT_FAMILIES=True;native.ERROR_REFERENCE='product_exact_linear';native.OUTPUT_NAME='MIDPOINT_ERROR_COMPOSITION_NATIVE_RAW_V1.json';native.main()
 raw=json.loads((P/native.OUTPUT_NAME).read_text());prior=json.loads((P/'MIDPOINT_ROLE_LINEAR_NATIVE_V1.json').read_text());rows=[];checks=[]
 for d in ['fineweb','code']:
  for name in ['paired256','separate256','source256']:
   for family in ['removal','same_token','source_only','context_only']:
    key='product_'+name;checks.append(abs(raw['summary'][d][key][family]['centered_effect_relative_error']-prior['summary'][d][key][family]['centered_effect_relative_error']))
    rr=[v for v in raw['records'] if v['domain']==d and v['candidate']==key and v['family']==family];sums={k:sum(v['error_composition'][k] for v in rr) for k in ['baseline_error_energy','increment_error_energy','twice_error_cross','total_error_energy']};en=sums['total_error_energy'];checks.append(abs(en-sums['baseline_error_energy']-sums['increment_error_energy']-sums['twice_error_cross'])/en)
    rows.append(dict(domain=d,program=name,family=family,baseline_fraction=sums['baseline_error_energy']/en,increment_fraction=sums['increment_error_energy']/en,cross_fraction=sums['twice_error_cross']/en))
 target=[v for v in rows if v['program']=='separate256' and v['family']=='same_token'];pred=dict(pred_a_instrument=max(checks)<1e-8,pred_b_alignment=all(v['cross_fraction']>=.1 for v in target),pred_c_increment=all(v['increment_fraction']>=.1 for v in target))
 out=P/'MIDPOINT_ERROR_COMPOSITION_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,checks=max(checks),records=rows,scope='Exact decomposition of errors in centered final-logit effects: baseline plus incremental prediction change. After nonlinearities, increment is not itself a standalone linear-write effect. No fitting, reused panels; signed cross contribution.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
