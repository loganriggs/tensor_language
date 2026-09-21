#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_late pred_c_early
"""Position-specific conditional preservation; no fitting,48captures.
pred_a_instrument pooled prior replay and bins partition energies<1e-8.
pred_b_late separate256/exact whole-swap error ratio late>=1.1early bothdomains.
pred_c_early separate256 early whole-swap error<=1.05exact bothdomains.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,position_bins=[[0,64],[64,128],[128,256]],fit=False)));return
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_ROLE_SHARED_LINEAR_GRAPHS_V1.pt';native.SOURCE_CONTEXT_FAMILIES=True;native.POSITION_BINS=True;native.OUTPUT_NAME='MIDPOINT_POSITION_AUDIT_RAW_V1.json';native.main()
 raw=json.loads((P/native.OUTPUT_NAME).read_text());old=json.loads((P/'MIDPOINT_ROLE_LINEAR_NATIVE_V1.json').read_text());checks=[];rows=[]
 for d in ['fineweb','code']:
  for name in ['exact_linear','paired256','separate256']:
   for f in ['removal','same_token','source_only','context_only']:
    key='product_'+name;checks.append(abs(raw['summary'][d][key][f]['centered_effect_relative_error']-old['summary'][d][key][f]['centered_effect_relative_error']))
    rr=[v for v in raw['records'] if v['domain']==d and v['candidate']==key and v['family']==f]
    for v in rr:
     for field in ['native_centered_effect_energy','centered_effect_error_energy']:checks.append(abs(sum(b[field] for b in v['position_bins'].values())-v[field])/max(v[field],1e-30))
    for bin in ['0:64','64:128','128:256']:
     bb=[v['position_bins'][bin] for v in rr if bin in v['position_bins']];ref=sum(v['native_centered_effect_energy'] for v in bb);error=sum(v['centered_effect_error_energy'] for v in bb)
     rows.append(dict(domain=d,program=name,family=f,position_bin=bin,sites=sum(v['sites'] for v in bb),effect_error=(error/ref)**.5,reference_energy=ref,error_energy=error))
 value=lambda d,k,b:next(v['effect_error'] for v in rows if v['domain']==d and v['program']==k and v['family']=='same_token' and v['position_bin']==b)
 ratio=lambda d,b:value(d,'separate256',b)/value(d,'exact_linear',b)
 pred=dict(pred_a_instrument=max(checks)<1e-8,pred_b_late=all(ratio(d,'128:256')>=1.1*ratio(d,'0:64') for d in ['fineweb','code']),pred_c_early=all(ratio(d,'0:64')<=1.05 for d in ['fineweb','code']))
 out=P/'MIDPOINT_POSITION_AUDIT_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,checks_max=max(checks),records=rows,scope='Position-stratified existing interventions with fixed donor mapping. Recipient-position bins only; donor positions may differ. Calibration positions0:64; diagnostic early evaluation starts16. No fitting or new candidate selection.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
