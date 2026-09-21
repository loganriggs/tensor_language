#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_preserve pred_c_jointmetric
"""Matched8product separable-vs-exactjointmoment swap screen, noheldfitting.
pred_a_instrument botharms donor/hash/selfedit pass;
pred_b_preserve jointmetric same-token allindividualerror<.3/cos>.95 andjoint<=1.25confirmed bothdomains;
pred_c_jointmetric jointmetric jointMSE<=.8separable bothdomains.
Null exactpaired momentfit fails to transfer. Price96nativeforwards,8products/18,432inputcoeff/32outputmix.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=96,products=8,arms=['separable','joint'])));return
 import run_direct_midpoint_swap_v1 as swap
 results={}
 for arm,file in [('separable','MIDPOINT_CP8_SEPARABLE_V1.pt'),('joint','MIDPOINT_JOINT_MOMENT_REFIT_V1.pt')]:
  swap.PROGRAM_FILE=file;swap.OUTPUT_NAME=f'MIDPOINT_CP8_SWAP_{arm.upper()}_V1.json';swap.main();results[arm]=json.loads((P/swap.OUTPUT_NAME).read_text())
 old=json.loads((P/'MIDPOINT_SWAP_V1.json').read_text());pred=dict(pred_a_instrument=all(r['predictions']['pred_a_instrument'] for r in results.values()),pred_b_preserve=results['joint']['predictions']['pred_b_features'] and all(results['joint']['summary'][d]['same_token']['joint']['centered_effect_relative_error']<=1.25*old['summary'][d]['same_token']['joint']['centered_effect_relative_error'] for d in ['fineweb','code']),pred_c_jointmetric=all(results['joint']['summary'][d]['same_token']['joint']['centered_effect_error_energy']<=.8*results['separable']['summary'][d]['same_token']['joint']['centered_effect_error_energy'] for d in ['fineweb','code']))
 result=dict(predictions=pred,results={k:v['summary'] for k,v in results.items()},scope='Matched8product programs; exactjointmoment vsseparable moment weightmatching. Samefixedfour outputfeatures andwriters. Reuseddiagnostics, no semanticidentity orfreshconfirmation.')
 out=P/'MIDPOINT_CP8_SWAP_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
