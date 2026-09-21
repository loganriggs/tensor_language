#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_features pred_c_preserve
"""Shared graph native swap replay; primary joint error <=1.25confirmed bothdomains.
pred_a_instrument donor/hash/selfedit checks; pred_b_features same-token all error<.3/cos>.95;
pred_c_preserve refit joint error<=1.25confirmed same-token. Price48nativeforwards.
Null weighted tensor refit still fails native graph-preservation. No native-data refit.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=48,products=16,reader_coefficients=18688)));return
 import run_direct_midpoint_swap_v1 as swap
 swap.PROGRAM_FILE='MIDPOINT_JOINT_CORE_REFIT_V1.pt';swap.OUTPUT_NAME='MIDPOINT_REFIT_SWAP_RAW_V1.json';swap.main();r=json.loads((P/swap.OUTPUT_NAME).read_text());old=json.loads((P/'MIDPOINT_SWAP_V1.json').read_text());pred=dict(pred_a_instrument=r['predictions']['pred_a_instrument'],pred_b_features=r['predictions']['pred_b_features'],pred_c_preserve=all(r['summary'][d]['same_token']['joint']['centered_effect_relative_error']<=1.25*old['summary'][d]['same_token']['joint']['centered_effect_relative_error'] for d in ['fineweb','code']));out=P/'MIDPOINT_REFIT_SWAP_V1.json';assert not out.exists();result=dict(predictions=pred,summary=r['summary'],baseline_summary=old['summary'],scope='Explicit shared input graph evaluated, not collapsed dense proxy. Reused diagnostics. Targetfeatures andwriters unchanged; no semanticidentification claim.');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
