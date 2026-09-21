#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_improve pred_c_features
"""Matched fixed-vs-learned output blocks: 96 native forwards, reused panels.
pred_a_instrument both executor checks and joint native energy replay<1e-5.
pred_b_improve learned joint error <= fixed in both domains/families.
pred_c_features learned individual same-token error<.3 and cosine>.95.
No fitting. Individual new features differ; only joint effects match across arms.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=96,products_per_arm=16)));return
 import run_direct_midpoint_swap_v1 as swap
 swap.FAMILIES=['removal','same_token'];arms={}
 for label,file in [('fixed','MIDPOINT_NATIVE_BTD_FIXED_V1.pt'),('learned','MIDPOINT_NATIVE_BTD_V1.pt')]:
  swap.PROGRAM_FILE=file;swap.OUTPUT_NAME=f'MIDPOINT_BTD_{label.upper()}_RAW_V1.json';swap.main();arms[label]=json.loads((P/swap.OUTPUT_NAME).read_text())
 replay=max(abs(arms['fixed']['summary'][d][f]['joint']['native_centered_effect_energy']/arms['learned']['summary'][d][f]['joint']['native_centered_effect_energy']-1) for d in ['fineweb','code'] for f in swap.FAMILIES)
 pred=dict(pred_a_instrument=all(r['predictions']['pred_a_instrument'] for r in arms.values()) and replay<1e-5,pred_b_improve=all(arms['learned']['summary'][d][f]['joint']['centered_effect_relative_error']<=arms['fixed']['summary'][d][f]['joint']['centered_effect_relative_error'] for d in ['fineweb','code'] for f in swap.FAMILIES),pred_c_features=arms['learned']['predictions']['pred_b_features'])
 result=dict(predictions=pred,native_joint_energy_replay=replay,summary={k:v['summary'] for k,v in arms.items()},scope='Matched original-weight fixed/learned output block candidates. Reused panels. Individual operational features differ; joint teacher must be invariant.');out=P/'MIDPOINT_BTD_NATIVE_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
