#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_netreplay pred_b_blocks pred_c_joint
"""Eight independent signed blocks: removal andsame-token swaps,48nativeforwards.
pred_a_netreplay original24productsame-token jointnativeeffect energy replay<1e-4 anddonor/hash checks;
pred_b_blocks all8block centered effect error<.3/cos>.95 bothdomains/bothinterventions;
pred_c_joint joint8block centered effect error<.2 bothdomains/interventions.
Null stablecoefficientblocks/netfidelity hide poor independentblockeffects.
Positive/negative coefficient blocks can have either activation sign; writersareopposite.
Program uses owncalibrationmeans; teacher uses fullspectralblockmeans. Positions16:256.
"""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=48,blocks=8,products=24,interventions=['removal','same_token'])));return
 import run_direct_midpoint_swap_v1 as swap
 swap.FAMILIES=['removal','same_token'];swap.PROGRAM_FILE='MIDPOINT_CENTERED_BLOCK_PROGRAM_V1.pt';swap.OUTPUT_NAME='MIDPOINT_INDEPENDENT_BLOCKS_RAW_V1.json';swap.main();r=json.loads((P/swap.OUTPUT_NAME).read_text());old=json.loads((P/'MIDPOINT_BLOCK3_SWAP_V1.json').read_text());checks=[abs(r['summary'][d]['same_token']['joint']['native_centered_effect_energy']/old['summary'][d]['same_token']['joint']['native_centered_effect_energy']-1) for d in ['fineweb','code']]
 pred=dict(pred_a_netreplay=max(checks)<1e-4 and r['predictions']['pred_a_instrument'],pred_b_blocks=all(r['summary'][d][f][str(g)]['centered_effect_relative_error']<.3 and r['summary'][d][f][str(g)]['centered_effect_cosine']>.95 for d in ['fineweb','code'] for f in ['removal','same_token'] for g in range(8)),pred_c_joint=all(r['summary'][d][f]['joint']['centered_effect_relative_error']<.2 for d in ['fineweb','code'] for f in ['removal','same_token']))
 result=dict(predictions=pred,net_effect_energy_replay=max(checks),summary=r['summary'],scope='Independent fixedmetric signed-block interventions. Oppositewriters share fouroutputdirections. Teacher blocks fromweights; approximation24products. Reuseddiagnostics, nosemanticselectivityclaim.');out=P/'MIDPOINT_INDEPENDENT_BLOCKS_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
