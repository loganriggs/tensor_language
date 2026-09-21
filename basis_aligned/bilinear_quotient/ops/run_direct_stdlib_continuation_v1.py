#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_selectivity pred_c_writer
"""Frozen continuation-group domain shift, no fitting.
Identical prespecified bars to FineWeb: pred_a counts/hashes, pred_b real
continuationCE>.05 and abs(spacedCE)<.02, pred_c real-minus-sham oddsdecrease>.02.
Secondary same-token contrast retained without weakening coverage requirements.
"""
import run_direct_continuation_group_v1 as parent
parent.PLAN_NAME='MIDPOINT_STDLIB_CONTINUATION_PLAN_V1.json'
parent.TOKEN_NAME='MIDPOINT_STDLIB_CONTINUATION_TOKENS_V1.pt'
parent.OUTPUT_NAME='MIDPOINT_STDLIB_CONTINUATION_NATIVE_V1.json'
parent.FORWARDS=16
parent.SCOPE='Frozen FineWeb-discovered group tested on docstring-stripped Python standard-library function snippets; code shift, pretraining overlap unknown.'
def main():
 import os,json
 parent.main()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):return
 result=json.loads((parent.P/parent.OUTPUT_NAME).read_text());s=result['summary'];a=s['real'];b=s['sham']
 pred=dict(pred_a_instrument=result['predictions']['pred_a_instrument'],pred_b_selectivity=a['continuation']['ce_added']>.05 and abs(a['spaced_word']['ce_added'])<.02,pred_c_writer=a['continuation']['mean_logodds_decrease']-b['continuation']['mean_logodds_decrease']>.02)
 assert pred==result['predictions']
if __name__=='__main__':main()
