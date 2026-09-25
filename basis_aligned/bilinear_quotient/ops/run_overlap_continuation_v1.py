#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_components pred_c_metrics pred_d_arithmetic
"""Equal additional optimizer budget on unchanged whole-block graph.
pred_a: relative execution/dense normalized loss<1e-8.
pred_b: all3values<=.15/1.10baseline and zJacobians<=1.10baseline.
pred_c: both coefficient errors<=1.10baseline.
pred_d: source multiplications<=.80baseline and fewer coefficients.
Null: extra optimization alone fails the original fidelity requirements.
"""
import os,json
from run_joint_overlap_v1 import main,P
if __name__=='__main__':
 main(prefix='OVERLAP_CONTINUATION')
 if not (os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL')):
  record=json.loads((P/'OVERLAP_CONTINUATION_V1.json').read_text());r=record['predictions']
  predictions=dict(pred_a_instrument=r['pred_a_instrument'],pred_b_components=r['pred_b_components'],pred_c_metrics=r['pred_c_metrics'],pred_d_arithmetic=r['pred_d_arithmetic']);assert predictions==r
