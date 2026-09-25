#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_components pred_c_metrics pred_d_arithmetic
"""Mixed shared/private source products at the same graph budget.
Use frozen toy-selected optimizer, original native fidelity bars and paired
whole-block continuation control; no new model/data collection.
pred_a: relative source execution/dense normalized loss<1e-8.
pred_b: all3values<=.15/1.10baseline and zJacobians<=1.10baseline.
pred_c: both coefficient errors<=1.10baseline.
pred_d: source multiplications<=.80baseline and fewer coefficients.
Null: direct cross paths do not improve faithful sharing at fixed cost.
"""
import os,json
from run_joint_overlap_v1 import main,P
if __name__=='__main__':
 main(prefix='CROSS_SLOT_JOINT')
 if not (os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL')):
  record=json.loads((P/'CROSS_SLOT_JOINT_V1.json').read_text());r=record['predictions']
  predictions=dict(pred_a_instrument=r['pred_a_instrument'],pred_b_components=r['pred_b_components'],pred_c_metrics=r['pred_c_metrics'],pred_d_arithmetic=r['pred_d_arithmetic']);assert predictions==r
