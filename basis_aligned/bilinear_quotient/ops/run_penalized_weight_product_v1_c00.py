#!/usr/bin/env python3
# BQGATE: unsupervised penalized weight fit;0forwards540s fitting,900s alarm.
"""pred_a instrument; pred_b convergence; pred_c cancellation<=10 and capture>=75%reference and objective gain>=1e-4."""
import os,sys
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(ROOT),str(POLY)]
PREDICTIONS={'pred_a_instrument':'finite regularized solve; checkpoint replay<=1e-10',
    'pred_b_converged':'five penalized objective checks plus unchanged gradient bars; false unfinished',
    'pred_c_stable_useful':'cancellation<=10; capture>=.75*reference; objective gain>=1e-4 over fixed-reader baseline'}
from penalized_weight_product_runner_v1 import run
if __name__=='__main__':run(RUNNER,POLY/'PENALIZED_WEIGHT_PRODUCT_V1_C00_BINDING.json',0,dry=bool(os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL')))
