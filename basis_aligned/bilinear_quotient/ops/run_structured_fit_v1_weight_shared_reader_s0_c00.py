#!/usr/bin/env python3
# BQGATE: unsupervised convergent fit;0forwards540s fitting chunk, checkpoint resume.
"""pred_a instrument; pred_b converged (false=unfinished); pred_c checkpoint replay."""
import os,sys
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(ROOT),str(POLY)]
PREDICTIONS = {'pred_a_instrument': 'finite and conditioned; replay <=1e-10', 'pred_b_converged': 'five loss checks plus full gradient; false means unfinished', 'pred_c_checkpoint_replay': 'independent saved-function replay <=1e-10'}
from structured_quadratic_campaign_runner_v1 import run
if __name__=='__main__':run('weight_shared_reader_s0',0,RUNNER,POLY/'STRUCTURED_FIT_V1_weight_shared_reader_s0_C00_BINDING.json',dry=bool(os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL')))
