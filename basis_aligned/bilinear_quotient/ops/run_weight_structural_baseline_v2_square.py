#!/usr/bin/env python3
# BQGATE: 0forwards0seq; weight-only structural baseline,540s, no data.
# pred_a_instrument replay; pred_b_converged; pred_c_capture_gain.
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'basis_aligned/polynomial_causal'))
from run_weight_structural_baseline_v2 import main
REGISTERED={'pred_a_instrument': 'Dense controls and native saved objective replay within1e-10', 'pred_b_converged': 'Five checks plateau and full gradient thresholds', 'pred_c_capture_gain': 'Capture at least0.09698912596, unmatched capacity screen'}
if __name__=='__main__':main('square',REGISTERED)
