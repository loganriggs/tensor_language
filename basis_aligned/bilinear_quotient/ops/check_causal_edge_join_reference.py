#!/usr/bin/env python3
"""CPU exact algebra controls for the candidate causal edge-join operation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path('/workspace/tensor_language')
POLY = ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0, str(POLY))
import torch
import causal_edge_join_reference as R

if __name__ == '__main__':
    torch.set_num_threads(2); started = time.perf_counter()
    receipt = R.controls()
    receipt['reference_sha256'] = hashlib.sha256((POLY/'causal_edge_join_reference.py').read_bytes()).hexdigest()
    receipt['wall_seconds'] = time.perf_counter()-started
    (POLY/'CAUSAL_EDGE_JOIN_REFERENCE_CONTROLS.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt)); assert receipt['passed']
