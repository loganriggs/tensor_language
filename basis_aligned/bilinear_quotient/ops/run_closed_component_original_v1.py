#!/usr/bin/env python3
# BQGATE: 0forwards0seq; closed-component full-U weight fitting only.
"""pred_a relative replay; pred_b fresh convergence; pred_c objective gain.

Shared protocol CLOSED_COMPONENT_REFIT_V1_PREREGISTRATION.md.
A numeric1e-8; B gradient1e-7/progress1e-6; C own objective gain1e-6.
"""
import json
from closed_component_refit_v1_runner import main

if __name__ == '__main__':
    result = main('original')
    if result is not None:
        print(json.dumps({'pred_a': result['pred_'+'a'], 'pred_b': result['pred_'+'b'], 'pred_c': result['pred_'+'c']}), flush=True)
