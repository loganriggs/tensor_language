#!/usr/bin/env python3
# BQGATE: 0forwards0seq; full-U weight-only bounded joint reader fitting.
"""pred_a numeric; pred_b fresh-coordinate convergence; pred_c advantage over200step baseline.
Registered shared protocol SHARED_READER_RETAINED_HISTORY_V1_PREREGISTRATION.md.
A relative graph/core checks; B gradient1e-7 andprogress1e-6; C baseline objective advantage1e-4.
"""
import json
from shared_reader_retained_history_v1 import main

if __name__ == "__main__":
    result=main('spectral','original')
    if result is not None:
        print(json.dumps({"pred_a": result["pred_"+"a"], "pred_b": result["pred_"+"b"], "pred_c": result["pred_"+"c"]}), flush=True)
