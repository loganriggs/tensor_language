#!/usr/bin/env python3
# BQGATE: 0forwards0seq; full-U weight-only bounded joint reader fitting.
"""pred_a numeric; pred_b fresh-coordinate convergence; pred_c objective gain.
Registered shared protocol SHARED_READER_JOINT_FIT_V1_PREREGISTRATION.md.
A relative graph/core checks; B gradient1e-7 andprogress1e-6; C gain1e-4.
"""
import json
from shared_reader_joint_fit_v1 import main

if __name__ == "__main__":
    result=main('native','original')
    if result is not None:
        print(json.dumps({"pred_a": result["pred_"+"a"], "pred_b": result["pred_"+"b"], "pred_c": result["pred_"+"c"]}), flush=True)
