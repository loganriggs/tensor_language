"""Registered saved-coefficient linear-gate falsifier; no fitted factor search."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import torch
import shared_query_gate_reference as G

BASE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter()
    out=BASE/'QUERY_SHARED_LINEAR_GATE_V1_RESULT.json';assert not out.exists()
    source=BASE/'QUERY_COMMON_SQUARED_CHANNELS_V1_COEFFICIENTS.pt'
    assert digest(source)=='3f195fe72cc86a62b96ba08733c89e4768a6462608b97a0cbf4483fe1b4e3151'
    parent=BASE/'QUERY_COMMON_SQUARED_CHANNELS_V1_RESULT.json'
    assert json.loads(parent.read_text())['predictions']['pred_a_instrument']
    controls=G.controls();assert controls['passed']
    saved=torch.load(source,map_location='cpu',weights_only=True)
    certificate=G.obstruction(saved['forms'].tolist())
    result={'scope':'Exact necessary-condition certificate on stored FP64 forms in one opened context; no ideal-real interval certificate.',
        'predictions':{'pred_a_instrument':controls['passed'],
                       'pred_b_single_linear_gate_rejected':certificate['terminal']=='no_common_linear_gate'},
        'certificate':certificate,'controls':controls,'opaque_export_constants':387968,
        'source_sha256':digest(source),'parent_sha256':digest(parent),'runner_sha256':digest(Path(__file__)),
        'reference_sha256':digest(BASE/'shared_query_gate_reference.py'),
        'prereg_sha256':digest(BASE/'QUERY_SHARED_LINEAR_GATE_V1_PREREGISTRATION.md'),
        'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
