"""Replay the complete hybrid predictor without a model checkpoint or CUDA."""
import json
from pathlib import Path
import torch
from hybrid_response_runtime import execute
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_hybrid_v688.pt'

def values(v):
    if isinstance(v,torch.Tensor):return v.numel()
    if isinstance(v,dict):return sum(values(x) for x in v.values())
    if isinstance(v,list):return sum(values(x) for x in v)
    return 0

def main():
    torch.set_num_threads(2)
    p=torch.load(ART,map_location='cpu',weights_only=True)
    errors=[];zero_errors=[];rows=0;reader_count=0;input_count=0;reference_count=0
    for case in p['cases']:
        reader_count+=case['readers'].numel()
        for name,source in case['sources'].items():
            predicted=execute(p['runtime'],p['encoder'],case,source['delta'])
            errors.append(float((predicted-source['reference']).abs().max()))
            zero=execute(p['runtime'],p['encoder'],case,torch.zeros_like(source['delta']))
            # Number baseline includes a float32 native-margin port; tolerate its readout replay floor.
            zero_errors.append(float(zero.abs().max()))
            input_count+=source['delta'].numel();reference_count+=source['reference'].numel();rows+=len(predicted)
    assert max(errors)<=1e-8
    assert max(zero_errors)<=1e-4
    assert not torch.cuda.is_initialized()
    result=dict(source_cases=rows,contexts=len(p['cases']),max_replay_error=max(errors),max_zero_response=max(zero_errors),
        runtime_values=values(p['runtime']),encoder_values=p['encoder'].numel(),prepared_case_values=values(p['cases']),
        reader_values=reader_count,source_input_values=input_count,reference_values=reference_count,artifact_bytes=ART.stat().st_size,
        cuda_initialized=torch.cuda.is_initialized(),scope=p['scope'])
    Path(__file__).with_name('HYBRID_RESPONSE_CPU_REPLAY_2026-09-20.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
