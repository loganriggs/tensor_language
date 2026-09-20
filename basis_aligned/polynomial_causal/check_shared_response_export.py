"""CPU-only consumer replay; checkpoint and native model code are not loaded."""
import json
from pathlib import Path
import torch
from shared_response_runtime import execute


def values(obj):
    if isinstance(obj,torch.Tensor):return obj.numel()
    if isinstance(obj,dict):return sum(values(v) for v in obj.values())
    if isinstance(obj,list):return sum(values(v) for v in obj)
    return 0


def main():
    root=Path(__file__).resolve().parents[2]
    artifact=root/'basis_aligned/bilinear_quotient/circuits/followups/subject_response_v655_program.pt'
    package=torch.load(artifact,map_location='cpu',weights_only=True)
    runtime=package['runtime'];maximum=0.;rows=0
    for case in package['cases']:
        actual=execute(runtime,case['initial'],case['contexts'],case['final_context'])
        maximum=max(maximum,float((actual-case['reference_logits']).abs().max()))
        rows+=len(actual)
    assert maximum<=1e-8
    assert not torch.cuda.is_initialized()
    result=dict(cpu_replay_max_absolute_error=maximum,rows=rows,
        runtime_tensor_values=values(runtime),producer_tensor_values=values(package['producer']),
        unique_quadratic_products_per_evaluation=sum(b['inputs']*(b['inputs']+1)//2 for b in runtime['blocks']),
        quadratic_coefficients=sum(b['coefficients'].numel() for b in runtime['blocks']),
        context_values_per_example=package['metadata']['context_values_per_example'],
        artifact_bytes=artifact.stat().st_size,cuda_initialized=False,
        scope='Replay at saved conditional context boundary; no checkpoint access, token-input closure or new OOD claim')
    out=Path(__file__).with_name('SHARED_RESPONSE_EXPORT_CPU_2026-09-20.json')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
